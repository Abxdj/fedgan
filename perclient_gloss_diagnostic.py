"""
Per-Client G-Loss Diagnostic — cGAN Non-IID Divergence Mechanism
FedGAN Failure Analysis - Anish Bharadwaj

Finding #2 (CLAUDE.md) asserts a mechanism for why the cGAN diverges
under non-IID in 3/3 seeds: clients missing classes locally develop
discriminators that reject all generated samples of those classes,
and FedAvg then averages contradictory generator gradients. That
mechanism was never evidenced — only the aggregate G-loss divergence
was. This script re-runs the cGAN Non-IID federation (seed 42, the
same seed as the seed42 multiseed run) with two added instruments:

1. Per-client, per-class discriminator logit on generated ("fake")
   images, using a FIXED noise/label eval set each round, evaluated
   under model.eval() so it doesn't perturb BatchNorm/InstanceNorm
   running stats used by the real training loop.
   -> tests: do clients reject classes absent from their local data?

2. Cosine similarity between each pair of clients' generator weight
   updates (local G after local training minus global G before it),
   per round, before FedAvg averages them.
   -> tests: do client updates become contradictory (negative
      cosine similarity) as training progresses, especially around
      the point the aggregate G-loss starts diverging?

Does NOT touch the original cgan_train_multiseed.py results —
this is a standalone diagnostic run, single seed, not part of the
headline multi-seed numbers.
"""

import os
import copy
import json
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PathMNIST
from cgan import ConditionalGenerator, ConditionalDiscriminator, LATENT_DIM, NUM_CLASSES

SEED         = 42   # matches the seed42 run in the multiseed sweep
NUM_CLIENTS  = 5
NUM_ROUNDS   = 30
LOCAL_STEPS  = 100
BATCH_SIZE   = 64
LR_G         = 0.0002
LR_D         = 0.0002
BETA1        = 0.5
EVAL_BATCH   = 32     # fake images per class per client per round, for D-logit probe
DATA_DIR     = "./data"
LOG_DIR      = "./logs"
FIG_DIR      = "./figures"
SPLIT_FILE   = "./data/non_iid_split.json"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Seed: {SEED}")


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_client_loaders(split_file):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = PathMNIST(split="train", transform=transform,
                         download=False, root=DATA_DIR)
    with open(split_file) as f:
        split = json.load(f)

    all_labels = np.array([dataset[i][1].item() for i in range(len(dataset))])

    loaders, sizes, class_counts = {}, {}, {}
    for client_id, indices in split.items():
        cid = int(client_id)
        subset = Subset(dataset, indices)
        loaders[cid] = DataLoader(subset, batch_size=BATCH_SIZE,
                                   shuffle=True, drop_last=True, num_workers=0)
        sizes[cid] = len(indices)
        class_counts[cid] = np.bincount(all_labels[indices], minlength=NUM_CLASSES).tolist()
    return loaders, sizes, class_counts


# ── FedAvg ────────────────────────────────────────────────────────────────────
def fedavg(global_G, client_generators, client_sizes, selected_clients):
    total     = sum(client_sizes[c] for c in selected_clients)
    new_state = copy.deepcopy(global_G.state_dict())
    for key in new_state:
        new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)
    for cid in selected_clients:
        weight = client_sizes[cid] / total
        for key in new_state:
            new_state[key] += weight * client_generators[cid].state_dict()[key].float()
    global_G.load_state_dict(new_state)
    return global_G


# ── Local Training (identical to cgan_train_multiseed.py) ────────────────────
def train_local(client_id, generator, discriminator, loader, opt_G, opt_D, steps):
    criterion = nn.BCEWithLogitsLoss()
    generator.train()
    discriminator.train()

    g_losses, d_losses = [], []
    data_iter = iter(loader)

    for _ in range(steps):
        try:
            real_imgs, real_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real_imgs, real_labels = next(data_iter)

        real_imgs   = real_imgs.to(device)
        real_labels = real_labels.squeeze().long().to(device)
        bs = real_imgs.size(0)

        real_target = torch.ones(bs, 1).to(device)
        fake_target = torch.zeros(bs, 1).to(device)

        # Train Discriminator
        opt_D.zero_grad()
        d_real = discriminator(real_imgs, real_labels)
        loss_d_real = criterion(d_real, real_target)

        z           = torch.randn(bs, LATENT_DIM).to(device)
        fake_labels = torch.randint(0, NUM_CLASSES, (bs,)).to(device)
        fake_imgs   = generator(z, fake_labels).detach()
        d_fake      = discriminator(fake_imgs, fake_labels)
        loss_d_fake = criterion(d_fake, fake_target)

        loss_D = (loss_d_real + loss_d_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # Train Generator
        opt_G.zero_grad()
        z           = torch.randn(bs, LATENT_DIM).to(device)
        fake_labels = torch.randint(0, NUM_CLASSES, (bs,)).to(device)
        fake_imgs   = generator(z, fake_labels)
        d_out       = discriminator(fake_imgs, fake_labels)
        loss_G      = criterion(d_out, real_target)
        loss_G.backward()
        opt_G.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())

    return np.mean(g_losses), np.mean(d_losses)


# ── Diagnostic Instruments ────────────────────────────────────────────────────
@torch.no_grad()
def probe_d_logits(generator, discriminator, fixed_z_per_class):
    """Mean D logit on fake images, per class, for one client. Higher = D fooled,
    lower/more negative = D confidently rejects that class."""
    generator.eval()
    discriminator.eval()
    logits_per_class = []
    for c in range(NUM_CLASSES):
        z = fixed_z_per_class[c]
        labels = torch.full((z.size(0),), c, dtype=torch.long, device=device)
        fake = generator(z, labels)
        logit = discriminator(fake, labels)
        logits_per_class.append(logit.mean().item())
    generator.train()
    discriminator.train()
    return logits_per_class


def flatten_generator(gen):
    return torch.cat([p.detach().reshape(-1) for p in gen.parameters()])


def pairwise_cosine_sims(update_vectors):
    """update_vectors: dict[cid] -> flat tensor. Returns list of cosine sims
    for every unordered client pair."""
    sims = []
    for cid_a, cid_b in itertools.combinations(update_vectors.keys(), 2):
        a, b = update_vectors[cid_a], update_vectors[cid_b]
        sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        sims.append(sim)
    return sims


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f" Per-Client G-Loss Diagnostic — cGAN Non-IID (seed {SEED})")
    print(f"{'='*60}")

    loaders, client_sizes, class_counts = load_client_loaders(SPLIT_FILE)
    present_mask = {cid: [count > 0 for count in class_counts[cid]] for cid in range(NUM_CLIENTS)}

    print("\nPer-client local class counts (0 = class absent locally):")
    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid}: {class_counts[cid]}")

    global_G  = ConditionalGenerator().to(device)
    client_Gs = {i: copy.deepcopy(global_G) for i in range(NUM_CLIENTS)}
    client_Ds = {i: ConditionalDiscriminator().to(device) for i in range(NUM_CLIENTS)}

    opt_Gs = {i: optim.Adam(client_Gs[i].parameters(), lr=LR_G, betas=(BETA1, 0.999))
              for i in range(NUM_CLIENTS)}
    opt_Ds = {i: optim.Adam(client_Ds[i].parameters(), lr=LR_D, betas=(BETA1, 0.999))
              for i in range(NUM_CLIENTS)}

    # Fixed eval noise per class, shared across all clients/rounds so that
    # changes in D-logit reflect weight changes only, not sampling noise.
    eval_gen = torch.Generator(device=device).manual_seed(SEED + 1000)
    fixed_z_per_class = [
        torch.randn(EVAL_BATCH, LATENT_DIM, device=device, generator=eval_gen)
        for _ in range(NUM_CLASSES)
    ]

    log = {
        "round": [], "g_loss": [], "d_loss": [],
        "d_logit_per_client_class": [],   # list of round -> {cid: [9 logits]}
        "cosine_sim_pairwise_mean": [],
        "cosine_sim_pairwise_min": [],
        "class_present_mask": present_mask,
        "class_counts": class_counts,
    }

    for rnd in range(1, NUM_ROUNDS + 1):
        global_state_before = copy.deepcopy(global_G.state_dict())
        for cid in range(NUM_CLIENTS):
            client_Gs[cid].load_state_dict(copy.deepcopy(global_state_before))

        round_g, round_d = [], []
        round_d_logits = {}
        update_vectors = {}

        for cid in range(NUM_CLIENTS):
            g_loss, d_loss = train_local(
                client_id=cid, generator=client_Gs[cid], discriminator=client_Ds[cid],
                loader=loaders[cid], opt_G=opt_Gs[cid], opt_D=opt_Ds[cid], steps=LOCAL_STEPS
            )
            round_g.append(g_loss)
            round_d.append(d_loss)

            round_d_logits[cid] = probe_d_logits(client_Gs[cid], client_Ds[cid], fixed_z_per_class)

            local_flat  = flatten_generator(client_Gs[cid])
            global_flat = torch.cat([p.detach().reshape(-1) for p in global_G.parameters()])
            update_vectors[cid] = local_flat - global_flat

        sims = pairwise_cosine_sims(update_vectors)

        global_G = fedavg(global_G, client_Gs, client_sizes, list(range(NUM_CLIENTS)))

        avg_g, avg_d = float(np.mean(round_g)), float(np.mean(round_d))
        log["round"].append(rnd)
        log["g_loss"].append(avg_g)
        log["d_loss"].append(avg_d)
        log["d_logit_per_client_class"].append(round_d_logits)
        log["cosine_sim_pairwise_mean"].append(float(np.mean(sims)))
        log["cosine_sim_pairwise_min"].append(float(np.min(sims)))

        if rnd % 5 == 0 or rnd == 1:
            print(f"Round {rnd:3d}/{NUM_ROUNDS} | G: {avg_g:.4f} | D: {avg_d:.4f} | "
                  f"cos_sim(mean/min): {np.mean(sims):+.3f}/{np.min(sims):+.3f}")

    log_path = os.path.join(LOG_DIR, "perclient_gloss_diagnostic_seed42.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDiagnostic log saved -> {log_path}")

    print_verdict(log, present_mask)


def print_verdict(log, present_mask):
    present_logits, absent_logits = [], []
    for round_logits in log["d_logit_per_client_class"]:
        for cid_str, logits in round_logits.items():
            cid = int(cid_str)
            for c, logit in enumerate(logits):
                (present_logits if present_mask[cid][c] else absent_logits).append(logit)

    print(f"\n{'='*60}\n VERDICT\n{'='*60}")
    print(f"Mean D logit on fake images, classes PRESENT locally: {np.mean(present_logits):+.4f}")
    print(f"Mean D logit on fake images, classes ABSENT locally : {np.mean(absent_logits):+.4f}")
    gap = np.mean(present_logits) - np.mean(absent_logits)
    print(f"Gap (present - absent): {gap:+.4f}  "
          f"{'-> supports rejection hypothesis' if gap > 0 else '-> does NOT support rejection hypothesis'}")

    first_half_cos = np.mean(log["cosine_sim_pairwise_mean"][:15])
    second_half_cos = np.mean(log["cosine_sim_pairwise_mean"][15:])
    print(f"\nMean pairwise cosine similarity of client G updates, rounds 1-15: {first_half_cos:+.4f}")
    print(f"Mean pairwise cosine similarity of client G updates, rounds 16-30: {second_half_cos:+.4f}")
    print(f"{'-> updates became MORE contradictory over training (supports mechanism)' if second_half_cos < first_half_cos else '-> updates did NOT become more contradictory over training'}")


if __name__ == "__main__":
    main()
