"""
Per-Client Divergence Diagnostic — IID CONTROL
FedGAN Failure Analysis - Anish Bharadwaj

Extends perclient_gloss_diagnostic.py with the missing control.

WHY THIS EXISTS
The original diagnostic instrumented only the cGAN Non-IID run. That
establishes what happens in the diverging configuration but cannot
establish that non-IID CAUSES it, because there is no counterfactual.
Two specific risks it cannot rule out:

  1. Declining pairwise cosine similarity between client generator
     updates may simply be what FedAvg does over time, in any
     configuration. Note that ~0.09 at round 1 is already close to
     orthogonal for vectors of this dimensionality, so a drift to
     ~0.01 is not self-evidently meaningful without a baseline.

  2. The present-vs-absent D logit gap is undefined under IID (every
     client holds every class), so the IID run instead provides the
     "no absent classes" floor: how much per-class logit spread
     exists purely from class difficulty and sampling.

This script runs the IDENTICAL diagnostic on the IID split and the
non-IID split back to back, in one process, and prints a side-by-side
comparison. Only if the two diverge does the mechanism claim hold.

Usage:
    python diagnostic_iid_control.py --seed 42

Run for seeds 42, 123, 2024 to address the n=1 limitation of the
original diagnostic. Outputs one JSON per (condition, seed).
"""

import os
import copy
import json
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PathMNIST
from cgan import ConditionalGenerator, ConditionalDiscriminator, LATENT_DIM, NUM_CLASSES

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
SEED = args.seed

# ── Config (identical to perclient_gloss_diagnostic.py) ──────────────────────
NUM_CLIENTS = 5
NUM_ROUNDS  = 30
LOCAL_STEPS = 100
BATCH_SIZE  = 64
LR_G        = 0.0002
LR_D        = 0.0002
BETA1       = 0.5
EVAL_BATCH  = 32
DATA_DIR    = "./data"
LOG_DIR     = "./logs"

CONDITIONS = {
    "iid"     : "./data/iid_split.json",
    "non_iid" : "./data/non_iid_split.json",
}

os.makedirs(LOG_DIR, exist_ok=True)
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
        class_counts[cid] = np.bincount(all_labels[indices],
                                        minlength=NUM_CLASSES).tolist()
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


# ── Local Training ────────────────────────────────────────────────────────────
def train_local(generator, discriminator, loader, opt_G, opt_D, steps):
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

        # Discriminator
        opt_D.zero_grad()
        loss_d_real = criterion(discriminator(real_imgs, real_labels), real_target)
        z           = torch.randn(bs, LATENT_DIM).to(device)
        fake_labels = torch.randint(0, NUM_CLASSES, (bs,)).to(device)
        fake_imgs   = generator(z, fake_labels).detach()
        loss_d_fake = criterion(discriminator(fake_imgs, fake_labels), fake_target)
        loss_D = (loss_d_real + loss_d_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # Generator
        opt_G.zero_grad()
        z           = torch.randn(bs, LATENT_DIM).to(device)
        fake_labels = torch.randint(0, NUM_CLASSES, (bs,)).to(device)
        loss_G = criterion(discriminator(generator(z, fake_labels), fake_labels),
                           real_target)
        loss_G.backward()
        opt_G.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())

    return float(np.mean(g_losses)), float(np.mean(d_losses))


# ── Instruments ───────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_d_logits(generator, discriminator, fixed_z_per_class):
    generator.eval()
    discriminator.eval()
    out = []
    for c in range(NUM_CLASSES):
        z = fixed_z_per_class[c]
        labels = torch.full((z.size(0),), c, dtype=torch.long, device=device)
        out.append(discriminator(generator(z, labels), labels).mean().item())
    generator.train()
    discriminator.train()
    return out


def flatten_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def pairwise_cosine_sims(update_vectors):
    sims = []
    for a, b in itertools.combinations(update_vectors.keys(), 2):
        sims.append(torch.nn.functional.cosine_similarity(
            update_vectors[a].unsqueeze(0),
            update_vectors[b].unsqueeze(0)).item())
    return sims


# ── One Condition ─────────────────────────────────────────────────────────────
def run_condition(condition_name, split_file):
    print(f"\n{'='*62}")
    print(f" DIAGNOSTIC — cGAN {condition_name.upper()} (seed {SEED})")
    print(f"{'='*62}")

    # Reseed per condition so both start from identical initialization —
    # any difference between conditions is then attributable to the split.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    loaders, client_sizes, class_counts = load_client_loaders(split_file)
    present_mask = {cid: [c > 0 for c in class_counts[cid]]
                    for cid in range(NUM_CLIENTS)}

    n_absent = sum(1 for cid in range(NUM_CLIENTS)
                   for p in present_mask[cid] if not p)
    print(f"Client-class cells with zero local samples: {n_absent}/"
          f"{NUM_CLIENTS * NUM_CLASSES}")
    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid}: {class_counts[cid]}")

    global_G  = ConditionalGenerator().to(device)
    client_Gs = {i: copy.deepcopy(global_G) for i in range(NUM_CLIENTS)}
    client_Ds = {i: ConditionalDiscriminator().to(device) for i in range(NUM_CLIENTS)}

    opt_Gs = {i: optim.Adam(client_Gs[i].parameters(), lr=LR_G, betas=(BETA1, 0.999))
              for i in range(NUM_CLIENTS)}
    opt_Ds = {i: optim.Adam(client_Ds[i].parameters(), lr=LR_D, betas=(BETA1, 0.999))
              for i in range(NUM_CLIENTS)}

    eval_gen = torch.Generator(device=device).manual_seed(SEED + 1000)
    fixed_z_per_class = [
        torch.randn(EVAL_BATCH, LATENT_DIM, device=device, generator=eval_gen)
        for _ in range(NUM_CLASSES)
    ]

    log = {
        "condition": condition_name, "seed": SEED,
        "round": [], "g_loss": [], "d_loss": [],
        "d_logit_per_client_class": [],
        "cosine_sim_pairwise_mean": [], "cosine_sim_pairwise_min": [],
        "class_present_mask": present_mask, "class_counts": class_counts,
    }

    for rnd in range(1, NUM_ROUNDS + 1):
        global_state_before = copy.deepcopy(global_G.state_dict())
        # Hoisted out of the client loop — this is the pre-round global
        # parameter vector, identical for every client this round.
        global_flat = flatten_params(global_G).clone()

        for cid in range(NUM_CLIENTS):
            client_Gs[cid].load_state_dict(copy.deepcopy(global_state_before))

        round_g, round_d = [], []
        round_d_logits, update_vectors = {}, {}

        for cid in range(NUM_CLIENTS):
            g_loss, d_loss = train_local(
                client_Gs[cid], client_Ds[cid], loaders[cid],
                opt_Gs[cid], opt_Ds[cid], LOCAL_STEPS)
            round_g.append(g_loss)
            round_d.append(d_loss)
            round_d_logits[cid] = probe_d_logits(
                client_Gs[cid], client_Ds[cid], fixed_z_per_class)
            update_vectors[cid] = flatten_params(client_Gs[cid]) - global_flat

        sims = pairwise_cosine_sims(update_vectors)
        global_G = fedavg(global_G, client_Gs, client_sizes,
                          list(range(NUM_CLIENTS)))

        avg_g, avg_d = float(np.mean(round_g)), float(np.mean(round_d))
        log["round"].append(rnd)
        log["g_loss"].append(avg_g)
        log["d_loss"].append(avg_d)
        log["d_logit_per_client_class"].append(round_d_logits)
        log["cosine_sim_pairwise_mean"].append(float(np.mean(sims)))
        log["cosine_sim_pairwise_min"].append(float(np.min(sims)))

        if rnd % 5 == 0 or rnd == 1:
            print(f"Round {rnd:3d}/{NUM_ROUNDS} | G: {avg_g:.4f} | D: {avg_d:.4f} "
                  f"| cos(mean/min): {np.mean(sims):+.4f}/{np.min(sims):+.4f}")

    path = os.path.join(LOG_DIR,
                        f"diagnostic_{condition_name}_seed{SEED}.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved -> {path}")

    return log


# ── Analysis ──────────────────────────────────────────────────────────────────
def logit_stats(log):
    """Returns (present_mean, absent_mean, per-round gap trend)."""
    mask = log["class_present_mask"]
    present, absent = [], []
    gap_by_round = []

    for round_logits in log["d_logit_per_client_class"]:
        rp, ra = [], []
        for cid_str, logits in round_logits.items():
            cid = int(cid_str)
            for c, val in enumerate(logits):
                if mask[cid][c]:
                    present.append(val); rp.append(val)
                else:
                    absent.append(val);  ra.append(val)
        gap_by_round.append(
            (float(np.mean(rp)) - float(np.mean(ra))) if ra else None)

    return (float(np.mean(present)) if present else None,
            float(np.mean(absent)) if absent else None,
            gap_by_round)


def cos_trend(log):
    c = log["cosine_sim_pairwise_mean"]
    return float(np.mean(c[:15])), float(np.mean(c[15:])), c


def compare(logs):
    iid, non_iid = logs["iid"], logs["non_iid"]

    print(f"\n{'='*72}")
    print(f" SIDE-BY-SIDE COMPARISON (seed {SEED})")
    print(f"{'='*72}")

    # ── G-loss trajectory ──
    print("\n[1] G-loss trajectory (does only non-IID diverge?)")
    for name, lg in (("IID", iid), ("Non-IID", non_iid)):
        first, last = lg["g_loss"][0], lg["g_loss"][-1]
        direction = "DIVERGED (rising)" if last > first else "converged (falling)"
        print(f"  {name:<9} round 1: {first:.4f} -> round 30: {last:.4f}   {direction}")

    # ── D logit gap ──
    print("\n[2] D logit on fake images, present vs absent classes")
    for name, lg in (("IID", iid), ("Non-IID", non_iid)):
        p, a, gaps = logit_stats(lg)
        if a is None:
            print(f"  {name:<9} present {p:+.4f} | absent: N/A "
                  f"(every client holds every class — expected under IID)")
        else:
            valid = [g for g in gaps if g is not None]
            early = float(np.mean(valid[:5]))
            late  = float(np.mean(valid[-5:]))
            trend = ("WIDENS" if late > early else
                     "narrows" if late < early else "flat")
            print(f"  {name:<9} present {p:+.4f} | absent {a:+.4f} "
                  f"| gap {p - a:+.4f}")
            print(f"  {'':<9} gap rounds 1-5: {early:+.4f} -> "
                  f"rounds 26-30: {late:+.4f}   [{trend}]")

    # ── Cosine similarity ──
    print("\n[3] Pairwise cosine similarity of client G updates")
    print("    (THE control comparison — if IID declines the same way,")
    print("     the decline is a FedAvg property, not a non-IID effect)")
    deltas = {}
    for name, lg in (("IID", iid), ("Non-IID", non_iid)):
        h1, h2, series = cos_trend(lg)
        deltas[name] = h2 - h1
        print(f"  {name:<9} rounds 1-15: {h1:+.4f} -> rounds 16-30: {h2:+.4f} "
              f"| delta {h2 - h1:+.4f} | min over run {min(series):+.4f}")

    print(f"\n{'='*72}")
    print(" VERDICT")
    print(f"{'='*72}")

    d_iid, d_non = deltas["IID"], deltas["Non-IID"]
    if d_non < d_iid - 1e-4:
        print(" [3] Cosine similarity declines MORE under non-IID than IID.")
        print("     -> Misalignment is attributable to the data split, not to")
        print("        FedAvg in general. Mechanism claim SUPPORTED.")
    elif abs(d_non - d_iid) <= 1e-4:
        print(" [3] Cosine similarity declines EQUALLY in both conditions.")
        print("     -> The decline is a property of FedAvg, NOT a non-IID")
        print("        effect. Do NOT present it as the divergence mechanism.")
    else:
        print(" [3] Cosine similarity declines LESS under non-IID than IID.")
        print("     -> Contradicts the mechanism. Report honestly.")

    all_series = (iid["cosine_sim_pairwise_mean"]
                  + non_iid["cosine_sim_pairwise_mean"])
    if min(all_series) > 0:
        print("\n Note: cosine similarity never goes negative in either run.")
        print(" Use 'increasingly uncorrelated / misaligned', never")
        print(" 'contradictory', 'opposing', or 'canceling'.")

    print("\n Reminder: n=1 seed. Run --seed 123 and --seed 2024 before")
    print(" making any mechanism claim in the paper.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logs = {}
    for name, split in CONDITIONS.items():
        logs[name] = run_condition(name, split)
    compare(logs)
