"""
Multi-Seed Federated Training — Conditional GAN (cGAN)
FedGAN Failure Analysis - Anish Bharadwaj

Runs all 3 conditional experiments (IID, Non-IID, Noisy Client)
for a single seed, passed via command line. Run this 3 times with
different seeds to get the full multi-seed sweep:

    python cgan_train_multiseed.py --seed 42
    python cgan_train_multiseed.py --seed 123
    python cgan_train_multiseed.py --seed 2024
"""

import os
import copy
import json
import argparse
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
parser.add_argument("--seed", type=int, required=True,
                     help="Random seed for this run (e.g. 42, 123, 2024)")
args = parser.parse_args()
SEED = args.seed

# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    NUM_CLIENTS       = 5
    NUM_ROUNDS         = 30
    LOCAL_STEPS         = 100
    BATCH_SIZE          = 64
    LR_G                = 0.0002
    LR_D                = 0.0002
    BETA1               = 0.5
    DATA_DIR            = "./data"
    CKPT_DIR            = "./checkpoints"
    LOG_DIR             = "./logs"

cfg = Config()
os.makedirs(cfg.CKPT_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR, exist_ok=True)

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
                        download=False, root=cfg.DATA_DIR)
    with open(split_file) as f:
        split = json.load(f)
    loaders, sizes = {}, {}
    for client_id, indices in split.items():
        cid = int(client_id)
        subset = Subset(dataset, indices)
        loaders[cid] = DataLoader(subset, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=0)
        sizes[cid] = len(indices)
    return loaders, sizes


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
def train_local(client_id, generator, discriminator, loader,
                opt_G, opt_D, steps, noisy=False):
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

        if noisy and client_id == 0:
            noise = torch.randn_like(real_imgs) * 0.5
            real_imgs = torch.clamp(real_imgs + noise, -1, 1)

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


# ── Main Federated Training ───────────────────────────────────────────────────
def federated_train_cgan(split_file, experiment_tag, noisy=False):
    print(f"\n{'='*55}")
    print(f" [Seed {SEED}] Federated cGAN Training — {experiment_tag.upper()}")
    print(f"{'='*55}")

    loaders, client_sizes = load_client_loaders(split_file)

    global_G  = ConditionalGenerator().to(device)
    client_Gs = {i: copy.deepcopy(global_G) for i in range(cfg.NUM_CLIENTS)}
    client_Ds = {i: ConditionalDiscriminator().to(device) for i in range(cfg.NUM_CLIENTS)}

    opt_Gs = {i: optim.Adam(client_Gs[i].parameters(),
                             lr=cfg.LR_G, betas=(cfg.BETA1, 0.999))
              for i in range(cfg.NUM_CLIENTS)}
    opt_Ds = {i: optim.Adam(client_Ds[i].parameters(),
                             lr=cfg.LR_D, betas=(cfg.BETA1, 0.999))
              for i in range(cfg.NUM_CLIENTS)}

    log = {"round": [], "g_loss": [], "d_loss": []}

    for rnd in range(1, cfg.NUM_ROUNDS + 1):
        selected = list(range(cfg.NUM_CLIENTS))
        for cid in selected:
            client_Gs[cid].load_state_dict(copy.deepcopy(global_G.state_dict()))

        round_g, round_d = [], []
        for cid in selected:
            g_loss, d_loss = train_local(
                client_id=cid, generator=client_Gs[cid],
                discriminator=client_Ds[cid], loader=loaders[cid],
                opt_G=opt_Gs[cid], opt_D=opt_Ds[cid],
                steps=cfg.LOCAL_STEPS, noisy=noisy
            )
            round_g.append(g_loss)
            round_d.append(d_loss)

        global_G = fedavg(global_G, client_Gs, client_sizes, selected)

        avg_g, avg_d = np.mean(round_g), np.mean(round_d)
        log["round"].append(rnd)
        log["g_loss"].append(avg_g)
        log["d_loss"].append(avg_d)
        if rnd % 5 == 0 or rnd == 1:
            print(f"Round {rnd:3d}/{cfg.NUM_ROUNDS} | G: {avg_g:.4f} | D: {avg_d:.4f}")

    tag_with_seed = f"{experiment_tag}_seed{SEED}"
    ckpt_path = os.path.join(cfg.CKPT_DIR, f"generator_{tag_with_seed}_final.pt")
    torch.save(global_G.state_dict(), ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    log_path = os.path.join(cfg.LOG_DIR, f"losses_{tag_with_seed}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Losses saved    → {log_path}")

    return global_G, log


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    federated_train_cgan("./data/iid_split.json", "cgan_iid", noisy=False)
    federated_train_cgan("./data/non_iid_split.json", "cgan_non_iid", noisy=False)
    federated_train_cgan("./data/iid_split.json", "cgan_noisy_client", noisy=True)

    print(f"\nAll 3 conditional experiments complete for seed {SEED}.")
