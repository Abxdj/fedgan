"""
Centralized Baseline Training — GAN and cGAN
FedGAN Failure Analysis - Anish Bharadwaj

BLOCKING FIX #1 from the adversarial review.

Trains the identical GAN and cGAN architectures CENTRALLY on the
full pooled PathMNIST training set, with a total generator update
budget matched to the federated runs:

    Federated budget = 30 rounds x 100 local steps x 5 clients
                     = 15,000 total generator updates

The centralized runs therefore perform 15,000 update steps on the
pooled data. This is the conservative total-update matching: the
centralized model receives exactly as many gradient updates as the
sum of all federated clients combined.

Why this experiment is mandatory: without it, no observed failure
can be attributed to FEDERATION as opposed to the architecture,
resolution, dataset, or training budget. The
federated-minus-centralized delta per architecture is what
licenses the word "federated" in every claim of the paper.

Usage (run for all 3 seeds):
    python centralized_train.py --seed 42
    python centralized_train.py --seed 123
    python centralized_train.py --seed 2024

Each invocation trains BOTH the centralized GAN and the
centralized cGAN for that seed. Checkpoints:
    checkpoints/generator_central_gan_seed{S}_final.pt
    checkpoints/generator_central_cgan_seed{S}_final.pt
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from dcgan import Generator, Discriminator, LATENT_DIM as UNCOND_LATENT_DIM
from cgan import (ConditionalGenerator, ConditionalDiscriminator,
                  LATENT_DIM as COND_LATENT_DIM, NUM_CLASSES)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
SEED = args.seed

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_STEPS  = 15000     # matches 30 rounds x 100 steps x 5 clients
BATCH_SIZE   = 64
LR_G         = 0.0002
LR_D         = 0.0002
BETA1        = 0.5
LOG_EVERY    = 500       # log-averaged loss every N steps
DATA_DIR     = "./data"
CKPT_DIR     = "./checkpoints"
LOG_DIR      = "./logs"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Seed: {SEED} | Budget: {TOTAL_STEPS} steps")


# ── Pooled Data ───────────────────────────────────────────────────────────────
def get_pooled_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = PathMNIST(split="train", transform=transform,
                        download=False, root=DATA_DIR)
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=True, drop_last=True, num_workers=0)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# ── Centralized Unconditional GAN ─────────────────────────────────────────────
def train_central_gan(loader):
    print(f"\n{'='*55}")
    print(f" [Seed {SEED}] Centralized GAN — pooled data, {TOTAL_STEPS} steps")
    print(f"{'='*55}")

    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    data = infinite(loader)
    log = {"step": [], "g_loss": [], "d_loss": []}
    g_window, d_window = [], []

    G.train(); D.train()
    for step in range(1, TOTAL_STEPS + 1):
        real_imgs, _ = next(data)
        real_imgs = real_imgs.to(device)
        bs = real_imgs.size(0)

        real_labels = torch.ones(bs, 1).to(device)
        fake_labels = torch.zeros(bs, 1).to(device)

        # Discriminator
        opt_D.zero_grad()
        loss_d_real = criterion(D(real_imgs), real_labels)
        z = torch.randn(bs, UNCOND_LATENT_DIM, 1, 1).to(device)
        fake_imgs = G(z).detach()
        loss_d_fake = criterion(D(fake_imgs), fake_labels)
        loss_D = (loss_d_real + loss_d_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # Generator
        opt_G.zero_grad()
        z = torch.randn(bs, UNCOND_LATENT_DIM, 1, 1).to(device)
        loss_G = criterion(D(G(z)), real_labels)
        loss_G.backward()
        opt_G.step()

        g_window.append(loss_G.item())
        d_window.append(loss_D.item())

        if step % LOG_EVERY == 0:
            avg_g, avg_d = np.mean(g_window), np.mean(d_window)
            log["step"].append(step)
            log["g_loss"].append(avg_g)
            log["d_loss"].append(avg_d)
            print(f"Step {step:5d}/{TOTAL_STEPS} | G: {avg_g:.4f} | D: {avg_d:.4f}")
            g_window, d_window = [], []

    ckpt = os.path.join(CKPT_DIR, f"generator_central_gan_seed{SEED}_final.pt")
    torch.save(G.state_dict(), ckpt)
    print(f"Checkpoint saved → {ckpt}")

    log_path = os.path.join(LOG_DIR, f"losses_central_gan_seed{SEED}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Losses saved    → {log_path}")


# ── Centralized Conditional GAN ───────────────────────────────────────────────
def train_central_cgan(loader):
    print(f"\n{'='*55}")
    print(f" [Seed {SEED}] Centralized cGAN — pooled data, {TOTAL_STEPS} steps")
    print(f"{'='*55}")

    G = ConditionalGenerator().to(device)
    D = ConditionalDiscriminator().to(device)
    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    data = infinite(loader)
    log = {"step": [], "g_loss": [], "d_loss": []}
    g_window, d_window = [], []

    G.train(); D.train()
    for step in range(1, TOTAL_STEPS + 1):
        real_imgs, real_lbls = next(data)
        real_imgs = real_imgs.to(device)
        real_lbls = real_lbls.squeeze().long().to(device)
        bs = real_imgs.size(0)

        real_target = torch.ones(bs, 1).to(device)
        fake_target = torch.zeros(bs, 1).to(device)

        # Discriminator
        opt_D.zero_grad()
        loss_d_real = criterion(D(real_imgs, real_lbls), real_target)
        z = torch.randn(bs, COND_LATENT_DIM).to(device)
        gen_lbls = torch.randint(0, NUM_CLASSES, (bs,)).to(device)
        fake_imgs = G(z, gen_lbls).detach()
        loss_d_fake = criterion(D(fake_imgs, gen_lbls), fake_target)
        loss_D = (loss_d_real + loss_d_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # Generator
        opt_G.zero_grad()
        z = torch.randn(bs, COND_LATENT_DIM).to(device)
        gen_lbls = torch.randint(0, NUM_CLASSES, (bs,)).to(device)
        loss_G = criterion(D(G(z, gen_lbls), gen_lbls), real_target)
        loss_G.backward()
        opt_G.step()

        g_window.append(loss_G.item())
        d_window.append(loss_D.item())

        if step % LOG_EVERY == 0:
            avg_g, avg_d = np.mean(g_window), np.mean(d_window)
            log["step"].append(step)
            log["g_loss"].append(avg_g)
            log["d_loss"].append(avg_d)
            print(f"Step {step:5d}/{TOTAL_STEPS} | G: {avg_g:.4f} | D: {avg_d:.4f}")
            g_window, d_window = [], []

    ckpt = os.path.join(CKPT_DIR, f"generator_central_cgan_seed{SEED}_final.pt")
    torch.save(G.state_dict(), ckpt)
    print(f"Checkpoint saved → {ckpt}")

    log_path = os.path.join(LOG_DIR, f"losses_central_cgan_seed{SEED}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Losses saved    → {log_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = get_pooled_loader()
    train_central_gan(loader)
    train_central_cgan(loader)
    print(f"\nBoth centralized baselines complete for seed {SEED}.")
