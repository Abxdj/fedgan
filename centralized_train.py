"""
Centralized Baseline Training (v2, with --steps) — blocking fix C3
FedGAN Failure Analysis - Anish Bharadwaj

Replaces the previous centralized_train.py. Backward compatible: the default
--steps 15000 writes the SAME checkpoint names as before, so existing
15k checkpoints and evaluate_centralized.py keep working.

Why a second budget: the existing 15,000-step baselines match the SUM of
federated generator updates (30 rounds x 100 steps x 5 clients). Each
individual federated model, however, receives only 3,000 updates. A reviewer
can argue either matching is the fair one, so we report both and interpret
the federated result against the bracket:

    --steps 15000   -> generator_central_{gan,cgan}_seed{S}_final.pt   (existing)
    --steps 3000    -> generator_central_{gan,cgan}_3k_seed{S}_final.pt

If federated IID is worse than BOTH centralized budgets, the federation cost
is robust to budget choice. If it falls between them, the cost is
budget-dependent and must be reported as a range.

Usage:
    python centralized_train.py --steps 3000 --seed 42
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
from dcgan import Generator, Discriminator, LATENT_DIM as U_LATENT
from cgan import (ConditionalGenerator, ConditionalDiscriminator,
                  LATENT_DIM as C_LATENT, NUM_CLASSES)

parser = argparse.ArgumentParser()
parser.add_argument("--seed",  type=int, required=True)
parser.add_argument("--steps", type=int, default=15000)
args = parser.parse_args()
SEED, TOTAL_STEPS = args.seed, args.steps

SUFFIX    = "" if TOTAL_STEPS == 15000 else f"_{TOTAL_STEPS // 1000}k"
LOG_EVERY = 500 if TOTAL_STEPS >= 10000 else 250
BATCH_SIZE, LR, BETA1 = 64, 0.0002, 0.5
DATA_DIR, CKPT_DIR, LOG_DIR = "./data", "./checkpoints", "./logs"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Seed: {SEED} | Budget: {TOTAL_STEPS} steps")


def get_pooled_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    ds = PathMNIST(split="train", transform=transform, download=False, root=DATA_DIR)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                      drop_last=True, num_workers=0)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def train(loader, conditional):
    name = "cgan" if conditional else "gan"
    tag  = f"central_{name}{SUFFIX}"
    print(f"\n{'='*58}\n [Seed {SEED}] Centralized {name.upper()} — "
          f"{TOTAL_STEPS} steps\n{'='*58}")

    G = (ConditionalGenerator() if conditional else Generator()).to(device)
    D = (ConditionalDiscriminator() if conditional else Discriminator()).to(device)
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
    crit  = nn.BCEWithLogitsLoss()

    data = infinite(loader)
    log = {"step": [], "g_loss": [], "d_loss": []}
    gw, dw = [], []
    G.train(); D.train()

    for step in range(1, TOTAL_STEPS + 1):
        imgs, lbls = next(data)
        imgs = imgs.to(device)
        lbls = lbls.view(-1).long().to(device)
        bs = imgs.size(0)
        ones  = torch.ones(bs, 1, device=device)
        zeros = torch.zeros(bs, 1, device=device)

        opt_D.zero_grad()
        if conditional:
            z  = torch.randn(bs, C_LATENT, device=device)
            fl = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            fake = G(z, fl).detach()
            loss_D = 0.5 * (crit(D(imgs, lbls), ones) + crit(D(fake, fl), zeros))
        else:
            z = torch.randn(bs, U_LATENT, 1, 1, device=device)
            fake = G(z).detach()
            loss_D = 0.5 * (crit(D(imgs), ones) + crit(D(fake), zeros))
        loss_D.backward()
        opt_D.step()

        opt_G.zero_grad()
        if conditional:
            z  = torch.randn(bs, C_LATENT, device=device)
            fl = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            loss_G = crit(D(G(z, fl), fl), ones)
        else:
            z = torch.randn(bs, U_LATENT, 1, 1, device=device)
            loss_G = crit(D(G(z)), ones)
        loss_G.backward()
        opt_G.step()

        gw.append(loss_G.item())
        dw.append(loss_D.item())
        if step % LOG_EVERY == 0:
            log["step"].append(step)
            log["g_loss"].append(float(np.mean(gw)))
            log["d_loss"].append(float(np.mean(dw)))
            print(f"Step {step:5d}/{TOTAL_STEPS} | G: {np.mean(gw):.4f} "
                  f"| D: {np.mean(dw):.4f}")
            gw, dw = [], []

    ckpt = os.path.join(CKPT_DIR, f"generator_{tag}_seed{SEED}_final.pt")
    torch.save(G.state_dict(), ckpt)
    with open(os.path.join(LOG_DIR, f"losses_{tag}_seed{SEED}.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Checkpoint -> {ckpt}")


if __name__ == "__main__":
    loader = get_pooled_loader()
    train(loader, conditional=False)
    train(loader, conditional=True)
    print(f"\nBoth centralized baselines ({TOTAL_STEPS} steps) complete for seed {SEED}.")
