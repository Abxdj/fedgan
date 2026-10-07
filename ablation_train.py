"""
Unified Federated Training — blocking fixes C1 and C2
FedGAN Failure Analysis - Anish Bharadwaj

One script, four flags, so every new run shares identical training code:

  --arch  {gan, cgan}
  --split {iid, non_iid, non_iid_perseed}
            non_iid          = the original frozen Dirichlet draw
            non_iid_perseed  = a fresh draw per seed (make_perseed_splits.py)
  --sync  {G, DG}
            G  = generator-only aggregation (Sync-G, Fan & Liu 2020) — the
                 protocol behind every headline result
            DG = generator AND discriminator aggregation (Sync D&G, the
                 scheme used by FedGAN, Rasouli et al. 2020)
  --seed  int

What each blocking fix runs:
  C1  Aggregation ablation:  --arch cgan --split {iid,non_iid} --sync DG
      Same frozen partition as the headline runs, so aggregation is the ONLY
      variable changed. Question: does the non-IID divergence survive once
      discriminators are synchronized too?
  C2  Partition robustness:  --arch {gan,cgan} --split non_iid_perseed --sync G
      Same protocol as the headline runs, new partition per seed. Question:
      does the divergence replicate across partitions, not just seeds?

Hyperparameters are identical to the headline multiseed scripts. Per-client
Adam state persists across rounds while weights are overwritten by the
broadcast global model — the convention the headline runs use for G, applied
to D as well under DG.

Output:
  checkpoints/generator_{arch}_{split}_sync{sync}_seed{S}_final.pt
  logs/losses_{arch}_{split}_sync{sync}_seed{S}.json
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
from dcgan import Generator, Discriminator, LATENT_DIM as U_LATENT
from cgan import (ConditionalGenerator, ConditionalDiscriminator,
                  LATENT_DIM as C_LATENT, NUM_CLASSES)

parser = argparse.ArgumentParser()
parser.add_argument("--arch",  choices=["gan", "cgan"], required=True)
parser.add_argument("--split", choices=["iid", "non_iid", "non_iid_perseed"],
                    required=True)
parser.add_argument("--sync",  choices=["G", "DG"], default="G")
parser.add_argument("--seed",  type=int, required=True)
args = parser.parse_args()

NUM_CLIENTS = 5
NUM_ROUNDS  = 30
LOCAL_STEPS = 100
BATCH_SIZE  = 64
LR_G        = 0.0002
LR_D        = 0.0002
BETA1       = 0.5
DATA_DIR    = "./data"
CKPT_DIR    = "./checkpoints"
LOG_DIR     = "./logs"

COND = args.arch == "cgan"
TAG  = f"{args.arch}_{args.split}_sync{args.sync}"
SPLIT_FILE = {
    "iid"            : os.path.join(DATA_DIR, "iid_split.json"),
    "non_iid"        : os.path.join(DATA_DIR, "non_iid_split.json"),
    "non_iid_perseed": os.path.join(DATA_DIR, f"non_iid_split_seed{args.seed}.json"),
}[args.split]

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | {TAG} | seed {args.seed}")


# ── Data ──────────────────────────────────────────────────────────────────────
def load_client_loaders():
    if not os.path.exists(SPLIT_FILE):
        raise SystemExit(f"Missing {SPLIT_FILE} — run make_perseed_splits.py first.")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = PathMNIST(split="train", transform=transform,
                        download=False, root=DATA_DIR)
    labels = np.asarray(dataset.labels).reshape(-1)
    with open(SPLIT_FILE) as f:
        split = json.load(f)

    loaders, sizes, counts = {}, {}, {}
    for k, idx in split.items():
        cid = int(k)
        if len(idx) < BATCH_SIZE:
            raise SystemExit(f"Client {cid} has {len(idx)} samples (< batch size).")
        loaders[cid] = DataLoader(Subset(dataset, idx), batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=0)
        sizes[cid]  = len(idx)
        counts[cid] = np.bincount(labels[idx], minlength=NUM_CLASSES).tolist()
    return loaders, sizes, counts


# ── Models ────────────────────────────────────────────────────────────────────
def new_G():
    return (ConditionalGenerator() if COND else Generator()).to(device)


def new_D():
    return (ConditionalDiscriminator() if COND else Discriminator()).to(device)


def fedavg(global_model, client_models, sizes):
    """Dataset-size-weighted average of every state_dict entry."""
    total = sum(sizes.values())
    new_state = {k: torch.zeros_like(v, dtype=torch.float32)
                 for k, v in global_model.state_dict().items()}
    for cid, m in client_models.items():
        w = sizes[cid] / total
        for k, v in m.state_dict().items():
            new_state[k] += w * v.float()
    global_model.load_state_dict(new_state)
    return global_model


# ── Local Training ────────────────────────────────────────────────────────────
def train_local(G, D, loader, opt_G, opt_D, steps):
    crit = nn.BCEWithLogitsLoss()
    G.train(); D.train()
    g_losses, d_losses = [], []
    it = iter(loader)

    for _ in range(steps):
        try:
            imgs, lbls = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, lbls = next(it)

        imgs = imgs.to(device)
        lbls = lbls.view(-1).long().to(device)
        bs   = imgs.size(0)
        ones  = torch.ones(bs, 1, device=device)
        zeros = torch.zeros(bs, 1, device=device)

        # Discriminator
        opt_D.zero_grad()
        if COND:
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

        # Generator
        opt_G.zero_grad()
        if COND:
            z  = torch.randn(bs, C_LATENT, device=device)
            fl = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            loss_G = crit(D(G(z, fl), fl), ones)
        else:
            z = torch.randn(bs, U_LATENT, 1, 1, device=device)
            loss_G = crit(D(G(z)), ones)
        loss_G.backward()
        opt_G.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())

    return float(np.mean(g_losses)), float(np.mean(d_losses))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    loaders, sizes, counts = load_client_loaders()
    empty = sum(1 for c in counts.values() for x in c if x == 0)
    print(f"Split: {SPLIT_FILE} | empty client-class cells: "
          f"{empty}/{NUM_CLIENTS * NUM_CLASSES}")

    global_G  = new_G()
    client_Gs = {i: copy.deepcopy(global_G) for i in range(NUM_CLIENTS)}

    if args.sync == "DG":
        # Synced D: all clients start from one global D
        global_D  = new_D()
        client_Ds = {i: copy.deepcopy(global_D) for i in range(NUM_CLIENTS)}
    else:
        # Local D: independent initializations, as in the headline runs
        global_D  = None
        client_Ds = {i: new_D() for i in range(NUM_CLIENTS)}

    opt_Gs = {i: optim.Adam(client_Gs[i].parameters(), lr=LR_G, betas=(BETA1, 0.999))
              for i in range(NUM_CLIENTS)}
    opt_Ds = {i: optim.Adam(client_Ds[i].parameters(), lr=LR_D, betas=(BETA1, 0.999))
              for i in range(NUM_CLIENTS)}

    log = {"tag": TAG, "seed": args.seed, "arch": args.arch,
           "split": args.split, "sync": args.sync,
           "empty_cells": empty, "class_counts": counts,
           "round": [], "g_loss": [], "d_loss": [], "client_g_loss": []}

    for rnd in range(1, NUM_ROUNDS + 1):
        g_state = copy.deepcopy(global_G.state_dict())
        for i in range(NUM_CLIENTS):
            client_Gs[i].load_state_dict(g_state)
        if global_D is not None:
            d_state = copy.deepcopy(global_D.state_dict())
            for i in range(NUM_CLIENTS):
                client_Ds[i].load_state_dict(d_state)

        round_g, round_d = [], []
        for i in range(NUM_CLIENTS):
            g, d = train_local(client_Gs[i], client_Ds[i], loaders[i],
                               opt_Gs[i], opt_Ds[i], LOCAL_STEPS)
            round_g.append(g)
            round_d.append(d)

        global_G = fedavg(global_G, client_Gs, sizes)
        if global_D is not None:
            global_D = fedavg(global_D, client_Ds, sizes)

        log["round"].append(rnd)
        log["g_loss"].append(float(np.mean(round_g)))
        log["d_loss"].append(float(np.mean(round_d)))
        log["client_g_loss"].append(round_g)

        if rnd % 5 == 0 or rnd == 1:
            print(f"Round {rnd:3d}/{NUM_ROUNDS} | G: {np.mean(round_g):.4f} "
                  f"| D: {np.mean(round_d):.4f}")

    early = float(np.mean(log["g_loss"][:5]))
    late  = float(np.mean(log["g_loss"][-5:]))
    log.update({"early_g": early, "late_g": late, "diverged": late > early})

    ckpt = os.path.join(CKPT_DIR, f"generator_{TAG}_seed{args.seed}_final.pt")
    torch.save(global_G.state_dict(), ckpt)
    with open(os.path.join(LOG_DIR, f"losses_{TAG}_seed{args.seed}.json"), "w") as f:
        json.dump(log, f, indent=2)

    status = "DIVERGED (rising)" if late > early else "converged (falling)"
    print(f"\nG-loss rounds 1-5: {early:.4f} -> rounds 26-30: {late:.4f}  [{status}]")
    print(f"Checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
