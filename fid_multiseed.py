"""
Multi-Seed FID Aggregation
FedGAN Failure Analysis - Anish Bharadwaj

Computes FID for all 18 checkpoints (6 experiments x 3 seeds),
then aggregates into mean +/- std per experiment for the paper.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import PathMNIST
from scipy import linalg
import warnings
warnings.filterwarnings("ignore")
from dcgan import Generator, LATENT_DIM as UNCOND_LATENT_DIM
from cgan import ConditionalGenerator, LATENT_DIM as COND_LATENT_DIM, NUM_CLASSES

DATA_DIR   = "./data"
CKPT_DIR   = "./checkpoints"
LOG_DIR    = "./logs"
N_SAMPLES  = 5000
BATCH_SIZE = 128
SEEDS      = [42, 123, 2024]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
    "GAN — IID Baseline"    : ("iid",               False),
    "GAN — Non-IID"         : ("non_iid",           False),
    "GAN — Noisy Client"    : ("noisy_client",      False),
    "cGAN — IID Baseline"   : ("cgan_iid",          True),
    "cGAN — Non-IID"        : ("cgan_non_iid",      True),
    "cGAN — Noisy Client"   : ("cgan_noisy_client", True),
}

print(f"Device: {DEVICE}\n")


class InceptionFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(299, 299),
                                       mode="bilinear", align_corners=False)
        return self.features(x).view(x.size(0), -1)


@torch.no_grad()
def get_real_features(inception, n_samples=N_SAMPLES):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = PathMNIST(split="test", transform=transform,
                        download=False, root=DATA_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=0)
    feats, total = [], 0
    for imgs, _ in loader:
        imgs = ((imgs + 1) / 2).clamp(0, 1).to(DEVICE)
        feats.append(inception(imgs).cpu().numpy())
        total += imgs.size(0)
        if total >= n_samples:
            break
    return np.concatenate(feats)[:n_samples]


@torch.no_grad()
def get_fake_features_uncond(generator, inception, n_samples=N_SAMPLES):
    generator.eval()
    all_imgs, generated = [], 0
    while generated < n_samples:
        bs   = min(BATCH_SIZE, n_samples - generated)
        z    = torch.randn(bs, UNCOND_LATENT_DIM, 1, 1).to(DEVICE)
        all_imgs.append(generator(z).cpu())
        generated += bs
    all_imgs = torch.cat(all_imgs)[:n_samples]
    feats = []
    for i in range(0, len(all_imgs), BATCH_SIZE):
        batch = ((all_imgs[i:i+BATCH_SIZE] + 1) / 2).clamp(0, 1).to(DEVICE)
        feats.append(inception(batch).cpu().numpy())
    return np.concatenate(feats)


@torch.no_grad()
def get_fake_features_cond(generator, inception, n_samples=N_SAMPLES):
    generator.eval()
    all_imgs, generated = [], 0
    while generated < n_samples:
        bs     = min(BATCH_SIZE, n_samples - generated)
        z      = torch.randn(bs, COND_LATENT_DIM).to(DEVICE)
        labels = torch.tensor([i % NUM_CLASSES for i in range(generated,
                               generated + bs)]).to(DEVICE)
        all_imgs.append(generator(z, labels).cpu())
        generated += bs
    all_imgs = torch.cat(all_imgs)[:n_samples]
    feats = []
    for i in range(0, len(all_imgs), BATCH_SIZE):
        batch = ((all_imgs[i:i+BATCH_SIZE] + 1) / 2).clamp(0, 1).to(DEVICE)
        feats.append(inception(batch).cpu().numpy())
    return np.concatenate(feats)


def calculate_fid(real_feats, fake_feats):
    mu_r, mu_g   = np.mean(real_feats, axis=0), np.mean(fake_feats, axis=0)
    sig_r, sig_g = np.cov(real_feats, rowvar=False), np.cov(fake_feats, rowvar=False)
    diff         = mu_r - mu_g
    covmean, _   = linalg.sqrtm(sig_r.dot(sig_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig_r + sig_g - 2 * covmean))


if __name__ == "__main__":
    print("Loading Inception-v3...")
    inception = InceptionFeatures().to(DEVICE)
    print("Extracting real image features (shared across all runs)...")
    real_feats = get_real_features(inception)
    print(f"Real features: {real_feats.shape}\n")

    # raw[exp_name] = [fid_seed42, fid_seed123, fid_seed2024]
    raw = {name: [] for name in EXPERIMENTS}

    for exp_name, (tag, is_conditional) in EXPERIMENTS.items():
        print(f"[{exp_name}]")
        for seed in SEEDS:
            ckpt_path = os.path.join(CKPT_DIR, f"generator_{tag}_seed{seed}_final.pt")
            if not os.path.exists(ckpt_path):
                print(f"  MISSING seed {seed}: {ckpt_path}")
                continue

            if is_conditional:
                G = ConditionalGenerator().to(DEVICE)
                G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                fake_feats = get_fake_features_cond(G, inception)
            else:
                G = Generator().to(DEVICE)
                G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                fake_feats = get_fake_features_uncond(G, inception)

            fid = calculate_fid(real_feats, fake_feats)
            raw[exp_name].append(fid)
            print(f"  Seed {seed}: FID = {fid:.2f}")
        print()

    # ── Aggregate ──
    print("="*70)
    print(" MULTI-SEED FID RESULTS (mean +/- std across 3 seeds)")
    print("="*70)
    print(f"  {'Experiment':<26} {'Mean':>8} {'Std':>8}   {'Individual Seeds'}")
    print(f"  {'-'*66}")

    summary = {}
    for name, vals in raw.items():
        if not vals:
            continue
        mean, std = np.mean(vals), np.std(vals)
        summary[name] = {"mean": round(mean, 2), "std": round(std, 2),
                         "seeds": [round(v, 2) for v in vals]}
        seed_str = ", ".join(f"{v:.1f}" for v in vals)
        print(f"  {name:<26} {mean:>8.2f} {std:>8.2f}   [{seed_str}]")
    print("="*70)

    log_path = os.path.join(LOG_DIR, "fid_multiseed.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {log_path}")
