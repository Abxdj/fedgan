"""
Precision & Recall for Generative Models
FedGAN Failure Analysis - Anish Bharadwaj

Implements the Improved Precision and Recall metric
(Kynkaanniemi et al., NeurIPS 2019) using k-NN manifold estimation.

Why this matters:
FID collapses image quality and diversity into a single number.
A generator that produces perfect images of only 2 out of 9 classes
can have a similar FID to one that produces mediocre images of all
9 classes. Precision and Recall separate these two failure modes:

  Precision = fraction of GENERATED samples that fall within the
              REAL data manifold. Low precision = poor image quality.

  Recall    = fraction of REAL samples that fall within the
              GENERATED data manifold. Low recall = mode collapse
              (the generator is not covering the full real
              distribution, even if what it does produce looks fine).

This directly explains the singular covariance matrix warning seen
in the cGAN Non-IID FID computation: we expect that experiment to
show reasonable precision but collapsed recall.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import PathMNIST
from dcgan import Generator, LATENT_DIM as UNCOND_LATENT_DIM
from cgan import ConditionalGenerator, LATENT_DIM as COND_LATENT_DIM, NUM_CLASSES

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "./data"
CKPT_DIR   = "./checkpoints"
LOG_DIR    = "./logs"
N_SAMPLES  = 5000
BATCH_SIZE = 128
K_NEIGHBORS = 5          # standard choice from the original paper
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
    "GAN — IID Baseline"    : ("generator_iid_final.pt",               False),
    "GAN — Non-IID"         : ("generator_non_iid_final.pt",           False),
    "GAN — Noisy Client"    : ("generator_noisy_client_final.pt",      False),
    "cGAN — IID Baseline"   : ("generator_cgan_iid_final.pt",          True),
    "cGAN — Non-IID"        : ("generator_cgan_non_iid_final.pt",      True),
    "cGAN — Noisy Client"   : ("generator_cgan_noisy_client_final.pt", True),
}

print(f"Device: {DEVICE}\n")


# ── Inception Feature Extractor (same as fid_all.py) ─────────────────────────
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


# ── Feature Extraction ────────────────────────────────────────────────────────
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
        imgs = generator(z)
        all_imgs.append(imgs.cpu())
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
        imgs   = generator(z, labels)
        all_imgs.append(imgs.cpu())
        generated += bs
    all_imgs = torch.cat(all_imgs)[:n_samples]
    feats = []
    for i in range(0, len(all_imgs), BATCH_SIZE):
        batch = ((all_imgs[i:i+BATCH_SIZE] + 1) / 2).clamp(0, 1).to(DEVICE)
        feats.append(inception(batch).cpu().numpy())
    return np.concatenate(feats)


# ── Precision & Recall (Kynkaanniemi et al. 2019) ─────────────────────────────
def compute_pairwise_distances(X, Y):
    """
    Euclidean distance matrix between rows of X and rows of Y.

    Uses scipy's cdist rather than the norm-expansion formula
    (||x||^2 + ||y||^2 - 2*x.y). That formula subtracts two large
    numbers to recover a small one, which loses precision badly in
    float32 on high-magnitude, high-dimensional features like raw
    Inception-v3 pool activations (unbounded, often large scale).
    The resulting cancellation error was corrupting k-NN radius
    estimates for nearby points, artificially inflating detected
    "duplicate" points (radius ~ 0) and collapsing recall to near
    zero regardless of true generator diversity. cdist computes
    differences directly and is numerically stable for this case.
    """
    from scipy.spatial.distance import cdist
    return cdist(X.astype(np.float64), Y.astype(np.float64), metric="euclidean")


def compute_manifold_radii(features, k=K_NEIGHBORS):
    """
    For each point, radius = distance to its k-th nearest neighbor
    within the same feature set. This defines a local manifold ball
    around each real (or fake) sample.
    """
    dists = compute_pairwise_distances(features, features)
    # Sort each row, skip the 0th (self-distance = 0)
    sorted_dists = np.sort(dists, axis=1)
    radii = sorted_dists[:, k]   # k-th neighbor (0-indexed, skip self at 0)
    return radii


def compute_precision_recall(real_feats, fake_feats, k=K_NEIGHBORS):
    """
    Precision: fraction of fake samples within the real manifold.
    Recall:    fraction of real samples within the fake manifold.
    """
    real_radii = compute_manifold_radii(real_feats, k)
    fake_radii = compute_manifold_radii(fake_feats, k)

    # Precision: for each fake sample, is it within radius of ANY real sample?
    dist_fake_to_real = compute_pairwise_distances(fake_feats, real_feats)
    within_real_manifold = (dist_fake_to_real <= real_radii[np.newaxis, :])
    precision = within_real_manifold.any(axis=1).mean()

    # Recall: for each real sample, is it within radius of ANY fake sample?
    dist_real_to_fake = compute_pairwise_distances(real_feats, fake_feats)
    within_fake_manifold = (dist_real_to_fake <= fake_radii[np.newaxis, :])
    recall = within_fake_manifold.any(axis=1).mean()

    return float(precision), float(recall)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading Inception-v3...")
    inception  = InceptionFeatures().to(DEVICE)

    print("Extracting real image features...")
    real_feats = get_real_features(inception)
    print(f"Real features: {real_feats.shape}\n")

    results = {}

    for exp_name, (ckpt_file, is_conditional) in EXPERIMENTS.items():
        ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"MISSING: {ckpt_path} — skipping\n")
            continue

        print(f"[{exp_name}]")
        if is_conditional:
            G = ConditionalGenerator().to(DEVICE)
            G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            fake_feats = get_fake_features_cond(G, inception)
        else:
            G = Generator().to(DEVICE)
            G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            fake_feats = get_fake_features_uncond(G, inception)

        precision, recall = compute_precision_recall(real_feats, fake_feats)
        results[exp_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }
        print(f"  Precision: {precision:.4f}  (image quality/fidelity)")
        print(f"  Recall   : {recall:.4f}  (distribution coverage/diversity)\n")

    # ── Summary ──
    print("="*70)
    print(" PRECISION & RECALL RESULTS — ALL EXPERIMENTS")
    print("="*70)
    print(f"  {'Experiment':<26} {'Precision':>10} {'Recall':>10}   {'Interpretation'}")
    print(f"  {'-'*66}")
    for name, vals in results.items():
        p, r = vals["precision"], vals["recall"]
        if r < 0.15:
            note = "SEVERE mode collapse"
        elif r < 0.35:
            note = "Moderate mode collapse"
        elif p < 0.35:
            note = "Poor image quality"
        else:
            note = "Healthy"
        print(f"  {name:<26} {p:>10.4f} {r:>10.4f}   {note}")
    print("="*70)
    print("Precision = fidelity (quality of generated images)")
    print("Recall    = diversity (coverage of real data distribution)")
    print("Low recall + reasonable precision = mode collapse signature\n")

    log_path = os.path.join(LOG_DIR, "precision_recall_all.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {log_path}")
