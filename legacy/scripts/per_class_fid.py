"""
Per-Class (Intra-Class) FID
FedGAN Failure Analysis - Anish Bharadwaj

Standard aggregate FID hides which specific tissue classes a
generator handles well or poorly. This script computes FID
separately for each of the 9 PathMNIST classes, giving an
image-quality counterpart to the existing per-class accuracy
heatmap (Figure 4 in silent_failure_all.py).

Important framing note (include this in the paper):
- For the CONDITIONAL GAN, each class's synthetic images are
  generated using that class's explicit label — this is a true
  test of class-specific generation quality.
- For the UNCONDITIONAL GAN, there is no class control. Images
  are drawn from the same undifferentiated output distribution
  and bucketed by round-robin label assignment (same scheme used
  in silent_failure_all.py for consistency). We therefore expect
  per-class FID to be roughly UNIFORM across classes for the
  unconditional GAN — this uniformity is itself evidence that the
  architecture cannot differentiate by class, contrasted against
  the conditional GAN's expected per-class variation.

Uses the same FID computation as fid_all.py — no new metric,
just sliced by class. Same 5000-sample budget, split across
classes according to real per-class availability.
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

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "./data"
CKPT_DIR     = "./checkpoints"
LOG_DIR      = "./logs"
FIG_DIR      = "./figures"
MAX_PER_CLASS = 300     # cap per class — balances stability vs compute time
BATCH_SIZE   = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
    "GAN — IID"          : ("generator_iid_final.pt",               False),
    "GAN — Non-IID"       : ("generator_non_iid_final.pt",           False),
    "GAN — Noisy Client"  : ("generator_noisy_client_final.pt",      False),
    "cGAN — IID"          : ("generator_cgan_iid_final.pt",          True),
    "cGAN — Non-IID"      : ("generator_cgan_non_iid_final.pt",      True),
    "cGAN — Noisy Client" : ("generator_cgan_noisy_client_final.pt", True),
}

CLASS_NAMES = ["adipose", "background", "debris", "lymphocytes",
               "mucus", "smooth muscle", "normal colon",
               "cancer stroma", "adenocarcinoma"]

print(f"Device: {DEVICE}\n")


# ── Inception Feature Extractor ───────────────────────────────────────────────
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
def extract_features_from_tensor(imgs, inception):
    """imgs expected in [-1,1] range, shape (N,3,28,28)."""
    feats = []
    for i in range(0, len(imgs), BATCH_SIZE):
        batch = ((imgs[i:i+BATCH_SIZE] + 1) / 2).clamp(0, 1).to(DEVICE)
        feats.append(inception(batch).cpu().numpy())
    return np.concatenate(feats)


# ── Real Data Grouped by Class ────────────────────────────────────────────────
def get_real_images_by_class():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = PathMNIST(split="test", transform=transform,
                        download=False, root=DATA_DIR)

    by_class = {c: [] for c in range(NUM_CLASSES)}
    for img, label in dataset:
        c = int(label.item()) if hasattr(label, "item") else int(label)
        if len(by_class[c]) < MAX_PER_CLASS:
            by_class[c].append(img)

    for c in by_class:
        by_class[c] = torch.stack(by_class[c])

    return by_class


# ── Synthetic Generation Per Class ────────────────────────────────────────────
@torch.no_grad()
def generate_unconditional_pool(generator, n_total):
    """Generate a large unconditional pool, to be bucketed by round-robin label."""
    generator.eval()
    imgs, generated = [], 0
    while generated < n_total:
        bs   = min(BATCH_SIZE, n_total - generated)
        z    = torch.randn(bs, UNCOND_LATENT_DIM, 1, 1).to(DEVICE)
        imgs.append(generator(z).cpu())
        generated += bs
    return torch.cat(imgs)[:n_total]


@torch.no_grad()
def generate_conditional_per_class(generator, class_id, n_samples):
    """Generate n_samples images explicitly conditioned on class_id."""
    generator.eval()
    imgs, generated = [], 0
    while generated < n_samples:
        bs     = min(BATCH_SIZE, n_samples - generated)
        z      = torch.randn(bs, COND_LATENT_DIM).to(DEVICE)
        labels = torch.full((bs,), class_id, dtype=torch.long).to(DEVICE)
        imgs.append(generator(z, labels).cpu())
        generated += bs
    return torch.cat(imgs)[:n_samples]


# ── FID ───────────────────────────────────────────────────────────────────────
def calculate_fid(real_feats, fake_feats):
    mu_r, mu_g   = np.mean(real_feats, axis=0), np.mean(fake_feats, axis=0)
    sig_r, sig_g = np.cov(real_feats, rowvar=False), np.cov(fake_feats, rowvar=False)
    diff         = mu_r - mu_g
    covmean, _   = linalg.sqrtm(sig_r.dot(sig_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig_r + sig_g - 2 * covmean))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading Inception-v3...")
    inception = InceptionFeatures().to(DEVICE)

    print("Loading real images grouped by class...")
    real_by_class = get_real_images_by_class()
    for c in range(NUM_CLASSES):
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(real_by_class[c])} real images")

    print("\nExtracting real per-class features...")
    real_feats_by_class = {
        c: extract_features_from_tensor(real_by_class[c], inception)
        for c in range(NUM_CLASSES)
    }

    # results[experiment_name][class_id] = fid
    results = {name: {} for name in EXPERIMENTS}

    for exp_name, (ckpt_file, is_conditional) in EXPERIMENTS.items():
        ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"\nMISSING: {ckpt_path} — skipping")
            continue

        print(f"\n[{exp_name}]")

        if is_conditional:
            G = ConditionalGenerator().to(DEVICE)
            G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

            for c in range(NUM_CLASSES):
                n_needed = len(real_by_class[c])
                fake_imgs = generate_conditional_per_class(G, c, n_needed)
                fake_feats = extract_features_from_tensor(fake_imgs, inception)
                fid = calculate_fid(real_feats_by_class[c], fake_feats)
                results[exp_name][c] = round(fid, 2)
                print(f"  {CLASS_NAMES[c]:<20} FID: {fid:.2f}")

        else:
            G = Generator().to(DEVICE)
            G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

            # Generate one large pool, bucket by round-robin label
            total_needed = sum(len(real_by_class[c]) for c in range(NUM_CLASSES))
            pool = generate_unconditional_pool(G, total_needed)
            pool_labels = torch.tensor([i % NUM_CLASSES for i in range(total_needed)])

            for c in range(NUM_CLASSES):
                mask = pool_labels == c
                fake_imgs = pool[mask][:len(real_by_class[c])]
                fake_feats = extract_features_from_tensor(fake_imgs, inception)
                fid = calculate_fid(real_feats_by_class[c], fake_feats)
                results[exp_name][c] = round(fid, 2)
                print(f"  {CLASS_NAMES[c]:<20} FID: {fid:.2f}")

    # ── Save ──
    log_path = os.path.join(LOG_DIR, "per_class_fid.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {log_path}")

    # ── Uniformity check for unconditional GANs ──
    print("\n" + "="*60)
    print(" UNIFORMITY CHECK (unconditional GAN — expected near-flat)")
    print("="*60)
    for exp_name, (_, is_cond) in EXPERIMENTS.items():
        if is_cond or exp_name not in results:
            continue
        vals = list(results[exp_name].values())
        print(f"  {exp_name:<22} std across classes: {np.std(vals):.2f}"
              f"  (range: {min(vals):.1f}–{max(vals):.1f})")

    print("\n" + "="*60)
    print(" VARIATION CHECK (conditional GAN — expected real spread)")
    print("="*60)
    for exp_name, (_, is_cond) in EXPERIMENTS.items():
        if not is_cond or exp_name not in results:
            continue
        vals = list(results[exp_name].values())
        print(f"  {exp_name:<22} std across classes: {np.std(vals):.2f}"
              f"  (range: {min(vals):.1f}–{max(vals):.1f})")

    print("\nPer-class FID complete.")
