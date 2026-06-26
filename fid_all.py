"""
FID Measurement — All Experiments
FedGAN Failure Analysis - Anish Bharadwaj

Covers:
1. Unconditional GAN: IID, Non-IID, Noisy Client
2. Conditional GAN  : IID, Non-IID, Noisy Client
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
from dcgan import Generator, LATENT_DIM as UNCOND_LATENT_DIM
from cgan import ConditionalGenerator, LATENT_DIM as COND_LATENT_DIM, NUM_CLASSES

# Config  
DATA_DIR   = "./data"
CKPT_DIR   = "./checkpoints"
LOG_DIR    = "./logs"
N_SAMPLES  = 5000
BATCH_SIZE = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
    # name : (checkpoint_file, is_conditional)
    "GAN — IID Baseline"    : ("generator_iid_final.pt",              False),
    "GAN — Non-IID"         : ("generator_non_iid_final.pt",          False),
    "GAN — Noisy Client"    : ("generator_noisy_client_final.pt",     False),
    "cGAN — IID Baseline"   : ("generator_cgan_iid_final.pt",         True),
    "cGAN — Non-IID"        : ("generator_cgan_non_iid_final.pt",     True),
    "cGAN — Noisy Client"   : ("generator_cgan_noisy_client_final.pt",True),
}

print(f"Device: {DEVICE}\n")


# Inception Feature Extractor
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


# Feature Extraction
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


# FID
def calculate_fid(real_feats, fake_feats):
    mu_r, mu_g   = np.mean(real_feats, axis=0), np.mean(fake_feats, axis=0)
    sig_r, sig_g = np.cov(real_feats, rowvar=False), np.cov(fake_feats, rowvar=False)
    diff         = mu_r - mu_g
    covmean, _   = linalg.sqrtm(sig_r.dot(sig_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig_r + sig_g - 2 * covmean))


# Main
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

        fid = calculate_fid(real_feats, fake_feats)
        results[exp_name] = round(fid, 2)
        print(f"  FID: {fid:.2f}\n")

    # Summary
    print("="*55)
    print(" FID RESULTS — ALL EXPERIMENTS")
    print("="*55)
    print(f"  {'Experiment':<28} {'FID':>8}  {'vs GAN-IID':>10}")
    print(f"  {'-'*50}")
    baseline = results.get("GAN — IID Baseline", 1)
    for name, fid in results.items():
        delta = f"{((fid - baseline) / baseline * 100):+.1f}%" \
                if name != "GAN — IID Baseline" else "—"
        print(f"  {name:<28} {fid:>8.2f}  {delta:>10}")
    print("="*55)
    print("Lower FID = better\n")

    log_path = os.path.join(LOG_DIR, "fid_results_all.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {log_path}")
