"""
Centralized Baseline Evaluation — FID + Downstream Accuracy
FedGAN Failure Analysis - Anish Bharadwaj

Completes BLOCKING FIX #1: evaluates the 6 centralized checkpoints
(GAN and cGAN x 3 seeds) on the same FID and downstream-accuracy
protocols used for the federated runs, so the federated-vs-
centralized delta can be computed per architecture.

Run AFTER centralized_train.py has completed for all 3 seeds.

Usage:
    python evaluate_centralized.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
from medmnist import PathMNIST
from scipy import linalg
import warnings
warnings.filterwarnings("ignore")
from dcgan import Generator, LATENT_DIM as UNCOND_LATENT_DIM
from cgan import ConditionalGenerator, LATENT_DIM as COND_LATENT_DIM, NUM_CLASSES

DATA_DIR          = "./data"
CKPT_DIR          = "./checkpoints"
LOG_DIR           = "./logs"
N_SAMPLES         = 5000      # for FID — matches federated protocol
N_SYNTHETIC       = 10000     # for downstream — matches federated protocol
BATCH_SIZE        = 128
CLASSIFIER_EPOCHS = 15
CLASSIFIER_LR     = 0.001
SEEDS             = [42, 123, 2024]
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CENTRAL_EXPERIMENTS = {
    "Centralized GAN"  : ("central_gan",  False),
    "Centralized cGAN" : ("central_cgan", True),
}

print(f"Device: {DEVICE}\n")


# ── Inception ─────────────────────────────────────────────────────────────────
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


# ── Classifier ────────────────────────────────────────────────────────────────
class TissueClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128 * 9, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


# ── Feature / Generation Helpers ──────────────────────────────────────────────
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
def generate_images(generator, is_conditional, n_samples):
    """Returns (imgs in [-1,1], labels). Same protocols as federated eval."""
    generator.eval()
    imgs, lbls, generated = [], [], 0
    while generated < n_samples:
        bs = min(BATCH_SIZE, n_samples - generated)
        if is_conditional:
            z = torch.randn(bs, COND_LATENT_DIM).to(DEVICE)
            labels = torch.tensor([i % NUM_CLASSES for i in range(generated,
                                   generated + bs)], dtype=torch.long).to(DEVICE)
            imgs.append(generator(z, labels).cpu())
            lbls.append(labels.cpu())
        else:
            z = torch.randn(bs, UNCOND_LATENT_DIM, 1, 1).to(DEVICE)
            imgs.append(generator(z).cpu())
            lbls.append(torch.tensor([i % NUM_CLASSES
                                       for i in range(generated, generated + bs)],
                                      dtype=torch.long))
        generated += bs
    return torch.cat(imgs)[:n_samples], torch.cat(lbls)[:n_samples]


@torch.no_grad()
def features_from_imgs(imgs, inception):
    feats = []
    for i in range(0, len(imgs), BATCH_SIZE):
        batch = ((imgs[i:i+BATCH_SIZE] + 1) / 2).clamp(0, 1).to(DEVICE)
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


def train_classifier(train_loader, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model     = TissueClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    for _ in range(CLASSIFIER_EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE).squeeze().long()
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


@torch.no_grad()
def evaluate_classifier(model, test_loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.squeeze().long()
        preds  = model(imgs).argmax(dim=1).cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading Inception-v3...")
    inception = InceptionFeatures().to(DEVICE)
    print("Extracting real features...")
    real_feats = get_real_features(inception)
    print(f"Real features: {real_feats.shape}\n")

    transform   = transforms.Compose([transforms.ToTensor()])
    test_ds     = PathMNIST(split="test", transform=transform,
                            download=False, root=DATA_DIR)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    results = {name: {"fid": [], "accuracy": []} for name in CENTRAL_EXPERIMENTS}

    for exp_name, (tag, is_cond) in CENTRAL_EXPERIMENTS.items():
        print("="*60)
        print(f" {exp_name}")
        print("="*60)

        for seed in SEEDS:
            ckpt = os.path.join(CKPT_DIR, f"generator_{tag}_seed{seed}_final.pt")
            if not os.path.exists(ckpt):
                print(f"  MISSING seed {seed}: {ckpt}")
                continue

            if is_cond:
                G = ConditionalGenerator().to(DEVICE)
            else:
                G = Generator().to(DEVICE)
            G.load_state_dict(torch.load(ckpt, map_location=DEVICE))

            # FID
            fake_imgs_fid, _ = generate_images(G, is_cond, N_SAMPLES)
            fake_feats = features_from_imgs(fake_imgs_fid, inception)
            fid = calculate_fid(real_feats, fake_feats)
            results[exp_name]["fid"].append(fid)

            # Downstream
            syn_imgs, syn_labels = generate_images(G, is_cond, N_SYNTHETIC)
            syn_imgs = ((syn_imgs + 1) / 2).clamp(0, 1)
            syn_loader = DataLoader(TensorDataset(syn_imgs, syn_labels),
                                    batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0)
            clf = train_classifier(syn_loader, seed)
            acc = evaluate_classifier(clf, test_loader)
            results[exp_name]["accuracy"].append(acc)

            print(f"  Seed {seed}: FID = {fid:.2f} | Accuracy = {acc:.2f}%")
        print()

    # ── Summary ──
    print("="*72)
    print(" CENTRALIZED BASELINE RESULTS (mean +/- std, n=3 seeds)")
    print("="*72)
    print(f"  {'Configuration':<22} {'FID':>16}   {'Downstream Acc':>18}")
    print(f"  {'-'*66}")

    summary = {}
    for name, vals in results.items():
        if not vals["fid"]:
            continue
        fid_m, fid_s = np.mean(vals["fid"]), np.std(vals["fid"])
        acc_m, acc_s = np.mean(vals["accuracy"]), np.std(vals["accuracy"])
        summary[name] = {
            "fid_mean": round(fid_m, 2), "fid_std": round(fid_s, 2),
            "fid_seeds": [round(v, 2) for v in vals["fid"]],
            "acc_mean": round(acc_m, 2), "acc_std": round(acc_s, 2),
            "acc_seeds": [round(v, 2) for v in vals["accuracy"]],
        }
        print(f"  {name:<22} {fid_m:>8.2f} +/- {fid_s:<5.2f} "
              f"{acc_m:>10.2f}% +/- {acc_s:<5.2f}")
    print("="*72)

    print("\nNEXT STEP — compute these deltas for the paper:")
    print("  federated_IID_FID  −  centralized_FID   (per architecture)")
    print("  federated_IID_acc  −  centralized_acc   (per architecture)")
    print("  Compare against your federated multi-seed results in")
    print("  logs/fid_multiseed.json and logs/silent_failure_multiseed.json")

    log_path = os.path.join(LOG_DIR, "centralized_results.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {log_path}")
