"""
Multi-Seed Silent Failure Aggregation
FedGAN Failure Analysis - Anish Bharadwaj

Trains a downstream classifier on synthetic data from each of the
18 checkpoints (6 experiments x 3 seeds), evaluates on real test
data, and aggregates into mean +/- std per experiment.

The real-data reference classifier is trained once (not seed-varied)
since it establishes the ceiling and is not part of the experimental
variable being tested.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from medmnist import PathMNIST
from dcgan import Generator, LATENT_DIM as UNCOND_LATENT_DIM
from cgan import ConditionalGenerator, LATENT_DIM as COND_LATENT_DIM, NUM_CLASSES

DATA_DIR          = "./data"
CKPT_DIR          = "./checkpoints"
LOG_DIR           = "./logs"
N_SYNTHETIC        = 10000
BATCH_SIZE         = 128
CLASSIFIER_EPOCHS  = 15
CLASSIFIER_LR      = 0.001
SEEDS              = [42, 123, 2024]
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
    "GAN — IID Baseline"    : ("iid",               False),
    "GAN — Non-IID"         : ("non_iid",           False),
    "GAN — Noisy Client"    : ("noisy_client",      False),
    "cGAN — IID Baseline"   : ("cgan_iid",          True),
    "cGAN — Non-IID"        : ("cgan_non_iid",      True),
    "cGAN — Noisy Client"   : ("cgan_noisy_client", True),
}

print(f"Device: {DEVICE}\n")


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


def get_test_loader():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset   = PathMNIST(split="test", transform=transform,
                          download=False, root=DATA_DIR)
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0)


@torch.no_grad()
def generate_unconditional(generator, n_samples):
    generator.eval()
    imgs, generated = [], 0
    while generated < n_samples:
        bs   = min(BATCH_SIZE, n_samples - generated)
        z    = torch.randn(bs, UNCOND_LATENT_DIM, 1, 1).to(DEVICE)
        imgs.append(generator(z).cpu())
        generated += bs
    imgs   = torch.cat(imgs)[:n_samples]
    imgs   = ((imgs + 1) / 2).clamp(0, 1)
    labels = torch.tensor([i % NUM_CLASSES for i in range(n_samples)],
                          dtype=torch.long)
    return imgs, labels


@torch.no_grad()
def generate_conditional(generator, n_samples):
    generator.eval()
    imgs, all_labels, generated = [], [], 0
    while generated < n_samples:
        bs     = min(BATCH_SIZE, n_samples - generated)
        z      = torch.randn(bs, COND_LATENT_DIM).to(DEVICE)
        labels = torch.tensor([i % NUM_CLASSES for i in range(generated,
                               generated + bs)], dtype=torch.long).to(DEVICE)
        imgs.append(generator(z, labels).cpu())
        all_labels.append(labels.cpu())
        generated += bs
    imgs   = torch.cat(imgs)[:n_samples]
    imgs   = ((imgs + 1) / 2).clamp(0, 1)
    labels = torch.cat(all_labels)[:n_samples]
    return imgs, labels


def train_classifier(train_loader):
    model     = TissueClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(CLASSIFIER_EPOCHS):
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
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.squeeze().long()
        preds  = model(imgs).argmax(dim=1).cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100


if __name__ == "__main__":
    test_loader = get_test_loader()

    # ── Reference: real data (single run, not seed-varied) ──
    print("="*60)
    print(" REFERENCE — Classifier trained on REAL data")
    print("="*60)
    transform     = transforms.Compose([transforms.ToTensor()])
    real_train_ds = PathMNIST(split="train", transform=transform,
                              download=False, root=DATA_DIR)
    real_loader   = DataLoader(real_train_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0)
    ref_model = train_classifier(real_loader)
    ref_acc   = evaluate(ref_model, test_loader)
    print(f"  Real Data Reference Accuracy: {ref_acc:.2f}%\n")

    # raw[exp_name] = [acc_seed42, acc_seed123, acc_seed2024]
    raw = {name: [] for name in EXPERIMENTS}

    for exp_name, (tag, is_conditional) in EXPERIMENTS.items():
        print("="*60)
        print(f" {exp_name}")
        print("="*60)

        for seed in SEEDS:
            ckpt_path = os.path.join(CKPT_DIR, f"generator_{tag}_seed{seed}_final.pt")
            if not os.path.exists(ckpt_path):
                print(f"  MISSING seed {seed}: {ckpt_path}")
                continue

            if is_conditional:
                G = ConditionalGenerator().to(DEVICE)
                G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                syn_imgs, syn_labels = generate_conditional(G, N_SYNTHETIC)
            else:
                G = Generator().to(DEVICE)
                G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                syn_imgs, syn_labels = generate_unconditional(G, N_SYNTHETIC)

            syn_loader = DataLoader(TensorDataset(syn_imgs, syn_labels),
                                    batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0)
            model = train_classifier(syn_loader)
            acc   = evaluate(model, test_loader)
            raw[exp_name].append(acc)
            print(f"  Seed {seed}: Accuracy = {acc:.2f}%")
        print()

    # ── Aggregate ──
    print("="*70)
    print(" MULTI-SEED SILENT FAILURE RESULTS (mean +/- std across 3 seeds)")
    print("="*70)
    print(f"  Real Data (Reference): {ref_acc:.2f}%\n")
    print(f"  {'Experiment':<26} {'Mean':>8} {'Std':>8}   {'Individual Seeds'}")
    print(f"  {'-'*66}")

    summary = {"Real Data (Reference)": {"mean": round(ref_acc, 2), "std": 0.0}}
    for name, vals in raw.items():
        if not vals:
            continue
        mean, std = np.mean(vals), np.std(vals)
        summary[name] = {"mean": round(mean, 2), "std": round(std, 2),
                         "seeds": [round(v, 2) for v in vals]}
        seed_str = ", ".join(f"{v:.1f}" for v in vals)
        print(f"  {name:<26} {mean:>8.2f} {std:>8.2f}   [{seed_str}]")
    print("="*70)

    log_path = os.path.join(LOG_DIR, "silent_failure_multiseed.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {log_path}")
