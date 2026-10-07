"""
Reference Classifiers — Multi-Seed, Full and Matched-Size
FedGAN Failure Analysis - Anish Bharadwaj

BLOCKING FIXES #2 and #4 from the adversarial review.

Trains the downstream reference classifiers properly:

  (a) Real-full     : all 89,996 real training images, 3 seeds.
      Your earlier single runs of this drifted 74.99% -> 85.89%
      across invocations — an unstable ceiling. Must be reported
      as mean +/- std like everything else.

  (b) Real-matched  : a random 10,000-image subset of real data,
      3 seeds. This matches the synthetic training set size and
      isolates the QUALITY effect from the QUANTITY effect:

          (real-full - real-matched)  = pure data-quantity effect
          (real-matched - synthetic)  = pure data-quality effect

      Without this control, part of the reported "silent failure
      gap" is simply that the reference saw 9x more data.

Usage (one invocation runs everything):
    python reference_classifiers.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PathMNIST
from cgan import NUM_CLASSES

DATA_DIR          = "./data"
LOG_DIR           = "./logs"
N_MATCHED         = 10000
BATCH_SIZE        = 128
CLASSIFIER_EPOCHS = 15
CLASSIFIER_LR     = 0.001
SEEDS             = [42, 123, 2024]
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(LOG_DIR, exist_ok=True)
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


def get_datasets():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = PathMNIST(split="train", transform=transform,
                         download=False, root=DATA_DIR)
    test_ds  = PathMNIST(split="test", transform=transform,
                         download=False, root=DATA_DIR)
    return train_ds, test_ds


def train_classifier(train_loader, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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
    train_ds, test_ds = get_datasets()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    results = {"real_full": [], "real_matched_10k": []}

    # ── (a) Real-full, 3 seeds ──
    print("="*60)
    print(" REAL-FULL REFERENCE (89,996 images, 3 seeds)")
    print("="*60)
    for seed in SEEDS:
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)
        model = train_classifier(loader, seed)
        acc = evaluate(model, test_loader)
        results["real_full"].append(acc)
        print(f"  Seed {seed}: {acc:.2f}%")

    # ── (b) Real-matched 10k, 3 seeds ──
    print("\n" + "="*60)
    print(f" REAL-MATCHED REFERENCE ({N_MATCHED} images, 3 seeds)")
    print("="*60)
    for seed in SEEDS:
        # Seed the subset selection too, so each seed gets its own
        # independent random 10k draw — variance then reflects both
        # subset choice and training stochasticity, matching how the
        # synthetic sets differ per seed.
        rng = np.random.RandomState(seed)
        subset_idx = rng.choice(len(train_ds), N_MATCHED, replace=False)
        subset = Subset(train_ds, subset_idx.tolist())
        loader = DataLoader(subset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)
        model = train_classifier(loader, seed)
        acc = evaluate(model, test_loader)
        results["real_matched_10k"].append(acc)
        print(f"  Seed {seed}: {acc:.2f}%")

    # ── Summary ──
    print("\n" + "="*70)
    print(" REFERENCE CLASSIFIER RESULTS (mean +/- std, n=3 seeds)")
    print("="*70)
    summary = {}
    for name, vals in results.items():
        mean, std = np.mean(vals), np.std(vals)
        summary[name] = {"mean": round(mean, 2), "std": round(std, 2),
                         "seeds": [round(v, 2) for v in vals]}
        seed_str = ", ".join(f"{v:.1f}" for v in vals)
        print(f"  {name:<20} {mean:>7.2f} +/- {std:<5.2f}  [{seed_str}]")

    quantity_effect = (np.mean(results["real_full"])
                       - np.mean(results["real_matched_10k"]))
    print(f"\n  Data-QUANTITY effect (full − matched): "
          f"{quantity_effect:+.2f} percentage points")
    print("  Data-QUALITY effect = (matched − synthetic) — compute against")
    print("  your synthetic multi-seed means from silent_failure_multiseed.json")
    print("="*70)

    summary["quantity_effect_pp"] = round(quantity_effect, 2)
    log_path = os.path.join(LOG_DIR, "reference_classifiers.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {log_path}")
