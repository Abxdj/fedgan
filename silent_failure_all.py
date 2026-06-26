"""
Silent Failure Test — All Experiments
FedGAN Failure Analysis - Anish Bharadwaj

Trains a classifier on synthetic data from each of 6 generators,
then tests every classifier on the same real PathMNIST test set.

The accuracy gap = silent failure gap.

Key difference for cGAN:
- Labels are assigned correctly (class i → generator told to produce class i)
- This is the fair test — cGAN should perform better than unconditional GAN
  IF the federated training preserved class-discriminative features.
- If cGAN still fails, the problem is federated training, not label assignment.
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

# Config 
DATA_DIR          = "./data"
CKPT_DIR          = "./checkpoints"
LOG_DIR           = "./logs"
N_SYNTHETIC       = 10000
BATCH_SIZE        = 128
CLASSIFIER_EPOCHS = 15
CLASSIFIER_LR     = 0.001
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
    "GAN — IID Baseline"    : ("generator_iid_final.pt",               False),
    "GAN — Non-IID"         : ("generator_non_iid_final.pt",           False),
    "GAN — Noisy Client"    : ("generator_noisy_client_final.pt",      False),
    "cGAN — IID Baseline"   : ("generator_cgan_iid_final.pt",          True),
    "cGAN — Non-IID"        : ("generator_cgan_non_iid_final.pt",      True),
    "cGAN — Noisy Client"   : ("generator_cgan_noisy_client_final.pt", True),
}

CLASS_NAMES = ["adipose", "background", "debris", "lymphocytes",
               "mucus", "smooth muscle", "normal colon",
               "cancer stroma", "adenocarcinoma"]

print(f"Device: {DEVICE}\n")


# Classifier 
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


# Data
def get_test_loader():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset   = PathMNIST(split="test", transform=transform,
                          download=False, root=DATA_DIR)
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0)


@torch.no_grad()
def generate_unconditional(generator, n_samples):
    """Round-robin label assignment — same as before, intentionally naive."""
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
    """
    Correct label assignment — generator is told which class to produce.
    This is the fair test for cGAN. If accuracy is still low,
    the federated training failed to preserve class features,
    not the label assignment.
    """
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


# Train Classifier
def train_classifier(train_loader):
    model     = TissueClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(CLASSIFIER_EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE).squeeze().long()
            optimizer.zero_grad()
            loss   = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d}/{CLASSIFIER_EPOCHS} "
                  f"| Loss: {total_loss/len(train_loader):.4f}")

    return model


# Evaluate 
@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    per_class_correct = torch.zeros(NUM_CLASSES)
    per_class_total   = torch.zeros(NUM_CLASSES)

    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.squeeze().long()
        preds  = model(imgs).argmax(dim=1).cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        for c in range(NUM_CLASSES):
            mask = labels == c
            per_class_correct[c] += (preds[mask] == labels[mask]).sum()
            per_class_total[c]   += mask.sum()

    overall      = correct / total * 100
    per_class    = (per_class_correct /
                    per_class_total.clamp(min=1) * 100).tolist()
    return overall, per_class


# Main 
if __name__ == "__main__":
    test_loader = get_test_loader()
    results     = {}

    # Reference: real data 
    print("="*58)
    print(" REFERENCE — Classifier trained on REAL data")
    print("="*58)
    transform     = transforms.Compose([transforms.ToTensor()])
    real_train_ds = PathMNIST(split="train", transform=transform,
                              download=False, root=DATA_DIR)
    real_loader   = DataLoader(real_train_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0)
    ref_model   = train_classifier(real_loader)
    ref_acc, ref_per = evaluate(ref_model, test_loader)
    results["Real Data (Reference)"] = {
        "overall": round(ref_acc, 2),
        "per_class": [round(x, 2) for x in ref_per]
    }
    print(f"  → Accuracy: {ref_acc:.2f}%\n")

    # Synthetic classifiers 
    for exp_name, (ckpt_file, is_conditional) in EXPERIMENTS.items():
        ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"MISSING: {ckpt_path} — skipping\n")
            continue

        print("="*58)
        print(f" {exp_name}")
        print("="*58)

        if is_conditional:
            G = ConditionalGenerator().to(DEVICE)
            G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            syn_imgs, syn_labels = generate_conditional(G, N_SYNTHETIC)
        else:
            G = Generator().to(DEVICE)
            G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            syn_imgs, syn_labels = generate_unconditional(G, N_SYNTHETIC)

        print(f"  Generated {N_SYNTHETIC} images | "
              f"Label distribution: {torch.bincount(syn_labels).tolist()}")

        syn_loader = DataLoader(TensorDataset(syn_imgs, syn_labels),
                                batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0)
        model      = train_classifier(syn_loader)
        acc, per   = evaluate(model, test_loader)

        results[exp_name] = {
            "overall": round(acc, 2),
            "per_class": [round(x, 2) for x in per]
        }
        print(f"  → Accuracy on Real Test Set: {acc:.2f}%\n")

    # Summary 
    ref = results["Real Data (Reference)"]["overall"]

    print("="*58)
    print(" SILENT FAILURE RESULTS — ALL EXPERIMENTS")
    print("="*58)
    print(f"  {'Experiment':<30} {'Accuracy':>9} {'Gap vs Real':>12}")
    print(f"  {'-'*54}")
    for name, data in results.items():
        acc = data["overall"]
        gap = f"{acc - ref:+.2f}%" if name != "Real Data (Reference)" else "—"
        print(f"  {name:<30} {acc:>8.2f}% {gap:>12}")
    print("="*58)

    # Per-class breakdown
    print("\nPer-class accuracy (%):")
    header = f"  {'Class':<20}"
    for name in results:
        short = name.replace("GAN — ", "").replace("cGAN — ", "c")[:10]
        header += f"  {short:>10}"
    print(header)
    print("  " + "-" * (20 + 12 * len(results)))
    for i, cls in enumerate(CLASS_NAMES):
        row = f"  {cls:<20}"
        for name in results:
            row += f"  {results[name]['per_class'][i]:>9.1f}%"
        print(row)

    # Save
    log_path = os.path.join(LOG_DIR, "silent_failure_all.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {log_path}")
    print("\nAll silent failure experiments complete.")
