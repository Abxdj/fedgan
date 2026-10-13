"""
Pseudo-Labeled Downstream Evaluation — Unconditional GAN
FedGAN Failure Analysis - Anish Bharadwaj

Addresses the labeling-protocol confound in the unconditional results.

THE PROBLEM
The unconditional GAN provides no class control, so the main downstream
protocol assigns labels round-robin. Those labels are random with
respect to image content BY CONSTRUCTION, which guarantees chance-level
accuracy regardless of generation quality. Proof: the centralized GAN
has the best FID in the entire study (84.13) yet scores 4.59%. Those
numbers measure the protocol, not the generator.

THE FIX
Label the unconditional generator's output using an independent
classifier trained on real data (a "pseudo-labeler"), then run the
standard downstream protocol on those pseudo-labeled images. If the
generator has learned class-discriminative structure, the pseudo-labeler
can recover it and the downstream classifier should score well above
chance. If accuracy stays at chance, the generator genuinely produces
images without recoverable class structure — which is then a real
finding, not an artifact.

IMPORTANT INTERPRETATION LIMIT
Pseudo-labeling introduces a ceiling: the downstream classifier can only
be as good as the pseudo-labeler's own accuracy on synthetic images,
which is unmeasurable (there is no ground truth for a generated image).
The pseudo-labeler is also trained on real data, so this measures
"class structure recoverable by a real-data classifier", not absolute
generation quality. Report accordingly — this rescues the comparison
from being a pure artifact, it does not make it a clean quality metric.

We also record the pseudo-label distribution entropy. If the generator
mode-collapsed, the pseudo-labeler will assign most images to a few
classes, and low entropy flags that directly.

Usage:
    python pseudo_label_eval.py
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
from cgan import NUM_CLASSES

DATA_DIR          = "./data"
CKPT_DIR          = "./checkpoints"
LOG_DIR           = "./logs"
N_SYNTHETIC       = 10000
BATCH_SIZE        = 128
CLASSIFIER_EPOCHS = 15
CLASSIFIER_LR     = 0.001
SEEDS             = [42, 123, 2024]
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All unconditional generators, federated and centralized
EXPERIMENTS = {
    "GAN — IID Baseline"  : "iid",
    "GAN — Non-IID"       : "non_iid",
    "GAN — Noisy Client"  : "noisy_client",
    "Centralized GAN"     : "central_gan",
}

os.makedirs(LOG_DIR, exist_ok=True)
print(f"Device: {DEVICE}\n")


# ── Classifier (identical to the main downstream protocol) ───────────────────
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


# ── Generation and pseudo-labeling ────────────────────────────────────────────
@torch.no_grad()
def generate_unconditional(generator, n_samples):
    """Returns images in [0,1], no labels — labels come from the pseudo-labeler."""
    generator.eval()
    imgs, generated = [], 0
    while generated < n_samples:
        bs = min(BATCH_SIZE, n_samples - generated)
        z  = torch.randn(bs, UNCOND_LATENT_DIM, 1, 1).to(DEVICE)
        imgs.append(generator(z).cpu())
        generated += bs
    imgs = torch.cat(imgs)[:n_samples]
    return ((imgs + 1) / 2).clamp(0, 1)


@torch.no_grad()
def pseudo_label(images, labeler):
    """Assign labels to synthetic images using a real-data-trained classifier."""
    labeler.eval()
    labels, confidences = [], []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE].to(DEVICE)
        logits = labeler(batch)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        labels.append(pred.cpu())
        confidences.append(conf.cpu())
    return torch.cat(labels), torch.cat(confidences)


def label_entropy(labels):
    """Normalized entropy of the pseudo-label distribution.
    1.0 = perfectly uniform across 9 classes, 0.0 = all one class.
    Low entropy indicates the generator collapsed onto few classes."""
    counts = np.bincount(labels.numpy(), minlength=NUM_CLASSES).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(NUM_CLASSES))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds  = PathMNIST(split="train", transform=transform,
                          download=False, root=DATA_DIR)
    test_ds   = PathMNIST(split="test", transform=transform,
                          download=False, root=DATA_DIR)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # ── Train the pseudo-labeler on real data (one per seed) ──
    print("="*64)
    print(" PSEUDO-LABELER — trained on real data")
    print("="*64)
    labelers, labeler_accs = {}, []
    for seed in SEEDS:
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)
        m = train_classifier(loader, seed)
        acc = evaluate(m, test_loader)
        labelers[seed] = m
        labeler_accs.append(acc)
        print(f"  Seed {seed}: {acc:.2f}% on real test set")
    print(f"  Mean pseudo-labeler accuracy: {np.mean(labeler_accs):.2f}%")
    print("  (this is the effective ceiling — a pseudo-labeled downstream")
    print("   classifier cannot exceed the quality of its own labels)\n")

    results = {}

    for exp_name, tag in EXPERIMENTS.items():
        print("="*64)
        print(f" {exp_name} — pseudo-labeled")
        print("="*64)

        accs, entropies, confs = [], [], []

        for seed in SEEDS:
            ckpt = os.path.join(CKPT_DIR, f"generator_{tag}_seed{seed}_final.pt")
            if not os.path.exists(ckpt):
                print(f"  MISSING seed {seed}: {ckpt}")
                continue

            G = Generator().to(DEVICE)
            G.load_state_dict(torch.load(ckpt, map_location=DEVICE))

            syn_imgs = generate_unconditional(G, N_SYNTHETIC)
            syn_labels, syn_conf = pseudo_label(syn_imgs, labelers[seed])

            ent = label_entropy(syn_labels)
            dist = np.bincount(syn_labels.numpy(),
                               minlength=NUM_CLASSES).tolist()

            syn_loader = DataLoader(TensorDataset(syn_imgs, syn_labels),
                                    batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0)
            clf = train_classifier(syn_loader, seed)
            acc = evaluate(clf, test_loader)

            accs.append(acc)
            entropies.append(ent)
            confs.append(float(syn_conf.mean()))

            print(f"  Seed {seed}: acc {acc:.2f}% | label entropy {ent:.3f} "
                  f"| mean conf {syn_conf.mean():.3f}")
            print(f"           pseudo-label distribution: {dist}")

        if accs:
            results[exp_name] = {
                "acc_mean": round(float(np.mean(accs)), 2),
                "acc_std":  round(float(np.std(accs)), 2),
                "acc_seeds": [round(a, 2) for a in accs],
                "label_entropy_mean": round(float(np.mean(entropies)), 3),
                "mean_confidence": round(float(np.mean(confs)), 3),
            }
        print()

    # ── Summary ──
    chance = 100 / NUM_CLASSES
    print("="*72)
    print(" PSEUDO-LABELED DOWNSTREAM RESULTS (mean +/- std, n=3 seeds)")
    print("="*72)
    print(f"  Pseudo-labeler ceiling: {np.mean(labeler_accs):.2f}%")
    print(f"  Chance level          : {chance:.2f}%\n")
    print(f"  {'Configuration':<24} {'Accuracy':>14} {'Entropy':>9} {'Conf':>7}")
    print(f"  {'-'*58}")
    for name, r in results.items():
        print(f"  {name:<24} {r['acc_mean']:>7.2f}% +/-{r['acc_std']:<5.2f} "
              f"{r['label_entropy_mean']:>8.3f} {r['mean_confidence']:>7.3f}")
    print("="*72)

    # ── Interpretation ──
    print("\n INTERPRETATION")
    print("-"*72)
    for name, r in results.items():
        margin = r["acc_mean"] - chance
        if margin > 3 * max(r["acc_std"], 1.0):
            verdict = ("CLEARLY above chance — generator DOES carry recoverable "
                       "class structure")
        elif margin > 0:
            verdict = ("marginally above chance — within noise, treat as "
                       "inconclusive")
        else:
            verdict = ("at or below chance — no recoverable class structure")
        print(f"  {name}: {r['acc_mean']:.2f}% vs {chance:.2f}% chance")
        print(f"    -> {verdict}")
        if r["label_entropy_mean"] < 0.7:
            print(f"    -> LOW label entropy ({r['label_entropy_mean']:.3f}): "
                  f"pseudo-labels concentrate on few classes, indicating "
                  f"mode collapse")
    print("-"*72)
    print("\n Caveat for the paper: pseudo-labeling measures class structure")
    print(" recoverable by a real-data classifier, bounded by that labeler's")
    print(" own accuracy. It removes the round-robin artifact; it does not")
    print(" make these numbers directly comparable to the conditional GAN,")
    print(" whose labels are exact by construction.")

    log_path = os.path.join(LOG_DIR, "pseudo_label_results.json")
    with open(log_path, "w") as f:
        json.dump({
            "pseudo_labeler_accuracy_mean": round(float(np.mean(labeler_accs)), 2),
            "chance_level": round(chance, 2),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved → {log_path}")
