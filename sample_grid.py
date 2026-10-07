"""
Qualitative Sample Grid — Figure 5
FedGAN Failure Analysis - Anish Bharadwaj

BLOCKING FIX #5 from the adversarial review.

Generates one grid figure: top row real PathMNIST images, then
one row of generated samples per configuration (seed 42
checkpoints). Reviewers of generative-model papers expect to SEE
the outputs; this figure also makes the cGAN Non-IID collapse
visually inspectable.

For conditional generators, each column is conditioned on a
different class (0-8), so within-row diversity across columns
directly shows whether class control works. For unconditional
generators, columns are independent random draws.

Usage:
    python sample_grid.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
import warnings
warnings.filterwarnings("ignore")
from dcgan import Generator, LATENT_DIM as UNCOND_LATENT_DIM
from cgan import ConditionalGenerator, LATENT_DIM as COND_LATENT_DIM, NUM_CLASSES

DATA_DIR = "./data"
CKPT_DIR = "./checkpoints"
FIG_DIR  = "./figures"
SEED     = 42          # one representative seed for the qualitative figure
N_COLS   = 9           # one column per class for conditional rows
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(FIG_DIR, exist_ok=True)

ROWS = [
    ("Real (one per class)",  None,                                        None),
    ("GAN — IID",             f"generator_iid_seed{SEED}_final.pt",          False),
    ("GAN — Non-IID",         f"generator_non_iid_seed{SEED}_final.pt",      False),
    ("GAN — Noisy Client",    f"generator_noisy_client_seed{SEED}_final.pt", False),
    ("cGAN — IID",            f"generator_cgan_iid_seed{SEED}_final.pt",     True),
    ("cGAN — Non-IID",        f"generator_cgan_non_iid_seed{SEED}_final.pt", True),
    ("cGAN — Noisy Client",   f"generator_cgan_noisy_client_seed{SEED}_final.pt", True),
]

CLASS_NAMES = ["adipose", "backgr.", "debris", "lymph.", "mucus",
               "sm.muscle", "colon", "stroma", "adenoca."]

COLORS = {"bg": "#0F1117", "text": "#E8EAF0", "subtext": "#8B90A0",
          "warn": "#E84C4C"}

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_real_one_per_class():
    """One real example image per class, in class order."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = PathMNIST(split="test", transform=transform,
                        download=False, root=DATA_DIR)
    found = {}
    for img, label in dataset:
        c = int(label.item()) if hasattr(label, "item") else int(label)
        if c not in found:
            found[c] = img
        if len(found) == NUM_CLASSES:
            break
    return torch.stack([found[c] for c in range(NUM_CLASSES)])


@torch.no_grad()
def generate_row(ckpt_file, is_conditional):
    ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
    if not os.path.exists(ckpt_path):
        return None

    if is_conditional:
        G = ConditionalGenerator().to(DEVICE)
        G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        G.eval()
        z = torch.randn(NUM_CLASSES, COND_LATENT_DIM).to(DEVICE)
        labels = torch.arange(NUM_CLASSES).to(DEVICE)   # one per class
        imgs = G(z, labels).cpu()
    else:
        G = Generator().to(DEVICE)
        G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        G.eval()
        z = torch.randn(NUM_CLASSES, UNCOND_LATENT_DIM, 1, 1).to(DEVICE)
        imgs = G(z).cpu()

    return imgs


def to_display(img):
    """(3,28,28) in [-1,1] -> (28,28,3) in [0,1] for imshow."""
    return ((img.permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy()


if __name__ == "__main__":
    print(f"Device: {DEVICE} | Building sample grid from seed {SEED} checkpoints\n")

    n_rows = len(ROWS)
    fig, axes = plt.subplots(n_rows, N_COLS,
                             figsize=(N_COLS * 1.15, n_rows * 1.3))
    fig.patch.set_facecolor(COLORS["bg"])

    for r, (row_name, ckpt_file, is_cond) in enumerate(ROWS):
        if ckpt_file is None:
            imgs = get_real_one_per_class()
            print(f"Row {r}: {row_name} — real images loaded")
        else:
            imgs = generate_row(ckpt_file, is_cond)
            status = "OK" if imgs is not None else "MISSING CHECKPOINT"
            print(f"Row {r}: {row_name} — {status}")

        for c in range(N_COLS):
            ax = axes[r, c]
            ax.set_facecolor(COLORS["bg"])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if imgs is not None:
                ax.imshow(to_display(imgs[c]))

            # Column headers on top row only — class names apply to the
            # real row and to conditional rows (conditioned per column)
            if r == 0:
                ax.set_title(CLASS_NAMES[c], fontsize=7,
                             color=COLORS["subtext"], pad=3)

            # Row label on first column
            if c == 0:
                label_color = (COLORS["warn"] if "Non-IID" in row_name
                               and "cGAN" in row_name else COLORS["text"])
                ax.set_ylabel(row_name, fontsize=7.5, color=label_color,
                              rotation=0, ha="right", va="center",
                              labelpad=48)

    fig.suptitle(
        "Figure 5 — Qualitative Samples: real vs. generated (seed 42)\n"
        "Conditional rows: each column conditioned on that class. "
        "Unconditional rows: independent random draws.",
        fontsize=9.5, color=COLORS["text"], y=1.005)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_sample_grid.png")
    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"\nSaved → {path}")
