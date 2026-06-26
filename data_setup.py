"""
Step 1: MedMNIST Data Setup + Client Splitting
FedGAN Failure Analysis - Anish Bharadwaj

Handles:
- Loading PathMNIST (colorectal cancer tissue, 9 classes)
- IID client split (baseline)
- Non-IID client split (failure injection)
- Basic dataset stats and verification
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import medmnist
from medmnist import PathMNIST
import matplotlib.pyplot as plt
import os
import json

# Config
NUM_CLIENTS     = 5
IMG_SIZE        = 28
BATCH_SIZE      = 64
DATA_DIR        = "./data"
PLOTS_DIR       = "./plots"
NON_IID_ALPHA   = 0.1   # Dirichlet alpha — lower = more skewed. 0.1 is severe.
SEED            = 42

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Transform 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])  # normalise to [-1, 1] for GAN
])

# Load Dataset
def load_pathmnist():
    train_dataset = PathMNIST(split="train", transform=transform,
                               download=True, root=DATA_DIR)
    test_dataset  = PathMNIST(split="test",  transform=transform,
                               download=True, root=DATA_DIR)

    print(f"Train size : {len(train_dataset)}")
    print(f"Test size  : {len(test_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Num classes: {len(train_dataset.info['label'])}")
    print(f"Labels     : {list(train_dataset.info['label'].values())}\n")

    return train_dataset, test_dataset

 
# IID Split
def iid_split(dataset, num_clients):
    """Randomly and evenly distribute data across clients."""
    indices = np.random.permutation(len(dataset))
    client_indices = np.array_split(indices, num_clients)
    return {i: client_indices[i].tolist() for i in range(num_clients)}


# Non-IID Split (Dirichlet)
def non_iid_split(dataset, num_clients, alpha=NON_IID_ALPHA):
    """
    Dirichlet-based non-IID split.
    alpha → 0  : each client gets only 1 class (extreme)
    alpha → inf: approaches IID
    alpha=0.1 is severe and realistic for hospital-style skew.
    """
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}

    client_indices = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        # Convert to integer counts
        counts = (proportions * len(class_indices[c])).astype(int)
        # Fix rounding so we use all samples
        counts[-1] = len(class_indices[c]) - counts[:-1].sum()

        np.random.shuffle(class_indices[c])
        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(class_indices[c][start:start + count])
            start += count

    return client_indices


# Verify + Visualise Splits
def verify_split(dataset, client_indices, split_name, num_classes=9):
    """Print class distribution per client and save a bar chart."""
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    distributions = {}

    print(f"\n{'='*50}")
    print(f" {split_name} Split — Class Distribution per Client")
    print(f"{'='*50}")

    for client_id, indices in client_indices.items():
        client_labels = labels[indices]
        class_counts  = np.bincount(client_labels, minlength=num_classes)
        distributions[client_id] = class_counts
        dist_str = "  ".join([f"C{c}:{class_counts[c]:4d}" for c in range(num_classes)])
        print(f"Client {client_id} ({len(indices):5d} samples): {dist_str}")

    # Plot
    fig, axes = plt.subplots(1, NUM_CLIENTS, figsize=(16, 3), sharey=True)
    fig.suptitle(f"{split_name} Split — Per-Client Class Distribution", fontsize=13)
    for i, ax in enumerate(axes):
        ax.bar(range(num_classes), distributions[i], color="steelblue", edgecolor="white")
        ax.set_title(f"Client {i}", fontsize=10)
        ax.set_xlabel("Class")
        if i == 0:
            ax.set_ylabel("Samples")
        ax.set_xticks(range(num_classes))

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"{split_name.lower().replace(' ', '_')}_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved → {save_path}")

    return distributions


# DataLoader Factory 
def get_client_loaders(dataset, client_indices, batch_size=BATCH_SIZE):
    """Return a dict of DataLoaders, one per client."""
    loaders = {}
    for client_id, indices in client_indices.items():
        subset = Subset(dataset, indices)
        loaders[client_id] = DataLoader(subset, batch_size=batch_size,
                                         shuffle=True, drop_last=True)
    return loaders


# Save Split Metadata
def save_split(client_indices, filename):
    path = os.path.join(DATA_DIR, filename)
    serialisable = {str(k): v for k, v in client_indices.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f)
    print(f"Split saved → {path}")


# Main
if __name__ == "__main__":
    train_dataset, test_dataset = load_pathmnist()

    # IID split
    iid_indices = iid_split(train_dataset, NUM_CLIENTS)
    verify_split(train_dataset, iid_indices, "IID")
    save_split(iid_indices, "iid_split.json")

    # Non-IID split
    non_iid_indices = non_iid_split(train_dataset, NUM_CLIENTS, alpha=NON_IID_ALPHA)
    verify_split(train_dataset, non_iid_indices, "Non-IID")
    save_split(non_iid_indices, "non_iid_split.json")

    # Sanity check — no data leakage between clients
    iid_all = [idx for v in iid_indices.values() for idx in v]
    assert len(iid_all) == len(set(iid_all)), "Duplicate indices in IID split!"
    print("\nSanity check passed — no data leakage between clients.")

    print("\nStep 1 complete. Ready for Step 2 (DCGAN).")
