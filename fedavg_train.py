"""
Step 3: FedAvg Training Loop
FedGAN Failure Analysis - Anish Bharadwaj

What this does:
- Each round: selected clients train G and D locally for N steps
- Server aggregates generator weights via FedAvg (weighted by dataset size)
- Discriminator stays LOCAL — each client keeps its own D
  (this is standard in FedGAN literature and more stable)
- Saves generator checkpoints and loss curves per round
"""

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from medmnist import PathMNIST
from dcgan import Generator, Discriminator, LATENT_DIM

# Config 
class Config:
    # Federated
    NUM_CLIENTS         = 5
    CLIENTS_PER_ROUND   = 5        # all clients participate each round
    NUM_ROUNDS          = 30       # federated rounds
    LOCAL_STEPS         = 100      # GAN update steps per client per round

    # GAN training
    BATCH_SIZE          = 64
    LR_G                = 0.0002
    LR_D                = 0.0002
    BETA1               = 0.5      # Adam momentum — standard for GANs
    LATENT_DIM          = LATENT_DIM

    # Paths
    DATA_DIR            = "./data"
    CKPT_DIR            = "./checkpoints"
    SAMPLE_DIR          = "./samples"
    LOG_DIR             = "./logs"

    SEED                = 42

cfg = Config()

for d in [cfg.CKPT_DIR, cfg.SAMPLE_DIR, cfg.LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# seeds removed for non-IID experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Data Loading
def load_client_loaders(split_file):
    """Load DataLoaders for each client from a saved split JSON."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = PathMNIST(split="train", transform=transform,
                        download=False, root=cfg.DATA_DIR)

    with open(split_file) as f:
        split = json.load(f)

    loaders = {}
    sizes   = {}
    for client_id, indices in split.items():
        cid = int(client_id)
        subset = Subset(dataset, indices)
        loaders[cid] = DataLoader(subset, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, drop_last=True,
                                  num_workers=0)
        sizes[cid] = len(indices)

    return loaders, sizes


# FedAvg Aggregation 
def fedavg(global_G, client_generators, client_sizes, selected_clients):
    """
    Weighted average of generator weights.
    Weight = client dataset size / total selected dataset size.
    Only the Generator is aggregated — Discriminators stay local.
    """
    total = sum(client_sizes[c] for c in selected_clients)
    new_state = copy.deepcopy(global_G.state_dict())

    for key in new_state:
        new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)

    for cid in selected_clients:
        weight = client_sizes[cid] / total
        client_state = client_generators[cid].state_dict()
        for key in new_state:
            new_state[key] += weight * client_state[key].float()

    global_G.load_state_dict(new_state)
    return global_G


# Local GAN Training
def train_local(client_id, generator, discriminator, loader,
                opt_G, opt_D, steps, round_num):
    """
    Train G and D locally for `steps` update steps.
    Returns average G loss and D loss over those steps.
    """
    criterion = nn.BCEWithLogitsLoss()
    generator.train()
    discriminator.train()

    g_losses, d_losses = [], []
    data_iter = iter(loader)
    step = 0

    while step < steps:
        # Get real batch
        try:
            real_imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real_imgs, _ = next(data_iter)

        real_imgs = real_imgs.to(device)
        # Noisy client injection — Client 0 gets Gaussian noise added to its images
        if client_id == 0:
            noise = torch.randn_like(real_imgs) * 0.5
            real_imgs = torch.clamp(real_imgs + noise, -1, 1)
        bs = real_imgs.size(0)

        real_labels = torch.ones(bs, 1).to(device)
        fake_labels = torch.zeros(bs, 1).to(device)

        # Train Discriminator
        opt_D.zero_grad()
        d_real = discriminator(real_imgs)
        loss_d_real = criterion(d_real, real_labels)

        z = torch.randn(bs, cfg.LATENT_DIM, 1, 1).to(device)
        fake_imgs = generator(z).detach()
        d_fake = discriminator(fake_imgs)
        loss_d_fake = criterion(d_fake, fake_labels)

        loss_D = (loss_d_real + loss_d_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # Train Generator
        opt_G.zero_grad()
        z = torch.randn(bs, cfg.LATENT_DIM, 1, 1).to(device)
        fake_imgs = generator(z)
        d_out = discriminator(fake_imgs)
        loss_G = criterion(d_out, real_labels)   # G wants D to say "real"
        loss_G.backward()
        opt_G.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())
        step += 1

    return np.mean(g_losses), np.mean(d_losses)


# Sample Images
def save_samples(generator, round_num, tag=""):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(25, cfg.LATENT_DIM, 1, 1).to(device)
        samples = generator(z)
        samples = (samples + 1) / 2   # [-1,1] → [0,1] for saving
    path = os.path.join(cfg.SAMPLE_DIR, f"round_{round_num:03d}{tag}.png")
    save_image(samples, path, nrow=5)


# Main Federated Training Loop
def federated_train(split_file, experiment_tag="iid"):
    print(f"\n{'='*55}")
    print(f" Federated Training — {experiment_tag.upper()}")
    print(f"{'='*55}")

    loaders, client_sizes = load_client_loaders(split_file)

    # Initialise global generator
    global_G = Generator().to(device)

    # Initialise per-client discriminators and optimisers
    # Each client gets its own D — never aggregated
    client_Ds   = {cid: Discriminator().to(device)
                   for cid in range(cfg.NUM_CLIENTS)}
    client_Gs   = {cid: copy.deepcopy(global_G)
                   for cid in range(cfg.NUM_CLIENTS)}

    opt_Gs = {cid: optim.Adam(client_Gs[cid].parameters(),
                               lr=cfg.LR_G, betas=(cfg.BETA1, 0.999))
              for cid in range(cfg.NUM_CLIENTS)}
    opt_Ds = {cid: optim.Adam(client_Ds[cid].parameters(),
                               lr=cfg.LR_D, betas=(cfg.BETA1, 0.999))
              for cid in range(cfg.NUM_CLIENTS)}

    # Logging
    log = {"round": [], "g_loss": [], "d_loss": []}

    for rnd in range(1, cfg.NUM_ROUNDS + 1):
        selected = list(range(cfg.NUM_CLIENTS))   # all clients this round

        # Distribute global G weights to selected clients
        for cid in selected:
            client_Gs[cid].load_state_dict(
                copy.deepcopy(global_G.state_dict())
            )

        # Local training
        round_g_losses, round_d_losses = [], []

        for cid in selected:
            g_loss, d_loss = train_local(
                client_id    = cid,
                generator    = client_Gs[cid],
                discriminator= client_Ds[cid],
                loader       = loaders[cid],
                opt_G        = opt_Gs[cid],
                opt_D        = opt_Ds[cid],
                steps        = cfg.LOCAL_STEPS,
                round_num    = rnd
            )
            round_g_losses.append(g_loss)
            round_d_losses.append(d_loss)

        # FedAvg: aggregate generator weights 
        global_G = fedavg(global_G, client_Gs, client_sizes, selected)

        # Logging
        avg_g = np.mean(round_g_losses)
        avg_d = np.mean(round_d_losses)
        log["round"].append(rnd)
        log["g_loss"].append(avg_g)
        log["d_loss"].append(avg_d)

        print(f"Round {rnd:3d}/{cfg.NUM_ROUNDS} | "
              f"G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

        # Save samples every 5 rounds
        if rnd % 5 == 0:
            save_samples(global_G, rnd, tag=f"_{experiment_tag}")

    # Save final checkpoint and logs
    ckpt_path = os.path.join(cfg.CKPT_DIR, f"generator_{experiment_tag}_final.pt")
    torch.save(global_G.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")

    log_path = os.path.join(cfg.LOG_DIR, f"losses_{experiment_tag}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Losses saved    → {log_path}")

    return global_G, log


# Entry Point
if __name__ == "__main__":
    # GPU check
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("WARNING: No GPU detected. Training will be slow.\n")

    # Run IID baseline — this is Experiment 1
    global_G_iid, log_iid = federated_train(
        split_file     = "./data/non_iid_split.json",
        experiment_tag = "noisy_client"
    )

    print("\nStep 3 complete. IID baseline training done.")
    print("Run non-IID next: change split_file to non_iid_split.json")
