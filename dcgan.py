"""
Step 2: DCGAN Architecture
FedGAN Failure Analysis - Anish Bharadwaj

Generator  : latent vector (100,) → 3x28x28 RGB image
Discriminator: 3x28x28 RGB image → real/fake scalar

Design choices:
- No BatchNorm in Discriminator (more stable in federated setting)
  Using InstanceNorm instead — each client normalises independently
- LeakyReLU in Discriminator, ReLU in Generator (standard DCGAN)
- Tanh output on Generator (matches [-1,1] normalisation in data_setup.py)
- Weights initialised per DCGAN paper (mean=0, std=0.02)
"""

import torch
import torch.nn as nn


# Config
LATENT_DIM  = 100
IMG_CHANNELS = 3
FEATURE_G   = 64   # base feature map size for generator
FEATURE_D   = 64   # base feature map size for discriminator


# Weight Initialisation
def weights_init(m):
    """Apply DCGAN-style weight init to Conv and BatchNorm layers."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname or "InstanceNorm" in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# Generator
class Generator(nn.Module):
    """
    Input : (batch, LATENT_DIM, 1, 1)
    Output: (batch, 3, 28, 28)

    Architecture: series of ConvTranspose2d upsampling blocks
    4x4 → 7x7 → 14x14 → 28x28
    """
    def __init__(self, latent_dim=LATENT_DIM, feature_g=FEATURE_G,
                 img_channels=IMG_CHANNELS):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1: latent → 4×4
            # (latent_dim, 1, 1) → (feature_g*4, 4, 4)
            nn.ConvTranspose2d(latent_dim, feature_g * 4,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            # Block 2: 4×4 → 7×7
            # stride=1 + padding=0 with kernel=4 gives exact 7
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            # Block 3: 7×7 → 14×14
            nn.ConvTranspose2d(feature_g * 2, feature_g,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            # Block 4: 14×14 → 28×28 → final RGB output
            nn.ConvTranspose2d(feature_g, img_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z):
        return self.net(z)


# Discriminator
class Discriminator(nn.Module):
    """
    Input : (batch, 3, 28, 28)
    Output: (batch, 1) — raw logit (use BCEWithLogitsLoss)

    Uses InstanceNorm instead of BatchNorm:
    - BatchNorm shares stats across the batch, which causes instability
      when each federated client has a skewed class distribution.
    - InstanceNorm normalises per sample — safer for non-IID clients.
    """
    def __init__(self, img_channels=IMG_CHANNELS, feature_d=FEATURE_D):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1: 28×28 → 14×14
            nn.Conv2d(img_channels, feature_d,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 14×14 → 7×7
            nn.Conv2d(feature_d, feature_d * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_d * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: 7×7 → 4×4
            nn.Conv2d(feature_d * 2, feature_d * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_d * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: 3×3 → 1×1 (scalar output)
            nn.Conv2d(feature_d * 4, 1,
                      kernel_size=3, stride=1, padding=0, bias=False),
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.net(x).view(-1, 1)  # flatten to (batch, 1)


# Smoke Test 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    G = Generator().to(device)
    D = Discriminator().to(device)

    # Test forward pass
    z    = torch.randn(16, LATENT_DIM, 1, 1).to(device)
    fake = G(z)
    out  = D(fake)

    print(f"Generator")
    print(f"  Input  : {z.shape}")
    print(f"  Output : {fake.shape}   ← should be (16, 3, 28, 28)")
    print(f"  Min/Max: {fake.min():.3f} / {fake.max():.3f}   ← should be in [-1, 1]\n")

    print(f"Discriminator")
    print(f"  Input  : {fake.shape}")
    print(f"  Output : {out.shape}   ← should be (16, 1)")

    # Parameter count
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\nParameter count")
    print(f"  Generator    : {g_params:,}")
    print(f"  Discriminator: {d_params:,}")
    print(f"  Total        : {g_params + d_params:,}")

    print("\nStep 2 complete. DCGAN architecture verified.")
