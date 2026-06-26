"""
Conditional GAN (cGAN) Architecture
FedGAN Failure Analysis - Anish Bharadwaj

Difference from DCGAN:
- Generator takes (noise vector + class label) as input
- Discriminator takes (image + class label) as input
- Labels are embedded and concatenated to the input at each network

This allows the generator to produce class-specific images.
During the silent failure test, generated images will have
correct label-image correspondence — unlike the unconditional GAN.

This lets us isolate: is the failure from the GAN itself (unconditional),
or from the federated training? The cGAN + FL comparison answers that.
"""

import torch
import torch.nn as nn

# Config
LATENT_DIM   = 100
NUM_CLASSES  = 9
IMG_CHANNELS = 3
EMBED_DIM    = 50    # class embedding dimension
FEATURE_G    = 64
FEATURE_D    = 64


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname or "InstanceNorm" in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# Conditional Generator
class ConditionalGenerator(nn.Module):
    """
    Input : noise (batch, LATENT_DIM) + label (batch,)
    Output: (batch, 3, 28, 28)

    Class label is embedded to EMBED_DIM and concatenated
    with the noise vector before the first conv layer.
    """
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES,
                 embed_dim=EMBED_DIM, feature_g=FEATURE_G,
                 img_channels=IMG_CHANNELS):
        super().__init__()

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        input_dim = latent_dim + embed_dim

        self.net = nn.Sequential(
            # Block 1: (input_dim, 1, 1) → (feature_g*4, 4, 4)
            nn.ConvTranspose2d(input_dim, feature_g * 4,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            # Block 2: 4×4 → 7×7
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            # Block 3: 7×7 → 14×14
            nn.ConvTranspose2d(feature_g * 2, feature_g,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            # Block 4: 14×14 → 28×28
            nn.ConvTranspose2d(feature_g, img_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z, labels):
        # Embed label and concatenate with noise
        label_emb = self.label_embed(labels)           # (batch, embed_dim)
        x = torch.cat([z, label_emb], dim=1)           # (batch, latent+embed)
        x = x.unsqueeze(2).unsqueeze(3)                # (batch, latent+embed, 1, 1)
        return self.net(x)


# Conditional Discriminator 
class ConditionalDiscriminator(nn.Module):
    """
    Input : image (batch, 3, 28, 28) + label (batch,)
    Output: (batch, 1) scalar logit

    Class label is embedded, projected to a spatial map,
    and concatenated as an extra channel to the input image.
    This gives the discriminator class-awareness:
    it can penalise a generator that produces the wrong tissue type.
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM,
                 img_channels=IMG_CHANNELS, feature_d=FEATURE_D):
        super().__init__()

        # Project label embedding to a single spatial channel (28×28)
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.label_proj  = nn.Linear(embed_dim, 28 * 28)

        # Input is image (3 channels) + label map (1 channel) = 4 channels
        in_channels = img_channels + 1

        self.net = nn.Sequential(
            # Block 1: 28×28 → 14×14
            nn.Conv2d(in_channels, feature_d,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 14×14 → 7×7
            nn.Conv2d(feature_d, feature_d * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_d * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: 7×7 → 3×3
            nn.Conv2d(feature_d * 2, feature_d * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_d * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: 3×3 → 1×1
            nn.Conv2d(feature_d * 4, 1,
                      kernel_size=3, stride=1, padding=0, bias=False),
        )

        self.apply(weights_init)

    def forward(self, x, labels):
        # Embed label → spatial map → concatenate as extra channel
        label_emb  = self.label_embed(labels)               # (batch, embed_dim)
        label_map  = self.label_proj(label_emb)             # (batch, 28*28)
        label_map  = label_map.view(x.size(0), 1, 28, 28)  # (batch, 1, 28, 28)
        x = torch.cat([x, label_map], dim=1)                # (batch, 4, 28, 28)
        return self.net(x).view(-1, 1)


# Smoke Test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    G = ConditionalGenerator().to(device)
    D = ConditionalDiscriminator().to(device)

    batch  = 16
    z      = torch.randn(batch, LATENT_DIM).to(device)
    labels = torch.randint(0, NUM_CLASSES, (batch,)).to(device)

    fake = G(z, labels)
    out  = D(fake, labels)

    print(f"Generator")
    print(f"  Noise input : {z.shape}")
    print(f"  Labels      : {labels[:4].tolist()} ...")
    print(f"  Output      : {fake.shape}   ← should be (16, 3, 28, 28)")
    print(f"  Min/Max     : {fake.min():.3f} / {fake.max():.3f}\n")

    print(f"Discriminator")
    print(f"  Image input : {fake.shape}")
    print(f"  Output      : {out.shape}   ← should be (16, 1)")

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\nParameter count")
    print(f"  Generator    : {g_params:,}")
    print(f"  Discriminator: {d_params:,}")
    print(f"  Total        : {g_params + d_params:,}")

    print("\ncGAN architecture verified.")
