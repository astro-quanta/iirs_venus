"""
STEP 3 — Training the Convolutional Denoising Autoencoder

Input  : data/patches_noisy.npy  (N, 50, 8, 8)
         data/patches_clean.npy  (N, 50, 8, 8)
Output : data/autoencoder.pth    (model weights + normalisation stats)

Loss = MSE(reconstruction, clean) + 0.5 × MSE(Δrecon, Δclean)
The derivative term encourages the network to preserve spectral
gradients, which helps retain absorption-line shapes.

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os


EPOCHS      = 40
BATCH_SIZE  = 16
LR          = 2e-4
DERIV_WEIGHT = 0.7
SAVE_PATH   = "data/autoencoder_VI0044.pth"



class SpectralDenoiser(nn.Module):
    """
    Convolutional autoencoder that operates on (bands, H, W) patches.

    Channels act as spectral bands; the 3×3 spatial convolutions share
    information across neighbouring pixels, which helps the decoder
    distinguish real spectral structure from pixel-level noise.

    Architecture:
    Encoder: bands → 64 → 32  (ReLU activations)
    Bottleneck: 32 feature maps at full spatial resolution
    Decoder: 32 → 64 → bands
    """
    def __init__(self, bands: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bands, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def spectral_derivative(t: torch.Tensor) -> torch.Tensor:
    """First-order difference along the band (channel) axis."""
    return t[:, 1:, :, :] - t[:, :-1, :, :]


def build_loaders(noisy_norm, clean_norm, batch_size, val_fraction=0.15):
    """Splitting into train / validation and returning DataLoaders."""
    N = noisy_norm.shape[0]
    n_val = max(1, int(N * val_fraction))

    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_train = torch.tensor(noisy_norm[train_idx], dtype=torch.float32)
    y_train = torch.tensor(clean_norm[train_idx],  dtype=torch.float32)
    x_val   = torch.tensor(noisy_norm[val_idx],   dtype=torch.float32)
    y_val   = torch.tensor(clean_norm[val_idx],    dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                              batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(x_val,   y_val),
                              batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


if __name__ == "__main__":
    noisy = np.load("data/patches_noisy_VI0044.npy").astype(np.float32)
    clean = np.load("data/patches_clean_VI0044.npy").astype(np.float32)
    print(f"Loaded  noisy: {noisy.shape}   clean: {clean.shape}")

    # Sanity check: values should be physical
    assert clean.max() < 1e6, (
        "Max radiance > 1e6 — endianness is still wrong in step2. "
        "Ensure DTYPE = np.dtype('>f4')."
    )


    vmin, vmax = float(np.percentile(clean, 1)), float(np.percentile(clean, 99))
    noisy = np.clip(noisy, vmin, vmax)
    clean = np.clip(clean, vmin, vmax)

    mean = float(clean.mean())
    std  = float(clean.std()) + 1e-6

    noisy_norm = (noisy - mean) / std
    clean_norm = (clean - mean) / std

    print(f"Normalisation  mean={mean:.4f}  std={std:.4f}")
    print(f"Normalised range  [{noisy_norm.min():.2f}, {noisy_norm.max():.2f}]")


    train_loader, val_loader = build_loaders(noisy_norm, clean_norm, BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    bands  = clean.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model     = SpectralDenoiser(bands).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5
    )

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss  = criterion(recon, yb) + DERIV_WEIGHT * criterion(
                spectral_derivative(recon), spectral_derivative(yb)
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon  = model(xb)
                loss   = criterion(recon, yb) + DERIV_WEIGHT * criterion(
                    spectral_derivative(recon), spectral_derivative(yb)
                )
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        marker = "  ← best" if val_loss < best_val_loss else ""
        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("data", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "bands":       bands,
                "mean":        mean,
                "std":         std,
                "vmin":        vmin,
                "vmax":        vmax,
            }, SAVE_PATH)

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Model saved → {SAVE_PATH}")
