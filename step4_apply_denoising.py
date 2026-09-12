"""
STEP 4 — Apply the Trained Denoiser to the Full Subcube

Loading the trained model and running it patch-by-patch across the same
spatial region used in training.

Output:
    data/subcube_original.npy   (50, H, W)  — raw radiance
    data/subcube_denoised.npy   (50, H, W)  — denoised radiance
"""

import numpy as np
import torch
import torch.nn as nn
import os

CAL_PATH          = r"C:\Projects\IIRS_Internship_2026\IIRS_VenusProject\data\VI0044_01.CAL"
MODEL_PATH        = "data/autoencoder_VI0044.pth"
RECORD_BYTES      = 512
Q_OBJECT_RECORD   = 14
ALL_BANDS         = 432
LINES             = 240
SAMPLES           = 256
DTYPE             = np.dtype(">f4")
INVALID_THRESHOLD = -900.0


class SpectralDenoiser(nn.Module):
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


def load_virtis_cal(cal_path):
    header_bytes = (Q_OBJECT_RECORD - 1) * RECORD_BYTES
    n_values     = ALL_BANDS * LINES * SAMPLES
    with open(cal_path, "rb") as f:
        f.seek(header_bytes)
        arr = np.fromfile(f, dtype=DTYPE, count=n_values)
    cube = arr.reshape((ALL_BANDS, LINES, SAMPLES)).astype(np.float32)
    cube[cube < INVALID_THRESHOLD] = np.nan
    cube *= 1000.0
    return cube


def denoise_subcube(subcube, model, mean, std, vmin, vmax, patch_size, device):
    B, H, W   = subcube.shape
    denoised  = np.zeros_like(subcube)

    model.eval()
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, patch_size):
            for x in range(0, W - patch_size + 1, patch_size):
                patch = subcube[:, y:y + patch_size, x:x + patch_size].copy()

                patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

                patch = np.clip(patch, vmin, vmax)
                patch_norm = (patch - mean) / (std + 1e-6)

                t = torch.tensor(patch_norm[None, ...],
                                 dtype=torch.float32).to(device)
                recon = model(t).cpu().numpy()[0]           # (B, patch, patch)

                denoised[:, y:y + patch_size, x:x + patch_size] = (
                    recon * std + mean
                )

    return denoised


if __name__ == "__main__":
    ckpt   = torch.load(MODEL_PATH, map_location="cpu")
    bands  = ckpt["bands"]
    mean   = ckpt["mean"]
    std    = ckpt["std"]
    vmin   = ckpt["vmin"]
    vmax   = ckpt["vmax"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SpectralDenoiser(bands).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Model loaded from {MODEL_PATH}  (bands={bands}, device={device})")

    meta         = np.load("data/patch_meta_VI0044.npy")
    band_start   = int(meta[0])
    band_end     = int(meta[1])
    patch_size   = int(meta[2])
    line_start   = int(meta[4])
    line_end     = int(meta[5])
    sample_start = int(meta[6])
    sample_end   = int(meta[7])

    print("Loading CAL cube …")
    cube    = load_virtis_cal(CAL_PATH)
    subcube = cube[
        band_start:band_end,
        line_start:line_end,
        sample_start:sample_end
    ].copy()
    print(f"  Subcube shape: {subcube.shape}")

    print("Running denoiser …")
    denoised = denoise_subcube(subcube, model, mean, std, vmin, vmax,
                               patch_size, device)
    print("  Done.")

    os.makedirs("data", exist_ok=True)
    np.save("data/subcube_original_VI0044.npy", subcube)
    np.save("data/subcube_denoised_VI0044.npy", denoised)
    np.save("data/band_axis_VI0044.npy",
            np.arange(band_start, band_end, dtype=np.int32))

    print("\nSaved:")
    print(f"  data/subcube_original.npy  {subcube.shape}")
    print(f"  data/subcube_denoised.npy  {denoised.shape}")
    print(f"  data/band_axis.npy")
    print("Step 4 done. Run step5_visualise.py next.")
