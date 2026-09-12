"""
STEP 2 — Prepare Training Patches

Cuts the calibrated cube into small spatial patches.
Produces TWO patch files:
  • data/patches_noisy.npy  — raw (original) patches
  • data/patches_clean.npy  — spectrally-smoothed versions (training targets)

The smoothing acts as a pseudo "ground-truth": the network learns to
reproduce the smooth spectral shape while suppressing band-to-band noise,
yet absorption features that span several bands are preserved because the
smoothing window (5 bands) is narrower than real spectral features.

"""

import numpy as np
import os


CAL_PATH          = r"C:\Projects\IIRS_Internship_2026\IIRS_VenusProject\data\VI0044_01.CAL"
RECORD_BYTES      = 512
Q_OBJECT_RECORD   = 14
ALL_BANDS         = 432
LINES             = 289
SAMPLES           = 256
DTYPE             = np.dtype(">f4")   # big-endian float32
INVALID_THRESHOLD = -900.0


BAND_START = 100
BAND_END   = 150


LINE_START,   LINE_END   = 20, 268
SAMPLE_START, SAMPLE_END = 20, 236

PATCH_SIZE    = 8
SMOOTH_WINDOW = 5


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


def extract_patches(subcube, patch_size):
    B, H, W = subcube.shape
    patches = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patches.append(subcube[:, y:y + patch_size, x:x + patch_size].copy())
    return np.stack(patches, axis=0)


def smooth_patches_spectrally(patches, window):
    patches_smooth = patches.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    N, B, H, W = patches_smooth.shape
    for i in range(N):
        for y in range(H):
            for x in range(W):
                spec = patches_smooth[i, :, y, x]
                if not np.any(np.isnan(spec)):
                    patches_smooth[i, :, y, x] = np.convolve(spec, kernel, mode="same")
    return patches_smooth


def filter_bad_patches(patches_noisy, patches_clean, nan_limit=0.50):
    nan_frac = np.isnan(patches_noisy).mean(axis=(1, 2, 3))
    keep = nan_frac <= nan_limit
    print(f"  Kept {keep.sum()} / {len(keep)} patches "
          f"(removed {(~keep).sum()} with >{nan_limit*100:.0f}% NaN)")
    return patches_noisy[keep], patches_clean[keep]


if __name__ == "__main__":
    print("Loading CAL cube …")
    cube = load_virtis_cal(CAL_PATH)
    print(f"  Cube shape: {cube.shape}")

    subcube = cube[BAND_START:BAND_END, LINE_START:LINE_END, SAMPLE_START:SAMPLE_END]
    print(f"  Subcube shape: {subcube.shape}")

    print("Extracting patches …")
    patches_noisy = extract_patches(subcube, PATCH_SIZE)
    print(f"  Raw patches: {patches_noisy.shape}")

    print(f"Smoothing (window={SMOOTH_WINDOW}) …")
    patches_clean = smooth_patches_spectrally(patches_noisy, SMOOTH_WINDOW)

    print("Filtering bad patches …")
    patches_noisy, patches_clean = filter_bad_patches(patches_noisy, patches_clean)

    patches_noisy = np.nan_to_num(patches_noisy, nan=0.0, posinf=0.0, neginf=0.0)
    patches_clean = np.nan_to_num(patches_clean, nan=0.0, posinf=0.0, neginf=0.0)

    os.makedirs("data", exist_ok=True)
    np.save("data/patches_noisy_VI0044.npy", patches_noisy)
    np.save("data/patches_clean_VI0044.npy", patches_clean)
    np.save("data/patch_meta_VI0044.npy", np.array([
        BAND_START, BAND_END, PATCH_SIZE, SMOOTH_WINDOW,
        LINE_START, LINE_END, SAMPLE_START, SAMPLE_END
    ]))

    print("\nSaved:")
    print(f"  patches_noisy: {patches_noisy.shape}  [{patches_noisy.min():.3f}, {patches_noisy.max():.3f}]")
    print(f"  patches_clean: {patches_clean.shape}  [{patches_clean.min():.3f}, {patches_clean.max():.3f}]")
