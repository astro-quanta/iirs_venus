"""
STEP 1 — Load and Validate the VIRTIS CAL Cube

Expected output:
    Cube shape: (432, 240, 256)
    Valid pixel count: <some large number>
    Nanmin / Nanmax: something like 0.5 / 280.0
    Band 125 mean radiance: ~100–200

"""

import numpy as np
import matplotlib.pyplot as plt
import os

CAL_PATH = r"C:\Projects\IIRS_Internship_2026\IIRS_VenusProject\data\VI0044_01.CAL"

RECORD_BYTES    = 512
Q_OBJECT_RECORD = 14
ALL_BANDS       = 432
LINES           = 289
SAMPLES         = 256

# VIRTIS CAL files are big-endian IEEE 754 single-precision floats
DTYPE = np.dtype(">f4")

# Physical sentinel / error codes used by VIRTIS pipeline
INVALID_THRESHOLD = -900.0


def load_virtis_cal(cal_path):
    """
    Loading a VIRTIS-M calibrated (CAL) PDS3 QUBE file.

    OUput:
    cube : np.ndarray, shape (bands, lines, samples), dtype float32, native endian
           Invalid pixels are replaced with np.nan.
    """
    header_bytes = (Q_OBJECT_RECORD - 1) * RECORD_BYTES
    n_values     = ALL_BANDS * LINES * SAMPLES

    with open(cal_path, "rb") as f:
        f.seek(header_bytes)
        arr = np.fromfile(f, dtype=DTYPE, count=n_values)

    if arr.size != n_values:
        raise RuntimeError(
            f"Expected {n_values} values but read {arr.size}. "
            "Check RECORD_BYTES / Q_OBJECT_RECORD in the PDS3 label."
        )

    # Converting to native float32 for all downstream operations
    cube = arr.reshape((ALL_BANDS, LINES, SAMPLES)).astype(np.float32)

    # Masking VIRTIS fill / saturation codes
    cube[cube < INVALID_THRESHOLD] = np.nan
    cube *= 1000.0

    return cube


def validate_cube(cube):
    """Printing diagnostics and sanity-check that values look physical."""
    print("=" * 55)
    print("Cube shape          :", cube.shape)
    valid_mask  = np.isfinite(cube)
    n_valid     = valid_mask.sum()
    n_total     = cube.size
    print(f"Valid pixels        : {n_valid:,} / {n_total:,}  "
          f"({100*n_valid/n_total:.1f} %)")
    print(f"Radiance min (valid): {np.nanmin(cube):.4f}")
    print(f"Radiance max (valid): {np.nanmax(cube):.4f}")
    print(f"Radiance mean       : {np.nanmean(cube):.4f}")

    band_125 = cube[125]
    print(f"Band 125 — mean     : {np.nanmean(band_125):.4f}  "
          f"std: {np.nanstd(band_125):.4f}")

    if np.nanmax(cube) > 1e10:
        print("\n⚠  WARNING: max radiance > 1e10.  "
              "Endianness is likely still wrong — ensure DTYPE = np.dtype('>f4').")
    else:
        print("\n✓  Radiance range looks physical.")
    print("=" * 55)


def plot_sample_spectra(cube, n_pixels=5, band_start=100, band_end=150):
    """Plotting a few raw spectra to visually confirm the data look reasonable."""
    rng = np.random.default_rng(42)
    lines_idx   = rng.integers(20, LINES - 20,   size=n_pixels)
    samples_idx = rng.integers(20, SAMPLES - 20, size=n_pixels)
    band_idx    = np.arange(band_start, band_end)

    plt.figure(figsize=(9, 4))
    for li, si in zip(lines_idx, samples_idx):
        spec = cube[band_start:band_end, li, si]
        if np.all(np.isnan(spec)):
            continue
        plt.plot(band_idx, spec, alpha=0.8, linewidth=1.2,
                 label=f"L{li} S{si}")

    plt.xlabel("Band index")
    plt.ylabel("Radiance [W/m²/sr/µm]")
    plt.title("Raw VIRTIS spectra — sanity check (bands 100–149)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/step1_raw_spectra.png", dpi=150)
    plt.show()
    print("Saved → data/step1_raw_spectra.png")


if __name__ == "__main__":
    print(f"Loading: {CAL_PATH}")
    cube = load_virtis_cal(CAL_PATH)
    validate_cube(cube)
    plot_sample_spectra(cube)
