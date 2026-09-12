"""
STEP 5 — Visualising Results: Spectra, Absorption Lines, Spatial Images

  OUtput:
  Single pixel: original vs denoised spectrum with
             absorption-line markers automatically detected.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
import os


PIXEL_LINE   = 60    # adjust to a point of interest
PIXEL_SAMPLE = 60


CENTRE_LINE   = 60
CENTRE_SAMPLE = 60

# Known VIRTIS-M IR absorption band centres (approximate band indices).
# These are Venus atmospheric / surface features.
KNOWN_ABSORPTIONS = {
    "CO₂ 1.05 µm":  108,
    "CO₂ 1.18 µm":  121,
    "H₂O / CO₂":    135,
    "CO₂ 1.74 µm":  143,
}

PLOT_DIR = "data/plots"
# ──────────────────────────────────────────────────────────────────────────────


def load_data():
    original = np.load("data/subcube_original_VI0044.npy")   # adjust suffix if needed
    denoised = np.load("data/subcube_denoised_VI0044.npy")
    bands    = np.load("data/band_axis_VI0044.npy")
    denoised = np.clip(denoised, -0.5, None)   # only remove extreme outliers, not near-zero dips
    return original, denoised, bands


def detect_absorption_lines(spectrum, bands, prominence=0.5, width=1):
    """
    Detecting local minima (absorption dips) in the denoised spectrum.

    Returns list of (band_index, radiance) tuples.
    """
    inv = -spectrum
    if np.all(np.isnan(inv)):
        return []
    inv = np.nan_to_num(inv, nan=np.nanmin(inv))

    prom_abs = prominence * (np.nanmax(spectrum) - np.nanmin(spectrum))
    prom_abs = max(prom_abs, 1e-6)

    peaks, props = find_peaks(inv, prominence=prom_abs, width=width)
    return [(bands[p], spectrum[p]) for p in peaks]


def fig1_single_spectrum(original, denoised, bands):
    orig_spec = original[:, PIXEL_LINE, PIXEL_SAMPLE]
    den_spec  = denoised[:, PIXEL_LINE, PIXEL_SAMPLE]

    abs_lines = detect_absorption_lines(den_spec, bands, prominence=0.03, width=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(bands, orig_spec, color="#9ecae1", linewidth=1.0,
            alpha=0.85, label="Original (noisy)")
    ax.plot(bands, den_spec, color="#084594", linewidth=2.0,
            label="Denoised")

    for b_idx, rad in abs_lines:
        ax.axvline(b_idx, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.annotate(f"b{b_idx}", xy=(b_idx, rad),
                    xytext=(b_idx + 0.3, rad - 0.05 * (np.nanmax(den_spec) - np.nanmin(den_spec))),
                    fontsize=7, color="red", rotation=90, va="top")

    for label, bidx in KNOWN_ABSORPTIONS.items():
        if bands[0] <= bidx <= bands[-1]:
            ax.axvline(bidx, color="green", linewidth=1.0,
                       linestyle=":", alpha=0.8)
            ax.text(bidx + 0.2, ax.get_ylim()[0],
                    label, fontsize=7, color="green",
                    rotation=90, va="bottom")

    ax.set_xlabel("Band index", fontsize=12)
    ax.set_ylabel("Radiance [W/m²/sr/µm]", fontsize=12)
    ax.set_title(
        f"VIRTIS spectrum — pixel (line={PIXEL_LINE}, sample={PIXEL_SAMPLE})\n"
        "Red dashes = auto-detected dips  |  Green dots = known absorptions",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "fig1_single_spectrum.png")
    fig.savefig(path, dpi=180)
    plt.show()
    print(f"  Saved → {path}")
    if abs_lines:
        print(f"  Auto-detected {len(abs_lines)} absorption dip(s):")
        for b, r in abs_lines:
            print(f"    band {b:3d}  radiance={r:.4f}")
    else:
        print("  No significant absorption dips auto-detected at current threshold.")



if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Loading saved subcubes …")
    original, denoised, bands = load_data()
    print(f"  original: {original.shape}  range [{np.nanmin(original):.3f}, {np.nanmax(original):.3f}]")
    print(f"  denoised: {denoised.shape}  range [{np.nanmin(denoised):.3f}, {np.nanmax(denoised):.3f}]")
    print(f"  bands:    {bands[0]} … {bands[-1]}")

    print("\nFigure 1 — single spectrum …")
    fig1_single_spectrum(original, denoised, bands)

