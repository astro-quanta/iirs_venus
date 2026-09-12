"""
Scatter plot: Magellan SAR backscatter vs VIRTIS denoised radiance
Requires both datasets sampled to the same grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.ndimage import zoom

# 1. File Paths
mag_path = r"C:\Projects\IIRS_Internship_2026\IIRS_VenusProject\data\magellan\777br06s102.ibg"
vir_path = "data/virtis_band135_CO_denoised.tif"

# 2. Open Datasets
mag_ds = gdal.Open(mag_path)
vir_ds = gdal.Open(vir_path)

mag_raw = mag_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
vir_arr = vir_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

# 3. Resample Magellan Array to Match VIRTIS Grid Dimensions (220, 216)
zoom_factors = (vir_arr.shape[0] / mag_raw.shape[0], vir_arr.shape[1] / mag_raw.shape[1])
mag_dn = zoom(mag_raw, zoom_factors, order=1)

# 4. Correct Magellan SAR Calibration
# If raw array is already in 0-255 float format:
if mag_dn.max() > 50:
    mag_db = 0.32 * (mag_dn - 1.0) - 20.0
else:
    # If array was pre-offset by GDAL, shift back to physical range (-25 to +5 dB)
    mag_db = mag_dn - 30.0

# 5. Mask Valid Non-Zero Points
# 5. Mask Valid Non-Background Data
valid = (
    (mag_db > -20.0) & (mag_db < 15.0) &  # Exclude background (-20.32 dB) & right truncation
    (vir_arr > 0.1) &                     # Exclude 0 or near-zero background radiance
    np.isfinite(mag_db) &
    np.isfinite(vir_arr)
)

x = mag_db[valid]
y = vir_arr[valid]

# 6. Plot Corrected Hexbin Density
plt.figure(figsize=(9, 6), dpi=150)
hb = plt.hexbin(x, y, gridsize=50, cmap='inferno', mincnt=1)
plt.colorbar(hb, label='Pixel Count')

if len(x) >= 2:
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_line, slope * x_line + intercept, 'r--', label=f'Linear fit (m={slope:.3f})')
    plt.legend(loc='upper left')

plt.xlabel(r'Magellan SAR Backscatter $\sigma^0$ [dB]')
plt.ylabel(r'VIRTIS Radiance at 2.30 $\mu$m [$W \cdot m^{-2} \cdot sr^{-1} \cdot \mu m^{-1}$]')
plt.title('Surface Roughness vs. Thermal Emission — Ovda Regio')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()