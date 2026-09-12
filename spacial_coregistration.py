"""
Extract lat/lon grid from VIRTIS GEO file and export
denoised band image as a GeoTIFF for loading into QGIS.
"""
import numpy as np
from osgeo import gdal, osr

# ── Load GEO cube ─────────────────────────────────────────
GEO_PATH      = r"C:\Users\shriy\PycharmProjects\IIRS_VenusProject\data\VI0044_01.GEO"
RECORD_BYTES  = 512
GEO_Q_RECORD  = 10
GEO_BANDS     = 33
LINES         = 289
SAMPLES       = 256
GEO_DTYPE     = np.dtype(">i4")

geo_header = (GEO_Q_RECORD - 1) * RECORD_BYTES
with open(GEO_PATH, "rb") as f:
    f.seek(geo_header)
    arr = np.fromfile(f, dtype=GEO_DTYPE,
                      count=GEO_BANDS * LINES * SAMPLES)
geo = arr.reshape((GEO_BANDS, LINES, SAMPLES)).astype(np.float32)

# ── Read label to find which GEO bands are lat/lon ────────
# Standard VIRTIS GEO delivery band order:
# Inspect your label with diag3_read_label.py for VI0044.GEO
# Typical order: 0=lon, 1=lat (in degrees * 1000 as integer)
# Try both and check if values are in [-180,180] and [-90,90]
lon_raw = geo[0]   # degrees * 1000
lat_raw = geo[1]   # degrees * 1000

# Convert from millidegrees to degrees
lon = lon_raw / 1000.0
lat = lat_raw / 1000.0

print("Longitude range:", np.nanmin(lon), "to", np.nanmax(lon))
print("Latitude  range:", np.nanmin(lat), "to", np.nanmax(lat))
# Expected for Ovda Regio: lon ~90-120°E, lat ~-10 to +10°

# ── Load denoised subcube ─────────────────────────────────
denoised = np.load("data/subcube_denoised_VI0044.npy")   # (50, H, W)
meta      = np.load("data/patch_meta_VI0044.npy")
line_start   = int(meta[4])
sample_start = int(meta[6])

# ── Export one band as GeoTIFF ────────────────────────────
# Use band index 35 (local) = global band 135 = CO 2.30 µm window
# This is the most scientifically interesting band for Ovda
band_local = 35
band_img   = denoised[band_local]    # (H, W)
H, W       = band_img.shape

# Get the lat/lon corners of the subcube region
lat_sub = lat[line_start:line_start+H, sample_start:sample_start+W]
lon_sub = lon[line_start:line_start+H, sample_start:sample_start+W]

lat_min, lat_max = np.nanmin(lat_sub), np.nanmax(lat_sub)
lon_min, lon_max = np.nanmin(lon_sub), np.nanmax(lon_sub)
print(f"\nSubcube covers: lon {lon_min:.2f}–{lon_max:.2f}°E, "
      f"lat {lat_min:.2f}–{lat_max:.2f}°")

# Write GeoTIFF with geographic coordinates
driver = gdal.GetDriverByName("GTiff")
ds = driver.Create("data/virtis_band135_CO_denoised.tif",
                   W, H, 1, gdal.GDT_Float32)

# Geotransform: (top-left-lon, pixel-width, 0, top-left-lat, 0, -pixel-height)
pixel_w = (lon_max - lon_min) / W
pixel_h = (lat_max - lat_min) / H
ds.SetGeoTransform([lon_min, pixel_w, 0, lat_max, 0, -pixel_h])

# Set Venus geographic CRS (GCS_Venus_2000)
srs = osr.SpatialReference()
srs.SetGeogCS("GCS_Venus_2000", "D_Venus_2000",
              "Venus_2000_IAU_IAG",
              6051800.0, 0.0)  # Venus mean radius, sphere
ds.SetProjection(srs.ExportToWkt())

band = ds.GetRasterBand(1)
band.WriteArray(band_img)
band.SetNoDataValue(np.nan)
ds.FlushCache()
ds = None
print("Saved → data/virtis_band135_CO_denoised.tif")
print("Load this in QGIS and overlay with your Magellan layer.")