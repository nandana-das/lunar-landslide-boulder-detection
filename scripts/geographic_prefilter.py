"""
geographic_prefilter.py
========================
Geographic pre-filtering of OHRC tiles to identify high-value candidates
for landslide/boulder annotation.

Pipeline:
  1. Build a lat/lon lookup table for every pixel in all 12 products
     by interpolating from the sparse geometry CSV grid.
  2. Assign each usable tile a center lat/lon.
  3. Score tiles by:
       a. Proximity to known feature locations (crater rims, scarps)
       b. Brightness variance (std deviation — high texture = possible boulders)
       c. Composite priority score
  4. Export top-500 ranked tiles to:
       results/geofilter/priority_tiles.csv
  5. Copy top-500 tiles to:
       data/priority_sample/

IMPORTANT:
  - Original tiles are NEVER modified or deleted.
  - Copies only.
  - Uses only existing manifest + geometry CSVs + tile PNGs.

Usage:
  python scripts/geographic_prefilter.py

Runtime: ~5-10 min (geometry interpolation for 12 products + std-dev scan of tiles)
"""
import re
import sys
import shutil
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
MANIFEST_CSV   = PROJECT_ROOT / "results" / "dataset_manifest.csv"
EXTRACTED_BASE = PROJECT_ROOT / "data" / "extracted"
TILES_USABLE   = PROJECT_ROOT / "data" / "tiles" / "usable"
OUT_DIR        = PROJECT_ROOT / "results" / "geofilter"
PRIORITY_DIR   = PROJECT_ROOT / "data" / "priority_sample"
OUT_CSV        = OUT_DIR / "priority_tiles.csv"
GEO_CACHE_CSV  = OUT_DIR / "tile_geolocation.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PRIORITY_DIR.mkdir(parents=True, exist_ok=True)

TOP_N          = 500
TILE_SIZE      = 640
MIN_STD        = 0.0   # accept all std values; scoring handles weighting

# ── Known feature locations (selenographic, degrees) ──────────────────────
# Source: published lunar south-pole geology / LROC literature
# Format: (name, lat_deg, lon_deg, radius_deg)
# radius_deg: tiles within this distance get proximity credit
KNOWN_FEATURES = [
    # Major craters with steep inner walls — known landslide/boulder sources
    ("Shackleton_rim",    -89.9,   0.0,  1.5),
    ("de Gerlache_rim",   -88.5, -87.5,  1.0),
    ("Haworth",           -87.5,   0.0,  1.5),
    ("Nobile",            -85.2,  53.5,  1.5),
    ("Malapert",          -84.9,   0.0,  1.0),
    ("Slater",            -88.1,  83.0,  1.0),
    ("Shoemaker",         -88.1, 144.5,  1.5),
    ("Faustini",          -87.3,  77.0,  1.5),
    ("Amundsen",          -84.5,  83.0,  1.5),
    # Broader south-pole region (everything south of -80° is candidate)
    ("SouthPole_general", -85.0,   0.0, 10.0),
]


def extract_product(filename: str) -> str:
    stem = filename[:-4] if filename.endswith(".png") else filename
    m = re.match(r"^(.+)_x\d+_y\d+$", stem)
    return m.group(1) if m else "UNKNOWN"


# ── Step 1: Find geometry CSV for each product ────────────────────────────
def find_geometry_csv(product_id: str) -> Path | None:
    """Search for the geometry CSV in the extracted bundle."""
    bundle_dir = EXTRACTED_BASE / product_id
    if not bundle_dir.exists():
        return None
    # geometry/calibrated/<date>/<product_id_with_g_grd>.csv
    for csv_path in bundle_dir.glob("geometry/calibrated/*/*.csv"):
        return csv_path
    return None


def build_interpolator(geo_csv: Path):
    """
    Build a 2D bilinear interpolator from the sparse geometry grid.

    The CSV has columns: Longitude, Latitude, Pixel, Scan
    Pixel = sample axis (x), Scan = line axis (y)
    Grid spacing: every 100 pixels in x, every scan line in y.

    Returns two RegularGridInterpolator objects:
      interp_lon(pixel, scan) -> longitude
      interp_lat(pixel, scan) -> latitude
    """
    df = pd.read_csv(geo_csv)
    df.columns = df.columns.str.strip()

    # Pivot to grids
    pixel_vals = np.sort(df["Pixel"].unique())
    scan_vals  = np.sort(df["Scan"].unique())

    # Build 2D arrays — df may be large, use pivot
    lon_grid = (
        df.pivot(index="Scan", columns="Pixel", values="Longitude")
          .reindex(index=scan_vals, columns=pixel_vals)
    )
    lat_grid = (
        df.pivot(index="Scan", columns="Pixel", values="Latitude")
          .reindex(index=scan_vals, columns=pixel_vals)
    )

    # Fill any NaN gaps via forward-fill
    lon_grid = lon_grid.ffill(axis=1).ffill(axis=0).bfill(axis=1).bfill(axis=0)
    lat_grid = lat_grid.ffill(axis=1).ffill(axis=0).bfill(axis=1).bfill(axis=0)

    interp_lon = RegularGridInterpolator(
        (scan_vals, pixel_vals), lon_grid.values,
        method="linear", bounds_error=False, fill_value=None
    )
    interp_lat = RegularGridInterpolator(
        (scan_vals, pixel_vals), lat_grid.values,
        method="linear", bounds_error=False, fill_value=None
    )
    return interp_lon, interp_lat, scan_vals.min(), scan_vals.max(), pixel_vals.min(), pixel_vals.max()


# ── Step 2: Assign lat/lon to tile center ─────────────────────────────────
def tile_center_latlon(interp_lon, interp_lat, tile_x: int, tile_y: int) -> tuple[float, float]:
    """
    tile_x = Pixel offset (sample axis)
    tile_y = Scan offset (line axis)
    Center = (tile_x + 320, tile_y + 320)
    """
    cx = tile_x + TILE_SIZE // 2
    cy = tile_y + TILE_SIZE // 2
    pt = np.array([[cy, cx]])   # RegularGridInterpolator expects (scan, pixel)
    lon = float(interp_lon(pt)[0])
    lat = float(interp_lat(pt)[0])
    return lat, lon


# ── Step 3a: Proximity score ──────────────────────────────────────────────
def proximity_score(lat: float, lon: float) -> float:
    """
    Returns a score 0–1 based on proximity to known feature locations.
    Score = max across all features of: max(0, 1 - dist/radius)
    """
    best = 0.0
    for name, f_lat, f_lon, radius in KNOWN_FEATURES:
        # Simple Euclidean distance in degrees (good enough near south pole)
        dlat = lat - f_lat
        dlon = lon - f_lon
        dist = np.sqrt(dlat**2 + dlon**2)
        score = max(0.0, 1.0 - dist / radius)
        best = max(best, score)
    return best


# ── Step 3b: Variance score from PNG ─────────────────────────────────────
def compute_std(tile_path: Path) -> float:
    """Load tile PNG and return pixel std deviation."""
    try:
        arr = np.array(Image.open(tile_path), dtype=np.float32)
        return float(arr.std())
    except Exception:
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("OHRC Geographic Pre-Filter")
    print("=" * 65)

    # Load manifest — usable tiles only
    print(f"\n[1] Loading manifest from {MANIFEST_CSV.name}...")
    mdf = pd.read_csv(MANIFEST_CSV)
    usable = mdf[mdf["status"] == "usable"].copy()
    print(f"  Usable tiles in manifest: {len(usable):,}")

    # Fix stale source paths in manifest (may point to old date-based dirs)
    # We don't need source paths — we use product_id from tile filename
    usable["product_id"] = usable["tile_filename"].apply(extract_product)

    # ── Build geometry interpolators per product ──────────────────────────
    print("\n[2] Building geometry interpolators for each product...")
    interpolators = {}  # product_id -> (interp_lon, interp_lat)
    skipped = []

    for product_id in sorted(usable["product_id"].unique()):
        geo_csv = find_geometry_csv(product_id)
        if geo_csv is None:
            print(f"  [WARN] No geometry CSV found for {product_id} — will skip geolocation")
            skipped.append(product_id)
            continue
        print(f"  Loading {geo_csv.name} ...", end=" ", flush=True)
        try:
            interp_lon, interp_lat, smin, smax, pmin, pmax = build_interpolator(geo_csv)
            interpolators[product_id] = (interp_lon, interp_lat)
            print(f"OK  (scan {smin:.0f}–{smax:.0f}, pixel {pmin:.0f}–{pmax:.0f})")
        except Exception as e:
            print(f"ERROR: {e}")
            skipped.append(product_id)

    # ── Compute lat/lon + std for every usable tile ───────────────────────
    print(f"\n[3] Scoring {len(usable):,} tiles (lat/lon + std + proximity)...")
    print("    This may take 5-10 minutes...")

    records = []
    no_geo_count = 0

    for _, row in tqdm(usable.iterrows(), total=len(usable), unit="tile"):
        fname      = row["tile_filename"]
        product_id = row["product_id"]
        tile_x     = int(row["tile_x"])
        tile_y     = int(row["tile_y"])

        # Lat/lon
        if product_id in interpolators:
            interp_lon, interp_lat = interpolators[product_id]
            lat, lon = tile_center_latlon(interp_lon, interp_lat, tile_x, tile_y)
            prox = proximity_score(lat, lon)
        else:
            lat, lon, prox = float("nan"), float("nan"), 0.0
            no_geo_count += 1

        # Std deviation
        tile_path = TILES_USABLE / fname
        std = compute_std(tile_path)

        records.append({
            "tile_filename": fname,
            "source_product": product_id,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "center_lat": round(lat, 6),
            "center_lon": round(lon, 6),
            "std_dev": round(std, 3),
            "prox_score": round(prox, 4),
            "mean_pixel": round(float(row.get("mean_pixel_value", 0)), 2),
        })

    geo_df = pd.DataFrame(records)

    # ── Composite priority score ──────────────────────────────────────────
    # Normalize std_dev to [0,1] across dataset
    std_max = geo_df["std_dev"].max()
    std_min = geo_df["std_dev"].min()
    if std_max > std_min:
        geo_df["std_norm"] = (geo_df["std_dev"] - std_min) / (std_max - std_min)
    else:
        geo_df["std_norm"] = 0.0

    # Priority = 0.6 × proximity + 0.4 × normalized_std
    # Proximity is the dominant factor since features are location-specific
    geo_df["priority_score"] = (
        0.6 * geo_df["prox_score"] + 0.4 * geo_df["std_norm"]
    ).round(4)

    # Save full geolocation table
    geo_df.to_csv(GEO_CACHE_CSV, index=False)
    print(f"\n  Geolocation table → {GEO_CACHE_CSV}")
    if no_geo_count:
        print(f"  WARNING: {no_geo_count} tiles had no geometry data (no lat/lon)")

    # ── Coverage summary ──────────────────────────────────────────────────
    print("\n[4] Coverage summary per product:")
    for prod, grp in geo_df.groupby("source_product"):
        lat_range = f"{grp['center_lat'].min():.2f}° to {grp['center_lat'].max():.2f}°"
        lon_range = f"{grp['center_lon'].min():.2f}° to {grp['center_lon'].max():.2f}°"
        near_pole = (grp["center_lat"] < -80).sum()
        print(f"  {prod[-30:]}  lat:{lat_range}  lon:{lon_range}  tiles<-80°: {near_pole}")

    # ── Select top-N by priority score ───────────────────────────────────
    print(f"\n[5] Selecting top {TOP_N} tiles by priority score...")
    top = geo_df.nlargest(TOP_N, "priority_score")

    # Export ranked CSV
    export_cols = [
        "tile_filename", "source_product", "tile_x", "tile_y",
        "center_lat", "center_lon", "std_dev", "prox_score",
        "priority_score", "mean_pixel"
    ]
    top[export_cols].to_csv(OUT_CSV, index=False)
    print(f"  Priority tile list → {OUT_CSV}")

    # ── Copy top tiles ────────────────────────────────────────────────────
    print(f"\n[6] Copying top {TOP_N} tiles to {PRIORITY_DIR.name}/...")
    copied, missing = 0, 0
    for fname in tqdm(top["tile_filename"], unit="tile"):
        src = TILES_USABLE / fname
        dst = PRIORITY_DIR / fname
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

    print(f"  Copied: {copied}  |  Not found: {missing}")

    # ── Score distribution summary ────────────────────────────────────────
    print("\n[7] Score distribution:")
    print(f"  Tiles with prox_score > 0  (near known features): "
          f"{(geo_df['prox_score'] > 0).sum():,}")
    print(f"  Tiles with prox_score > 0.5 (close to feature):  "
          f"{(geo_df['prox_score'] > 0.5).sum():,}")
    print(f"  Tiles with prox_score == 1.0 (at feature center):"
          f"{(geo_df['prox_score'] == 1.0).sum():,}")
    print(f"  Std dev — mean: {geo_df['std_dev'].mean():.1f}  "
          f"max: {geo_df['std_dev'].max():.1f}  "
          f"p95: {geo_df['std_dev'].quantile(0.95):.1f}")

    print("\n  Top-10 tiles by priority:")
    print(top[["tile_filename","center_lat","center_lon",
               "std_dev","prox_score","priority_score"]].head(10).to_string(index=False))

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Geographic Pre-Filter — Complete")
    print(f"\n  Priority tiles CSV: {OUT_CSV}")
    print(f"  Priority tile copies: {PRIORITY_DIR}")
    print(f"  Full geolocation table: {GEO_CACHE_CSV}")
    print("\n  Next step: open priority_tiles.csv and review the top tiles.")
    print("  Then run: python scripts/review_priority_tiles.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
