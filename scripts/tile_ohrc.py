"""
tile_ohrc.py
============
Generates 640 × 640 pixel non-overlapping tiles from every extracted OHRC
calibrated product, applies dark/shadow filtering, and saves usable PNG tiles.

Design decisions (documented for reproducibility):
  ─────────────────────────────────────────────────
  1. TILING STRATEGY: Non-overlapping 640 × 640 tiles.
     - Overlap is NOT used (overlap=0) for the initial tiling pass.
       Overlap would expand the dataset but complicate annotation deduplication.
     - Edge tiles (where the image width/height is not a multiple of 640)
       are ZERO-PADDED to exactly 640 × 640.  They are flagged as
       'edge_padded=True' in the manifest. They are NOT silently stretched
       or dropped. They participate in dark filtering like any other tile.

  2. DARK TILE THRESHOLD DOMAIN:
     - Data type: UnsignedByte (uint8, 0–255).
     - No scaling_factor or value_offset is declared in the PDS4 XML for
       these calibrated (NCP) products.
     - Therefore the dark tile threshold of mean < 20 is applied to the
       NATIVE calibrated pixel values (raw uint8), not to any normalized
       representation.
     - Normalization (percentile stretch → uint8) is performed ONLY when
       writing the PNG tile files for machine-learning use. The threshold
       decision is made BEFORE normalization.
     - This separation ensures that filtering criteria are physically
       meaningful and reproducible regardless of per-image normalization.

  3. PNG CONVERSION:
     - Each tile is normalized (percentile stretch p2–p98 → uint8).
     - Saved as lossless PNG (no JPEG — lossy compression is undesirable
       for annotation imagery).
     - Normalization is computed per-tile (not per-image) to maximize local
       contrast in each 640 × 640 region.
     - The .img source file is never written to.

  4. MEMORY EFFICIENCY:
     - The full image is never loaded into RAM at once.
     - Instead, we read horizontal strips of STRIP_LINES lines at a time.
     - For a 12000-sample UnsignedByte image:
         640 lines × 12000 px × 1 byte = 7.68 MB per strip
     - Peak RAM usage per product ≈ 8–16 MB (well within 8 GB).

Usage:
    python scripts/tile_ohrc.py [--product <substring>]

Outputs:
    data/tiles/all/<product_id>_x<XXXX>_y<YYYY>.png   (all tiles)
    data/tiles/usable/<product_id>_x<XXXX>_y<YYYY>.png (usable tiles)
    results/tile_statistics.csv
    results/preprocessing_errors.log
"""

import sys
import csv
import shutil
import argparse
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inspect_pds4 import (
    find_pds4_pairs,
    parse_pds4_label,
    pds4_dtype_to_numpy,
    log_error,
    EXTRACT_BASE,
)

TILES_ALL_DIR = PROJECT_ROOT / "data" / "tiles" / "all"
TILES_USABLE_DIR = PROJECT_ROOT / "data" / "tiles" / "usable"
RESULTS_DIR = PROJECT_ROOT / "results"
TILE_STATS_CSV = RESULTS_DIR / "tile_statistics.csv"

TILES_ALL_DIR.mkdir(parents=True, exist_ok=True)
TILES_USABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 640          # px — tile width and height
DARK_THRESHOLD = 20      # mean native pixel value below which tile is discarded
TILE_STATS_FIELDS = [
    "product_id", "date",
    "total_tiles", "usable_tiles", "discarded_tiles", "pct_discarded",
    "image_lines", "image_samples",
    "n_tile_cols", "n_tile_rows",
    "tile_size", "overlap", "edge_handling",
    "dark_threshold", "threshold_domain",
    "image_pix_min", "image_pix_max", "image_pix_mean",
]

MANIFEST_ROWS = []  # populated during tiling, flushed by create_dataset_manifest


def normalize_tile_for_png(tile: np.ndarray) -> np.ndarray:
    """
    Percentile stretch (p2–p98) within the tile → uint8.
    Operates on a copy; never modifies native scientific values.
    NOTE: This is used for the PNG output files only. The dark tile
    filtering decision uses the native values (see DARK THRESHOLD DOMAIN).
    """
    p_low = float(np.percentile(tile, 2))
    p_high = float(np.percentile(tile, 98))
    if p_high <= p_low:
        return np.zeros(tile.shape, dtype=np.uint8)
    norm = np.clip(tile.astype(np.float32), p_low, p_high)
    norm = (norm - p_low) / (p_high - p_low) * 255.0
    return norm.astype(np.uint8)


def tile_product(pair: dict) -> dict | None:
    """
    Tile a single product. Returns per-product statistics dict or None.
    """
    product_id = pair["product_id"]
    xml_path = pair["xml_path"]
    img_path = pair["img_path"]
    date = pair["date"]

    # Check if this product is already tiled (look for any tile with this prefix)
    existing = list(TILES_ALL_DIR.glob(f"{product_id}_*.png"))
    if existing:
        print(f"  [SKIP] {product_id} — {len(existing)} tiles already present in all/")
        # Recompute quick stats from existing tiles for reporting
        usable = list(TILES_USABLE_DIR.glob(f"{product_id}_*.png"))
        return {
            "product_id": product_id, "date": date,
            "total_tiles": len(existing), "usable_tiles": len(usable),
            "discarded_tiles": len(existing) - len(usable),
            "pct_discarded": round(100 * (len(existing) - len(usable)) / max(len(existing), 1), 2),
            "image_lines": "?", "image_samples": "?",
            "n_tile_cols": "?", "n_tile_rows": "?",
            "tile_size": TILE_SIZE, "overlap": 0,
            "edge_handling": "zero_pad",
            "dark_threshold": DARK_THRESHOLD, "threshold_domain": "native_uint8",
            "image_pix_min": "?", "image_pix_max": "?", "image_pix_mean": "?",
        }

    print(f"\n[TILE] {product_id}")

    label = parse_pds4_label(xml_path)
    lines = label["lines"]
    samples = label["samples"]
    data_type = label["data_type"]
    offset = label["offset"]

    if None in (lines, samples, data_type):
        msg = "Could not parse label."
        print(f"  [ERROR] {msg}")
        log_error(product_id, msg)
        return None

    dtype = pds4_dtype_to_numpy(data_type)
    if dtype is None:
        msg = f"Unknown dtype: {data_type}"
        print(f"  [ERROR] {msg}")
        log_error(product_id, msg)
        return None

    bytes_per_pixel = np.dtype(dtype).itemsize
    row_bytes = samples * bytes_per_pixel

    n_tile_cols = (samples + TILE_SIZE - 1) // TILE_SIZE
    n_tile_rows = (lines + TILE_SIZE - 1) // TILE_SIZE
    total_tiles = n_tile_cols * n_tile_rows

    print(f"  {lines} lines × {samples} samples | {data_type}")
    print(f"  Tile grid: {n_tile_rows} rows × {n_tile_cols} cols = {total_tiles} tiles")

    total_count = 0
    usable_count = 0
    discarded_count = 0
    image_pix_sum = 0.0
    image_pix_min = np.inf
    image_pix_max = -np.inf
    image_pixels = 0

    tile_row_stats = []  # list of dicts for manifest

    with open(img_path, "rb") as f:
        # Iterate strip by strip (each strip = TILE_SIZE lines)
        for tile_row in tqdm(range(n_tile_rows), desc=f"  Tiling {product_id[:30]}", unit="strip"):
            start_line = tile_row * TILE_SIZE
            end_line = min(start_line + TILE_SIZE, lines)
            n_lines_this = end_line - start_line

            # Read the strip
            f.seek(offset + start_line * row_bytes)
            raw = f.read(n_lines_this * row_bytes)
            if len(raw) < n_lines_this * row_bytes:
                log_error(product_id, f"Truncated read at strip {tile_row}")
                break

            strip = np.frombuffer(raw, dtype=dtype).reshape(n_lines_this, samples)

            # Track image-level stats from native values
            image_pix_sum += strip.sum(dtype=np.float64)
            image_pix_min = min(image_pix_min, float(strip.min()))
            image_pix_max = max(image_pix_max, float(strip.max()))
            image_pixels += strip.size

            # Pad strip to TILE_SIZE rows if it's an edge strip
            if n_lines_this < TILE_SIZE:
                pad_rows = TILE_SIZE - n_lines_this
                strip = np.pad(strip, ((0, pad_rows), (0, 0)), mode="constant", constant_values=0)

            # Slice columns into tiles
            for tile_col in range(n_tile_cols):
                start_sample = tile_col * TILE_SIZE
                end_sample = min(start_sample + TILE_SIZE, samples)
                n_samples_this = end_sample - start_sample

                tile_data = strip[:, start_sample:end_sample]

                is_edge_padded = (n_lines_this < TILE_SIZE) or (n_samples_this < TILE_SIZE)

                # Pad column direction if needed
                if n_samples_this < TILE_SIZE:
                    pad_cols = TILE_SIZE - n_samples_this
                    tile_data = np.pad(tile_data, ((0, 0), (0, pad_cols)),
                                       mode="constant", constant_values=0)

                # ── DARK TILE FILTERING (native uint8 values) ─────────────────
                # Threshold is applied to native calibrated pixel values (0–255).
                # No normalization is applied before filtering.
                tile_mean = float(tile_data.mean())
                tile_min = float(tile_data.min())
                tile_max = float(tile_data.max())
                is_dark = tile_mean < DARK_THRESHOLD

                # Build output filename
                fname = f"{product_id}_x{start_sample:05d}_y{start_line:05d}.png"

                # ── PNG CONVERSION (normalization for ML imagery) ─────────────
                # Normalization happens HERE, after the filtering decision,
                # to keep the two operations strictly separate.
                tile_png = normalize_tile_for_png(tile_data)
                img_obj = Image.fromarray(tile_png)  # dtype=uint8 → mode 'L' inferred

                # Always save to all/
                all_path = TILES_ALL_DIR / fname
                img_obj.save(all_path, format="PNG", compress_level=6)

                # Save to usable/ only if not dark
                status = "dark_discarded" if is_dark else "usable"
                if not is_dark:
                    usable_path = TILES_USABLE_DIR / fname
                    img_obj.save(usable_path, format="PNG", compress_level=6)
                    usable_count += 1
                else:
                    discarded_count += 1

                total_count += 1

                # Record for manifest
                tile_row_stats.append({
                    "source_product": product_id,
                    "source_date": date,
                    "source_xml": str(xml_path),
                    "source_img": str(img_path),
                    "tile_filename": fname,
                    "tile_x": start_sample,
                    "tile_y": start_line,
                    "tile_width": TILE_SIZE,
                    "tile_height": TILE_SIZE,
                    "mean_pixel_value": round(tile_mean, 3),
                    "min_pixel_value": tile_min,
                    "max_pixel_value": tile_max,
                    "is_edge_padded": is_edge_padded,
                    "status": status,
                })

    MANIFEST_ROWS.extend(tile_row_stats)

    pct_discarded = round(100 * discarded_count / max(total_count, 1), 2)
    image_pix_mean = image_pix_sum / max(image_pixels, 1)

    print(f"  Total tiles:     {total_count}")
    print(f"  Usable tiles:    {usable_count}")
    print(f"  Discarded dark:  {discarded_count} ({pct_discarded:.1f}%)")

    return {
        "product_id": product_id,
        "date": date,
        "total_tiles": total_count,
        "usable_tiles": usable_count,
        "discarded_tiles": discarded_count,
        "pct_discarded": pct_discarded,
        "image_lines": lines,
        "image_samples": samples,
        "n_tile_cols": n_tile_cols,
        "n_tile_rows": n_tile_rows,
        "tile_size": TILE_SIZE,
        "overlap": 0,
        "edge_handling": "zero_pad",
        "dark_threshold": DARK_THRESHOLD,
        "threshold_domain": "native_uint8",
        "image_pix_min": round(image_pix_min, 3) if image_pix_min != np.inf else "nan",
        "image_pix_max": round(image_pix_max, 3) if image_pix_max != -np.inf else "nan",
        "image_pix_mean": round(image_pix_mean, 3),
    }


def save_tile_stats(stats_list: list[dict]) -> None:
    if not stats_list:
        return
    mode = "a" if TILE_STATS_CSV.exists() else "w"
    fieldnames = TILE_STATS_FIELDS
    with open(TILE_STATS_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(stats_list)
    print(f"\nTile statistics saved to: {TILE_STATS_CSV}")


def save_manifest_rows(rows: list[dict]) -> None:
    """Append tile manifest rows to results/dataset_manifest.csv."""
    if not rows:
        return
    manifest_path = RESULTS_DIR / "dataset_manifest.csv"
    mode = "a" if manifest_path.exists() else "w"
    fieldnames = list(rows[0].keys())
    with open(manifest_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Tile OHRC products into 640x640 PNG tiles.")
    parser.add_argument("--product", type=str, default=None,
                        help="Filter to product containing this substring (e.g. 20190906)")
    args = parser.parse_args()

    print("=" * 70)
    print("Chandrayaan-2 OHRC — 640×640 Tiling")
    print("=" * 70)
    print(f"Tile size:      {TILE_SIZE} × {TILE_SIZE} px")
    print(f"Overlap:        None (non-overlapping)")
    print(f"Edge handling:  Zero-padding to {TILE_SIZE} × {TILE_SIZE}")
    print(f"Dark threshold: mean < {DARK_THRESHOLD} (native uint8 calibrated values)")
    print(f"PNG norm:       Per-tile percentile stretch p2–p98 (display only)")
    print()

    pairs = find_pds4_pairs(EXTRACT_BASE, product_filter=args.product)
    if not pairs:
        print("[ERROR] No PDS4 pairs found.")
        return

    print(f"Found {len(pairs)} product(s) to tile.\n")

    stats_list = []
    success, failed = 0, 0

    for pair in pairs:
        try:
            stats = tile_product(pair)
            if stats:
                stats_list.append(stats)
                success += 1
            else:
                failed += 1
        except Exception as e:
            msg = f"Unhandled exception: {e}\n{traceback.format_exc()}"
            print(f"  [ERROR] {pair['product_id']}: {msg}")
            log_error(pair["product_id"], msg)
            failed += 1

    save_tile_stats(stats_list)
    save_manifest_rows(MANIFEST_ROWS)

    total_tiles = sum(s["total_tiles"] for s in stats_list if isinstance(s["total_tiles"], int))
    usable_tiles = sum(s["usable_tiles"] for s in stats_list if isinstance(s["usable_tiles"], int))
    discarded_tiles = sum(s["discarded_tiles"] for s in stats_list if isinstance(s["discarded_tiles"], int))

    print("\n" + "=" * 70)
    print("Tiling summary:")
    print(f"  Products processed: {success}/{len(pairs)}")
    print(f"  Total tiles:        {total_tiles:,}")
    print(f"  Usable tiles:       {usable_tiles:,}")
    print(f"  Discarded dark:     {discarded_tiles:,}")
    pct = round(100 * discarded_tiles / max(total_tiles, 1), 1)
    print(f"  % discarded:        {pct}%")
    print(f"  All tiles dir:      {TILES_ALL_DIR}")
    print(f"  Usable tiles dir:   {TILES_USABLE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
