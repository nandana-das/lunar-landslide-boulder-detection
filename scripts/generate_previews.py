"""
generate_previews.py
====================
Generates normalized PNG preview images for all extracted OHRC calibrated
products. Previews are for human inspection only — scientific pixel values
are never modified.

Design decisions:
  - Large images (101k × 12k) are SUBSAMPLED for preview (every N-th row/col)
    to keep output manageable. The subsampling factor is auto-computed so the
    preview is at most ~2000 px tall.
  - Normalization for display: percentile stretch (p2–p98) → uint8.
    This is done on the subsampled array to keep RAM usage low.
  - Colormap: grayscale (panchromatic instrument).
  - The original .img file is never written to.

Usage:
    python scripts/generate_previews.py [--product <substring>]

Outputs:
    data/previews/<product_id>_preview.png
"""

import sys
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
)

PREVIEWS_DIR = PROJECT_ROOT / "data" / "previews"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# Target maximum preview height in pixels (auto-subsampling will target this)
MAX_PREVIEW_HEIGHT = 2000
# Target maximum preview width — OHRC is 12000 px wide, subsample to this
MAX_PREVIEW_WIDTH = 2000


def subsample_factor(lines: int, samples: int) -> tuple[int, int]:
    """Compute row and column subsampling factors."""
    row_factor = max(1, lines // MAX_PREVIEW_HEIGHT)
    col_factor = max(1, samples // MAX_PREVIEW_WIDTH)
    return row_factor, col_factor


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """
    Percentile stretch (p2–p98) to uint8 for display.
    This does NOT modify scientific values — only operates on the subsampled
    preview array derived for visualization purposes.
    """
    p_low = float(np.percentile(arr, 2))
    p_high = float(np.percentile(arr, 98))
    if p_high == p_low:
        return np.zeros_like(arr, dtype=np.uint8)
    clipped = np.clip(arr.astype(np.float32), p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low) * 255.0
    return normalized.astype(np.uint8)


def generate_preview(pair: dict) -> bool:
    """Generate a single product preview. Returns True on success."""
    product_id = pair["product_id"]
    xml_path = pair["xml_path"]
    img_path = pair["img_path"]

    out_path = PREVIEWS_DIR / f"{product_id}_preview.png"
    if out_path.exists():
        print(f"  [SKIP] Preview already exists: {out_path.name}")
        return True

    print(f"\n[PREVIEW] {product_id}")

    # Parse label
    label = parse_pds4_label(xml_path)
    lines = label["lines"]
    samples = label["samples"]
    data_type = label["data_type"]
    offset = label["offset"]

    if None in (lines, samples, data_type):
        msg = "Could not parse label fields."
        print(f"  [ERROR] {msg}")
        log_error(product_id, msg)
        return False

    dtype = pds4_dtype_to_numpy(data_type)
    if dtype is None:
        msg = f"Unknown PDS4 data type: {data_type}"
        print(f"  [ERROR] {msg}")
        log_error(product_id, msg)
        return False

    row_factor, col_factor = subsample_factor(lines, samples)
    print(f"  Image: {lines} × {samples} px | dtype: {data_type}")
    print(f"  Subsampling: every {row_factor} rows, every {col_factor} cols")
    print(f"  Preview size: ~{lines // row_factor} × {samples // col_factor} px")

    # Read subsampled rows from the raw binary file
    bytes_per_pixel = np.dtype(dtype).itemsize
    row_bytes = samples * bytes_per_pixel

    sampled_rows = []
    selected_line_indices = range(0, lines, row_factor)

    with open(img_path, "rb") as f:
        for line_idx in tqdm(selected_line_indices, desc="  Reading rows", leave=False, unit="row"):
            f.seek(offset + line_idx * row_bytes)
            raw = f.read(row_bytes)
            if len(raw) < row_bytes:
                break
            row = np.frombuffer(raw, dtype=dtype)
            # Subsample columns
            row_sub = row[::col_factor]
            sampled_rows.append(row_sub)

    if not sampled_rows:
        msg = "No rows read — image may be empty or unreadable."
        print(f"  [ERROR] {msg}")
        log_error(product_id, msg)
        return False

    preview_arr = np.stack(sampled_rows, axis=0)
    print(f"  Subsampled array shape: {preview_arr.shape}")
    print(f"  Native value range: min={preview_arr.min()}, max={preview_arr.max()}, "
          f"mean={preview_arr.mean():.2f}")

    # Normalize ONLY for display (separate from scientific processing)
    display_arr = normalize_for_display(preview_arr)

    # Save as PNG (lossless, grayscale)
    img = Image.fromarray(display_arr)  # uint8 → 'L' inferred
    img.save(out_path, format="PNG", compress_level=6)

    file_kb = out_path.stat().st_size // 1024
    print(f"  Preview saved: {out_path.name} ({file_kb:,} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate OHRC PDS4 preview images.")
    parser.add_argument("--product", type=str, default=None,
                        help="Filter to product containing this substring (e.g. 20190906)")
    args = parser.parse_args()

    print("=" * 70)
    print("Chandrayaan-2 OHRC — Preview Generation")
    print("=" * 70)

    from inspect_pds4 import EXTRACT_BASE
    pairs = find_pds4_pairs(EXTRACT_BASE, product_filter=args.product)
    if not pairs:
        print("[ERROR] No PDS4 pairs found.")
        return

    print(f"Found {len(pairs)} product(s).\n")

    success, skipped, failed = 0, 0, 0
    for pair in pairs:
        out = PREVIEWS_DIR / f"{pair['product_id']}_preview.png"
        if out.exists():
            skipped += 1
            print(f"  [SKIP] {pair['product_id']}")
            continue
        try:
            ok = generate_preview(pair)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            msg = f"Unhandled exception: {e}\n{traceback.format_exc()}"
            print(f"  [ERROR] {pair['product_id']}: {msg}")
            log_error(pair["product_id"], msg)
            failed += 1

    print("\n" + "=" * 70)
    print(f"Preview generation complete:")
    print(f"  Generated: {success}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Output:    {PREVIEWS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
