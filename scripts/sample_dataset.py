"""
sample_dataset.py  — Steps 4 & 5
==================================
Stratified proportional random sampling of 200 tiles from the 32,169
usable tile pool across all 12 OHRC source products.

Sampling strategy:
  - Proportional allocation: each product contributes
    round(200 * product_tiles / total_tiles) images.
  - Random seed: 42 (fully reproducible).
  - Tiles are COPIED (not moved) to data/annotation_sample/.
  - Original files are never modified.

Outputs:
  data/annotation_sample/          (200 PNG copies)
  results/quality_control/sampling_manifest.csv

Usage:
  python scripts/sample_dataset.py
"""
import re
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
TILES_DIR     = PROJECT_ROOT / "data" / "tiles" / "usable"
SAMPLE_DIR    = PROJECT_ROOT / "data" / "annotation_sample"
QC_DIR        = PROJECT_ROOT / "results" / "quality_control"
MANIFEST_CSV  = QC_DIR / "sampling_manifest.csv"

TOTAL_SAMPLE  = 200
RANDOM_SEED   = 42
SAMPLING_METHOD = "stratified_proportional_random"

def extract_product(filename: str) -> str:
    """
    Extract source OHRC product_id from a tile filename.
    Strips the trailing _x<digits>_y<digits>.png suffix.
    Robust to any product_id format — does not depend on fixed digit counts.

    Example:
      ch2_ohr_ncp_20190906T2241285714_d_img_gds_x00000_y00000.png
      -> ch2_ohr_ncp_20190906T2241285714_d_img_gds
    """
    stem = filename[:-4] if filename.endswith(".png") else filename
    m = re.match(r"^(.+)_x\d+_y\d+$", stem)
    return m.group(1) if m else "UNKNOWN"



def allocate_proportional(product_counts: dict[str, int], total: int) -> dict[str, int]:
    """
    Proportional allocation with rounding that sums exactly to `total`.
    Uses largest-remainder method to fix rounding drift.
    """
    grand = sum(product_counts.values())
    exact = {p: total * cnt / grand for p, cnt in product_counts.items()}
    floored = {p: int(v) for p, v in exact.items()}
    remainder = total - sum(floored.values())
    # Assign remaining slots to products with largest fractional parts
    fracs = sorted(exact.items(), key=lambda x: -(x[1] - int(x[1])))
    for i in range(remainder):
        floored[fracs[i][0]] += 1
    return floored


def main():
    print("=" * 60)
    print("Dataset Stratified Sampling — Steps 4 & 5")
    print(f"  Target: {TOTAL_SAMPLE} tiles  |  Seed: {RANDOM_SEED}")
    print("=" * 60)

    all_files = sorted(TILES_DIR.glob("*.png"))
    if not all_files:
        print(f"ERROR: No PNG files found under {TILES_DIR}")
        sys.exit(1)

    total_tiles = len(all_files)
    print(f"  Total usable tiles: {total_tiles:,}")

    # Group by product
    by_product: dict[str, list[Path]] = defaultdict(list)
    for p in all_files:
        by_product[extract_product(p.name)].append(p)

    n_products = len(by_product)
    print(f"  Source products:    {n_products}")

    # Proportional allocation
    product_counts = {k: len(v) for k, v in by_product.items()}
    allocation = allocate_proportional(product_counts, TOTAL_SAMPLE)

    print(f"\n  Allocation (seed={RANDOM_SEED}):")
    rng = random.Random(RANDOM_SEED)

    selected: list[dict] = []
    sample_id = 1
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    for prod in sorted(by_product.keys()):
        pool  = sorted(by_product[prod])  # sorted for reproducibility
        k     = allocation.get(prod, 0)
        drawn = rng.sample(pool, min(k, len(pool)))
        print(f"    {prod}  pool={len(pool):5d}  draw={k:3d}")

        for src in drawn:
            dst = SAMPLE_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)   # copy2 preserves metadata
            selected.append({
                "sample_id":     sample_id,
                "tile_filename": src.name,
                "source_product": prod,
                "random_seed":   RANDOM_SEED,
                "sampling_method": SAMPLING_METHOD,
            })
            sample_id += 1

    # Write manifest
    QC_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(selected)
    df.to_csv(MANIFEST_CSV, index=False)

    print(f"\n  Selected {len(selected)} tiles → {SAMPLE_DIR}")
    print(f"  Manifest → {MANIFEST_CSV}")
    print("=" * 60)
    return len(selected)


if __name__ == "__main__":
    main()
