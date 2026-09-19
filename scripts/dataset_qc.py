"""
dataset_qc.py  — Steps 1, 2, 3, 9, 10
======================================
Chandrayaan-2 OHRC usable-tile dataset quality control.

Steps performed:
  1. Inspect dataset (count, dimensions, mode, file-size stats)
  2. Validate every PNG (header-only read, batch of 500)
  3. Source-product distribution
  9. Visual-quality flags from file-size heuristics (reported only)
  10. Exact-duplicate detection via MD5 (streaming, no full load)

Outputs:
  results/quality_control/image_validation.csv
  results/quality_control/source_tile_distribution.csv
  results/quality_control/duplicate_report.csv

Usage:
  python scripts/dataset_qc.py
"""
import sys
import csv
import hashlib
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TILES_DIR    = PROJECT_ROOT / "data" / "tiles" / "usable"
QC_DIR       = PROJECT_ROOT / "results" / "quality_control"
QC_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_CSV   = QC_DIR / "image_validation.csv"
DISTRIBUTION_CSV = QC_DIR / "source_tile_distribution.csv"
DUPLICATE_CSV    = QC_DIR / "duplicate_report.csv"

EXPECTED_W = 640
EXPECTED_H = 640
EXPECTED_MODE = "L"
BATCH_SIZE = 500
# File-size thresholds for visual-quality heuristics (bytes)
VERY_SMALL_KB = 80   # likely blank / near-uniform / mostly shadow
VERY_LARGE_KB = 280  # unusual — flag for inspection


# ── Helpers ────────────────────────────────────────────────────────────────
# Tile filenames follow the pattern:
#   <product_id>_x<XXXXX>_y<YYYYY>.png
# where product_id = ch2_ohr_ncp_<8-digit-date>T<10-digit-time>_d_img_<3-char-suffix>
# e.g.: ch2_ohr_ncp_20190906T2241285714_d_img_gds_x00000_y00000.png
#
# BUG FIX: original regex used \d{13} for the timestamp — the actual
# timestamp component has 10 digits (e.g. 2241285714), not 13.
# Using a robust stem-strip approach instead of a fixed digit count.

def extract_product(filename: str) -> str:
    """
    Extract the source OHRC product_id from a tile filename.
    Strategy: strip the trailing _x<digits>_y<digits>.png suffix.
    This is robust to any product_id format.
    """
    stem = filename
    if stem.endswith(".png"):
        stem = stem[:-4]
    # Match and strip _x<digits>_y<digits> at the end
    m = re.match(r"^(.+)_x\d+_y\d+$", stem)
    return m.group(1) if m else "UNKNOWN"


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ── Step 1+2: File inventory and validation ────────────────────────────────
def run_validation(all_files: list[Path]):
    print(f"\n[Step 1+2] Validating {len(all_files):,} PNG files in batches of {BATCH_SIZE}...")
    rows = []
    invalid_count = 0
    size_list = []
    mode_counts = defaultdict(int)
    dim_counts  = defaultdict(int)

    # Sample pixel stats from first 200 valid files
    sample_pix = []

    with open(VALIDATION_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["filename", "valid", "width", "height",
                         "mode", "file_size_bytes", "error", "vq_flag"])

        for i in range(0, len(all_files), BATCH_SIZE):
            batch = all_files[i : i + BATCH_SIZE]
            for p in tqdm(batch, desc=f"Batch {i//BATCH_SIZE+1}", leave=False, unit="img"):
                fname     = p.name
                fsize     = p.stat().st_size
                size_list.append(fsize)
                valid     = False
                w = h = None
                mode = ""
                error = ""
                vq_flag = ""

                # Zero-byte check
                if fsize == 0:
                    error = "zero_byte"
                    invalid_count += 1
                else:
                    try:
                        img = Image.open(p)   # header only — no .load()
                        w, h = img.size
                        mode = img.mode
                        valid = True
                        mode_counts[mode] += 1
                        dim_counts[(w, h)] += 1

                        # Dimension check
                        if w != EXPECTED_W or h != EXPECTED_H:
                            error = f"wrong_dims_{w}x{h}"
                            valid = False
                            invalid_count += 1
                        # Mode check
                        elif mode != EXPECTED_MODE:
                            error = f"wrong_mode_{mode}"
                            valid = False
                            invalid_count += 1
                        else:
                            # Visual-quality heuristic from file size
                            kb = fsize / 1024
                            if kb < VERY_SMALL_KB:
                                vq_flag = "VERY_SMALL_FILE_possible_blank_or_shadow"
                            elif kb > VERY_LARGE_KB:
                                vq_flag = "VERY_LARGE_FILE_possible_noise_or_artifact"

                            # Collect pixel sample
                            if len(sample_pix) < 200:
                                arr = np.array(img)
                                sample_pix.append((
                                    int(arr.min()), int(arr.max()),
                                    float(arr.mean())
                                ))

                    except Exception as exc:
                        error = str(exc)[:120]
                        invalid_count += 1

                writer.writerow([fname, valid, w, h, mode, fsize, error, vq_flag])

    # Summary stats
    sizes = np.array(size_list)
    print(f"\n  PNG count:      {len(all_files):,}")
    print(f"  Invalid:        {invalid_count}")
    print(f"  File size — min: {sizes.min()/1024:.1f} KB  "
          f"max: {sizes.max()/1024:.1f} KB  "
          f"mean: {sizes.mean()/1024:.1f} KB  "
          f"median: {np.median(sizes)/1024:.1f} KB")
    print(f"  Modes seen:     {dict(mode_counts)}")
    print(f"  Dimension sets: {dict(dim_counts)}")
    if sample_pix:
        mins  = [r[0] for r in sample_pix]
        maxes = [r[1] for r in sample_pix]
        means = [r[2] for r in sample_pix]
        print(f"  Pixel range (sample of {len(sample_pix)}): "
              f"min={min(mins)}, max={max(maxes)}, mean~{sum(means)/len(means):.1f}")

    return len(all_files), invalid_count


# ── Step 3: Source distribution ────────────────────────────────────────────
def run_source_distribution(all_files: list[Path]):
    print("\n[Step 3] Building source-product distribution...")
    product_counts = defaultdict(int)
    for p in all_files:
        product_counts[extract_product(p.name)] += 1

    total = len(all_files)
    rows = []
    for prod, cnt in sorted(product_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total
        rows.append({"source_product": prod, "tile_count": cnt, "percentage_of_dataset": round(pct, 2)})
        flag = "  [DOMINANT >20%]" if pct > 20 else ""
        print(f"  {prod}  {cnt:5d} tiles  ({pct:5.1f}%){flag}")

    df = pd.DataFrame(rows)
    df.to_csv(DISTRIBUTION_CSV, index=False)
    print(f"\n  Source products: {len(product_counts)}")
    max_pct = max(r["percentage_of_dataset"] for r in rows)
    if max_pct > 20:
        print(f"  WARNING: One product contributes {max_pct:.1f}% of tiles — potential imbalance.")
    else:
        print(f"  Max single-product share: {max_pct:.1f}% — balanced dataset.")
    return product_counts


# ── Step 10: Duplicate detection ──────────────────────────────────────────
def run_duplicate_check(all_files: list[Path]):
    print(f"\n[Step 10] Computing MD5 hashes for {len(all_files):,} files...")
    hash_map = defaultdict(list)
    for p in tqdm(all_files, desc="Hashing", unit="file"):
        h = md5_file(p)
        hash_map[h].append(p.name)

    dup_groups = {h: fnames for h, fnames in hash_map.items() if len(fnames) > 1}
    total_dups = sum(len(v) - 1 for v in dup_groups.values())
    print(f"  Exact duplicate groups: {len(dup_groups)}")
    print(f"  Total duplicate files:  {total_dups}")

    rows = []
    for h, fnames in sorted(dup_groups.items()):
        for i, fname in enumerate(sorted(fnames)):
            rows.append({
                "md5": h,
                "filename": fname,
                "role": "original" if i == 0 else "duplicate",
                "group_size": len(fnames),
            })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["md5","filename","role","group_size"])
    df.to_csv(DUPLICATE_CSV, index=False)
    return total_dups


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Chandrayaan-2 OHRC Dataset QC — Steps 1, 2, 3, 9, 10")
    print("=" * 60)

    all_files = sorted(TILES_DIR.glob("*.png"))
    if not all_files:
        print(f"ERROR: No PNG files found under {TILES_DIR}")
        sys.exit(1)

    total, invalid = run_validation(all_files)
    product_counts  = run_source_distribution(all_files)
    total_dups      = run_duplicate_check(all_files)

    print("\n" + "=" * 60)
    print("QC Steps 1-3, 9, 10 complete.")
    print(f"  image_validation.csv    → {VALIDATION_CSV}")
    print(f"  source_tile_dist.csv    → {DISTRIBUTION_CSV}")
    print(f"  duplicate_report.csv    → {DUPLICATE_CSV}")
    print("=" * 60)

    return total, invalid, len(product_counts), total_dups


if __name__ == "__main__":
    main()
