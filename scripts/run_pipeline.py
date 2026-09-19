"""
run_pipeline.py
===============
Orchestrator for the full Chandrayaan-2 OHRC preprocessing pipeline.

Stages:
  1. discover  — find all calibrated NCP ZIPs
  2. extract   — unzip products into data/extracted/
  3. inspect   — read PDS4 labels, compute statistics
  4. preview   — generate per-product preview PNG
  5. tile      — generate 640×640 tiles, apply dark filtering
  6. manifest  — finalize manifest and QC contact sheets

Usage:
    # Full pipeline (all 12 calibrated products):
    python scripts/run_pipeline.py

    # Validate ONLY the already-extracted 20190906 product first:
    python scripts/run_pipeline.py --validate-only

    # Process a specific product by date substring:
    python scripts/run_pipeline.py --product 20190906

    # Skip extraction (products already extracted):
    python scripts/run_pipeline.py --skip-extract
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_script(script_name: str, extra_args: list[str] = None) -> bool:
    """Run a pipeline script as a subprocess and return True if successful."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    print(f"\n{'─'*70}")
    print(f"Running: {script_name} {' '.join(extra_args or [])}")
    print(f"{'─'*70}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(SCRIPTS_DIR))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  → {script_name}: {status} in {elapsed:.1f}s")
    return result.returncode == 0


def print_final_report(args):
    """Print a final pipeline summary."""
    print("\n" + "=" * 70)
    print("OHRC PREPROCESSING PIPELINE — FINAL REPORT")
    print("=" * 70)

    # Count products
    metadata_csv = RESULTS_DIR / "ohrc_dataset_metadata.csv"
    tile_stats_csv = RESULTS_DIR / "tile_statistics.csv"
    error_log = RESULTS_DIR / "preprocessing_errors.log"
    manifest_csv = RESULTS_DIR / "dataset_manifest.csv"

    if metadata_csv.exists():
        with open(metadata_csv, "r") as f:
            n_products = sum(1 for _ in f) - 1  # subtract header
        print(f"  Calibrated products processed: {n_products}")
    else:
        print("  Calibrated products processed: (metadata CSV not found)")

    if tile_stats_csv.exists():
        import csv
        with open(tile_stats_csv, "r") as f:
            rows = list(csv.DictReader(f))
        total_tiles = sum(int(r["total_tiles"]) for r in rows if r["total_tiles"].isdigit())
        usable_tiles = sum(int(r["usable_tiles"]) for r in rows if r["usable_tiles"].isdigit())
        discarded = sum(int(r["discarded_tiles"]) for r in rows if r["discarded_tiles"].isdigit())
        pct = round(100 * discarded / max(total_tiles, 1), 1)
        print(f"  Total tiles:                   {total_tiles:,}")
        print(f"  Usable tiles:                  {usable_tiles:,}")
        print(f"  Discarded (dark):              {discarded:,} ({pct}%)")
    else:
        print("  Tile statistics: (CSV not found)")

    print()
    print(f"  Metadata CSV:    {metadata_csv}")
    print(f"  Tile stats CSV:  {tile_stats_csv}")
    print(f"  Manifest CSV:    {manifest_csv}")
    print(f"  Previews:        {PROJECT_ROOT / 'data' / 'previews'}")
    print(f"  Usable tiles:    {PROJECT_ROOT / 'data' / 'tiles' / 'usable'}")
    print(f"  QC output:       {RESULTS_DIR / 'quality_control'}")

    if error_log.exists() and error_log.stat().st_size > 0:
        with open(error_log, "r") as f:
            errors = f.readlines()
        print(f"\n  ⚠  Errors logged: {len(errors)} — see {error_log}")
    else:
        print(f"\n  ✓  No errors logged.")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Orchestrate OHRC preprocessing pipeline.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run only on the already-extracted 20190906 product.")
    parser.add_argument("--product", type=str, default=None,
                        help="Process only the product containing this substring.")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip ZIP extraction (use if already extracted).")
    args = parser.parse_args()

    product_filter = []
    if args.validate_only:
        product_filter = ["--product", "20190906"]
        print("\n*** VALIDATION MODE: Processing 20190906 product only ***")
        print("*** Confirm results before running the full pipeline.   ***")
    elif args.product:
        product_filter = ["--product", args.product]

    print("\n" + "=" * 70)
    print("Chandrayaan-2 OHRC Preprocessing Pipeline")
    print("=" * 70)

    # Stage 1: Discover
    ok = run_script("discover_ohrc_products.py")
    if not ok:
        print("[FATAL] Discovery failed. Aborting.")
        return

    # Stage 2: Extract (skip if --skip-extract or --validate-only)
    if not args.skip_extract and not args.validate_only:
        ok = run_script("extract_ohrc.py", product_filter)
        if not ok:
            print("[WARN] Extraction had errors. Continuing with already-extracted products.")
    elif args.validate_only:
        print("\n[SKIP] Extraction skipped (validate-only mode — 20190906 already extracted).")
    else:
        print("\n[SKIP] Extraction skipped (--skip-extract).")

    # Stage 3: Inspect PDS4
    ok = run_script("inspect_pds4.py", product_filter)
    if not ok:
        print("[WARN] Inspection had errors.")

    # Stage 4: Preview
    ok = run_script("generate_previews.py", product_filter)
    if not ok:
        print("[WARN] Preview generation had errors.")

    # Stage 5: Tile
    ok = run_script("tile_ohrc.py", product_filter)
    if not ok:
        print("[WARN] Tiling had errors.")

    # Stage 6: Manifest + QC
    ok = run_script("create_dataset_manifest.py")
    if not ok:
        print("[WARN] Manifest/QC generation had errors.")

    print_final_report(args)


if __name__ == "__main__":
    main()
