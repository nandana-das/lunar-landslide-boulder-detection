"""
extract_ohrc.py
===============
Extracts Chandrayaan-2 OHRC calibrated (NCP) ZIP files into the
data/extracted/ directory.

Rules:
  - Only processes ch2_ohr_ncp_*.zip files (calibrated products)
  - Skips any ZIP if a sentinel file (.extracted) already exists
  - Never deletes, modifies, or overwrites source ZIPs
  - Preserves internal ZIP directory structure
  - Each product is extracted to: data/extracted/<YYYYMMDD>/

Usage:
    python scripts/extract_ohrc.py [--product <YYYYMMDD>]

Optional:
    --product   Extract only the product with the given date substring.
                Example: --product 20190906
"""

import argparse
import zipfile
from pathlib import Path
from tqdm import tqdm

from discover_ohrc_products import discover_calibrated_zips

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PRADAN_DIR = PROJECT_ROOT / "pradan.issdc.gov.in"
EXTRACT_BASE = PROJECT_ROOT / "data" / "extracted"
EXTRACT_BASE.mkdir(parents=True, exist_ok=True)

SENTINEL_FILENAME = ".extracted"


def get_extraction_dir(product: dict) -> Path:
    """
    Determine the extraction directory for a product.
    Uses the full product_id (not just date) to avoid collisions when
    multiple observations share the same date (e.g. 4 × 20240425 products).
    Products are placed in: data/extracted/<product_id>/
    """
    return EXTRACT_BASE / product["product_id"]


def already_extracted(product: dict) -> bool:
    """Check if a product's sentinel file exists, meaning it was already extracted."""
    sentinel = get_extraction_dir(product) / SENTINEL_FILENAME
    return sentinel.exists()


def extract_product(product: dict, dry_run: bool = False) -> bool:
    """
    Extract a single calibrated ZIP product.

    Returns True on success, False on failure.
    """
    zip_path = Path(product["path"])
    extract_dir = get_extraction_dir(product)

    if already_extracted(product):
        print(f"  [SKIP] {product['name']} — already extracted at {extract_dir}")
        return True

    print(f"  [EXTRACT] {product['name']} ({product['size_mb']} MB) → {extract_dir}")

    if dry_run:
        print("    [DRY RUN] Would extract here.")
        return True

    try:
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            # Filter: only extract data/ and adjacent PDS4 structure.
            # Skip browse images (they are not scientific input) if desired,
            # but we keep them for completeness since they are small.
            for member in tqdm(members, desc=f"  Extracting {product['date']}", leave=False, unit="file"):
                zf.extract(member, extract_dir)

        # Write sentinel
        sentinel = extract_dir / SENTINEL_FILENAME
        with open(sentinel, "w") as f:
            f.write(f"Extracted from: {zip_path}\n")
            f.write(f"Product: {product['product_id']}\n")
            f.write(f"Source size MB: {product['size_mb']}\n")

        print(f"    [OK] Extracted {len(members)} files.")
        return True

    except zipfile.BadZipFile as e:
        print(f"    [ERROR] Bad ZIP file: {e}")
        return False
    except OSError as e:
        print(f"    [ERROR] OS error during extraction: {e}")
        return False
    except Exception as e:
        print(f"    [ERROR] Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract Chandrayaan-2 OHRC calibrated ZIP products."
    )
    parser.add_argument(
        "--product",
        type=str,
        default=None,
        help="Extract only the product whose name contains this substring (e.g. 20190906)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without actually extracting.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Chandrayaan-2 OHRC — ZIP Extraction")
    print("=" * 70)

    calibrated = discover_calibrated_zips(PRADAN_DIR)

    if not calibrated:
        print("[ERROR] No calibrated products found.")
        return

    if args.product:
        calibrated = [p for p in calibrated if args.product in p["name"]]
        if not calibrated:
            print(f"[ERROR] No product found matching: {args.product}")
            return
        print(f"\nFiltered to {len(calibrated)} product(s) matching '{args.product}'.")

    print(f"\nProcessing {len(calibrated)} calibrated product(s)...\n")

    successes, skipped, failures = 0, 0, 0

    for product in calibrated:
        if already_extracted(product):
            skipped += 1
            print(f"  [SKIP] {product['name']} (already extracted)")
        else:
            ok = extract_product(product, dry_run=args.dry_run)
            if ok:
                successes += 1
            else:
                failures += 1

    print("\n" + "=" * 70)
    print(f"Extraction complete:")
    print(f"  Extracted:        {successes}")
    print(f"  Already present:  {skipped}")
    print(f"  Failed:           {failures}")
    print("=" * 70)


if __name__ == "__main__":
    main()
