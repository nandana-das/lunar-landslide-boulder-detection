"""
discover_ohrc_products.py
=========================
Recursively discovers all Chandrayaan-2 OHRC calibrated (NCP) ZIP files
under the PRADAN download directory and reports their inventory.

Usage:
    python scripts/discover_ohrc_products.py

Outputs:
    - Console table of discovered calibrated products
    - results/product_inventory.csv
"""

import os
import csv
from pathlib import Path

# ── Project root is the parent of scripts/ ────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PRADAN_DIR = PROJECT_ROOT / "pradan.issdc.gov.in"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def discover_calibrated_zips(pradan_dir: Path) -> list[dict]:
    """
    Recursively find all ch2_ohr_ncp_*.zip files.
    Returns a list of dicts with keys: name, path, size_mb, date
    """
    products = []
    for zip_path in sorted(pradan_dir.rglob("ch2_ohr_ncp_*.zip")):
        name = zip_path.name
        stem = zip_path.stem  # e.g. ch2_ohr_ncp_20190906T2241285714_d_img_gds
        # Extract date portion (after ncp_)
        parts = stem.split("_")
        date_token = parts[3] if len(parts) > 3 else "unknown"
        date = date_token[:8]  # YYYYMMDD

        products.append(
            {
                "name": name,
                "path": str(zip_path),
                "size_mb": round(zip_path.stat().st_size / (1024 * 1024), 1),
                "date": date,
                "product_id": stem,
            }
        )
    return products


def discover_raw_zips(pradan_dir: Path) -> list[dict]:
    """Find all ch2_ohr_nrp_*.zip files (raw, to be ignored)."""
    return [
        {"name": p.name, "path": str(p), "size_mb": round(p.stat().st_size / (1024 * 1024), 1)}
        for p in sorted(pradan_dir.rglob("ch2_ohr_nrp_*.zip"))
    ]


def save_inventory(products: list[dict], output_path: Path) -> None:
    """Save product list to CSV."""
    if not products:
        return
    fieldnames = list(products[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(products)


def main():
    print("=" * 70)
    print("Chandrayaan-2 OHRC Product Discovery")
    print("=" * 70)

    if not PRADAN_DIR.exists():
        print(f"[ERROR] PRADAN directory not found: {PRADAN_DIR}")
        return []

    calibrated = discover_calibrated_zips(PRADAN_DIR)
    raw = discover_raw_zips(PRADAN_DIR)

    print(f"\nFound {len(calibrated)} calibrated (NCP) products:")
    print(f"{'#':<4} {'Product ID':<55} {'Date':<10} {'Size MB'}")
    print("-" * 85)
    for i, p in enumerate(calibrated, 1):
        print(f"{i:<4} {p['product_id']:<55} {p['date']:<10} {p['size_mb']}")

    print(f"\nFound {len(raw)} raw (NRP) products (will be ignored):")
    for p in raw:
        print(f"  [SKIP] {p['name']}")

    # Save inventory
    inv_path = RESULTS_DIR / "product_inventory.csv"
    save_inventory(calibrated, inv_path)
    print(f"\nInventory saved to: {inv_path}")
    print(f"\nTotal calibrated products: {len(calibrated)}")
    print(f"Total raw products (skipped): {len(raw)}")

    return calibrated


if __name__ == "__main__":
    main()
