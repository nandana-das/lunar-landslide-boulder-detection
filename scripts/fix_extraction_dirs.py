"""
fix_extraction_dirs.py
======================
One-time migration script to rename date-based extraction directories
to product_id-based ones, fixing the date-collision bug.

Reads each existing .extracted sentinel to determine which product_id
was extracted there, then renames the directory accordingly.

Run once, then re-run:
  python scripts/extract_ohrc.py    (to get the remaining 7 products)
  python scripts/run_pipeline.py --skip-extract  (to process all 12)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXTRACT_BASE = PROJECT_ROOT / "data" / "extracted"
SENTINEL = ".extracted"


def parse_product_id_from_sentinel(sentinel_path: Path) -> str | None:
    """Read the .extracted sentinel and extract the product_id line."""
    with open(sentinel_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Product:"):
                return line.split(":", 1)[1].strip()
    return None


def main():
    print("=" * 60)
    print("Fixing extraction directory names (date → product_id)")
    print("=" * 60)

    # Only rename date-style directories (8 chars, all digits: YYYYMMDD)
    renamed, skipped = 0, 0

    for subdir in sorted(EXTRACT_BASE.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        # Check if this looks like a YYYYMMDD directory (old format)
        if not (len(name) == 8 and name.isdigit()):
            print(f"  [SKIP] {name} — already product_id format")
            skipped += 1
            continue

        sentinel = subdir / SENTINEL
        if not sentinel.exists():
            print(f"  [WARN] {name} — no sentinel file, skipping")
            continue

        product_id = parse_product_id_from_sentinel(sentinel)
        if not product_id:
            print(f"  [WARN] {name} — could not parse product_id from sentinel")
            continue

        new_dir = EXTRACT_BASE / product_id
        if new_dir.exists():
            print(f"  [SKIP] {name} → {product_id} — target already exists")
            skipped += 1
            continue

        print(f"  [RENAME] {name} → {product_id}")
        subdir.rename(new_dir)
        renamed += 1

    print(f"\nRenamed: {renamed}  |  Skipped: {skipped}")
    print("\nNow run:")
    print("  python scripts/extract_ohrc.py        # extract remaining 7 products")
    print("  python scripts/run_pipeline.py --skip-extract  # process all 12")


if __name__ == "__main__":
    main()
