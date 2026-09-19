"""
download_prieur.py  — Step 1b
==============================
Downloads the Prieur et al. (2024) boulder detection dataset from Zenodo.

Dataset: "Boulder populations around 82 fresh simple impact craters on the
Moon and 15 fresh simple impact craters on Mars"
Zenodo record: 14253940
DOI: 10.5281/zenodo.14253940

Content:
  - coldspots/           Lunar cold spot craters (boulder populations)
  - fresh_craters_not_coldspots/   Fresh lunar simple craters
  - mars/                Martian craters (EXCLUDED — we use lunar only)

Each crater sub-folder contains:
  raster/   *.tif         LROC NAC GeoTIFF image tiles
  shp/      *-YOLO-predictions.shp   YOLOv8-predicted boulder bounding boxes

NOTE: Total download is potentially large (5–15 GB).
      The script downloads ONLY the lunar subsets (coldspots + fresh_craters_not_coldspots).
      It skips mars/ entirely.
      Resume-capable: already-downloaded files are skipped.

Usage:
  python scripts/download_prieur.py [--dry-run] [--max-craters N]

  --dry-run       Print file list without downloading
  --max-craters N Download only the first N craters (for testing)
"""
import sys
import json
import argparse
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
OUT_DIR        = PROJECT_ROOT / "data" / "source_domain" / "prieur2023"
ZENODO_RECORD  = "14253940"
ZENODO_API     = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
CHUNK_SIZE     = 1 << 20   # 1 MB chunks

# Folders to download (case-sensitive match against Zenodo file key prefixes)
INCLUDE_PREFIXES = ["coldspots/", "fresh_craters_not_coldspots/"]
SKIP_PREFIXES    = ["mars/"]


def get_record_files() -> list[dict]:
    """Fetch file list from Zenodo API."""
    print(f"Fetching Zenodo record {ZENODO_RECORD}...")
    r = requests.get(ZENODO_API, timeout=30)
    r.raise_for_status()
    data = r.json()
    files = data.get("files", [])
    print(f"  Total files in record: {len(files)}")
    return files


def filter_files(files: list[dict]) -> list[dict]:
    """Keep only lunar subset files."""
    kept = []
    for f in files:
        key = f["key"]
        # Skip mars
        if any(key.startswith(p) for p in SKIP_PREFIXES):
            continue
        # Only keep if under a lunar prefix (or if no filtering needed)
        if INCLUDE_PREFIXES:
            if not any(key.startswith(p) for p in INCLUDE_PREFIXES):
                continue
        kept.append(f)
    return kept


def download_file(url: str, dest: Path, expected_md5: str | None = None):
    """Download a single file with resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Resume: check existing size
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    r = requests.get(url, headers=headers, stream=True, timeout=60)

    if r.status_code == 416:
        # File fully downloaded
        return
    r.raise_for_status()

    total = int(r.headers.get("Content-Length", 0)) + existing
    mode = "ab" if existing > 0 else "wb"

    with open(dest, mode) as f, tqdm(
        total=total, initial=existing, unit="B", unit_scale=True,
        desc=dest.name, leave=False
    ) as pbar:
        for chunk in r.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-craters", type=int, default=None,
                        help="Download only first N crater subdirectories")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = get_record_files()
    files = filter_files(files)

    print(f"\n  Lunar files to download: {len(files)}")
    total_bytes = sum(f.get("size", 0) for f in files)
    print(f"  Total size: {total_bytes / 1e9:.2f} GB")

    if args.dry_run:
        print("\n  [DRY RUN] Files that would be downloaded:")
        for f in files[:20]:
            sz = f.get("size", 0) / 1e6
            print(f"    {f['key']}  ({sz:.1f} MB)")
        if len(files) > 20:
            print(f"    ... and {len(files)-20} more")
        return

    # Limit by crater if requested
    if args.max_craters:
        # Group by second path component (crater name)
        seen = set()
        limited = []
        for f in files:
            parts = f["key"].split("/")
            crater = parts[1] if len(parts) > 1 else ""
            if crater not in seen:
                seen.add(crater)
            if len(seen) <= args.max_craters or crater in list(seen)[:args.max_craters]:
                limited.append(f)
        files = limited
        print(f"  Limited to first {args.max_craters} craters: {len(files)} files")

    print(f"\nDownloading to: {OUT_DIR}")
    print("Already-complete files will be skipped.\n")

    for f in files:
        key  = f["key"]
        url  = f["links"]["self"]
        size = f.get("size", 0)
        dest = OUT_DIR / key

        if dest.exists() and dest.stat().st_size == size:
            continue  # already complete

        print(f"  {key}  ({size/1e6:.1f} MB)")
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"  ERROR downloading {key}: {e}")
            continue

    print(f"\nDownload complete. Files at: {OUT_DIR}")
    print("\nNext step: python scripts/prepare_rmam.py")
    print("      then: python scripts/prepare_prieur.py")


if __name__ == "__main__":
    main()
