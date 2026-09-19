"""
create_dataset_manifest.py
==========================
Creates the final dataset_manifest.csv from tile statistics and
generates quality control contact sheets showing examples of:
  - Usable lunar surface tiles
  - Discarded dark/shadow tiles

Usage:
    python scripts/create_dataset_manifest.py

Outputs:
    results/dataset_manifest.csv   (if not already created by tile_ohrc.py)
    results/quality_control/contact_sheet_usable.png
    results/quality_control/contact_sheet_dark.png
    results/quality_control/qc_summary.txt
"""

import sys
import csv
import random
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TILES_ALL_DIR = PROJECT_ROOT / "data" / "tiles" / "all"
TILES_USABLE_DIR = PROJECT_ROOT / "data" / "tiles" / "usable"
QC_DIR = RESULTS_DIR / "quality_control"
QC_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = RESULTS_DIR / "dataset_manifest.csv"
TILE_STATS_CSV = RESULTS_DIR / "tile_statistics.csv"

# Contact sheet parameters
CONTACT_COLS = 6       # tiles per row in contact sheet
CONTACT_ROWS = 4       # rows
THUMB_SIZE = 128       # thumbnail px (resized from 640 for contact sheet)
FONT_SIZE = 11


def load_manifest() -> list[dict]:
    if not MANIFEST_CSV.exists():
        print(f"[WARN] {MANIFEST_CSV} not found. Run tile_ohrc.py first.")
        return []
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_contact_sheet(
    tile_paths: list[Path],
    title: str,
    output_path: Path,
    n_cols: int = CONTACT_COLS,
    n_rows: int = CONTACT_ROWS,
    thumb_size: int = THUMB_SIZE,
) -> None:
    """
    Build a contact sheet showing up to n_cols × n_rows tile thumbnails.
    Tiles are randomly sampled from the provided list.
    """
    n_tiles = n_cols * n_rows
    if len(tile_paths) > n_tiles:
        sampled = random.sample(tile_paths, n_tiles)
    else:
        sampled = tile_paths

    # Padding
    PAD = 4
    HEADER_H = 30
    cell_w = thumb_size + PAD
    cell_h = thumb_size + PAD

    sheet_w = n_cols * cell_w + PAD
    sheet_h = n_rows * cell_h + PAD + HEADER_H

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    # Title
    draw.text((PAD, 5), title, fill=(220, 220, 220))

    for idx, tile_path in enumerate(sampled):
        col = idx % n_cols
        row = idx // n_cols
        x = PAD + col * cell_w
        y = HEADER_H + PAD + row * cell_h

        try:
            tile_img = Image.open(tile_path).convert("RGB")
            tile_img = tile_img.resize((thumb_size, thumb_size), Image.LANCZOS)
            sheet.paste(tile_img, (x, y))
        except Exception as e:
            # Draw placeholder
            draw.rectangle([x, y, x + thumb_size, y + thumb_size], fill=(40, 40, 40))
            draw.text((x + 2, y + 2), "ERR", fill=(200, 50, 50))

    sheet.save(output_path, format="PNG")
    print(f"  Contact sheet saved: {output_path.name} ({len(sampled)} tiles shown)")


def main():
    print("=" * 70)
    print("Chandrayaan-2 OHRC — Quality Control & Manifest")
    print("=" * 70)

    rows = load_manifest()

    if not rows:
        # Try to build manifest from tile directories
        print("[INFO] Building manifest from tile directories...")
        all_tiles = sorted(TILES_ALL_DIR.glob("*.png"))
        usable_tiles = {p.name for p in TILES_USABLE_DIR.glob("*.png")}

        rows = []
        for tp in all_tiles:
            name = tp.name
            status = "usable" if name in usable_tiles else "dark_discarded"
            rows.append({
                "source_product": name.rsplit("_x", 1)[0],
                "source_date": "",
                "source_xml": "",
                "source_img": "",
                "tile_filename": name,
                "tile_x": "",
                "tile_y": "",
                "tile_width": 640,
                "tile_height": 640,
                "mean_pixel_value": "",
                "min_pixel_value": "",
                "max_pixel_value": "",
                "is_edge_padded": "",
                "status": status,
            })

        if rows:
            with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Manifest written: {MANIFEST_CSV}")

    if not rows:
        print("[ERROR] No tile data available. Run tile_ohrc.py first.")
        return

    usable_rows = [r for r in rows if r["status"] == "usable"]
    dark_rows = [r for r in rows if r["status"] == "dark_discarded"]

    print(f"\nManifest summary:")
    print(f"  Total tiles:   {len(rows):,}")
    print(f"  Usable:        {len(usable_rows):,}")
    print(f"  Dark/shadow:   {len(dark_rows):,}")
    pct = round(100 * len(dark_rows) / max(len(rows), 1), 1)
    print(f"  % discarded:   {pct}%")

    # Build contact sheets
    print("\n[QC] Building contact sheets...")

    usable_paths = [TILES_USABLE_DIR / r["tile_filename"] for r in usable_rows
                    if (TILES_USABLE_DIR / r["tile_filename"]).exists()]
    dark_paths = [TILES_ALL_DIR / r["tile_filename"] for r in dark_rows
                  if (TILES_ALL_DIR / r["tile_filename"]).exists()]

    if usable_paths:
        build_contact_sheet(
            usable_paths,
            title=f"USABLE TILES — mean >= 20 (native uint8) | {len(usable_rows):,} total",
            output_path=QC_DIR / "contact_sheet_usable.png",
        )
    else:
        print("  [WARN] No usable tiles found for contact sheet.")

    if dark_paths:
        build_contact_sheet(
            dark_paths,
            title=f"DARK/SHADOW TILES — mean < 20 (native uint8) | {len(dark_rows):,} total",
            output_path=QC_DIR / "contact_sheet_dark.png",
        )
    else:
        print("  [WARN] No dark tiles found for contact sheet.")

    # QC summary text
    summary_path = QC_DIR / "qc_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Chandrayaan-2 OHRC Preprocessing — Quality Control Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total tiles in manifest:  {len(rows):,}\n")
        f.write(f"Usable tiles:             {len(usable_rows):,}\n")
        f.write(f"Dark/shadow discarded:    {len(dark_rows):,}\n")
        f.write(f"% discarded:              {pct}%\n\n")
        f.write("Dark tile filtering decision:\n")
        f.write("  Threshold:  mean < 20\n")
        f.write("  Domain:     native calibrated uint8 pixel values (0-255)\n")
        f.write("  Rationale:  PDS4 data type is UnsignedByte; no scaling\n")
        f.write("              factor or value_offset declared in XML labels.\n")
        f.write("              Filtering before normalization keeps the criterion\n")
        f.write("              physically reproducible.\n\n")
        f.write("Tiling strategy:\n")
        f.write("  Tile size:    640 × 640 px\n")
        f.write("  Overlap:      None (non-overlapping)\n")
        f.write("  Edge tiles:   Zero-padded to 640 × 640 (flagged in manifest)\n")
        f.write("  PNG norm:     Per-tile percentile stretch p2-p98 (display only)\n")
    print(f"  QC summary saved: {summary_path.name}")

    print("\n" + "=" * 70)
    print(f"QC outputs saved to: {QC_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
