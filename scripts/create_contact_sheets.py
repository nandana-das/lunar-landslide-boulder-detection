"""
create_contact_sheets.py  — Step 6
====================================
Creates 4 contact-sheet PNGs from the 200 sampled tiles.
Each sheet holds 50 tiles in a 5-column × 10-row grid.
Tiles are shown at 128×128 thumbnail size with a filename
label below each cell.

Outputs:
  results/quality_control/contact_sheets/sample_sheet_001.png
  results/quality_control/contact_sheets/sample_sheet_002.png
  results/quality_control/contact_sheets/sample_sheet_003.png
  results/quality_control/contact_sheets/sample_sheet_004.png

The original tiles are NOT modified.

Usage:
  python scripts/create_contact_sheets.py
"""
import sys
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
SAMPLE_DIR     = PROJECT_ROOT / "data" / "annotation_sample"
MANIFEST_CSV   = PROJECT_ROOT / "results" / "quality_control" / "sampling_manifest.csv"
SHEET_DIR      = PROJECT_ROOT / "results" / "quality_control" / "contact_sheets"
SHEET_DIR.mkdir(parents=True, exist_ok=True)

THUMB_SIZE     = 128          # px — thumbnail side length
LABEL_HEIGHT   = 20           # px — space below each thumb for filename
N_COLS         = 5            # thumbnails per row
N_PER_SHEET    = 50           # tiles per contact sheet (= 5 × 10)
PADDING        = 4            # px — gap between cells
BG_COLOR       = (30, 30, 30) # dark background
LABEL_COLOR    = (200, 200, 200)
BORDER_COLOR   = (80, 80, 80)

CELL_W = THUMB_SIZE + PADDING * 2
CELL_H = THUMB_SIZE + LABEL_HEIGHT + PADDING * 2


def try_load_font(size: int = 9):
    """Try to load a small monospace font; fall back to default."""
    for name in ["cour.ttf", "Courier New Bold.ttf", "DejaVuSansMono.ttf",
                 "consola.ttf", "LiberationMono-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_sheet(tiles: list[tuple[int, str]], sheet_path: Path, font):
    """Render one contact sheet from a list of (sample_id, filename) pairs."""
    n = len(tiles)
    n_rows = math.ceil(n / N_COLS)
    sheet_w = N_COLS * CELL_W + PADDING
    sheet_h = n_rows * CELL_H + PADDING + 30  # +30 for top banner
    canvas = Image.new("RGB", (sheet_w, sheet_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    # Banner
    banner = sheet_path.stem
    draw.text((PADDING, 6), f"OHRC Tile Sample — {banner}", fill=(220, 180, 80), font=font)

    for idx, (sid, fname) in enumerate(tiles):
        row = idx // N_COLS
        col = idx % N_COLS
        x0 = col * CELL_W + PADDING
        y0 = row * CELL_H + PADDING + 30

        img_path = SAMPLE_DIR / fname
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                thumb = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            except Exception:
                thumb = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (80, 0, 0))
        else:
            thumb = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (50, 50, 50))

        # Draw border
        draw.rectangle([x0 - 1, y0 - 1, x0 + THUMB_SIZE, y0 + THUMB_SIZE], outline=BORDER_COLOR)
        canvas.paste(thumb, (x0, y0))

        # Sample ID badge
        draw.rectangle([x0, y0, x0 + 24, y0 + 12], fill=(0, 80, 160))
        draw.text((x0 + 2, y0 + 1), f"#{sid:03d}", fill="white", font=font)

        # Filename label (truncated to fit)
        short = fname
        # Extract just the tile coordinates
        try:
            short = fname.split("_x")[1].replace("_y", ",").replace(".png", "")
            short = f"x,y={short}"
        except Exception:
            short = fname[-20:]
        draw.text((x0, y0 + THUMB_SIZE + PADDING), short, fill=LABEL_COLOR, font=font)

    canvas.save(sheet_path, format="PNG", optimize=True)
    return sheet_path


def main():
    print("=" * 60)
    print("Creating Contact Sheets — Step 6")
    print("=" * 60)

    if not MANIFEST_CSV.exists():
        print(f"ERROR: sampling_manifest.csv not found at {MANIFEST_CSV}")
        print("Run scripts/sample_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_CSV)
    tiles = list(zip(df["sample_id"].tolist(), df["tile_filename"].tolist()))

    font = try_load_font(9)

    n_sheets = math.ceil(len(tiles) / N_PER_SHEET)
    print(f"  Tiles: {len(tiles)}  |  Sheets: {n_sheets}  |  {N_PER_SHEET} tiles/sheet")

    for i in range(n_sheets):
        batch = tiles[i * N_PER_SHEET : (i + 1) * N_PER_SHEET]
        sheet_path = SHEET_DIR / f"sample_sheet_{i+1:03d}.png"
        make_sheet(batch, sheet_path, font)
        kb = sheet_path.stat().st_size // 1024
        print(f"  Sheet {i+1}: {sheet_path.name}  ({kb} KB)")

    print(f"\n  Contact sheets → {SHEET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
