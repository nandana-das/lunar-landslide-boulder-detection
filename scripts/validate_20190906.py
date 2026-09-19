"""
Standalone validation script for the 20190906 product.
Generates a preview PNG without importing inspect_pds4.py.
Run this from the project root:
  python scripts/validate_20190906.py
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

XML_PATH = (PROJECT_ROOT / "data/extracted/20190906/data/calibrated/20190906"
            / "ch2_ohr_ncp_20190906T2241285714_d_img_gds.xml")
IMG_PATH = XML_PATH.with_suffix(".img")
PREVIEWS_DIR = PROJECT_ROOT / "data" / "previews"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# Known from PDS4 XML inspection
LINES = 101075
SAMPLES = 12000
DTYPE = "uint8"
OFFSET = 0

# Subsampling for preview
ROW_STEP = 51     # every 51 rows -> ~1981 rows in preview
COL_STEP = 6      # every 6 cols  -> 2000 cols in preview

import numpy as np
from PIL import Image
from tqdm import tqdm

out_path = PREVIEWS_DIR / "ch2_ohr_ncp_20190906T2241285714_d_img_gds_preview.png"

if out_path.exists():
    print(f"Preview already exists: {out_path}")
    sys.exit(0)

print("Reading subsampled rows from .img file...")
row_bytes = SAMPLES * 1  # 1 byte per pixel (uint8)

selected_lines = range(0, LINES, ROW_STEP)
rows = []
with open(IMG_PATH, "rb") as f:
    for line_idx in tqdm(selected_lines, desc="Reading", unit="row"):
        f.seek(OFFSET + line_idx * row_bytes)
        raw = f.read(row_bytes)
        row = np.frombuffer(raw, dtype=np.uint8)[::COL_STEP]
        rows.append(row)

preview = np.stack(rows, axis=0)
print(f"Preview array: {preview.shape}  dtype={preview.dtype}")
print(f"Native min={preview.min()}  max={preview.max()}  mean={preview.mean():.2f}")

# Percentile stretch for display only
p2 = float(np.percentile(preview, 2))
p98 = float(np.percentile(preview, 98))
print(f"p2={p2:.1f}  p98={p98:.1f}")
stretched = np.clip(preview.astype(np.float32), p2, p98)
stretched = ((stretched - p2) / (p98 - p2) * 255).astype(np.uint8)

img = Image.fromarray(stretched, mode="L")
img.save(out_path, format="PNG", compress_level=6)
print(f"Saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")
