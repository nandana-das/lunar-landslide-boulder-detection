# Detecting Lunar Landslides and Boulder Fields from Chandrayaan-2 OHRC Imagery
### M.Tech AI & Data Science Research Project

**Domain-Adaptive Transfer Learning for Lunar Geological Feature Detection**

---

## Project Overview

This repository contains the preprocessing pipeline for the M.Tech research project
*"Detecting Lunar Landslides and Boulder Fields from Chandrayaan-2 OHRC Imagery via
Domain-Adaptive Transfer Learning"*.

The pipeline converts raw Chandrayaan-2 Orbiter High Resolution Camera (OHRC)
calibrated PDS4 products into 640×640 pixel PNG tiles suitable for object detection
model training and annotation.

---

## Dataset Source

**Instrument**: Chandrayaan-2 Orbiter High Resolution Camera (OHRC)
- Passive electro-optical imaging camera, panchromatic band (500–800 nm)
- Ground Sampling Distance: **0.24–0.25 m/pixel** from ~100 km orbit
- Pushbroom sensor with TDI (Time Delay Integration)

**Data portal**: [ISRO ISSDC PRADAN](https://pradan.issdc.gov.in/)
- Downloads are located under `pradan.issdc.gov.in/` (not committed to Git)
- 12 calibrated (NCP) products processed; 13 raw (NRP) products ignored

---

## PDS4 Format

OHRC products are distributed as **PDS4** (Planetary Data System version 4) products.
Each product is a ZIP archive containing at minimum:

```
data/calibrated/<YYYYMMDD>/
    <product_id>.xml    ← PDS4 label (metadata)
    <product_id>.img    ← raw binary science image
geometry/
browse/
miscellaneous/
```

### Key PDS4 Label Fields (from 20190906 product)

| Field | Value |
|---|---|
| `data_type` | `UnsignedByte` |
| Lines (height) | 101,075 |
| Samples (width) | 12,000 |
| Offset | 0 bytes |
| Axis order | Last Index Fastest (row-major, C order) |
| File size | 1,212,900,000 bytes = 101,075 × 12,000 × 1 ✓ |
| Scaling factor | **None declared** |
| Value offset | **None declared** |
| Special constants | **None declared** |
| Processing level | Calibrated (radiometric LUT applied) |

> **Important**: The `.img` file is a **raw binary array**. It is NOT a JPEG, PNG,
> or TIFF. Do not open it with standard image viewers. Always use the `.xml` PDS4
> label to determine dimensions, data type, and byte offset.

---

## Directory Structure

```
lunar-landslide-boulder-detection/
│
├── pradan.issdc.gov.in/       ← ORIGINAL DOWNLOADS (never commit, never modify)
│   └── ch2/.../downloadData/
│       ├── ch2_ohr_ncp_*.zip  ← 12 calibrated products (PROCESS THESE)
│       └── ch2_ohr_nrp_*.zip  ← 13 raw products (IGNORE)
│
├── data/
│   ├── extracted/             ← Unzipped PDS4 products
│   │   └── <YYYYMMDD>/        ← One directory per observation date
│   ├── previews/              ← Full-image preview PNGs (subsampled)
│   └── tiles/
│       ├── all/               ← All 640×640 PNG tiles
│       └── usable/            ← Tiles passing dark/shadow filter
│
├── results/
│   ├── product_inventory.csv       ← Discovered calibrated ZIPs
│   ├── ohrc_dataset_metadata.csv   ← Per-product PDS4 metadata & statistics
│   ├── tile_statistics.csv         ← Per-product tiling statistics
│   ├── dataset_manifest.csv        ← Per-tile manifest (all tiles)
│   ├── preprocessing_errors.log    ← Failed products/errors
│   └── quality_control/
│       ├── contact_sheet_usable.png
│       ├── contact_sheet_dark.png
│       └── qc_summary.txt
│
├── scripts/
│   ├── discover_ohrc_products.py   ← Find calibrated ZIPs
│   ├── extract_ohrc.py             ← Unzip products
│   ├── inspect_pds4.py             ← Read PDS4 metadata + statistics
│   ├── generate_previews.py        ← Full-image preview generation
│   ├── tile_ohrc.py                ← 640×640 tiling + dark filtering
│   ├── create_dataset_manifest.py  ← Manifest + QC contact sheets
│   └── run_pipeline.py             ← Pipeline orchestrator
│
├── models/                    ← Future: trained model weights
├── notebooks/                 ← Future: EDA and visualization notebooks
├── preprocessing/             ← Future: advanced preprocessing modules
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd lunar-landslide-boulder-detection

# Install dependencies (Python 3.11+)
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pds4-tools` | ≥1.4 | PDS4 label parsing |
| `numpy` | ≥1.24 | Array operations |
| `Pillow` | ≥10.0 | PNG I/O |
| `opencv-python` | ≥4.8 | Image processing |
| `matplotlib` | ≥3.7 | Visualization |
| `tqdm` | ≥4.65 | Progress bars |

---

## How to Run the Preprocessing Pipeline

### Step 1 — Validate on a single product first (recommended)

Before processing all 12 products, validate the pipeline on the already-extracted
20190906 product:

```bash
python scripts/run_pipeline.py --validate-only
```

This runs: discover → inspect → preview → tile → QC for the 20190906 product only.
Examine `results/` and `data/previews/` before proceeding.

### Step 2 — Full pipeline (all 12 calibrated products)

```bash
python scripts/run_pipeline.py
```

This runs all 6 stages. Extraction of ZIPs is included.

### Step 3 — Skip extraction if already extracted

```bash
python scripts/run_pipeline.py --skip-extract
```

### Individual script usage

```bash
# Discover all calibrated ZIPs
python scripts/discover_ohrc_products.py

# Extract a single product
python scripts/extract_ohrc.py --product 20190906

# Inspect PDS4 metadata for one product
python scripts/inspect_pds4.py --product 20190906

# Generate preview for one product
python scripts/generate_previews.py --product 20190906

# Tile one product
python scripts/tile_ohrc.py --product 20190906

# Build manifest + QC contact sheets
python scripts/create_dataset_manifest.py
```

---

## Calibrated Pixel Representation

The OHRC calibrated (NCP) products store pixel values as **`UnsignedByte` (uint8, 0–255)**.

- Calibration applies a radiometric Look-Up Table (LUT) to convert raw DN to
  calibrated DN. The output is still stored as integer 0–255.
- **No `scaling_factor` or `value_offset` is declared** in the PDS4 XML labels
  of the products examined. Values are therefore native uint8 calibrated counts.
- The `.img` file is a pure binary array with no embedded header. Byte offset = 0.

---

## Normalization

**Two separate normalization contexts are used, and they are strictly kept separate:**

### 1. Dark tile filtering (no normalization)
- Applied to **native calibrated uint8 pixel values** (0–255)
- Threshold: `tile_mean < 20` → tile is discarded
- Rationale: since data is already uint8 with no declared scaling, the threshold
  is physically meaningful in native units

### 2. PNG conversion (normalization for display/ML)
- Applied to each tile separately (per-tile percentile stretch)
- Method: percentile stretch p2–p98 → rescale to 0–255 → save as uint8 PNG
- Purpose: maximize local contrast for human annotation and ML training
- Does **not** affect the filtering decision
- The original `.img` files are never written to

> **Reproducibility note**: Per-tile normalization means each tile is independently
> stretched. A globally consistent normalization (e.g., using dataset-wide min/max)
> could alternatively be used for quantitative analysis, but per-tile stretch is
> preferred for annotation visibility. This choice is documented here.

---

## Tiling Strategy

| Parameter | Value |
|-----------|-------|
| Tile size | 640 × 640 pixels |
| Overlap | None (non-overlapping) |
| Edge handling | Zero-padding to 640 × 640 |
| Edge tiles in manifest | Flagged as `is_edge_padded=True` |

**Why non-overlapping?**
- Reduces total dataset size and avoids annotation duplication
- Simplifies manifest tracking
- Overlapping tiling can be added later for augmentation if needed

**Why zero-padding for edge tiles?**
- Prevents silent aspect-ratio distortion
- Preserves the spatial meaning of each tile's coordinates
- Edge tiles are retained (not discarded) but flagged so they can be excluded
  if annotation requires fully valid pixels

---

## Dark Tile Filtering

Tiles with `mean pixel value < 20` (in native uint8 calibrated space) are classified
as dark/shadow tiles and moved to `data/tiles/all/` but not copied to `data/tiles/usable/`.

**Rationale for threshold = 20 in native uint8 space:**
- OHRC panchromatic calibrated values represent relative surface reflectance
- Regions with mean < 20/255 ≈ 7.8% of maximum possible value are predominantly
  unlit shadow regions carrying no useful surface texture for detection
- The threshold was chosen from domain knowledge and should be reviewed after
  inspecting the actual pixel-value histogram per product

---

## Output File Descriptions

### `results/ohrc_dataset_metadata.csv`
One row per product. Columns include: product_id, date, xml_path, img_path,
lines, samples, data_type, pix_min, pix_max, pix_mean, scaling_factor, etc.

### `results/tile_statistics.csv`
One row per product. Summarises: total_tiles, usable_tiles, discarded_tiles,
pct_discarded, image min/max/mean.

### `results/dataset_manifest.csv`
One row per tile. Columns: source_product, source_date, source_xml, source_img,
tile_filename, tile_x, tile_y, tile_width, tile_height, mean_pixel_value,
min_pixel_value, max_pixel_value, is_edge_padded, status.

### `results/quality_control/`
- `contact_sheet_usable.png` — random sample of usable tiles
- `contact_sheet_dark.png` — random sample of discarded dark tiles
- `qc_summary.txt` — text summary of filtering decisions and counts

---

## Limitations

1. **Approximate median**: The per-product median reported in `ohrc_dataset_metadata.csv`
   is computed as the median of per-strip means (strip = 640 lines). This is an
   approximation; computing the true global median would require sorting all ~1.2B
   values and is memory-prohibitive on 8 GB RAM.

2. **Edge tile zero-padding**: Edge tiles contain padded zeros along one or two borders.
   These zero regions have very low pixel values. If an edge tile's zero-padded area
   causes its mean to fall below the dark threshold, it may be incorrectly discarded.
   Edge tiles are flagged in the manifest for post-hoc review.

3. **Per-tile PNG normalization**: Different tiles may have different brightness scales
   after normalization. For quantitative analysis requiring consistent pixel intensity
   scale, use the raw binary `.img` data and the PDS4 label.

4. **No georeferencing**: PNG tiles do not embed geographic coordinates. The manifest
   records tile pixel coordinates (`tile_x`, `tile_y`) relative to the source image
   origin, which can be combined with the PDS4 geometry information for spatial
   referencing.

5. **Disk space**: Processing all 12 calibrated products requires ~15–20 GB for
   extracted `.img` files plus ~5–10 GB for generated PNG tiles.

---

## GitHub / Version Control Policy

The following are excluded from version control (see `.gitignore`):
- All `.zip` files (PRADAN downloads)
- All `.img` files (raw binary PDS4 science data)
- `pradan.issdc.gov.in/` directory
- `data/extracted/`, `data/tiles/`, `data/previews/`
- Authentication credentials, cookies, session files

Results CSVs and scripts ARE committed. Large generated binary files are NOT.

---

## Future Work

- [ ] Manual annotation of landslide and boulder features (Roboflow)
- [ ] YOLOv5 / YOLOv8 baseline object detection training
- [ ] Faster R-CNN fine-tuning on annotated OHRC tiles
- [ ] Domain-adaptive transfer learning (source: terrestrial; target: lunar)
- [ ] Evaluation metrics: mAP, precision/recall per class
- [ ] Geographic visualization of detections on OHRC footprints

---

## Citation

If you use this pipeline or dataset in your research, please cite:

> Nandana Das. "Detecting Lunar Landslides and Boulder Fields from Chandrayaan-2
> OHRC Imagery via Domain-Adaptive Transfer Learning." M.Tech Thesis, [Institution],
> 2026.

OHRC data courtesy of Indian Space Research Organisation (ISRO) via the ISSDC PRADAN portal.
