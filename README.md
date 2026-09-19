# Detecting Lunar Rockfalls from Chandrayaan-2 OHRC Imagery via Domain-Adaptive Transfer Learning

### M.Tech AI & Data Science Research Project — Alliance University, Batch 2025–2027

---

## Overview

This repository implements a domain-adaptive transfer learning pipeline for rockfall detection on Chandrayaan-2 Orbiter High Resolution Camera (OHRC) imagery — the first deep learning detection framework applied to OHRC data.

**Key result:** 989 rockfall detections on 31,769 unlabeled OHRC tiles using YOLOv8n with histogram matching — an 18× improvement over no domain adaptation.

---

## Paper

**Title:** Detecting Lunar Rockfalls from Chandrayaan-2 OHRC Imagery via Domain-Adaptive Transfer Learning

**Authors:** Nandana Narayan Das, Gowri Kannan

**Target venue:** IEEE GRSL / ICNLP 2027

---

## Method

Three-stage domain-adaptive transfer learning:

```
COCO pretrained YOLOv8n
    ↓ Stage 1: Fine-tune on RMaM-2020 lunar subset (frozen backbone, 50ep)
    ↓ Stage 2: Full fine-tune on histogram-matched RMaM-2020 (50ep)
    ↓ Stage 3: Inference on 31,769 unlabeled OHRC tiles
```

Histogram matching adapts LROC NAC pixel distribution to match OHRC target domain before training — no OHRC annotations required.

---

## Results

### Source Domain (RMaM-2020 lunar test set)

| Model | mAP@0.5 | Precision | Recall |
|-------|---------|-----------|--------|
| YOLOv8n Stage 1 (frozen) | 0.256 | 0.350 | 0.337 |
| YOLOv8n Stage 2 no HM | 0.342 | 0.419 | 0.349 |
| YOLOv8n Stage 2 + HM | 0.349 | 0.382 | **0.440** |
| YOLOv5s no HM | **0.452** | **0.615** | 0.366 |
| YOLOv5s + HM | 0.389 | 0.519 | 0.386 |

### Target Domain (Chandrayaan-2 OHRC — unlabeled)

| Model | Detections | Tiles hit |
|-------|-----------|-----------|
| YOLOv8n no HM | 54 | 52 |
| YOLOv8n + HM | 989 | 914 |
| YOLOv5s + HM | **1,628** | **1,310** |

---

## Dataset

**Source domain (labeled):** RMaM-2020 — 2,822 lunar rockfall images from LROC NAC. Download from [EDMOND](https://edmond.mpdl.mpg.de) — search "RMaM".

**Target domain (unlabeled):** Chandrayaan-2 OHRC calibrated products from [ISSDC PRADAN](https://pradan.issdc.gov.in). 12 products → 31,769 usable 640×640 PNG tiles.

---

## Repository Structure

```
lunar-landslide-boulder-detection/
├── scripts/
│   ├── convert_rmam_labels.py      # CSV → YOLO format conversion
│   ├── histogram_matching.py       # Pixel distribution alignment
│   ├── train_yolov8.py             # Stage 1 + Stage 2 training
│   ├── train_baseline_yolov5.py    # YOLOv5s baseline
│   ├── run_inference_ohrc.py       # OHRC tile inference
│   ├── create_dataset_yaml.py      # YAML config generation
│   ├── tile_ohrc.py                # OHRC tiling pipeline
│   ├── inspect_pds4.py             # PDS4 metadata reader
│   ├── generate_previews.py        # Preview generation
│   └── run_pipeline.py             # Full pipeline orchestrator
├── data/rmam/
│   ├── dataset.yaml                # RMaM training config
│   └── dataset_hm.yaml             # Histogram-matched config
├── results/
│   ├── dataset_manifest.csv
│   ├── tile_statistics.csv
│   └── geofilter/
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/nandana-das/lunar-landslide-boulder-detection.git
cd lunar-landslide-boulder-detection
pip install -r requirements.txt
```

---

## Quickstart

```bash
# 1. Convert RMaM labels to YOLO format
python scripts/convert_rmam_labels.py

# 2. Run histogram matching
python scripts/histogram_matching.py

# 3. Train YOLOv8n (Stage 1 + Stage 2)
python scripts/train_yolov8.py

# 4. Run inference on OHRC tiles
python scripts/run_inference_ohrc.py --conf 0.20
```

---

## Data Access

OHRC data is not committed (too large). To reproduce:

1. Register at https://pradan.issdc.gov.in (free)
2. Browse and Download → OHRC → Calibrated
3. Download rows 600–624 (best solar incidence angles, SI 75–85°)
4. Run `python scripts/run_pipeline.py`

RMaM-2020: https://edmond.mpdl.mpg.de → search "RMaM" → download RMaM-2020.zip

---

## Citation

```bibtex
@article{das2026lunar,
  title={Detecting Lunar Rockfalls from Chandrayaan-2 OHRC Imagery via Domain-Adaptive Transfer Learning},
  author={Das, Nandana Narayan and Kannan, Gowri},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2026}
}
```

OHRC data courtesy of ISRO via ISSDC PRADAN portal.
RMaM-2020 dataset: Bickel et al. (2021), Frontiers in Remote Sensing.
