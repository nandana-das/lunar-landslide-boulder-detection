# Detecting Lunar Rockfalls from Chandrayaan-2 OHRC Imagery via Domain-Adaptive Transfer Learning

### M.Tech AI & Data Science Research Project — Alliance University, Batch 2025–2027

---

## Overview

This repository implements a domain-adaptive transfer learning pipeline for rockfall detection on Chandrayaan-2 Orbiter High Resolution Camera (OHRC) imagery. The pipeline transfers knowledge from labeled NASA LROC NAC source domain images to unlabeled OHRC target domain images without requiring any OHRC annotations.

**Key results:**
- YOLO26n achieves mAP@0.5 = **0.640** after domain-adaptive fine-tuning on combined RMaM-2020 + Prieur lunar data
- **4,565 rockfall detections** on 31,769 OHRC tiles (YOLO26n, τ=0.20)
- Multi-reference histogram matching (20-tile averaged reference) improves OHRC detection count **3.1×** over single-reference baseline

---

## Paper

**Title:** Detecting Lunar Rockfalls from Chandrayaan-2 OHRC Imagery via Domain-Adaptive Transfer Learning

**Authors:** Nandana Narayan Das, Gowri Kannan

**Target venue:** IEEE Geoscience and Remote Sensing Letters (GRSL) / ICNLP 2027

---

## Method

Three-stage domain-adaptive transfer learning pipeline:

```
Stage 1: COCO pretrained YOLO26n/YOLOv8n/YOLOv5s
    ↓ Fine-tune on RMaM-2020 + Prieur lunar (4,068 images)
    ↓ Frozen backbone, LR=1e-3, 50 epochs

Stage 2: Multi-reference Histogram Matching
    ↓ Average 20 OHRC tiles → reference distribution
    ↓ Apply HM to all 4,068 training images + validation set
    ↓ Full fine-tune, LR=1e-4, cosine annealing, 50 epochs

Stage 3: OHRC Inference
    ↓ Run adapted model on 31,769 unlabeled OHRC tiles
    ↓ τ = 0.20, stream inference
```

---

## Results

### Source Domain Validation (Prieur lunar HM val set, 697 images)

| Model | Stage | mAP@0.5 | Precision | Recall |
|-------|-------|---------|-----------|--------|
| YOLO26n | Stage 1 | 0.549 | 0.623 | 0.511 |
| YOLOv8n | Stage 1 | 0.520 | 0.604 | 0.483 |
| YOLOv5s | Stage 1 | 0.600 | 0.656 | 0.548 |
| **YOLO26n** | **Stage 2 + Multi-HM** | **0.640** | **0.660** | 0.590 |
| YOLOv8n | Stage 2 + Multi-HM | 0.596 | 0.620 | 0.568 |
| YOLOv5s | Stage 2 + Multi-HM | 0.649 | 0.653 | **0.607** |

### OHRC Target Domain Inference (31,769 tiles, τ=0.20)

| Model | Pipeline | Detections | Tiles |
|-------|----------|-----------|-------|
| YOLOv8n | RMaM only, no HM | 54 | 52 |
| YOLOv8n | RMaM only, single-HM | 989 | 914 |
| YOLOv5s | RMaM only, single-HM | 1,628 | 1,310 |
| **YOLO26n** | **RMaM+Prieur, Multi-HM** | **4,565** | **1,193** |
| YOLOv5s | RMaM+Prieur, Multi-HM | 5,026 | 1,532 |
| YOLOv8n† | RMaM+Prieur, Multi-HM | 3,118† | 1,257† |

†YOLOv8n over-detects at τ=0.20 (41,941 detections); τ=0.50 used here.

---

## Dataset

### Source Domain (Labeled)
| Dataset | Source | Images | Format |
|---------|--------|--------|--------|
| RMaM-2020 (lunar only) | [EDMOND](https://edmond.mpdl.mpg.de) | 349 train, 17 test | CSV bbox → YOLO |
| Prieur et al. 2023 (lunar only) | [Zenodo 14250874](https://zenodo.org/records/14250874) | 3,719 train, 697 val, 262 test | Polygon → YOLO bbox |

### Target Domain (Unlabeled)
- **Chandrayaan-2 OHRC calibrated products** from [ISSDC PRADAN](https://pradan.issdc.gov.in)
- 12 products (rows 600–624 of OHRC catalog)
- 6 south pole products (lat ≈ −70°S, SI 78°–84°) + 6 equatorial (lat ≈ +60°N, SI 59°–68°)
- 31,769 usable 640×640 PNG tiles after dark tile filtering (mean pixel < 20)

---

## Repository Structure

```
lunar-landslide-boulder-detection/
├── scripts/
│   ├── convert_rmam_labels.py      # RMaM CSV → YOLO bbox format
│   ├── convert_prieur_labels.py    # Prieur polygon → YOLO bbox
│   ├── filter_moon_only.py         # Filter Moon subset from Prieur
│   ├── combine_datasets.py         # Merge RMaM + Prieur → combined/
│   ├── histogram_matching.py       # Multi-reference HM (RMaM images)
│   ├── hm_val.py                   # Apply HM to Prieur val set
│   ├── combined_hm_prep.py         # HM Prieur train + merge with RMaM HM
│   ├── tile_ohrc.py                # PDS4 → GeoTIFF → 640×640 tiles
│   ├── inspect_pds4.py             # PDS4 metadata reader
│   ├── generate_previews.py        # OHRC image preview generation
│   ├── train_stage1_all.py         # Stage 1 — all 3 models
│   ├── train_stage2.py             # Stage 2 — single model
│   ├── train_stage2_v2.py          # Stage 2 — all 3 models, HM val
│   ├── infer_ohrc.py               # OHRC tile inference + save detections
│   ├── check_confidence.py         # Confidence threshold analysis
│   └── run_pipeline.py             # Full pipeline orchestrator
├── data/rmam/
│   ├── dataset.yaml                # RMaM training config
│   ├── dataset_hm.yaml             # Single-ref HM config
│   └── dataset_hm_multi.yaml       # Multi-ref HM config
├── results/
│   ├── inference_results_summary.txt  # All detection counts + mAP results
│   ├── dataset_manifest.csv
│   ├── tile_statistics.csv
│   └── ohrc_dataset_metadata.csv
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
# 1. Convert RMaM labels
python scripts/convert_rmam_labels.py

# 2. Convert + filter Prieur dataset (Moon only)
python scripts/convert_prieur_labels.py
python scripts/filter_moon_only.py

# 3. Combine datasets
python scripts/combine_datasets.py

# 4. Apply multi-reference histogram matching
python scripts/histogram_matching.py
python scripts/hm_val.py
python scripts/combined_hm_prep.py

# 5. Train Stage 1 (all 3 models)
python scripts/train_stage1_all.py

# 6. Train Stage 2 (all 3 models, HM val)
python scripts/train_stage2_v2.py

# 7. Run OHRC inference
python scripts/infer_ohrc.py --conf 0.20

# 8. Confidence analysis
python scripts/check_confidence.py
```

---

## Data Access

**OHRC data** is not committed (large files). To reproduce:
1. Register at https://pradan.issdc.gov.in (free)
2. Browse and Download → OHRC → Calibrated
3. Navigate to page 7 (rows 600–624) — best solar incidence angles
4. Download bulk script → run to download products
5. Run `python scripts/tile_ohrc.py`

**RMaM-2020:** https://edmond.mpdl.mpg.de — search "RMaM" → download RMaM-2020.zip

**Prieur 2023:** https://zenodo.org/records/14250874 — download `bouldering_dataset_2024_YOLO_and_detectron2_format.zip`

---

## Hardware

Trained on NVIDIA RTX 3050 Laptop GPU (4GB VRAM), FP16 mixed precision, batch size 8.

---

## Citation

```bibtex
@article{das2026lunar,
  title={Detecting Lunar Rockfalls from Chandrayaan-2 OHRC Imagery
         via Domain-Adaptive Transfer Learning},
  author={Das, Nandana Narayan and Kannan, Gowri},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2026}
}
```

**Source datasets:**
- Bickel et al. (2021). RMaM-2020. *Frontiers in Remote Sensing*. doi:10.3389/frsen.2021.640034
- Prieur et al. (2023). Boulder dataset. Zenodo. doi:10.5281/zenodo.14250874
- OHRC data courtesy of ISRO via ISSDC PRADAN portal.
