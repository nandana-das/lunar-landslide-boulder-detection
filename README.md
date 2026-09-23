# Detecting Lunar Rockfalls from Chandrayaan-2 OHRC Imagery via Domain-Adaptive Transfer Learning

### M.Tech AI & Data Science Research Project — Alliance University, Batch 2025–2027

---

## Overview

This repository implements an end-to-end domain-adaptive transfer learning pipeline for lunar rockfall and boulder detection on ultra-high-resolution Chandrayaan-2 Orbiter High Resolution Camera (OHRC) imagery (~0.25–0.32 m/pixel). The pipeline transfers knowledge from labeled NASA LROC NAC source domain images to completely unlabeled OHRC target domain images without requiring any manual OHRC annotations.

We systematically evaluate four detector architectures spanning both Convolutional Neural Networks (**YOLO26n**, **YOLOv5s**, **YOLOv8n**) and real-time Vision Transformers (**RT-DETR-L**).

**Key Findings & Results:**
- **Superior Held-Out Generalization:** On the held-out Prieur lunar test split (262 histogram-matched images), **YOLO26n** leads with **mAP@0.5 = 0.617** (mAP@0.5:0.95 = 0.236, F1 = 0.607), followed closely by **YOLOv5s** (**0.612**, F1 = 0.606), significantly outperforming **RT-DETR-L** (**0.474**, F1 = 0.515).
- **Source Domain Validation:** YOLO26n achieves **mAP@0.5 = 0.640** on the validation set (697 images) after domain adaptation, while YOLOv5s achieves **0.649**.
- **Target Domain Scale:** YOLO26n detects **4,565 rockfalls** across 31,769 OHRC tiles (1,193 positive tiles, 3.75% hit rate, $\tau=0.20$).
- **Multi-Reference Histogram Matching:** Constructing an empirical reference distribution by averaging 20 representative OHRC tiles delivers a **3.1× detection gain** over single-reference histogram matching.
- **Inductive Bias Discovery:** CNN architectures provide superior inductive bias for sparse planetary boulder detection, whereas global self-attention in Vision Transformers (RT-DETR-L) triggers severe texture hallucinations on unannotated regolith (80.2% tile hit rate).

---

## Paper

**Title:** Detecting Lunar Rockfalls from Chandrayaan-2 OHRC Imagery via Domain-Adaptive Transfer Learning  
**Authors:** Nandana Narayan Das, Gowri Kannan  
**Affiliation:** Department of Artificial Intelligence & Data Science, Alliance University, Bengaluru, India  
**Target Venue:** IEEE Geoscience and Remote Sensing Letters (GRSL) / ICNLP 2027  

---

## Method

The transfer learning framework operates across three distinct stages with a dedicated multi-reference radiometric adaptation step:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Source Domain Pre-training                                   │
│  - Architectures: YOLO26n, YOLOv5s, YOLOv8n, RT-DETR-L                 │
│  - Dataset: Combined RMaM-2020 + Prieur Moon subset (4,068 images)     │
│  - Setup: Frozen backbone, LR=1e-3, 50 epochs                          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Multi-Reference Radiometric Domain Adaptation                │
│  - Compute mean cumulative CDF from 20 diverse OHRC reference tiles   │
│  - Apply multi-reference histogram matching (HM) to all training,     │
│    validation (697 images), and held-out test (262 images) data        │
│  - Full fine-tuning: LR=1e-4, Cosine Annealing, 50 epochs              │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Target Domain OHRC Inference                                 │
│  - Input: 31,769 raw calibrated OHRC tiles (640×640 px)                │
│  - Streaming inference at operational confidence threshold τ = 0.20    │
│  - Multi-model cross-validation and regional divergence analysis      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Results

### 1. Held-Out Lunar Test Split Benchmark (Prieur Moon HM Test Set, 262 images)

Evaluated under identical protocol using `scripts/evaluate_test_split.py` and `scripts/evaluate_rtdetr_test.py`:

| Model | Architecture Type | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **YOLO26n** | CNN (Modern Anchor-Free) | **0.6166** | **0.2359** | **0.6367** | 0.5804 | **0.6072** |
| **YOLOv5s** | CNN (Anchor-Based Baseline) | 0.6122 | 0.2336 | 0.6303 | **0.5827** | 0.6056 |
| **YOLOv8n** | CNN (Anchor-Free Baseline) | 0.5663 | 0.2078 | 0.5962 | 0.5593 | 0.5772 |
| **RT-DETR-L** | Vision Transformer (Hybrid Encoder) | 0.4742 | 0.1765 | 0.5144 | 0.5153 | 0.5149 |

### 2. Source Domain Validation Split (Prieur Moon HM Val Set, 697 images)

| Model | Stage | mAP@0.5 | Precision | Recall | F1 Score |
|---|---|:---:|:---:|:---:|:---:|
| YOLO26n | Stage 1 (No HM) | 0.5490 | 0.6230 | 0.5110 | 0.5615 |
| YOLOv8n | Stage 1 (No HM) | 0.5200 | 0.6040 | 0.4830 | 0.5368 |
| YOLOv5s | Stage 1 (No HM) | 0.6000 | 0.6560 | 0.5480 | 0.5971 |
| **YOLO26n** | **Stage 2 + Multi-HM** | **0.6400** | **0.6600** | 0.5900 | 0.6230 |
| YOLOv5s | Stage 2 + Multi-HM | 0.6490 | 0.6530 | **0.6070** | **0.6292** |
| YOLOv8n | Stage 2 + Multi-HM | 0.5960 | 0.6200 | 0.5680 | 0.5929 |
| RT-DETR-L | Stage 2 + Multi-HM | 0.4843 | 0.5483 | 0.5371 | 0.5426 |

### 3. OHRC Target Domain Inference (31,769 tiles, $\tau=0.20$)

| Model | Domain Adaptation Pipeline | Total Detections | Positive Tiles | Tile Hit Rate | Mean Confidence | Recommended $\tau$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| YOLOv8n | RMaM only, No HM | 54 | 52 | 0.16% | 0.274 | — |
| YOLOv8n | RMaM only, Single-Ref HM | 989 | 914 | 2.88% | 0.288 | — |
| YOLOv5s | RMaM only, Single-Ref HM | 1,628 | 1,310 | 4.12% | 0.295 | — |
| **YOLO26n** | **RMaM + Prieur, Multi-Ref HM** | **4,565** | **1,193** | **3.75%** | **0.297** | **0.20** |
| YOLOv5s | RMaM + Prieur, Multi-Ref HM | 5,026 | 1,532 | 4.82% | 0.312 | 0.20 |
| YOLOv8n* | RMaM + Prieur, Multi-Ref HM | 41,941* | 6,621* | 20.84% | 0.313 | $\ge 0.35$ |
| RT-DETR-L† | RMaM + Prieur, Multi-Ref HM | 436,896† | 25,483† | 80.21% | 0.2959 | Experimental |

*\*YOLOv8n over-detects at $\tau=0.20$ due to low-confidence boundary noise; at calibrated threshold $\tau=0.50$, it yields 3,118 detections across 1,257 tiles.*  
*†RT-DETR-L exhibits severe vision transformer texture hallucination on unannotated planetary regolith, firing on fine grain shadow patterns.*

### 4. CNN vs. Transformer Inductive Bias in Planetary Remote Sensing

The 4-model evaluation yields an important theoretical finding for extraterrestrial object detection:
- **CNNs (YOLO26n, YOLOv5s):** Strong translation equivariance and local receptive fields act as an effective regularizer against regolith texture noise, confining detections to discrete physical boulders with cast shadows.
- **Vision Transformers (RT-DETR-L):** Global self-attention without strong local inductive priors causes the model to associate subtle regolith texture gradients with boulder morphology, producing massive false-positive cascades in out-of-domain target inference.

---

## Interactive Presentation Demo

A self-contained HTML/CSS/JS presentation demo is available in `demo/index.html`. It runs directly in any modern browser without web servers or build tools:

```bash
# Open directly in browser
start demo/index.html   # Windows
# or open demo/index.html in Chrome/Firefox/Edge
```

**Demo Features:**
1. **Interactive Pipeline Flowchart:** Step-through visualization of the 5 stages (Source Domain $\rightarrow$ Stage 1 $\rightarrow$ Multi-Ref HM $\rightarrow$ Stage 2 $\rightarrow$ OHRC Inference).
2. **Comprehensive Model Comparison:** 3-tab view comparing Source Validation (Table 2), Held-Out Lunar Test Split, and Target OHRC Inference (Table 3).
3. **High-Resolution Detection Gallery:** Curated OHRC detections (single isolated rockfall, boulder cluster, crater-rim context) with full-screen lightbox modal inspection.
4. **Regional Divergence Analysis:** South Pole polar terrain vs. Equatorial mare/highland distribution breakdown with embedded analytical charts.
5. **Operational Calibration Curves:** Interactive precision-recall and F1 curves for confidence threshold tuning.

---

## Repository Structure

```
lunar-landslide-boulder-detection/
├── demo/
│   ├── index.html                  # Self-contained presentation demo
│   └── images/                     # Annotated detection tiles & calibration plots
├── scripts/
│   ├── evaluate_test_split.py      # Held-out lunar test split evaluation (YOLO models)
│   ├── evaluate_rtdetr_test.py     # Held-out lunar test split evaluation (RT-DETR-L)
│   ├── train_rtdetr_pipeline.py    # RT-DETR Stage 1 & 2 training & eval pipeline
│   ├── analyze_regional_detections.py # Regional breakdown & comparison plot
│   ├── generate_ohrc_confidence_curves.py # Precision-recall & F1 threshold curves
│   ├── convert_rmam_labels.py      # RMaM CSV → YOLO bbox format
│   ├── convert_prieur_labels.py    # Prieur polygon → YOLO bbox
│   ├── filter_moon_only.py         # Filter Moon subset from Prieur dataset
│   ├── combine_datasets.py         # Merge RMaM + Prieur → combined/
│   ├── histogram_matching.py       # Multi-reference HM (RMaM training images)
│   ├── hm_val.py                   # Apply multi-ref HM to Prieur validation set
│   ├── hm_test.py                  # Apply multi-ref HM to Prieur held-out test set
│   ├── combined_hm_prep.py         # HM Prieur train + merge with RMaM HM
│   ├── tile_ohrc.py                # PDS4 → GeoTIFF → 640×640 tiles
│   ├── inspect_pds4.py             # PDS4 metadata reader & parser
│   ├── generate_previews.py        # OHRC image preview generator
│   ├── train_stage1_all.py         # Stage 1 — train all 3 YOLO models
│   ├── train_stage2.py             # Stage 2 — single model fine-tuning
│   ├── train_stage2_v2.py          # Stage 2 — all 3 models with HM validation
│   ├── infer_ohrc.py               # Streaming OHRC tile inference
│   ├── check_confidence.py         # Confidence threshold sweep & metrics
│   └── run_pipeline.py             # Master pipeline orchestrator
├── data/rmam/
│   ├── dataset.yaml                # Source RMaM training config
│   ├── dataset_hm.yaml             # Single-ref HM config
│   └── dataset_hm_multi.yaml       # Multi-ref HM config
├── results/
│   ├── test_metrics_stage2.csv     # Quantitative test split metrics (all 4 models)
│   ├── test_metrics_summary.md     # Markdown test split summary table
│   ├── table2_rtdetr_comparison.csv # 4-model validation benchmark
│   ├── table3_rtdetr_comparison.csv # 4-model OHRC inference comparison
│   ├── confidence_analysis/        # PR/F1 curves, regional plots & threshold tables
│   ├── ohrc_inference/             # Raw detection CSVs per model
│   ├── inference_results_summary.txt # Detection counts & summary statistics
│   ├── dataset_manifest.csv        # Dataset catalog & split counts
│   ├── tile_statistics.csv         # OHRC tile dimension and radiance stats
│   └── ohrc_dataset_metadata.csv   # Image product IDs, incidence angles, locations
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

## Reproduction & Workflow

```bash
# 1. Label Conversion & Preprocessing
python scripts/convert_rmam_labels.py
python scripts/convert_prieur_labels.py
python scripts/filter_moon_only.py
python scripts/combine_datasets.py

# 2. Multi-Reference Histogram Matching
python scripts/histogram_matching.py
python scripts/hm_val.py
python scripts/hm_test.py
python scripts/combined_hm_prep.py

# 3. Stage 1 Pre-training (Source Domain)
python scripts/train_stage1_all.py

# 4. Stage 2 Domain Adaptation
python scripts/train_stage2_v2.py
python scripts/train_rtdetr_pipeline.py

# 5. Held-Out Lunar Test Split Evaluation
python scripts/evaluate_test_split.py
python scripts/evaluate_rtdetr_test.py

# 6. OHRC Target Domain Streaming Inference
python scripts/infer_ohrc.py --conf 0.20

# 7. Analysis & Visualizations
python scripts/check_confidence.py
python scripts/generate_ohrc_confidence_curves.py
python scripts/analyze_regional_detections.py
```

---

## Data Access

**Chandrayaan-2 OHRC Imagery:**
1. Register at the ISRO ISSDC PRADAN portal: https://pradan.issdc.gov.in (free registration)
2. Browse and Download $\rightarrow$ Orbiter High Resolution Camera (OHRC) $\rightarrow$ Calibrated Products
3. Select observation tracks with low-to-medium solar incidence angles ($45^\circ \le i \le 75^\circ$) for prominent boulder shadow morphology
4. Run `python scripts/tile_ohrc.py` to convert PDS4 products to 640×640 inference tiles

**Source Domain Datasets:**
- **RMaM-2020:** https://edmond.mpdl.mpg.de — search "RMaM" $\rightarrow$ download `RMaM-2020.zip`
- **Prieur 2023:** https://zenodo.org/records/14250874 — download `bouldering_dataset_2024_YOLO_and_detectron2_format.zip`

---

## Hardware & Training Configuration

- **Compute:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM)
- **Precision:** Mixed Precision FP16
- **Batch Size:** 8 (YOLO models), 4 (RT-DETR-L)
- **Optimizer:** AdamW with Cosine Annealing learning rate schedule

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

**References:**
- Bickel, V. T., et al. (2021). "Automated Detection of Lunar Rockfalls Using Deep Learning." *Frontiers in Remote Sensing*, 2, 640034. doi:10.3389/frsen.2021.640034
- Prieur, N. C., et al. (2023). "Automated boulder detection on the Moon and other airless bodies." Zenodo. doi:10.5281/zenodo.14250874
- Chandrayaan-2 OHRC data courtesy of Indian Space Research Organisation (ISRO) via ISSDC PRADAN.
