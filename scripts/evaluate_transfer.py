"""
evaluate_transfer.py  — Step 5
================================
Evaluates the domain adaptation quality and detection transfer.

Produces:
  1. mAP on held-out LROC test set (quantitative, source-domain)
  2. Confidence distribution comparison: LROC val vs OHRC tiles
  3. Transfer gap score: KL divergence of confidence distributions
  4. Contact sheet of top-50 OHRC detections
  5. results/transfer_evaluation/transfer_report.md

Usage:
  python scripts/evaluate_transfer.py
"""
import sys
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
MODELS_DIR     = PROJECT_ROOT / "models"
DEFAULT_MODEL  = MODELS_DIR / "yolov8n_lroc_adapted.pt"
ADAPTED_YAML   = PROJECT_ROOT / "data" / "source_domain" / "yolo_dataset_adapted" / "dataset.yaml"
RUNS_DIR       = PROJECT_ROOT / "results" / "training_runs"
DET_CSV        = PROJECT_ROOT / "results" / "ohrc_inference" / "detections.csv"
TILES_DIR      = PROJECT_ROOT / "data" / "tiles" / "usable"
OUT_DIR        = PROJECT_ROOT / "results" / "transfer_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_val_on_test_split(model, data_yaml: Path) -> dict:
    """Run validation on the held-out LROC test split."""
    print("\n[1] Running validation on LROC test split...")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        verbose=False,
    )
    results = {
        "map50":    float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":   float(metrics.box.mr),
    }
    print(f"  mAP@0.5:      {results['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['map50_95']:.4f}")
    print(f"  Precision:    {results['precision']:.4f}")
    print(f"  Recall:       {results['recall']:.4f}")
    return results


def get_lroc_val_confidences(model, data_yaml: Path) -> np.ndarray:
    """Run inference on LROC val split and collect confidence scores."""
    import yaml
    data = yaml.safe_load(data_yaml.read_text())
    val_dir = Path(data["path"]) / data.get("val", "images/val")

    val_imgs = list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpg"))
    if not val_imgs:
        return np.array([])

    print(f"\n[2] Collecting LROC val confidences ({len(val_imgs)} images)...")
    confs = []
    for img_path in val_imgs:
        results = model.predict(str(img_path), conf=0.1, verbose=False)
        for r in results:
            if r.boxes:
                confs.extend(r.boxes.conf.cpu().numpy().tolist())
    return np.array(confs)


def get_ohrc_confidences(det_csv: Path) -> np.ndarray:
    """Load OHRC detection confidences from inference CSV."""
    if not det_csv.exists():
        print(f"\n  [WARN] OHRC detections CSV not found: {det_csv}")
        print("  Run run_inference_ohrc.py first for OHRC confidence data.")
        return np.array([])

    df = pd.read_csv(det_csv)
    return df["confidence"].values if "confidence" in df.columns else np.array([])


def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
    """Compute KL(P || Q) between two confidence distributions."""
    if len(p) == 0 or len(q) == 0:
        return float("nan")
    eps = 1e-10
    h_p, edges = np.histogram(p, bins=bins, range=(0, 1), density=True)
    h_q, _     = np.histogram(q, bins=bins, range=(0, 1), density=True)
    h_p = h_p + eps
    h_q = h_q + eps
    h_p /= h_p.sum()
    h_q /= h_q.sum()
    return float(np.sum(h_p * np.log(h_p / h_q)))


def plot_confidence_distributions(lroc_confs: np.ndarray, ohrc_confs: np.ndarray,
                                   out_path: Path):
    """Plot overlapping confidence histograms: LROC vs OHRC."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    if len(lroc_confs) > 0:
        ax.hist(lroc_confs, bins=40, range=(0, 1), alpha=0.6,
                color="#4cc9f0", label=f"LROC val (n={len(lroc_confs)})",
                density=True)
    if len(ohrc_confs) > 0:
        ax.hist(ohrc_confs, bins=40, range=(0, 1), alpha=0.6,
                color="#f77f00", label=f"OHRC target (n={len(ohrc_confs)})",
                density=True)

    ax.set_xlabel("Detection Confidence", color="white")
    ax.set_ylabel("Density", color="white")
    ax.set_title("Confidence Distribution: LROC Source vs OHRC Target", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#16213e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Confidence plot saved: {out_path}")


def make_top50_contact_sheet(det_csv: Path, tiles_dir: Path, out_path: Path):
    """Render a 5×10 contact sheet of top-50 OHRC tiles by max detection confidence."""
    if not det_csv.exists():
        print("  [SKIP] Contact sheet — no detections CSV found")
        return

    df = pd.read_csv(det_csv)
    if df.empty:
        print("  [SKIP] Contact sheet — no detections")
        return

    top = (df.groupby("tile_filename")["confidence"]
             .max()
             .nlargest(50)
             .reset_index())

    THUMB = 128
    COLS, ROWS = 10, 5
    sheet = Image.new("RGB", (COLS * THUMB, ROWS * THUMB), color=(20, 20, 30))

    for i, row in enumerate(top.itertuples()):
        if i >= COLS * ROWS:
            break
        tile_path = tiles_dir / row.tile_filename
        if not tile_path.exists():
            continue
        try:
            thumb = Image.open(tile_path).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
            col = i % COLS
            r   = i // COLS
            sheet.paste(thumb, (col * THUMB, r * THUMB))
        except Exception:
            continue

    sheet.save(out_path)
    print(f"  Top-50 contact sheet saved: {out_path}")


def write_report(val_metrics: dict, lroc_confs: np.ndarray,
                 ohrc_confs: np.ndarray, kl: float, out_dir: Path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lroc_mean = float(lroc_confs.mean()) if len(lroc_confs) else float("nan")
    ohrc_mean = float(ohrc_confs.mean()) if len(ohrc_confs) else float("nan")

    report = f"""# Transfer Evaluation Report

**Generated:** {ts}

## 1. Source-Domain Performance (LROC test split)

| Metric | Value |
|---|---|
| mAP@0.5 | {val_metrics.get('map50', float('nan')):.4f} |
| mAP@0.5:0.95 | {val_metrics.get('map50_95', float('nan')):.4f} |
| Precision | {val_metrics.get('precision', float('nan')):.4f} |
| Recall | {val_metrics.get('recall', float('nan')):.4f} |

## 2. Domain Transfer Gap

| Metric | Value |
|---|---|
| LROC val detections | {len(lroc_confs)} |
| OHRC target detections | {len(ohrc_confs)} |
| LROC mean confidence | {lroc_mean:.4f} |
| OHRC mean confidence | {ohrc_mean:.4f} |
| KL divergence (LROC \\| OHRC) | {kl:.4f} |

**Interpretation:**
- KL < 0.1: Excellent transfer (distributions nearly identical)
- KL 0.1–0.5: Moderate gap (domain shift present but manageable)
- KL > 0.5: Large gap (CycleGAN style transfer recommended)
- KL = nan: No OHRC detections above threshold (run inference first)

## 3. Files

| File | Description |
|---|---|
| `confidence_distributions.png` | Overlapping LROC/OHRC confidence histograms |
| `top50_ohrc_detections.png` | Contact sheet of top-50 OHRC detections |
| `../../ohrc_inference/detections.csv` | All OHRC detections |
| `../../ohrc_inference/top_tiles/` | Visual overlays (top 100) |

## 4. Next Steps

- If KL < 0.3 and mAP@0.5 > 0.5: **Proceed to annotation of top OHRC detections**
- If KL > 0.5 or mAP@0.5 < 0.3: **Run CycleGAN style transfer** (see `train_cyclegan.py`)
- Optionally fine-tune with Prieur pseudo-labels: `python scripts/prepare_prieur.py`
"""
    out_path = out_dir / "transfer_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"  Transfer report: {out_path}")


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("Transfer Evaluation — LROC → OHRC")
    print("=" * 60)

    if not DEFAULT_MODEL.exists():
        print(f"ERROR: Model not found: {DEFAULT_MODEL}")
        print("Run train_yolov8.py first.")
        sys.exit(1)

    model = YOLO(str(DEFAULT_MODEL))

    # Source-domain test set evaluation
    val_metrics = {}
    if ADAPTED_YAML.exists():
        try:
            val_metrics = run_val_on_test_split(model, ADAPTED_YAML)
        except Exception as e:
            print(f"  [WARN] Could not run val: {e}")

    # Confidence distributions
    lroc_confs = get_lroc_val_confidences(model, ADAPTED_YAML) if ADAPTED_YAML.exists() else np.array([])
    ohrc_confs = get_ohrc_confidences(DET_CSV)

    kl = kl_divergence(lroc_confs, ohrc_confs)
    print(f"\n[3] KL divergence (transfer gap): {kl:.4f}")

    # Plots
    print("\n[4] Generating plots...")
    plot_confidence_distributions(
        lroc_confs, ohrc_confs,
        OUT_DIR / "confidence_distributions.png"
    )
    make_top50_contact_sheet(DET_CSV, TILES_DIR, OUT_DIR / "top50_ohrc_detections.png")

    # Report
    print("\n[5] Writing transfer report...")
    write_report(val_metrics, lroc_confs, ohrc_confs, kl, OUT_DIR)

    print("\n" + "=" * 60)
    print("Evaluation complete")
    print(f"  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
