"""
generate_ohrc_confidence_curves.py
===================================
Confidence calibration analysis on OHRC lunar tiles for Stage-2 models:
YOLO26n, YOLOv8n, YOLOv5s.

Computes:
  1. Precision_proxy vs confidence and Recall_proxy vs confidence curves
     (Proxy precision via cross-model NMS consensus and overlap consistency)
  2. F1-score vs threshold curve per model
  3. Optimal operating confidence threshold per model (max F1)
  4. Saves 300 DPI publication-quality figures and calibration summary CSV.

NOTE: This is NOT ground truth precision/recall (no annotated ground truth
exists for Chandrayaan-2 OHRC target tiles). Framed as confidence calibration
and consensus analysis only.
"""

from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
RUNS = BASE / "runs"
RESULTS_DIR = BASE / "results" / "confidence_analysis"
OHRC_INFER_DIR = BASE / "results" / "ohrc_inference"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OHRC_INFER_DIR.mkdir(parents=True, exist_ok=True)

# Candidate OHRC tiles
TILES_914 = list(Path("C:/ohrc_detections/images").glob("*.png"))
TILES_PRIO = list((BASE / "data/priority_sample").glob("*.png"))
ALL_TILES = sorted(list(set([str(f) for f in TILES_914 + TILES_PRIO])))

MODELS = {
    "YOLO26n": RUNS / "stage2_yolo26n_combined_hm_v2" / "weights" / "best.pt",
    "YOLOv8n": RUNS / "stage2_yolov8n_combined_hm_v2" / "weights" / "best.pt",
    "YOLOv5s": RUNS / "stage2_yolov5s_combined_hm_v2" / "weights" / "best.pt",
}

def box_iou(box1, box2):
    """Compute IoU between [x1, y1, x2, y2] and [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def collect_detections():
    """Run inference or load cached raw detections at tau=0.20."""
    detections = {}
    for name, weights in MODELS.items():
        csv_file = OHRC_INFER_DIR / f"raw_detections_{name.lower()}.csv"
        if csv_file.exists():
            print(f"Loading cached detections for {name} from {csv_file}...")
            df = pd.read_csv(csv_file)
            detections[name] = df.to_dict("records")
            print(f"  {name}: {len(detections[name])} detections across {df['tile'].nunique()} tiles")
            continue

        print(f"\nRunning inference for {name} on {len(ALL_TILES)} OHRC tiles (tau=0.20)...")
        model = YOLO(str(weights))
        model_dets = []
        out_lbl_dir = Path(f"C:/ohrc_detections/{name.lower()}/labels")
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        seen_tiles = set()
        for source_dir in [Path("C:/ohrc_detections/images"), BASE / "data/priority_sample"]:
            if not source_dir.exists():
                continue
            for r in model.predict(source=str(source_dir), imgsz=640, conf=0.20, stream=True, device=0, verbose=False):
                tile_name = Path(r.path).name
                if tile_name in seen_tiles:
                    continue
                seen_tiles.add(tile_name)
                count += 1
                if len(r.boxes) > 0:
                    lbl_lines = []
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        xywh = box.xywh[0].cpu().numpy().tolist()
                        lbl_lines.append(f"0 {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {conf:.4f}")
                        model_dets.append({
                            "tile": tile_name,
                            "conf": round(conf, 4),
                            "x1": round(xyxy[0], 2),
                            "y1": round(xyxy[1], 2),
                            "x2": round(xyxy[2], 2),
                            "y2": round(xyxy[3], 2),
                        })
                    (out_lbl_dir / (Path(r.path).stem + ".txt")).write_text("\n".join(lbl_lines), encoding="utf-8")
                if count % 300 == 0:
                    print(f"  [{name}] {count}/{len(ALL_TILES)} tiles processed ({len(model_dets)} detections)...")

        # Save to CSV
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tile", "conf", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(model_dets)

        detections[name] = model_dets
        tiles_hit = len(set(d["tile"] for d in model_dets))
        print(f"  {name} complete: {len(model_dets)} detections on {tiles_hit} tiles saved to {csv_file}")

    return detections

def build_consensus_pseudo_gt(detections):
    """
    Build consensus reference boulders per tile.
    A candidate boulder is formed by clustering detections across models (IoU >= 0.45).
    Consensus candidates require support from >= 2 models or high confidence agreement.
    """
    print("\nBuilding multi-model consensus pseudo-ground truth...")
    all_tiles = set()
    for dets in detections.values():
        all_tiles.update(d["tile"] for d in dets)

    consensus = {}
    total_consensus_boxes = 0

    for tile in all_tiles:
        tile_boxes = {}
        for m_name, dets in detections.items():
            tile_boxes[m_name] = [
                ([d["x1"], d["y1"], d["x2"], d["y2"]], d["conf"])
                for d in dets if d["tile"] == tile
            ]

        # Form consensus clusters across models
        matched_clusters = []
        models_list = list(detections.keys())
        
        # Take YOLO26n and YOLOv5s as primary consensus anchors
        for box_a, conf_a in tile_boxes[models_list[0]]:
            cluster = [box_a]
            supports = 1
            for m_other in models_list[1:]:
                best_iou = 0
                best_box = None
                for box_b, conf_b in tile_boxes[m_other]:
                    iou = box_iou(box_a, box_b)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = box_b
                if best_iou >= 0.45:
                    cluster.append(best_box)
                    supports += 1
            
            # Consensus: supported by at least 2 models, or anchor conf >= 0.40
            if supports >= 2 or conf_a >= 0.40:
                avg_box = [
                    np.mean([b[0] for b in cluster]),
                    np.mean([b[1] for b in cluster]),
                    np.mean([b[2] for b in cluster]),
                    np.mean([b[3] for b in cluster]),
                ]
                matched_clusters.append(avg_box)

        # Also add supported boxes from model 2 & 3 not yet captured
        for box_b, conf_b in tile_boxes[models_list[2]]: # YOLOv5s
            if not any(box_iou(box_b, cb) >= 0.45 for cb in matched_clusters):
                # Check agreement with YOLOv8n
                if any(box_iou(box_b, b8[0]) >= 0.45 for b8 in tile_boxes[models_list[1]]):
                    matched_clusters.append(box_b)

        consensus[tile] = matched_clusters
        total_consensus_boxes += len(matched_clusters)

    print(f"Total consensus reference boulder candidates: {total_consensus_boxes} across {len(all_tiles)} tiles")
    return consensus

def evaluate_thresholds(detections, consensus):
    """Sweep threshold tau from 0.20 to 0.85 and compute proxy metrics."""
    thresholds = np.arange(0.20, 0.86, 0.02)
    records = []

    total_ref = sum(len(boxes) for boxes in consensus.values())
    all_tiles = list(consensus.keys())

    for tau in thresholds:
        tau = round(float(tau), 2)
        for name, dets in detections.items():
            filtered = [d for d in dets if d["conf"] >= tau]
            
            # Group by tile
            tile_dets = {}
            for d in filtered:
                tile_dets.setdefault(d["tile"], []).append([d["x1"], d["y1"], d["x2"], d["y2"]])

            tp = 0
            fp = 0
            for tile, ref_boxes in consensus.items():
                pred_boxes = tile_dets.get(tile, [])
                matched_ref = set()
                for p_box in pred_boxes:
                    hit = False
                    for r_idx, r_box in enumerate(ref_boxes):
                        if r_idx not in matched_ref and box_iou(p_box, r_box) >= 0.45:
                            matched_ref.add(r_idx)
                            tp += 1
                            hit = True
                            break
                    if not hit:
                        fp += 1

            prec_proxy = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec_proxy = tp / total_ref if total_ref > 0 else 0.0
            f1_proxy = (2 * prec_proxy * rec_proxy) / (prec_proxy + rec_proxy + 1e-16) if (prec_proxy + rec_proxy) > 0 else 0.0

            records.append({
                "threshold": tau,
                "model": name,
                "precision_proxy": round(prec_proxy, 4),
                "recall_proxy": round(rec_proxy, 4),
                "f1": round(f1_proxy, 4),
                "detections": len(filtered),
                "tiles_hit": len(tile_dets),
            })

    df = pd.DataFrame(records)
    return df

def generate_plots(df):
    """Generate 300 DPI publication plots."""
    models = df["model"].unique()
    colors = {"YOLO26n": "#00b4d8", "YOLOv8n": "#ff595e", "YOLOv5s": "#8ac926"}
    styles = {"YOLO26n": "-", "YOLOv8n": "--", "YOLOv5s": "-."}

    # 1. Precision & Recall vs Confidence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    for name in models:
        m_df = df[df["model"] == name]
        ax1.plot(m_df["threshold"], m_df["precision_proxy"], label=f"{name}", color=colors[name], lw=2.2, linestyle=styles[name])
        ax2.plot(m_df["threshold"], m_df["recall_proxy"], label=f"{name}", color=colors[name], lw=2.2, linestyle=styles[name])

    ax1.set_title("Proxy Precision vs. Confidence Threshold\n[Limitation: Proxy via Consensus Overlap; No OHRC Ground Truth]", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("Confidence Threshold (τ)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Proxy Precision", fontsize=10, fontweight="bold")
    ax1.set_xlim(0.20, 0.85)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="#f8f9fa")

    ax2.set_title("Proxy Recall vs. Confidence Threshold\n[Limitation: Normalized to Consensus Candidate Universe]", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Confidence Threshold (τ)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Proxy Recall", fontsize=10, fontweight="bold")
    ax2.set_xlim(0.20, 0.85)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, facecolor="#f8f9fa")

    plt.tight_layout()
    p_pr_path = RESULTS_DIR / "precision_recall_vs_confidence.png"
    fig.savefig(p_pr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p_pr_path}")

    # 2. F1 vs Threshold
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    best_points = {}
    for name in models:
        m_df = df[df["model"] == name]
        best_idx = m_df["f1"].idxmax()
        best_row = m_df.loc[best_idx]
        best_points[name] = best_row

        ax.plot(m_df["threshold"], m_df["f1"], label=f"{name} (max F1={best_row['f1']:.3f} @ τ={best_row['threshold']:.2f})", color=colors[name], lw=2.4, linestyle=styles[name])
        ax.scatter([best_row["threshold"]], [best_row["f1"]], color=colors[name], s=90, zorder=5)
        ax.annotate(f"τ={best_row['threshold']:.2f}\nF1={best_row['f1']:.3f}", 
                    xy=(best_row["threshold"], best_row["f1"]),
                    xytext=(best_row["threshold"] + 0.03, best_row["f1"] - 0.05),
                    arrowprops=dict(arrowstyle="->", color=colors[name], lw=1.2),
                    fontsize=9, fontweight="bold", color="#1f2421",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor=colors[name], alpha=0.9))

    ax.set_title("F1-Score vs. Operating Confidence Threshold (τ)\n[Confidence Calibration Analysis — Target OHRC Domain]", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Confidence Threshold (τ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Proxy F1-Score", fontsize=11, fontweight="bold")
    ax.set_xlim(0.20, 0.85)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#f8f9fa", loc="lower left", fontsize=9.5)

    caption = "NOTE: Evaluated on target Chandrayaan-2 OHRC tiles. Precision and recall are calibrated proxies derived from cross-architecture NMS consensus.\nThis analysis is for threshold calibration only and does not represent verified ground truth accuracy."
    fig.text(0.5, -0.05, caption, ha="center", fontsize=8, style="italic", color="#555555")

    plt.tight_layout()
    f1_path = RESULTS_DIR / "f1_vs_threshold.png"
    fig.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {f1_path}")

    return best_points

def main():
    print("=" * 75)
    print("OHRC Confidence Calibration & Operating Threshold Analysis")
    print("=" * 75)

    # 1. Collect / cache detections
    detections = collect_detections()

    # 2. Consensus reference
    consensus = build_consensus_pseudo_gt(detections)

    # 3. Evaluate sweep
    df = evaluate_thresholds(detections, consensus)

    # Save summary CSV
    csv_out = RESULTS_DIR / "confidence_calibration_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved calibration table: {csv_out}")

    # 4. Generate plots
    best_points = generate_plots(df)

    # 5. Print recommended thresholds
    print("\n" + "=" * 75)
    print("RECOMMENDED OPERATING THRESHOLDS (MAX F1 POINT)")
    print("=" * 75)
    rec_rows = []
    for name, row in best_points.items():
        print(f"Model: {name:<8} | Optimal τ* = {row['threshold']:.2f} | Max F1 = {row['f1']:.4f} | Proxy Prec = {row['precision_proxy']:.4f} | Proxy Rec = {row['recall_proxy']:.4f} | Dets = {int(row['detections'])}")
        rec_rows.append({
            "Model": name,
            "Recommended_Threshold": row["threshold"],
            "Max_F1": row["f1"],
            "Proxy_Precision": row["precision_proxy"],
            "Proxy_Recall": row["recall_proxy"],
            "Active_Detections": int(row["detections"]),
            "Active_Tiles": int(row["tiles_hit"]),
        })

    rec_df = pd.DataFrame(rec_rows)
    rec_csv = RESULTS_DIR / "recommended_operating_thresholds.csv"
    rec_df.to_csv(rec_csv, index=False)
    print(f"Saved recommendations table: {rec_csv}")

if __name__ == "__main__":
    main()
