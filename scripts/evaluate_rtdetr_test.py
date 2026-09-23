"""Evaluate trained RT-DETR-L Stage-2 weights on Prieur lunar test split."""
import sys
from pathlib import Path
import csv
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Bypass thop profiling for RT-DETR
import ultralytics.utils.torch_utils as tu
tu.get_flops = lambda model, imgsz=640: 0.0

from ultralytics import RTDETR

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA = BASE / "data/combined_hm/dataset_combined_hm.yaml"
WEIGHTS = BASE / "runs/stage2_rtdetr_l_combined_hm/weights/best.pt"
RESULTS_DIR = BASE / "results"

def main():
    print("=" * 70)
    print("Evaluating RT-DETR-L Stage-2 on Prieur Lunar HM Test Split (262 images)")
    print(f"Weights: {WEIGHTS}")
    print(f"Data: {DATA}")
    print("=" * 70)

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights file not found: {WEIGHTS}")

    torch.cuda.empty_cache()
    model = RTDETR(str(WEIGHTS))

    # Same eval protocol as evaluate_test_split.py
    metrics = model.val(
        data=str(DATA),
        split='test',
        imgsz=640,
        device=0,
        verbose=True
    )

    p = float(metrics.box.mp)
    r = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = (2 * p * r) / (p + r + 1e-16) if (p + r) > 0 else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS FOR RT-DETR-L (TEST SPLIT):")
    print(f"  mAP@0.5:      {map50:.4f}")
    print(f"  mAP@0.5:0.95: {map50_95:.4f}")
    print(f"  Precision:    {p:.4f}")
    print(f"  Recall:       {r:.4f}")
    print(f"  F1-Score:     {f1:.4f}")
    print("=" * 70)

    rtdetr_row = {
        'Model': 'RT-DETR-L',
        'mAP@0.5': round(map50, 4),
        'mAP@0.5:0.95': round(map50_95, 4),
        'Precision': round(p, 4),
        'Recall': round(r, 4),
        'F1': round(f1, 4),
    }

    # 1. Append/Update results/test_metrics_stage2.csv
    csv_path = RESULTS_DIR / "test_metrics_stage2.csv"
    existing_rows = []
    if csv_path.exists():
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Model') != 'RT-DETR-L':
                    existing_rows.append(row)

    existing_rows.append(rtdetr_row)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1'])
        writer.writeheader()
        writer.writerows(existing_rows)
    print(f"\n[Updated CSV] {csv_path}")

    # 2. Append/Update results/test_metrics_summary.md
    md_lines = [
        "# Stage-2 Model Evaluation on Prieur Lunar Test Split (HM, 262 images)",
        "",
        "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for row in existing_rows:
        m_name = row['Model']
        m50 = float(row['mAP@0.5'])
        m50_95 = float(row['mAP@0.5:0.95'])
        prec = float(row['Precision'])
        rec = float(row['Recall'])
        f1_val = float(row['F1'])
        md_lines.append(f"| **{m_name}** | {m50:.4f} | {m50_95:.4f} | {prec:.4f} | {rec:.4f} | {f1_val:.4f} |")

    md_content = "\n".join(md_lines) + "\n"
    md_path = RESULTS_DIR / "test_metrics_summary.md"
    md_path.write_text(md_content, encoding='utf-8')
    print(f"[Updated Markdown] {md_path}")
    print("\n" + md_content)

if __name__ == '__main__':
    main()
