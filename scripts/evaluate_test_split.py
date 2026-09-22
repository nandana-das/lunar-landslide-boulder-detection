"""Evaluate Stage-2 models on Prieur lunar HM test split (262 images)."""
from pathlib import Path
import csv
from ultralytics import YOLO

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA = BASE / "data/combined_hm/dataset_combined_hm.yaml"
RUNS = BASE / "runs"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    'YOLO26n': RUNS / 'stage2_yolo26n_combined_hm_v2' / 'weights' / 'best.pt',
    'YOLOv8n': RUNS / 'stage2_yolov8n_combined_hm_v2' / 'weights' / 'best.pt',
    'YOLOv5s': RUNS / 'stage2_yolov5s_combined_hm_v2' / 'weights' / 'best.pt',
}

def main():
    print("=" * 70)
    print("Evaluating Stage-2 Models on Prieur Lunar HM Test Split (262 images)")
    print("=" * 70)

    rows = []
    for name, weights in MODELS.items():
        print(f"\nEvaluating {name} from {weights}...")
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(DATA),
            split='test',
            imgsz=640,
            device=0,
            verbose=False
        )

        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        f1 = (2 * p * r) / (p + r + 1e-16) if (p + r) > 0 else 0.0

        print(f"  mAP@0.5:      {map50:.4f}")
        print(f"  mAP@0.5:0.95: {map50_95:.4f}")
        print(f"  Precision:    {p:.4f}")
        print(f"  Recall:       {r:.4f}")
        print(f"  F1-Score:     {f1:.4f}")

        rows.append({
            'Model': name,
            'mAP@0.5': round(map50, 4),
            'mAP@0.5:0.95': round(map50_95, 4),
            'Precision': round(p, 4),
            'Recall': round(r, 4),
            'F1': round(f1, 4),
        })

    # Save CSV
    csv_path = RESULTS_DIR / "test_metrics_stage2.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[CSV saved] {csv_path}")

    # Generate and save Markdown summary
    md_lines = [
        "# Stage-2 Model Evaluation on Prieur Lunar Test Split (HM, 262 images)",
        "",
        "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(f"| **{row['Model']}** | {row['mAP@0.5']:.4f} | {row['mAP@0.5:0.95']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} |")

    md_content = "\n".join(md_lines) + "\n"
    md_path = RESULTS_DIR / "test_metrics_summary.md"
    md_path.write_text(md_content, encoding='utf-8')
    print(f"[Markdown saved] {md_path}")
    print("\n" + md_content)

if __name__ == '__main__':
    main()
