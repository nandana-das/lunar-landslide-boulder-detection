"""Full 3-Stage Pipeline for RT-DETR-L on Lunar Boulder Detection:
Stage 1: Pretrained backbone freeze (freeze=10), RMaM+Prieur combined (4,068 images), 50 epochs, LR=1e-3, AdamW.
Stage 2: Unfreeze all (freeze=0), histogram-matched combined data (combined_hm), 50 epochs, LR=1e-4, cos_lr=True, patience=20.
Stage 3: Full OHRC inference on all 31,769 usable tiles at tau=0.20.
Table II & III: Validation evaluation on Prieur 697 HM val set and OHRC detection aggregation.
"""

import os
import sys
import time
from pathlib import Path
import torch
import pandas as pd

# Reconfigure stdout/stderr for clean utf-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Bypass thop profiling to eliminate 'ReLU' object has no attribute 'total_ops' bug
import ultralytics.utils.torch_utils as tu
tu.get_flops = lambda model, imgsz=640: 0.0

from ultralytics import RTDETR

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA_STAGE1 = str(BASE / "data/combined/dataset_combined.yaml")
DATA_STAGE2 = str(BASE / "data/combined_hm/dataset_combined_hm.yaml")
RUNS = BASE / "runs"
OHRC_TILES = BASE / "data/tiles/usable"
RESULTS_DIR = BASE / "results"
OHRC_INFERENCE_DIR = RESULTS_DIR / "ohrc_inference"

def run_stage1():
    print("\n" + "=" * 70)
    print("STARTING STAGE 1: RT-DETR-L Pretrained Backbone Freeze")
    print("Dataset: RMaM + Prieur Combined (4,068 images)")
    print("Config: epochs=50, lr0=0.001, optimizer=AdamW, freeze=10, batch=4")
    print("=" * 70)

    save_dir = RUNS / "stage1_rtdetr_l_combined"
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"

    # Check if Stage 1 already reached 50 epochs
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            if len(df) >= 50 and best_weights.exists():
                print(f"Stage 1 already complete (50/50 epochs)! Found weights at: {best_weights}")
                return str(best_weights)
        except Exception:
            pass

    torch.cuda.empty_cache()

    if last_weights.exists():
        print(f"Found existing checkpoint at {last_weights}. Resuming Stage 1...")
        model = RTDETR(str(last_weights))
        model.train(resume=True)
    else:
        model = RTDETR('rtdetr-l.pt')
        model.train(
            data=DATA_STAGE1,
            epochs=50,
            imgsz=640,
            batch=4,
            lr0=0.001,
            optimizer='AdamW',
            freeze=10,
            project=str(RUNS),
            name='stage1_rtdetr_l_combined',
            device=0,
            workers=0,
            amp=True,
            plots=True,
            save=True
        )

    if not best_weights.exists():
        raise RuntimeError(f"Stage 1 training finished but best weights not found at {best_weights}")

    print(f"Stage 1 Complete! Best weights: {best_weights}")
    return str(best_weights)

def run_stage2(stage1_weights: str):
    print("\n" + "=" * 70)
    print("STARTING STAGE 2: RT-DETR-L Full Unfreeze on Histogram-Matched Data")
    print("Dataset: Combined HM (4,068 images, multi-reference histogram matched)")
    print("Config: epochs=50, lr0=0.0001, cos_lr=True, patience=20, freeze=0, batch=4")
    print("=" * 70)

    save_dir = RUNS / "stage2_rtdetr_l_combined_hm"
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"

    # Check if Stage 2 already completed
    if results_csv.exists() and best_weights.exists():
        try:
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            if len(df) >= 50:
                print(f"Stage 2 already complete (50/50 epochs)! Found weights at: {best_weights}")
                return str(best_weights)
        except Exception:
            pass

    torch.cuda.empty_cache()

    if last_weights.exists():
        print(f"Found existing checkpoint at {last_weights}. Resuming Stage 2...")
        model = RTDETR(str(last_weights))
        model.train(resume=True)
    else:
        model = RTDETR(stage1_weights)
        model.train(
            data=DATA_STAGE2,
            epochs=50,
            imgsz=640,
            batch=4,
            lr0=0.0001,
            cos_lr=True,
            patience=20,
            freeze=0,
            project=str(RUNS),
            name='stage2_rtdetr_l_combined_hm',
            device=0,
            workers=0,
            amp=True,
            plots=True,
            save=True
        )

    if not best_weights.exists():
        raise RuntimeError(f"Stage 2 training finished but best weights not found at {best_weights}")

    print(f"Stage 2 Complete! Best weights: {best_weights}")
    return str(best_weights)

def run_evaluation_table2(stage2_weights: str):
    print("\n" + "=" * 70)
    print("EVALUATING TABLE II: Source Domain Validation (Prieur HM Val Set, 697 images)")
    print("=" * 70)

    torch.cuda.empty_cache()
    model = RTDETR(stage2_weights)
    
    # Run validation on combined_hm val split (Prieur lunar HM val set)
    metrics = model.val(
        data=DATA_STAGE2,
        split='val',
        imgsz=640,
        device=0,
        workers=0,
        conf=0.20,
        plots=True
    )
    
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    print(f"RT-DETR-L Table II Metrics: mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    
    # Compile Table II comparing with existing YOLO models
    table2_rows = [
        {'Model': 'YOLO26n', 'mAP@0.5': 0.6400, 'Precision': 0.6600, 'Recall': 0.5900, 'F1': round(2*0.66*0.59/(0.66+0.59), 4)},
        {'Model': 'YOLOv8n', 'mAP@0.5': 0.5960, 'Precision': 0.6200, 'Recall': 0.5680, 'F1': round(2*0.62*0.568/(0.62+0.568), 4)},
        {'Model': 'YOLOv5s', 'mAP@0.5': 0.6490, 'Precision': 0.6530, 'Recall': 0.6070, 'F1': round(2*0.653*0.607/(0.653+0.607), 4)},
        {'Model': 'RT-DETR-L', 'mAP@0.5': round(map50, 4), 'Precision': round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}
    ]
    t2_df = pd.DataFrame(table2_rows)
    t2_csv = RESULTS_DIR / "table2_rtdetr_comparison.csv"
    t2_df.to_csv(t2_csv, index=False)
    print(f"Table II saved to: {t2_csv}")
    print(t2_df.to_string(index=False))
    return t2_df

def run_stage3_ohrc_inference(stage2_weights: str):
    print("\n" + "=" * 70)
    print("STARTING STAGE 3: Full OHRC Inference (31,769 usable tiles at tau=0.20)")
    print(f"Source Directory: {OHRC_TILES}")
    print("=" * 70)

    out_csv = OHRC_INFERENCE_DIR / "raw_detections_rtdetr_l.csv"
    out_img = Path("C:/ohrc_detections/rtdetr_l/images")
    out_lbl = Path("C:/ohrc_detections/rtdetr_l/labels")
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    OHRC_INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    model = RTDETR(stage2_weights)

    all_detections = []
    total_detections = 0
    positive_tiles = 0
    scanned_tiles = 0
    t0 = time.time()

    # Stream through tiles
    for r in model.predict(source=str(OHRC_TILES), imgsz=640, conf=0.20, stream=True, device=0):
        scanned_tiles += 1
        tile_name = Path(r.path).name
        n_boxes = len(r.boxes)
        
        if n_boxes > 0:
            positive_tiles += 1
            total_detections += n_boxes
            r.save(filename=str(out_img / tile_name))
            r.save_txt(str(out_lbl / (Path(r.path).stem + '.txt')))
            
            for b in r.boxes:
                conf = float(b.conf[0])
                coords = b.xyxy[0].tolist()
                all_detections.append({
                    'tile': tile_name,
                    'conf': round(conf, 4),
                    'x1': round(coords[0], 2),
                    'y1': round(coords[1], 2),
                    'x2': round(coords[2], 2),
                    'y2': round(coords[3], 2)
                })

        if scanned_tiles % 1000 == 0 or scanned_tiles == 31769:
            elapsed = time.time() - t0
            fps = scanned_tiles / elapsed if elapsed > 0 else 0
            print(f"[{scanned_tiles}/31769] Tiles processed ({fps:.1f} tiles/sec) | Detections: {total_detections:,} | Positive Tiles: {positive_tiles:,}")

    det_df = pd.DataFrame(all_detections)
    det_df.to_csv(out_csv, index=False)
    print(f"\nRaw detections saved to: {out_csv}")
    print(f"Total OHRC Detections: {total_detections:,}")
    print(f"Tiles with Detections: {positive_tiles:,} / {scanned_tiles:,}")

    # Compile Table III comparing with existing YOLO models
    table3_rows = [
        {'Model': 'YOLO26n', 'Total Detections': 4565, 'Positive Tiles': 1193, 'Evaluated Tiles': 31769, 'Mean Confidence': 0.297},
        {'Model': 'YOLOv5s', 'Total Detections': 5026, 'Positive Tiles': 1532, 'Evaluated Tiles': 31769, 'Mean Confidence': 0.312},
        {'Model': 'YOLOv8n', 'Total Detections': 41941, 'Positive Tiles': 6621, 'Evaluated Tiles': 31769, 'Mean Confidence': 0.313},
        {'Model': 'RT-DETR-L', 'Total Detections': total_detections, 'Positive Tiles': positive_tiles, 'Evaluated Tiles': scanned_tiles, 'Mean Confidence': round(det_df['conf'].mean() if len(det_df) else 0.0, 4)}
    ]
    t3_df = pd.DataFrame(table3_rows)
    t3_csv = RESULTS_DIR / "table3_rtdetr_comparison.csv"
    t3_df.to_csv(t3_csv, index=False)
    print(f"Table III saved to: {t3_csv}")
    print(t3_df.to_string(index=False))

    # Append to inference_results_summary.txt
    summary_file = RESULTS_DIR / "inference_results_summary.txt"
    if summary_file.exists():
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n## RT-DETR-L Results (RMaM+Prieur, Multi-HM, tau=0.20)\n")
            f.write(f"RT-DETR-L: detections={total_detections}, tiles={positive_tiles} (evaluated={scanned_tiles})\n")

    return t3_df

def main():
    print("=" * 70)
    print("STARTING COMPLETE RT-DETR-L 3-STAGE PIPELINE")
    print("=" * 70)
    
    t_start = time.time()
    
    # 1. Stage 1 Training
    stage1_weights = run_stage1()
    
    # 2. Stage 2 Training on histogram-matched data
    stage2_weights = run_stage2(stage1_weights)
    
    # 3. Source Domain Validation (Table II)
    run_evaluation_table2(stage2_weights)
    
    # 4. Stage 3 Full OHRC Inference (Table III)
    run_stage3_ohrc_inference(stage2_weights)
    
    total_time_h = (time.time() - t_start) / 3600
    print("\n" + "=" * 70)
    print(f"ALL 3 STAGES COMPLETE! Total elapsed time: {total_time_h:.2f} hours")
    print("=" * 70)

if __name__ == '__main__':
    main()
