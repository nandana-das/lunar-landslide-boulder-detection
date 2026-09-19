"""
run_inference_ohrc.py  — Step 4
=================================
Runs the trained YOLOv8n detector on all 31,769 OHRC target tiles.

Processes tiles in batches for efficiency.
Saves:
  results/ohrc_inference/detections.csv    Per-tile detection records
  results/ohrc_inference/top_tiles/        Visual overlays of top 100 tiles
  results/ohrc_inference/summary.txt       Run summary

Usage:
  python scripts/run_inference_ohrc.py [--model PATH] [--conf FLOAT] [--batch-size N]
"""
import sys
import csv
import shutil
import argparse
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
TILES_DIR      = PROJECT_ROOT / "data" / "tiles" / "usable"
MODELS_DIR     = PROJECT_ROOT / "models"
DEFAULT_MODEL  = MODELS_DIR / "yolov8n_lroc_adapted.pt"
OUT_DIR        = PROJECT_ROOT / "results" / "ohrc_inference"
TOP_TILES_DIR  = OUT_DIR / "top_tiles"
DET_CSV        = OUT_DIR / "detections.csv"
SUMMARY_TXT    = OUT_DIR / "summary.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_TILES_DIR.mkdir(parents=True, exist_ok=True)


def draw_detections(img_arr, boxes, scores, labels, class_names):
    """Draw bounding boxes on a numpy image array. Returns numpy array."""
    import numpy as np
    import cv2
    # Convert L to BGR for drawing
    if img_arr.ndim == 2:
        vis = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
    else:
        vis = img_arr.copy()

    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        name = class_names[label] if label < len(class_names) else str(label)
        color = (0, 255, 64)  # bright green
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(vis, f"{name} {score:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf",       type=float, default=0.30,
                        help="Detection confidence threshold")
    parser.add_argument("--iou",        type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size. Reduce if OOM.")
    parser.add_argument("--max-tiles",  type=int, default=None,
                        help="Limit number of tiles (for testing)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
        import torch
        import numpy as np
        import cv2
        from PIL import Image
    except ImportError as e:
        print(f"ERROR: {e}\nRun: pip install ultralytics opencv-python")
        sys.exit(1)

    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        print("Run train_yolov8.py first.")
        sys.exit(1)

    print("=" * 60)
    print("OHRC Inference — YOLOv8n Target Domain Detection")
    print("=" * 60)
    print(f"  Model:      {args.model.name}")
    print(f"  Conf:       {args.conf}")
    print(f"  IOU:        {args.iou}")
    print(f"  Batch size: {args.batch_size}")

    model = YOLO(str(args.model))
    class_names = model.names if hasattr(model, "names") else {0: "rockfall"}

    # Collect tiles
    all_tiles = sorted(TILES_DIR.glob("*.png"))
    if args.max_tiles:
        all_tiles = all_tiles[:args.max_tiles]

    print(f"\n  Tiles to process: {len(all_tiles):,}")
    print(f"  Output at: {OUT_DIR}\n")

    # ── Run inference ─────────────────────────────────────────────────────
    tile_records = []   # (filename, n_detections, max_conf, total_conf)
    all_det_rows = []   # individual detection rows for CSV

    for i in tqdm(range(0, len(all_tiles), args.batch_size), unit="batch"):
        batch_paths = all_tiles[i : i + args.batch_size]
        results = model.predict(
            [str(p) for p in batch_paths],
            conf=args.conf,
            iou=args.iou,
            imgsz=640,
            verbose=False,
        )

        for img_path, result in zip(batch_paths, results):
            boxes  = result.boxes
            n_det  = len(boxes) if boxes else 0
            max_c  = 0.0
            total_c = 0.0

            if n_det > 0:
                confs  = boxes.conf.cpu().numpy()
                xyxy   = boxes.xyxy.cpu().numpy()
                cls    = boxes.cls.cpu().numpy().astype(int)
                max_c  = float(confs.max())
                total_c = float(confs.sum())

                for det_idx in range(n_det):
                    all_det_rows.append({
                        "tile_filename": img_path.name,
                        "det_x1": float(xyxy[det_idx][0]),
                        "det_y1": float(xyxy[det_idx][1]),
                        "det_x2": float(xyxy[det_idx][2]),
                        "det_y2": float(xyxy[det_idx][3]),
                        "confidence": float(confs[det_idx]),
                        "class_id": int(cls[det_idx]),
                        "class_name": class_names.get(int(cls[det_idx]), "rockfall"),
                    })

            tile_records.append((img_path.name, n_det, max_c, total_c))

    # ── Save detections CSV ───────────────────────────────────────────────
    print(f"\nSaving detections to {DET_CSV}...")
    with open(DET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tile_filename", "det_x1", "det_y1", "det_x2", "det_y2",
            "confidence", "class_id", "class_name"
        ])
        writer.writeheader()
        writer.writerows(all_det_rows)

    # ── Rank tiles by detection confidence ───────────────────────────────
    tile_records.sort(key=lambda r: r[2], reverse=True)  # sort by max_conf desc
    tiles_with_dets = [(r) for r in tile_records if r[1] > 0]

    print(f"\nTiles with detections: {len(tiles_with_dets)} / {len(tile_records)}")

    # ── Save overlay images for top 100 ──────────────────────────────────
    n_top = min(100, len(tiles_with_dets))
    print(f"Saving visual overlays for top {n_top} tiles...")

    import numpy as np
    import cv2

    top_dets_by_tile = {}
    for row in all_det_rows:
        fname = row["tile_filename"]
        if fname not in top_dets_by_tile:
            top_dets_by_tile[fname] = []
        top_dets_by_tile[fname].append(row)

    for rank, (fname, n_det, max_c, _) in enumerate(tiles_with_dets[:n_top]):
        src = TILES_DIR / fname
        if not src.exists():
            continue
        img_arr = np.array(Image.open(src).convert("L"))
        dets = top_dets_by_tile.get(fname, [])
        boxes  = [(d["det_x1"], d["det_y1"], d["det_x2"], d["det_y2"]) for d in dets]
        scores = [d["confidence"] for d in dets]
        clss   = [d["class_id"] for d in dets]
        vis    = draw_detections(img_arr, boxes, scores, clss,
                                 {0: "rockfall", 1: "landslide"})
        # Add rank label
        cv2.putText(vis, f"#{rank+1} n={n_det} conf={max_c:.2f}",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        out_name = f"rank{rank+1:03d}_{fname}"
        cv2.imwrite(str(TOP_TILES_DIR / out_name), vis)

    # ── Summary ───────────────────────────────────────────────────────────
    n_tiles  = len(tile_records)
    n_det_tiles = len(tiles_with_dets)
    n_total_dets = len(all_det_rows)
    detection_rate = 100 * n_det_tiles / n_tiles if n_tiles else 0

    summary = f"""OHRC Inference Summary
======================
Model:           {args.model.name}
Confidence:      {args.conf}
Tiles processed: {n_tiles:,}
Tiles with dets: {n_det_tiles:,}  ({detection_rate:.1f}%)
Total detections:{n_total_dets:,}
Avg det/tile:    {n_total_dets/max(n_det_tiles,1):.1f}
Top tile:        {tile_records[0][0] if tile_records else 'N/A'}
Top tile conf:   {tile_records[0][2] if tile_records else 'N/A'}
Run timestamp:   {datetime.now().isoformat()}
"""
    SUMMARY_TXT.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"  Detections CSV:  {DET_CSV}")
    print(f"  Visual overlays: {TOP_TILES_DIR}")
    print(f"  Summary:         {SUMMARY_TXT}")
    print("\nNext: python scripts/evaluate_transfer.py")


if __name__ == "__main__":
    main()
