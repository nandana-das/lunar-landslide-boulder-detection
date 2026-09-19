"""
Run trained detector on all OHRC tiles.
Saves detection images and labels only for tiles with detections.
Results saved to C:/ohrc_detections/ to avoid filling D drive.
"""
from ultralytics import YOLO
from pathlib import Path
import argparse
import csv

BASE = Path(__file__).parent.parent
TILES_DIR = BASE / "data/tiles/usable"
DEFAULT_WEIGHTS = BASE / "runs/stage2_hm/weights/best.pt"
DEFAULT_OUT = Path("C:/ohrc_detections")

def run_inference(weights, out_dir, conf=0.20):
    model = YOLO(str(weights))
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    hits = 0
    log = []

    print(f"Running inference on {TILES_DIR}")
    print(f"Weights: {weights}")
    print(f"Output: {out_dir}")

    for r in model.predict(
        source=str(TILES_DIR),
        imgsz=640,
        conf=conf,
        stream=True,
        device=0
    ):
        n = len(r.boxes)
        tile_name = Path(r.path).name
        if n > 0:
            r.save(filename=str(out_img / tile_name))
            r.save_txt(str(out_lbl / (Path(r.path).stem + ".txt")))
            total += n
            hits += 1
            for box in r.boxes:
                log.append({
                    "tile": tile_name,
                    "conf": round(float(box.conf), 4),
                    "x1": round(float(box.xyxy[0][0]), 1),
                    "y1": round(float(box.xyxy[0][1]), 1),
                    "x2": round(float(box.xyxy[0][2]), 1),
                    "y2": round(float(box.xyxy[0][3]), 1),
                })

    # Save detection log
    log_path = out_dir / "detections.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile","conf","x1","y1","x2","y2"])
        writer.writeheader()
        writer.writerows(log)

    print(f"\nTotal detections: {total}")
    print(f"Tiles with detections: {hits}")
    print(f"Detection log saved to: {log_path}")
    return total, hits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--conf", type=float, default=0.20)
    args = parser.parse_args()
    run_inference(Path(args.weights), Path(args.out), args.conf)
