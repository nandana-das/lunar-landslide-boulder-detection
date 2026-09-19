"""
YOLOv5s baseline training on RMaM-2020 lunar dataset.
Two variants: with and without histogram matching.
"""
from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).parent.parent
RUNS = BASE / "runs"
DATA_YAML = BASE / "data/rmam/dataset.yaml"
DATA_HM_YAML = BASE / "data/rmam/dataset_hm.yaml"

def train_baseline(use_hm=True):
    name = "baseline_yolov5s_hm" if use_hm else "baseline_yolov5s_noHM"
    data = DATA_HM_YAML if use_hm else DATA_YAML
    print(f"=== YOLOv5s baseline — HM={use_hm} ===")
    model = YOLO("yolov5su.pt")
    model.train(
        data=str(data),
        epochs=50,
        imgsz=640,
        batch=8,
        lr0=0.001,
        project=str(RUNS),
        name=name,
        device=0
    )
    return RUNS / name / "weights" / "best.pt"

if __name__ == "__main__":
    train_baseline(use_hm=False)
    train_baseline(use_hm=True)
    print("Baseline training complete.")
