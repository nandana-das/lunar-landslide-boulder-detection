"""
Stage 1 + Stage 2 YOLOv8n training on RMaM-2020 lunar dataset.
Stage 1: frozen backbone, 50 epochs
Stage 2: full fine-tune on histogram-matched data, 50 epochs
"""
from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).parent.parent
RUNS = BASE / "runs"
DATA_YAML = BASE / "data/rmam/dataset.yaml"
DATA_HM_YAML = BASE / "data/rmam/dataset_hm.yaml"

def stage1():
    print("=== Stage 1: Frozen backbone training ===")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        lr0=0.001,
        freeze=10,
        project=str(RUNS),
        name="stage1_rmam_moon",
        device=0
    )
    return RUNS / "stage1_rmam_moon" / "weights" / "best.pt"

def stage2(stage1_weights):
    print("=== Stage 2: Full fine-tune on histogram-matched data ===")
    model = YOLO(str(stage1_weights))
    model.train(
        data=str(DATA_HM_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        lr0=0.0001,
        freeze=0,
        project=str(RUNS),
        name="stage2_hm",
        device=0
    )
    return RUNS / "stage2_hm" / "weights" / "best.pt"

if __name__ == "__main__":
    w1 = stage1()
    w2 = stage2(w1)
    print(f"Training complete. Best weights: {w2}")
