"""Stage 1 training — all 3 models on combined RMaM+Prieur dataset."""
from ultralytics import YOLO
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA = str(BASE / "data/combined/dataset_combined.yaml")
RUNS = str(BASE / "runs")

if __name__ == '__main__':
    for model_name, weights in [
        ('yolo26n', 'yolo26n.pt'),
        ('yolov8n', 'yolov8n.pt'),
        ('yolov5s', 'yolov5su.pt'),
    ]:
        print(f'Training {model_name} Stage 1...')
        model = YOLO(weights)
        model.train(
            data=DATA,
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=0.001,
            freeze=10,
            project=RUNS,
            name=f'stage1_{model_name}_combined',
            device=0
        )
        print(f'{model_name} Stage 1 done')
