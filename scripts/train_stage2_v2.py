"""Stage 2 training — all 3 models on combined HM data with HM validation."""
from ultralytics import YOLO
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA = str(BASE / "data/combined_hm/dataset_combined_hm.yaml")
RUNS = Path(str(BASE / "runs"))

if __name__ == '__main__':
    for model_name in ['yolo26n', 'yolov8n', 'yolov5s']:
        weights = RUNS / f'stage1_{model_name}_combined' / 'weights' / 'best.pt'
        print(f'Stage 2 {model_name}...')
        model = YOLO(str(weights))
        model.train(
            data=DATA,
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=0.0001,
            freeze=0,
            project=str(RUNS),
            name=f'stage2_{model_name}_combined_hm_v2',
            device=0
        )
        print(f'{model_name} Stage 2 done')
