from ultralytics import YOLO
from pathlib import Path

RUNS = Path(r'D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\runs')
DATA = r'D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\rmam\dataset_hm_multi.yaml'

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
        name=f'stage2_{model_name}_hm_multi',
        device=0
    )
    print(f'{model_name} Stage 2 done')
