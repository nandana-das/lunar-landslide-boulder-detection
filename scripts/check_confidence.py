"""Analyze detection confidence distribution on OHRC tiles."""
from ultralytics import YOLO
from pathlib import Path
import statistics

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
TILES = str(BASE / "data/tiles/usable")
RUNS = Path(str(BASE / "runs"))

MODELS = {
    'yolo26n': RUNS / 'stage2_yolo26n_combined_hm_v2' / 'weights' / 'best.pt',
    'yolov5s': RUNS / 'stage2_yolov5s_combined_hm_v2' / 'weights' / 'best.pt',
}

if __name__ == '__main__':
    for model_name, weights in MODELS.items():
        model = YOLO(str(weights))
        confs = [float(box.conf)
                 for r in model.predict(source=TILES, imgsz=640,
                                        conf=0.20, stream=True, device=0)
                 for box in r.boxes]
        if confs:
            print(f'{model_name}: N={len(confs)} '
                  f'Mean={round(statistics.mean(confs),3)} '
                  f'Median={round(statistics.median(confs),3)} '
                  f'<0.30={sum(1 for c in confs if c<0.30)} '
                  f'>=0.30={sum(1 for c in confs if c>=0.30)}')
