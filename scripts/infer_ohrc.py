"""Run inference on all OHRC tiles — saves detection images to C drive."""
from ultralytics import YOLO
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
TILES = str(BASE / "data/tiles/usable")
RUNS = Path(str(BASE / "runs"))

MODELS = {
    'yolo26n': RUNS / 'stage2_yolo26n_combined_hm_v2' / 'weights' / 'best.pt',
    'yolov8n': RUNS / 'stage2_yolov8n_combined_hm_v2' / 'weights' / 'best.pt',
    'yolov5s': RUNS / 'stage2_yolov5s_combined_hm_v2' / 'weights' / 'best.pt',
}

if __name__ == '__main__':
    for model_name, weights in MODELS.items():
        out_img = Path(f'C:/ohrc_detections/{model_name}/images')
        out_lbl = Path(f'C:/ohrc_detections/{model_name}/labels')
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(weights))
        total = 0
        hits = 0
        for r in model.predict(source=TILES, imgsz=640, conf=0.20,
                               stream=True, device=0):
            if len(r.boxes) > 0:
                r.save(filename=str(out_img / Path(r.path).name))
                r.save_txt(str(out_lbl / (Path(r.path).stem + '.txt')))
                total += len(r.boxes)
                hits += 1
        print(f'{model_name}: detections={total} tiles={hits}')
