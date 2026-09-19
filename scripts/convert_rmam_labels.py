import csv
from PIL import Image
from pathlib import Path
from collections import defaultdict

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\rmam\moon")
# Change these 3 lines in the script:
IMG_DIR = BASE / "test_images"
LBL_CSV = BASE / "test_labels" / "test_labels_m.csv"
OUT_DIR = BASE / "test_labels_yolo"
OUT_DIR.mkdir(exist_ok=True)

labels = defaultdict(list)
with open(LBL_CSV) as f:
    for row in csv.reader(f):
        if len(row) < 6 or not all(row[:5]):
            continue
        fname, x1, y1, x2, y2, cls = row
        try:
            labels[fname].append((int(x1), int(y1), int(x2), int(y2)))
        except ValueError:
            continue

for fname, boxes in labels.items():
    img_path = IMG_DIR / fname
    if not img_path.exists():
        continue
    w, h = Image.open(img_path).size
    out_path = OUT_DIR / (Path(fname).stem + ".txt")
    with open(out_path, "w") as f:
        for x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

print(f"Done — {len(labels)} label files written to {OUT_DIR}")