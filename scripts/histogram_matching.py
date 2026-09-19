import numpy as np
from PIL import Image
from pathlib import Path
from skimage.exposure import match_histograms
import random, shutil

# Paths
LROC_DIR = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\rmam\moon\train\images")
OHRC_DIR = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\tiles\usable")
OUT_DIR  = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\rmam\moon\train_hm\images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick random OHRC reference tile
ohrc_tiles = list(OHRC_DIR.glob("*.png"))
reference = np.array(Image.open(random.choice(ohrc_tiles)).convert("L"))

# Match all LROC train images to OHRC histogram
lroc_imgs = list(LROC_DIR.glob("*.tif"))
for i, img_path in enumerate(lroc_imgs):
    img = np.array(Image.open(img_path).convert("L"))
    matched = match_histograms(img, reference)
    matched = matched.astype(np.uint8)
    Image.fromarray(matched).save(OUT_DIR / (img_path.stem + ".png"))
    if i % 50 == 0:
        print(f"{i}/{len(lroc_imgs)} done")

print(f"Done — {len(lroc_imgs)} images histogram matched to OHRC style")
