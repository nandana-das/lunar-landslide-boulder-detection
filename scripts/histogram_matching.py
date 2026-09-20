import numpy as np
from PIL import Image
from pathlib import Path
from skimage.exposure import match_histograms
import random

LROC_DIR = Path(r'D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\rmam\moon\train\images')
OHRC_DIR = Path(r'D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\tiles\usable')
OUT_DIR  = Path(r'D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\rmam\moon\train_hm_multi\images')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample 20 random OHRC tiles and average their histograms
ohrc_tiles = list(OHRC_DIR.glob('*.png'))
sample = random.sample(ohrc_tiles, 20)

# Build average reference image
refs = [np.array(Image.open(f).convert('L')).astype(float) for f in sample]
avg_ref = np.mean(refs, axis=0).astype(np.uint8)
reference = avg_ref

# Match all LROC train images
lroc_imgs = list(LROC_DIR.glob('*.tif'))
for i, img_path in enumerate(lroc_imgs):
    img = np.array(Image.open(img_path).convert('L'))
    matched = match_histograms(img, reference).astype(np.uint8)
    Image.fromarray(matched).save(OUT_DIR / (img_path.stem + '.png'))
    if i % 50 == 0:
        print(f'{i}/{len(lroc_imgs)} done')

print(f'Done — {len(lroc_imgs)} images matched to averaged OHRC reference (n=20)')
