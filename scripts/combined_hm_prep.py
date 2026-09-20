"""Apply HM to Prieur moon images and merge with RMaM HM for Stage 2 training."""
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.exposure import match_histograms
import random, shutil

PRIEUR = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\prieur\moon_only\train\images")
OHRC = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\tiles\usable")
RMAM_HM = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\rmam\moon\train_hm_multi\images")
RMAM_LBL = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\rmam\moon\train\labels")
OUT_IMG = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\combined_hm\train\images")
OUT_LBL = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\combined_hm\train\labels")

if __name__ == '__main__':
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)
    refs = [np.array(Image.open(f).convert('L')).astype(float)
            for f in random.sample(list(OHRC.glob('*.png')), 20)]
    reference = np.mean(refs, axis=0).astype(np.uint8)
    prieur_imgs = list(PRIEUR.glob('*.png'))
    for i, f in enumerate(prieur_imgs):
        img = np.array(Image.open(f).convert('L'))
        matched = match_histograms(img, reference).astype(np.uint8)
        Image.fromarray(matched).save(OUT_IMG / ('prieur_' + f.name))
        lbl = f.parent.parent / 'labels' / (f.stem + '.txt')
        if lbl.exists():
            shutil.copy(lbl, OUT_LBL / ('prieur_' + f.stem + '.txt'))
        if i % 200 == 0:
            print(f'Prieur: {i}/{len(prieur_imgs)}')
    for f in RMAM_HM.glob('*.png'):
        shutil.copy(f, OUT_IMG / ('rmam_' + f.name))
        lbl = RMAM_LBL / (f.stem + '.txt')
        if lbl.exists():
            shutil.copy(lbl, OUT_LBL / ('rmam_' + f.stem + '.txt'))
    print('Done')
