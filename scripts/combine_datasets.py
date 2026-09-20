"""Merge RMaM-2020 lunar + Prieur moon_only into combined training set."""
from pathlib import Path
import shutil

RMAM = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\rmam\moon\train")
PRIEUR = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\prieur\moon_only\train")
OUT = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\combined\train")

if __name__ == '__main__':
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / 'labels').mkdir(parents=True, exist_ok=True)
    count = 0
    for f in (RMAM / 'images').glob('*.tif'):
        shutil.copy(f, OUT / 'images' / ('rmam_' + f.stem + '.png'))
        lbl = RMAM / 'labels' / (f.stem + '.txt')
        if lbl.exists():
            shutil.copy(lbl, OUT / 'labels' / ('rmam_' + f.stem + '.txt'))
        count += 1
    for f in (PRIEUR / 'images').glob('*.png'):
        shutil.copy(f, OUT / 'images' / ('prieur_' + f.name))
        lbl = PRIEUR / 'labels' / (f.stem + '.txt')
        if lbl.exists():
            shutil.copy(lbl, OUT / 'labels' / ('prieur_' + f.stem + '.txt'))
        count += 1
    print(f'Combined: {count} total files')
