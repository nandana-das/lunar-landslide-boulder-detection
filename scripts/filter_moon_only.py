"""Filter Moon-only images from Prieur et al. 2023 boulder dataset."""
from pathlib import Path
import shutil

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\prieur\boulder2024")
MOON_OUT = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\prieur\moon_only")

MOON_PREFIXES = ('M', 'NAC')

if __name__ == '__main__':
    for split in ['train', 'validation', 'test']:
        img_out = MOON_OUT / split / 'images'
        lbl_out = MOON_OUT / split / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        count = 0
        for img in (BASE / split / 'images').glob('*.png'):
            prefix = img.name.split('_')[0]
            if any(prefix.startswith(p) for p in MOON_PREFIXES):
                lbl = BASE / split / 'labels' / (img.stem + '.txt')
                shutil.copy(img, img_out / img.name)
                if lbl.exists():
                    shutil.copy(lbl, lbl_out / lbl.name)
                count += 1
        print(f'{split}: {count} moon images copied')
