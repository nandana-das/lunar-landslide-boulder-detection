"""Convert Prieur et al. 2023 YOLO polygon labels to bounding boxes."""
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\prieur\boulder2024")

if __name__ == '__main__':
    for split in ['train', 'validation', 'test']:
        src = BASE / split / 'labels_seg'
        dst = BASE / split / 'labels'
        dst.mkdir(exist_ok=True)
        count = 0
        for f in src.glob('*.txt'):
            lines_out = []
            for line in f.read_text().strip().split('\n'):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                coords = list(map(float, parts[1:]))
                xs = coords[0::2]
                ys = coords[1::2]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                lines_out.append(f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
            if lines_out:
                (dst / f.name).write_text('\n'.join(lines_out))
                count += 1
        print(f'{split}: {count} files converted')
