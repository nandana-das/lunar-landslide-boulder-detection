"""
prepare_rmam.py  — Step 2a
===========================
Converts the RMaM-2020 (Rockfall Mars Moon 2020) dataset to YOLOv8 format.

Expected input layout after manual download from EDMOND:
  data/source_domain/rmam2020/
    images/          *.png or *.jpg — LROC NAC / HiRISE image patches
    labels.csv       Columns: image_id, x_min, y_min, x_max, y_max, class

  OR (alternative EDMOND layout):
    *.png            Images at root
    *.csv            One CSV per image OR a single master labels file

The script auto-detects the layout and handles both.

Output:
  data/source_domain/yolo_dataset/
    images/
      train/   val/   test/
    labels/
      train/   val/   test/
    dataset.yaml

Splits: 70% train / 15% val / 15% test  (reproducible, seed=42)

Class mapping:
  0 = rockfall / boulder

Usage:
  python scripts/prepare_rmam.py [--rmam-dir PATH]
"""
import sys
import csv
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
RMAM_DIR     = PROJECT_ROOT / "data" / "source_domain" / "rmam2020"
YOLO_DIR     = PROJECT_ROOT / "data" / "source_domain" / "yolo_dataset"
RANDOM_SEED  = 42
SPLIT        = (0.70, 0.15, 0.15)  # train, val, test
CLASS_NAMES  = ["rockfall"]
TARGET_SIZE  = 640   # resize images to this (square)


def find_rmam_layout(rmam_dir: Path) -> str:
    """Detect whether EDMOND download has images/ subfolder or flat layout."""
    if (rmam_dir / "images").is_dir():
        return "subdir"
    pngs = list(rmam_dir.glob("*.png")) + list(rmam_dir.glob("*.jpg"))
    if pngs:
        return "flat"
    return "unknown"


def load_labels(rmam_dir: Path) -> dict[str, list[tuple]]:
    """
    Load all annotations.
    Returns dict: image_stem -> [(x_min, y_min, x_max, y_max), ...]
    Tries multiple CSV formats used by EDMOND releases.
    """
    csvs = list(rmam_dir.glob("*.csv")) + list(rmam_dir.glob("labels/*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV label files found in {rmam_dir}.\n"
            "Expected: a labels.csv or per-image CSV files.\n"
            "Please verify your EDMOND download is complete."
        )

    labels = defaultdict(list)

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip().lower() for c in df.columns]
        except Exception as e:
            print(f"  [WARN] Could not read {csv_path.name}: {e}")
            continue

        # Try to identify columns
        possible_id_cols   = ["image_id", "image", "file", "filename", "img_id"]
        possible_xmin_cols = ["x_min", "xmin", "x1", "left"]
        possible_ymin_cols = ["y_min", "ymin", "y1", "top"]
        possible_xmax_cols = ["x_max", "xmax", "x2", "right"]
        possible_ymax_cols = ["y_max", "ymax", "y2", "bottom"]

        def pick(cols, options):
            for o in options:
                if o in cols:
                    return o
            return None

        col_id   = pick(df.columns, possible_id_cols)
        col_xmin = pick(df.columns, possible_xmin_cols)
        col_ymin = pick(df.columns, possible_ymin_cols)
        col_xmax = pick(df.columns, possible_xmax_cols)
        col_ymax = pick(df.columns, possible_ymax_cols)

        if not all([col_xmin, col_ymin, col_xmax, col_ymax]):
            print(f"  [WARN] {csv_path.name}: could not identify bbox columns. "
                  f"Found: {list(df.columns)}")
            continue

        for _, row in df.iterrows():
            if col_id:
                img_id = str(row[col_id]).strip()
                # Strip extension if present
                if img_id.endswith((".png", ".jpg", ".jpeg")):
                    img_id = Path(img_id).stem
            else:
                img_id = csv_path.stem  # per-image CSV named after image

            try:
                box = (float(row[col_xmin]), float(row[col_ymin]),
                       float(row[col_xmax]), float(row[col_ymax]))
                labels[img_id].append(box)
            except (ValueError, KeyError):
                continue

    return dict(labels)


def pixel_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert pixel bbox to YOLO format (normalized cx, cy, w, h)."""
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.001, min(1.0, w))
    h  = max(0.001, min(1.0, h))
    return cx, cy, w, h


def build_yolo_dataset(rmam_dir: Path, labels: dict, yolo_dir: Path):
    """Copy/resize images and write YOLO label .txt files."""
    # Find all images
    all_imgs = (
        list(rmam_dir.glob("*.png")) +
        list(rmam_dir.glob("*.jpg")) +
        list(rmam_dir.glob("images/*.png")) +
        list(rmam_dir.glob("images/*.jpg"))
    )
    all_imgs = [p for p in all_imgs if p.stem in labels]

    if not all_imgs:
        # Try matching without exact stem (EDMOND sometimes adds suffixes)
        all_imgs_all = (
            list(rmam_dir.glob("*.png")) +
            list(rmam_dir.glob("*.jpg")) +
            list(rmam_dir.glob("images/*.png")) +
            list(rmam_dir.glob("images/*.jpg"))
        )
        # Fuzzy match: find images whose stem appears in any label key
        label_keys = set(labels.keys())
        all_imgs = [p for p in all_imgs_all
                    if any(k in p.stem or p.stem in k for k in label_keys)]

    if not all_imgs:
        raise RuntimeError(
            f"No labeled images found in {rmam_dir}.\n"
            f"Labels loaded for {len(labels)} image IDs.\n"
            f"Images found at root/images: {len(list(rmam_dir.glob('**/*.png')))}\n"
            "Check that image filenames match the CSV image_id values."
        )

    print(f"  Images with labels: {len(all_imgs)}")

    # Reproducible split
    random.seed(RANDOM_SEED)
    random.shuffle(all_imgs)
    n = len(all_imgs)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    splits = {
        "train": all_imgs[:n_train],
        "val":   all_imgs[n_train:n_train + n_val],
        "test":  all_imgs[n_train + n_val:],
    }

    # Create output dirs
    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0, "skipped": 0, "total_boxes": 0}

    for split, imgs in splits.items():
        for img_path in tqdm(imgs, desc=f"  {split}", unit="img"):
            stem = img_path.stem

            # Find matching label key
            label_key = stem
            if stem not in labels:
                # Try fuzzy
                matches = [k for k in labels if k in stem or stem in k]
                if not matches:
                    stats["skipped"] += 1
                    continue
                label_key = matches[0]

            boxes = labels[label_key]
            if not boxes:
                stats["skipped"] += 1
                continue

            # Load and resize image
            try:
                img = Image.open(img_path).convert("L")  # grayscale like OHRC
                orig_w, orig_h = img.size
                img_resized = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            except Exception as e:
                print(f"  [WARN] Cannot open {img_path.name}: {e}")
                stats["skipped"] += 1
                continue

            # Save image
            dst_img = yolo_dir / "images" / split / f"{stem}.png"
            img_resized.save(dst_img, format="PNG", compress_level=6)

            # Write YOLO label
            dst_lbl = yolo_dir / "labels" / split / f"{stem}.txt"
            with open(dst_lbl, "w") as f:
                for (xmin, ymin, xmax, ymax) in boxes:
                    # Scale boxes if image was resized
                    scale_x = TARGET_SIZE / orig_w
                    scale_y = TARGET_SIZE / orig_h
                    xmin_s = xmin * scale_x
                    ymin_s = ymin * scale_y
                    xmax_s = xmax * scale_x
                    ymax_s = ymax * scale_y
                    cx, cy, w, h = pixel_to_yolo(
                        xmin_s, ymin_s, xmax_s, ymax_s, TARGET_SIZE, TARGET_SIZE
                    )
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    stats["total_boxes"] += 1

            stats[split] += 1

    return stats


def write_dataset_yaml(yolo_dir: Path):
    yaml_content = f"""# YOLOv8 dataset config — RMaM-2020 (lunar rockfall)
path: {yolo_dir.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (yolo_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmam-dir", type=Path, default=RMAM_DIR)
    args = parser.parse_args()

    rmam_dir = args.rmam_dir

    print("=" * 60)
    print("Preparing RMaM-2020 → YOLOv8 format")
    print("=" * 60)

    if not rmam_dir.exists():
        print(f"\nERROR: RMaM directory not found: {rmam_dir}")
        print("Please download RMaM-2020 from EDMOND first:")
        print("  https://edmond.mpdl.mpg.de/imeji/collection/DowTY91csU3jv9S2")
        print(f"  Extract into: {rmam_dir}")
        sys.exit(1)

    layout = find_rmam_layout(rmam_dir)
    print(f"\n  Detected EDMOND layout: '{layout}'")

    print("\n[1] Loading label files...")
    labels = load_labels(rmam_dir)
    total_boxes = sum(len(v) for v in labels.values())
    print(f"  Labeled images: {len(labels)}")
    print(f"  Total bounding boxes: {total_boxes}")

    print(f"\n[2] Building YOLOv8 dataset in: {YOLO_DIR}")
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    stats = build_yolo_dataset(rmam_dir, labels, YOLO_DIR)

    print(f"\n[3] Writing dataset.yaml...")
    write_dataset_yaml(YOLO_DIR)

    print("\n" + "=" * 60)
    print("RMaM-2020 conversion complete")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")
    print(f"  Test:  {stats['test']} images")
    print(f"  Skipped (no labels/unreadable): {stats['skipped']}")
    print(f"  Total bounding boxes written: {stats['total_boxes']}")
    print(f"\n  Dataset at: {YOLO_DIR}")
    print(f"  Config:     {YOLO_DIR / 'dataset.yaml'}")
    print("\nNext: python scripts/domain_adapt_histogram.py")


if __name__ == "__main__":
    main()
