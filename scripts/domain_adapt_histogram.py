"""
domain_adapt_histogram.py  — Step 2c
======================================
Histogram matching domain adaptation: makes LROC source images look like
OHRC target images before training.

This is the BASELINE domain adaptation approach (fast, no GPU training).

Method:
  For each LROC training image:
    1. Compute CDF of source image pixel values
    2. Compute CDF of target OHRC pixel distribution (from a sampled subset)
    3. Apply the CDF mapping to shift LROC pixels to match OHRC statistics

The OHRC target distribution is computed from a random sample of 500 tiles
to keep memory use low.

Output:
  data/source_domain/yolo_dataset_adapted/
    images/train/   val/   test/   (histogram-matched versions)
    labels/train/   val/   test/   (COPIED unchanged from yolo_dataset)
    dataset.yaml

Usage:
  python scripts/domain_adapt_histogram.py
"""
import sys
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
YOLO_SRC       = PROJECT_ROOT / "data" / "source_domain" / "yolo_dataset"
YOLO_ADAPTED   = PROJECT_ROOT / "data" / "source_domain" / "yolo_dataset_adapted"
OHRC_TILES_DIR = PROJECT_ROOT / "data" / "tiles" / "usable"
N_OHRC_SAMPLE  = 500   # OHRC tiles to sample for target distribution
RANDOM_SEED    = 42


def compute_cdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute CDF of pixel values 0–255."""
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 255))
    cdf = hist.cumsum().astype(np.float64)
    cdf /= cdf[-1]  # normalize to [0, 1]
    bins = np.arange(256)
    return bins, cdf


def build_ohrc_cdf(tiles_dir: Path, n_sample: int, seed: int) -> np.ndarray:
    """Sample OHRC tiles and compute aggregate CDF of pixel distribution."""
    all_tiles = sorted(tiles_dir.glob("*.png"))
    if not all_tiles:
        raise FileNotFoundError(f"No OHRC tiles found in {tiles_dir}")

    rng = random.Random(seed)
    sample = rng.sample(all_tiles, min(n_sample, len(all_tiles)))

    print(f"  Computing OHRC target CDF from {len(sample)} tile sample...")
    total_hist = np.zeros(256, dtype=np.float64)
    for tile_path in tqdm(sample, unit="tile", leave=False):
        try:
            arr = np.array(Image.open(tile_path).convert("L"), dtype=np.uint8)
            h, _ = np.histogram(arr.ravel(), bins=256, range=(0, 255))
            total_hist += h
        except Exception:
            continue

    cdf = total_hist.cumsum()
    cdf /= cdf[-1]
    return cdf   # shape (256,), normalized CDF


def histogram_match(src_arr: np.ndarray, src_cdf: np.ndarray,
                    tgt_cdf: np.ndarray) -> np.ndarray:
    """
    Map pixel values in src_arr so that src CDF → tgt CDF.
    Uses the inverse CDF method (standard histogram matching).
    """
    # Build lookup table: for each src value, find tgt value with closest CDF
    lut = np.zeros(256, dtype=np.uint8)
    for v in range(256):
        target_val = np.searchsorted(tgt_cdf, src_cdf[v])
        lut[v] = min(target_val, 255)
    return lut[src_arr]


def adapt_split(src_split_dir: Path, dst_split_dir: Path,
                ohrc_cdf: np.ndarray, split_name: str):
    """Apply histogram matching to all images in a split."""
    dst_split_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(src_split_dir.glob("*.png")) + sorted(src_split_dir.glob("*.jpg"))

    if not imgs:
        return 0

    adapted = 0
    for img_path in tqdm(imgs, desc=f"  {split_name}", unit="img"):
        dst_path = dst_split_dir / (img_path.stem + ".png")
        if dst_path.exists():
            adapted += 1
            continue
        try:
            arr = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
            _, src_cdf = compute_cdf(arr)
            adapted_arr = histogram_match(arr, src_cdf, ohrc_cdf)
            Image.fromarray(adapted_arr, mode="L").save(dst_path, format="PNG", compress_level=6)
            adapted += 1
        except Exception as e:
            print(f"  [WARN] {img_path.name}: {e}")

    return adapted


def copy_labels(src_dir: Path, dst_dir: Path):
    """Copy label .txt files unchanged."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for lbl in src_dir.glob("*.txt"):
        shutil.copy2(lbl, dst_dir / lbl.name)


def write_dataset_yaml(yolo_dir: Path, class_names: list[str]):
    content = f"""# YOLOv8 dataset — RMaM adapted to OHRC histogram
path: {yolo_dir.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    (yolo_dir / "dataset.yaml").write_text(content, encoding="utf-8")


def main():
    print("=" * 60)
    print("Domain Adaptation — Histogram Matching Baseline")
    print("=" * 60)

    if not YOLO_SRC.exists():
        print(f"\nERROR: Source YOLO dataset not found: {YOLO_SRC}")
        print("Run prepare_rmam.py first.")
        sys.exit(1)

    if not OHRC_TILES_DIR.exists() or not any(OHRC_TILES_DIR.glob("*.png")):
        print(f"\nERROR: OHRC tiles not found in {OHRC_TILES_DIR}")
        sys.exit(1)

    YOLO_ADAPTED.mkdir(parents=True, exist_ok=True)

    # Compute OHRC target CDF
    print(f"\n[1] Building OHRC target pixel distribution...")
    ohrc_cdf = build_ohrc_cdf(OHRC_TILES_DIR, N_OHRC_SAMPLE, RANDOM_SEED)

    ohrc_mean = int((ohrc_cdf > 0.5).argmax())
    print(f"  OHRC median pixel value (approx): {ohrc_mean}")

    # Check source distribution for comparison
    src_sample = list((YOLO_SRC / "images" / "train").glob("*.png"))[:50]
    if src_sample:
        src_vals = []
        for p in src_sample:
            arr = np.array(Image.open(p).convert("L"))
            src_vals.append(float(arr.mean()))
        print(f"  LROC source mean pixel (sample): {np.mean(src_vals):.1f}")

    print(f"\n[2] Applying histogram matching to all splits...")
    total = 0
    for split in ["train", "val", "test"]:
        src_img = YOLO_SRC / "images" / split
        dst_img = YOLO_ADAPTED / "images" / split
        src_lbl = YOLO_SRC / "labels" / split
        dst_lbl = YOLO_ADAPTED / "labels" / split

        if src_img.exists():
            n = adapt_split(src_img, dst_img, ohrc_cdf, split)
            total += n
            copy_labels(src_lbl, dst_lbl)

    # Read class names from source yaml
    src_yaml = YOLO_SRC / "dataset.yaml"
    class_names = ["rockfall"]
    if src_yaml.exists():
        for line in src_yaml.read_text().splitlines():
            if line.strip().startswith("names:"):
                try:
                    import ast
                    class_names = ast.literal_eval(line.split(":", 1)[1].strip())
                except Exception:
                    pass

    print(f"\n[3] Writing adapted dataset.yaml...")
    write_dataset_yaml(YOLO_ADAPTED, class_names)

    # Save the OHRC CDF for later use (CycleGAN / evaluation)
    np.save(YOLO_ADAPTED / "ohrc_cdf.npy", ohrc_cdf)
    print(f"  OHRC CDF saved to: {YOLO_ADAPTED / 'ohrc_cdf.npy'}")

    print("\n" + "=" * 60)
    print("Histogram matching complete")
    print(f"  Adapted images: {total}")
    print(f"  Dataset at: {YOLO_ADAPTED}")
    print("\nNext: python scripts/train_yolov8.py")


if __name__ == "__main__":
    main()
