"""
Create YOLO dataset YAML files for RMaM-2020 moon subset.
Run once after convert_rmam_labels.py.
"""
from pathlib import Path

BASE = Path(__file__).parent.parent
RMAM_DIR = BASE / "data/rmam/moon"

def write_yaml(path, train_dir, val_dir):
    content = f"""path: {RMAM_DIR.as_posix()}
train: {train_dir}
val: {val_dir}
nc: 1
names: [rockfall]
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")

if __name__ == "__main__":
    write_yaml(BASE / "data/rmam/dataset.yaml", "train/images", "val/images")
    write_yaml(BASE / "data/rmam/dataset_hm.yaml", "train_hm/images", "val/images")
    print("Done.")
