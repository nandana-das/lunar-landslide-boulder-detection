"""
train_yolov8.py  — Step 3
===========================
Trains YOLOv8n on the histogram-matched LROC source domain dataset.

Model: YOLOv8n (nano) — 3.2M parameters
  - Fits in 4GB VRAM (RTX 3050)
  - Pretrained on COCO for strong edge/shape features

Stage 1: Source-domain training on histogram-matched LROC images
  Output: models/yolov8n_lroc_adapted.pt

Hyperparameters tuned for:
  - Small dataset (~2,800 RMaM images)
  - 4GB VRAM constraint (batch=8, imgsz=640)
  - Greyscale input (single-channel → replicated to 3ch for COCO pretrain)

Usage:
  python scripts/train_yolov8.py [--epochs N] [--batch B] [--resume]
"""
import sys
import argparse
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
ADAPTED_YAML  = PROJECT_ROOT / "data" / "source_domain" / "yolo_dataset_adapted" / "dataset.yaml"
MODELS_DIR    = PROJECT_ROOT / "models"
RUNS_DIR      = PROJECT_ROOT / "results" / "training_runs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=8,
                        help="Batch size. Reduce to 4 if OOM on 4GB VRAM")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint if run exists")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader workers (keep low on Windows)")
    args = parser.parse_args()

    # ── Import here so error messages are clean before ultralytics loads ──
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("Run: pip install ultralytics")
        sys.exit(1)

    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("\n[WARN] CUDA not available — training on CPU (will be slow)")
        print("  Verify CUDA/cuDNN installation if RTX 3050 is present")

    print("=" * 60)
    print("YOLOv8n Training — LROC Source Domain (Adapted)")
    print("=" * 60)
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  Dataset: {ADAPTED_YAML}")

    if not ADAPTED_YAML.exists():
        print(f"\nERROR: Dataset YAML not found: {ADAPTED_YAML}")
        print("Run domain_adapt_histogram.py first.")
        sys.exit(1)

    # Load model
    if args.resume:
        # Look for last.pt in most recent run
        run_dirs = sorted(RUNS_DIR.glob("train*/weights/last.pt"))
        if run_dirs:
            last_ckpt = run_dirs[-1]
            print(f"\n  Resuming from: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print("\n  No checkpoint found for resume — starting fresh")
            model = YOLO("yolov8n.pt")
    else:
        print("\n  Starting from COCO pretrained yolov8n.pt")
        model = YOLO("yolov8n.pt")

    # ── Training hyperparameters ──────────────────────────────────────────
    # Tuned for:
    #   - Small grayscale dataset (3-channel via repeat)
    #   - RTX 3050 4GB VRAM
    #   - Good transfer to planetary imagery
    results = model.train(
        data=str(ADAPTED_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(RUNS_DIR),
        name="lroc_adapted",
        exist_ok=True,

        # Optimizer
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,      # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,

        # Augmentation — aggressive for small dataset
        hsv_h=0.0,     # No hue shift (grayscale)
        hsv_s=0.0,     # No saturation (grayscale)
        hsv_v=0.4,     # Brightness variation (simulate illumination change)
        flipud=0.5,    # Vertical flip (no "up" in space)
        fliplr=0.5,    # Horizontal flip
        mosaic=1.0,    # Mosaic augmentation (4-image composite)
        mixup=0.1,     # MixUp
        degrees=180.0, # Rotation (boulders have no fixed orientation)
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0,
        copy_paste=0.0,

        # Loss weights (tuned for small objects like boulders)
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Early stopping
        patience=20,

        # Save settings
        save=True,
        save_period=10,   # Save checkpoint every 10 epochs
        val=True,

        # Misc
        verbose=True,
        seed=42,
        deterministic=True,
    )

    print("\n" + "=" * 60)
    print("Training complete")

    # Save best model to models/
    best_pt = RUNS_DIR / "lroc_adapted" / "weights" / "best.pt"
    if best_pt.exists():
        dst = MODELS_DIR / "yolov8n_lroc_adapted.pt"
        import shutil
        shutil.copy2(best_pt, dst)
        print(f"  Best model saved to: {dst}")
    else:
        print(f"  [WARN] best.pt not found at {best_pt}")

    print(f"\n  Training run at: {RUNS_DIR / 'lroc_adapted'}")
    print(f"  Metrics:         {RUNS_DIR / 'lroc_adapted' / 'results.csv'}")
    print("\nNext: python scripts/run_inference_ohrc.py")


if __name__ == "__main__":
    main()
