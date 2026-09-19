"""
review_sample.py  — Step 7
============================
Interactive visual review tool for the 200 sampled tiles.
Displays tiles one at a time using matplotlib and lets you rate each:

  0 = no obvious target feature
  1 = possible landslide
  2 = possible boulder field
  3 = possible both
  4 = unclear / needs expert review
  s = skip (leave unrated for now)
  q = quit and save progress

Ratings are saved incrementally to:
  results/quality_control/visual_review.csv

You can rerun the script to continue where you left off.
Already-rated tiles are skipped automatically.

NOTE: This is a human QC tool only.
      No automatic annotation or AI classification is performed.

Usage:
  python scripts/review_sample.py
"""
import sys
import csv
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_DIR   = PROJECT_ROOT / "data" / "annotation_sample"
MANIFEST_CSV = PROJECT_ROOT / "results" / "quality_control" / "sampling_manifest.csv"
REVIEW_CSV   = PROJECT_ROOT / "results" / "quality_control" / "visual_review.csv"

RATING_LABELS = {
    0: "No target feature",
    1: "Possible landslide",
    2: "Possible boulder field",
    3: "Possible both",
    4: "Unclear / expert review",
}

RATING_COLORS = {
    0: "#444444",
    1: "#c84b00",
    2: "#0055aa",
    3: "#007744",
    4: "#886600",
}


def load_existing_ratings() -> dict[str, int]:
    """Load previously saved ratings keyed by tile_filename."""
    if not REVIEW_CSV.exists():
        return {}
    df = pd.read_csv(REVIEW_CSV)
    if "rating" not in df.columns:
        return {}
    # Only keep rows with a numeric rating
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    return dict(zip(df["tile_filename"], df["rating"]))


def save_rating(tile_filename: str, sample_id: int, source_product: str, rating: int):
    """Append a single rating to the CSV."""
    file_exists = REVIEW_CSV.exists()
    with open(REVIEW_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["sample_id", "tile_filename", "source_product", "rating", "rating_label"])
        writer.writerow([sample_id, tile_filename, source_product,
                         rating, RATING_LABELS[rating]])


def run_review():
    if not MANIFEST_CSV.exists():
        print(f"ERROR: {MANIFEST_CSV} not found. Run sample_dataset.py first.")
        sys.exit(1)

    manifest = pd.read_csv(MANIFEST_CSV)
    existing = load_existing_ratings()

    unrated = manifest[~manifest["tile_filename"].isin(existing.keys())]
    n_total = len(manifest)
    n_done  = len(existing)

    print(f"Tiles total:   {n_total}")
    print(f"Already rated: {n_done}")
    print(f"Remaining:     {len(unrated)}")

    if unrated.empty:
        print("All tiles are rated. See visual_review.csv")
        return

    print("\nRating codes:")
    for k, v in RATING_LABELS.items():
        print(f"  {k} = {v}")
    print("\nKeyboard shortcuts: 0-4 to rate | s = skip | q = quit")
    print("Ratings are saved automatically after each key press.\n")

    # Use non-interactive backend if no display is available
    try:
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))
        fig.patch.set_facecolor("#1e1e1e")
    except Exception:
        print("No display available — using text-mode fallback.")
        _text_mode_review(manifest, existing)
        return

    state = {"idx": 0, "done": False}
    rows_to_review = unrated.to_dict("records")

    def show_tile(i):
        row = rows_to_review[i]
        fname = row["tile_filename"]
        img_path = SAMPLE_DIR / fname
        axes.clear()
        axes.set_facecolor("#1e1e1e")
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            axes.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            axes.text(0.5, 0.5, "FILE NOT FOUND", transform=axes.transAxes,
                      ha="center", va="center", color="red", fontsize=14)
        total_left = len(rows_to_review) - i
        axes.set_title(
            f"[{i+1}/{len(rows_to_review)}] Sample #{row['sample_id']}  "
            f"({n_done + i + 1}/{n_total} total)\n{fname}",
            color="white", fontsize=7, pad=6
        )
        axes.axis("off")
        fig.canvas.draw()

    def on_key(event):
        i = state["idx"]
        if i >= len(rows_to_review):
            return
        row = rows_to_review[i]

        if event.key == "q":
            print("\nQuitting. Progress saved.")
            state["done"] = True
            plt.close(fig)
            return

        if event.key == "s":
            print(f"  [{i+1}] SKIPPED: {row['tile_filename']}")
            state["idx"] += 1
        elif event.key in ("0", "1", "2", "3", "4"):
            rating = int(event.key)
            save_rating(row["tile_filename"], row["sample_id"], row["source_product"], rating)
            label = RATING_LABELS[rating]
            color = RATING_COLORS[rating]
            print(f"  [{i+1}] {rating} — {label}: {row['tile_filename']}")
            state["idx"] += 1
        else:
            return

        if state["idx"] >= len(rows_to_review):
            print("\nAll remaining tiles reviewed! Session complete.")
            plt.close(fig)
            return

        show_tile(state["idx"])

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Legend
    legend_text = "Keys: 0=no feature  1=landslide  2=boulder  3=both  4=unclear  s=skip  q=quit"
    fig.text(0.5, 0.01, legend_text, ha="center", color="#aaaaaa", fontsize=7.5)

    show_tile(0)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

    # Summary on exit
    if REVIEW_CSV.exists():
        df = pd.read_csv(REVIEW_CSV)
        print(f"\nSession ended. Total rated so far: {len(df)}/{n_total}")


def _text_mode_review(manifest, existing):
    """Fallback for headless environments — prompt user on CLI."""
    unrated = manifest[~manifest["tile_filename"].isin(existing.keys())]
    print(f"\n[Text mode] {len(unrated)} tiles to review.")
    for _, row in unrated.iterrows():
        fname = row["tile_filename"]
        print(f"\nTile #{row['sample_id']}: {fname}")
        print(f"Image path: {SAMPLE_DIR / fname}")
        while True:
            try:
                ans = input("Rating (0-4) or s=skip or q=quit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "q"
            if ans == "q":
                print("Quitting.")
                return
            if ans == "s":
                break
            if ans in ("0", "1", "2", "3", "4"):
                save_rating(fname, int(row["sample_id"]), row["source_product"], int(ans))
                print(f"  Saved: {ans} = {RATING_LABELS[int(ans)]}")
                break
            print("  Invalid input. Enter 0-4, s, or q.")


if __name__ == "__main__":
    run_review()
