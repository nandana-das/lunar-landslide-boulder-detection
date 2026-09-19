"""
review_priority_tiles.py
========================
Interactive visual review for the geographic pre-filtered priority tiles.
Same interface as review_sample.py — shows tiles one at a time,
accepts keyboard ratings 0-4, saves to:
  results/geofilter/priority_review.csv

Rating codes:
  0 = no obvious target feature
  1 = possible landslide
  2 = possible boulder field
  3 = possible both
  4 = unclear / needs expert review
  s = skip
  q = quit (progress saved, resumable)

Usage:
  python scripts/review_priority_tiles.py
"""
import sys
import csv
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = Path(__file__).parent.parent
PRIORITY_DIR = PROJECT_ROOT / "data" / "priority_sample"
PRIORITY_CSV = PROJECT_ROOT / "results" / "geofilter" / "priority_tiles.csv"
REVIEW_CSV   = PROJECT_ROOT / "results" / "geofilter" / "priority_review.csv"

RATING_LABELS = {
    0: "No target feature",
    1: "Possible landslide",
    2: "Possible boulder field",
    3: "Possible both",
    4: "Unclear / expert review",
}


def load_existing() -> set[str]:
    if not REVIEW_CSV.exists():
        return set()
    df = pd.read_csv(REVIEW_CSV)
    return set(df["tile_filename"].tolist())


def save_rating(row: dict, rating: int):
    exists = REVIEW_CSV.exists()
    with open(REVIEW_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["tile_filename","source_product","center_lat","center_lon",
                        "std_dev","prox_score","priority_score","rating","rating_label"])
        w.writerow([
            row["tile_filename"], row["source_product"],
            row.get("center_lat",""), row.get("center_lon",""),
            row.get("std_dev",""), row.get("prox_score",""),
            row.get("priority_score",""),
            rating, RATING_LABELS[rating]
        ])


def main():
    if not PRIORITY_CSV.exists():
        print(f"ERROR: {PRIORITY_CSV} not found.")
        print("Run geographic_prefilter.py first.")
        sys.exit(1)

    manifest = pd.read_csv(PRIORITY_CSV)
    already_done = load_existing()
    unrated = manifest[~manifest["tile_filename"].isin(already_done)]

    print(f"Priority tiles: {len(manifest)}")
    print(f"Already rated:  {len(already_done)}")
    print(f"Remaining:      {len(unrated)}")

    if unrated.empty:
        print("All priority tiles rated. See priority_review.csv")
        return

    print("\n0=no feature  1=landslide  2=boulder  3=both  4=unclear  s=skip  q=quit\n")

    rows = unrated.to_dict("records")
    state = {"idx": 0}

    try:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        fig.patch.set_facecolor("#1a1a2e")
    except Exception:
        _text_mode(rows, already_done)
        return

    def show(i):
        row = rows[i]
        fname = row["tile_filename"]
        img_path = PRIORITY_DIR / fname
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        if img_path.exists():
            ax.imshow(mpimg.imread(str(img_path)), cmap="gray", vmin=0, vmax=255)
        else:
            ax.text(0.5, 0.5, "FILE NOT FOUND", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=14)

        lat  = row.get("center_lat", "?")
        lon  = row.get("center_lon", "?")
        prio = row.get("priority_score", "?")
        prox = row.get("prox_score", "?")
        std  = row.get("std_dev", "?")
        ax.set_title(
            f"[{i+1}/{len(rows)}]  #{i+1+len(already_done)}  priority={prio}  "
            f"prox={prox}  std={std}\n"
            f"lat={lat}°  lon={lon}°\n{fname}",
            color="white", fontsize=7, pad=6
        )
        ax.axis("off")
        fig.canvas.draw()

    def on_key(event):
        i = state["idx"]
        if i >= len(rows):
            return
        row = rows[i]

        if event.key == "q":
            print("\nQuitting. Progress saved.")
            plt.close(fig)
            return
        if event.key == "s":
            print(f"  [{i+1}] SKIPPED")
            state["idx"] += 1
        elif event.key in ("0","1","2","3","4"):
            rating = int(event.key)
            save_rating(row, rating)
            print(f"  [{i+1}] {rating} — {RATING_LABELS[rating]}: {row['tile_filename']}")
            state["idx"] += 1
        else:
            return

        if state["idx"] >= len(rows):
            print("\nAll priority tiles reviewed!")
            _print_summary()
            plt.close(fig)
            return
        show(state["idx"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.text(0.5, 0.01,
             "0=no feature  1=landslide  2=boulder  3=both  4=unclear  s=skip  q=quit",
             ha="center", color="#aaaaaa", fontsize=7.5)

    show(0)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    _print_summary()


def _print_summary():
    if not REVIEW_CSV.exists():
        return
    df = pd.read_csv(REVIEW_CSV)
    n = len(df)
    if n == 0:
        return
    print(f"\nSummary: {n} tiles rated")
    counts = df["rating"].value_counts().sort_index()
    labels = {0:"no_feature", 1:"landslide", 2:"boulder", 3:"both", 4:"unclear"}
    for r, label in labels.items():
        cnt = int(counts.get(r, 0))
        print(f"  {r} {label}: {cnt} ({100*cnt/n:.1f}%)")
    positives = sum(int(counts.get(r, 0)) for r in [1, 2, 3])
    print(f"  POSITIVE rate: {positives}/{n} = {100*positives/n:.1f}%")


def _text_mode(rows, already_done):
    for row in rows:
        fname = row["tile_filename"]
        print(f"\n{fname}")
        print(f"  lat={row.get('center_lat','?')}  lon={row.get('center_lon','?')}  "
              f"priority={row.get('priority_score','?')}")
        while True:
            try:
                ans = input("  Rating (0-4/s/q): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "q"
            if ans == "q":
                return
            if ans == "s":
                break
            if ans in ("0","1","2","3","4"):
                save_rating(row, int(ans))
                break
            print("  Invalid. Enter 0-4, s, or q.")


if __name__ == "__main__":
    main()
