"""
summarize_review.py  — Step 8
================================
Reads the completed visual_review.csv and prints a concise summary
of the manual feature-presence assessment.

DO NOT run this until manual review is complete.

Usage:
  python scripts/summarize_review.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
REVIEW_CSV   = PROJECT_ROOT / "results" / "quality_control" / "visual_review.csv"
MANIFEST_CSV = PROJECT_ROOT / "results" / "quality_control" / "sampling_manifest.csv"

RATING_LABELS = {
    0: "no_target",
    1: "possible_landslide",
    2: "possible_boulder_field",
    3: "possible_both",
    4: "unclear",
}

def main():
    if not REVIEW_CSV.exists():
        print(f"ERROR: {REVIEW_CSV} not found.")
        print("Complete the manual review first: python scripts/review_sample.py")
        sys.exit(1)

    manifest = pd.read_csv(MANIFEST_CSV)
    review   = pd.read_csv(REVIEW_CSV)

    n_total  = len(manifest)
    n_rated  = len(review.dropna(subset=["rating"]))
    n_skipped = n_total - n_rated

    print("=" * 55)
    print("Manual Visual Review Summary — Step 8")
    print("=" * 55)
    print(f"  Sampled tiles:    {n_total}")
    print(f"  Rated:            {n_rated}")
    print(f"  Skipped/Pending:  {n_skipped}")

    if n_rated == 0:
        print("\n  No ratings found yet. Run review_sample.py first.")
        return

    review["rating"] = review["rating"].astype(int)
    counts = review["rating"].value_counts().sort_index()

    print("\n  Feature Assessment:")
    for rating, label in RATING_LABELS.items():
        cnt = int(counts.get(rating, 0))
        pct = 100 * cnt / n_rated if n_rated > 0 else 0
        print(f"    {rating} — {label:<24}  {cnt:4d}  ({pct:5.1f}%)")

    # Per-source breakdown
    if "source_product" in review.columns:
        print("\n  Ratings by source product:")
        grp = review.groupby(["source_product", "rating"]).size().unstack(fill_value=0)
        print(grp.to_string())

    print("\n  NOTE: Counts above are preliminary human assessments.")
    print("        Do NOT use these counts as ground-truth labels.")
    print("=" * 55)

    # Save summary CSV
    summary_rows = []
    for rating, label in RATING_LABELS.items():
        cnt = int(counts.get(rating, 0))
        summary_rows.append({
            "rating_code": rating,
            "label": label,
            "count": cnt,
            "pct_of_rated": round(100 * cnt / n_rated, 2) if n_rated else 0,
        })
    import pandas as pd
    out = PROJECT_ROOT / "results" / "quality_control" / "review_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print(f"\n  Summary saved to: {out}")


if __name__ == "__main__":
    main()
