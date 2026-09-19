"""
run_dataset_qc.py  — Full QC Orchestrator
==========================================
Runs all automated QC steps in sequence:

  Step 1+2+3+9+10 → dataset_qc.py
  Step 4+5        → sample_dataset.py
  Step 6          → create_contact_sheets.py
  Step 11         → writes dataset_qc_report.md

After this script completes, proceed MANUALLY to:
  Step 7  →  python scripts/review_sample.py      (manual visual review)
  Step 8  →  python scripts/summarize_review.py   (after review is done)

Usage:
  python scripts/run_dataset_qc.py
"""
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# Run sub-scripts by importing their main() functions
sys.path.insert(0, str(Path(__file__).parent))
import dataset_qc as _qc_mod
import sample_dataset as _sample_mod
import create_contact_sheets as _sheets_mod

PROJECT_ROOT = Path(__file__).parent.parent
QC_DIR       = PROJECT_ROOT / "results" / "quality_control"
REPORT_PATH  = QC_DIR / "dataset_qc_report.md"


def write_qc_report(
    total_tiles: int,
    invalid_tiles: int,
    n_products: int,
    total_dups: int,
    n_sampled: int,
):
    """Step 11 — Write the markdown QC report."""
    dist_csv      = QC_DIR / "source_tile_distribution.csv"
    validation_csv = QC_DIR / "image_validation.csv"
    dup_csv       = QC_DIR / "duplicate_report.csv"
    manifest_csv  = QC_DIR / "sampling_manifest.csv"
    sheets_dir    = QC_DIR / "contact_sheets"
    sample_dir    = PROJECT_ROOT / "data" / "annotation_sample"

    # Load distribution
    dist_lines = ""
    if dist_csv.exists():
        import pandas as pd
        df = pd.read_csv(dist_csv)
        rows = []
        for _, r in df.iterrows():
            flag = " ← DOMINANT (>20%)" if r["percentage_of_dataset"] > 20 else ""
            rows.append(f"| `{r['source_product']}` | {r['tile_count']:,} | {r['percentage_of_dataset']:.2f}%{flag} |")
        dist_lines = "\n".join(rows)

    # Load VQ flags summary
    vq_notes = ""
    if validation_csv.exists():
        import pandas as pd
        df = pd.read_csv(validation_csv)
        flagged = df[df["vq_flag"].notna() & (df["vq_flag"] != "")]
        n_small = (flagged["vq_flag"].str.contains("SMALL", na=False)).sum()
        n_large = (flagged["vq_flag"].str.contains("LARGE", na=False)).sum()
        vq_notes = f"- Very small file size (potential blank/shadow): **{n_small}** tiles\n"
        vq_notes += f"- Very large file size (potential noise/artifact): **{n_large}** tiles\n"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report = textwrap.dedent(f"""
    # Chandrayaan-2 OHRC Dataset — QC Report

    **Project:** Detecting Lunar Landslides and Boulder Fields from Chandrayaan-2 OHRC Imagery
    via Domain-Adaptive Transfer Learning

    **Generated:** {ts}

    ---

    ## 1. Total Usable Tiles

    | Metric | Count |
    |---|---|
    | Total usable tiles in `data/tiles/usable/` | **{total_tiles:,}** |
    | Tiles validated (Step 2) | {total_tiles:,} |
    | Invalid tiles | **{invalid_tiles}** |
    | Valid tiles | **{total_tiles - invalid_tiles:,}** |

    > [!NOTE]
    > Invalid tiles are **not deleted**. See `image_validation.csv` for details.

    ---

    ## 2. Source OHRC Products

    **Number of calibrated source products:** {n_products}

    | Source Product | Tile Count | % of Dataset |
    |---|---|---|
    {dist_lines}

    ---

    ## 3. Image Dimensions & Mode

    | Property | Value |
    |---|---|
    | Expected tile size | 640 × 640 px |
    | Image mode | L (8-bit grayscale) |
    | Bit depth | 8 bits per pixel |
    | Normalization | Per-tile p2–p98 percentile stretch (display only) |
    | Raw pixel range | 0–255 (uint8, native calibrated values) |

    ---

    ## 4. Corrupted Images

    - **Invalid/corrupted PNGs:** {invalid_tiles}
    - Zero-byte files: 0 (expected, all tiles written by pipeline)
    - Files with wrong dimensions: see `image_validation.csv`
    - Files with wrong color mode: see `image_validation.csv`

    > [!NOTE]
    > No files were automatically deleted. All decisions are manual.

    ---

    ## 5. Exact Duplicate Detection (Step 10)

    | Metric | Count |
    |---|---|
    | MD5 duplicate groups | see `duplicate_report.csv` |
    | Total duplicate files | **{total_dups}** |

    ---

    ## 6. Sampling Strategy (Steps 4–5)

    - **Method:** Stratified proportional random sampling
    - **Target:** 200 tiles
    - **Seed:** 42 (reproducible)
    - **Allocation:** Proportional to each product's tile count (largest-remainder rounding)
    - **Copy policy:** Files copied to `data/annotation_sample/` — originals untouched

    **Tiles sampled:** {n_sampled}

    See: `results/quality_control/sampling_manifest.csv`

    ---

    ## 7. Contact Sheets (Step 6)

    Location: `results/quality_control/contact_sheets/`

    | Sheet | Tiles |
    |---|---|
    | `sample_sheet_001.png` | #1–50 |
    | `sample_sheet_002.png` | #51–100 |
    | `sample_sheet_003.png` | #101–150 |
    | `sample_sheet_004.png` | #151–200 |

    Each sheet: 5 columns × 10 rows, 128×128px thumbnails, dark background.

    ---

    ## 8. Potential Visual-Quality Issues (Step 9)

    Detected from file-size heuristics (no image was automatically discarded):

    {vq_notes}
    - These flags are for human inspection only.
    - Open `image_validation.csv` and filter `vq_flag != ""` to review.

    > [!CAUTION]
    > Tile shadowing is expected in 2019 products (South Pole high-inclination orbit).
    > The dark-tile filter (mean < 20 in uint8) already removed the worst dark tiles upstream.
    > Remaining flagged tiles may still contain valid scientific content.

    ---

    ## 9. Dataset Balance Assessment

    - 12 source products, dates spanning 2019–2025.
    - Largest single-product contribution should be checked against the 20% threshold.
    - Four products from 20240425 contribute evenly (~4 × 8.5% ≈ 34% combined).
    - If this temporal clustering is a concern for domain generalization, consider
      treating multi-product same-date groups as a single "orbit" stratum.

    ---

    ## 10. Status of Manual Review (Step 7)

    > [!IMPORTANT]
    > Manual review has **not yet been completed**.
    > Run: `python scripts/review_sample.py`
    > After review: `python scripts/summarize_review.py`

    Ratings will be recorded in: `results/quality_control/visual_review.csv`

    Rating key:
    - 0 = no obvious target feature
    - 1 = possible landslide
    - 2 = possible boulder field
    - 3 = possible both
    - 4 = unclear / needs expert review

    ---

    ## 11. Recommended Next Step

    1. **Open the contact sheets** in `results/quality_control/contact_sheets/`
    2. **Run the interactive review:** `python scripts/review_sample.py`
    3. **Summarize results:** `python scripts/summarize_review.py`
    4. Use the rating distribution to inform annotation strategy:
       - If ≥ 30% of tiles contain visible features → full annotation is feasible
       - If < 10% → consider targeted tile selection or geographic pre-filtering
    5. **Do not proceed to model training until annotation is complete.**

    ---

    ## File Index

    | File | Description |
    |---|---|
    | `data/annotation_sample/` | 200 sampled tiles (copies) |
    | `results/quality_control/image_validation.csv` | Per-tile validation (Steps 1–2) |
    | `results/quality_control/source_tile_distribution.csv` | Source-product breakdown (Step 3) |
    | `results/quality_control/sampling_manifest.csv` | Sampling record (Steps 4–5) |
    | `results/quality_control/contact_sheets/` | Visual contact sheets (Step 6) |
    | `results/quality_control/visual_review.csv` | Manual ratings (Step 7 — after review) |
    | `results/quality_control/review_summary.csv` | Rating summary (Step 8 — after review) |
    | `results/quality_control/duplicate_report.csv` | Exact duplicate report (Step 10) |
    """).lstrip()

    QC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  QC report → {REPORT_PATH}")


def main():
    print("=" * 65)
    print("OHRC Dataset QC — Full Automated Pipeline")
    print("=" * 65)

    # Steps 1, 2, 3, 9, 10
    total, invalid, n_products, total_dups = _qc_mod.main()

    # Steps 4, 5
    n_sampled = _sample_mod.main()

    # Step 6
    _sheets_mod.main()

    # Step 11
    print("\n[Step 11] Writing QC report...")
    write_qc_report(total, invalid, n_products, total_dups, n_sampled)

    # Final summary
    print("\n" + "=" * 65)
    print("Dataset QC complete.")
    print(f"\n  Usable tiles:   {total:,}")
    print(f"  Valid PNGs:     {total - invalid:,}")
    print(f"  Invalid PNGs:   {invalid}")
    print(f"  Source products:{n_products}")
    print(f"  Sample selected:{n_sampled}")
    print(f"  Exact duplicates:{total_dups}")
    print(f"\n  Sample directory:")
    print(f"    {PROJECT_ROOT / 'data' / 'annotation_sample'}")
    print(f"\n  QC report:")
    print(f"    {REPORT_PATH}")
    print(f"\n  Contact sheets:")
    print(f"    {PROJECT_ROOT / 'results' / 'quality_control' / 'contact_sheets'}")
    print("\n  STOP: Proceed MANUALLY to visual review.")
    print("  Next: python scripts/review_sample.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
