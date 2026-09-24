r"""
process_polar_batch.py
======================
Automated pipeline for the 20 downloaded ultra-deep polar calibrated OHRC products (lat <= -89.5°S):
1. Process 20 .zip products from D:\Users\NANS\Downloads\.
2. Extract data/calibrated/ .xml (PDS4 label) and .img (calibrated radiance image).
3. Tile into 640x640 non-overlapping tiles, discarding dark/dropout tiles (native mean < 20),
   applying identical percentile stretch (p2-p98 -> uint8) as original Module I.
4. Clean up extracted .img immediately after tiling each product to conserve disk space.
5. Run Stage 3 inference using trained Stage-2 weights:
   - YOLO26n (stage2_yolo26n_combined_hm_v2)
   - YOLOv8n (stage2_yolov8n_combined_hm_v2)
   - YOLOv5s (stage2_yolov5s_combined_hm_v2)
   - RT-DETR-L (stage2_rtdetr_l_combined_hm)
6. Generate comparison Table IV matching the existing regional detection analysis.
"""

import os
import sys
import csv
import time
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Bypass thop profiling for RT-DETR
try:
    import ultralytics.utils.torch_utils as tu
    tu.get_flops = lambda model, imgsz=640: 0.0
except Exception:
    pass

from ultralytics import YOLO, RTDETR

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = Path(r"D:\Users\NANS\Downloads")
RESULTS_DIR = ROOT / "results"
POLAR_CSV = RESULTS_DIR / "selected_20_polar_products.csv"

POLAR_EXTRACT_TMP = ROOT / "data" / "extracted_polar_tmp"
POLAR_TILES_USABLE = ROOT / "data" / "tiles" / "polar_usable"
POLAR_INFERENCE_DIR = RESULTS_DIR / "polar_inference"

# Clear previous tiles from faulty run if any
if POLAR_TILES_USABLE.exists():
    shutil.rmtree(POLAR_TILES_USABLE)
if POLAR_EXTRACT_TMP.exists():
    shutil.rmtree(POLAR_EXTRACT_TMP)

for d in [POLAR_EXTRACT_TMP, POLAR_TILES_USABLE, POLAR_INFERENCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUNS = ROOT / "runs"
MODELS = {
    'YOLO26n': (YOLO, RUNS / 'stage2_yolo26n_combined_hm_v2' / 'weights' / 'best.pt'),
    'YOLOv8n': (YOLO, RUNS / 'stage2_yolov8n_combined_hm_v2' / 'weights' / 'best.pt'),
    'YOLOv5s': (YOLO, RUNS / 'stage2_yolov5s_combined_hm_v2' / 'weights' / 'best.pt'),
    'RT-DETR-L': (RTDETR, RUNS / 'stage2_rtdetr_l_combined_hm' / 'weights' / 'best.pt'),
}

TILE_SIZE = 640
DARK_THRESHOLD = 20

def parse_pds4(xml_path):
    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    file_area = root.find(".//pds:File_Area_Observational", ns)
    if file_area is None:
        return None
    
    table = file_area.find(".//pds:Array_2D_Image", ns)
    if table is None:
        table = file_area.find(".//pds:Array_3D_Image", ns)
    if table is None:
        return None

    axes = table.findall(".//pds:Axis_Array", ns)
    lines, samples = None, None
    for a in axes:
        name = a.find("pds:axis_name", ns)
        elems = a.find("pds:elements", ns)
        if name is not None and elems is not None:
            n = name.text.strip().lower()
            if "line" in n:
                lines = int(elems.text)
            elif "sample" in n:
                samples = int(elems.text)

    elem_arr = table.find("pds:Element_Array", ns)
    data_type = elem_arr.find("pds:data_type", ns).text.strip() if elem_arr is not None else "UnsignedByte"
    offset_elem = table.find("pds:offset", ns)
    offset = int(offset_elem.text) if offset_elem is not None else 0
    return {"lines": lines, "samples": samples, "data_type": data_type, "offset": offset}

def normalize_tile(tile: np.ndarray) -> np.ndarray:
    """Exact identical normalization from scripts/tile_ohrc.py."""
    p_low = float(np.percentile(tile, 2))
    p_high = float(np.percentile(tile, 98))
    if p_high <= p_low:
        return np.zeros(tile.shape, dtype=np.uint8)
    norm = np.clip(tile.astype(np.float32), p_low, p_high)
    norm = (norm - p_low) / (p_high - p_low) * 255.0
    return norm.astype(np.uint8)

def tile_product(product_id, img_path, xml_path):
    meta = parse_pds4(xml_path)
    if not meta or not meta["lines"] or not meta["samples"]:
        print(f"  [ERROR] Could not parse dimensions for {xml_path.name}")
        return 0, 0
    lines = meta["lines"]
    samples = meta["samples"]
    offset = meta["offset"]
    dtype = np.uint8 if "byte" in meta["data_type"].lower() else np.uint16
    bytes_per_pixel = np.dtype(dtype).itemsize
    row_bytes = samples * bytes_per_pixel

    n_cols = (samples + TILE_SIZE - 1) // TILE_SIZE
    n_rows = (lines + TILE_SIZE - 1) // TILE_SIZE
    total_tiles = n_cols * n_rows
    usable_count = 0

    print(f"  Tiling {product_id}: {lines} lines x {samples} samples, grid: {n_rows}x{n_cols} = {total_tiles} tiles")
    with open(img_path, "rb") as f:
        for r_idx in range(n_rows):
            start_line = r_idx * TILE_SIZE
            end_line = min(start_line + TILE_SIZE, lines)
            n_lines_this = end_line - start_line

            f.seek(offset + start_line * row_bytes)
            raw = f.read(n_lines_this * row_bytes)
            if len(raw) < n_lines_this * row_bytes:
                break
            strip = np.frombuffer(raw, dtype=dtype).reshape(n_lines_this, samples)

            if n_lines_this < TILE_SIZE:
                strip = np.pad(strip, ((0, TILE_SIZE - n_lines_this), (0, 0)), mode="constant", constant_values=0)

            for c_idx in range(n_cols):
                start_sample = c_idx * TILE_SIZE
                end_sample = min(start_sample + TILE_SIZE, samples)
                tile = strip[:, start_sample:end_sample]
                if tile.shape[1] < TILE_SIZE:
                    tile = np.pad(tile, ((0, 0), (0, TILE_SIZE - tile.shape[1])), mode="constant", constant_values=0)

                # Dark tile filter: native mean < 20 (identical to Module I)
                if float(tile.mean()) < DARK_THRESHOLD:
                    continue

                norm_tile = normalize_tile(tile)
                out_name = f"{product_id}_x{start_sample:05d}_y{start_line:05d}.png"
                Image.fromarray(norm_tile).save(POLAR_TILES_USABLE / out_name)
                usable_count += 1
    return total_tiles, usable_count

def extract_and_tile_all():
    with open(POLAR_CSV, "r", encoding="utf-8") as f:
        polar_products = list(csv.DictReader(f))

    print(f"\n=======================================================")
    print(f"MODULE I: EXTRACTION & TILING ({len(polar_products)} PRODUCTS)")
    print(f"=======================================================")

    t0 = time.time()
    tiling_stats = []

    for i, prod in enumerate(polar_products, 1):
        pid = prod["product_id"]
        zip_path = DOWNLOADS_DIR / f"{pid}.zip"
        if not zip_path.exists():
            print(f"[{i}/{len(polar_products)}] MISSING: {zip_path.name}")
            continue

        print(f"\n[{i}/{len(polar_products)}] Extracting: {pid} ...")
        img_path = None
        xml_path = None

        with zipfile.ZipFile(zip_path, 'r') as z:
            for member in z.infolist():
                # Extract ONLY from data/calibrated/ (the real high-res observation)
                if member.filename.startswith("data/calibrated/"):
                    fname = Path(member.filename).name
                    if fname.lower().endswith('.img'):
                        target = POLAR_EXTRACT_TMP / fname
                        with z.open(member) as src, open(target, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        img_path = target
                    elif fname.lower().endswith('.xml'):
                        target = POLAR_EXTRACT_TMP / fname
                        with z.open(member) as src, open(target, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        xml_path = target

        if img_path and xml_path:
            tot, use = tile_product(pid, img_path, xml_path)
            disc = tot - use
            pct_disc = round(100.0 * disc / max(tot, 1), 2)
            print(f"  Finished: total={tot}, usable={use}, discarded(dark)={disc} ({pct_disc}%)")
            tiling_stats.append({
                "product_id": pid,
                "lat_min": prod["lat_min"],
                "lat_max": prod["lat_max"],
                "lon_min": prod["lon_min"],
                "lon_max": prod["lon_max"],
                "total_tiles": tot,
                "usable_tiles": use,
                "discarded_dark_tiles": disc,
                "pct_discarded": pct_disc
            })
            # Clean up uncompressed files immediately
            try:
                if img_path.exists(): img_path.unlink()
                if xml_path.exists(): xml_path.unlink()
            except Exception:
                pass
        else:
            print(f"  [ERROR] Could not find data/calibrated/ .img or .xml in {zip_path.name}")

    stats_df = pd.DataFrame(tiling_stats)
    stats_df.to_csv(POLAR_INFERENCE_DIR / "polar_tiling_statistics.csv", index=False)
    print(f"\nTiling complete in {time.time() - t0:.1f}s.")
    print(f"Total usable tiles produced: {stats_df['usable_tiles'].sum()}")
    return len(list(POLAR_TILES_USABLE.glob("*.png")))

def run_stage3_inference():
    tiles = sorted(list(POLAR_TILES_USABLE.glob("*.png")))
    total_tiles = len(tiles)
    print(f"\n=======================================================")
    print(f"STAGE 3: INFERENCE ON {total_tiles} USABLE ULTRA-DEEP POLAR TILES")
    print(f"=======================================================")

    results_records = []

    for model_name, (cls, weight_path) in MODELS.items():
        print(f"\n>>> Running Model: {model_name}")
        print(f"    Weights: {weight_path}")
        if not weight_path.exists():
            print(f"    [ERROR] Missing weights: {weight_path}")
            continue

        torch.cuda.empty_cache()
        model = cls(str(weight_path))
        
        detections = []
        positive_tiles = set()
        t0 = time.time()
        
        # Stream predict across all usable tiles
        for r in model.predict(source=str(POLAR_TILES_USABLE), imgsz=640, conf=0.20, stream=True, device=0, verbose=False):
            tile_name = Path(r.path).name
            if len(r.boxes) > 0:
                positive_tiles.add(tile_name)
                for b in r.boxes:
                    conf = float(b.conf.cpu().numpy()[0])
                    xyxy = b.xyxy.cpu().numpy()[0]
                    detections.append({
                        "tile": tile_name,
                        "conf": round(conf, 4),
                        "x1": round(float(xyxy[0]), 2),
                        "y1": round(float(xyxy[1]), 2),
                        "x2": round(float(xyxy[2]), 2),
                        "y2": round(float(xyxy[3]), 2),
                    })

        elapsed = time.time() - t0
        det_df = pd.DataFrame(detections)
        raw_csv = POLAR_INFERENCE_DIR / f"raw_detections_{model_name.lower().replace('-', '_')}.csv"
        det_df.to_csv(raw_csv, index=False)

        det_count = len(detections)
        pos_count = len(positive_tiles)
        eval_count = total_tiles
        rate = round(100.0 * pos_count / max(eval_count, 1), 2)
        mean_conf = round(float(det_df["conf"].mean()), 4) if det_count > 0 else 0.0
        std_conf = round(float(det_df["conf"].std()), 4) if det_count > 0 else 0.0
        det_per_pos = round(det_count / max(pos_count, 1), 2)

        print(f"    Completed in {elapsed:.1f}s ({elapsed/max(eval_count, 1)*1000:.1f} ms/tile)")
        print(f"    Detections: {det_count} | Positive Tiles: {pos_count}/{eval_count} ({rate}%) | Mean Conf: {mean_conf:.4f} ± {std_conf:.4f}")

        results_records.append({
            "region": "Ultra-Deep South Pole (lat ≤ -89.5°S)",
            "model": model_name,
            "detection_count": det_count,
            "tiles_with_detection": pos_count,
            "total_evaluated_tiles": eval_count,
            "tile_detection_rate_pct": rate,
            "mean_confidence": mean_conf,
            "std_confidence": std_conf,
            "detections_per_positive_tile": det_per_pos
        })

    polar_summary_df = pd.DataFrame(results_records)
    polar_summary_df.to_csv(POLAR_INFERENCE_DIR / "polar_regional_summary.csv", index=False)
    return polar_summary_df

def format_markdown_table(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)

def update_table4(polar_df):
    existing_t4_path = ROOT / "results" / "confidence_analysis" / "ohrc_regional_detection_comparison.csv"
    if existing_t4_path.exists():
        existing_df = pd.read_csv(existing_t4_path)
        combined_df = pd.concat([existing_df, polar_df], ignore_index=True)
    else:
        combined_df = polar_df

    out_csv = RESULTS_DIR / "table4_with_ultradeep_polar.csv"
    combined_df.to_csv(out_csv, index=False)
    print(f"\n=======================================================")
    print("TABLE IV — COMPREHENSIVE REGIONAL COMPARISON")
    print(f"=======================================================")
    print(combined_df.to_string(index=False))

    md_path = RESULTS_DIR / "table4_with_ultradeep_polar.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table IV: Chandrayaan-2 OHRC Regional Detection Comparison\n\n")
        f.write("Cross-regional boulder detection performance comparing the existing baseline regions against the newly added Ultra-Deep South Pole batch ($\\le -89.5^\\circ\\text{S}$).\n\n")
        f.write(format_markdown_table(combined_df))
        f.write("\n")

def main():
    print("Starting processing of 20 ultra-deep polar products...")
    n_tiles = extract_and_tile_all()
    if n_tiles == 0:
        print("[ERROR] No usable tiles were generated. Exiting.")
        return
    polar_df = run_stage3_inference()
    update_table4(polar_df)
    print("\nAll tasks finished successfully.")

if __name__ == "__main__":
    main()
