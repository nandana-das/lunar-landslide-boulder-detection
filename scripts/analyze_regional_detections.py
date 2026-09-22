"""Analyze and split OHRC Stage-2 detection results by product region.

Regions:
- South Pole (lat ≈ -70°S): 6 products (mean lat ~ -70.1°)
- Equatorial (lat ≈ +60°N): 6 products (mean lat ~ +60.5°)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Define product regions
SOUTH_POLE_PRODUCTS = {
    'ch2_ohr_ncp_20190906T2241285714_d_img_gds': -70.99,
    'ch2_ohr_ncp_20190907T0438126359_d_img_g26': -70.95,
    'ch2_ohr_ncp_20240425T1012478407_d_img_d18': -69.40,
    'ch2_ohr_ncp_20240425T1209509264_d_img_d18': -69.43,
    'ch2_ohr_ncp_20240425T1406019344_d_img_d18': -69.42,
    'ch2_ohr_ncp_20240425T1603031918_d_img_d18': -69.41,
}

EQUATORIAL_PRODUCTS = {
    'ch2_ohr_ncp_20250516T0948068899_d_img_d18': 60.46,
    'ch2_ohr_ncp_20250516T1145499313_d_img_d18': 60.45,
    'ch2_ohr_ncp_20250516T1342347288_d_img_d18': 60.43,
    'ch2_ohr_ncp_20250516T1540191774_d_img_d18': 60.47,
    'ch2_ohr_ncp_20250612T2031048828_d_img_d18': 60.59,
    'ch2_ohr_ncp_20250612T2229094979_d_img_d18': 60.62,
}

MODELS = ['yolo26n', 'yolov8n', 'yolov5s']
MODEL_DISPLAY = {
    'yolo26n': 'YOLO26n',
    'yolov8n': 'YOLOv8n',
    'yolov5s': 'YOLOv5s'
}

def extract_product_id(tile_filename: str) -> str:
    """Extract product ID from tile filename."""
    base = tile_filename.replace('.png', '').replace('.jpg', '')
    parts = base.split('_')
    # Product ID is everything before the trailing coordinates xNNNNN_yNNNNN
    return '_'.join(parts[:-2])

def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / 'results' / 'confidence_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)
    inference_dir = root / 'results' / 'ohrc_inference'
    
    # Check evaluated tiles count per region
    sample_dirs = [Path('C:/ohrc_detections/images'), root / 'data' / 'priority_sample']
    all_evaluated_tiles = set()
    for sdir in sample_dirs:
        if sdir.exists():
            for p in sdir.glob('*.png'):
                all_evaluated_tiles.add(p.name)
    
    south_eval_tiles = sum(1 for t in all_evaluated_tiles if extract_product_id(t) in SOUTH_POLE_PRODUCTS)
    eq_eval_tiles = sum(1 for t in all_evaluated_tiles if extract_product_id(t) in EQUATORIAL_PRODUCTS)
    print(f"Total evaluated candidate tiles: {len(all_evaluated_tiles)}")
    print(f"  South Pole tiles: {south_eval_tiles}")
    print(f"  Equatorial tiles: {eq_eval_tiles}")

    # Load raw detection CSVs
    dfs = {}
    for m in MODELS:
        csv_p = inference_dir / f'raw_detections_{m}.csv'
        if not csv_p.exists():
            raise FileNotFoundError(f"Missing detection CSV: {csv_p}")
        df = pd.read_csv(csv_p)
        df['product_id'] = df['tile'].apply(extract_product_id)
        df['region'] = df['product_id'].apply(
            lambda p: 'South Pole (lat ≈ -70°S)' if p in SOUTH_POLE_PRODUCTS
            else ('Equatorial (lat ≈ +60°N)' if p in EQUATORIAL_PRODUCTS else 'Unknown')
        )
        dfs[m] = df

    # 1. Summary comparison table per region & model
    summary_rows = []
    regions = ['South Pole (lat ≈ -70°S)', 'Equatorial (lat ≈ +60°N)']
    
    for r in regions:
        eval_tiles = south_eval_tiles if 'South Pole' in r else eq_eval_tiles
        for m in MODELS:
            sub = dfs[m][dfs[m]['region'] == r]
            det_count = len(sub)
            pos_tiles = sub['tile'].nunique()
            mean_conf = sub['conf'].mean() if det_count > 0 else 0.0
            std_conf = sub['conf'].std() if det_count > 0 else 0.0
            pos_tile_pct = (pos_tiles / eval_tiles * 100) if eval_tiles > 0 else 0.0
            det_per_pos_tile = (det_count / pos_tiles) if pos_tiles > 0 else 0.0
            
            summary_rows.append({
                'region': r,
                'model': MODEL_DISPLAY[m],
                'detection_count': det_count,
                'tiles_with_detection': pos_tiles,
                'total_evaluated_tiles': eval_tiles,
                'tile_detection_rate_pct': round(pos_tile_pct, 2),
                'mean_confidence': round(mean_conf, 4),
                'std_confidence': round(std_conf, 4),
                'detections_per_positive_tile': round(det_per_pos_tile, 2)
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / 'ohrc_regional_detection_comparison.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nRegional comparison saved to: {summary_csv}")
    print(summary_df.to_string(index=False))

    # 2. Per-product breakdown table
    product_rows = []
    all_prods = [(p, lat, 'South Pole (lat ≈ -70°S)') for p, lat in SOUTH_POLE_PRODUCTS.items()] + \
                 [(p, lat, 'Equatorial (lat ≈ +60°N)') for p, lat in EQUATORIAL_PRODUCTS.items()]
    
    for prod_id, lat, reg in all_prods:
        prod_eval = sum(1 for t in all_evaluated_tiles if extract_product_id(t) == prod_id)
        for m in MODELS:
            sub = dfs[m][dfs[m]['product_id'] == prod_id]
            det_count = len(sub)
            pos_tiles = sub['tile'].nunique()
            mean_conf = sub['conf'].mean() if det_count > 0 else 0.0
            product_rows.append({
                'region': reg,
                'product_id': prod_id,
                'mean_latitude_deg': lat,
                'model': MODEL_DISPLAY[m],
                'evaluated_tiles': prod_eval,
                'detection_count': det_count,
                'tiles_with_detection': pos_tiles,
                'mean_confidence': round(mean_conf, 4)
            })

    prod_df = pd.DataFrame(product_rows)
    prod_csv = results_dir / 'ohrc_per_product_detection_breakdown.csv'
    prod_df.to_csv(prod_csv, index=False)
    print(f"Per-product breakdown saved to: {prod_csv}")

    # 3. Create Grouped Bar Chart (300 DPI PNG)
    # Style settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    x = np.arange(len(regions))  # label locations
    width = 0.25  # bar width
    
    # Custom curated palette
    colors = {
        'YOLO26n': '#1f77b4',  # Deep Steel Blue
        'YOLOv8n': '#ff7f0e',  # Vibrant Coral/Orange
        'YOLOv5s': '#2ca02c'   # Forest Emerald Green
    }
    
    for i, m in enumerate(MODELS):
        disp = MODEL_DISPLAY[m]
        sub = summary_df[summary_df['model'] == disp]
        counts = sub['detection_count'].values
        mean_confs = sub['mean_confidence'].values
        pos_tiles = sub['tiles_with_detection'].values
        
        offset = (i - 1) * width
        rects = ax.bar(
            x + offset, 
            counts, 
            width, 
            label=f"{disp}",
            color=colors[disp],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.9
        )
        
        # Add labels above bars
        for rect, count, conf, pt in zip(rects, counts, mean_confs, pos_tiles):
            height = rect.get_height()
            ax.annotate(
                f"{count:,}\n({pt} tiles)\n[c̄={conf:.2f}]",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                color='#1a1a1a'
            )
            
    # Chart styling
    region_labels = [
        'South Pole Region\n(lat ≈ -70°S, 6 Products, 724 Tiles)',
        'Equatorial / Northern Region\n(lat ≈ +60°N, 6 Products, 690 Tiles)'
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, fontsize=11, fontweight='semibold')
    ax.set_ylabel('Total Detection Count (τ = 0.20)', fontsize=12, fontweight='bold')
    ax.set_title(
        'OHRC Stage-2 Boulder Detections by Product Region and Model\n(Chandrayaan-2 OHRC: 12 Products, 1,414 Evaluated Tiles)',
        fontsize=13, fontweight='bold', pad=16
    )
    
    # Set y-limit with breathing room for annotations
    max_count = summary_df['detection_count'].max()
    ax.set_ylim(0, max_count * 1.25)
    
    # Grid and legend
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10, loc='upper left')
    
    # Subtle note at the bottom
    plt.figtext(
        0.5, 0.02,
        "Note: Annotations show total detections, tiles-with-detection count, and mean confidence [c̄].\n"
        "South Pole exhibits high-shadow, low-sun angles (~10-15°); Equatorial features high-illumination angles.",
        ha='center', fontsize=8.5, style='italic', color='#444444'
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    plot_path = results_dir / 'regional_detections_by_model.png'
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"\nGrouped bar chart saved to: {plot_path}")

if __name__ == '__main__':
    main()
