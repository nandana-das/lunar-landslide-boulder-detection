import csv
import json
from pathlib import Path

csv_path = Path("results/pradan_ohrc_southpole_deep_filtered.csv")
with open(csv_path, mode="r", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["is_calibrated"] == "True"]

polar = [r for r in rows if float(r["lat_min"]) <= -89.5]

for r in polar:
    lmin = float(r["lon_min"])
    lmax = float(r["lon_max"])
    r["lon_mid"] = (lmin + lmax) / 2.0
    r["lon_span"] = lmax - lmin
    r["lat_min_f"] = float(r["lat_min"])
    r["lat_max_f"] = float(r["lat_max"])
    parts = r["product_id"].split("_")
    r["date"] = parts[3][:8] if len(parts) > 3 else ""
    r["time"] = parts[3][9:15] if len(parts) > 3 else ""

# Sort by longitude mid to see full spectrum
polar.sort(key=lambda x: x["lon_mid"])

# Select 20 products evenly spaced across the longitude spectrum,
# ensuring at most 1 product per (date, hour) to avoid clustered orbits.
selected = []
used_date_hours = set()

# Divide into 20 quantile bins or target 20 target longitudes from min to max
min_lon = min(p["lon_mid"] for p in polar)
max_lon = max(p["lon_mid"] for p in polar)
target_lons = [min_lon + i * (max_lon - min_lon) / 19.0 for i in range(20)]

for target in target_lons:
    # Find candidate closest to target not sharing date_hour
    best_cand = None
    best_dist = 999999
    for p in polar:
        dh = p["date"] + "_" + p["time"][:2]
        if dh in used_date_hours or p["product_id"] in [s["product_id"] for s in selected]:
            continue
        dist = abs(p["lon_mid"] - target)
        if dist < best_dist:
            best_dist = dist
            best_cand = p
    if best_cand:
        selected.append(best_cand)
        used_date_hours.add(best_cand["date"] + "_" + best_cand["time"][:2])

# If fewer than 20 due to constraints, fill in with largest gap
while len(selected) < 20:
    remaining = [p for p in polar if p["product_id"] not in [s["product_id"] for s in selected]]
    # Pick one that maximizes minimum distance to already selected
    best_cand = max(remaining, key=lambda p: min(abs(p["lon_mid"] - s["lon_mid"]) for s in selected))
    selected.append(best_cand)

selected.sort(key=lambda x: x["lon_mid"])

print(f"Selected {len(selected)} ultra-deep polar products:")
print(f"{'#':<3} {'Product ID':<44} {'Lat Range':<22} {'Lon Mid':<8} {'Lon Span':<8} {'Date':<10}")
print("-" * 105)
for i, s in enumerate(selected, 1):
    lat_str = f"[{s['lat_min_f']:.3f}, {s['lat_max_f']:.3f}]"
    print(f"{i:<3} {s['product_id']:<44} {lat_str:<22} {s['lon_mid']:<8.1f} {s['lon_span']:<8.1f} {s['date']:<10}")

out_csv = Path("results/selected_20_polar_products.csv")
with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
    writer.writeheader()
    writer.writerows(selected)
print(f"\nSaved selection to {out_csv}")

