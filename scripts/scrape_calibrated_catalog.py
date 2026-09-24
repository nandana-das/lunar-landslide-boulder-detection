#!/usr/bin/env python3
"""
Robust, direct metadata scraper for Chandrayaan-2 OHRC products on PRADAN.
Uses 'Jump to file' and PrimeFaces AJAX View requests to query every calibrated product,
extracts isda:area and Refined_Corner_Coordinates, and filters for South Pole (< -84°S).
"""

import os
import re
import sys
import time
import csv
from pathlib import Path
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_CSV = RESULTS_DIR / "pradan_ohrc_metadata_catalog.csv"
FILTERED_CSV = RESULTS_DIR / "pradan_ohrc_southpole_deep_filtered.csv"

# Existing Table I products (paper dataset)
TABLE_1_PRODUCTS = {
    'ch2_ohr_ncp_20190906T2241285714_d_img_gds',
    'ch2_ohr_ncp_20190907T0438126359_d_img_g26',
    'ch2_ohr_ncp_20240425T1012478407_d_img_d18',
    'ch2_ohr_ncp_20240425T1209509264_d_img_d18',
    'ch2_ohr_ncp_20240425T1406019344_d_img_d18',
    'ch2_ohr_ncp_20240425T1603031918_d_img_d18',
    'ch2_ohr_ncp_20250516T0948068899_d_img_d18',
    'ch2_ohr_ncp_20250516T1145499313_d_img_d18',
    'ch2_ohr_ncp_20250516T1342347288_d_img_d18',
    'ch2_ohr_ncp_20250516T1540191774_d_img_d18',
    'ch2_ohr_ncp_20250612T2031048828_d_img_d18',
    'ch2_ohr_ncp_20250612T2229094979_d_img_d18',
}

# 1. Load cookie and data_file_paths from download script
DOWNLOADS_SCRIPT = Path(r"D:\Users\NANS\Downloads\ohrc_2026Sep24T034658733.py")
with open(DOWNLOADS_SCRIPT, "r", encoding="utf-8") as f:
    code = f.read()

m_cookie = re.search(r'cookie_string\s*=\s*["\']([^"\']+)["\']', code)
cookie_string = m_cookie.group(1)

# Extract download paths map
download_paths = {}
for m in re.finditer(r'(/ch2/protected/downloadData/[^\"]+)', code):
    path = m.group(1)
    m_pid = re.search(r'(ch2_ohr_ncp_[0-9T]+_d_img_[a-z0-9]+)', path)
    if m_pid:
        download_paths[m_pid.group(1)] = path

# List of target calibrated products
with open(RESULTS_DIR / "pradan_calibrated_products_list.txt", "r", encoding="utf-8") as f:
    target_products = [line.strip() for line in f if line.strip()]

base_headers = {
    'Cookie': cookie_string,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

ajax_headers = {
    'Faces-Request': 'partial/ajax',
    'X-Requested-With': 'XMLHttpRequest',
    'Accept': 'application/xml, text/xml, */*; q=0.01',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Referer': 'https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=ohrc',
    'Origin': 'https://pradan.issdc.gov.in'
}

session = requests.Session()
session.headers.update(base_headers)

def parse_pds4_docdetail(xml_text):
    cdatas = re.findall(r'<!\[CDATA\[(.*?)\]\]>', xml_text, re.DOTALL)
    doc_cdata = None
    for c in cdatas:
        if 'docDetail' in c:
            doc_cdata = c
            break
    if not doc_cdata:
        return None
    
    soup = BeautifulSoup(doc_cdata, 'html.parser')
    rows = soup.find_all('tr')
    
    area = None
    for tr in rows:
        tds = [td.get_text(strip=True) for td in tr.find_all('td')]
        if len(tds) >= 2 and tds[0] == 'isda:area':
            area = tds[1]
            break
            
    refined_lats, refined_lons = [], []
    system_lats, system_lons = [], []
    
    for i, tr in enumerate(rows):
        tr_id = tr.get('id', '')
        tds = [td.get_text(strip=True) for td in tr.find_all('td')]
        if len(tds) >= 1:
            tag = tds[0]
            is_lat = 'latitude' in tag.lower()
            is_lon = 'longitude' in tag.lower()
            if is_lat or is_lon:
                val = None
                for j in range(i+1, min(i+4, len(rows))):
                    child_tds = [t.get_text(strip=True) for t in rows[j].find_all('td')]
                    if len(child_tds) >= 2 and child_tds[0] == 'content':
                        try:
                            val = float(child_tds[1])
                            break
                        except:
                            pass
                if val is not None:
                    if '0_8_5_1_1' in tr_id or 'refined' in tr_id.lower():
                        if is_lat:
                            refined_lats.append(val)
                        else:
                            refined_lons.append(val)
                    else:
                        if is_lat:
                            system_lats.append(val)
                        else:
                            system_lons.append(val)
                            
    lats = refined_lats if len(refined_lats) >= 4 else (system_lats if len(system_lats) >= 4 else refined_lats + system_lats)
    lons = refined_lons if len(refined_lons) >= 4 else (system_lons if len(system_lons) >= 4 else refined_lons + system_lons)
    
    lat_min = min(lats) if lats else None
    lat_max = max(lats) if lats else None
    lon_min = min(lons) if lons else None
    lon_max = max(lons) if lons else None
    
    return {
        'area': area,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lats': lats,
        'lons': lons
    }

def main():
    print("=" * 70, flush=True)
    print("AUTOMATED PRADAN METADATA SCRAPER FOR CALIBRATED OHRC PRODUCTS", flush=True)
    print("=" * 70, flush=True)
    
    # Load existing processed products
    processed = {}
    if CATALOG_CSV.exists():
        with open(CATALOG_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed[row['product_id']] = row
        print(f"Resuming: {len(processed)} products already in catalog CSV", flush=True)
    else:
        with open(CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['product_id', 'area', 'lat_min', 'lat_max', 'lon_min', 'lon_max', 'zip_download_path', 'is_calibrated'])

    # Get initial page
    browse_url = 'https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=ohrc'
    resp = session.get(browse_url, timeout=30)
    if "login" in resp.url.lower() or "idp.issdc" in resp.url.lower():
        print("Session expired! Please generate a new PRADAN download script.", flush=True)
        sys.exit(1)
        
    m = re.search(r'value="([^"]*)"[^>]*autocomplete="off"', resp.text)
    if not m:
        m = re.search(r'name="javax\.faces\.ViewState"[^>]*value="([^"]*)"', resp.text)
    view_state = m.group(1)
    print(f"Initial ViewState acquired: {view_state[:30]}...", flush=True)

    post_url = 'https://pradan.issdc.gov.in/ch2/protected/browse.xhtml'
    
    to_scrape = [p for p in target_products if p not in processed]
    print(f"Total calibrated products to scrape: {len(to_scrape)} / {len(target_products)}", flush=True)
    
    count = 0
    for idx, pid in enumerate(to_scrape, 1):
        target_zip = pid + ".zip"
        download_path = download_paths.get(pid, f"/ch2/protected/downloadData/.../{target_zip}?ohrc")
        
        # Step A: Jump to file
        p_jump = {
            'javax.faces.partial.ajax': 'true',
            'javax.faces.source': 'tableForm:j_idt184',
            'javax.faces.partial.execute': 'tableForm:skipToFilePanel',
            'javax.faces.partial.render': '@all',
            'tableForm:j_idt183': target_zip,
            'tableForm:j_idt184': 'tableForm:j_idt184',
            'tableForm': 'tableForm',
            'javax.faces.ViewState': view_state
        }
        
        try:
            r_jump = session.post(post_url, data=p_jump, headers=ajax_headers, timeout=30)
            m_vs = re.search(r'<update id="[^"]*javax\.faces\.ViewState[^"]*"><!\[CDATA\[(.*?)\]\]>', r_jump.text)
            if m_vs:
                view_state = m_vs.group(1)
        except Exception as e:
            print(f"[{idx}/{len(to_scrape)}] Error during jump to {pid}: {e}", flush=True)
            time.sleep(2)
            continue
            
        time.sleep(0.3)
        
        # Step B: View row 0
        p_view = {
            'javax.faces.partial.ajax': 'true',
            'javax.faces.source': 'tableForm:lazyDocTable:0:j_idt173',
            'javax.faces.partial.execute': '@all',
            'javax.faces.partial.render': 'tableForm:docDetail',
            'tableForm:lazyDocTable:0:j_idt173': 'tableForm:lazyDocTable:0:j_idt173',
            'tableForm': 'tableForm',
            'javax.faces.ViewState': view_state
        }
        
        try:
            r_view = session.post(post_url, data=p_view, headers=ajax_headers, timeout=30)
            m_vs = re.search(r'<update id="[^"]*javax\.faces\.ViewState[^"]*"><!\[CDATA\[(.*?)\]\]>', r_view.text)
            if m_vs:
                view_state = m_vs.group(1)
                
            meta = parse_pds4_docdetail(r_view.text)
            if not meta:
                print(f"[{idx}/{len(to_scrape)}] {pid}: No metadata parsed", flush=True)
                continue
                
            area = meta['area'] or 'Unknown'
            lat_min_str = f"{meta['lat_min']:.6f}" if meta['lat_min'] is not None else ''
            lat_max_str = f"{meta['lat_max']:.6f}" if meta['lat_max'] is not None else ''
            lon_min_str = f"{meta['lon_min']:.6f}" if meta['lon_min'] is not None else ''
            lon_max_str = f"{meta['lon_max']:.6f}" if meta['lon_max'] is not None else ''
            
            row_dict = {
                'product_id': pid,
                'area': area,
                'lat_min': lat_min_str,
                'lat_max': lat_max_str,
                'lon_min': lon_min_str,
                'lon_max': lon_max_str,
                'zip_download_path': download_path,
                'is_calibrated': 'True'
            }
            
            with open(CATALOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    row_dict['product_id'],
                    row_dict['area'],
                    row_dict['lat_min'],
                    row_dict['lat_max'],
                    row_dict['lon_min'],
                    row_dict['lon_max'],
                    row_dict['zip_download_path'],
                    row_dict['is_calibrated']
                ])
                
            processed[pid] = row_dict
            count += 1
            
            sp_match = " [SP MATCH (< -84°)]" if (area.lower() == 'south pole' and meta['lat_min'] is not None and meta['lat_min'] < -84.0) else ""
            print(f"[{idx}/{len(to_scrape)}] {pid[:40]}: Area={area}, LatMin={lat_min_str}{sp_match}", flush=True)
            
        except Exception as e:
            print(f"[{idx}/{len(to_scrape)}] Error viewing {pid}: {e}", flush=True)
            time.sleep(1)
            
        # Rate limit pause
        time.sleep(0.4)
        if idx % 25 == 0:
            try:
                session.get("https://pradan.issdc.gov.in/ch2/protected/payload.xhtml", timeout=15)
            except:
                pass

    print("\n" + "=" * 70, flush=True)
    print("STEP 5 & 6: FILTERING AND CROSS-REFERENCING", flush=True)
    print("=" * 70, flush=True)
    
    # Filter rows: isda:area == "South Pole" and lat_min < -84
    matching_rows = []
    with open(CATALOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            area = r['area'].strip()
            try:
                lat_min = float(r['lat_min']) if r['lat_min'] else 999.0
            except:
                lat_min = 999.0
                
            if area.lower() == "south pole" and lat_min < -84.0:
                is_new = r['product_id'] not in TABLE_1_PRODUCTS
                r['is_new_addition'] = is_new
                matching_rows.append(r)

    # Save to FILTERED_CSV
    fieldnames = ['product_id', 'area', 'lat_min', 'lat_max', 'lon_min', 'lon_max', 'zip_download_path', 'is_calibrated', 'is_new_addition']
    with open(FILTERED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in matching_rows:
            writer.writerow(r)
            
    new_additions = [r for r in matching_rows if r['is_new_addition']]
    new_calibrated = [r for r in new_additions if r.get('is_calibrated') in (True, 'True', 'true')]
    
    print(f"Total catalog products scraped      : {len(processed)}", flush=True)
    print(f"Matching South Pole (lat_min < -84°): {len(matching_rows)}", flush=True)
    print(f"Total NEW additions (not in Table I): {len(new_additions)}", flush=True)
    print(f"  - New Calibrated Products (ncp)   : {len(new_calibrated)}", flush=True)
    print(f"Saved filtered results to: {FILTERED_CSV.name}", flush=True)
    print("=" * 70, flush=True)
    
    print("\nAll Qualifying NEW Calibrated Products (lat < -84°S):", flush=True)
    for i, r in enumerate(new_calibrated, 1):
        print(f" {i:2d}. {r['product_id']}", flush=True)
        print(f"     Lat Range: [{r['lat_min']}°, {r['lat_max']}°] | Lon: [{r['lon_min']}°, {r['lon_max']}°]", flush=True)
        print(f"     Download : {r['zip_download_path']}", flush=True)

if __name__ == '__main__':
    main()
