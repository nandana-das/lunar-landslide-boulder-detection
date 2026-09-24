#!/usr/bin/env python3
"""
Automated metadata scraper for Chandrayaan-2 OHRC products on PRADAN portal.
Iterates through all products via reverse-engineered PrimeFaces AJAX View requests,
extracts isda:area and Refined_Corner_Coordinates, filters South Pole (< -84° lat),
and cross-references against existing Table I products.
"""

import os
import re
import sys
import time
import csv
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_CSV = RESULTS_DIR / "pradan_ohrc_metadata_catalog.csv"
FILTERED_CSV = RESULTS_DIR / "pradan_ohrc_southpole_deep_filtered.csv"

# Existing Table I products (paper dataset)
TABLE_1_PRODUCTS = {
    # South pole (~ -70°S)
    'ch2_ohr_ncp_20190906T2241285714_d_img_gds',
    'ch2_ohr_ncp_20190907T0438126359_d_img_g26',
    'ch2_ohr_ncp_20240425T1012478407_d_img_d18',
    'ch2_ohr_ncp_20240425T1209509264_d_img_d18',
    'ch2_ohr_ncp_20240425T1406019344_d_img_d18',
    'ch2_ohr_ncp_20240425T1603031918_d_img_d18',
    # Equatorial (~ +60°N)
    'ch2_ohr_ncp_20250516T0948068899_d_img_d18',
    'ch2_ohr_ncp_20250516T1145499313_d_img_d18',
    'ch2_ohr_ncp_20250516T1342347288_d_img_d18',
    'ch2_ohr_ncp_20250516T1540191774_d_img_d18',
    'ch2_ohr_ncp_20250612T2031048828_d_img_d18',
    'ch2_ohr_ncp_20250612T2229094979_d_img_d18',
}

# 1. Load cookie
DOWNLOADS_SCRIPT = Path(r"D:\Users\NANS\Downloads\ohrc_2026Sep24T034658733.py")
if not DOWNLOADS_SCRIPT.exists():
    print(f"Error: Could not find script at {DOWNLOADS_SCRIPT}")
    sys.exit(1)

with open(DOWNLOADS_SCRIPT, "r", encoding="utf-8") as f:
    code = f.read()

m_cookie = re.search(r'cookie_string\s*=\s*["\']([^"\']+)["\']', code)
if not m_cookie:
    print("Error: Could not extract cookie_string")
    sys.exit(1)
cookie_string = m_cookie.group(1)

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

def keep_alive_ping():
    try:
        session.get("https://pradan.issdc.gov.in/ch2/protected/payload.xhtml", timeout=15)
    except:
        pass

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
    print("=" * 70)
    print("CHANDRAYAAN-2 OHRC PRADAN CATALOG METADATA SCRAPER")
    print("=" * 70)
    
    # Check already processed products to support resuming
    processed_products = {}
    if CATALOG_CSV.exists():
        with open(CATALOG_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_products[row['product_id']] = row
        print(f"Resuming: Loaded {len(processed_products)} existing records from {CATALOG_CSV.name}")
    else:
        with open(CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['product_id', 'area', 'lat_min', 'lat_max', 'lon_min', 'lon_max', 'zip_download_path', 'is_calibrated'])

    # Step 1: Initial GET
    browse_url = 'https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=ohrc'
    print("\n[1/3] Accessing browse page and obtaining ViewState...")
    resp = session.get(browse_url, timeout=30)
    if "login" in resp.url.lower() or "idp.issdc" in resp.url.lower():
        print("Session expired! Please generate a new PRADAN download script.")
        sys.exit(1)
        
    m = re.search(r'name="javax\.faces\.ViewState"[^>]*value="([^"]*)"', resp.text)
    if not m:
        m = re.search(r'value="([^"]*)"[^>]*autocomplete="off"', resp.text)
    if not m:
        print("Could not find initial ViewState!")
        sys.exit(1)
    view_state = m.group(1)
    print(f"Initial ViewState acquired: {view_state[:30]}...")

    # Detect total rows
    m_total = re.search(r'Total Rows:\s*([0-9]+)', resp.text)
    total_rows = int(m_total.group(1)) if m_total else 624
    print(f"Total catalog rows reported: {total_rows}")

    PAGE_SIZE = 50
    total_pages = (total_rows + PAGE_SIZE - 1) // PAGE_SIZE
    post_url = 'https://pradan.issdc.gov.in/ch2/protected/browse.xhtml'

    print(f"\n[2/3] Iterating {total_pages} pages ({PAGE_SIZE} rows/page)...")
    
    total_scraped = len(processed_products)
    req_counter = 0

    for page_idx in range(total_pages):
        first_row = page_idx * PAGE_SIZE
        print(f"\n--- Page {page_idx + 1}/{total_pages} (rows {first_row} to {min(first_row + PAGE_SIZE, total_rows)}) ---")
        
        # Paginate to this chunk
        page_payload = {
            'javax.faces.partial.ajax': 'true',
            'javax.faces.source': 'tableForm:lazyDocTable',
            'javax.faces.partial.execute': 'tableForm:lazyDocTable',
            'javax.faces.partial.render': 'tableForm:lazyDocTable',
            'tableForm:lazyDocTable': 'tableForm:lazyDocTable',
            'tableForm:lazyDocTable_pagination': 'true',
            'tableForm:lazyDocTable_first': str(first_row),
            'tableForm:lazyDocTable_rows': str(PAGE_SIZE),
            'tableForm:lazyDocTable_rppDD': str(PAGE_SIZE),
            'tableForm:lazyDocTable_skipChildren': 'true',
            'tableForm:lazyDocTable_encodeFeature': 'true',
            'tableForm': 'tableForm',
            'javax.faces.ViewState': view_state
        }
        
        try:
            r_page = session.post(post_url, data=page_payload, headers=ajax_headers, timeout=30)
            m_vs = re.search(r'<update id="[^"]*javax\.faces\.ViewState[^"]*"><!\[CDATA\[(.*?)\]\]>', r_page.text)
            if m_vs:
                view_state = m_vs.group(1)
        except Exception as e:
            print(f"Error fetching page {page_idx}: {e}")
            time.sleep(2)
            continue
            
        # Parse CDATA of lazyDocTable
        cdatas = re.findall(r'<!\[CDATA\[(.*?)\]\]>', r_page.text, re.DOTALL)
        table_html = None
        for c in cdatas:
            if 'tableForm:lazyDocTable' in c or 'data-ri' in c:
                table_html = c
                break
        if not table_html:
            print("Warning: could not find table HTML in page response")
            continue
            
        soup_page = BeautifulSoup(table_html, 'html.parser')
        rows = soup_page.find_all('tr', {'data-ri': True})
        print(f"Found {len(rows)} data rows on page {page_idx + 1}")
        
        for row in rows:
            ri = row.get('data-ri')
            a_tag = row.find('a', href=True)
            if not a_tag:
                continue
            href = a_tag['href']
            filename = a_tag.get_text(strip=True)
            product_id = filename.replace('.zip', '')
            is_calibrated = 'ch2_ohr_ncp' in product_id
            
            # Check if already processed
            if product_id in processed_products:
                continue
                
            # Rate limit pause
            time.sleep(0.4)
            req_counter += 1
            if req_counter % 25 == 0:
                keep_alive_ping()
                
            # Trigger View button for this row
            view_payload = {
                'javax.faces.partial.ajax': 'true',
                'javax.faces.source': f'tableForm:lazyDocTable:{ri}:j_idt173',
                'javax.faces.partial.execute': '@all',
                'javax.faces.partial.render': 'tableForm:docDetail',
                f'tableForm:lazyDocTable:{ri}:j_idt173': f'tableForm:lazyDocTable:{ri}:j_idt173',
                'tableForm': 'tableForm',
                'javax.faces.ViewState': view_state
            }
            
            try:
                r_view = session.post(post_url, data=view_payload, headers=ajax_headers, timeout=30)
                m_vs = re.search(r'<update id="[^"]*javax\.faces\.ViewState[^"]*"><!\[CDATA\[(.*?)\]\]>', r_view.text)
                if m_vs:
                    view_state = m_vs.group(1)
                    
                meta = parse_pds4_docdetail(r_view.text)
                if not meta:
                    print(f"  [Row {ri}] {product_id[:35]}: No metadata returned")
                    continue
                    
                row_data = {
                    'product_id': product_id,
                    'area': meta['area'] or 'Unknown',
                    'lat_min': f"{meta['lat_min']:.6f}" if meta['lat_min'] is not None else '',
                    'lat_max': f"{meta['lat_max']:.6f}" if meta['lat_max'] is not None else '',
                    'lon_min': f"{meta['lon_min']:.6f}" if meta['lon_min'] is not None else '',
                    'lon_max': f"{meta['lon_max']:.6f}" if meta['lon_max'] is not None else '',
                    'zip_download_path': href,
                    'is_calibrated': is_calibrated
                }
                
                # Append to CSV immediately
                with open(CATALOG_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        row_data['product_id'],
                        row_data['area'],
                        row_data['lat_min'],
                        row_data['lat_max'],
                        row_data['lon_min'],
                        row_data['lon_max'],
                        row_data['zip_download_path'],
                        row_data['is_calibrated']
                    ])
                    
                processed_products[product_id] = row_data
                total_scraped += 1
                
                # Print progress
                pole_flag = " [SP MATCH]" if (row_data['area'] == 'South Pole' and meta['lat_min'] is not None and meta['lat_min'] < -84.0) else ""
                print(f"  ({total_scraped}/{total_rows}) {product_id[:40]}: Area={row_data['area']}, LatMin={row_data['lat_min']}{pole_flag}")
                
            except Exception as e:
                print(f"  Error querying row {ri} ({product_id}): {e}")
                time.sleep(1)

    print("\n[3/3] Performing Step 5 & 6 Filtering and Cross-Referencing...")
    
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
            
    print(f"\nSaved {len(matching_rows)} matching products to {FILTERED_CSV}")
    
    new_additions = [r for r in matching_rows if r['is_new_addition']]
    new_calibrated = [r for r in new_additions if r.get('is_calibrated') in (True, 'True', 'true')]
    
    print("\n" + "=" * 70)
    print("FILTERING & CROSS-REFERENCE SUMMARY")
    print("=" * 70)
    print(f"Total catalog products parsed       : {total_scraped}")
    print(f"Matching South Pole (lat_min < -84°): {len(matching_rows)}")
    print(f"Total NEW additions (not in Table I): {len(new_additions)}")
    print(f"  - New Calibrated Products (ncp)   : {len(new_calibrated)}")
    print(f"  - New Raw Products (nrp)          : {len(new_additions) - len(new_calibrated)}")
    print("=" * 70)
    
    print("\nTop Qualifying NEW Calibrated Products:")
    for i, r in enumerate(new_calibrated[:15], 1):
        print(f" {i:2d}. {r['product_id']}")
        print(f"     Lat Range: [{r['lat_min']}°, {r['lat_max']}°] | Lon: [{r['lon_min']}°, {r['lon_max']}°]")
        print(f"     Zip URL  : {r['zip_download_path']}")

if __name__ == '__main__':
    main()
