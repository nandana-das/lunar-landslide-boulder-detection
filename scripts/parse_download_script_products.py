import re

with open('scripts/pradan_bulk_download.py', 'r', encoding='utf-8') as f:
    code = f.read()

matches = re.findall(r'/ch2/protected/downloadData/[^\"]+', code)
print(f'Total download paths in script: {len(matches)}')
calibrated = [m for m in matches if '/calibrated/' in m]
raw = [m for m in matches if '/raw/' in m]
print(f'Calibrated paths: {len(calibrated)}')
print(f'Raw paths: {len(raw)}')

# Extract product filenames
calibrated_products = [re.search(r'(ch2_ohr_ncp_[^\.?]+)', m).group(1) for m in calibrated if re.search(r'(ch2_ohr_ncp_[^\.?]+)', m)]
print(f'Unique calibrated product IDs: {len(set(calibrated_products))}')

# Save list of unique calibrated product IDs
with open('results/pradan_calibrated_products_list.txt', 'w', encoding='utf-8') as f:
    for pid in sorted(set(calibrated_products)):
        f.write(pid + '\n')
print('Saved to results/pradan_calibrated_products_list.txt')
