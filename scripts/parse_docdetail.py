import re
from bs4 import BeautifulSoup

with open('results/ajax_view_result.xml', 'r', encoding='utf-8') as f:
    text = f.read()

soup = BeautifulSoup(text, 'html.parser')

data = {}
for tr in soup.find_all('tr'):
    tds = tr.find_all('td')
    if len(tds) >= 2:
        k = tds[0].get_text(strip=True)
        v = tds[1].get_text(strip=True)
        data[k] = v

print("Total key-value pairs parsed from docDetail:", len(data))
print("\nKey properties:")
for target in [
    'isda:area',
    'isda:upper_left_latitude',
    'isda:upper_right_latitude',
    'isda:lower_left_latitude',
    'isda:lower_right_latitude',
    'isda:upper_left_longitude',
    'isda:upper_right_longitude',
    'isda:lower_left_longitude',
    'isda:lower_right_longitude',
    'isda:solar_incidence',
    'lidvid_reference',
    'logical_identifier'
]:
    for k, v in data.items():
        if target.lower() in k.lower():
            print(f"  {k} = {v}")
