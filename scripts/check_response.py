import requests
import re

script_path = r"D:\Users\NANS\Downloads\ohrc_2026Sep24T034658733.py"
with open(script_path, "r", encoding="utf-8") as f:
    code = f.read()

cookie = re.search(r'cookie_string\s*=\s*["\']([^"\']+)["\']', code).group(1)
headers = {
    'Cookie': cookie,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

resp = requests.get('https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=ohrc', headers=headers)
print('Status:', resp.status_code)
print('URL:', resp.url)
print('Length:', len(resp.text))
print('Has ViewState:', 'ViewState' in resp.text)
if 'ViewState' in resp.text:
    for m in re.finditer(r'id="([^"]*ViewState[^"]*)"\s+value="([^"]*)"', resp.text):
        print("Match 1:", m.group(1), m.group(2))
    for m in re.finditer(r'value="([^"]*)"\s+autocomplete="off"', resp.text):
        print("Match 2:", m.group(1))
else:
    print("First 500 chars:")
    print(resp.text[:500])
