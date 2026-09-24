import re
import requests
from bs4 import BeautifulSoup

script_path = r"D:\Users\NANS\Downloads\ohrc_2026Sep24T034658733.py"
with open(script_path, "r", encoding="utf-8") as f:
    content = f.read()

cookie_string = re.search(r'cookie_string\s*=\s*["\']([^"\']+)["\']', content).group(1)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Cookie": cookie_string
}

session = requests.Session()
session.headers.update(headers)

resp = session.get("https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=ohrc")
with open("results/live_browse_page.html", "w", encoding="utf-8") as f:
    f.write(resp.text)

soup = BeautifulSoup(resp.text, "html.parser")
print("Page Title:", soup.title.string if soup.title else "No title")

# Find forms and tables
forms = soup.find_all("form")
print(f"Found {len(forms)} form(s)")
for i, form in enumerate(forms):
    print(f"Form {i}: id={form.get('id')}, action={form.get('action')}")

tables = soup.find_all("table")
print(f"Found {len(tables)} table(s)")
for i, table in enumerate(tables):
    print(f"Table {i}: id={table.get('id')}, class={table.get('class')}")

# Check any link or button with 'View' or related onclick
view_items = []
for el in soup.find_all(["a", "button", "span", "input"]):
    text = el.get_text(strip=True)
    onclick = el.get("onclick", "")
    href = el.get("href", "")
    cid = el.get("id", "")
    if "view" in text.lower() or "view" in onclick.lower() or "view" in href.lower() or "view" in cid.lower():
        view_items.append((el.name, text, cid, href, onclick))

print(f"\nFound {len(view_items)} View-related elements:")
for item in view_items[:10]:
    print(" ", item)
