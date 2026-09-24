import re
import requests
import sys

script_path = r"D:\Users\NANS\Downloads\ohrc_2026Sep24T023336150.py"

print(f"Reading {script_path}...")
try:
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Extract cookie_string
match = re.search(r'cookie_string\s*=\s*["\']([^"\']+)["\']', content)
if not match:
    print("Could not find cookie_string in the script.")
    sys.exit(1)

cookie_string = match.group(1)
print(f"Extracted cookie_string ({len(cookie_string)} chars):")
for part in cookie_string.split(";"):
    if part.strip():
        name = part.split("=")[0].strip()
        val = part.split("=")[1].strip() if "=" in part else ""
        print(f"  - {name}: {val[:20]}..." if len(val) > 20 else f"  - {name}: {val}")

test_url = "https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=ohrc"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Cookie": cookie_string
}

print(f"\nTesting session against {test_url} ...")
try:
    resp = requests.get(test_url, headers=headers, timeout=30, allow_redirects=True)
    print(f"HTTP Status: {resp.status_code}")
    print(f"Final Landed URL: {resp.url}")
    
    if "idp.issdc.gov.in" in resp.url.lower() or "login" in resp.url.lower():
        print("\n>>> RESULT: EXPIRED")
        print("Redirected to ISSDC Keycloak Identity Provider (Login required).")
    else:
        print("\n>>> RESULT: VALID")
        print(f"Successfully reached protected area! Content size: {len(resp.text)} bytes")
except Exception as e:
    print(f"Connection error: {e}")
