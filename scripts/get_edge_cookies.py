import os
import json
import base64
import sqlite3
import shutil
import tempfile
import ctypes
from ctypes import wintypes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ('cbData', wintypes.DWORD),
        ('pbData', ctypes.POINTER(ctypes.c_char))
    ]

CryptUnprotectData = ctypes.windll.crypt32.CryptUnprotectData
CryptUnprotectData.argtypes = [
    ctypes.POINTER(DATA_BLOB),
    ctypes.POINTER(wintypes.LPWSTR),
    ctypes.POINTER(DATA_BLOB),
    ctypes.c_void_p,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(DATA_BLOB)
]
CryptUnprotectData.restype = wintypes.BOOL

def dpapi_decrypt(data):
    blob_in = DATA_BLOB(len(data), ctypes.create_string_buffer(data, len(data)))
    blob_out = DATA_BLOB()
    if CryptUnprotectData(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
        decrypted = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)
        return decrypted
    raise RuntimeError("DPAPI decryption failed")

edge_local_state = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Local State")
edge_cookie_db = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\Network\Cookies")

if not os.path.exists(edge_local_state):
    print("Edge Local State not found")
    exit(1)

with open(edge_local_state, "r", encoding="utf-8") as f:
    local_state = json.load(f)

encrypted_key = base64.b64decode(local_state["os_crypt"]["encrypted_key"])
encrypted_key = encrypted_key[5:] # strip DPAPI prefix
master_key = dpapi_decrypt(encrypted_key)

if not os.path.exists(edge_cookie_db):
    print("Edge Cookies DB not found")
    exit(1)

def read_file_shared(path):
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 1
    FILE_SHARE_WRITE = 2
    FILE_SHARE_DELETE = 4
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    handle = ctypes.windll.kernel32.CreateFileW(
        path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None
    )
    if handle == -1 or handle == 0xFFFFFFFFFFFFFFFF:
        raise ctypes.WinError()
    chunks = []
    buf = ctypes.create_string_buffer(64 * 1024)
    bytes_read = wintypes.DWORD()
    while ctypes.windll.kernel32.ReadFile(handle, buf, len(buf), ctypes.byref(bytes_read), None) and bytes_read.value > 0:
        chunks.append(buf.raw[:bytes_read.value])
    ctypes.windll.kernel32.CloseHandle(handle)
    return b''.join(chunks)

temp_db = tempfile.NamedTemporaryFile(delete=False).name
with open(temp_db, "wb") as f_out:
    f_out.write(read_file_shared(edge_cookie_db))


conn = sqlite3.connect(temp_db)
cursor = conn.cursor()
cursor.execute("SELECT host_key, name, encrypted_value FROM cookies WHERE host_key LIKE '%issdc%'")

cookies = []
for host_key, name, encrypted_val in cursor.fetchall():
    try:
        if encrypted_val.startswith(b'v10') or encrypted_val.startswith(b'v11'):
            nonce = encrypted_val[3:15]
            ciphertext = encrypted_val[15:]
            aesgcm = AESGCM(master_key)
            decrypted = aesgcm.decrypt(nonce, ciphertext, None).decode('utf-8')
            cookies.append((host_key, name, decrypted))
    except Exception as e:
        pass

conn.close()
try:
    os.remove(temp_db)
except:
    pass

print(f"Found {len(cookies)} cookies for issdc:")
for host, name, val in cookies:
    print(f"Host: {host}, Name: {name}, Value: {val[:15]}...")

if cookies:
    # Build cookie header for pradan.issdc.gov.in
    pradan_cookies = [f"{name}={val}" for host, name, val in cookies if "pradan" in host or host.startswith(".issdc")]
    c_str = "; ".join(pradan_cookies)
    print("\nConstructed Cookie String:")
    print(c_str)
    with open("results/edge_issdc_cookies.txt", "w", encoding="utf-8") as f:
        f.write(c_str)
