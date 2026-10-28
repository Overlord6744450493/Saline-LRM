"""
Hackathon data downloader.
Run once:  python download_data.py
Everything is saved into a  data/  folder next to this script.
The API key stops working at 00:00 on 29/10, so run this today!

The key is NOT in this file (it must never be pushed to git). Put it in a file
api_key.txt next to this script (git ignores that file), or set the environment
variable AISO_API_KEY.
"""
import os
import json
import sys
import tarfile
import requests

BASE = "https://aiso-water-data.vercel.app"
VLE = "https://sp02.aveiro.vle.global"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
KEY_FILE = os.path.join(HERE, "api_key.txt")


def read_api_key():
    """The hackathon API key from AISO_API_KEY or api_key.txt (both stay off git)."""
    key = os.environ.get("AISO_API_KEY", "").strip()
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE) as fh:
            key = fh.read().strip()
    if not key:
        sys.exit("No API key found. Put it in api_key.txt next to this script, "
                 "or set the environment variable AISO_API_KEY.")
    return key


H = {"X-API-Key": read_api_key()}


def download(path):
    """Download one file from the API into data/<path>. Skips files already downloaded."""
    local = os.path.join(OUT, path)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f"  already have {path}")
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with requests.get(f"{BASE}/files/{path}", headers=H, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(local + ".part", "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    os.replace(local + ".part", local)
    print(f"  saved {path} ({os.path.getsize(local) / 1e6:.1f} MB)")
    return local


def file_path(entry):
    """The /files list may contain plain strings or dicts; get the path either way."""
    if isinstance(entry, str):
        return entry
    for key in ("path", "name", "key", "file"):
        if key in entry:
            return entry[key]
    raise ValueError(f"Don't know how to read this file entry: {entry}")


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. List every file and save the list
    print("1/4 Listing files...")
    resp = requests.get(f"{BASE}/files", headers=H, timeout=60)
    resp.raise_for_status()
    files = [file_path(f) for f in resp.json()["files"]]
    with open(os.path.join(OUT, "file_list.txt"), "w") as fh:
        fh.write("\n".join(files))
    print(f"  {len(files)} files available (list saved to data/file_list.txt)")

    # 2. TAI: everything in one tar, then unpack it
    print("2/4 Downloading TAI (150 MB, may take a minute)...")
    tar_path = download("tai/tai_all.tar")
    with tarfile.open(tar_path) as t:
        t.extractall(os.path.join(OUT, "tai"))
    print("  TAI unpacked into data/tai/")

    # 3. AGV: every file under agv/
    agv = [f for f in files if f.startswith("agv/")]
    print(f"3/4 Downloading {len(agv)} AGV files...")
    failed = []
    for f in agv:
        try:
            download(f)
        except Exception as e:
            print(f"  FAILED {f}: {e}")
            failed.append(f)

    # 4. VLE Global (no key needed)
    print("4/4 Downloading VLE Global...")
    for name in ("latest", "history"):
        try:
            r = requests.get(f"{VLE}/api/{name}", timeout=120)
            r.raise_for_status()
            with open(os.path.join(OUT, f"vle_{name}.json"), "w") as fh:
                json.dump(r.json(), fh)
            print(f"  saved vle_{name}.json")
        except Exception as e:
            print(f"  FAILED vle {name}: {e}")

    print("\nDone! Everything is in:", OUT)
    if failed:
        print("These failed, just run the script again to retry them:", failed)


if __name__ == "__main__":
    main()
