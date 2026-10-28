"""
Unzips every .gz file inside the data/ folder so VS Code and Excel can open them.
Put this file in your hackathon folder (next to data/) and run:  python unzip_data.py
The original .gz files are kept; a normal file is created next to each one.
"""
import gzip
import os
import shutil

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

count = 0
for folder, _, names in os.walk(DATA):
    for name in names:
        if not name.endswith(".gz"):
            continue
        src = os.path.join(folder, name)
        dst = src[:-3]  # "file.csv.gz" -> "file.csv"
        if os.path.exists(dst):
            continue
        with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        count += 1
        print("unzipped", os.path.relpath(dst, DATA))

print(f"\nDone! Unzipped {count} files.")
