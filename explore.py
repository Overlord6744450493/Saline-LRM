"""
Load the downloaded data. Run download_data.py first.
Tip: in VS Code, the  # %%  lines turn into runnable cells (Shift+Enter).
"""
# %%
import glob
import json
import pandas as pd

# %% TAI files (comma separated, use encoding_errors="replace")
tai_files = sorted(glob.glob("data/tai/**/*.csv*", recursive=True))
print(len(tai_files), "TAI files, e.g.", tai_files[:5])
tai = pd.read_csv(tai_files[0], encoding_errors="replace")
tai.head()

# %% AGV parameter explanation: read this first
params = pd.read_excel("data/agv/Parameter_uitleg.xlsx")
params.head(20)

# %% AGV chemistry (FYCHEM) uses ";"
fychem_files = glob.glob("data/agv/waterkwaliteit/**/*FYCHEM*", recursive=True)
print(fychem_files)
if fychem_files:
    fychem = pd.read_csv(fychem_files[0], sep=";", encoding_errors="replace")
    fychem.head()

# %% AGV hydrobiology (HB) uses ","
hb_files = glob.glob("data/agv/waterkwaliteit/**/*HB*", recursive=True)
print(hb_files)
if hb_files:
    hb = pd.read_csv(hb_files[0], sep=",", encoding_errors="replace")
    hb.head()

# %% AGV water quantity uses ";"
qty_files = sorted(glob.glob("data/agv/waterkwantiteit/**/*.*", recursive=True))
print(qty_files)
# The 10-minute polder flows come in 4 parts with different headers: load each part separately.

# %% VLE Global
with open("data/vle_latest.json") as fh:
    vle_latest = pd.json_normalize(json.load(fh))
vle_latest.head()
