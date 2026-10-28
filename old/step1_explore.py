"""
STEP 1 - Explore the data (don't assume anything yet).
Run each cell with Shift+Enter in VS Code. Run them top to bottom.

IMPORTANT: some files are HUGE (the 10-min polder flows are ~1.3 GB each,
TAI MeasurementValues is 3.6 GB in total). So in this script we only read
the column names and a few rows, or we read big files in small pieces
("chunks"). Never do pd.read_csv() on a whole 1 GB file - it will freeze.
"""
# %% Cell 1: imports and settings
import glob
import json
import os

import pandas as pd

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

DATA = "data"
OUT = "output"  # small result files we save go here
os.makedirs(OUT, exist_ok=True)

# One regular expression ("search pattern") for salt-related words.
#   geleidend / geleidbaar = Dutch: conductivity
#        (NOT just "geleid", because that also matches "afgeleid" = derived)
#   EGV     = elektrisch geleidend vermogen (Dutch: electrical conductivity)
#   chloride as a whole word (NOT chlorofyl = algae, NOT Chloridazon = pesticide)
#   ^Cl_ / ^CL$ = parameter codes for chloride, like "Cl_mg/l" or "CL"
#   salin   = salinity, \bsal\b = the short name in the VLE data
#   \bzout  = Dutch: salt (NOT "ijzerzout" = iron salt, a chemical)
# (?i) at the start means: ignore upper/lower case.
SALT_PATTERN = (r"(?i)geleidend|geleidbaar|conduct|\bEGV|\bEC\b|\bchloride\b"
                r"|^Cl_|^CL$|salin|\bsal\b|\bzout")


def find_salt(series):
    """True for every text value that contains a salt-related word."""
    return series.astype(str).str.contains(SALT_PATTERN, regex=True)


# %% Cell 2: overview of the data folder (how many files, how big)
# We skip the .gz files, they are just compressed copies of the .csv files.
rows = []
for folder, subfolders, files in os.walk(DATA):
    files = [f for f in files if not f.endswith((".gz", ".tar"))]
    if files:
        size_mb = sum(os.path.getsize(os.path.join(folder, f)) for f in files) / 1e6
        rows.append({"folder": folder, "n_files": len(files),
                     "size_MB": round(size_mb, 1), "example": files[0]})
overview = pd.DataFrame(rows)
print(overview.to_string(index=False))


# %% Cell 3: Parameter_uitleg.xlsx - what do the parameter codes mean?
params = pd.read_excel(f"{DATA}/agv/Parameter_uitleg.xlsx")
print("Shape (rows, columns):", params.shape)
print("Columns:", list(params.columns))
params[["ID", "NAME", "GROUP", "UNIT"]].head(10)


# %% Cell 4: which parameter codes are about salt?
mask = find_salt(params["ID"]) | find_salt(params["NAME"])
salt_params = params.loc[mask, ["ID", "NAME", "GROUP", "UNIT", "Bron"]]
print(len(salt_params), "salt-related parameter codes:")
salt_params


# %% Cell 5: column names of every AGV file (only reads the first 3 rows)
agv_files = {
    # file name pattern                          separator
    "waterkwaliteit/FYCHEM_*.csv":                ";",
    "waterkwaliteit/HB_*.csv":                    ",",
    "waterkwantiteit/Amstelsluizen_*.csv":        ";",
    "waterkwantiteit/Berlagebrug_*.csv":          ";",
    "waterkwantiteit/polderdebieten_10min_*.csv": ";",
    "waterkwantiteit/polderdebieten_uur_*.csv":   ";",
    "waterkwantiteit/poldergemalen_*.csv":        ";",
}
for pattern, sep in agv_files.items():
    for path in sorted(glob.glob(f"{DATA}/agv/{pattern}")):
        head = pd.read_csv(path, sep=sep, nrows=3, encoding_errors="replace")
        salt_cols = [c for c in head.columns if find_salt(pd.Series([c]))[0]]
        print("=" * 80)
        print(os.path.basename(path))
        print("  columns:", list(head.columns))
        print("  salt words in column names:", salt_cols or "none")
        print(head.to_string(index=False)[:600])

# NOTE: Amstelsluizen and Berlagebrug have TWO header rows:
#   row 1 = location code, row 2 = parameter (e.g. WATHTE.mNAP.dag.gem = daily
#   mean water level in m NAP, Q.m3.dag = flow in m3 per day). We'll handle that later.


# %% Cell 6: search the CHEMISTRY file (FYCHEM, 700 MB) for salt parameters
# The salt info is not in the column names but in the VALUES of the
# 'fewsparameter' column. We read the file in chunks of 500,000 rows,
# keep only salt rows, and throw the rest away. Takes about 1-3 minutes.
fychem_path = glob.glob(f"{DATA}/agv/waterkwaliteit/FYCHEM_*.csv")[0]
keep_cols = ["locatiecode", "locatie omschrijving", "locatie x", "locatie y",
             "datum", "fewsparameter", "fewsparameternaam", "meetwaarde", "eenheid"]
salt_parts = []
for i, chunk in enumerate(pd.read_csv(fychem_path, sep=";", usecols=keep_cols, dtype=str,
                                      chunksize=500_000, encoding_errors="replace")):
    is_salt = find_salt(chunk["fewsparameter"]) | find_salt(chunk["fewsparameternaam"])
    salt_parts.append(chunk[is_salt])
    print(f"chunk {i + 1} done", end="\r")

fychem_salt = pd.concat(salt_parts, ignore_index=True)
fychem_salt["datum"] = pd.to_datetime(fychem_salt["datum"], errors="coerce")
fychem_salt["meetwaarde"] = pd.to_numeric(
    fychem_salt["meetwaarde"].str.replace(",", "."), errors="coerce")

# Save the small salt-only table so we never have to scan the big file again.
fychem_salt.to_csv(f"{OUT}/fychem_salt.csv", index=False)
print(f"\nFound {len(fychem_salt):,} salt measurements -> saved to {OUT}/fychem_salt.csv")


# %% Cell 7: summary - which salt parameters, how many, which years?
summary = (fychem_salt.groupby(["fewsparameter", "fewsparameternaam", "eenheid"])
           .agg(n=("meetwaarde", "size"),
                n_locations=("locatiecode", "nunique"),
                first=("datum", "min"),
                last=("datum", "max"),
                median=("meetwaarde", "median"))
           .sort_values("n", ascending=False))
summary


# %% Cell 8: which LOCATIONS have the most salt measurements?
top_locations = (fychem_salt.groupby(["locatiecode", "locatie omschrijving", "fewsparameter"])
                 .agg(n=("meetwaarde", "size"),
                      first=("datum", "min"), last=("datum", "max"),
                      median=("meetwaarde", "median"), max=("meetwaarde", "max"))
                 .sort_values("n", ascending=False))
top_locations.head(25)


# %% Cell 9: HYDROBIOLOGY file (HB, 628 MB) - quick check for salt parameters
# HB is mostly about plants and animals, but let's check instead of assuming.
hb_path = glob.glob(f"{DATA}/agv/waterkwaliteit/HB_*.csv")[0]
hb_counts = pd.Series(dtype=int)
for chunk in pd.read_csv(hb_path, sep=",", usecols=["fewsparameter", "fewsparameternaam"],
                         dtype=str, chunksize=500_000, encoding_errors="replace"):
    is_salt = find_salt(chunk["fewsparameter"]) | find_salt(chunk["fewsparameternaam"])
    hb_counts = hb_counts.add(chunk.loc[is_salt, "fewsparameternaam"].value_counts(),
                              fill_value=0)
print("Salt-related parameters in HB:")
print(hb_counts.sort_values(ascending=False) if len(hb_counts) else "none")


# %% Cell 10: TAI data (sewage treatment plants) - what is measured?
# MeasurementPoints describes WHAT each sensor measures. It is small (7 MB),
# so we can load all files. (MeasurementValues is 3.6 GB - not now!)
mp_files = sorted(glob.glob(f"{DATA}/tai/MeasurementPoints/*.csv"))
mp = pd.concat([pd.read_csv(f, encoding_errors="replace").assign(file=os.path.basename(f))
                for f in mp_files], ignore_index=True)
print(len(mp_files), "files,", len(mp), "measurement points")
print("Columns:", list(mp.columns))

text_cols = ["MeasurementObjectDescription", "ProductDescription",
             "QuantityDescription", "Parameter", "ParameterDescription"]
is_salt = pd.concat([find_salt(mp[c]) for c in text_cols], axis=1).any(axis=1)
tai_salt = mp[is_salt]
print(len(tai_salt), "salt-related measurement points in",
      tai_salt["file"].nunique(), "installations")

# Add the plant names (from the Installations folder) so we know WHERE they are.
inst = pd.concat([pd.read_csv(f, encoding_errors="replace")
                  for f in glob.glob(f"{DATA}/tai/Installations/*.csv")]).drop_duplicates()
names = dict(zip(inst["Installation"] + ".csv", inst["Installation Description"]))
tai_salt = tai_salt.assign(plant=tai_salt["file"].map(names))

# NL.11 = Waterschap Amstel, Gooi en Vecht (the Amsterdam area, same as the AGV data)
tai_salt[tai_salt["file"].str.startswith("NL.11")][
    ["plant", "MeasurementPointID", "MeasurementObjectDescription",
     "QuantityDescription", "Unit"]]


# %% Cell 11: TAI - what kinds of things are measured overall? (to spot alternatives)
mp["ParameterDescription"].value_counts().head(30)


# %% Cell 12: VLE Global sensor (Aveiro, Portugal)
with open(f"{DATA}/vle_latest.json") as fh:
    vle_latest = json.load(fh)
with open(f"{DATA}/vle_history.json") as fh:
    vle_history = pd.DataFrame(json.load(fh))
print("Latest reading:", vle_latest)
print(vle_history.shape)
vle_history.head()
# 'sal' = salinity in PSU (~35 = pure seawater). This sensor is in Portugal,
# not Amsterdam, so it can only serve as a "what does seawater look like" reference.
