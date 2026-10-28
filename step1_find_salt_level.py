"""
STEP 1 - What is the stakeholder's "salt level"?
The stakeholder says: "when the salt level is NEGATIVE, ocean water flows in".
A salt CONCENTRATION can never be negative, so we check every candidate:
which measurements actually go below zero, and how often?

Run each cell with Shift+Enter, top to bottom.
One-time install (in the VS Code terminal):
    python -m pip install matplotlib truststore
"""
# %% Cell 1: imports
import glob

import matplotlib.pyplot as plt
import pandas as pd

from helpers import daily_sea_level, download_sea_level, load_agv_level, load_storage_level

pd.set_option("display.width", 200)


# %% Cell 2: salt words in Parameter_uitleg.xlsx
params = pd.read_excel("data/agv/Parameter_uitleg.xlsx")
salt_words = r"(?i)\bzout|salt|salin|\bchloride\b|^Cl_|^CL$|geleidend|conduct|\bEGV|\bEC\b"
is_salt = (params["ID"].astype(str).str.contains(salt_words)
           | params["NAME"].astype(str).str.contains(salt_words))
params.loc[is_salt, ["ID", "NAME", "UNIT", "Bron"]]
# -> chloride (Cl, mg/l) and conductivity (GELDHD / EGV, mS/m). No "salt level".


# %% Cell 3: does chloride or conductivity ever go negative?
# We scan the big chemistry file in chunks (about 1-2 minutes).
fychem = glob.glob("data/agv/waterkwaliteit/FYCHEM_*.csv")[0]
lowest = []
for chunk in pd.read_csv(fychem, sep=";", usecols=["fewsparameter", "meetwaarde"], dtype=str,
                         chunksize=1_000_000, encoding_errors="replace"):
    values = pd.to_numeric(chunk["meetwaarde"].str.replace(",", "."), errors="coerce")
    lowest.append(values.groupby(chunk["fewsparameter"]).min())
lowest = pd.concat(lowest).groupby(level=0).min()
print("Parameters with negative values in FYCHEM:", (lowest < 0).sum())
print("Lowest chloride:", lowest.get("Cl_mg/l"), " lowest conductivity:", lowest.get("GELDHD_mS/m_25oC"))
# RESULT: 0 parameters go negative. So chloride/conductivity is NOT the "salt level".


# %% Cell 4: which WATER columns go negative?
storage = load_storage_level()                       # Amstelsluizen, m NAP
berlage = load_agv_level("Berlagebrug_debiet_waterhoogte")
berlage.columns = ["berlage_level", "berlage_flow"]   # m NAP, m3/day
berlage.loc[berlage["berlage_level"] < -5, "berlage_level"] = None  # -32.768 = sensor error

pumps = pd.read_csv("data/agv/waterkwantiteit/poldergemalen_aan_uit.csv", sep=";")
flows = pd.read_csv("data/agv/waterkwantiteit/polderdebieten_uur_fews.csv", sep=";",
                    usecols=["waarde"])

candidates = pd.DataFrame({
    "share_negative": [(storage < 0).mean(), (berlage["berlage_level"] < 0).mean(),
                       (berlage["berlage_flow"] < 0).mean(), (pumps["waarde"] < 0).mean(),
                       (flows["waarde"] < 0).mean()],
}, index=["canal level Amstelsluizen (m NAP)", "level Berlagebrug (m NAP)",
          "flow Berlagebrug (m3/day)", "pump on/off status", "polder pump flows (m3/hour)"])
candidates.round(3)
# - Canal levels are ALWAYS negative: the canals are kept ~0.4 m below NAP (sea level reference).
#   "Negative" is normal there, so it can't mean "ocean water is coming in".
# - Berlagebrug flow is negative 93% of the time = its normal flow direction.
# - Pump status -2 is an error/unknown code, pump flows are never negative.


# %% Cell 5: download real SEA level (IJmuiden, where the sea meets the canals)
# Rijkswaterstaat (free, no key). 3 years as a test: ~1 minute the first time,
# then it is saved in data/external and loads instantly.
sea = download_sea_level("2022-01-01", "2024-12-01")
sea_daily = daily_sea_level(sea)
sea_daily.head()


# %% Cell 6: the level DIFFERENCE  "storage level - sea level"
# Negative = the sea is HIGHER than the canals, so water wants to flow IN.
both = sea_daily.join(storage, how="inner")
for kind in ["sea_mean", "sea_low", "sea_high"]:
    both[f"diff_{kind}"] = both["storage_level"] - both[kind]

summary = pd.DataFrame({
    "share_negative": (both.filter(like="diff_") < 0).mean(),
    "median_m": both.filter(like="diff_").median(),
}).round(3)
summary.index = ["storage - sea (daily MEAN)", "storage - sea (at LOW tide)",
                 "storage - sea (at HIGH tide)"]
summary
# - vs daily mean / high tide: negative almost EVERY day (the sea is usually higher).
# - vs LOW tide: negative on ~8% of days. Those are the days when even at low tide
#   the sea stays above the canal level -> water can't drain out, sea water pushes in.
#   This is the best match for "negative salt level = ocean water flows in".


# %% Cell 7: CHART - what does a "negative" day look like? (one stormy month)
month = sea[(sea["time"] >= "2023-12-15") & (sea["time"] < "2024-01-15")]
days = both.loc["2023-12-15":"2024-01-14"]
negative_days = days.index[days["diff_sea_low"] < 0]

fig, ax = plt.subplots(figsize=(11, 4.5))
for day in negative_days:
    ax.axvspan(day, day + pd.Timedelta(days=1), color="#fde2e1", zorder=0)
ax.plot(month["time"], month["sea_level"], color="#2a78d6", linewidth=1, label="sea level IJmuiden")
ax.plot(days.index + pd.Timedelta(hours=12), days["storage_level"], color="#eb6834",
        linewidth=2, label="canal level Amsterdam")
ax.set_ylabel("water level (m NAP)")
ax.set_title("Pink days: even at low tide the sea stays above the canals (2023–24 winter)",
             loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e1e0d9")
fig.tight_layout()
fig.savefig("output/step1_salt_level_example.png", dpi=150)
plt.show()
