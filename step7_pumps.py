"""
STEP 7 - Which pumping stations seem to help?

What the data allows (checked first):
  + 83 pumps at 44 pumping stations, hourly flow 1996-2025 (polderdebieten_uur_fews.csv)
  - NO names, NO locations for the pump codes (KGM...), so we can't link a pump
    to an area or to a salt sample point. And our inflow signal is ONE number for
    the whole system (canal - sea at IJmuiden), not one per area.
So we rank STATIONS by their link with the SYSTEM-WIDE salt level:

  1. For every inflow day, predict how much the salt level should recover by the
     next day from the storm alone (sea, wind, tide, rain): the "expected" recovery.
  2. For each station: on days it RAN HARD, did the salt level recover FASTER than
     expected, compared with days it did not?
  3. Rank stations by that difference.

This is a CORRELATION, not proof that a pump causes the recovery.
Needs output/daily_dataset.csv (step 3).
"""
# %% Cell 1: imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

BLUE, MUTED, INK_2, GRID = "#2a78d6", "#898781", "#52514e", "#e1e0d9"


# %% Cell 2: hourly pump flows -> one volume per pump per day (first time ~1-2 minutes)
# The file is 779 MB, so we read it in chunks and only keep the daily totals.
DAILY_FILE = "output/pump_daily.csv"
if not os.path.exists(DAILY_FILE):
    parts = []
    for chunk in pd.read_csv("data/agv/waterkwantiteit/polderdebieten_uur_fews.csv", sep=";",
                             usecols=["datum", "waarde", "locatie"], chunksize=2_000_000):
        chunk["date"] = chunk["datum"].str[:10]  # "1996-01-01 05:00:00" -> "1996-01-01"
        parts.append(chunk.groupby(["locatie", "date"])["waarde"].sum())
    pump_daily = pd.concat(parts).groupby(level=[0, 1]).sum().rename("m3").reset_index()
    pump_daily.to_csv(DAILY_FILE, index=False)
pump_daily = pd.read_csv(DAILY_FILE, parse_dates=["date"])
pump_daily["station"] = pump_daily["locatie"].str[:8]  # KGM00357_P002 -> KGM00357
print(len(pump_daily), "pump-days,", pump_daily["locatie"].nunique(), "pumps,",
      pump_daily["station"].nunique(), "stations")


# %% Cell 3: station overview - pumps, capacity, how often they run
# Capacity: no capacity column exists, so we use the biggest daily volume ever
# pumped (99th percentile, to ignore glitches) as a stand-in.
station_daily = pump_daily.pivot_table(index="date", columns="station", values="m3", aggfunc="sum")
stations = pd.DataFrame({
    "pumps": pump_daily.groupby("station")["locatie"].unique().map(lambda p: ", ".join(x[9:] for x in sorted(p))),
    "capacity_m3_day": station_daily.quantile(0.99),
    "typical_m3_day": station_daily.mean(),
    "share_days_running": (station_daily > 0).mean(),
})
stations.sort_values("capacity_m3_day", ascending=False).round(2).head(15)


# %% Cell 4: join with the salt level and build the "expected recovery"
data = pd.read_csv("output/daily_dataset.csv", index_col="date", parse_dates=True).asfreq("D")
data["recovery"] = data["salt_level"].shift(-1) - data["salt_level"]  # rise by tomorrow (m)
data["wind_from_nw_next"] = data["wind_from_nw"].shift(-1)
data["tide_pred_low_next"] = data["tide_pred_low"].shift(-1)

storm = ["salt_level", "wind_from_nw", "wind_from_nw_next", "tide_pred_low_next", "rain"]
days = data[data["inflow"] == 1].dropna(subset=storm + ["recovery"])
days = days[days.index.isin(station_daily.index)]
print(len(days), "inflow days with pump data")

# Simple straight-line formula: recovery = a*salt_level + b*wind + ... + c
expected_model = LinearRegression().fit(days[storm], days["recovery"])
days["expected"] = expected_model.predict(days[storm])
days["faster_than_expected"] = days["recovery"] - days["expected"]  # > 0 = recovered faster
print(f"The storm alone explains {expected_model.score(days[storm], days['recovery']):.0%} of the recovery")


# %% Cell 5: for each station - did salt recover faster when it ran hard?
# "Ran hard" = pumped more than its own usual on rainy winter days
# (above its 75th percentile on all inflow days).
rows = []
for station in station_daily.columns:
    volume = station_daily.loc[days.index, station]
    if volume.notna().sum() < 50:  # too little data for this station
        continue
    hard = volume > volume.quantile(0.75)
    a = days.loc[hard, "faster_than_expected"]
    b = days.loc[~hard & volume.notna(), "faster_than_expected"]
    test = stats.ttest_ind(a, b, equal_var=False)
    rows.append({"station": station, "days_ran_hard": len(a),
                 "extra_recovery_cm": (a.mean() - b.mean()) * 100, "p_value": test.pvalue})

ranking = pd.DataFrame(rows).set_index("station").join(stations)
ranking["verdict"] = np.select(
    [(ranking["p_value"] < 0.1) & (ranking["extra_recovery_cm"] > 0),
     (ranking["p_value"] < 0.1) & (ranking["extra_recovery_cm"] < 0)],
    ["seems to help", "no help (salt recovered slower)"], default="no clear effect")
ranking = ranking.sort_values("extra_recovery_cm", ascending=False)
ranking.to_csv("output/station_ranking.csv")
ranking[["pumps", "capacity_m3_day", "days_ran_hard", "extra_recovery_cm", "p_value", "verdict"]].round(3)
# extra_recovery_cm: how many cm MORE the salt level rose by the next day (vs expected)
# when this station ran hard. p_value < 0.1 = probably not luck.


# %% Cell 6: summary
print(ranking["verdict"].value_counts().to_string())
# WATCH OUT: we did 44 tests. With p < 0.1, about 44 x 0.1 = 4 stations would
# look like they "help" by pure luck. So a handful of "seems to help" is NOT evidence.
print(f"Stations expected to pass by luck alone: {len(ranking) * 0.1:.0f}")
print("\nTop 5 stations:")
for station, row in ranking.head(5).iterrows():
    print(f"  {station} (pumps {row['pumps']}): +{row['extra_recovery_cm']:.1f} cm faster, "
          f"p={row['p_value']:.2f}, {row['verdict']}")


# %% Cell 7: CHART - stations ranked by extra recovery
show = ranking.iloc[::-1]  # best on top
colors = [BLUE if v == "seems to help" else MUTED for v in show["verdict"]]
fig, ax = plt.subplots(figsize=(9, 10))
ax.barh(range(len(show)), show["extra_recovery_cm"], color=colors, height=0.7)
ax.set_yticks(range(len(show)), show.index, fontsize=8)
ax.axvline(0, color=INK_2, linewidth=1)
ax.set_xlabel("extra salt-level recovery by next day when the station ran hard (cm)")
ax.set_title("No pumping station clearly speeds up recovery\n"
             "(blue = p < 0.1, but with 44 stations ~4 pass by luck alone)",
             loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("output/step7_station_ranking.png", dpi=150)
plt.show()


# %% Cell 8: all stations TOGETHER - does more total pumping go with faster recovery?
# Most stations pump hard on the same days, so test the total instead.
total = station_daily.sum(axis=1)
days["total_pumped_m3"] = total.reindex(days.index)
days["pumping_level"] = pd.qcut(days["total_pumped_m3"], 4, labels=["low", "medium", "high", "very high"])
by_level = days.groupby("pumping_level", observed=True)["faster_than_expected"].agg(["mean", "size"])
by_level["mean"] = (by_level["mean"] * 100).round(1)
by_level.columns = ["faster than expected (cm)", "inflow days"]
r, p = stats.pearsonr(days["total_pumped_m3"], days["faster_than_expected"])
print(f"Correlation total pumping vs faster recovery: r = {r:.2f}, p = {p:.3f}")
by_level
