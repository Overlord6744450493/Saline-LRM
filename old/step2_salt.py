"""
STEP 2 - Find the salt-water signal: WHERE is it salty, and WHEN?
Run each cell with Shift+Enter, top to bottom.
Needs output/fychem_salt.csv from step 1 (cell 6).
If you get "No module named matplotlib", run this once in the terminal:
    python -m pip install matplotlib
"""
# %% Cell 1: imports and chart style
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "output"

# One consistent colour set for all our charts (same look on every slide).
BLUE = "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # max 5 lines
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DRINKING_LIMIT = 150  # mg/l chloride, the Dutch drinking-water limit (for scale)

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 10, "lines.linewidth": 2,
})


# %% Cell 2: load the salt measurements we saved in step 1
salt = pd.read_csv(f"{OUT}/fychem_salt.csv", parse_dates=["datum"])

# Give the parameters simple names. Both Cl codes are chloride (mg/l);
# GELDHD_mS/m_25oC is conductivity (EC) in mS/m. We skip the others.
salt["kind"] = salt["fewsparameter"].map({
    "Cl_mg/l": "chloride", "Cl_mg/l_nf": "chloride", "GELDHD_mS/m_25oC": "EC"})
salt = salt.dropna(subset=["kind", "meetwaarde"])

# Remove impossible values (typing errors in the database, e.g. 130000 mg/l).
# Seawater is ~19000 mg/l chloride and ~5000 mS/m, so anything above is wrong.
impossible = ((salt["kind"] == "chloride") & (salt["meetwaarde"] > 20000)) | \
             ((salt["kind"] == "EC") & (salt["meetwaarde"] > 5000))
print("Removed", impossible.sum(), "impossible values")
salt = salt[~impossible]

salt["year"] = salt["datum"].dt.year
salt["month"] = salt["datum"].dt.month
print(salt.groupby("kind")["meetwaarde"].describe().round(0))


# %% Cell 3: is there salt data INSIDE the sewage system? (check, don't assume)
# rwzi Uithoorn has a conductivity sensor on its influent (incoming sewage):
# 11.UIwlON.mi-001.GELDHD.25. Let's count how many values it actually has.
tai_ui = pd.read_csv("data/tai/MeasurementValues/NL.11.UI.csv", encoding_errors="replace")
n = tai_ui["MeasurementPointID"].str.contains("GELDHD").sum()
print("Uithoorn influent conductivity values:", n)
print("Uithoorn data covers", tai_ui["Date"].min(), "to", tai_ui["Date"].max())
# RESULT: 0 values. The sensor is listed, but no data was delivered.
# So we have NO salt measurements inside the sewage system. Our best
# alternative is the salt in the surface water next to the pumping stations
# (= the water that can leak into the sewers).


# %% Cell 4: salt at the PUMPING STATIONS (gemalen)
# Many sample points are "Voor krooshek gemaal X" = right in front of pump X.
def pump_name(description):
    """'Voor krooshek gemaal Botshol groot' -> 'Botshol groot'."""
    m = re.search(r"gemaal\s+(.+?)(?:\s+(?:langs|op|thv|bij|eind|aan)\b.*)?$",
                  description, flags=re.IGNORECASE)
    return m.group(1) if m else description


recent = salt[salt["year"] >= 2010]  # recent years = today's situation
# "voor ... gemaal" / "bij ... gemaal" = sampled right at the pump. We skip
# points like "amstel 500m-n.gemaal middelpolder" (in the river, 500 m away).
at_pumps = recent[recent["locatie omschrijving"].str.contains(
                      r"(?:voor|bij|naar)\b.*gemaal", case=False, na=False)
                  & (recent["kind"] == "chloride")]

pump_salt = (at_pumps.groupby(["locatiecode", "locatie omschrijving", "locatie x", "locatie y"])
             ["meetwaarde"].agg(median="median", max="max", n="size")
             .reset_index()
             .query("n >= 20")  # only locations with enough samples
             .sort_values("median", ascending=False))
pump_salt["pump"] = pump_salt["locatie omschrijving"].map(pump_name)
pump_salt["times_drinking_limit"] = (pump_salt["median"] / DRINKING_LIMIT).round(1)
pump_salt.to_csv(f"{OUT}/salt_at_pumps.csv", index=False)

print(len(pump_salt), "pump locations with enough chloride samples (2010+)")
pump_salt[["locatiecode", "pump", "median", "max", "n", "times_drinking_limit"]].head(15)


# %% Cell 5: CHART 1 - which pumping stations have the saltiest water?
top15 = pump_salt.head(15).iloc[::-1]  # reverse so the saltiest is on top
y = np.arange(len(top15))  # bar positions 0, 1, 2, ...
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(y, top15["median"], color=BLUE, height=0.6)
ax.set_yticks(y, top15["pump"])
ax.axvline(DRINKING_LIMIT, color=INK_2, linewidth=1, linestyle="--")
ax.text(DRINKING_LIMIT + 15, -0.9, "drinking-water limit 150 mg/l", color=INK_2, fontsize=9)
for pos, v in zip(y, top15["median"]):
    ax.text(v + 15, pos, f"{v:,.0f}", va="center", color=INK_2, fontsize=9)
ax.set_title("Saltiest water at pumping stations (median chloride, 2010–2025)")
ax.set_xlabel("chloride (mg/l)")
ax.grid(axis="y", visible=False)
ax.set_ylim(-1.3, len(top15) - 0.4)
fig.tight_layout()
fig.savefig(f"{OUT}/chart1_salt_per_pump.png", dpi=150)
plt.show()


# %% Cell 6: CHART 2 - map: where is the salt? (all sample points, 2010+)
# 'locatie x' and 'locatie y' are Dutch RD coordinates in metres.
cl_recent = recent[recent["kind"] == "chloride"]
points = (cl_recent.groupby(["locatiecode", "locatie x", "locatie y"])["meetwaarde"]
          .median().reset_index().sort_values("meetwaarde"))  # salty dots drawn last

fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(points["locatie x"], points["locatie y"], c=points["meetwaarde"],
                cmap="Blues", vmin=0, vmax=1000, s=22, edgecolors="#fcfcfb", linewidths=0.5)
for _, row in pump_salt.head(5).iterrows():  # label the 5 saltiest pumps
    ax.annotate(row["pump"], (row["locatie x"], row["locatie y"]), xytext=(6, 4),
                textcoords="offset points", fontsize=9, color=INK)
fig.colorbar(sc, ax=ax, shrink=0.6, label="median chloride (mg/l), capped at 1000")
ax.set_title("Where is the water salty? (each dot = a sample point)")
ax.set_aspect("equal")
ax.margins(x=0.2)  # room for the labels on the right
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax.spines[:].set_visible(False)  # a map needs no axis lines
fig.tight_layout()
fig.savefig(f"{OUT}/chart2_salt_map.png", dpi=150)
plt.show()
# Tip: Amsterdam centre is around x=121000, y=487000. The saltiest points are
# in the south-east (Bijlmer / Bullewijk / Botshol / De Winkel).


# %% Cell 7: WHEN is it salty? Month-by-month pattern
# Every location has its own "normal" salt level. To compare locations fairly,
# we divide each value by that location's median: 1.0 = normal, 1.2 = 20% saltier.
cl_recent = cl_recent.assign(
    relative=cl_recent["meetwaarde"] / cl_recent.groupby("locatiecode")["meetwaarde"].transform("median"))
season = cl_recent.groupby("month")["relative"].median()
print("Salt compared with normal, per month (all locations, 2010+):")
print(season.round(2).to_string())


# %% Cell 8: CHART 3 - seasonal pattern at the 5 saltiest pumping stations
top5 = pump_salt.head(5)
fig, ax = plt.subplots(figsize=(9, 5))
for color, (_, row) in zip(SERIES, top5.iterrows()):
    data = salt[(salt["locatiecode"] == row["locatiecode"]) & (salt["kind"] == "chloride")
                & (salt["year"] >= 2010)]
    monthly = data.groupby("month")["meetwaarde"].median()
    ax.plot(monthly.index, monthly.values, color=color, marker="o", markersize=4, label=row["pump"])
ax.axhline(DRINKING_LIMIT, color=INK_2, linewidth=1, linestyle="--")
ax.text(1, DRINKING_LIMIT + 30, "drinking-water limit 150 mg/l", color=INK_2, fontsize=9)
ax.axvspan(4.5, 9.5, color=GRID, alpha=0.5, zorder=0)
ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
ax.text(7, ax.get_ylim()[1] * 0.96, "May–Sep: saltiest", ha="center", color=INK_2, fontsize=9)
ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("median chloride (mg/l)")
ax.set_title("Salt peaks in summer at the saltiest pumping stations (2010–2025)")
ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
fig.savefig(f"{OUT}/chart3_salt_by_month.png", dpi=150)
plt.show()


# %% Cell 9: is it getting worse over the years? (5-year medians)
five_years = (salt[salt["locatiecode"].isin(top5["locatiecode"]) & (salt["kind"] == "chloride")]
              .assign(period=lambda d: d["year"] // 5 * 5)
              .groupby(["locatiecode", "period"])["meetwaarde"].median()
              .unstack().round(0))
five_years
# NOTE: the "2025" column only has data until June 2025, so treat it with care.


# %% Cell 10: CHART 4 - can a cheap conductivity sensor replace a chloride lab test?
# Find samples where BOTH chloride and EC were measured (same place, same time).
pairs = salt.pivot_table(index=["locatiecode", "datum"], columns="kind",
                         values="meetwaarde", aggfunc="mean").dropna()
slope, intercept = np.polyfit(pairs["EC"], pairs["chloride"], 1)
r = pairs["EC"].corr(pairs["chloride"])
print(f"{len(pairs):,} paired samples. Rule: chloride = {slope:.2f} x EC {intercept:+.0f}  (r = {r:.2f})")
ec_at_limit = (DRINKING_LIMIT - intercept) / slope
print(f"So chloride 150 mg/l is about EC {ec_at_limit:.0f} mS/m")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(pairs["EC"], pairs["chloride"], s=8, color=BLUE, alpha=0.15, linewidths=0)
x = np.array([0, 700])
ax.plot(x, slope * x + intercept, color=INK, linewidth=1.5)
ax.text(30, 1850, f"chloride ≈ {slope:.1f} × EC {intercept:+.0f}\nr = {r:.2f}",
        color=INK, fontsize=11, va="top")
ax.set_xlim(0, 700); ax.set_ylim(0, 2000)
ax.set_xlabel("conductivity EC (mS/m)")
ax.set_ylabel("chloride (mg/l)")
ax.set_title("Conductivity predicts chloride well")
fig.tight_layout()
fig.savefig(f"{OUT}/chart4_chloride_vs_ec.png", dpi=150)
plt.show()


# %% Cell 11: summary of what we found
print("SALTIEST PUMPING STATIONS (median chloride 2010+):")
for _, row in top5.iterrows():
    print(f"  {row['locatiecode']}  {row['pump']:<35} {row['median']:>6,.0f} mg/l"
          f"  ({row['times_drinking_limit']}x drinking-water limit)")
print(f"\nSaltiest months: {season.idxmax()} (x{season.max():.2f} normal), "
      f"least salty: {season.idxmin()} (x{season.min():.2f} normal)")
print(f"Conductivity rule: chloride = {slope:.2f} x EC {intercept:+.0f}, r = {r:.2f}")
print("Charts saved in the output folder.")
