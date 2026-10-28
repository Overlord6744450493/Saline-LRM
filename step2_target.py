"""
STEP 2 - Build the target: on which days does ocean water flow in?

"salt level" = canal level - sea level at LOW tide   (chosen in step 1)
inflow = 1 when that is below 0 (even at low tide the sea is higher than the canals)

Run each cell with Shift+Enter. The first run downloads 25 years of sea level
(about 5-7 minutes); after that it loads from data/external in seconds.
"""
# %% Cell 1: imports
import matplotlib.pyplot as plt
import pandas as pd

from features import add_salt_level
from helpers import daily_sea_level, download_sea_level, load_storage_level

BLUE, ORANGE, INK_2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e1e0d9"


# %% Cell 2: load canal level (local data) and sea level (Rijkswaterstaat)
storage = load_storage_level()                       # m NAP, one value per day
sea = download_sea_level("2000-01-01", "2025-05-01")  # m NAP, every 10 minutes
sea_daily = daily_sea_level(sea)

days = sea_daily.join(storage, how="inner")          # keep days that have BOTH
print(f"{len(days):,} days with both values, from {days.index.min():%Y-%m-%d} to {days.index.max():%Y-%m-%d}")


# %% Cell 3: the "salt level" and the inflow target
# add_salt_level (features.py) does: salt_level = storage_level - sea_low, inflow = salt_level < 0
days = add_salt_level(days)

print(f"Inflow days: {days['inflow'].sum()} of {len(days)} = {days['inflow'].mean():.1%}")

# An "event" = a run of inflow days in a row (a storm can last 3 days = 1 event).
new_event = (days["inflow"] == 1) & (days["inflow"].shift(1, fill_value=0) == 0)
days["event_id"] = new_event.cumsum().where(days["inflow"] == 1)
event_lengths = days.groupby("event_id").size()
print(f"Inflow events: {len(event_lengths)}, typical length {event_lengths.median():.0f} day(s), "
      f"longest {event_lengths.max()} days")

days.to_csv("output/daily_target.csv")
days[["storage_level", "sea_low", "salt_level", "inflow"]].tail()


# %% Cell 4: how often per year and per month?
per_year = days.groupby(days.index.year)["inflow"].sum()
per_month = days.groupby(days.index.month)["inflow"].mean() * 100
print("Inflow days per year:\n", per_year.to_string())
print("\n% of days with inflow, per month:\n", per_month.round(1).to_string())


# %% Cell 5: CHART - inflow over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})

ax1.bar(per_year.index, per_year.values, color=BLUE, width=0.7)
ax1.set_title("Inflow days per year", loc="left", fontweight="bold")
ax1.set_ylabel("days")
if days.index.max().month < 12:  # last year is not complete
    ax1.text(per_year.index[-1], per_year.iloc[-1] + 0.8, f"until\n{days.index.max():%b}",
             ha="center", color=INK_2, fontsize=8)

ax2.bar(range(1, 13), per_month.values, color=BLUE, width=0.7)
ax2.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax2.set_title("Share of days with inflow, per month", loc="left", fontweight="bold")
ax2.set_ylabel("% of days")

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("output/step2_inflow_over_time.png", dpi=150)
plt.show()
