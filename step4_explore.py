"""
STEP 4 - Explore the link: what does the weather look like in the days
BEFORE ocean water flows in?
Needs output/daily_dataset.csv from step 3.
"""
# %% Cell 1: imports and data
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, INK_2, GRID, PINK = "#2a78d6", "#eb6834", "#52514e", "#e1e0d9", "#fde2e1"

data = pd.read_csv("output/daily_dataset.csv", index_col="date", parse_dates=True)
weather_cols = ["wind_from_nw", "wind", "gust_max", "pressure", "rain", "tide_pred_low"]


# %% Cell 2: average weather on inflow days vs normal days, and 1-2 days BEFORE
# .shift(-k) moves the target k days back, so we compare the weather on day t
# with whether inflow happens on day t+k.
rows = {}
for k, label in [(0, "same day"), (1, "1 day before"), (2, "2 days before")]:
    inflow_later = data["inflow"].shift(-k)
    rows[f"inflow, {label}"] = data.loc[inflow_later == 1, weather_cols].mean()
rows["normal day"] = data.loc[data["inflow"] == 0, weather_cols].mean()
compare = pd.DataFrame(rows).T.round(1)
compare
# How to read it: look at how each column changes from "normal day" to "2 days before"
# to "same day". The bigger the change, the more useful that weather is as a warning.


# %% Cell 3: how strong is each link? (correlation with inflow 2 days later)
corr = {}
for col in weather_cols + ["salt_level"]:
    corr[col] = data[col].corr(data["inflow"].shift(-2))
pd.Series(corr, name="correlation with inflow in 2 days").sort_values().round(2)
# Close to 0 = no link.  Further from 0 = stronger link.  The sign shows the direction.


# %% Cell 4: CHART - share of days with inflow, by north-west wind
bins = [-10, -2, 0, 2, 4, 6, 8, 15]
data["nw_bin"] = pd.cut(data["wind_from_nw"], bins)
share = data.groupby("nw_bin", observed=True)["inflow"].mean() * 100
counts = data.groupby("nw_bin", observed=True).size()

fig, ax = plt.subplots(figsize=(9, 4.5))
labels = ["< -2\n(from SE)", "-2–0", "0–2", "2–4", "4–6", "6–8", "> 8\n(strong NW)"]
ax.bar(labels, share.values, color=BLUE, width=0.6)
for i, (v, n) in enumerate(zip(share.values, counts.values)):
    ax.text(i, v + 1, f"{v:.0f}%", ha="center", color=INK_2, fontsize=9)
    ax.text(i, -6, f"n={n}", ha="center", color="#898781", fontsize=8)
ax.set_ylim(-8, max(share.values) + 8)
ax.set_xlabel("average wind from the north-west that day (m/s)")
ax.set_ylabel("% of days with inflow")
ax.set_title("The stronger the north-west wind, the more often seawater flows in",
             loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("output/step4_inflow_vs_wind.png", dpi=150)
plt.show()


# %% Cell 5: CHART - one winter: salt level and north-west wind
season = data.loc["2023-10-01":"2024-03-31"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for day in season.index[season["inflow"] == 1]:
    for ax in (ax1, ax2):
        ax.axvspan(day, day + pd.Timedelta(days=1), color=PINK, zorder=0)
ax1.plot(season.index, season["salt_level"], color=BLUE)
ax1.axhline(0, color=INK_2, linewidth=1)
ax1.set_ylabel("salt level (m)\ncanal − sea at low tide")
ax1.set_title("Winter 2023–24: inflow days (pink) come with strong north-west wind",
              loc="left", fontweight="bold")
ax2.plot(season.index, season["wind_from_nw"], color=ORANGE)
ax2.axhline(0, color=INK_2, linewidth=1)
ax2.set_ylabel("wind from NW (m/s)")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID)
fig.tight_layout()
fig.savefig("output/step4_winter_example.png", dpi=150)
plt.show()
