"""
STEP 10 - Seasonality, other effects, and how to COUNTER the inflow.

Part A - what else drives inflow (and does the model get better with it?)
  1. season           inflow happens September to February
  2. spring/neap tide twice a month the tide range is small and low tide stays HIGH
  3. long-term trend  the astronomical low tide rose ~5 cm since 2000, the canal level did not
  4. model gain       the step 9 models with and without these inputs (features.context_features)
Part B - what can we do about it?
  5. canal buffer     how many inflow days a higher canal level would have avoided
  6. pumps            what the polder pumps do to the canal level
  7. flushing         is the water less salty when the Amstel carries more fresh water?

These are CORRELATIONS in historical data, not proof of cause and effect.
Needs output/daily_dataset.csv (step 3), output/pump_daily.csv (step 7), output/fychem_salt.csv.
"""
# %% Cell 1: imports and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import NEW_MOON, SYNODIC_MONTH, build_features, build_target

BLUE, RED, MUTED, INK_2, GRID = "#2a78d6", "#d62728", "#898781", "#52514e", "#e1e0d9"

data = pd.read_csv("output/daily_dataset.csv", index_col="date", parse_dates=True).asfreq("D")
inflow = data["inflow"] == 1
print(f"{inflow.sum()} inflow days in {data.index.min():%Y}-{data.index.max():%Y}")


def style(ax, title):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


# %% Cell 2: SEASON - share of days with inflow, per month
by_month = data.groupby(data.index.month)["inflow"].mean() * 100
print(by_month.round(1).to_string())

# %% Cell 3: SPRING/NEAP - share of inflow days by moon phase
# Around full and new moon the tide range is LARGE (spring tide): very low low tides drain
# the canals well. Around half moon it is SMALL (neap tide): low tide stays high.
moon_age = ((data.index - NEW_MOON).total_seconds() / 86400) % SYNODIC_MONTH
by_phase = data.groupby(pd.cut(moon_age, np.linspace(0, SYNODIC_MONTH, 9)), observed=True)["inflow"].mean() * 100
by_phase.index = [f"day {i.left:.0f}-{i.right:.0f}" for i in by_phase.index]
print("inflow % by moon age (0 = new moon, ~15 = full moon):")
print(by_phase.round(1).to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.bar(by_month.index, by_month, color=[RED if v > 5 else MUTED for v in by_month])
ax1.set_xticks(range(1, 13), list("JFMAMJJASOND"))
ax1.set_ylabel("% of days with inflow")
style(ax1, "Inflow season: September to February")
ax2.bar(range(len(by_phase)), by_phase, color=BLUE)
ax2.set_xticks(range(len(by_phase)), ["new", "", "1st qtr", "", "full", "", "last qtr", ""])
ax2.set_ylabel("% of days with inflow")
style(ax2, "Neap tides (half moon) bring more inflow")
fig.tight_layout()
fig.savefig("output/step10_season_moon.png", dpi=150)
plt.show()


# %% Cell 4: TREND - the safety margin is shrinking
# margin = canal level - astronomical low tide: how much room there is before a storm surge
winter = data[data.index.month.isin([9, 10, 11, 12, 1, 2])]
winter = winter[winter.index < "2025-01-01"]  # 2025 only has January-February
per_year = winter.groupby(winter.index.year).agg(
    inflow_days=("inflow", "sum"), canal=("storage_level", "mean"), tide_low=("tide_pred_low", "mean"))
per_year["margin_cm"] = (per_year["canal"] - per_year["tide_low"]) * 100
print(per_year.round(3).to_string())
trend = np.polyfit(per_year.index, per_year["tide_low"] * 100, 1)[0]
print(f"Astronomical low tide rises {trend * 10:.1f} cm per 10 years; "
      f"margin {per_year['margin_cm'].iloc[:5].mean():.0f} cm (2000-04) -> "
      f"{per_year['margin_cm'].iloc[-5:].mean():.0f} cm (2020-24)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(per_year.index, per_year["inflow_days"], color=MUTED, label="inflow days (Sep-Feb)")
ax.set_ylabel("inflow days per winter")
ax2 = ax.twinx()
ax2.plot(per_year.index, per_year["margin_cm"], color=RED, marker="o", label="margin (cm)")
ax2.set_ylabel("canal level - astronomical low tide (cm)", color=RED)
style(ax, "The margin between canal and low tide is shrinking")
fig.tight_layout()
fig.savefig("output/step10_trend.png", dpi=150)
plt.show()


# %% Cell 5: MODEL GAIN - do these inputs make the forecast better?
# Same recipe as step 9 (logistic regression), with and without features.context_features.
def test_score(h, context):
    X, y = build_features(data, h, context), build_target(data, h)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok].astype(int)
    train, test = X.index < "2017-01-01", X.index >= "2020-01-01"
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
    model.fit(X[train], y[train])
    prob = model.predict_proba(X[test])[:, 1]
    return average_precision_score(y[test], prob), roc_auc_score(y[test], prob)


rows = {}
for h in range(1, 8):
    (ap_without, auc_without), (ap_with, auc_with) = test_score(h, False), test_score(h, True)
    rows[h] = {"without": ap_without, "with seasonality & effects": ap_with,
               "AUC without": auc_without, "AUC with seasonality & effects": auc_with}
gain = pd.DataFrame(rows).T.rename_axis("horizon_days")
gain["gain"] = gain["with seasonality & effects"] - gain["without"]
gain["AUC gain"] = gain["AUC with seasonality & effects"] - gain["AUC without"]
print("TEST 2020-2025, average precision (AP) and area under the ROC curve (AUC):")
print(gain.round(3).to_string())
gain.to_csv("output/step10_feature_gain.csv")


# %% Cell 6: CANAL BUFFER - what if the canal level had been higher?
# Inflow = canal level below the sea at low tide. Every cm the canal is higher is a cm of margin.
deficit_cm = -data.loc[inflow, "salt_level"] * 100  # how far below 0 the salt level was
buffer = pd.DataFrame({"raise_cm": range(0, 41)})
buffer["share_avoided"] = [(deficit_cm < cm).mean() for cm in buffer["raise_cm"]]
buffer.to_csv("output/step10_canal_buffer.csv", index=False)
for cm in [5, 10, 15, 20, 30]:
    print(f"canal {cm:>2} cm higher -> {(deficit_cm < cm).mean():.0%} of inflow days avoided")

# Do the operators already raise the canal before a storm? (canal level vs normal for the month)
normal = data.groupby(data.index.month)["storage_level"].transform("median")
start = inflow & ~inflow.shift(1, fill_value=False)
around = pd.Series({lag: ((data["storage_level"] - normal).shift(-lag)[start] * 100).mean()
                    for lag in range(-3, 4)})
print("canal level vs normal around the start of an inflow event (cm):")
print(around.round(1).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(buffer["raise_cm"], buffer["share_avoided"] * 100, color=BLUE, linewidth=2)
for cm in [10, 20]:
    share = buffer.loc[buffer["raise_cm"] == cm, "share_avoided"].iloc[0] * 100
    ax.annotate(f"+{cm} cm: {share:.0f}%", (cm, share), textcoords="offset points", xytext=(8, -12))
    ax.plot(cm, share, "o", color=BLUE)
ax.set_xlabel("canal level kept higher by (cm)")
ax.set_ylabel("% of inflow days avoided")
style(ax, "A higher canal level before a storm avoids most inflow days")
fig.tight_layout()
fig.savefig("output/step10_canal_buffer.png", dpi=150)
plt.show()


# %% Cell 7: PUMPS - what do the polder pumps do to the canal level?
# The polder pumps lift water FROM the polders INTO the canals, so they raise the canal level.
pumped = pd.read_csv("output/pump_daily.csv", parse_dates=["date"]).groupby("date")["m3"].sum()
data["pumped_million_m3"] = pumped.reindex(data.index) / 1e6
data["canal_rise_cm"] = (data["storage_level"].shift(-1) - data["storage_level"]) * 100
cols = ["pumped_million_m3", "rain", "inflow", "wind_from_nw"]
ok = data[cols + ["canal_rise_cm"]].dropna()
fit = LinearRegression().fit(ok[cols], ok["canal_rise_cm"])
per_million = fit.coef_[0]
print(f"polder pumping: {data.loc[~inflow, 'pumped_million_m3'].mean():.2f} million m3/day normally, "
      f"{data.loc[inflow, 'pumped_million_m3'].mean():.2f} on inflow days")
print(f"each million m3 pumped goes with +{per_million:.2f} cm canal level the next day "
      f"-> +10 cm would take ~{10 / per_million:.0f} million m3, "
      f"~{10 / per_million / data['pumped_million_m3'].mean():.0f} days of normal pumping")


# %% Cell 8: FLUSHING - is the water less salty when the Amstel carries more fresh water?
# Chloride samples (fychem) compared with each location's own median (log scale),
# against the Amstel discharge at Berlagebrug (7-day mean, 2012-2025).
berlage = pd.read_csv("data/agv/waterkwantiteit/Berlagebrug_debiet_waterhoogte.csv", sep=";", skiprows=[1])
berlage.index = pd.to_datetime(berlage["GMT+1"], dayfirst=True).dt.normalize()
discharge = berlage["MPP00976_FIT001_Q"].where(berlage["MPP00976_FIT001_Q"] > -900)  # -999 = missing
discharge_7d = discharge.asfreq("D").rolling(7, min_periods=4).mean()

salt = pd.read_csv("output/fychem_salt.csv", parse_dates=["datum"])
salt = salt[(salt["datum"] >= "2000-01-01") & (salt["meetwaarde"] > 0)]
salt["date"] = salt["datum"].dt.normalize()
salt["anomaly"] = np.log(salt["meetwaarde"]) - salt.groupby("locatiecode")["meetwaarde"].transform(
    lambda v: np.log(v).median())
salt["amstel_7d"] = salt["date"].map(discharge_7d)
salt["inflow_last_14d"] = salt["date"].map(data["inflow"].rolling(14, min_periods=1).sum())

with_flow = salt.dropna(subset=["amstel_7d"])
flushing = with_flow.groupby(pd.qcut(with_flow["amstel_7d"], 4, labels=["low", "medium", "high", "very high"]),
                             observed=True)["anomaly"].agg(["mean", "size"])
flushing["chloride_vs_normal"] = (np.exp(flushing["mean"]) - 1).map("{:+.0%}".format)
print("chloride vs normal, by Amstel discharge:")
print(flushing[["chloride_vs_normal", "size"]].to_string())

after = salt.groupby(pd.cut(salt["inflow_last_14d"], [-1, 0, 2, 5, 14], labels=["0", "1-2", "3-5", "6+"]),
                     observed=True)["anomaly"].agg(["mean", "size"])
after["chloride_vs_normal"] = (np.exp(after["mean"]) - 1).map("{:+.0%}".format)
print("chloride vs normal, by inflow days in the 14 days before the sample:")
print(after[["chloride_vs_normal", "size"]].to_string())
# Surprise: the samples are NOT saltier after inflow. Most sample points are in the polders, where
# the salt comes from brackish seepage water; storms bring rain that dilutes it. So the polder samples
# can't show seawater inflow - a measuring point near IJmuiden / the Noordzeekanaal would be needed.
