"""
STEP 6 - Turn the prediction into a pump decision.

Rule:  "If the predicted inflow probability for the day after tomorrow is above
        THRESHOLD, prepare and release the pumps."
We choose THRESHOLD on the validation years (2017-2019), then check it honestly
on the test years (2020-2025).
Needs output/predictions.csv from step 5.
"""
# %% Cell 1: imports and data
import matplotlib.pyplot as plt
import pandas as pd

from helpers import choose_threshold, evaluate_rule

BLUE, ORANGE, INK_2, GRID, PINK, MUTED = "#2a78d6", "#eb6834", "#52514e", "#e1e0d9", "#fde2e1", "#898781"

pred = pd.read_csv("output/predictions.csv", index_col="date", parse_dates=True)
# A row is made on day t and is about day t+2. Move it to the day it is ABOUT:
pred.index = pred.index + pd.Timedelta(days=2)
pred.index.name = "target_day"


# %% Cell 2: helper - count EVENTS caught and unnecessary pump days
# evaluate_rule lives in helpers.py (the web app uses it too).
# An event = consecutive inflow days; it is "caught" if at least one day had an alarm.
evaluate = evaluate_rule


# %% Cell 3: try thresholds on the VALIDATION years
valid = pred[pred["period"] == "validation"]
table = pd.DataFrame([evaluate(valid, t) for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
table.round(2)
# Low threshold = catch almost everything, but pump more often for nothing.
# High threshold = fewer wasted pump runs, but miss more events.


# %% Cell 4: choose the threshold
# Our choice: the HIGHEST threshold that still catches at least 90% of events
# (missing seawater is worse than an extra pump run). Change 0.9 if the
# stakeholder thinks differently.
# choose_threshold (helpers.py) is the same rule step 5 uses to save the threshold with the model.
THRESHOLD = choose_threshold(valid)
print("Chosen threshold:", THRESHOLD)


# %% Cell 5: honest check on the TEST years (2020-2025)
test = pred[pred["period"] == "test"]
result = evaluate(test, THRESHOLD)

# baseline for comparison: "pump in 2 days if there is inflow today"
base = test.copy()
base["prob"] = pred["inflow_t+2"].shift(2).reindex(test.index).fillna(0)  # inflow on day t
base_result = evaluate(base, 0.5)

summary = pd.DataFrame([result, base_result], index=["our rule", "baseline: inflow today"]).round(2)
summary
# -> how many events we catch, and how many pump days were not needed


# %% Cell 6: CHART - test period timeline
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(test.index, test["prob"], color=BLUE, linewidth=1, label="predicted inflow probability (2 days ahead)")
ax.axhline(THRESHOLD, color=ORANGE, linewidth=1.5, linestyle="--", label=f"pump threshold {THRESHOLD}")
inflow_days = test.index[test["inflow_t+2"] == 1]
ax.scatter(inflow_days, [1.04] * len(inflow_days), marker="|", s=80, color=INK_2, label="real inflow day")
ax.set_ylim(0, 1.1)
ax.set_ylabel("probability")
ax.set_title("Test years 2020–2025: the forecast rises before real inflow days",
             loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRID)
fig.tight_layout()
fig.savefig("output/step6_test_timeline.png", dpi=150)
plt.show()


# %% Cell 7: CHART - our rule vs baseline (events caught and wasted pump days)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
names = ["baseline\n(inflow today)", "our rule"]
caught = [base_result["share_caught"] * 100, result["share_caught"] * 100]
wasted = [base_result["unnecessary_per_year"], result["unnecessary_per_year"]]
ax1.bar(names, caught, color=[MUTED, BLUE], width=0.5)
ax1.set_title("Inflow events caught 2 days ahead", loc="left", fontweight="bold")
ax1.set_ylabel("% of events")
ax1.set_ylim(0, 110)
ax2.bar(names, wasted, color=[MUTED, BLUE], width=0.5)
ax2.set_title("Unnecessary pump days per year", loc="left", fontweight="bold")
ax2.set_ylabel("days per year")
for ax, values, fmt in [(ax1, caught, "{:.0f}%"), (ax2, wasted, "{:.0f}")]:
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, fmt.format(v), ha="center", color=INK_2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("output/step6_rule_vs_baseline.png", dpi=150)
plt.show()
