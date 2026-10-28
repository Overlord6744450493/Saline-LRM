"""
STEP 5 - Machine learning: predict inflow 2 DAYS in advance.

Every morning of day t we ask: "will ocean water flow in on day t+2?"
What we may use on day t:
  - weather FORECAST for day t, t+1 and t+2 (in real life from a weather forecast;
    here we use the real weather as a stand-in, see limitations)
  - predicted ASTRONOMICAL tide for day t+2 (moon + sun, known years ahead)
  - measured canal / sea levels up to day t ONLY (no peeking into the future!)

Split by TIME:  train 2000-2016 | validation 2017-2019 (to choose settings) | test 2020-2025
Needs output/daily_dataset.csv from step 3.
One-time install:  python -m pip install scikit-learn
"""
# %% Cell 1: imports and data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import LOCATION, WEATHER_UNITS, build_features, build_target, weather_only
from helpers import choose_threshold, evaluate_rule

data = pd.read_csv("output/daily_dataset.csv", index_col="date", parse_dates=True)
data = data.asfreq("D")  # one row per calendar day, so shift(2) really means "2 days"


# %% Cell 2: build the features (one row = "today", day t)
# The inputs are made in features.py, the SAME code predict.py uses for the live forecast:
#   weather for t, t+1, t+2 (wind_from_nw, wind, gust_max, pressure, pressure_min, rain)
#   astronomical tide on t+2, and water levels of today and yesterday (no peeking ahead).
X = build_features(data)
y = build_target(data)  # TARGET: inflow on day t+2

ok = X.notna().all(axis=1) & y.notna()  # drop days with missing values
X, y = X[ok], y[ok].astype(int)
print(f"{len(X):,} usable days, {X.shape[1]} features, inflow in 2 days on {y.mean():.1%} of them")


# %% Cell 3: split by time
train = X.index < "2017-01-01"
valid = (X.index >= "2017-01-01") & (X.index < "2020-01-01")
test = X.index >= "2020-01-01"
for name, part in [("train", train), ("validation", valid), ("test", test)]:
    print(f"{name:<11} {X.index[part].min():%Y-%m-%d} to {X.index[part].max():%Y-%m-%d}: "
          f"{part.sum():>5} days, {y[part].sum():>4} inflow days")


# %% Cell 4: helper to print results in plain words
def report(name, y_true, alarm):
    p = precision_score(y_true, alarm, zero_division=0)
    r = recall_score(y_true, alarm)
    print(f"{name:<22} precision {p:5.0%}  recall {r:5.0%}   "
          f"-> {p:.0%} of alarms are right, we catch {r:.0%} of inflow days "
          f"({alarm.sum()} alarms, {y_true.sum()} inflow days)")
    return p, r


# %% Cell 5: BASELINE - "inflow in 2 days = inflow today"
baseline = X["inflow_t"]
print("TEST period 2020-2025:")
p, r = report("baseline (persistence)", y[test], baseline[test])


# %% Cell 6: train two models on 2000-2016
# class_weight="balanced": inflow is rare, so tell the model each inflow day counts more.
models = {
    "logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
    "random forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
        random_state=0, n_jobs=-1),
}
valid_score = {}
for name, model in models.items():
    model.fit(X[train], y[train])
    prob = model.predict_proba(X[valid])[:, 1]
    valid_score[name] = average_precision_score(y[valid], prob)
    print(f"{name:<20} validation score (average precision): {valid_score[name]:.2f}")
# Average precision: 1.0 = perfect, the share of inflow days (~0.05) = random guessing.
best_name = max(valid_score, key=valid_score.get)
print("Best on validation ->", best_name)


# %% Cell 7: results on the TEST period (2020-2025), never seen during training
print("TEST period 2020-2025, alarm when probability > 0.5:")
report("baseline (persistence)", y[test], baseline[test])
for name, model in models.items():
    prob = model.predict_proba(X[test])[:, 1]
    report(name, y[test], (prob > 0.5).astype(int))
    print(f"{'':<22} average precision {average_precision_score(y[test], prob):.2f}")


# %% Cell 8: which features matter most? (the random forest can tell us this easily)
importance = pd.Series(models["random forest"].feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(12).round(3)


# %% Cell 9: how much does the WEATHER FORECAST help?
# Same kind of model, but WITHOUT the forecast for t+1 and t+2 (only weather up to today).
no_forecast = [c for c in X.columns if not (c.endswith("_t+1") or c.endswith("_t+2"))
               or c.startswith("tide")]
model_now = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
model_now.fit(X.loc[train, no_forecast], y[train])
prob_now = model_now.predict_proba(X.loc[test, no_forecast])[:, 1]
p, r = report("without forecast", y[test], (prob_now > 0.5).astype(int))
print(f"{'':<22} average precision {average_precision_score(y[test], prob_now):.2f}")


# %% Cell 10: save predictions of the BEST model for step 6
everything = pd.DataFrame(index=X.index)
everything["inflow_t+2"] = y
everything["prob"] = models[best_name].predict_proba(X)[:, 1]
everything["period"] = "train"
everything.loc[valid, "period"] = "validation"
everything.loc[test, "period"] = "test"
everything.to_csv("output/predictions.csv")
print("saved output/predictions.csv")


# %% Cell 11: FALLBACKS - what if we don't have live water levels?
# The canal level (Amstelsluizen) is only in the AGV files we downloaded (they end in May 2025);
# it is not online. The sea level IS online (Rijkswaterstaat, free, no key), but can be down.
# We check two fallbacks on the same test years:
#   A) canal level unknown -> use the typical canal level for that month (from the TRAINING years)
#   B) no water levels at all -> a weather-only model (weather forecast + astronomical tide)
canal_by_month = data.loc[:"2016-12-31", "storage_level"].groupby(data.loc[:"2016-12-31"].index.month).median()

typical = data.copy()
typical["storage_level"] = typical.index.month.map(canal_by_month)
typical["salt_level"] = typical["storage_level"] - typical["sea_low"]
typical["inflow"] = (typical["salt_level"] < 0).astype(int).where(typical["salt_level"].notna())
X_typical = build_features(typical).loc[X.index]

weather_features = weather_only(list(X.columns))
model_weather = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
model_weather.fit(X.loc[train, weather_features], y[train])


def as_rule_input(prob, period):
    """Predictions -> the table evaluate_rule wants (index = the day the prediction is about)."""
    part = pd.DataFrame({"prob": prob, "inflow_t+2": y}, index=X.index)[period]
    part.index = part.index + pd.Timedelta(days=2)
    return part


best = models[best_name]
variants = {
    "full model (real canal level)": (best, X, list(X.columns)),
    "A: typical canal level": (best, X_typical, list(X.columns)),
    "B: weather + tide only": (model_weather, X, weather_features),
}
thresholds, scores = {}, {}
for name, (model, inputs, cols) in variants.items():
    prob = model.predict_proba(inputs[cols])[:, 1]
    # the threshold of a variant is chosen on the validation years, like in step 6
    thresholds[name] = choose_threshold(as_rule_input(prob, valid))
    result = evaluate_rule(as_rule_input(prob, test), thresholds[name])
    scores[name] = {"average_precision": average_precision_score(y[test], prob[test]),
                    "threshold": thresholds[name], "events_caught": result["share_caught"],
                    "alarm_right": result["alarm_right"],
                    "unnecessary_per_year": result["unnecessary_per_year"]}
scores = pd.DataFrame(scores).T
print("TEST period 2020-2025:")
print(scores.round(2).to_string())


# %% Cell 12: save the trained models, so predict.py and the web app (app.py) can use them
# Saved with the model: the exact input names and order, the pump threshold, the units of
# the weather data, and the range of every input (predict.py checks new data against these).
import joblib  # comes with scikit-learn

bundle = {
    "model": best, "name": best_name, "features": list(X.columns),
    "threshold": thresholds["full model (real canal level)"],
    "weather_only": {"model": model_weather, "features": weather_features,
                     "threshold": thresholds["B: weather + tide only"]},
    "canal_level_by_month": canal_by_month.round(3).to_dict(),
    "threshold_typical_canal_level": thresholds["A: typical canal level"],
    "weather_units": WEATHER_UNITS, "location": LOCATION, "water_level_unit": "m NAP",
    "feature_ranges": {c: (float(X.loc[train, c].min()), float(X.loc[train, c].max())) for c in X.columns},
    "trained_on": f"{X.index[train].min():%Y-%m-%d} to {X.index[train].max():%Y-%m-%d}",
    "test_scores": scores.to_dict("index"),
}
joblib.dump(bundle, "output/model.joblib")
print("saved output/model.joblib")
