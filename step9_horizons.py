"""
STEP 9 - One model per forecast horizon: inflow in 1, 2, 3, ... 7 days.

Same recipe as step 5 (which is the 2-day model), repeated for every horizon h:
  inputs:  weather for t, t+h-1 and t+h, astronomical tide on t+h, water levels up to today,
           plus the seasonality and other effects from step 10 (features.context_features)
  target:  inflow on day t+h
  split:   train 2000-2016 | validation 2017-2019 | test 2020-2025
  model:   logistic regression or random forest, whichever is best on validation
  calibration: the balanced class weights make the raw probabilities 2-5x too high, so they are
             rescaled on the validation years (Platt scaling: a logistic fit on the log-odds)
  threshold: the highest one that still catches 90% of events in the validation years

Saves output/models_horizons.joblib (used by the web app, also everything its "Model audit" page
shows: all predictions, baselines, the model comparison and input importance) and
output/horizon_scores.csv.
Needs output/daily_dataset.csv from step 3.
One-time install:  python -m pip install scikit-learn shap
"""
# %% Cell 1: imports and data
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features, build_target
from helpers import choose_threshold, evaluate_rule, predict_inflow, shap_values

HORIZONS = range(1, 8)
# calibrated probabilities are lower than the raw ones, so the threshold options are too
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)

data = pd.read_csv("output/daily_dataset.csv", index_col="date", parse_dates=True)
data = data.asfreq("D")


def make_models():
    return {
        "logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=0, n_jobs=-1),
    }


def as_rule_input(prob, y, part, horizon):
    """Predictions -> the table evaluate_rule wants (index = the day the prediction is about)."""
    # evaluate_rule calls the truth column "inflow_t+2"; here it is the inflow on day t+horizon
    table = pd.DataFrame({"prob": prob, "inflow_t+2": y}, index=y.index)[part]
    table.index = table.index + pd.Timedelta(days=horizon)
    return table


# %% Cell 2: train one model per horizon
bundle = {"horizons": {}}
scores = {}
for h in HORIZONS:
    X = build_features(data, h, context=True)
    y = build_target(data, h)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok].astype(int)
    train = X.index < "2017-01-01"
    valid = (X.index >= "2017-01-01") & (X.index < "2020-01-01")
    test = X.index >= "2020-01-01"

    models = make_models()
    valid_score = {}
    for name, model in models.items():
        model.fit(X[train], y[train])
        valid_score[name] = average_precision_score(y[valid], model.predict_proba(X[valid])[:, 1])
    best_name = max(valid_score, key=valid_score.get)
    best = models[best_name]

    target_month = pd.Series((X.index + pd.Timedelta(days=h)).month, index=X.index)
    month_rate = y[train].groupby(target_month[train]).mean()

    raw = best.predict_proba(X)[:, 1].clip(1e-6, 1 - 1e-6)
    log_odds = np.log(raw / (1 - raw)).reshape(-1, 1)
    platt = LogisticRegression().fit(log_odds[valid], y[valid])
    entry = {"model": best, "features": list(X.columns),
             "calibration": (float(platt.coef_[0, 0]), float(platt.intercept_[0]))}
    prob = predict_inflow(entry, X)
    threshold = choose_threshold(as_rule_input(prob, y, valid, h), options=THRESHOLDS)
    result = evaluate_rule(as_rule_input(prob, y, test, h), threshold)
    scores[h] = {
        "model": best_name,
        "average_precision": average_precision_score(y[test], prob[test]),
        "threshold": threshold,
        "events_caught": result["share_caught"],
        "alarm_right": result["alarm_right"],
        "unnecessary_per_year": result["unnecessary_per_year"],
    }
    print(f"t+{h}: {best_name:<20} test average precision {scores[h]['average_precision']:.2f}, "
          f"catches {result['share_caught']:.0%} of events, {result['alarm_right']:.0%} of alarms right")

    entry.update({
        "name": best_name, "threshold": threshold,
        # a sample of training days: the "typical day" SHAP compares each forecast with
        "background": X[train].sample(200, random_state=0),
        "feature_ranges": {c: (float(X.loc[train, c].min()), float(X.loc[train, c].max()))
                           for c in X.columns},
        "test_scores": scores[h],
        # for the audit page: every prediction (index = day t, the forecast is for t+h),
        # how both candidate models did on validation, and two simple baselines on the test years
        "predictions": pd.DataFrame({"inflow": y, "prob": prob, "raw_prob": raw, "period": "train",
                                     # two simple baselines to compare with (audit page):
                                     # persistence = "inflow then if there is inflow today"
                                     "persistence": X["inflow_t"],
                                     # season only = share of inflow days in the target day's calendar
                                     # month in the training years: what you know without any forecast
                                     "season_only": target_month.map(month_rate)}, index=X.index),
        "validation_scores": valid_score,
    })
    entry["predictions"].loc[valid, "period"] = "validation"
    entry["predictions"].loc[test, "period"] = "test"
    # which inputs matter most on the test years: average size of their SHAP value
    entry["importance"] = shap_values(entry, X[test])[0].abs().mean().sort_values(ascending=False)
    bundle["horizons"][h] = entry

# %% Cell 3: how fast does the skill drop with the horizon?
scores = pd.DataFrame(scores).T.rename_axis("horizon_days")
print(scores.round(2).to_string())
scores.to_csv("output/horizon_scores.csv")

# %% Cell 4: save for the web app
bundle["trained_on"] = "2000-01-01 to 2016-12-31"
bundle["periods"] = {"train": "2000-2016", "validation": "2017-2019", "test": "2020-2025"}
joblib.dump(bundle, "output/models_horizons.joblib")
print("saved output/models_horizons.joblib and output/horizon_scores.csv")
