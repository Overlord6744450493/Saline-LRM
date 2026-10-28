"""
PREDICT - will seawater flow in 2 days from now, and which pumps should be ready?

Run in the VS Code terminal:
    python predict.py                      live: today's forecast
    python predict.py --date 2023-12-19    demo: pretend it is that day (historical weather)
                                           and show what really happened 2 days later
Options:
    --canal-level -0.38   real canal level today in m NAP (it is not online; default = typical for the month)
    --threshold 0.5       pump when the probability is above this (default = saved with the model)
    --weather-only        use the model without water levels (weather forecast + astronomical tide)

Needs output/model.joblib (step 5) and output/station_ranking.csv (step 7).
The model inputs are made by features.py, the same code the model was trained with.
Writes the result to predictions/latest.csv (live) or predictions/demo_<date>.csv (demo).
"""
import argparse
import os
import sys

import joblib
import pandas as pd

from features import FeatureMismatch, build_features, check_features, check_weather_units, daily_sea_level, \
    live_daily_sea_level
from helpers import (DataSourceError, daily_inputs, download_sea_level, fetch_sea_level, forecast_weather,
                     historical_weather, load_storage_level, pump_plan, risk_label, tidy_sea_level, today_gmt1)

AREA = "Amsterdam canal system (IJmuiden)"  # the data has ONE inflow signal, so one area
TIDE_CACHE = "data/external/tide_ijmuiden_ahead.csv"
OUT_DIR = "predictions"


def stop(message, code=1):
    """Print a clear message and stop, instead of a long Python error."""
    print(f"\nCannot make a prediction: {message}")
    sys.exit(code)


# ---------- getting the data ----------
def live_sea_level(today):
    """Live sea level from Rijkswaterstaat. If RWS is down: the astronomical tide we saved
    earlier (it is known months ahead), and None as a sign that measurements are missing."""
    try:
        sea = tidy_sea_level(fetch_sea_level(today - pd.Timedelta(days=2), today + pd.Timedelta(days=3),
                                             timeout=60))
    except DataSourceError as error:
        print(f"Warning: {error}")
        if not os.path.exists(TIDE_CACHE):
            return None, None
        print("         Using the saved astronomical tide and the weather-only model instead.")
        tide = tidy_sea_level(pd.read_csv(TIDE_CACHE))
        return tide[["time", "tide_pred"]].assign(sea_level=float("nan")), "unavailable (RWS down)"
    save_tide_ahead(today)
    return sea, "Rijkswaterstaat, measured + RWS forecast"


def save_tide_ahead(today, days=120):
    """Keep ~4 months of future astronomical tide on disk, for when RWS is down."""
    if os.path.exists(TIDE_CACHE):
        cached = pd.to_datetime(pd.read_csv(TIDE_CACHE, usecols=["time"])["time"], utc=True)
        if cached.max().tz_localize(None) > today + pd.Timedelta(days=30):
            return
    try:
        raw = fetch_sea_level(today, today + pd.Timedelta(days=days), timeout=60)
        raw[raw["kind"] == "astronomisch"].to_csv(TIDE_CACHE, index=False)
    except DataSourceError:
        pass  # not important now, we try again next time


def canal_level_for(day, args, bundle, measured=None):
    """Canal level (m NAP) and where it comes from: the user, the AGV file, or the typical value."""
    if args.canal_level is not None:
        return args.canal_level, "given with --canal-level", True
    if measured is not None and day in measured.index and day - pd.Timedelta(days=1) in measured.index:
        return measured.loc[day - pd.Timedelta(days=1):day], "measured (AGV file)", True
    typical = bundle["canal_level_by_month"][day.month]
    return typical, f"typical for {day:%B} (real value is not online)", False


def live_inputs(today, args, bundle):
    try:
        weather = forecast_weather(past_days=2, forecast_days=4)
    except DataSourceError as error:
        stop(f"{error}\nThe weather forecast is the most important input, so we can't predict without it. "
             "Try again in a few minutes.")
    sea, sea_source = live_sea_level(today)
    if sea is None:
        stop("Rijkswaterstaat is down and no saved tide was found. Try again later.")
    canal, canal_source, canal_real = canal_level_for(today, args, bundle)
    days = daily_inputs(live_daily_sea_level(sea), weather, canal)
    sources = {"weather": "Open-Meteo forecast API", "sea_level": sea_source,
               "canal_level": canal_source, "canal_level_m_nap": canal}
    return days, sources, canal_real, sea_source.startswith("Rijkswaterstaat")


def demo_inputs(today, args, bundle):
    """The same daily table as live, but with HISTORICAL weather and measured sea level."""
    first, last = today - pd.Timedelta(days=2), today + pd.Timedelta(days=3)
    try:
        weather = historical_weather(first, last)
        sea = download_sea_level(first.to_period("M").to_timestamp(), last)
    except DataSourceError as error:
        stop(f"{error}\nCould not get the historical data for {today:%Y-%m-%d}.")
    measured = load_storage_level()
    canal, canal_source, canal_real = canal_level_for(today, args, bundle, measured)
    sea_daily = daily_sea_level(sea)  # measured only, exactly like in training
    days = daily_inputs(sea_daily, weather, canal)

    # What REALLY happened 2 days later (same definition as the training target)
    target = today + pd.Timedelta(days=2)
    truth = None
    if target in sea_daily.index:
        real_canal = measured.get(target, bundle["canal_level_by_month"][target.month])
        salt = real_canal - sea_daily.loc[target, "sea_low"]
        truth = {"actual_salt_level_m": round(salt, 3), "actual_inflow": int(salt < 0),
                 "actual_sea_low_m_nap": sea_daily.loc[target, "sea_low"],
                 "actual_based_on": "measured" if target in measured.index else "typical canal level"}
    sources = {"weather": "Open-Meteo historical (same files as training)",
               "sea_level": "Rijkswaterstaat measured", "canal_level": canal_source,
               "canal_level_m_nap": canal if not isinstance(canal, pd.Series) else canal.iloc[-1]}
    return days, sources, canal_real, True, truth


# ---------- predicting ----------
def predict(days, today, bundle, use_water_levels, canal_real, threshold):
    """Pick the model, build + check the inputs, return probability and details."""
    if use_water_levels:
        model, features = bundle["model"], bundle["features"]
        name = "full model (weather + tide + water levels)"
        default = bundle["threshold"] if canal_real else bundle["threshold_typical_canal_level"]
    else:
        model, features = bundle["weather_only"]["model"], bundle["weather_only"]["features"]
        name = "weather-only model (weather + astronomical tide)"
        default = bundle["weather_only"]["threshold"]

    X = build_features(days)
    if today not in X.index:
        raise FeatureMismatch(f"no data at all for {today:%Y-%m-%d}")
    X = X.loc[[today], features]
    warnings = check_features(X, features, bundle["feature_ranges"])
    prob = float(model.predict_proba(X)[0, 1])
    return prob, name, threshold if threshold is not None else default, X, warnings


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="2-day seawater inflow forecast and pump plan")
    parser.add_argument("--date", help="demo mode: pretend it is this day (YYYY-MM-DD), uses historical data")
    parser.add_argument("--canal-level", type=float, help="real canal level today in m NAP")
    parser.add_argument("--threshold", type=float, help="pump when the probability is above this (0-1)")
    parser.add_argument("--weather-only", action="store_true", help="use the model without water levels")
    args = parser.parse_args()

    if not os.path.exists("output/model.joblib"):
        stop("output/model.joblib not found. Run  python step5_model.py  first.", 2)
    bundle = joblib.load("output/model.joblib")
    if "weather_only" not in bundle:
        stop("output/model.joblib is from an older version. Run  python step5_model.py  again.", 2)
    check_weather_units(bundle["weather_units"])  # the saved model must match features.py

    demo = args.date is not None
    if demo:
        try:
            today = pd.Timestamp(args.date).normalize()
        except ValueError:
            stop(f"--date {args.date!r} is not a date. Use YYYY-MM-DD, e.g. --date 2023-12-19", 2)
        if today + pd.Timedelta(days=3) > today_gmt1():
            stop("--date must be at least 3 days in the past (for the future, run without --date).", 2)
        print(f"DEMO MODE: pretending today is {today:%A %d %B %Y}, using historical data")
        days, sources, canal_real, has_sea, truth = demo_inputs(today, args, bundle)
    else:
        today = today_gmt1()
        print(f"LIVE: forecast made on {today:%A %d %B %Y}")
        days, sources, canal_real, has_sea = live_inputs(today, args, bundle)
        truth = None
    target = today + pd.Timedelta(days=2)

    use_water = has_sea and not args.weather_only
    try:
        prob, model_name, threshold, X, warnings = predict(days, today, bundle, use_water, canal_real,
                                                          args.threshold)
    except FeatureMismatch as error:
        if not use_water:
            stop(str(error), 2)
        print(f"Warning: {error}. Trying the weather-only model instead.")
        prob, model_name, threshold, X, warnings = predict(days, today, bundle, False, canal_real,
                                                          args.threshold)
        sources["sea_level"] += " (incomplete, not used)"
    for w in warnings:
        print("Warning:", w)

    try:
        stations = pd.read_csv("output/station_ranking.csv", index_col="station")
        risk, plan = pump_plan(prob, threshold, stations)
    except FileNotFoundError:
        risk, plan = risk_label(prob, threshold), []
        if risk == "HIGH":
            plan = ["(run step7_pumps.py to get the station list)"]
    codes = [line.split(" ")[0] for line in plan]  # "KGM00110 (pumps ...)" -> "KGM00110"

    # ---------- print ----------
    action = "activate pumps at stations " + ", ".join(codes) if plan else "no action"
    print("\n" + "=" * 78)
    print(f"Forecast for {target:%a %d %b %Y}: inflow risk {risk} ({prob:.0%}) at {AREA} -> {action}")
    print("=" * 78)
    if plan:
        print("Prepare and release the pumps. Largest pumping stations:")
        for line in plan:
            print("   -", line)
        print("   Note: past data shows no station is measurably more effective than another,")
        print("   so they are ranked by capacity. Ask the operator which ones push water out to sea.")
    else:
        print(f"Probability is below the pump threshold of {threshold:.0%}.")

    row = X.iloc[0]
    print(f"\nWhy (inputs for {target:%a %d %b}, the target day):")
    print(f"   wind from north-west {row['wind_from_nw_t+2']:5.1f} m/s   (inflow likely above ~8 m/s)")
    print(f"   strongest gust       {row['gust_max_t+2']:5.1f} m/s")
    print(f"   air pressure         {row['pressure_t+2']:6.0f} hPa  (low = storm)")
    print(f"   astronomical low tide {row['tide_pred_low_t+2']:+.2f} m NAP")
    if "salt_level_t" in row:
        print(f"   salt level today     {row['salt_level_t']:+.2f} m (negative = inflow today)")
    print(f"\nModel: {model_name}, pump threshold {threshold:.0%}")
    for key in ["weather", "sea_level", "canal_level"]:
        print(f"   {key.replace('_', ' '):<12} {sources[key]}")

    if demo:
        print("\nWhat REALLY happened on", f"{target:%a %d %b %Y}:")
        if truth is None:
            print("   no measured sea level for that day")
        else:
            real = "INFLOW (salt level negative)" if truth["actual_inflow"] else "no inflow"
            right = (risk == "HIGH") == bool(truth["actual_inflow"])
            print(f"   {real}: salt level {truth['actual_salt_level_m']:+.2f} m "
                  f"({truth['actual_based_on']} canal level)")
            print("   -> the forecast was", "RIGHT" if right else "WRONG",
                  "(pump alarm)" if risk == "HIGH" else "(no alarm)")
        if today < pd.Timestamp("2017-01-01"):
            print("   Careful: the model was TRAINED on this day. Use a date in 2020 or later for an honest demo.")

    # ---------- save ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    result = {
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "mode": "demo" if demo else "live",
        "forecast_made_on": f"{today:%Y-%m-%d}", "target_day": f"{target:%Y-%m-%d}", "area": AREA,
        "inflow_probability": round(prob, 3), "risk": risk, "threshold": threshold,
        "action": action, "model": model_name,
        **{f"source_{k}": v for k, v in sources.items()},
        **{k: round(float(v), 3) for k, v in row.items()},
        **(truth or {}), "warnings": " | ".join(warnings),
    }
    path = f"{OUT_DIR}/demo_{today:%Y-%m-%d}.csv" if demo else f"{OUT_DIR}/latest.csv"
    pd.DataFrame([result]).to_csv(path, index=False)
    history = f"{OUT_DIR}/history.csv"
    pd.DataFrame([result]).to_csv(history, mode="a", header=not os.path.exists(history), index=False)
    print(f"\nSaved to {path} (and added to {history})")


if __name__ == "__main__":
    main()
