"""
FEATURES - the ONE place where model inputs are made.

Training (step2/step3/step5), the live forecast (predict.py) and the web app all
call these functions, so the inputs can never get out of sync.

The chain:
    hourly weather (Open-Meteo)   --hourly_to_daily-->   one row per day
    10-minute sea level (RWS)     --daily_sea_level-->   one row per day
    + canal level                 --add_salt_level-->    salt_level, inflow
    all daily columns together    --build_features-->    the model inputs for day t
                                                         (24 for the 2-day model)
"""
import numpy as np
import pandas as pd

# ---------- weather: what we ask Open-Meteo for, and in which units ----------
HOURLY_VARS = "precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
WEATHER_UNITS = {  # the units Open-Meteo must answer with (we ask wind_speed_unit=ms)
    "precipitation": "mm",
    "pressure_msl": "hPa",
    "wind_speed_10m": "m/s",
    "wind_direction_10m": "°",
    "wind_gusts_10m": "m/s",
}
LOCATION = {"lat": 52.46, "lon": 4.60, "name": "coast at IJmuiden"}

# daily weather columns made by hourly_to_daily
WEATHER = ["wind_from_nw", "wind", "gust_max", "pressure", "pressure_min", "rain"]


def check_weather_units(units):
    """Stop with a clear message if Open-Meteo answers in other units than training used."""
    wrong = {var: (units.get(var), unit) for var, unit in WEATHER_UNITS.items() if units.get(var) != unit}
    if wrong:
        raise ValueError("Weather API answered in different units than the model was trained on: "
                         + ", ".join(f"{v}: got {got!r}, expected {want!r}" for v, (got, want) in wrong.items()))


def hourly_to_daily(hourly):
    """Hourly weather (Open-Meteo columns) -> one row per day."""
    hourly = hourly.dropna().copy()
    hourly["time"] = pd.to_datetime(hourly["time"]) + pd.Timedelta(hours=1)  # GMT -> GMT+1

    # Wind direction = where the wind comes FROM, in degrees (0 = north, 270 = west).
    # Storm surges on the Dutch coast come with north-west wind (315 degrees), so we
    # compute how much of the wind blows from the north-west (negative = from south-east).
    angle = np.radians(hourly["wind_direction_10m"] - 315)
    hourly["wind_from_nw"] = hourly["wind_speed_10m"] * np.cos(angle)

    daily = hourly.groupby(hourly["time"].dt.normalize()).agg(
        rain=("precipitation", "sum"),
        pressure=("pressure_msl", "mean"),
        pressure_min=("pressure_msl", "min"),
        wind=("wind_speed_10m", "mean"),
        gust_max=("wind_gusts_10m", "max"),
        wind_from_nw=("wind_from_nw", "mean"),
    )
    daily.index.name = "date"
    return daily


# ---------- sea level ----------
def daily_sea_level(sea):
    """10-minute sea levels -> one row per day:
    sea_mean, sea_low (low tide), sea_high (high tide), tide_pred_low / tide_pred_high."""
    by_day = sea.groupby(sea["time"].dt.normalize())
    daily = by_day["sea_level"].agg(sea_mean="mean", sea_low="min", sea_high="max", n="count")
    daily["tide_pred_low"] = by_day["tide_pred"].min()
    daily["tide_pred_high"] = by_day["tide_pred"].max()
    daily = daily[daily["n"] >= 100]  # a full day has 144 values; skip days with big gaps
    daily.index.name = "date"
    return daily.drop(columns="n")


def live_daily_sea_level(sea):
    """Same as daily_sea_level, for LIVE data around today:
    - today is not finished: where there is no measurement yet, use Rijkswaterstaat's forecast
    - future days have no measurement at all, but the astronomical tide is known, so keep it."""
    sea = sea.copy()
    if "rws_forecast" in sea:
        sea["sea_level"] = sea["sea_level"].fillna(sea["rws_forecast"])
    daily = daily_sea_level(sea).drop(columns=["tide_pred_low", "tide_pred_high"])
    by_day = sea.groupby(sea["time"].dt.normalize())["tide_pred"]
    tide = by_day.agg(tide_pred_low="min", tide_pred_high="max", n="count")
    tide = tide[tide["n"] >= 100].drop(columns="n")
    return daily.join(tide, how="outer").rename_axis("date")


# ---------- salt level and inflow ----------
def add_salt_level(days):
    """salt level = canal level - sea level at LOW tide (chosen in step 1).
    inflow = 1 when it is below 0: even at low tide the sea is higher than the canals."""
    days = days.copy()
    days["salt_level"] = days["storage_level"] - days["sea_low"]
    days["inflow"] = (days["salt_level"] < 0).astype(int).where(days["salt_level"].notna())
    return days


# ---------- the model inputs ----------
def day_suffix(k):
    """0 -> '_t', 2 -> '_t+2' (the names the 2-day model was trained with)."""
    return "_t" if k == 0 else f"_t+{k}"


SYNODIC_MONTH = 29.530588  # days from new moon to new moon
NEW_MOON = pd.Timestamp("2000-01-06 18:14")  # a known new moon


def context_features(daily, horizon):
    """Extra inputs found in step 10 (seasonality and other effects), for day t -> t+horizon:
    - season: time of year of the target day (inflow happens September to February)
    - spring/neap tide: twice a month the tide range is small (neap) and low tide stays HIGH,
      so inflow is ~2.5x as likely around half moon as around full/new moon
    - surge today: how far the sea already is above the astronomical tide (a storm in progress)
    - storm window: the worst wind and pressure over ALL days t..t+horizon, and 3 days of
      north-west wind before the target day (the North Sea needs time to pile up)
    - recent history: rain and inflow days in the last 7 days"""
    X = pd.DataFrame(index=daily.index)
    target = daily.index + pd.Timedelta(days=horizon)
    year_angle = 2 * np.pi * target.day_of_year / 365.25
    X["season_sin"], X["season_cos"] = np.sin(year_angle), np.cos(year_angle)
    moon_angle = 2 * np.pi * ((target - NEW_MOON).total_seconds() / 86400 % SYNODIC_MONTH) / SYNODIC_MONTH
    X["springneap_sin"], X["springneap_cos"] = np.sin(2 * moon_angle), np.cos(2 * moon_angle)  # 2 cycles/month
    X[f"tide_range_t+{horizon}"] = (daily["tide_pred_high"] - daily["tide_pred_low"]).shift(-horizon)
    X["surge_t"] = daily["sea_low"] - daily["tide_pred_low"]
    days_ahead = range(horizon + 1)
    X["nw_max_window"] = pd.concat([daily["wind_from_nw"].shift(-k) for k in days_ahead], axis=1).max(axis=1)
    X["pressure_min_window"] = pd.concat([daily["pressure_min"].shift(-k) for k in days_ahead],
                                         axis=1).min(axis=1)
    X[f"nw_3day_t+{horizon}"] = sum(daily["wind_from_nw"].shift(-k) for k in [horizon - 2, horizon - 1, horizon]) / 3
    X["rain_7d"] = daily["rain"].rolling(7).sum()
    X["inflow_7d"] = daily["inflow"].rolling(7).sum()
    return X


def build_features(daily, horizon=2, context=False):
    """Daily table (date index; weather, tide_pred_low, salt_level, inflow, sea_mean,
    storage_level columns) -> one row per day t with the model inputs to predict day t+horizon.
    context=True adds the seasonality and other effects of context_features (step 9 models).
    Missing values stay NaN; the caller decides what to do with them."""
    daily = daily.asfreq("D")  # one row per calendar day, so shift(2) really means "2 days"
    X = pd.DataFrame(index=daily.index)

    # Weather forecast for today (t), the day before the target day and the target day (t+horizon).
    # For horizon 2 that is t, t+1, t+2. shift(-k) = take the value from k days LATER.
    weather_days = sorted({0, horizon - 1, horizon})
    for col in WEATHER:
        for k in weather_days:
            X[f"{col}{day_suffix(k)}"] = daily[col].shift(-k)

    # Astronomical tide on the target day (always known in advance)
    X[f"tide_pred_low_t+{horizon}"] = daily["tide_pred_low"].shift(-horizon)

    # History: ONLY today and earlier. shift(1) = yesterday's value.
    X["salt_level_t"] = daily["salt_level"]
    X["salt_level_t-1"] = daily["salt_level"].shift(1)
    X["inflow_t"] = daily["inflow"]
    X["sea_mean_t"] = daily["sea_mean"]
    X["storage_level_t"] = daily["storage_level"]
    if context:
        X = X.join(context_features(daily, horizon))
    return X


def build_target(daily, horizon=2):
    """TARGET: inflow on day t+horizon (on the row of day t)."""
    return daily.asfreq("D")["inflow"].shift(-horizon)


# Inputs that need measured water levels (sea + canal). The weather-only model skips them.
WATER_LEVEL_FEATURES = ["salt_level_t", "salt_level_t-1", "inflow_t", "sea_mean_t", "storage_level_t",
                        "surge_t", "inflow_7d"]


def weather_only(features):
    """The inputs that are known without any water-level measurement:
    the weather forecast and the astronomical tide."""
    return [f for f in features if f not in WATER_LEVEL_FEATURES]


class FeatureMismatch(ValueError):
    """The new data does not look like the data the model was trained on."""


def check_features(X, expected, ranges=None):
    """Check the new inputs BEFORE predicting:
    1. exactly the same columns, in the same order, as in training
    2. no missing values
    3. values in a plausible range (catches unit mistakes, e.g. km/h instead of m/s).
    Raises FeatureMismatch with a clear message; returns a list of soft warnings."""
    got = list(X.columns)
    if got != list(expected):
        missing = [c for c in expected if c not in got]
        extra = [c for c in got if c not in expected]
        problem = []
        if missing:
            problem.append(f"missing: {missing}")
        if extra:
            problem.append(f"not in training: {extra}")
        if not missing and not extra:
            problem.append(f"same columns but in a different order: {got}")
        raise FeatureMismatch("The inputs don't match the trained model (" + "; ".join(problem) + ")")

    empty = [c for c in got if X[c].isna().any()]
    if empty:
        raise FeatureMismatch(f"These inputs have no value: {', '.join(empty)}")

    warnings = []
    for col, (low, high) in (ranges or {}).items():
        if col not in X:
            continue
        value, span = float(X[col].iloc[0]), high - low
        if value < low - span or value > high + span:  # far outside anything seen in 25 years
            raise FeatureMismatch(f"{col} = {value:.2f} is far outside the training range "
                                  f"[{low:.2f}, {high:.2f}]. Wrong units?")
        if value < low or value > high:
            warnings.append(f"{col} = {value:.2f} is outside the training range [{low:.2f}, {high:.2f}]")
    return warnings
