"""
Helper functions that several step files use. You don't need to run this file;
the step files do "from helpers import ..." to use it.
The model inputs themselves are made in features.py.
"""
import glob
import os

import pandas as pd

try:  # use the Windows certificate store, otherwise the Rijkswaterstaat download fails
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    print("Tip: run  python -m pip install truststore  if downloads give an SSL error")

import requests

# The feature code lives in features.py; imported here so "from helpers import ..." keeps working.
from features import (HOURLY_VARS, LOCATION, WEATHER, add_salt_level, build_features,  # noqa: F401
                      check_weather_units, daily_sea_level, hourly_to_daily, live_daily_sea_level)

EXTERNAL = "data/external"  # where we save data we download ourselves
os.makedirs(EXTERNAL, exist_ok=True)


class DataSourceError(RuntimeError):
    """An online data source is down, slow or gave a strange answer."""


def get_json(method, url, source, timeout, tries=2, **kwargs):
    """requests.get/post that turns every network problem into ONE clear DataSourceError."""
    last = None
    for _ in range(tries):
        try:
            answer = requests.request(method, url, timeout=timeout, **kwargs)
            answer.raise_for_status()
            return answer.json()
        except requests.Timeout:
            last = f"no answer within {timeout} seconds"
        except requests.ConnectionError:
            last = "could not connect (no internet, or the service is down)"
        except requests.HTTPError as error:
            last = f"HTTP error {error.response.status_code}: {error.response.text[:200]}"
        except ValueError:
            last = "the answer was not valid JSON"
    raise DataSourceError(f"{source}: {last}")


def load_agv_level(name):
    """Load an AGV water-level file that has TWO header rows.
    Row 1 = location code, row 2 = parameter name. We skip row 2."""
    df = pd.read_csv(f"data/agv/waterkwantiteit/{name}.csv", sep=";", skiprows=[1])
    df["date"] = pd.to_datetime(df["GMT+1"], dayfirst=True).dt.normalize()
    return df.drop(columns="GMT+1").set_index("date")


def load_storage_level():
    """Daily mean water level of the Amsterdam canals (Amstelsluizen), in m NAP."""
    return load_agv_level("Amstelsluizen_waterhoogte")["MPP00013_LIT054_1"].rename("storage_level")


RWS_URL = ("https://ddapi20-waterwebservices.rijkswaterstaat.nl/"
           "ONLINEWAARNEMINGENSERVICES/OphalenWaarnemingen")


def fetch_sea_level(begin, end, location="ijmuiden.buitenhaven", timeout=180):
    """Ask Rijkswaterstaat for all water-level series between two dates.
    Returns raw rows: time, kind, cm_nap. 'kind' (ProcesType) is
      'meting' = measured, 'astronomisch' = predicted tide,
      'verwachting' = Rijkswaterstaat's own forecast (only for the coming days)."""
    body = {
        "Locatie": {"Code": location},
        "AquoPlusWaarnemingMetadata": {"AquoMetadata": {
            "Compartiment": {"Code": "OW"}, "Grootheid": {"Code": "WATHTE"}}},
        "Periode": {"Begindatumtijd": f"{begin:%Y-%m-%d}T00:00:00.000+01:00",
                    "Einddatumtijd": f"{end:%Y-%m-%d}T00:00:00.000+01:00"},
    }
    answer = get_json("POST", RWS_URL, "Rijkswaterstaat sea level", timeout, json=body)
    rows = []
    for series in answer.get("WaarnemingenLijst", []):
        unit = series["AquoMetadata"].get("Eenheid", {}).get("Code", "cm")
        if unit != "cm":
            raise DataSourceError(f"Rijkswaterstaat sea level: unit is {unit!r}, expected 'cm'")
        kind = series["AquoMetadata"].get("ProcesType")
        for m in series["MetingenLijst"]:
            rows.append((m["Tijdstip"], kind, m["Meetwaarde"].get("Waarde_Numeriek")))
    return pd.DataFrame(rows, columns=["time", "kind", "cm_nap"])


def tidy_sea_level(raw):
    """Raw rows -> one row per 10 minutes with columns
    sea_level (measured), tide_pred (astronomical) and, if present, rws_forecast."""
    sea = raw.copy()
    # Same time zone as the AGV data (GMT+1, no summer time)
    sea["time"] = pd.to_datetime(sea["time"], utc=True).dt.tz_convert("Etc/GMT-1").dt.tz_localize(None)
    sea = sea[sea["cm_nap"].abs() < 1000]  # 999999999 = missing value code
    sea["m_nap"] = sea["cm_nap"] / 100  # cm -> m
    sea = sea.pivot_table(index="time", columns="kind", values="m_nap", aggfunc="mean")
    sea = sea.rename(columns={"meting": "sea_level", "astronomisch": "tide_pred",
                              "verwachting": "rws_forecast"})
    for col in ["sea_level", "tide_pred"]:  # make sure these exist, even if RWS sent nothing
        if col not in sea:
            sea[col] = float("nan")
    return sea.reset_index()


def download_sea_level(start, end, location="ijmuiden.buitenhaven"):
    """Download sea level (10-minute values, in metres NAP) from Rijkswaterstaat,
    one month at a time. Free, no API key. Two columns come back:
      sea_level = what was really MEASURED
      tide_pred = the ASTRONOMICAL tide (moon + sun only), known years in advance
    Already-downloaded months are saved in data/external and not downloaded again."""
    parts = []
    for month in pd.date_range(start, end, freq="MS"):
        cache = f"{EXTERNAL}/sea_{location}_{month:%Y-%m}.csv"
        if not os.path.exists(cache):
            raw = fetch_sea_level(month, month + pd.offsets.MonthBegin(1), location)
            raw.to_csv(cache, index=False)
            print(f"downloaded {month:%Y-%m}: {len(raw)} values")
        parts.append(pd.read_csv(cache))
    return tidy_sea_level(pd.concat(parts, ignore_index=True))


WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_open_meteo(url, source, timeout, **params):
    """One Open-Meteo request -> hourly table. Same variables and units for the
    historical (archive) and the forecast API, checked against features.WEATHER_UNITS."""
    params = {"latitude": LOCATION["lat"], "longitude": LOCATION["lon"], "hourly": HOURLY_VARS,
              "wind_speed_unit": "ms", "timezone": "GMT", **params}
    answer = get_json("GET", url, source, timeout, params=params)
    if answer.get("error"):
        raise DataSourceError(f"{source}: {answer.get('reason')}")
    check_weather_units(answer.get("hourly_units", {}))
    return pd.DataFrame(answer["hourly"])


def download_weather(start_year, end_year):
    """Download hourly historical weather from Open-Meteo (free, no API key)
    for the coast at IJmuiden, 5 years per request, saved in data/external.
    Returns one row per day."""
    parts = []
    for year in range(start_year, end_year + 1, 5):
        last = min(year + 4, end_year)
        cache = f"{EXTERNAL}/weather_{LOCATION['lat']}_{LOCATION['lon']}_{year}_{last}.csv"
        if not os.path.exists(cache):
            hourly = fetch_open_meteo(WEATHER_URL, "Open-Meteo historical weather", 300,
                                      start_date=f"{year}-01-01", end_date=f"{last}-12-31")
            hourly.to_csv(cache, index=False)
            print(f"downloaded weather {year}-{last}")
        parts.append(pd.read_csv(cache))
    return hourly_to_daily(pd.concat(parts, ignore_index=True))


def historical_weather(start, end):
    """Daily HISTORICAL weather for a few days (demo mode). Uses the files we trained on
    (data/external) when they cover the period, otherwise asks the Open-Meteo archive."""
    first, last = start - pd.Timedelta(days=1), end + pd.Timedelta(days=1)  # margin for GMT -> GMT+1
    files = glob.glob(f"{EXTERNAL}/weather_{LOCATION['lat']}_{LOCATION['lon']}_*.csv")
    if files:
        hourly = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates("time")
        times = pd.to_datetime(hourly["time"])
        hourly = hourly[(times >= first) & (times < last + pd.Timedelta(days=1))]
        daily = hourly_to_daily(hourly)
        if all(day in daily.index for day in pd.date_range(start, end)):
            return daily.loc[start:end]
    hourly = fetch_open_meteo(WEATHER_URL, "Open-Meteo historical weather", 60,
                              start_date=f"{first:%Y-%m-%d}", end_date=f"{last:%Y-%m-%d}")
    return hourly_to_daily(hourly).loc[start:end]


def forecast_weather(past_days=2, forecast_days=4):
    """LIVE weather forecast from Open-Meteo (free, no key): the last few days
    plus the coming days, one row per day (same columns as download_weather)."""
    hourly = fetch_open_meteo(FORECAST_URL, "Open-Meteo weather forecast", 30,
                              past_days=past_days, forecast_days=forecast_days)
    return hourly_to_daily(hourly)


def evaluate_rule(part, threshold):
    """Check the pump rule "pump when prob > threshold" on past predictions.
    part needs columns 'prob' and 'inflow_t+2' (index = the day the prediction is about).
    An inflow EVENT = consecutive inflow days; it is caught if at least one day had an alarm."""
    alarm = part["prob"] > threshold
    inflow = part["inflow_t+2"] == 1
    event_id = (inflow & ~inflow.shift(1, fill_value=False)).cumsum().where(inflow)
    caught = alarm.groupby(event_id).any()
    years = (part.index.max() - part.index.min()).days / 365.25
    return {
        "threshold": threshold,
        "events": len(caught),
        "events_caught": int(caught.sum()),
        "share_caught": caught.mean(),
        "pump_days": int(alarm.sum()),
        "unnecessary_pump_days": int((alarm & ~inflow).sum()),
        "unnecessary_per_year": (alarm & ~inflow).sum() / years,
        "alarm_right": (alarm & inflow).sum() / max(alarm.sum(), 1),
    }


def choose_threshold(part, min_caught=0.9, options=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """The HIGHEST threshold that still catches at least 90% of events
    (missing seawater is worse than an extra pump run). Used in step 5 and step 6."""
    table = pd.DataFrame([evaluate_rule(part, t) for t in options])
    good_enough = table[table["share_caught"] >= min_caught]
    return float(good_enough["threshold"].max()) if len(good_enough) else 0.5


def today_gmt1():
    """Today's date in the data's time zone (GMT+1, no summer time)."""
    return pd.Timestamp.now(tz="Etc/GMT-1").tz_localize(None).normalize()


def load_live_data(today):
    """Live sea level (Rijkswaterstaat) and weather forecast (Open-Meteo) around today."""
    sea = tidy_sea_level(fetch_sea_level(today - pd.Timedelta(days=2), today + pd.Timedelta(days=3),
                                         timeout=60))
    weather = forecast_weather(past_days=2, forecast_days=4)
    return sea, weather


def daily_inputs(sea_daily, weather, storage_level):
    """Put sea level, weather and canal level next to each other, one row per day.
    storage_level can be one number (live: the canal level isn't online) or a daily Series."""
    days = weather.join(sea_daily, how="outer")
    days["storage_level"] = storage_level
    return add_salt_level(days)


def build_live_features(today, sea, weather, storage_level):
    """Make the model inputs for 'today' (day t) from live data, with features.build_features."""
    days = daily_inputs(live_daily_sea_level(sea), weather, storage_level)
    return build_features(days).loc[[today]].reset_index(drop=True)


def risk_label(prob, threshold):
    """HIGH = above the pump threshold, MEDIUM = above half of it, else LOW."""
    if prob > threshold:
        return "HIGH"
    return "MEDIUM" if prob > threshold / 2 else "LOW"


def pump_plan(prob, threshold, stations, n=3):
    """Turn a probability into a pump plan. 'stations' = output/station_ranking.csv.
    The data showed NO station is measurably more effective (step 7), so we pick
    the stations that can move the most water (highest capacity)."""
    risk = risk_label(prob, threshold)
    if risk != "HIGH":
        return risk, []
    top = stations.sort_values("capacity_m3_day", ascending=False).head(n)
    return risk, [f"{code} (pumps {row['pumps']}, up to {row['capacity_m3_day']:,.0f} m3/day)"
                  for code, row in top.iterrows()]
