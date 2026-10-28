"""
STEP 3 - Get weather data and put it next to the inflow target.

Source: Open-Meteo historical weather (free, no API key), for the coast at
IJmuiden (52.46 N, 4.60 E), where the sea meets the Amsterdam canal system.
Hourly values are turned into daily values:
    rain          total rain that day (mm)
    pressure      average air pressure (hPa) - low pressure = storm
    pressure_min  lowest air pressure that day
    wind          average wind speed (m/s)
    gust_max      strongest gust (m/s)
    wind_from_nw  how hard the wind blows FROM the north-west (m/s, negative = from the south-east)
                  North-west wind pushes North Sea water onto the Dutch coast.
Needs output/daily_target.csv from step 2.
"""
# %% Cell 1: imports
import pandas as pd

from helpers import download_weather


# %% Cell 2: download (first time ~1 minute, then instant)
weather = download_weather(2000, 2025)
print(weather.shape)
weather.describe().round(1)


# %% Cell 3: join weather with the target (same dates)
target = pd.read_csv("output/daily_target.csv", index_col="date", parse_dates=True)
data = target.join(weather, how="inner")
print(f"{len(data):,} days with target AND weather")
data.to_csv("output/daily_dataset.csv")
data[["salt_level", "inflow", "rain", "pressure", "wind", "wind_from_nw"]].head()
