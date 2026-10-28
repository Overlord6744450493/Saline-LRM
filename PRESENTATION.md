# Keeping seawater out: a 2-day warning for the pumps

## 1. The problem
Seawater sometimes pushes into Amsterdam's water storage. The stakeholder has pumps
that can push it back out, but needs to know **in advance** when to use them.
Their signal: *"when the salt level is negative, ocean water flows in"*. Nobody
knew exactly which measurement that was.

## 2. What "salt level" turned out to be
- Salt **concentration** (chloride, conductivity) is never negative in 160,000+
  measurements, so it can't be the signal.
- Canal levels are *always* negative in m NAP (kept ~0.4 m below sea level), so
  "negative" alone means nothing there.
- What fits: **salt level = canal level − sea level at low tide** (IJmuiden).
  When it is negative, even at low tide the sea is higher than the canals: water
  can't drain out and seawater pushes in.
- This happened on **450 of 9,259 days (4.9%)**, 2000–2025: about 18 days a year,
  almost all from September to February. 2023 and 2024 had twice as many (~35).

**Chart:** `output/step1_salt_level_example.png`

## 3. What drives it: the weather
Strong **north-west wind** pushes North Sea water onto the coast. With a daily
north-west wind above 8 m/s, inflow happens on **70%** of days. With wind from the
south-east it happens on **1%**.

**Chart:** `output/step4_inflow_vs_wind.png`

## 4. The model
- **Question:** on day *t*, will seawater flow in on day *t+2*?
- **Inputs (24):** the weather forecast for the next 2 days (wind, gusts,
  north-west wind, pressure, rain), the astronomical tide, and today's water levels.
- **Model:** logistic regression. It beat a random forest on the 2017–2019
  validation years.
- **Honest test:** trained on 2000–2016, tested on 2020–2025, data it never saw.

## 5. Results and pump recommendation
**Rule:** *if the predicted inflow probability for the day after tomorrow is above
0.5, prepare and release the pumps.*

| Test years 2020–2025 | Our rule | Baseline ("inflow today") |
|---|---|---|
| Inflow events caught 2 days ahead | **84%** (80 of 95) | 21% |
| Alarms that were right | 38% | 20% |
| Unnecessary pump days per year | ~34 | ~20 |

The threshold is a dial. At 0.7 we would pump unnecessarily about 19 days a year,
but catch only ~81% of events.

**Chart:** `output/step6_rule_vs_baseline.png`

### Which pumps?
The data has 83 pumps at 44 pumping stations with hourly flows (1996–2025), but no
names or locations. So pumps can't be linked to areas, and inflow is one signal for
the whole system. We checked whether the salt level recovered faster than the storm
alone predicts when a station ran hard. **No station stands out.** Only 2 of 44 pass
the test, about what luck alone gives, and total pumping shows no link (r = −0.01).
These are polder pumps, which lift polder water into the canals. The pumps that push
water out to sea (IJmuiden) are not in the data. The plan therefore names the
**largest-capacity stations** (KGM00110, KGM00357, KGM00102) and says plainly that
their effectiveness is unproven.

**Chart (optional):** `output/step7_station_ranking.png`

## 6. Limitations (be upfront)
1. **"Salt level" is our interpretation.** It fits the stakeholder's description
   and the physics, but they should confirm it.
2. **Real weather, not real forecasts.** We trained and tested with the weather that
   actually happened. A real 2-day forecast has errors, so real performance will be
   somewhat lower. Without any forecast the model is much worse (average precision
   0.17 instead of 0.60), so the forecast is the key ingredient.
3. **No salt sensor in the sewers.** The only one (rwzi Uithoorn influent) has no data.
4. **Daily data only.** We predict the day, not the hour. Tides change every ~6 hours.
5. **Pump effect not measured.** The data can't show how much pumping lowers the salt.
6. **Inflow is increasing** (2023–24). The model should be retrained every year.

## 7. Next steps
- Confirm the "salt level" definition with the stakeholder.
- Test with archived real forecasts (Open-Meteo Historical Forecast API, from 2022).
- Add a conductivity sensor at the storage inlet to measure the pump effect.
