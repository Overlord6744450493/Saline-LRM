# Seawater inflow forecast: training specs and test-set evaluation

**Question the model answers:** on day *t*, will seawater flow into the Amsterdam canal system on day *t+2*?
**Main model:** logistic regression (scikit-learn), 24 inputs. Used by `predict.py` and `app.py`.
**Honest test:** 2020-01-01 to 2025-05-05 (1,952 days). No model saw these years during training or tuning.

**Headline result (test set, pump threshold 0.5)**
| | Main model | Simple baseline ("inflow today") |
|---|---|---|
| Inflow events caught 2 days ahead | **80 of 95 (84%)** | 20 of 95 (21%) |
| Average precision (1 = perfect, 0.07 = random) | **0.60** | 0.09 |
| ROC AUC | **0.96** | 0.57 |
| Alarms that were right (day level) | 38% | 20% |
| Unnecessary pump days per year | 34 | 20 |

---

## 1. Target definition

- **Salt level** = canal level (Amstelsluizen, m NAP) − sea level at **low tide** (IJmuiden, m NAP).
- **Inflow day** = salt level < 0. Even at low tide the sea stays above the canals, so the water can't drain and seawater pushes in.
- **Target** = inflow on day *t+2*, on the row of day *t* (`features.build_target`).
- **Event** = a run of consecutive inflow days. An event is "caught" if at least one of its days had an alarm 2 days before.

## 2. Data

| Source | What | Resolution | Access |
|---|---|---|---|
| AGV water board (hackathon files) | Canal level Amstelsluizen | daily | downloaded file, ends 2025-05-07 |
| Rijkswaterstaat | Sea level IJmuiden: measured + astronomical tide | 10 min → daily | free API, no key |
| Open-Meteo archive (ERA5) | Precipitation, pressure, wind speed/direction, gusts at 52.46 N, 4.60 E | hourly → daily | free API, no key |

- 9,259 days with all sources, 2000–2025; 9,256 usable after building features.
- Time zone: GMT+1 without summer time (same as the AGV data). Weather is requested in GMT and shifted +1 h.
- Units are fixed and checked at prediction time: mm, hPa, m/s, degrees; water levels in m NAP.

## 3. Features (24)

All inputs are built by **one function**, `features.build_features`. Training, `predict.py` (live and demo) and `app.py` all use it. A check showed demo-mode inputs are identical to the training inputs (max difference 0) on 7 dates from 2003 to 2025.

| Group | Features | Count | Available live? |
|---|---|---|---|
| Weather on *t*, *t+1*, *t+2* | `wind_from_nw`, `wind`, `gust_max`, `pressure`, `pressure_min`, `rain` (daily mean/max/min/sum) | 18 | yes, from the Open-Meteo forecast |
| Astronomical tide | `tide_pred_low_t+2` | 1 | yes, from Rijkswaterstaat (cached ~4 months ahead) |
| Water levels on *t* and *t−1* | `salt_level_t`, `salt_level_t-1`, `inflow_t`, `sea_mean_t`, `storage_level_t` | 5 | sea level yes; canal level **no** (typical monthly value used) |

`wind_from_nw` = wind speed × cos(direction − 315°): the part of the wind blowing from the north-west, which pushes North Sea water onto the coast.

## 4. Split

| Part | Days *t* | Days | Inflow days at *t+2* | Inflow rate | Used for |
|---|---|---|---|---|---|
| Train | 2000-01-02 to 2016-12-31 | 6,209 | 261 | 4.2% | fitting the models |
| Validation | 2017-01-01 to 2019-12-31 | 1,095 | 57 | 5.2% | choosing the model and the pump threshold |
| Test | 2020-01-01 to 2025-05-05 | 1,952 | 132 | 6.8% | final evaluation only |

The split is by time, not random, so the model never learns from days that come after the days it's tested on. Inflow is more frequent in the test years than in training.

## 5. Training specs

### 5.1 Logistic regression (main model, `step5_model.py`)
| Setting | Value |
|---|---|
| Pipeline | `StandardScaler` → `LogisticRegression` |
| Regularisation | L2, C = 1.0 |
| Solver | lbfgs, max_iter 2000, tol 1e-4 |
| Class imbalance | `class_weight="balanced"` (an inflow day weighs ~23× a normal day) |
| Selection | best validation average precision (0.62) vs random forest (0.58) |

### 5.2 Fallback models (saved in the same `output/model.joblib`)
| Model | Inputs | Used when | Threshold |
|---|---|---|---|
| Main model + typical canal level | 24; canal level = training median for the month | live default (real canal level is not online) | 0.4 |
| Weather + tide only | 19 (no water levels) | Rijkswaterstaat is down, or `--weather-only` | 0.5 |

### 5.3 Random forest (comparison only, not used)
300 trees, `min_samples_leaf=5`, `class_weight="balanced_subsample"`, `random_state=0`.

### 5.4 Neural network (comparison only, `step8_pytorch.py`)
| Setting | Value |
|---|---|
| Architecture | 24 → 32 (ReLU, dropout 0.2) → 16 (ReLU, dropout 0.2) → 1 |
| Input scaling | mean/std from the training years |
| Loss | `BCEWithLogitsLoss`, `pos_weight` = negatives/positives (~22.8) |
| Optimiser | Adam, learning rate 1e-3, weight decay 1e-4, batch 128 |
| Stopping | early stopping on validation average precision, patience 20, max 300 epochs; best epoch kept (~39) |
| Ensemble | 5 networks (seeds 0–4), probabilities averaged; single-run test AP 0.58–0.59 |

### 5.5 Pump threshold
`helpers.choose_threshold`: the **highest** threshold in {0.2 … 0.8} that still catches at least 90% of events on the **validation** years. The reasoning is that missing seawater is worse than an extra pump run. It is chosen separately for each model and saved with it.

### 5.6 Saved with the model (`output/model.joblib`)
Feature names and order, threshold(s), weather units, location, typical canal level per month, min/max of every feature on the training set, and the test scores. `predict.py` refuses to predict if the columns, order or units differ, or if a value is far outside the training range.

### 5.7 Software
Python 3.14.5, scikit-learn 1.9.1, pandas 3.1.0, PyTorch 2.14.1 (CPU).

## 6. Test-set evaluation (2020-01-01 to 2025-05-05)

### 6.1 All models
Threshold chosen on validation (see 5.5). Day-level counts: TP = alarm and inflow, FP = alarm and no inflow, FN = missed inflow day.

| Model | Thr. | Avg. precision | ROC AUC | Precision | Recall | F1 | TP | FP | FN | Events caught | Unnecessary pump days/yr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Logistic regression (main)** | 0.5 | **0.60** | **0.96** | 0.38 | 0.85 | **0.53** | 112 | 179 | 20 | 80/95 (84%) | 33.5 |
| LR + typical canal level (live) | 0.4 | 0.60 | 0.95 | 0.34 | 0.87 | 0.49 | 115 | 221 | 17 | 83/95 (87%) | 41.4 |
| Weather + tide only (fallback) | 0.5 | 0.58 | 0.95 | 0.39 | 0.80 | 0.52 | 106 | 168 | 26 | 77/95 (81%) | 31.5 |
| Neural network (PyTorch, 5 runs) | 0.5 | 0.59 | 0.95 | 0.37 | 0.83 | 0.51 | 110 | 189 | 22 | 80/95 (84%) | 35.4 |
| Random forest | 0.5 | 0.46 | 0.93 | 0.56 | 0.30 | 0.39 | 39 | 31 | 93 | 32/95 (34%) | 5.8 |
| Baseline: inflow today | – | 0.09 | 0.57 | 0.20 | 0.20 | 0.20 | 26 | 106 | 106 | 20/95 (21%) | 19.8 |

Random guessing has an average precision of 0.07 (the inflow rate).

**Reading the table**
- The logistic regression, the neural network and the weather-only model are practically tied. The neural network doesn't add anything on this data, so the simpler model is kept.
- The random forest **overfits**: average precision 0.98 on training vs 0.46 on test. The linear models score about the same on train, validation and test (0.64 / 0.62 / 0.60).
- Water-level history adds little on top of the weather: 0.60 vs 0.58. The weather on day *t+2* and the tide carry most of the signal.

### 6.2 Threshold trade-off (main model)
| Threshold | Events caught | Pump days/yr | Unnecessary/yr | Alarms right |
|---|---|---|---|---|
| 0.2 | 97% | 89 | 65 | 27% |
| 0.3 | 94% | 74 | 51 | 31% |
| 0.4 | 92% | 65 | 42 | 35% |
| **0.5** | **84%** | **55** | **34** | **38%** |
| 0.6 | 82% | 46 | 26 | 44% |
| 0.7 | 73% | 39 | 21 | 46% |
| 0.8 | 66% | 31 | 14 | 53% |
| 0.9 | 55% | 21 | 8 | 61% |

The threshold is a policy choice for the stakeholder, not a fixed property of the model.

### 6.3 Per year (main model, threshold 0.5, by target day)
| Year | Inflow days | Alarms | Inflow days caught | False alarms | Avg. precision |
|---|---|---|---|---|---|
| 2020 | 21 | 53 | 18 | 35 | 0.55 |
| 2021 | 21 | 54 | 18 | 36 | 0.42 |
| 2022 | 17 | 44 | 14 | 30 | 0.48 |
| 2023 | 35 | 84 | 32 | 52 | 0.77 |
| 2024 | 34 | 50 | 27 | 23 | 0.75 |
| 2025 (to May) | 4 | 6 | 3 | 3 | 0.50 |

Performance is stable across years and best in the stormy years 2023–2024.

### 6.4 Per season
| Season | Days | Inflow days | Alarms | False alarms | Missed inflow days |
|---|---|---|---|---|---|
| Sep–Feb | 964 | 116 | 244 | 144 | 16 |
| Mar–Aug | 988 | 16 | 47 | 35 | 4 |

88% of inflow days and 80% of false alarms fall in September–February.

### 6.5 Which events are missed?
95 events in the test set: 71 last 1 day, 15 last 2 days, 9 last 3 days or more.

| How deep the event went (lowest salt level) | Events | Caught |
|---|---|---|
| below −0.30 m (severe) | 14 | **100%** |
| −0.30 to −0.10 m | 38 | **95%** |
| −0.10 to 0 m (marginal) | 43 | 70% |

**All 15 missed events lasted a single day,** and 13 of them were within 10 cm of zero. Typically the north-west wind was weak (below 3 m/s), so the sea only just topped the canal level. **No severe event was missed.**

### 6.6 Calibration: what does "78%" mean?
| Predicted probability | Days | Average predicted | Actually inflow |
|---|---|---|---|
| 0–10% | 1,347 | 2% | 0% |
| 10–30% | 211 | 18% | 4% |
| 30–50% | 103 | 40% | 11% |
| 50–70% | 83 | 60% | 19% |
| 70–90% | 95 | 81% | 28% |
| 90–100% | 113 | 96% | 61% |

The ranking is reliable: a higher number really does mean a higher risk, and below 10% inflow almost never happens. But the numbers are **too high** as probabilities. This is a side effect of `class_weight="balanced"`, which makes the model treat inflow as common. Read the output as a **risk score**, not a literal chance. If a true probability is needed, the model can be recalibrated on the validation years (e.g. Platt scaling or isotonic regression), and the threshold re-chosen.

### 6.7 What drives the prediction
Logistic regression coefficients on standardised inputs (positive = more inflow risk):

| Input | Coefficient |
|---|---|
| `wind_from_nw_t+2` | +1.81 |
| `tide_pred_low_t+2` | +1.64 |
| `pressure_t+2` | −1.49 |
| `wind_t+2` | +1.41 |
| `pressure_t+1` | +0.79 |
| `wind_from_nw_t+1` | −0.52 |
| `pressure_min_t+1` / `pressure_min_t+2` | −0.51 / +0.51 |
| `storage_level_t` | −0.45 |

This matches the physics: strong north-west wind, a high astronomical low tide and low air pressure on the target day. The opposite-sign pairs (e.g. pressure on *t+1* vs *t+2*) mostly say how fast the pressure is dropping. The random forest ranks the same group on top: wind, gusts and north-west wind on *t+2*, then the tide.

## 7. Limitations of this evaluation

1. **Perfect weather.** Training and testing use the weather that actually happened (ERA5 reanalysis), not archived forecasts. Live, the model gets a 1–2-day forecast with errors, so expect lower scores. A fair check would use the Open-Meteo Historical Forecast API (archived forecasts from 2022).
2. **Gusts are measured differently.** The reanalysis and the forecast models differ in how they compute gusts, which can shift `gust_max`.
3. **Canal level is not live.** The live default uses the typical monthly level. On the test set this costs nothing in average precision (0.60 vs 0.60) but changes the threshold to 0.4.
4. **The target is our interpretation** of the stakeholder's "salt level". It isn't a measured salt concentration.
5. **One system-wide signal.** There is no per-area or per-measurement-point evaluation.
6. **Daily resolution.** The model can't tell which tide (or hour) the inflow happens on.
7. **Drift.** The inflow rate went from 4.2% (train) to 6.8% (test), and 2023–24 had about 35 inflow days a year. Retrain yearly.
8. **Overstated probabilities** (see 6.6).

## 8. Reproduce

```text
python step2_target.py      # target  -> output/daily_target.csv
python step3_weather.py     # weather -> output/daily_dataset.csv
python step5_model.py       # train + evaluate + save output/model.joblib
python step6_pump_rule.py   # threshold charts
python step8_pytorch.py     # neural network comparison -> output/nn_model.pt
python predict.py --date 2020-02-09   # demo on Storm Ciara (test period)
```
