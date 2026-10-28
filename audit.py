"""
AUDIT - the "Model audit" page of the web app (app.py): how good the 7-day models are,
what goes in and comes out, how they were trained, and what they can't do.
Everything shown comes from output/models_horizons.joblib (step 9).
"""
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

import style
from helpers import evaluate_rule

INPUTS = pd.DataFrame([
    ("north-west wind", "wind_from_nw_t, _t+h-1, _t+h", "How much of the wind blows from the north-west "
     "(315°): the direction that pushes the North Sea onto the Dutch coast", "m/s", "Open-Meteo forecast"),
    ("wind speed, gusts", "wind_*, gust_max_*", "Average wind and strongest gust of the day", "m/s",
     "Open-Meteo forecast"),
    ("air pressure", "pressure_*, pressure_min_*", "Mean and lowest air pressure. Low pressure lifts the sea "
     "(about 1 cm per hPa) and comes with storms", "hPa", "Open-Meteo forecast"),
    ("rain", "rain_*", "Rain on the day", "mm", "Open-Meteo forecast"),
    ("storm window", "nw_max_window, pressure_min_window, nw_3day_t+h", "Worst wind and pressure over ALL "
     "days until the target day, and 3 days of north-west wind before it", "m/s, hPa", "Open-Meteo forecast"),
    ("astronomical tide", "tide_pred_low_t+h, tide_range_t+h", "Predicted low tide and tide range on the "
     "target day (moon and sun only, known years ahead)", "m NAP", "Rijkswaterstaat"),
    ("spring/neap cycle", "springneap_sin, springneap_cos", "Where the target day is in the twice-monthly "
     "spring/neap tide cycle", "-", "computed from the date"),
    ("time of year", "season_sin, season_cos", "Day of the year of the target day", "-", "computed from the date"),
    ("water levels now", "salt_level_t, salt_level_t-1, sea_mean_t, storage_level_t, surge_t",
     "Today's and yesterday's salt level, mean sea level, canal level and storm surge", "m (NAP)",
     "Rijkswaterstaat (sea), canal level from the user"),
    ("recent history", "inflow_t, inflow_7d, rain_7d", "Inflow today, inflow days and rain in the last 7 days",
     "days, mm", "measured"),
], columns=["group", "inputs", "what it is", "unit", "source (live)"])


SPECIAL_LABELS = {"season_sin": "time of year (sin)", "season_cos": "time of year (cos)",
                  "springneap_sin": "spring/neap cycle (sin)", "springneap_cos": "spring/neap cycle (cos)"}


def input_label(feature, describe_input):
    """'wind_from_nw_t+2' -> 'north-west wind (t+2)', using the plain names of app.py."""
    if feature in SPECIAL_LABELS:
        return SPECIAL_LABELS[feature]
    name = re.sub(r", \w{3} \d{2} \w{3}$", "", describe_input(feature, pd.Timestamp("2000-01-01"))[0])
    day = re.search(r"_(t([+-]\d+)?)$", feature)
    return f"{name} ({day.group(1)})" if day else name


def chart_height(n):
    return max(160, 28 * n)


def render(bundle, history, describe_input):
    horizons = sorted(bundle["horizons"])
    entries = bundle["horizons"]
    st.header("Model audit")
    st.markdown("Everything about the 7-day forecast models: **how good they are**, **what goes in and "
                "comes out**, **how they were trained** and **what they can't do**. All numbers are on the "
                "**test years 2020-2025**, which the models never saw while they were built.")

    background, performance, io, method, limits = st.tabs(
        ["Background", "Performance", "Inputs & output", "Training method", "Limitations"])

    # ================= BACKGROUND =================
    with background:
        st.markdown(f"""
**The problem.** Seawater sometimes pushes into Amsterdam's water storage. Pumps can push it back out, but
they need to be prepared in advance: hence a forecast up to 7 days ahead.

**What we predict: the salt level.** *Salt level = canal level - sea level at low tide* (at IJmuiden).
When it is **negative**, even at low tide the sea is higher than the canals: water can't drain out and seawater
pushes in. We call such a day an **inflow day**. This happened on
{history['inflow'].mean():.1%} of days in {history.index.min():%Y}-{history.index.max():%Y}
(about {history['inflow'].mean() * 365:.0f} days a year), almost always from September to February.

**Data**
- Sea level at IJmuiden, measured and astronomical tide: Rijkswaterstaat (free, 10-minute values).
- Canal level at the Amstelsluizen, pump flows, chloride samples: water board AGV.
- Weather on the coast at IJmuiden, history and forecast: Open-Meteo (free).
""")
        for image, caption in [("output/step1_salt_level_example.png",
                                "An inflow day: even the low tide stays above the canal level"),
                               ("output/step4_inflow_vs_wind.png", "North-west wind drives inflow")]:
            try:
                st.image(image, caption=caption)
            except Exception:
                st.info(f"Run the step scripts to create {image}")

    # ================= PERFORMANCE =================
    with performance:
        st.markdown("""
**Metric: AUC** (area under the ROC curve). Take one random day *with* inflow and one random day
*without*: AUC is the chance that the model gives the inflow day the higher score. **1.0 = perfect,
0.5 = a coin flip.** It doesn't depend on the alarm level, so it measures the model itself.
""")
        style.note("""
<b>Baselines.</b> The models are compared with three reference forecasts that need no model:
<ul style="margin:6px 0 2px">
<li><b>Season only, the main baseline:</b> the forecast for a day is how often inflow happened in that
<i>calendar month</i> in 2000-2016 (e.g. December 11%, June 0.3%). This is what you know without any weather
or water-level data. It is a strong baseline, because inflow is so seasonal.</li>
<li><b>Persistence:</b> "inflow on the target day if there is inflow today".</li>
<li><b>Random guess:</b> AUC 0.5 by definition.</li>
</ul>
Our model has to beat the season-only baseline to be worth anything: everything above it comes from the
weather forecast, the tide and today's water levels.
""")
        rows = []
        for h in horizons:
            e, pred = entries[h], entries[h]["predictions"]
            auc = {period: roc_auc_score(part["inflow"], part["prob"]) for period, part in pred.groupby("period")}
            test = pred[pred["period"] == "test"]
            rows.append({"day ahead": f"t+{h}", "model": e["name"], "inputs": len(e["features"]),
                         "AUC test": auc["test"], "AUC validation": auc["validation"], "AUC train": auc["train"],
                         "season only AUC": roc_auc_score(test["inflow"], test["season_only"]),
                         "persistence AUC": roc_auc_score(test["inflow"], test["persistence"]),
                         "alarm level": e["threshold"], "events caught": e["test_scores"]["events_caught"],
                         "alarms right": e["test_scores"]["alarm_right"],
                         "unnecessary alarm days / yr": e["test_scores"]["unnecessary_per_year"]})
        table = pd.DataFrame(rows)

        st.subheader("All 7 models at a glance (test years 2020-2025)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC, our model", f"{table['AUC test'].min():.2f}-{table['AUC test'].max():.2f}",
                  help="1.0 = perfect, 0.5 = a coin flip")
        c2.metric("AUC, season-only baseline", f"{table['season only AUC'].mean():.2f}",
                  help="Forecast = how often inflow happened in that calendar month in 2000-2016")
        c3.metric("Inflow events caught", f"{table['events caught'].min():.0%}-{table['events caught'].max():.0%}",
                  help="An event = consecutive inflow days. Caught = at least one alarm on one of its days")
        c4.metric("Alarms that were right", f"{table['alarms right'].min():.0%}-{table['alarms right'].max():.0%}")

        long = table.assign(random=0.5).melt(
            id_vars="day ahead", value_vars=["AUC test", "season only AUC", "persistence AUC", "random"],
            var_name="forecast", value_name="AUC")
        names = {"AUC test": "our model", "season only AUC": "baseline: season only",
                 "persistence AUC": "baseline: persistence", "random": "baseline: random guess"}
        long["forecast"] = long["forecast"].map(names)
        st.altair_chart(alt.Chart(long).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("day ahead:N", title="forecast for day t+...", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.4, 1])),
            color=alt.Color("forecast:N", title=None, scale=alt.Scale(
                domain=list(names.values()),
                range=[style.GREEN, style.WATER, style.RISK_COLORS["MEDIUM"], style.RISK_COLORS["HIGH"]])),
            strokeDash=alt.condition(alt.datum.forecast == "our model", alt.value([1, 0]), alt.value([5, 4])),
            tooltip=["day ahead", "forecast", alt.Tooltip("AUC:Q", format=".3f")]
        ).properties(height=280), width="stretch")
        st.caption(f"Our model scores {table['AUC test'].min():.2f}-{table['AUC test'].max():.2f}, the season-only "
                   f"baseline {table['season only AUC'].mean():.2f} and persistence "
                   f"{table['persistence AUC'].min():.2f}-{table['persistence AUC'].max():.2f}. Knowing the season "
                   "already helps a lot; the weather, tide and water levels take it from there to near-perfect "
                   "ranking.")

        st.dataframe(table.set_index("day ahead").style.format({
            "AUC test": "{:.3f}", "AUC validation": "{:.3f}", "AUC train": "{:.3f}", "season only AUC": "{:.3f}",
            "persistence AUC": "{:.3f}", "alarm level": "{:.0%}", "events caught": "{:.0%}",
            "alarms right": "{:.0%}", "unnecessary alarm days / yr": "{:.0f}"}), width="stretch")
        drop = table["AUC train"] - table["AUC test"]
        st.caption(f"**Over-fitting check:** the test AUC differs from the training AUC by at most "
                   f"{drop.abs().max():.3f}, and is close to the validation AUC: the models did not memorise the "
                   "training years, they work as well on years they never saw.")

        st.divider()
        h = st.segmented_control("Model detail", horizons, default=2, key="audit_h",
                                 format_func=lambda k: f"t+{k}")
        h = h or 2
        e = entries[h]
        test = e["predictions"][e["predictions"]["period"] == "test"]
        alarm = test["prob"] > e["threshold"]
        inflow = test["inflow"] == 1

        st.subheader(f"Model for t+{h}: {e['name']}, {len(e['features'])} inputs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Inflow days caught", f"{(alarm & inflow).sum() / inflow.sum():.0%}",
                  help=f"{(alarm & inflow).sum()} of {inflow.sum()} inflow days had an alarm")
        c2.metric("Normal days without alarm", f"{(~alarm & ~inflow).sum() / (~inflow).sum():.0%}")
        c3.metric("Missed inflow days", f"{(~alarm & inflow).sum()}")
        c4.metric("False alarms", f"{(alarm & ~inflow).sum()}", help="days with an alarm but no inflow")

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**ROC curve: our model vs the baselines**")
            curves = []
            for column, label in [("prob", "our model"), ("season_only", "baseline: season only"),
                                  ("persistence", "baseline: persistence")]:
                false_rate, hit_rate, cut = roc_curve(test["inflow"], test[column])
                curves.append(pd.DataFrame({"false alarm rate": false_rate, "inflow days caught": hit_rate,
                                            "threshold": cut, "forecast": label}))
            curves = pd.concat(curves)
            ours = curves[curves["forecast"] == "our model"]
            here = ours.iloc[[(ours["threshold"] - e["threshold"]).abs().argmin()]]
            lines = alt.Chart(curves).mark_line(strokeWidth=3).encode(
                x=alt.X("false alarm rate:Q", title="false alarm rate (share of normal days with an alarm)"),
                y=alt.Y("inflow days caught:Q", title="share of inflow days caught"),
                color=alt.Color("forecast:N", title=None, legend=alt.Legend(orient="bottom-right"), scale=alt.Scale(
                    domain=["our model", "baseline: season only", "baseline: persistence"],
                    range=[style.GREEN, style.WATER, style.RISK_COLORS["MEDIUM"]])),
                tooltip=["forecast", alt.Tooltip("threshold:Q", format=".1%"),
                         alt.Tooltip("inflow days caught:Q", format=".0%"),
                         alt.Tooltip("false alarm rate:Q", format=".0%")])
            coin = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                strokeDash=[4, 4], color="#9aa79f").encode(x="x:Q", y="y:Q")
            dot = alt.Chart(here).mark_point(size=160, filled=True, color=style.RISK_COLORS["HIGH"]).encode(
                x="false alarm rate:Q", y="inflow days caught:Q")
            st.altair_chart((coin + lines + dot).properties(height=280), width="stretch")
            st.caption(f"The closer to the top-left corner, the better; the dashed diagonal is a coin flip "
                       f"(AUC 0.5). Red dot = the alarm level used ({e['threshold']:.0%}). AUC = the area under "
                       "each curve.")
        with right:
            st.markdown("**Calibration: does 30% mean 30%?**")
            bins = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1]
            cal = []
            for column, label in [("raw_prob", "before calibration"), ("prob", "after calibration")]:
                g = test.groupby(pd.cut(test[column], bins), observed=True).agg(
                    predicted=(column, "mean"), observed=("inflow", "mean"), days=("inflow", "size"))
                cal.append(g.assign(version=label))
            cal = pd.concat(cal).reset_index(drop=True)
            diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                strokeDash=[4, 4], color="#9aa79f").encode(x="x:Q", y="y:Q")
            points = alt.Chart(cal).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("predicted:Q", title="forecast chance", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("observed:Q", title="how often inflow really happened", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("version:N", scale=alt.Scale(domain=["before calibration", "after calibration"],
                                                             range=[style.RISK_COLORS["MEDIUM"], style.GREEN])),
                tooltip=["version", alt.Tooltip("predicted:Q", format=".0%"),
                         alt.Tooltip("observed:Q", format=".0%"), "days"])
            st.altair_chart((diagonal + points).properties(height=280), width="stretch")
            st.caption("On the dashed line a forecast of X% means inflow X% of the time. The raw model said "
                       "2-5x too much (a side effect of weighting the rare inflow days more in training); "
                       "after calibration on 2017-2019 the points lie close to the line.")

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Which inputs matter most** (average size of their SHAP value, test years)")
            importance = e["importance"].head(15).rename("importance").reset_index()
            importance["input"] = [input_label(f, describe_input) for f in importance["index"]]
            st.altair_chart(alt.Chart(importance).mark_bar(color=style.WATER).encode(
                x=alt.X("importance:Q", title="average |SHAP| (log-odds)"),
                y=alt.Y("input:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                tooltip=["index", alt.Tooltip("importance:Q", format=".2f")]).properties(
                height=chart_height(len(importance))), width="stretch")
            st.caption("t = today, t+1 = tomorrow, ... The weather on the target day and the tide matter most.")
        with right:
            st.markdown("**Choosing the model** (average precision on the validation years 2017-2019)")
            choice = pd.Series(e["validation_scores"]).rename("validation AP").reset_index()
            choice["chosen"] = np.where(choice["index"] == e["name"], "chosen", "not chosen")
            st.altair_chart(alt.Chart(choice).mark_bar(size=40).encode(
                x=alt.X("index:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("validation AP:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("chosen:N", legend=None, scale=alt.Scale(
                    domain=["chosen", "not chosen"], range=[style.GREEN, "#b9c8bd"])),
                tooltip=["index", alt.Tooltip("validation AP:Q", format=".3f")]).properties(height=200),
                width="stretch")
            rule = test[["prob", "inflow"]].rename(columns={"inflow": "inflow_t+2"})
            rule.index = rule.index + pd.Timedelta(days=h)
            events = evaluate_rule(rule, e["threshold"])
            st.markdown(f"**Per event:** {events['events_caught']} of {events['events']} inflow events "
                        f"caught ({events['share_caught']:.0%}); {events['pump_days']} alarm days, of which "
                        f"{events['unnecessary_pump_days']} without inflow "
                        f"(~{events['unnecessary_per_year']:.0f} a year).")

        st.markdown("**Forecast vs reality, one season at a time** (test years)")
        shown = test.copy()
        shown.index = shown.index + pd.Timedelta(days=h)  # the day the forecast is about
        seasons = sorted({d.year if d.month >= 7 else d.year - 1 for d in shown.index})
        season = st.select_slider("Season (July-June)", seasons, value=2023, key="audit_season",
                                  format_func=lambda y: f"{y}-{y + 1}")
        part = shown.loc[f"{season}-09-01":f"{season + 1}-03-31"].reset_index(names="day")
        base = alt.Chart(part).encode(x=alt.X("day:T", title=None))
        real = base.mark_bar(color=style.SKY, opacity=0.9).encode(
            y=alt.Y("inflow:Q", title="chance / inflow (1 = yes)"), tooltip=["day:T", "inflow"])
        line = base.mark_line(color=style.GREEN, strokeWidth=2).encode(
            y="prob:Q", tooltip=["day:T", alt.Tooltip("prob:Q", format=".0%")])
        level = alt.Chart(pd.DataFrame({"y": [e["threshold"]]})).mark_rule(
            strokeDash=[4, 4], color=style.RISK_COLORS["HIGH"]).encode(y="y:Q")
        st.altair_chart((real + line + level).properties(height=240), width="stretch")
        st.caption("Light blue bars = days with real inflow. Green line = the forecast made h days before. "
                   "Red dashed line = alarm level. September to March shown.")

    # ================= INPUTS & OUTPUT =================
    with io:
        st.subheader("Output")
        st.markdown("""
For each day ahead (**t+1 ... t+7**) one number: the **chance that seawater flows into the canals** on that
day, between 0 and 100%, calibrated so that 30% means "on such days inflow happened about 30% of the time".

**What counts as inflow?** The *salt level* = **canal level − sea level at low tide** (IJmuiden). When it is
**below 0**, even at low tide the sea is higher than the canals: water can't drain out and seawater pushes in.
This happened on about 5% of days in 2000-2025 (~18 days a year), almost all from September to February.

**Alarm level.** Every model has an alarm level, chosen on the validation years as the highest one that still
caught 90% of inflow events. Above it the dashboard says *Elevated*; from 30% it says *High*.
""")
        st.subheader("Inputs")
        st.markdown("*t* = the day the forecast is made, *t+h* = the target day. Only information that is "
                    "really available on day *t* is used: the weather **forecast**, the **astronomical** tide "
                    "and water levels **measured up to today**.")
        st.dataframe(INPUTS.set_index("group"), width="stretch")
        counts = pd.Series({f"t+{h}": len(entries[h]["features"]) for h in horizons}, name="inputs")
        st.caption("Inputs per model: " + ", ".join(f"{k}: {v}" for k, v in counts.items())
                   + ". The t+1 model has fewer because 'the day before the target day' is today.")
        with st.expander("Exact input names of each model"):
            st.dataframe(pd.DataFrame({f"t+{h}": pd.Series(entries[h]["features"]) for h in horizons}).fillna(""),
                         width="stretch")

    # ================= TRAINING METHOD =================
    with method:
        n = entries[2]["predictions"]["period"].value_counts()
        st.markdown(f"""
**Data.** One row per day, {history.index.min():%Y}-{history.index.max():%Y}: sea level at IJmuiden
(Rijkswaterstaat, 10-minute values), canal level at the Amstelsluizen (water board AGV) and hourly weather on
the coast at IJmuiden (Open-Meteo), all turned into daily values (*features.py*, the same code as live).

**Split by time, never shuffled**, so the model is always tested on years after the ones it learned from:

| part | years | days (t+2 model) | used for |
|---|---|---|---|
| train | 2000-2016 | {n.get('train', 0):,} | fitting the model |
| validation | 2017-2019 | {n.get('validation', 0):,} | choosing the model type, calibration, alarm level |
| test | 2020-2025 | {n.get('test', 0):,} | the scores on this page, used only once at the end |

**Steps, for every horizon h = 1 ... 7** (*step9_horizons.py*):
1. **Inputs** for day t and the target **inflow on day t+h** (*features.build_features*, *build_target*).
   Days with a missing value are skipped.
2. **Two candidate models** are trained on 2000-2016: a **logistic regression** (inputs scaled first) and a
   **random forest** (300 trees). Inflow is rare (~5% of days), so inflow days count more in training
   (*balanced class weights*); otherwise a model can be 95% right by always saying "no".
3. **Model choice**: the one with the best **average precision on 2017-2019** (like AUC, but it focuses on
   how right the alarms are on the rare inflow days). Logistic regression won for all 7 horizons.
4. **Calibration**: the class weights make the raw chances 2-5x too high, so they are rescaled with a logistic
   fit on the log-odds (*Platt scaling*), on 2017-2019.
5. **Alarm level**: the highest of 5, 10, ... 50% that still catches **90% of inflow events** in 2017-2019
   (missing seawater is worse than an extra pump run).
6. **Test** once on 2020-2025: AUC, compared with the season-only and persistence baselines, events
   caught, alarms right, false alarms per year.

**Explanations.** Every forecast is split into one part per input with **SHAP** (*LinearExplainer*, compared
with 200 typical training days). For a logistic regression this split is exact, and calibration only
rescales it.

**How we picked the inputs** (*step10_seasonality.py*). Season, spring/neap tide, storm surge today, the storm
window and the last 7 days were tested group by group. Together they raised the test AUC of every horizon
by 0.003-0.006 (AUC was already above 0.95), and the average precision, which looks at the rare inflow days
only, by 0.02-0.04.
""")
        try:
            gain = pd.read_csv("output/step10_feature_gain.csv")
            gain["day ahead"] = "t+" + gain["horizon_days"].astype(str)
            long = gain.melt(id_vars="day ahead", value_vars=["AUC without", "AUC with seasonality & effects"],
                             var_name="inputs", value_name="AUC (test)")
            long["inputs"] = long["inputs"].str.replace("AUC ", "")
            st.altair_chart(alt.Chart(long).mark_bar().encode(
                x=alt.X("day ahead:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset="inputs:N",
                y=alt.Y("AUC (test):Q", scale=alt.Scale(domain=[0.94, 0.96])),
                color=alt.Color("inputs:N", scale=alt.Scale(range=["#b9c8bd", style.GREEN])),
                tooltip=["day ahead", "inputs", alt.Tooltip("AUC (test):Q", format=".3f")]
            ).properties(height=240), width="stretch")
            st.caption("Note the y-axis starts at 0.94: the gain in AUC is small but the same for every horizon.")
        except FileNotFoundError:
            st.info("Run  python step10_seasonality.py  to see the gain of the extra inputs.")

    # ================= LIMITATIONS =================
    with limits:
        st.markdown("""
- **Tested with the real weather, not a forecast.** In 2020-2025 the models got the weather that really
  happened. Live they get a weather *forecast*, which gets worse every day ahead. So the scores here barely drop
  from t+1 to t+7, but live skill for days 5-7 will be clearly lower. Archived weather forecasts would allow an
  honest test.
- **The salt level is our definition** (canal level − low tide at IJmuiden). It is a stand-in for seawater
  inflow; the water board should confirm it. Chloride samples can't confirm it: most are in the polders, where
  the salt comes from seepage.
- **One signal for the whole city.** There is one inflow signal (IJmuiden), so no forecast per area.
- **Canal level live.** The canal level is not online; the app uses the value in the sidebar (-0.40 m NAP by
  default). The inputs that depend on it (salt level today, inflow today, last 7 days) are only as good as that.
- **Daily data.** The forecast is per day, not per hour or per tide.
- **The world changes.** The low tide rose ~5 cm since 2000 and 2023-24 had twice the usual inflow days. The
  models know the tide and season, but should be retrained every year (*step9_horizons.py*).
- **Missing inputs** (e.g. the weather forecast doesn't reach far enough) are replaced by a typical value, and
  the forecast page says so.
- **SHAP shows what the model uses, not cause and effect.** Inputs that measure almost the same thing (pressure
  on two days) share their effect in a way that can look odd; the forecast page adds them up per cause.
""")
