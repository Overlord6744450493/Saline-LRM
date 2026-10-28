"""
WEB APP - Seawater inflow warning for the pumps.

Start it in the VS Code terminal with:
    python -m streamlit run app.py
Your browser opens at http://localhost:8501. Stop it with Ctrl+C in the terminal.

Needs (from the step scripts): output/model.joblib and output/predictions.csv (step 5),
output/station_ranking.csv (step 7).
One-time install:  python -m pip install streamlit
"""
import joblib
import pandas as pd
import streamlit as st

import helpers
from helpers import build_live_features, evaluate_rule, pump_plan, today_gmt1

st.set_page_config(page_title="Seawater Warning", layout="wide")


# ---------- loading data (cached, so the app stays fast) ----------
@st.cache_resource
def load_model():
    return joblib.load("output/model.joblib")


@st.cache_data
def load_stations():
    return pd.read_csv("output/station_ranking.csv", index_col="station")


@st.cache_data
def load_past_predictions():
    pred = pd.read_csv("output/predictions.csv", index_col="date", parse_dates=True)
    pred.index = pred.index + pd.Timedelta(days=2)  # the day the prediction is ABOUT
    return pred


@st.cache_data(ttl=1800)  # download again at most every 30 minutes
def load_live_data(today):
    return helpers.load_live_data(today)


# ---------- sidebar: settings ----------
st.sidebar.title("Seawater Warning")
st.sidebar.caption("Predicts 2 days ahead whether seawater will flow into the Amsterdam canals.")
threshold = st.sidebar.slider("Pump threshold (probability)", 0.1, 0.9, 0.5, 0.05,
                              help="Alarm when the predicted probability is above this value. "
                                   "Lower = catch more events, but pump more often for nothing.")
storage_level = st.sidebar.number_input(
    "Canal level today (m NAP)", value=-0.40, step=0.01, format="%.2f",
    help="Amsterdam keeps its canals at about -0.40 m NAP. Enter the real value if you have it.")

bundle = load_model()
model, features = bundle["model"], bundle["features"]
st.sidebar.caption(f"Model: {bundle['name']}, {len(features)} inputs, trained on 2000–2016.")

tab_live, tab_rule, tab_stations, tab_how = st.tabs(
    ["Pump advice (live)", "Pump rule explorer", "Pumping stations", "How it works"])


# ---------- TAB 1: live advice ----------
with tab_live:
    today = today_gmt1()
    target_day = today + pd.Timedelta(days=2)
    st.header(f"Will seawater flow in on {target_day:%A %d %B}?")

    try:
        sea, weather = load_live_data(today)
        X = build_live_features(today, sea, weather, storage_level)
    except Exception as error:  # no internet, API down, ...
        st.error(f"Could not get live data: {error}")
        st.stop()

    if X.isna().any(axis=None):
        st.warning("Some live inputs are missing, so this prediction may be unreliable: "
                   + ", ".join(X.columns[X.isna().any()]))
        X = X.fillna(0)

    prob = model.predict_proba(X[features])[0, 1]
    risk, plan = pump_plan(prob, threshold, load_stations())
    col1, col2 = st.columns([1, 2])
    col1.metric("Inflow probability", f"{prob:.0%}", help="Predicted by the model for the day after tomorrow")
    col1.metric("Risk", risk, help="HIGH = above the threshold, MEDIUM = above half of it")
    with col2:
        if plan:
            st.error(f"### Prepare and release the pumps\nProbability {prob:.0%} is above the threshold "
                     f"{threshold:.0%}. Largest pumping stations:\n" + "\n".join(f"- {p}" for p in plan))
            st.caption("Past data shows no station is measurably more effective than another, so they "
                       "are ranked by capacity. See the 'Pumping stations' tab.")
        else:
            st.success(f"### No action needed\nProbability {prob:.0%} is below the threshold {threshold:.0%}.")
    st.caption("The risk is for the whole Amsterdam canal system: the data has one inflow signal "
               "(at IJmuiden), so a forecast per area is not possible.")

    st.subheader("Sea level at IJmuiden vs the canals")
    chart = sea.set_index("time")[[c for c in ["sea_level", "rws_forecast", "tide_pred"] if c in sea]].copy()
    chart = chart.rename(columns={"sea_level": "measured sea level", "rws_forecast": "Rijkswaterstaat forecast",
                                  "tide_pred": "astronomical tide"})
    chart["canal level"] = storage_level
    st.line_chart(chart, y_label="m NAP")
    st.caption("Inflow happens when even the LOW tide stays above the canal level.")

    st.subheader("Weather forecast (coast at IJmuiden)")
    days = [today, today + pd.Timedelta(days=1), target_day]
    table = weather.loc[days, ["wind_from_nw", "wind", "gust_max", "pressure", "rain"]].copy()
    table.index = ["today", "tomorrow", "day after tomorrow"]
    table.columns = ["wind from NW (m/s)", "avg wind (m/s)", "max gust (m/s)", "pressure (hPa)", "rain (mm)"]
    st.dataframe(table.round(1).T, width="stretch")
    st.caption("Strong north-west wind pushes North Sea water onto the coast: the main cause of inflow.")

    with st.expander("What the model sees (all inputs)"):
        st.dataframe(X[features].T.rename(columns={0: "value"}).round(3), width="stretch")


# ---------- TAB 2: pump rule explorer on past data ----------
with tab_rule:
    st.header("How would the pump rule have done in 2020–2025?")
    st.caption("Test years the model never saw during training. Move the threshold in the sidebar.")
    pred = load_past_predictions()
    test = pred[pred["period"] == "test"]
    result = evaluate_rule(test, threshold)

    c1, c2, c3 = st.columns(3)
    c1.metric("Inflow events caught", f"{result['share_caught']:.0%}",
              help=f"{result['events_caught']} of {result['events']} events had an alarm 2 days before")
    c2.metric("Alarms that were right", f"{result['alarm_right']:.0%}")
    c3.metric("Unnecessary pump days / year", f"{result['unnecessary_per_year']:.0f}")

    st.subheader("Trade-off: every threshold")
    options = pd.DataFrame([evaluate_rule(test, t / 100) for t in range(10, 95, 5)]).set_index("threshold")
    trade = pd.DataFrame({"% of events caught": options["share_caught"] * 100,
                          "unnecessary pump days per year": options["unnecessary_per_year"]})
    st.line_chart(trade, x_label="threshold")

    st.subheader("Predictions vs real inflow, one season at a time")
    seasons = sorted({d.year if d.month >= 7 else d.year - 1 for d in test.index})
    season = st.selectbox("Season (July–June)", seasons, index=len(seasons) - 2,
                          format_func=lambda y: f"{y}–{y + 1}")
    part = test.loc[f"{season}-07-01":f"{season + 1}-06-30"]
    st.line_chart(pd.DataFrame({"predicted probability": part["prob"],
                                "real inflow (1 = yes)": part["inflow_t+2"],
                                "threshold": threshold}))


# ---------- TAB 3: pumping stations ----------
with tab_stations:
    st.header("Which pumping stations help?")
    st.markdown("""
The data has **83 pumps at 44 pumping stations** with hourly flows (1996–2025), but **no names
or locations**, so pumps can't be linked to areas.

For every past inflow day we checked: when a station **ran hard**, did the salt level recover
**faster than the storm alone would predict**?

**Result: no station stands out.** Only 2 of 44 pass the test (p < 0.1), about what luck alone gives
(≈4). Total pumping of all stations together shows no link either (r = −0.01). These are polder pumps
that lift polder water into the canals; the pumps that push water out to sea (IJmuiden) are not in
the data. So the pump plan ranks stations by **capacity**.
""")
    table = load_stations()[["pumps", "capacity_m3_day", "typical_m3_day", "share_days_running",
                             "extra_recovery_cm", "p_value", "verdict"]]
    table = table.sort_values("capacity_m3_day", ascending=False)
    table.columns = ["pumps", "capacity (m3/day)", "typical (m3/day)", "share of days running",
                     "extra recovery when running hard (cm)", "p-value", "verdict"]
    st.dataframe(table.round(2), width="stretch")


# ---------- TAB 4: explanation ----------
with tab_how:
    st.header("How it works")
    st.markdown("""
**The problem.** Seawater sometimes pushes into Amsterdam's water storage. Pumps can push it
back out, but they need to be prepared in advance.

**What "salt level" means.** *canal level − sea level at low tide* (IJmuiden).
When it is **negative**, even at low tide the sea is higher than the canals: water can't drain
out and seawater pushes in. This happened on 4.9% of days in 2000–2025 (≈18 days a year),
almost always from September to February.

**The model.** Every day it predicts inflow for the day after tomorrow from
the weather forecast (wind, gusts, north-west wind, pressure, rain), the astronomical tide
and today's water levels. Logistic regression, trained on 2000–2016, tested on 2020–2025.

**Limitations.**
- The "salt level" definition is our interpretation; the stakeholder should confirm it.
- The model was tested with the real weather instead of real forecasts, so live accuracy will be lower.
- The live canal level is not available online; the app uses the value in the sidebar.
- Daily data only: the app warns for a day, not an hour.

**Data:** AGV water board, Rijkswaterstaat (sea level, free), Open-Meteo (weather, free).
""")
    for image, caption in [("output/step1_salt_level_example.png", "What a 'negative' day looks like"),
                           ("output/step4_inflow_vs_wind.png", "North-west wind drives inflow"),
                           ("output/step6_rule_vs_baseline.png", "Our rule vs the simple baseline")]:
        try:
            st.image(image, caption=caption)
        except Exception:
            st.info(f"Run the step scripts to create {image}")
