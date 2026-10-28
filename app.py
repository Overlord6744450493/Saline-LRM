"""
WEB APP - Seawater inflow warning for the pumps.

Start it in the VS Code terminal with:
    python -m streamlit run app.py
Your browser opens at http://localhost:8501. Stop it with Ctrl+C in the terminal.

Needs (from the step scripts): output/daily_dataset.csv (step 3), output/station_ranking.csv (step 7),
output/models_horizons.joblib (step 9). The command-line forecast (predict.py) uses the 2-day model of step 5.
One-time install:  python -m pip install streamlit shap
"""
import altair as alt
import joblib
import pandas as pd
import streamlit as st

import audit
import helpers
import style
from features import NEW_MOON, SYNODIC_MONTH
from helpers import build_live_features, evaluate_rule, explain_prediction, predict_inflow, pump_plan, today_gmt1

st.set_page_config(page_title="Seawater Watch", page_icon=":material/water:", layout="wide")
style.apply_style()


# ---------- loading data (cached, so the app stays fast) ----------
@st.cache_resource
def load_horizon_models():
    return joblib.load("output/models_horizons.joblib")


@st.cache_data
def load_stations():
    return pd.read_csv("output/station_ranking.csv", index_col="station")


@st.cache_data
def load_history():
    """The daily table the models were trained on (step 3), for the seasonality charts."""
    return pd.read_csv("output/daily_dataset.csv", index_col="date", parse_dates=True)


@st.cache_data(ttl=1800)  # download again at most every 30 minutes
def load_live_data(today, days_ahead):
    return helpers.load_live_data(today, days_ahead)


# ---------- plain-language names for the model inputs ----------
INPUT_NAMES = {
    "wind_from_nw": ("north-west wind", "m/s"), "wind": ("average wind", "m/s"),
    "gust_max": ("strongest gust", "m/s"), "pressure": ("air pressure", "hPa"),
    "pressure_min": ("lowest air pressure", "hPa"), "rain": ("rain", "mm"),
    "tide_pred_low": ("astronomical low tide", "m NAP"), "salt_level": ("salt level", "m"),
    "inflow": ("inflow (1 = yes)", ""), "sea_mean": ("mean sea level", "m NAP"),
    "storage_level": ("canal level", "m NAP"),
}


# the seasonality and other inputs of features.context_features
CONTEXT_NAMES = {
    "nw_max_window": ("strongest north-west wind until then", "m/s"),
    "pressure_min_window": ("lowest air pressure until then", "hPa"),
    "rain_7d": ("rain in the last 7 days", "mm"), "inflow_7d": ("inflow days in the last 7 days", ""),
    "surge_t": ("storm surge today (sea above tide)", "m"),
}


def describe_input(feature, today):
    """'wind_from_nw_t+2' -> ('north-west wind, Thu 30 Oct', 'm/s')."""
    if feature in CONTEXT_NAMES:
        return CONTEXT_NAMES[feature]
    name, suffix = feature.rsplit("_t", 1)
    offset = int(suffix) if suffix else 0
    if name == "tide_range":
        name, label, unit = None, "tide range (small = neap tide)", "m"
    elif name == "nw_3day":
        name, label, unit = None, "north-west wind, 3 days before", "m/s"
    else:
        label, unit = INPUT_NAMES.get(name, (name, ""))
    return f"{label}, {today + pd.Timedelta(days=offset):%a %d %b}", unit


def moon_phase(day):
    """Moon age in days -> plain words (spring tide around new/full moon, neap tide around the quarters)."""
    age = ((day - NEW_MOON).total_seconds() / 86400) % SYNODIC_MONTH
    names = ["Spring tide", "To neap", "Neap tide", "To spring", "Spring tide", "To neap", "Neap tide", "To spring"]
    return names[int((age / SYNODIC_MONTH * 8 + 0.5) % 8)]


def explain_table(contrib, row, today, horizon):
    """SHAP values -> one line per input in plain words. The sin/cos pairs of the season and the
    spring/neap cycle are added together (SHAP values may be added) and shown as one input."""
    target = today + pd.Timedelta(days=horizon)
    group = {"season_sin": "season", "season_cos": "season",
             "springneap_sin": "springneap", "springneap_cos": "springneap"}
    contrib = contrib.groupby(contrib.index.map(lambda f: group.get(f, f)), sort=False).sum()
    rows = []
    for feature, effect in contrib.items():
        if feature == "season":
            label, value = "time of year", f"{target:%d %B}"
        elif feature == "springneap":
            label, value = "spring/neap tide cycle", moon_phase(target)
        else:
            label, unit = describe_input(feature, today)
            value = f"{row[feature]:.2f} {unit}".strip()
        rows.append({"input": label, "value": value, "effect": effect})
    return pd.DataFrame(rows)


# Inputs that measure the same thing (pressure today, tomorrow, lowest pressure, ...) move together,
# so their SHAP values can be large with OPPOSITE signs. Added up per cause they read correctly.
CAUSES = [  # (cause, input names starting with, input shown as the value on the target day)
    ("north-west wind", ("wind_from_nw", "nw_"), "wind_from_nw_t+{h}"),
    ("wind speed and gusts", ("wind_", "gust_max"), "gust_max_t+{h}"),
    ("air pressure", ("pressure",), "pressure_min_t+{h}"),
    ("astronomical tide", ("tide_", "springneap"), "tide_pred_low_t+{h}"),
    ("rain", ("rain",), "rain_7d"),
    ("time of year", ("season",), None),
    ("water levels now", ("salt_level", "sea_mean", "storage_level", "surge", "inflow"), "salt_level_t"),
]


def explain_by_cause(contrib, row, today, horizon):
    """SHAP values added up per cause, with one telling value for each."""
    def cause_of(feature):
        return next(c for c, starts, _ in CAUSES if feature.startswith(starts))

    effects = contrib.groupby(contrib.index.map(cause_of)).sum()
    rows = []
    for cause, _, shown in CAUSES:
        if cause not in effects:
            continue
        if shown is None:
            value = f"{today + pd.Timedelta(days=horizon):%d %B}"
        else:
            feature = shown.format(h=horizon)
            value = f"{describe_input(feature, today)[0]}: {row[feature]:.2f} {describe_input(feature, today)[1]}"
        rows.append({"input": cause, "value": value, "effect": effects[cause]})
    return pd.DataFrame(rows)


def forecast_risk(prob, alarm_level):
    """Risk for the calibrated 7-day forecast: above the model's alarm level (chosen to catch 90% of
    events in 2017-2019) = MEDIUM, 30% or more = HIGH."""
    if prob >= max(0.3, alarm_level):
        return "HIGH"
    return "MEDIUM" if prob > alarm_level else "LOW"


# ---------- sidebar: settings ----------
st.sidebar.title("Seawater Watch")
st.sidebar.caption("Forecasts whether North Sea water will flow into the Amsterdam canals, "
                   "up to 7 days ahead.")
storage_level = st.sidebar.number_input(
    "Canal level today (m NAP)", value=-0.40, step=0.01, format="%.2f",
    help="Amsterdam keeps its canals at about -0.40 m NAP. Enter the real value if you have it.")

st.sidebar.caption("Models: one logistic regression per day ahead (t+1 ... t+7), trained on 2000-2016, "
                   "calibrated on 2017-2019, tested on 2020-2025. Details: *Model audit*.")

style.banner("Seawater Watch", "Seawater inflow forecast for the Amsterdam canal system, "
             "up to 7 days ahead")

tab_forecast, tab_plan, tab_counter, tab_rule, tab_stations, tab_audit = st.tabs(
    ["Forecast", "Pump plan", "Mitigation", "Alarm level explorer", "Pumping stations", "Model audit"])


try:
    horizon_bundle = load_horizon_models()
except FileNotFoundError:
    st.error("output/models_horizons.joblib not found. Run  python step9_horizons.py  first.")
    st.stop()

# ---------- TAB: model audit (first in the code: it needs no live data, so it works offline) ----------
with tab_audit:
    audit.render(horizon_bundle, load_history(), describe_input)


# ---------- TAB 0: 7-day forecast, explained with SHAP ----------
with tab_forecast:
    horizons = sorted(horizon_bundle["horizons"])
    today = today_gmt1()
    st.caption(f"Forecast made on **{today:%A %d %B %Y}** from the live weather forecast, the "
               "astronomical tide and today's sea level at IJmuiden. One model per day ahead.")

    try:
        sea, weather = load_live_data(today, max(horizons))
    except Exception as error:  # no internet, API down, ...
        st.error(f"Could not get live data: {error}")
        st.stop()

    def forecast_week(canal_level):
        """Calibrated probability for every day ahead, the model inputs, and the inputs that were missing."""
        rows, inputs, filled = [], {}, {}
        for h in horizons:
            entry = horizon_bundle["horizons"][h]
            X = build_live_features(today, sea, weather, canal_level, h, context=True)[entry["features"]]
            missing = list(X.columns[X.isna().any()])
            if missing:  # e.g. the weather forecast doesn't reach this far yet
                X = X.fillna(entry["background"].median())
                filled[h] = missing
            inputs[h] = X
            prob = float(predict_inflow(entry, X)[0])
            rows.append({"days ahead": h, "day": today + pd.Timedelta(days=h), "probability": prob,
                         "alarm level": entry["threshold"], "risk": forecast_risk(prob, entry["threshold"])})
        return pd.DataFrame(rows).set_index("days ahead"), inputs, filled

    forecast, inputs, filled = forecast_week(storage_level)

    # --- which day? opens on the riskiest day of the week ---
    def day_name(h):
        return "Tomorrow" if h == 1 else f"{today + pd.Timedelta(days=h):%a %d %b}"

    pick = st.segmented_control(
        "Forecast day", horizons, default=int(forecast["probability"].idxmax()),
        format_func=lambda h: f"{day_name(h)} · {forecast.loc[h, 'probability']:.0%}", key="forecast_day")
    if pick is None:  # clicking the selected day again un-selects it
        pick = int(forecast["probability"].idxmax())
    chosen = forecast.loc[pick]

    if chosen["risk"] == "HIGH":
        message = ("<b>Action recommended.</b> Prepare the pumps and consider keeping the canal level up "
                   "(see <i>Pump plan</i> and <i>Mitigation</i>).")
    elif chosen["risk"] == "MEDIUM":
        message = (f"<b>Heightened alert.</b> Above the alarm level of {chosen['alarm level']:.0%}; at that level "
                   "the forecast caught 90% of inflow events in 2017-2019.")
    else:
        message = f"<b>No action required.</b> Below the alarm level of {chosen['alarm level']:.0%}."
    ahead = "tomorrow" if pick == 1 else f"in {pick} days"
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        style.main_card(f"Chance of seawater inflow on <b>{chosen['day']:%A %d %B}</b> ({ahead})",
                        chosen["probability"], chosen["risk"], message)
    with col2:
        st.markdown("##### The coming 7 days")
        for (h, row), col in zip(forecast.iterrows(), st.columns(len(forecast))):
            with col:
                style.day_card(row["day"], row["probability"], row["risk"], main=h == pick)
        chart = forecast.assign(label=forecast["day"].dt.strftime("%a %d %b"))
        top = max(0.5, chart["probability"].max() * 1.15)
        bars = alt.Chart(chart).mark_bar(size=34).encode(
            x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, top]),
                    title="chance of inflow"),
            color=alt.Color("risk:N", legend=None, scale=alt.Scale(
                domain=list(style.RISK_COLORS), range=list(style.RISK_COLORS.values()))),
            opacity=alt.condition(alt.datum.label == f"{chosen['day']:%a %d %b}", alt.value(1), alt.value(0.55)),
            tooltip=[alt.Tooltip("label:N", title="day"), alt.Tooltip("probability:Q", format=".0%"), "risk:N"])
        alarm = alt.Chart(chart).mark_tick(color=style.DEEP, thickness=2, size=44).encode(
            x=alt.X("label:N", sort=None), y="alarm level:Q",
            tooltip=[alt.Tooltip("alarm level:Q", format=".0%")])
        st.altair_chart((bars + alarm).properties(height=210), width="stretch")
    st.caption("The chances are calibrated: on days forecast at 30%, inflow happened about 30% of the time. "
               "Dark tick = alarm level of that day's model. Further ahead the weather forecast is less "
               "certain, so treat days 5-7 as an early warning. Model performance: see *Model audit*.")
    if filled:
        st.warning("Some inputs were not available yet and were replaced by a typical value: "
                   + "; ".join(f"t+{h}: {', '.join(cols)}" for h, cols in filled.items()))

    # --- why? SHAP ---
    st.subheader(f"Explanation of the forecast for {chosen['day']:%A %d %B}")
    entry = horizon_bundle["horizons"][pick]
    contrib, base, unit = explain_prediction(entry, inputs[pick])
    explained = explain_table(contrib, inputs[pick].iloc[0], today, pick)
    by_cause = explain_by_cause(contrib, inputs[pick].iloc[0], today, pick)
    shap_table = by_cause.reindex(by_cause["effect"].abs().sort_values(ascending=False).index)
    shap_table["direction"] = shap_table["effect"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
    shap_chart = alt.Chart(shap_table).mark_bar().encode(
        x=alt.X("effect:Q", title=f"effect on the prediction ({unit})"),
        y=alt.Y("input:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=13)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                       range=[style.RISK_COLORS["HIGH"], style.GREEN]),
                        legend=None),
        tooltip=["input", "value", alt.Tooltip("effect:Q", format="+.2f"), "direction"])
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.altair_chart(shap_chart.properties(height=36 * len(shap_table)), width="stretch")
    with right:
        up = shap_table[shap_table["effect"] > 0].head(3)
        down = shap_table[shap_table["effect"] < 0].head(3)
        st.markdown(f"Compared with a typical day (the model's starting point), the chance for "
                    f"**{chosen['day']:%A}** is **{chosen['probability']:.0%}**. Explained with SHAP: "
                    "every cause gets its share of the prediction.")
        if len(up):
            st.markdown("**Raising the risk:** "
                        + "; ".join(f"{r.input} ({r.value})" for r in up.itertuples()))
        if len(down):
            st.markdown("**Lowering the risk:** "
                        + "; ".join(f"{r.input} ({r.value})" for r in down.itertuples()))
    with st.expander("All inputs and their SHAP values"):
        st.dataframe(explained.set_index("input").rename(columns={"effect": "SHAP effect"}).round(3),
                     width="stretch")
        st.caption(f"SHAP splits the prediction into one part per input. Typical day = {base:.2f} {unit}; "
                   "adding all effects gives this forecast. For logistic regression the split is exact. "
                   "Inputs that measure the same thing (e.g. pressure on two days) can get large effects "
                   "with opposite signs; that is why the chart above adds them up per cause.")

    # --- weather forecast behind it ---
    st.subheader("Weather forecast and tide (coast at IJmuiden)")
    days = [today + pd.Timedelta(days=h) for h in [0, *horizons]]
    table = weather.reindex(days)[["wind_from_nw", "wind", "gust_max", "pressure", "rain"]]
    tide = helpers.live_daily_sea_level(sea).reindex(days)
    table["tide"] = tide["tide_pred_low"]
    table["range"] = tide["tide_pred_high"] - tide["tide_pred_low"]
    table.index = [f"{d:%a %d %b}" for d in days]
    table.columns = ["wind from NW (m/s)", "avg wind (m/s)", "max gust (m/s)", "pressure (hPa)", "rain (mm)",
                     "astronomical low tide (m NAP)", "tide range (m)"]
    table = table.round(2).T.astype(str)  # text, so the moon row fits in the same table
    table.loc["spring/neap tide"] = [moon_phase(d) for d in days]
    st.dataframe(table, width="stretch")
    st.caption("Strong north-west wind and low pressure push North Sea water onto the coast: "
               "the main cause of inflow. Around half moon (neap tide) low tide stays high, so the "
               "canals drain worse: inflow is about 2.5 times as likely as around full or new moon.")


# ---------- TAB: how to counter the inflow ----------
with tab_counter:
    st.header("Mitigation")
    st.markdown("""
Inflow happens when the **canal level is below the sea at low tide** (salt level < 0).
We can't change the sea, but the **canal level can be managed**: every centimetre the canal is
higher before a storm is a centimetre of margin. The forecast gives 2 to 7 days to build that margin.
""")

    st.subheader("Scenario: a higher canal level")
    raise_cm = st.slider("Keep the canal level higher by (cm)", 0, 30, 10, 1)
    raised, _, _ = forecast_week(storage_level + raise_cm / 100)
    status_quo = f"Status quo: canal at {storage_level:+.2f} m NAP"
    what_if = f"What if: canal +{raise_cm} cm ({storage_level + raise_cm / 100:+.2f} m NAP)"
    compare = pd.DataFrame({"day": forecast["day"].dt.strftime("%a %d %b"), status_quo: forecast["probability"],
                            what_if: raised["probability"]}).set_index("day").T
    # status quo in light green, the scenario in the colour of the slider that sets it
    row_colors = {status_quo: f"background-color: {style.LIGHT_GREEN}; color: {style.INK}",
                  what_if: f"background-color: {style.GREEN}; color: white; font-weight: 600"}
    st.dataframe(compare.style.format("{:.0%}").apply(lambda row: [row_colors[row.name]] * len(row), axis=1),
                 width="stretch")
    st.caption("The dark green row is a scenario, not a forecast: the same forecast as the light green row, but "
               "with the canal level set by the slider above. The model learned this effect from history, so "
               "treat it as an indication, not a guarantee.")

    st.subheader("Historical evidence (2000-2025)")
    try:
        buffer = pd.read_csv("output/step10_canal_buffer.csv")
        c1, c2, c3 = st.columns(3)
        for col, cm in zip([c1, c2, c3], [10, 20, 30]):
            share = buffer.loc[buffer["raise_cm"] == cm, "share_avoided"].iloc[0]
            col.metric(f"Canal +{cm} cm", f"{share:.0%}", help="of all past inflow days would not have happened")
        st.altair_chart(alt.Chart(buffer.assign(pct=buffer["share_avoided"] * 100)).mark_area(
            color=style.SKY, line={"color": style.GREEN, "strokeWidth": 3}, opacity=0.8).encode(
            x=alt.X("raise_cm:Q", title="canal level kept higher by (cm)"),
            y=alt.Y("pct:Q", title="% of inflow days avoided"),
            tooltip=[alt.Tooltip("raise_cm:Q", title="canal higher by (cm)"),
                     alt.Tooltip("pct:Q", title="% avoided", format=".0f")]).properties(height=240),
            width="stretch")
    except FileNotFoundError:
        st.info("Run  python step10_seasonality.py  to create this analysis.")
    st.markdown("""
- **The canal is not raised before storms today.** Around the start of an inflow event the canal
  level is 0.5 to 1 cm *below* normal for the month: there is room to act on the forecast.
- **The polder pumps can't build the margin alone.** They pump about twice as much on inflow days,
  but each million m³ goes with only ~0.5 cm higher canal level the next day: +10 cm would take about a
  month of normal pumping. The margin has to come from **draining less to the sea** in the days
  before the storm (the sluices and pumps at IJmuiden / Schellingwoude, which are not in our data).
- **Fresh water flushing helps against salt.** When the Amstel carries the most water (Berlagebrug),
  chloride in the samples is about 11% below normal; when it carries the least, about 11% above.
- **The margin is shrinking.** The astronomical low tide rose about 5 cm since 2000, the canal level
  did not: the margin went from ~33 cm (2000-04) to ~27 cm (2020-24), and 2023-24 had twice the usual
  inflow days. Raising the target canal level a few cm in winter would win that back.

*All of this is correlation in historical data. The allowed canal level (quays, polder dikes, boats)
must be checked with the water board before acting on it.*
""")
    st.subheader("Seasonal pattern")
    try:
        history = load_history()
    except FileNotFoundError:
        st.info("Run the step scripts (step 3) to create output/daily_dataset.csv.")
        st.stop()
    months = history.groupby(history.index.month).agg(
        inflow=("inflow", "mean"), canal=("storage_level", "mean"), tide_low=("tide_pred_low", "mean"),
        storm=("wind_from_nw", lambda w: (w > 8).mean()))
    months["month"] = pd.to_datetime(months.index.astype(str), format="%m").strftime("%b")
    months["inflow %"] = months["inflow"] * 100
    months["margin (cm)"] = (months["canal"] - months["tide_low"]) * 100
    months["storm days %"] = months["storm"] * 100
    months["season"] = ["winter" if m in (9, 10, 11, 12, 1, 2) else "summer" for m in months.index]
    order = list(months["month"])

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Share of days with inflow, per month (2000-2025)**")
        st.altair_chart(alt.Chart(months).mark_bar(size=26).encode(
            x=alt.X("month:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("inflow %:Q", title="% of days with inflow"),
            color=alt.Color("season:N", legend=None, scale=alt.Scale(
                domain=["winter", "summer"], range=[style.WATER, style.REED])),
            tooltip=["month", alt.Tooltip("inflow %:Q", format=".1f")]).properties(height=260), width="stretch")
    with right:
        st.markdown("**Room before the sea gets in, and how often it storms**")
        base = alt.Chart(months).encode(x=alt.X("month:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)))
        margin = base.mark_area(color=style.SKY, line={"color": style.WATER}, opacity=0.8).encode(
            y=alt.Y("margin (cm):Q", title="canal level − low tide (cm)", scale=alt.Scale(zero=True)),
            tooltip=["month", alt.Tooltip("margin (cm):Q", format=".0f")])
        storms = base.mark_line(color=style.RISK_COLORS["HIGH"], point={"color": style.RISK_COLORS["HIGH"]}, strokeWidth=3).encode(
            y=alt.Y("storm days %:Q", title="days with strong north-west wind (%)",
                    axis=alt.Axis(titleColor=style.RISK_COLORS["HIGH"])),
            tooltip=["month", alt.Tooltip("storm days %:Q", format=".1f")])
        st.altair_chart(alt.layer(margin, storms).resolve_scale(y="independent").properties(height=260),
                        width="stretch")
        st.caption("Blue area: margin between the canal level and the astronomical low tide. "
                   "Red line: days with north-west wind above 8 m/s.")

    m = months.set_index("month")
    style.note(f"""
<b>Two things line up in autumn and winter: the sea's baseline is higher, and storms are more frequent.</b>
The canal level doesn't change with the season (about {m['canal'].mean() * 100:.0f} cm NAP all year),
so the margin shrinks.
<ol style="margin:8px 0 4px">
<li><b>The sea sits about 20 cm higher in autumn, even without wind.</b> The astronomical low tide is about
{m.loc['Jun', 'tide_low'] * 100:.0f} cm NAP in June but about {m.loc['Oct', 'tide_low'] * 100:.0f} cm in
October. Tide tables include a yearly cycle of mean sea level: warm water from the summer expands and peaks
in autumn, plus seasonal wind and pressure patterns. So the margin is about
{m.loc['Jun', 'margin (cm)']:.0f} cm in early summer but only about {m.loc['Oct', 'margin (cm)']:.0f} cm in
October, and a modest surge is enough to close it.</li>
<li><b>Winter brings the storms.</b> Strong north-west wind blows on {m.loc['Dec', 'storm days %']:.1f} to
{m.loc['Jan', 'storm days %']:.1f}% of days in December-January, versus almost none in May-August. That wind
pushes North Sea water onto the Dutch coast.</li>
</ol>
<b>How they combine:</b> September-October have few storms but the highest baseline, so even moderate wind
causes inflow. December-January have a falling baseline but the most storms. By March storms still happen,
but the baseline is about 20 cm lower, so inflow drops fast. Rain hardly differs by month and doesn't
explain the pattern.
""")

    st.subheader("Spring tide, neap tide and a shrinking margin")
    left, right = st.columns(2, gap="large")
    moon_age = ((history.index - NEW_MOON).total_seconds() / 86400) % SYNODIC_MONTH
    by_moon = history.groupby((moon_age // (SYNODIC_MONTH / 8)).astype(int))["inflow"].mean().mul(100)
    by_moon = by_moon.rename("inflow %").to_frame().assign(phase=[
        "new moon", "waxing", "first quarter", "waxing ", "full moon", "waning", "last quarter", "waning "])
    by_moon["tide"] = ["neap" if "quarter" in p else "spring" if "moon" in p else "between" for p in by_moon["phase"]]
    with left:
        st.markdown("**Inflow by moon phase**")
        st.altair_chart(alt.Chart(by_moon).mark_bar(size=30).encode(
            x=alt.X("phase:N", sort=list(by_moon["phase"]), title=None,
                  axis=alt.Axis(labelAngle=-30, labelOverlap=False)),
            y=alt.Y("inflow %:Q", title="% of days with inflow"),
            color=alt.Color("tide:N", title="tide", scale=alt.Scale(
                domain=["spring", "between", "neap"], range=[style.REED, style.SKY, style.WATER])),
            tooltip=["phase", alt.Tooltip("inflow %:Q", format=".1f")]).properties(height=260), width="stretch")
        st.caption("Around half moon (neap tide) the tide range is small: low tide stays high and the canals "
                   "drain worse, so inflow is about 2.5 times as likely as around full or new moon.")
    winters = history[history.index.month.isin([9, 10, 11, 12, 1, 2]) & (history.index < "2025-01-01")]
    per_year = winters.groupby(winters.index.year).agg(inflow=("inflow", "sum"), canal=("storage_level", "mean"),
                                                       tide_low=("tide_pred_low", "mean")).reset_index()
    per_year["margin (cm)"] = (per_year["canal"] - per_year["tide_low"]) * 100
    with right:
        st.markdown("**Inflow days per winter and the margin**")
        base = alt.Chart(per_year).encode(x=alt.X("date:O", title=None, axis=alt.Axis(labelAngle=-90)))
        bars = base.mark_bar(color=style.WATER, opacity=0.75).encode(
            y=alt.Y("inflow:Q", title="inflow days (Sep-Feb)"), tooltip=["date", "inflow"])
        line = base.mark_line(color=style.RISK_COLORS["HIGH"], point={"color": style.RISK_COLORS["HIGH"]}, strokeWidth=3).encode(
            y=alt.Y("margin (cm):Q", title="margin (cm)", scale=alt.Scale(zero=False),
                    axis=alt.Axis(titleColor=style.RISK_COLORS["HIGH"])),
            tooltip=["date", alt.Tooltip("margin (cm):Q", format=".0f")])
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=260),
                        width="stretch")
        st.caption("The astronomical low tide rose ~5 cm since 2000 and the canal level did not: the margin "
                   "went from ~33 cm to ~27 cm, and 2023-24 had twice the usual inflow days.")


# ---------- TAB: pump plan for the coming week ----------
with tab_plan:
    st.header("Pump plan for the coming week")
    style.note("<b>What this tab is.</b> It turns the forecast into actions: for every day whose chance of "
               "inflow is above the alarm level, which pumping stations to prepare. It uses the same models "
               "and numbers as the <i>Forecast</i> tab. The alarm level of each day was chosen to catch 90% of "
               "inflow events in 2017-2019; see the <i>Alarm level explorer</i> for what other levels would do.")
    stations = load_stations()
    plan_rows, alarm_days = [], []
    for h, row in forecast.iterrows():
        risk, plan = pump_plan(row["probability"], row["alarm level"], stations)
        if plan:
            alarm_days.append(row["day"])
        plan_rows.append({"day": f"{row['day']:%a %d %b}", "chance of inflow": row["probability"],
                          "alarm level": row["alarm level"],
                          "action": "Prepare pumps" if plan else "No action required"})
    plan_table = pd.DataFrame(plan_rows).set_index("day")
    action_colors = {"Prepare pumps": f"background-color: {style.RISK_COLORS['HIGH']}; color: white",
                     "No action required": f"background-color: {style.LIGHT_GREEN}"}
    st.dataframe(plan_table.style.format({"chance of inflow": "{:.0%}", "alarm level": "{:.0%}"})
                 .map(lambda v: action_colors.get(v, ""), subset=["action"]), width="stretch")

    if alarm_days:
        _, plan = pump_plan(1.0, 0.5, stations)
        st.error(f"**Prepare the pumps for {', '.join(f'{d:%a %d %b}' for d in alarm_days)}.** "
                 "Largest pumping stations:\n" + "\n".join(f"- {p}" for p in plan))
    else:
        st.success("**No action required this week.** No day is above its alarm level.")
    st.caption("Stations are ranked by capacity: past data shows no station is measurably more effective "
               "than another (see *Pumping stations*). The risk is for the whole canal system: the data has "
               "one inflow signal (IJmuiden), so a plan per area is not possible.")

    st.subheader("Sea level at IJmuiden vs the canal level")
    chart = sea.set_index("time")[[c for c in ["sea_level", "rws_forecast", "tide_pred"] if c in sea]].copy()
    chart = chart.rename(columns={"sea_level": "measured sea level", "rws_forecast": "Rijkswaterstaat forecast",
                                  "tide_pred": "astronomical tide"})
    chart["canal level"] = storage_level
    st.line_chart(chart, y_label="m NAP")
    st.caption("Inflow happens when even the LOW tide stays above the canal level.")


# ---------- TAB: alarm level explorer (back-test on the test years) ----------
with tab_rule:
    st.header("Alarm level explorer")
    style.note("<b>What this tab is.</b> A back-test of the rule <i>\"prepare the pumps when the chance of "
               "inflow is above the alarm level\"</i> on the years 2020-2025, which the models never saw while "
               "they were built. Use it to choose an alarm level: a lower level catches more inflow events but "
               "means more days of pumping for nothing. It changes nothing in the other tabs.")
    left, right = st.columns(2, gap="large")
    with left:
        h = st.segmented_control("Forecast made ... days ahead", horizons, default=2, key="rule_h",
                                 format_func=lambda k: f"t+{k}") or 2
    entry = horizon_bundle["horizons"][h]
    with right:
        level = st.slider("Alarm level", 0.01, 0.60, float(entry["threshold"]), 0.01, format="%.2f",
                          key=f"rule_level_{h}",
                          help=f"The level used in the forecast for t+{h} is {entry['threshold']:.0%}")
    pred = entry["predictions"]
    test = pred[pred["period"] == "test"][["prob", "inflow"]].rename(columns={"inflow": "inflow_t+2"})
    test.index = test.index + pd.Timedelta(days=h)  # the day the forecast is about
    result = evaluate_rule(test, level)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inflow events caught", f"{result['share_caught']:.0%}",
              help=f"{result['events_caught']} of {result['events']} events had at least one alarm")
    c2.metric("Alarms that were right", f"{result['alarm_right']:.0%}")
    c3.metric("Alarm days per year", f"{result['pump_days'] / ((test.index.max() - test.index.min()).days / 365.25):.0f}")
    c4.metric("False alarms per year", f"{result['unnecessary_per_year']:.0f}",
              help="Days with an alarm but no inflow: pump days for nothing")

    st.subheader("Trade-off at every alarm level")
    options = pd.DataFrame([evaluate_rule(test, t / 100) for t in range(1, 61)])
    trade = options.assign(**{"events caught (%)": options["share_caught"] * 100,
                              "unnecessary pump days per year": options["unnecessary_per_year"]})
    base = alt.Chart(trade).encode(x=alt.X("threshold:Q", title="alarm level", axis=alt.Axis(format="%")))
    caught = base.mark_line(color=style.GREEN, strokeWidth=3).encode(
        y=alt.Y("events caught (%):Q", axis=alt.Axis(titleColor=style.GREEN)),
        tooltip=[alt.Tooltip("threshold:Q", format=".0%"), alt.Tooltip("events caught (%):Q", format=".0f")])
    wasted = base.mark_line(color=style.RISK_COLORS["HIGH"], strokeWidth=3).encode(
        y=alt.Y("unnecessary pump days per year:Q", axis=alt.Axis(titleColor=style.RISK_COLORS["HIGH"])),
        tooltip=[alt.Tooltip("threshold:Q", format=".0%"),
                 alt.Tooltip("unnecessary pump days per year:Q", format=".0f")])
    chosen = alt.Chart(pd.DataFrame({"threshold": [level]})).mark_rule(
        strokeDash=[4, 4], color=style.DEEP).encode(x="threshold:Q")
    st.altair_chart(alt.layer(caught, wasted, chosen).resolve_scale(y="independent").properties(height=280),
                    width="stretch")
    st.caption("Green: share of inflow events with at least one alarm. Red: days with an alarm but no inflow. "
               "Dashed line: the alarm level chosen above.")

    st.subheader("Forecast vs real inflow, one season at a time")
    seasons = sorted({d.year if d.month >= 7 else d.year - 1 for d in test.index})
    season = st.select_slider("Season (July-June)", seasons, value=seasons[-2], key="rule_season",
                              format_func=lambda y: f"{y}-{y + 1}")
    part = test.loc[f"{season}-09-01":f"{season + 1}-03-31"].reset_index(names="day")
    timeline = alt.Chart(part).encode(x=alt.X("day:T", title=None))
    real = timeline.mark_bar(color=style.SKY).encode(y=alt.Y("inflow_t+2:Q", title="chance / inflow (1 = yes)"),
                                                     tooltip=["day:T"])
    line = timeline.mark_line(color=style.GREEN, strokeWidth=2).encode(
        y="prob:Q", tooltip=["day:T", alt.Tooltip("prob:Q", format=".0%")])
    rule = alt.Chart(pd.DataFrame({"y": [level]})).mark_rule(
        strokeDash=[4, 4], color=style.RISK_COLORS["HIGH"]).encode(y="y:Q")
    st.altair_chart((real + line + rule).properties(height=240), width="stretch")
    st.caption("Light blue bars: days with real inflow. Green line: the forecast. Red dashed line: the alarm level.")


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
