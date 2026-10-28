"""
STYLE - the look of the web app (app.py): colours, the canal banner and the risk cards.
The theme colours themselves are in .streamlit/config.toml.
"""
import altair as alt
import streamlit as st

# canal water, reeds and Amsterdam brick
GREEN, DEEP, WATER, SKY, REED, CREAM, INK = "#2f7d5b", "#1d4d3a", "#2b7a8c", "#cfe6e3", "#8db255", "#f5f3ea", "#1f2e27"
LIGHT_GREEN = "#dcebd8"
RISK_COLORS = {"LOW": "#3f9c6d", "MEDIUM": "#e0a526", "HIGH": "#c8452f"}
RISK_TEXT = {"LOW": "Low risk", "MEDIUM": "Elevated", "HIGH": "High risk"}

CSS = f"""
<style>
.block-container {{padding-top: 2.6rem; max-width: 1250px;}}
h1, h2, h3 {{color: {DEEP}; letter-spacing: -0.01em;}}
.stTabs [data-baseweb="tab-list"] {{gap: 4px; border-bottom: 2px solid #d6e2d3;}}
.stTabs [data-baseweb="tab"] {{font-size: 1rem; padding: 6px 4px;}}
.stTabs [aria-selected="true"] p {{color: {GREEN}; font-weight: 700;}}
[data-testid="stMetric"] {{background: white; border: 1px solid #d6e2d3; border-radius: 14px;
    padding: 12px 16px; box-shadow: 0 1px 3px rgba(29, 77, 58, 0.08);}}
[data-testid="stSidebar"] {{background: linear-gradient(180deg, #e6eee2 0%, #d5e7e2 100%);}}
.hero {{position: relative; border-radius: 18px; overflow: hidden; margin-bottom: 1rem;
    box-shadow: 0 4px 14px rgba(29, 77, 58, 0.18);}}
.hero svg {{display: block; width: 100%; height: 210px;}}
.hero-text {{position: absolute; left: 30px; top: 22px; max-width: 46%; color: white;
    text-shadow: 0 1px 6px rgba(0, 0, 0, 0.35);}}
.hero-text .title {{font-size: 2.3rem; font-weight: 700; line-height: 1.1;}}
.hero-text p {{margin: 6px 0 0; font-size: 1.02rem; line-height: 1.4; opacity: 0.95;}}
.card {{background: white; border-radius: 16px; padding: 18px 22px; border-left: 8px solid;
    box-shadow: 0 1px 4px rgba(29, 77, 58, 0.1); height: 100%;}}
.card .big {{font-size: 2.8rem; font-weight: 700; line-height: 1.1;}}
.card .label {{color: #5b6b63; font-size: 0.95rem;}}
.card .pill {{display: inline-block; color: white; border-radius: 999px; padding: 2px 12px;
    font-weight: 600; font-size: 0.9rem; margin-top: 6px;}}
.day {{background: white; border-radius: 14px; padding: 12px 8px; text-align: center;
    box-shadow: 0 1px 3px rgba(29, 77, 58, 0.1); border-top: 6px solid;}}
.day .name {{font-weight: 600; color: {DEEP};}}
.day .date {{font-size: 0.8rem; color: #6b7a72;}}
.day .pct {{font-size: 1.7rem; font-weight: 700; margin: 4px 0;}}
.day .drop {{height: 8px; border-radius: 4px; background: #e3ece6; overflow: hidden;}}
.day .drop div {{height: 100%;}}
.day.main {{outline: 3px solid {GREEN}; outline-offset: -1px;}}
.note {{background: #e6eee2; border-radius: 12px; padding: 14px 18px; border-left: 5px solid {REED};}}
</style>
"""

# Amsterdam canal at dusk: gabled houses, trees, an arched bridge and the water
BANNER = f"""
<svg viewBox="0 0 1200 200" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#3f8f86"/><stop offset="1" stop-color="#a9d4c4"/>
    </linearGradient>
    <linearGradient id="shade" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{DEEP}" stop-opacity="0.75"/><stop offset="0.55" stop-color="{DEEP}" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="water" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{WATER}"/><stop offset="1" stop-color="#174f5c"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="200" fill="url(#sky)"/>
  <circle cx="1080" cy="48" r="22" fill="#f4f1d8" opacity="0.85"/>
  <g fill="#7a3b2e" opacity="0.92">
    <path d="M560 140 V78 l18 -22 l18 22 V140z"/><path d="M600 140 V70 l12 -18 l12 18 V140z"/>
    <path d="M628 140 V84 l20 -24 l20 24 V140z"/><path d="M672 140 V74 l14 -20 l14 20 V140z"/>
    <path d="M704 140 V80 l18 -24 l18 24 V140z"/><path d="M744 140 V68 l13 -19 l13 19 V140z"/>
    <path d="M774 140 V82 l20 -22 l20 22 V140z"/><path d="M818 140 V72 l15 -21 l15 21 V140z"/>
    <path d="M852 140 V80 l18 -23 l18 23 V140z"/><path d="M892 140 V70 l12 -18 l12 18 V140z"/>
    <path d="M920 140 V84 l20 -24 l20 24 V140z"/><path d="M964 140 V76 l16 -21 l16 21 V140z"/>
    <path d="M1000 140 V82 l18 -22 l18 22 V140z"/><path d="M1040 140 V72 l14 -20 l14 20 V140z"/>
    <path d="M1072 140 V80 l18 -23 l18 23 V140z"/><path d="M1112 140 V70 l14 -19 l14 19 V140z"/>
    <path d="M1144 140 V82 l20 -22 l20 22 V140z"/>
  </g>
  <g fill="#f6e7b8" opacity="0.8">
    <rect x="570" y="96" width="6" height="8"/><rect x="582" y="112" width="6" height="8"/>
    <rect x="640" y="100" width="6" height="8"/><rect x="716" y="104" width="6" height="8"/>
    <rect x="786" y="98" width="6" height="8"/><rect x="864" y="108" width="6" height="8"/>
    <rect x="932" y="100" width="6" height="8"/><rect x="1012" y="104" width="6" height="8"/>
    <rect x="1084" y="96" width="6" height="8"/><rect x="1156" y="110" width="6" height="8"/>
  </g>
  <g fill="{DEEP}">
    <circle cx="540" cy="112" r="26"/><circle cx="520" cy="122" r="20"/><rect x="537" y="118" width="5" height="24"/>
    <circle cx="1190" cy="110" r="24"/><circle cx="470" cy="118" r="22"/><rect x="468" y="124" width="5" height="18"/>
  </g>
  <path d="M0 140 H1200 V200 H0z" fill="url(#water)"/>
  <path d="M760 140 q90 -46 180 0" fill="none" stroke="#5d2f25" stroke-width="10"/>
  <path d="M740 140 H960" stroke="#5d2f25" stroke-width="6"/>
  <g stroke="#cfe6e3" stroke-width="2" fill="none" opacity="0.55">
    <path d="M40 160 q15 -6 30 0 t30 0 t30 0"/><path d="M300 175 q15 -6 30 0 t30 0 t30 0"/>
    <path d="M620 165 q15 -6 30 0 t30 0"/><path d="M880 182 q15 -6 30 0 t30 0 t30 0"/>
    <path d="M1060 162 q15 -6 30 0 t30 0"/>
  </g>
  <g stroke="{REED}" stroke-width="3" stroke-linecap="round">
    <path d="M20 200 q4 -40 -2 -60"/><path d="M34 200 q2 -34 8 -52"/><path d="M48 200 q-2 -30 -8 -44"/>
    <path d="M1150 200 q4 -36 -2 -54"/><path d="M1166 200 q2 -30 8 -46"/>
  </g>
  <path d="M330 168 h70 l-10 10 h-52z" fill="#c8452f"/><rect x="352" y="156" width="22" height="12" fill="#f5f3ea"/>
  <rect width="1200" height="200" fill="url(#shade)"/>
</svg>
"""


def apply_style():
    """Theme for the page and for every Altair chart."""
    st.markdown(CSS, unsafe_allow_html=True)

    @alt.theme.register("canal", enable=True)
    def theme():
        return alt.theme.ThemeConfig({"config": {
            "background": "transparent", "view": {"stroke": None},
            "axis": {"labelColor": "#4a5a52", "titleColor": "#4a5a52", "gridColor": "#dfe7dc",
                     "domainColor": "#b9c8bd", "labelFontSize": 12, "titleFontSize": 12},
            "range": {"category": [GREEN, WATER, "#e0a526", "#c8452f", REED, "#7a3b2e"]},
            "bar": {"cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
        }})


def banner(title, subtitle):
    st.markdown(f'<div class="hero">{BANNER}<div class="hero-text"><div class="title">{title}</div>'
                f'<p>{subtitle}</p></div></div>', unsafe_allow_html=True)


def main_card(label, prob, risk, message):
    """The big card for the day after tomorrow."""
    color = RISK_COLORS[risk]
    st.markdown(f'<div class="card" style="border-color:{color}"><div class="label">{label}</div>'
                f'<div class="big" style="color:{color}">{prob:.0%}</div>'
                f'<span class="pill" style="background:{color}">{RISK_TEXT[risk]}</span>'
                f'<p style="margin:12px 0 0">{message}</p></div>', unsafe_allow_html=True)


def day_card(day, prob, risk, main=False):
    """One small card per day in the 7-day strip, with a bar that fills with the probability."""
    color = RISK_COLORS[risk]
    st.markdown(f'<div class="day{" main" if main else ""}" style="border-top-color:{color}">'
                f'<div class="name">{day:%a}</div><div class="date">{day:%d %b}</div>'
                f'<div class="pct" style="color:{color}">{prob:.0%}</div>'
                f'<div class="drop"><div style="width:{max(prob, 0.02):.0%};background:{color}"></div></div>'
                f'</div>', unsafe_allow_html=True)


def note(markdown_html):
    st.markdown(f'<div class="note">{markdown_html}</div>', unsafe_allow_html=True)
