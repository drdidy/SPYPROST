"""SPY Prophet design system.

Visual-only module: design tokens, Plotly chart template, and Streamlit
helper functions for rendering KPI cards, chart cards, status pills, and
empty states. No business logic lives here.
"""
from __future__ import annotations

from html import escape
from typing import Any, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

# Surfaces
BG_PAGE = "#0B1220"
BG_CARD = "#131C2E"
BG_CARD_HOVER = "#1A2540"
BG_ELEVATED = "#1E2A44"

# Borders
BORDER_SUBTLE = "rgba(255,255,255,0.06)"
BORDER_STRONG = "rgba(255,255,255,0.12)"

# Text
TEXT_PRIMARY = "#E8ECF4"
TEXT_SECONDARY = "#A0AEC8"
TEXT_MUTED = "#6B7B96"
TEXT_DISABLED = "#4A5772"

# Semantic
BULL = "#22C55E"
BULL_SOFT = "rgba(34,197,94,0.12)"
BEAR = "#EF4444"
BEAR_SOFT = "rgba(239,68,68,0.12)"
NEUTRAL = "#94A3B8"

# Brand accent
ACCENT = "#F5B642"
ACCENT_SOFT = "rgba(245,182,66,0.15)"

# Typography
FONT_UI = '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_MONO = '"JetBrains Mono", "Roboto Mono", Consolas, monospace'


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------

PROPHET_THEME: dict[str, Any] = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, system-ui, sans-serif",
            "size": 11,
            "color": TEXT_SECONDARY,
        },
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.05)",
            "linecolor": "rgba(255,255,255,0.08)",
            "tickfont": {"size": 10, "color": TEXT_MUTED},
            "showspikes": True,
            "spikecolor": "rgba(245,182,66,0.4)",
            "spikethickness": 1,
            "spikedash": "dot",
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.05)",
            "linecolor": "rgba(255,255,255,0.08)",
            "tickfont": {"size": 10, "color": TEXT_MUTED},
        },
        "margin": {"l": 48, "r": 16, "t": 8, "b": 32},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": BG_ELEVATED,
            "bordercolor": BORDER_STRONG,
            "font": {
                "family": "JetBrains Mono, monospace",
                "size": 11,
                "color": TEXT_PRIMARY,
            },
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11, "color": TEXT_SECONDARY},
            "bgcolor": "rgba(0,0,0,0)",
        },
    }
}


def apply_chart_theme(fig) -> Any:
    """Apply the Prophet Plotly template to a figure in place and return it."""
    fig.update_layout(**PROPHET_THEME["layout"])
    return fig


def register_plotly_default() -> None:
    """Register the Prophet template as Plotly's default so every figure
    in the app inherits its colors, fonts, gridlines, and hover styling
    without per-call changes. Idempotent.
    """
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
    except Exception:
        return
    template = go.layout.Template(layout=PROPHET_THEME["layout"])
    pio.templates["spy_prophet"] = template
    pio.templates.default = "spy_prophet"


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --bg-page: {BG_PAGE};
  --bg-card: {BG_CARD};
  --bg-card-hover: {BG_CARD_HOVER};
  --bg-elevated: {BG_ELEVATED};
  --border-subtle: {BORDER_SUBTLE};
  --border-strong: {BORDER_STRONG};
  --text-primary: {TEXT_PRIMARY};
  --text-secondary: {TEXT_SECONDARY};
  --text-muted: {TEXT_MUTED};
  --text-disabled: {TEXT_DISABLED};
  --bull: {BULL};
  --bull-soft: {BULL_SOFT};
  --bear: {BEAR};
  --bear-soft: {BEAR_SOFT};
  --neutral: {NEUTRAL};
  --accent: {ACCENT};
  --accent-soft: {ACCENT_SOFT};
}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background-color: var(--bg-page) !important;
  color: var(--text-primary);
  font-family: {FONT_UI};
  font-size: 15px;
  line-height: 1.55;
}}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
  color: var(--text-primary);
  font-family: {FONT_UI};
  font-weight: 600;
  letter-spacing: -0.01em;
}}
[data-testid="stMarkdownContainer"] h1 {{ font-size: 20px; line-height: 1.3; }}
[data-testid="stMarkdownContainer"] h2 {{ font-size: 18px; line-height: 1.3; }}
[data-testid="stMarkdownContainer"] h3 {{ font-size: 14px; line-height: 1.4; }}

[data-testid="stHeader"] {{
  background-color: var(--bg-page);
  border-bottom: 1px solid var(--border-subtle);
}}

[data-testid="stToolbar"] {{ background: transparent; }}

/* Sidebar */
[data-testid="stSidebar"] {{
  background-color: var(--bg-page) !important;
  border-right: 1px solid var(--border-subtle);
}}
[data-testid="stSidebar"] > div {{ background-color: var(--bg-page) !important; }}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}

/* Native metric → KPI-card aesthetic */
[data-testid="stMetric"] {{
  background-color: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 24px;
}}
[data-testid="stMetricLabel"] {{
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}
[data-testid="stMetricValue"] {{
  color: var(--text-primary);
  font-family: {FONT_UI};
  font-size: 24px;
  font-weight: 700;
  line-height: 1.2;
}}
[data-testid="stMetricDelta"] {{ font-size: 12px; }}

/* Plotly host */
.stPlotlyChart {{
  background: transparent;
  border-radius: 8px;
}}

/* Buttons */
.stButton > button {{
  background-color: var(--bg-card);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 8px 14px;
  font-family: {FONT_UI};
  font-size: 13px;
  font-weight: 500;
  transition: background-color 100ms ease-out, border-color 100ms ease-out;
}}
.stButton > button:hover {{
  background-color: var(--bg-card-hover);
  border-color: var(--border-strong);
}}
.stButton > button:focus {{
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--accent-soft);
}}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] > div {{
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  color: var(--text-primary) !important;
  border-radius: 8px !important;
  font-family: {FONT_UI};
  font-size: 13px;
}}

/* Tables — mono for tabular numerics */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  background-color: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
}}
[data-testid="stDataFrame"] [role="grid"],
[data-testid="stTable"] table {{
  font-family: {FONT_MONO};
  font-size: 12px;
  color: var(--text-primary);
}}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {{
  border-bottom: 1px solid var(--border-subtle);
  gap: 4px;
}}
[data-testid="stTabs"] [role="tab"] {{
  background: transparent;
  color: var(--text-muted);
  font-size: 13px;
  font-weight: 500;
  padding: 8px 12px;
  border-radius: 6px 6px 0 0;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
  color: var(--accent);
  border-bottom: 2px solid var(--accent);
}}

/* --- Custom components --- */

.kpi-card {{
  background-color: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.kpi-label {{
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}
.kpi-value {{
  color: var(--text-primary);
  font-size: 24px;
  font-weight: 700;
  line-height: 1.2;
  font-family: {FONT_UI};
}}
.kpi-value.bull {{ color: var(--bull); }}
.kpi-value.bear {{ color: var(--bear); }}
.kpi-value.accent {{ color: var(--accent); }}
.kpi-value.neutral {{ color: var(--neutral); }}
.kpi-sub {{
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 6px;
}}
.kpi-sub .kpi-arrow {{ font-size: 12px; }}
.kpi-sub.bull {{ color: var(--bull); }}
.kpi-sub.bear {{ color: var(--bear); }}

.chart-card {{
  background-color: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 24px;
}}
.chart-card .chart-header {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 16px;
}}
.chart-card .chart-header h3 {{
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}}
.chart-card .chart-meta {{
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}
.chart-card .chart-body {{ padding: 0; }}

.pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background-color: rgba(255,255,255,0.04);
  border: 1px solid var(--border-subtle);
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--text-secondary);
  font-family: {FONT_UI};
}}
.pill .pill-dot {{
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background-color: var(--neutral);
}}
.pill.live .pill-dot {{ background-color: var(--bull); }}
.pill.offline .pill-dot {{ background-color: var(--text-disabled); }}
.pill.warn .pill-dot {{ background-color: var(--accent); }}
.pill.error .pill-dot {{ background-color: var(--bear); }}

.empty-state {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 48px 24px;
  gap: 12px;
}}
.empty-state .empty-icon {{
  color: var(--text-muted);
  width: 32px;
  height: 32px;
}}
.empty-state .empty-title {{
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 600;
}}
.empty-state .empty-body {{
  color: var(--text-muted);
  font-size: 13px;
  max-width: 320px;
  line-height: 1.5;
}}

/* Header bar */
.app-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 56px;
  padding: 0 24px;
  border-bottom: 1px solid var(--border-subtle);
  background-color: var(--bg-page);
  margin-bottom: 16px;
}}
.app-header .app-title {{
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.01em;
}}
.app-header .app-meta {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.app-header .app-clock {{
  font-family: {FONT_MONO};
  font-size: 12px;
  color: var(--text-secondary);
}}

/* Sidebar nav */
.nav-group {{ margin-top: 16px; }}
.nav-group-label {{
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 8px 16px 4px;
}}
.nav-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-secondary);
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 100ms ease-out;
}}
.nav-item:hover {{ background-color: rgba(255,255,255,0.04); }}
.nav-item.active {{ background-color: var(--accent-soft); color: var(--accent); }}
.nav-item svg {{ width: 14px; height: 14px; flex-shrink: 0; }}

.brand-mark {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px;
}}
.brand-mark .brand-glyph {{ color: var(--accent); width: 18px; height: 18px; }}
.brand-mark .brand-word {{
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0.02em;
}}

/* Hide default Streamlit chrome we don't want */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* === TRADYTICS-STYLE SHELL OVERRIDES (last) === */

/* Brand block at top of sidebar */
.ds-brand{{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px 18px;
  margin: 0 0 8px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}}
.ds-brand-text{{display: flex; flex-direction: column; line-height: 1.2}}
.ds-brand-name{{
  color: #F4F7FB;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.06em;
}}
.ds-brand-tag{{
  color: #5A6479;
  font-size: 10px;
  font-weight: 500;
  margin-top: 2px;
  letter-spacing: 0.02em;
}}

/* Settings expander — tame the visual weight */
[data-testid="stSidebar"] [data-testid="stExpander"]{{
  background: transparent !important;
  border: none !important;
  margin: 12px 8px !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] details summary{{
  background: transparent !important;
  color: #5A6479 !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  padding: 6px 10px !important;
  border: none !important;
  border-radius: 6px !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
[data-testid="stSidebar"] [data-testid="stExpander"] details summary:hover{{
  background: rgba(255,255,255,0.03) !important;
  color: #A0AEC8 !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p{{
  color: #6B7B96 !important;
  font-size: 11px !important;
}}



/* Background: even darker, more cinematic */
html, body, .stApp, [data-testid="stAppViewContainer"]{{
  background: #0A0E14 !important;
}}
.main .block-container, .block-container{{
  padding-top: 1.75rem !important;
  padding-left: 40px !important;
  padding-right: 40px !important;
  max-width: 1520px !important;
}}

/* Larger labels app-wide */
[data-testid="stMetricLabel"], .ds-anal-l, .ds-l, .panel-hd, .tt-lbl, .pill-k, .stape-head, .tmap-head, .ds-nav-group{{
  font-size: 12px !important;
}}
[data-testid="stMetricValue"]{{ font-size: 28px !important; }}

/* Native widgets */
.stTextInput input, .stNumberInput input, .stDateInput input, [data-baseweb="select"] > div{{
  font-size: 14px !important;
  padding: 8px 12px !important;
}}
.stButton > button, .stDownloadButton > button{{
  font-size: 14px !important;
  padding: 9px 16px !important;
}}

/* KPI / card breathing */
.ds-anal-kpi{{ padding: 26px 28px !important; }}
.ds-anal-card{{ padding: 26px 28px !important; }}
.ds-anal-v{{ font-size: 42px !important; }}

/* Plotly text bump */
.stPlotlyChart text{{ font-size: 12px !important; }}

/* Sidebar — wider, darker, no border */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div{{
  width: 240px !important;
  min-width: 240px !important;
  background: #0A0E14 !important;
  border-right: 1px solid rgba(255,255,255,0.04) !important;
  padding-top: 8px !important;
}}
[data-testid="stSidebar"] [data-testid="stSidebarNav"]{{display:none !important}}

/* Sidebar nav group labels */
.ds-nav-group{{
  color: #5A6479;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  padding: 16px 16px 4px;
  margin-top: 4px;
}}
.ds-nav-wrap{{margin-bottom: 12px}}

/* Sidebar buttons styled as nav items — slim, flush, link-like */
[data-testid="stSidebar"] .stButton > button{{
  background: transparent !important;
  border: none !important;
  border-radius: 6px !important;
  color: #A0AEC8 !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 7px 14px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  margin: 0 8px !important;
  width: calc(100% - 16px) !important;
  box-shadow: none !important;
  height: auto !important;
  min-height: 0 !important;
  line-height: 1.3 !important;
  letter-spacing: 0.01em !important;
}}
[data-testid="stSidebar"] .stButton > button p{{
  font-size: 13px !important;
  font-weight: 500 !important;
  margin: 0 !important;
}}
[data-testid="stSidebar"] .stButton{{margin: 0 !important}}
[data-testid="stSidebar"] .stButton > button:hover{{
  background: rgba(255,255,255,0.04) !important;
  color: #E8ECF4 !important;
  border: none !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"]{{
  background: rgba(245,182,66,0.14) !important;
  color: #F5B642 !important;
  border: none !important;
  font-weight: 600 !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover{{
  background: rgba(245,182,66,0.20) !important;
  color: #F5B642 !important;
}}

/* Hide the now-empty Streamlit tab bar entirely */
[data-baseweb="tab-list"]{{display: none !important}}
[data-baseweb="tab-panel"]{{padding: 0 !important}}

/* Cards: nuke borders, use slightly lighter bg + breathing room */
[class*="-card"], [class*="-shell"], [class*="-plate"], [class*="-tile"]:not([class*="tile-label"]),
[class*="-mini"]:not([class*="mini-label"]):not([class*="mini-value"]):not([class*="mini-copy"]),
.terminal-hero, .brand-logo, .brief-card, .ai-verify-card, .citation-card,
.calendar-event, .data-notice, .decision-plate, .scenario-card,
.daily-guide-card, .source-card, .signal-card, .strike-card, .upgrade-card,
.flow-board-card, .ds-kpi, .ds-panel, .ds-hero{{
  background: #0F1623 !important;
  border: 1px solid rgba(255,255,255,0.04) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}}

/* Native metrics: also Tradytics-tuned */
[data-testid="stMetric"]{{
  background: #0F1623 !important;
  border: 1px solid rgba(255,255,255,0.04) !important;
  border-radius: 10px !important;
  padding: 20px 22px !important;
}}

/* Big numbers should be heavy and white */
.ds-hero-value, .hero-price, [data-testid="stMetricValue"]{{
  color: #F4F7FB !important;
  font-weight: 700 !important;
}}

/* Header bar — drop bottom border, subtler */
.app-header{{
  border-bottom: 1px solid rgba(255,255,255,0.04) !important;
  margin-bottom: 24px !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
}}

/* Refresh button in sidebar should match nav style */
[data-testid="stSidebar"] [data-testid="stButton-secondary"]{{
  background: rgba(255,255,255,0.03) !important;
  margin: 4px 16px 12px !important;
  width: calc(100% - 32px) !important;
}}

/* Tighten captions/labels */
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"]{{
  color: #5A6479 !important;
  font-size: 11px !important;
  padding: 2px 16px !important;
}}

/* Drop ALL border-radius >12px (Tradytics is squarish) */
[class*="radius"]{{border-radius: 10px !important}}

/* SPY price hero: bigger, more center stage */
.ds-hero{{padding: 40px 32px !important}}
.ds-hero-value{{font-size: 56px !important}}

/* KPI cards: bigger, more breathing */
.ds-kpi{{padding: 24px 22px !important; min-height: 110px}}
.ds-kpi .ds-v{{font-size: 26px !important}}

</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet. Call once at the top of `app.py`."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper components
# ---------------------------------------------------------------------------

_TONE_MAP = {
    "bull": "bull",
    "long": "bull",
    "favorable": "bull",
    "bear": "bear",
    "short": "bear",
    "adverse": "bear",
    "accent": "accent",
    "active": "accent",
    "neutral": "neutral",
    None: "",
    "": "",
}


def _tone_class(tone: Optional[str]) -> str:
    if tone is None:
        return ""
    return _TONE_MAP.get(tone.lower(), "")


def kpi_card(
    label: str,
    value: str,
    sub: Optional[str] = None,
    tone: Optional[str] = None,
) -> None:
    """Render a KPI card. `tone` is one of: bull, bear, accent, neutral."""
    tone_cls = _tone_class(tone)
    sub_html = ""
    if sub:
        arrow = ""
        if tone_cls == "bull":
            arrow = '<span class="kpi-arrow">↑</span>'
        elif tone_cls == "bear":
            arrow = '<span class="kpi-arrow">↓</span>'
        sub_html = (
            f'<div class="kpi-sub {tone_cls}">{arrow}'
            f'<span>{escape(sub)}</span></div>'
        )
    html = (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value {tone_cls}">{escape(value)}</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def chart_card(title: str, meta: str, fig) -> None:
    """Render a chart inside a styled card with header + meta."""
    apply_chart_theme(fig)
    header = (
        f'<div class="chart-card" style="padding-bottom: 8px;">'
        f'<div class="chart-header">'
        f'<h3>{escape(title)}</h3>'
        f'<div class="chart-meta">{escape(meta)}</div>'
        f'</div></div>'
    )
    st.markdown(header, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def status_pill(label: str, state: str = "neutral") -> str:
    """Return HTML for a status pill. `state`: live, offline, warn, error, neutral."""
    state_cls = state.lower() if state else "neutral"
    if state_cls not in {"live", "offline", "warn", "error", "neutral"}:
        state_cls = "neutral"
    return (
        f'<span class="pill {state_cls}">'
        f'<span class="pill-dot"></span>{escape(label)}'
        f'</span>'
    )


def render_app_header(title: str, pill_label: str, pill_state: str = "neutral",
                      clock_text: Optional[str] = None) -> None:
    """Render the top header bar: page title, status pill, CT clock.

    The clock is rendered server-side (Streamlit strips inline <script>
    tags from st.markdown), so it shows the time at last rerun rather
    than ticking every second. Pass `clock_text` formatted by the caller.
    """
    pill_html = status_pill(pill_label, pill_state)
    clock_display = clock_text or "—"
    html = f"""
<div class="app-header">
  <div class="app-title">{escape(title)}</div>
  <div class="app-meta">
    {pill_html}
    <span class="app-clock">{escape(clock_display)}</span>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str) -> None:
    """Render an empty/unconfigured-state card. `icon` is an inline SVG string."""
    html = (
        f'<div class="chart-card"><div class="empty-state">'
        f'<div class="empty-icon">{icon}</div>'
        f'<div class="empty-title">{escape(title)}</div>'
        f'<div class="empty-body">{escape(body)}</div>'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# Lucide-style inline SVG icons for use with empty_state() and nav.
ICON_TREND = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" width="100%" height="100%">'
    '<polyline points="3 17 9 11 13 15 21 7"/>'
    '<polyline points="14 7 21 7 21 14"/></svg>'
)
ICON_LOCK = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" width="100%" height="100%">'
    '<rect x="3" y="11" width="18" height="11" rx="2"/>'
    '<path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>'
)
ICON_INFO = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" width="100%" height="100%">'
    '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/>'
    '<path d="M12 8h.01"/></svg>'
)
