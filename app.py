"""
PRAGYAM (प्रज्ञम) — Portfolio Intelligence  |  A @thebullishvalue Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conviction-based portfolio curation with 80+ quantitative strategies.

Architecture:
  regime.py         → MarketRegimeDetector, compute_conviction_signals
  portfolio.py      → compute_conviction_based_weights()
  backdata.py       → generate_historical_data()
  intelligence.py   → conviction-weight calibration (passports)
  analytics.py      → portfolio-vs-benchmark performance metrics (Analytics tab)
  charts.py         → Plotly chart builders
  strategies.py     → 80+ BaseStrategy implementations

Conviction blend: 4 signals — RSI · Oscillator · Z-Score · MA-alignment. Weights
are regime-calibrated in Intelligence mode (4-simplex), the canonical
0.30 / 0.30 / 0.20 / 0.20 fallback otherwise.

Pipeline (2 phases):
  Phase 1: Data fetching + regime detection
  Phase 2: Conviction-based portfolio curation (ALL strategies)

Result tabs: Portfolio · Position Guide · Analytics (curated book vs benchmark) ·
Regime · Intelligence · Broker Sync (curated units → broker JSONs) · System.

Version: 8.1.0
Author: @thebullishvalue
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import warnings
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Tuple, Optional

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import html as html_module
import intelligence

# ── Imports ────────────────────────────────────────────────────────────────────
from logger_config import get_console
log = get_console()

from metrics import get_metrics
from ui.theme import inject_css, VERSION, PRODUCT_NAME, COMPANY, progress_bar
from ui.components import (
    render_header,
    render_section_header,
    render_metric_card,
    render_info_box,
    render_system_card,
    section_gap,
    render_warning_box,
    render_chart_skeleton,
    render_collapsible_section,
    render_collapsible_section_close,
    render_theme_toggle,
    render_export_button_row,
    render_interpretation_card,
    render_kv_table,
    get_icon,
)
import streamlit.components.v1 as components
from regime import (
    MarketRegimeDetector,
    REGIME_COLORS,
    REGIME_ICONS,
    REGIME_DESCRIPTIONS,
    get_regime_history_series,
    compute_conviction_signals,
)
from strategies import discover_strategies
from backdata import (
    generate_historical_data,
    get_default_universe,
    MAX_INDICATOR_PERIOD,
)
from universe import (
    resolve_universe,
    render_universe_selector,
    UNIVERSE_OPTIONS,
)
from portfolio import compute_conviction_based_weights

try:
    from charts import (
        COLORS,
        create_conviction_heatmap,
        create_regime_history_chart,
    )
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    COLORS = {
        "primary": "#FFC300",
        "success": "#10b981",
        "danger": "#ef4444",
        "warning": "#f59e0b",
        "info": "#06b6d4",
        "muted": "#888888",
        "card": "#1A1A1A",
        "border": "#2A2A2A",
        "text": "#EAEAEA",
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "v8.1.0"
PRODUCT_NAME = "Pragyam"
COMPANY = "@thebullishvalue"

st.set_page_config(
    page_title="PRAGYAM | Portfolio Intelligence",
    page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Obsidian Quant Terminal CSS
inject_css()

# Render theme toggle (dark/light mode)
render_theme_toggle()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_session_state():
    """Initialize session state with defaults."""
    defaults = {
        "portfolio": None,
        "current_df": None,
        "selected_date": None,
        "regime_result_dict": None,
        "training_data_window": None,
        "regime_history_series": None,
        "min_pos_pct": 0.01,
        "max_pos_pct": 0.10,
        "intelligence_mode": True,  # Use calibrated weights when a passport exists; falls back to defaults otherwise.
        "selected_universe": None,
        "selected_index": None,
        # Last intelligence outcome from Phase 1.5 — read by sidebar/result UI.
        # Shape: {"status": "reused"|"calibrated"|"skipped"|"failed", "reason": str,
        #         "universe": str, "index": Optional[str], "regime": str,
        #         "train_ir": float, "val_ir": float}
        "last_intel_outcome": None,
        "debug_info": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _intel_context() -> Tuple[str, Optional[str], str, str]:
    """(universe, selected_index, regime_name, mode) read from session state.

    Used by every site that constructs a passport or asks for active weights,
    so the four conviction weights are always sourced from the right key.
    """
    rd = st.session_state.get("regime_result_dict", {}) or {}
    universe = st.session_state.get("selected_universe") or "default"
    selected_index = st.session_state.get("selected_index")
    # "UNKNOWN" matches the regime-detector's own failure sentinel (see
    # _detect_regime_cached and MarketRegimeDetector.detect), so a passport lookup
    # before regime detection completes and a lookup after a failed detection
    # route to the same scope rather than two different ones.
    regime_name = rd.get("regime", "UNKNOWN")
    mode = "Intelligence" if st.session_state.get("intelligence_mode") else "Standard"
    return universe, selected_index, regime_name, mode


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _load_historical_data(end_date: datetime, lookback_files: int, symbols_key: str) -> List[Tuple[datetime, pd.DataFrame]]:
    """Fetch and cache historical indicator snapshots from yfinance."""
    # Resolve symbols from the cache key
    try:
        if symbols_key.startswith("UNIVERSE:"):
            universe_name, index = symbols_key.replace("UNIVERSE:", "", 1).split("|", 1)
            index = index if index != "None" else None
            symbols_list, _ = resolve_universe(universe_name, index)
        else:
            symbols_list = get_default_universe()
        
        if not symbols_list:
            raise ValueError("No symbols found in the selected universe.")
    except Exception as e:
        st.error(f"Error resolving universe: {e}")
        return []
    
    try:
        return generate_historical_data(
            symbols_to_process=symbols_list,
            start_date=end_date - timedelta(days=int((lookback_files + MAX_INDICATOR_PERIOD) * 1.5) + 30),
            end_date=end_date,
        )
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return []


# Single LOOKBACK used by both the regime detection cache and the main-flow
# fetch — so the regime card, the Phase 2 curation, and the Regime Score
# History chart all reason about the same historical panel.
_REGIME_LOOKBACK_FILES = 100


@st.cache_data(ttl=3600, show_spinner=False)
def _detect_regime_cached(end_date: datetime, symbols_key: str) -> Dict:
    """Detect market regime from the SAME cached panel the main flow uses.

    Reads `_load_historical_data(end_date, _REGIME_LOOKBACK_FILES, symbols_key)`
    so the trailing 10 days the detector consumes are identical to the trailing
    10 days the Regime Score History chart's last bucket consumes. This makes
    the sidebar card, the result page's regime banner, and the chart's last
    bar three views of the SAME computation on the SAME data.
    """
    try:
        hist = _load_historical_data(end_date, _REGIME_LOOKBACK_FILES, symbols_key)
    except Exception as e:
        return {
            "regime": "UNKNOWN",
            "mix_name": "Chop/Consolidate Mix",
            "confidence": 0.30,
            "composite_score": 0.0,
            "explanation": f"Data fetch error: {e}",
            "color": "#6b7280",
            "icon": "help-circle",
            "description": "",
        }

    if not hist or len(hist) < 5:
        return {
            "regime": "UNKNOWN",
            "mix_name": "Chop/Consolidate Mix",
            "confidence": 0.30,
            "composite_score": 0.0,
            "explanation": "Insufficient data for regime classification.",
            "color": REGIME_COLORS["UNKNOWN"],
            "icon": "help-circle",
            "description": "",
        }

    try:
        result = MarketRegimeDetector().detect(hist, analysis_date=end_date)
        return result.to_dict()
    except Exception as e:
        return {
            "regime": "UNKNOWN",
            "mix_name": "Chop/Consolidate Mix",
            "confidence": 0.30,
            "composite_score": 0.0,
            "explanation": f"Regime detection error: {e}",
            "color": "#6b7280",
            "icon": "help-circle",
            "description": "",
        }


@st.cache_data(ttl=1800, show_spinner=False)
def _analytics_series_cached(
    symbols: Tuple[str, ...], units: Tuple[float, ...],
    anchor_iso: str, days_back: int,
    bench_ticker: str, bench_name: str,
):
    """Cached wrapper around analytics.build_return_series.

    Keyed on the exact (symbols, units, anchor, benchmark) tuple so the yfinance
    fetch runs ONCE per unique window and every subsequent render/tab-switch hits
    cache — no repeated downloads. Returns (port_value, port_returns,
    bench_returns, err). Compute stays in analytics.py; caching lives here (the
    Streamlit boundary), mirroring _load_historical_data / _detect_regime_cached.
    """
    from analytics import build_return_series
    _port = pd.DataFrame({"symbol": list(symbols), "units": list(units)})
    anchor_dt = datetime.fromisoformat(anchor_iso)
    return build_return_series(_port, days_back, bench_ticker, bench_name, anchor_date=anchor_dt)


# ══════════════════════════════════════════════════════════════════════════════
# UI PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _section_header(title: str, subtitle: str = "") -> str:
    """Generate styled section header HTML."""
    sub = f"<p class='section-subtitle'>{subtitle}</p>" if subtitle else ""
    return f"<div class='section'><div class='section-header'><h3 class='section-title'>{title}</h3>{sub}</div></div>"


def _section_divider():
    """Render section divider."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_portfolio_tab(portfolio: pd.DataFrame, current_df: pd.DataFrame, capital: float):
    """Tab 1 — Curated portfolio with conviction signal overlay."""
    render_section_header(
        "Curated Portfolio Holdings",
        f"{len(portfolio)} positions · conviction-based curation",
        icon="briefcase",
        accent="amber",
    )

    universe, selected_index, regime_name, mode = _intel_context()
    portfolio_with_signals = compute_conviction_signals(
        portfolio, current_df,
        universe=universe, selected_index=selected_index,
        regime_name=regime_name, mode=mode,
    )

    # Portfolio table — Custom HTML with inline CSS via st_html
    table_rows = []
    for _, row in portfolio.iterrows():
        symbol_escaped = html_module.escape(row["symbol"])
        table_rows.append(
            f'<tr>'
            f'<td class="col-port-symbol symbol">{symbol_escaped}</td>'
            f'<td class="col-port-units numeric">{row["units"]:,.0f}</td>'
            f'<td class="col-port-price numeric currency">&#8377;{row["price"]:,.2f}</td>'
            f'<td class="col-port-weight numeric percentage">{row["weightage_pct"]:.2f}%</td>'
            f'<td class="col-port-value numeric currency">&#8377;{row["value"]:,.2f}</td>'
            f'</tr>'
        )

    # Full HTML with inline CSS for iframe rendering
    table_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'IBM Plex Mono', monospace; 
            background: transparent;
            color: #F1F5F9;
            padding: 0.5rem;
        }}
        .portfolio-table {{
            width: 100%;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.45) 0%, rgba(17, 24, 39, 0.4) 100%);
        }}
        .portfolio-table table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}
        .portfolio-table thead th {{
            background: linear-gradient(180deg, rgba(10, 14, 23, 0.95) 0%, rgba(10, 14, 23, 0.85) 100%);
            color: #4B5563;
            font-size: 0.62rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.75rem 1rem;
            border-bottom: 2px solid rgba(212, 168, 83, 0.3);
            text-align: left;
        }}
        .portfolio-table thead th.numeric {{ text-align: right; }}
        .portfolio-table tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            transition: background 0.2s ease;
        }}
        .portfolio-table tbody tr:nth-child(odd) {{ background: rgba(255, 255, 255, 0.01); }}
        .portfolio-table tbody tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.005); }}
        .portfolio-table tbody tr:hover {{ background: rgba(212, 168, 83, 0.05); }}
        .portfolio-table tbody td {{
            padding: 0.75rem 1rem;
            color: #F1F5F9;
        }}
        .portfolio-table tbody td.symbol {{
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .portfolio-table tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .col-port-symbol {{ width: 25%; }}
        .col-port-price {{ width: 18%; }}
        .col-port-units {{ width: 14%; }}
        .col-port-weight {{ width: 16%; }}
        .col-port-value {{ width: 27%; }}
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th class="numeric">Units</th>
                    <th class="numeric">Price (&#8377;)</th>
                    <th class="numeric">Weight %</th>
                    <th class="numeric">Value (&#8377;)</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    '''

    table_height = max(280, 220 + len(portfolio) * 42)
    components.html(table_html, height=table_height)

    # Conviction Signal Heatmap
    _section_divider()
    render_section_header(
        "Conviction Signals",
        "Real-time indicator alignment — RSI · Oscillator · Z-Score · MA Alignment",
        icon="activity",
        accent="cyan",
    )

    if CHARTS_AVAILABLE and not portfolio_with_signals.empty:
        st.markdown('<div class="chart-container portfolio">', unsafe_allow_html=True)
        fig_conv = create_conviction_heatmap(portfolio_with_signals)
        st.plotly_chart(fig_conv, width='stretch', key="tab1_conviction_heatmap")
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(
            "Green = bullish · Red = bearish · RSI (30%) · Oscillator (30%) · Z-Score (20%) · MA (20%)"
        )
    elif not portfolio_with_signals.empty:
        conv_cols = [c for c in ["symbol", "rsi_value", "osc_value", "zscore_value", "ma_count", "conviction_score"]
                     if c in portfolio_with_signals.columns]
        st.dataframe(portfolio_with_signals[conv_cols], width='stretch')
    else:
        st.info("Conviction signals unavailable.")

    _section_divider()

    # CSV Download
    first_cols = ["symbol", "price", "units"]
    other_cols = [c for c in portfolio.columns if c not in first_cols]
    download_df = portfolio[first_cols + other_cols]
    buf = io.BytesIO()
    download_df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download Portfolio CSV",
        data=buf.getvalue(),
        file_name=f"pragyam_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        width='stretch',
        key="tab1_csv_download",
    )


def _conviction_label(score) -> Tuple[str, str]:
    """Classify a conviction score (0–100) → (css_class, label).

    Single source of truth for both the Signal Distribution cards and the
    per-row Signal pill in the Position Guide. Whatever value type the score
    is (int, float, np.int64), it's coerced to float once so the thresholds
    apply uniformly: a stored 65.0 lands as Strong Buy, never as Buy.
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        s = 50.0
    if not (s == s):  # NaN check; falls back to the same default as a missing value
        s = 50.0
    if s >= 65.0:
        return "strong-buy", "Strong Buy"
    if s >= 50.0:
        return "buy", "Buy"
    if s >= 35.0:
        return "hold", "Hold"
    return "caution", "Caution"


def _render_position_guide_tab(portfolio: pd.DataFrame, current_df: pd.DataFrame):
    """Tab — Position Guide with entry conditions and conviction signals."""
    universe, selected_index, regime_name, mode = _intel_context()
    portfolio_with_signals = compute_conviction_signals(
        portfolio, current_df,
        universe=universe, selected_index=selected_index,
        regime_name=regime_name, mode=mode,
    )

    if "rsi_signal" not in portfolio_with_signals.columns:
        st.info("Position guide signals unavailable.")
        return

    # ── Signal Distribution ──────────────────────────────────────────────
    render_section_header("Signal Distribution", "Portfolio conviction breakdown", icon="target")

    c1, c2, c3, c4 = st.columns(4)

    # Single classification pass; both cards and per-row pills use the same labels.
    labels = [
        _conviction_label(row.get("conviction_score", 50))[1]
        for _, row in portfolio_with_signals.iterrows()
    ]
    strong_buy = sum(1 for L in labels if L == "Strong Buy")
    buy        = sum(1 for L in labels if L == "Buy")
    hold       = sum(1 for L in labels if L == "Hold")
    caution    = sum(1 for L in labels if L == "Caution")

    with c1:
        render_metric_card("Strong Buy", str(strong_buy), "High conviction (≥65)", "success")
    with c2:
        render_metric_card("Buy", str(buy), "Moderate conviction (50-64)", "info")
    with c3:
        render_metric_card("Hold", str(hold), "Neutral (35-49)", "warning")
    with c4:
        render_metric_card("Caution", str(caution), "Low conviction (<35)", "danger")

    _section_divider()

    # ── Position Guide Table ─────────────────────────────────────────────
    render_section_header(
        "Position Guide",
        "Entry conditions and conviction summary for all holdings",
        icon="crosshair",
    )

    if portfolio_with_signals.empty:
        st.info("No position guide data available.")
        return

    # Sort by conviction score descending
    sorted_df = portfolio_with_signals.sort_values('conviction_score', ascending=False).reset_index(drop=True)

    # SVG icons per signal class — visual identity at a glance.
    _SIGNAL_ICONS = {
        "strong-buy": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        "buy":        '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="10"/></svg>',
        "hold":       '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="10"/></svg>',
        "caution":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    }

    table_rows = []
    for _, row in sorted_df.iterrows():
        symbol_escaped = html_module.escape(row["symbol"])
        # Coerce once, classify once — same helper as the cards above.
        raw_conv = row.get("conviction_score", 50)
        signal_class, signal_text = _conviction_label(raw_conv)
        signal_icon = _SIGNAL_ICONS[signal_class]
        try:
            conviction_display = round(float(raw_conv))
        except (TypeError, ValueError):
            conviction_display = 50

        table_rows.append(
            f'<tr>'
            f'<td class="col-symbol symbol">{symbol_escaped}</td>'
            f'<td class="col-price numeric currency">&#8377;{row["price"]:,.2f}</td>'
            f'<td class="col-signal"><span class="signal-pill {signal_class}"><span class="signal-icon">{signal_icon}</span>{signal_text}</span></td>'
            f'<td class="col-conviction numeric conviction-score">{conviction_display}</td>'
            f'<td class="col-rsi numeric">{row["rsi_signal"]:+.1f}</td>'
            f'<td class="col-osc numeric">{row["osc_signal"]:+.1f}</td>'
            f'<td class="col-z numeric">{row["zscore_signal"]:+.2f}</td>'
            f'<td class="col-ma numeric">{row["ma_signal"]:+.1f}</td>'
            f'<td class="col-weight numeric">{row["weightage_pct"]:.2f}%</td>'
            f'</tr>'
        )

    table_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: transparent;
            color: #F1F5F9;
            padding: 0.5rem 0.5rem 1.5rem 0.5rem;
        }}
        .table-layout {{
            table-layout: fixed;
            width: 100%;
        }}
        .col-symbol {{ width: 15%; }}
        .col-price {{ width: 12%; }}
        .col-signal {{ width: 18%; }}
        .col-conviction {{ width: 10%; }}
        .col-rsi {{ width: 10%; }}
        .col-osc {{ width: 10%; }}
        .col-z {{ width: 10%; }}
        .col-ma {{ width: 10%; }}
        .col-weight {{ width: 15%; }}
        .portfolio-table {{
            width: 100%;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.45) 0%, rgba(17, 24, 39, 0.4) 100%);
        }}
        .portfolio-table table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}
        .portfolio-table thead th {{
            background: linear-gradient(180deg, rgba(10, 14, 23, 0.95) 0%, rgba(10, 14, 23, 0.85) 100%);
            color: #4B5563;
            font-size: 0.62rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.75rem 0.75rem;
            border-bottom: 2px solid rgba(212, 168, 83, 0.3);
            text-align: left;
            white-space: nowrap;
        }}
        .portfolio-table thead th.numeric {{ text-align: right; }}
        .portfolio-table tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            transition: background 0.2s ease;
        }}
        .portfolio-table tbody tr:nth-child(odd) {{ background: rgba(255, 255, 255, 0.01); }}
        .portfolio-table tbody tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.005); }}
        .portfolio-table tbody tr:hover {{ background: rgba(212, 168, 83, 0.05); }}
        .portfolio-table tbody td {{
            padding: 0.75rem 0.75rem;
            color: #F1F5F9;
            vertical-align: middle;
            font-size: 0.75rem;
        }}
        .portfolio-table tbody td.symbol {{
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .portfolio-table tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .conviction-score {{ font-weight: 700; color: #F59E0B; }}
        .signal-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.75rem;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 600;
        }}
        .signal-pill .signal-icon {{
            display: inline-flex;
            align-items: center;
            flex-shrink: 0;
        }}
        .signal-pill.strong-buy {{
            background: rgba(16, 185, 129, 0.18);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.55);
            font-weight: 700;
        }}
        .signal-pill.buy {{
            background: rgba(16, 185, 129, 0.06);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}
        .signal-pill.hold {{
            background: rgba(245, 158, 11, 0.06);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.2);
        }}
        .signal-pill.caution, .signal-pill.sell {{
            background: rgba(239, 68, 68, 0.06);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }}
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th class="col-symbol">Symbol</th>
                    <th class="col-price numeric">Price</th>
                    <th class="col-signal">Signal</th>
                    <th class="col-conviction numeric">Conviction</th>
                    <th class="col-rsi numeric">RSI</th>
                    <th class="col-osc numeric">Osc</th>
                    <th class="col-z numeric">Z</th>
                    <th class="col-ma numeric">MA</th>
                    <th class="col-weight numeric">Weight</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    '''

    # Each row in this table contains a pill (extra vertical padding) plus a
    # numeric grid, so it's taller than the Portfolio tab's plain row. Budget
    # ~50px per row + a 60px tail so the last row + body padding clears the
    # iframe edge; otherwise the bottom rows get clipped.
    table_height = max(320, 240 + len(sorted_df) * 50 + 60)
    components.html(table_html, height=table_height)


def _render_regime_tab(regime_result: Dict, regime_series: List, training_data: Optional[List] = None):
    """Tab 2 — Market regime analysis."""
    if not regime_result:
        st.info("Run analysis to populate regime data.")
        return

    regime_name = regime_result.get("regime", "UNKNOWN")
    mix_name = regime_result.get("mix_name", "—")
    confidence = regime_result.get("confidence", 0.0)
    score = regime_result.get("composite_score", 0.0)
    color = regime_result.get("color", "#888888")
    icon_key = regime_result.get("icon", "help-circle")
    description = regime_result.get("description", "")
    explanation = regime_result.get("explanation", "")
    factors_raw = regime_result.get("factors", {})

    # Current Regime Banner
    render_section_header("Current Market Regime", "10-day indicator window", icon="eye")

    # NOTE: the badge + factor scores are rendered as ONE self-contained HTML flex
    # row (not st.columns), so vertical centring is under our control — Streamlit's
    # column wrappers made the badge impossible to centre reliably. The flex row's
    # `align-items:center` centres the badge card against the factor list, period.

    # Fixed factor weights (regime detector is not calibrated); labelled so the
    # percentages match what the composite actually used (see RegimeResult).
    _fw = {
        "momentum": 0.30, "trend": 0.25, "breadth": 0.15,
        "velocity": 0.15, "extremes": 0.10, "volatility": 0.05, "correlation": 0.00,
    }
    factor_order = [
        ("momentum", "Momentum", "strength"),
        ("trend", "Trend", "quality"),
        ("breadth", "Breadth", "quality"),
        ("velocity", "Velocity", "acceleration"),
        ("extremes", "Extremes", "type"),
        ("volatility", "Volatility", "regime"),
        ("correlation", "Correlation", "regime"),
    ]
    # Each factor score is a SIGNED value in [-2, +2] (bearish ↔ bullish), rendered
    # as a CENTER-ANCHORED diverging bar: a zero line in the middle, the fill
    # growing RIGHT (emerald) for a positive score or LEFT (rose) for a negative
    # one, with magnitude = |score| / 2. A 0→100% fill would misread a signed value.
    _rows = []
    for fkey, fbase, label_key in factor_order:
        fd = factors_raw.get(fkey, {})
        fs = float(fd.get("score", 0.0))
        fl = fd.get(label_key, "—")
        _wpct = _fw.get(fkey)
        fname = f"{fbase} ({_wpct*100:.0f}%)" if _wpct is not None else fbase
        half = min(50.0, abs(fs) / 2.0 * 50.0)
        if fs > 0.05:
            val_color = "var(--emerald)"
            fill = (f'<div style="position:absolute; left:50%; top:0; bottom:0; '
                    f'width:{half}%; background:var(--emerald); border-radius:0 3px 3px 0;"></div>')
        elif fs < -0.05:
            val_color = "var(--rose)"
            fill = (f'<div style="position:absolute; right:50%; top:0; bottom:0; '
                    f'width:{half}%; background:var(--rose); border-radius:3px 0 0 3px;"></div>')
        else:
            val_color = "var(--ink-tertiary)"
            fill = ""
        _rows.append(
            f'<div style="margin:0 0 12px 0;">'
            f'<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">'
            f'<span style="color:var(--ink-primary); font-weight:600;">{fname}</span>'
            f'<span style="color:var(--ink-tertiary);">{fl} '
            f'<span style="color:{val_color}; font-weight:700;">({fs:+.1f})</span></span>'
            f'</div>'
            f'<div style="position:relative; height:8px; background:var(--bg-elevated); border-radius:3px; overflow:hidden;">'
            f'{fill}'
            f'<div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--border-active); transform:translateX(-0.5px);"></div>'
            f'</div>'
            f'</div>'
        )
    _factors_html = "".join(_rows)
    _badge_icon = get_icon(icon_key, size=40, stroke_width=1.5)
    _fs_icon = get_icon("activity", size=16, stroke_width=1.8)

    # ── ONE flex row: factor scores (left, flex:1.7) + regime badge (right, flex:1).
    #    align-items:stretch makes both columns equal-height; the badge card is
    #    height:100% so it runs FLUSH top-and-bottom with the factor list.
    st.markdown(
        f'<div style="display:flex; align-items:stretch; gap:24px; margin-top:8px;">'
        # LEFT — factor scores (flex:1.7)
        f'<div style="flex:1.7; min-width:0;">'
        f'<div style="display:flex; align-items:center; gap:8px; margin:0 0 4px 0;">'
        f'<span style="color:var(--cyan, #6CD3D7); display:inline-flex;">{_fs_icon}</span>'
        f'<span style="font-family:var(--display); font-size:0.95rem; font-weight:700; '
        f'text-transform:uppercase; letter-spacing:0.06em; color:var(--ink-primary);">Factor Scores</span>'
        f'</div>'
        f'<div style="font-family:var(--data); font-size:0.7rem; color:var(--ink-tertiary); margin:0 0 12px 0;">'
        f'Signed composite inputs · −2 bearish ↔ +2 bullish</div>'
        f'<div style="display:flex; justify-content:space-between; font-family:var(--data); '
        f'font-size:0.62rem; letter-spacing:0.08em; color:var(--ink-tertiary); '
        f'text-transform:uppercase; margin:0 0 8px 0;">'
        f'<span>−2 Bearish</span><span>0 Neutral</span><span>+2 Bullish</span></div>'
        f'{_factors_html}'
        f'</div>'
        # RIGHT — regime badge card (flex:1), flush to the factor list height
        f'<div style="flex:1; display:flex; min-width:0;">'
        f'<div class="regime-badge" style="width:100%; height:100%; border-color:{color}66; '
        f'background:linear-gradient(160deg, {color}12 0%, {color}05 45%, transparent 100%), var(--glass);">'
        f'<div class="regime-icon">{_badge_icon}</div>'
        f'<div class="regime-name" style="color:{color}; font-size:2.1rem;">{regime_name.replace("_", " ")}</div>'
        f'<div class="regime-sub">{mix_name}</div>'
        f'<div class="regime-score">Score: {score:+.2f}</div>'
        f'<div class="regime-conf">'
        f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{confidence*100:.0f}%; background:{color};"></div></div>'
        f'<span style="color:{color}; font-size:1.05rem; font-weight:700;">{confidence:.0%} confidence</span>'
        f'</div></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Full-width METHOD card (Obsidian Quant fidelity) — mirrors the
    #    Intelligence tab's method card: header + pill + lede + tile grid.
    _rmode_lbl = "Intelligence" if st.session_state.get("intelligence_mode") else "Standard"
    method_html = (
        '<div class="intel-method-card">'
            '<div class="intel-method-header">'
                '<div class="intel-method-title">How the Regime Is Read</div>'
                '<div class="intel-method-pill">'
                f'7-factor composite · {_rmode_lbl} run'
                '</div>'
            '</div>'
            '<div class="intel-method-lede">'
                'The market regime is a <strong>weighted composite</strong> of seven measured '
                'factors, each scored on a signed <code>[-2, +2]</code> scale (bearish ↔ bullish). '
                'The weighted sum places the market on the regime hierarchy from '
                '<strong>Strong Bull</strong> down to <strong>Crisis</strong>. The factor weights '
                'are fixed (Momentum 30% · Trend 25% · Breadth/Velocity 15% · Extremes 10% · '
                'Volatility 5%); the regime detector itself is not calibrated.'
            '</div>'
            '<div class="intel-method-grid">'

                '<div class="intel-method-tile tile-learns">'
                    '<div class="tile-label">Momentum &amp; Trend</div>'
                    '<div class="tile-body">'
                        'RSI trajectory, oscillator direction, price/MA alignment and the share of '
                        'names above their 200-DMA — the primary directional drivers (largest weights).'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-how">'
                    '<div class="tile-label">Breadth &amp; Velocity</div>'
                    '<div class="tile-body">'
                        'Cross-sectional participation and the first/second derivative of momentum '
                        '(is the move accelerating or decaying) — confirmation and turning-point cues.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-obj">'
                    '<div class="tile-label">Extremes &amp; Volatility</div>'
                    '<div class="tile-body">'
                        'Z-score crowding and the Bollinger band-width regime — stress and '
                        'mean-reversion context that modulates the directional read.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-safety">'
                    '<div class="tile-label">Reading the bars</div>'
                    '<div class="tile-body">'
                        'Each factor bar is <strong>centre-anchored</strong>: it grows right (green) '
                        'when bullish, left (red) when bearish, from a zero line — the magnitude is '
                        'how far the factor leans, not a simple fill.'
                    '</div>'
                '</div>'

            '</div>'
        '</div>'
    )
    st.markdown(method_html, unsafe_allow_html=True)

    # Regime History
    regime_series_to_use = regime_series
    if regime_series_to_use is None and training_data and len(training_data) >= 10:
        with st.spinner("Computing regime history…"):
            regime_series_to_use = get_regime_history_series(training_data, window_size=10, step=1)
        st.session_state.regime_history_series = regime_series_to_use

    if regime_series_to_use and len(regime_series_to_use) > 0:
        _section_divider()
        render_section_header("Regime Score History", "Rolling 10-day composite", icon="activity", accent="emerald")

        regimes_seq = [r.regime for r in regime_series_to_use]
        transitions = sum(1 for i in range(1, len(regimes_seq)) if regimes_seq[i] != regimes_seq[i-1])
        # The chart and the cards share the same underlying panel as the sidebar
        # regime card (see _detect_regime_cached + _load_historical_data), so the
        # last bar of the chart is the canonical regime by construction.
        last_regime = regimes_seq[-1] if regimes_seq else "—"
        prev_regime = regimes_seq[-2] if len(regimes_seq) > 1 else "—"

        if CHARTS_AVAILABLE:
            st.markdown('<div class="chart-container regime">', unsafe_allow_html=True)
            fig_rh = create_regime_history_chart(regime_series_to_use)
            st.plotly_chart(fig_rh, width='stretch', key="tab2_regime_history")
            st.markdown('</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        # Map regime names to semantic metric card colors
        def regime_color(regime: str) -> str:
            r = regime.upper().replace("-", "_")
            if r in ("STRONG_BULL", "BULL", "WEAK_BULL"):
                return "success"
            elif r in ("BEAR", "CRISIS"):
                return "danger"
            elif r == "WEAK_BEAR":
                return "warning"
            elif r in ("CHOP", "UNKNOWN"):
                return "info"
            return "neutral"

        with c1:
            render_metric_card("Transitions", str(transitions), "Over analysis window", "info")
        with c2:
            render_metric_card("Current", last_regime.replace("_", " "), "Latest", regime_color(last_regime))
        with c3:
            render_metric_card("Prior", prev_regime.replace("_", " "), "Previous", regime_color(prev_regime))


def _render_system_tab(training_window: List):
    """Tab — System configuration + methodology reference (Obsidian Quant)."""
    # ── Configuration — the run's settings as a clean KV readout ───────────────
    render_section_header("Configuration", "Run settings & data source", icon="settings", accent="cyan")
    details = {
        "Version": VERSION,
        "Curation Method": "Conviction-based (4-signal blend)",
        "Weight Formula": "(conviction / total) × 100",
        "Min Position": f"{st.session_state.min_pos_pct*100:.1f}%",
        "Max Position": f"{st.session_state.max_pos_pct*100:.1f}%",
        "Data Source": "yfinance (NSE)",
        "Lookback Period": f"{len(training_window)} days",
    }
    render_kv_table(details)

    _section_divider()

    # ── Methodology — full-width method card (mirrors Intelligence/Regime) ─────
    render_section_header("Methodology", "How a portfolio is curated", icon="target", accent="emerald")
    method_html = (
        '<div class="intel-method-card">'
            '<div class="intel-method-header">'
                '<div class="intel-method-title">Curation Pipeline</div>'
                '<div class="intel-method-pill">'
                'detect → score → select → weight'
                '</div>'
            '</div>'
            '<div class="intel-method-lede">'
                'Every symbol in the universe is scored 0–100 by a four-signal conviction blend, '
                'read against the market regime, then the highest-conviction names are curated into '
                'a bounded, dispersion-weighted book.'
            '</div>'
            '<div class="intel-method-grid">'

                '<div class="intel-method-tile tile-learns">'
                    '<div class="tile-label">Signals</div>'
                    '<div class="tile-body">'
                        'RSI · Oscillator · Z-Score · MA-alignment, each in '
                        '<code>[-2, +2]</code>. Blended by regime-calibrated weights, mapped to '
                        '<code>(raw + 2) / 4 × 100</code>.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-how">'
                    '<div class="tile-label">Regime</div>'
                    '<div class="tile-body">'
                        'A seven-factor composite places the market on the Strong Bull → Crisis '
                        'hierarchy and selects the conviction-weight passport used for scoring.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-obj">'
                    '<div class="tile-label">Selection</div>'
                    '<div class="tile-body">'
                        'Top N by conviction. No hard threshold — all symbols are eligible.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-safety">'
                    '<div class="tile-label">Weighting</div>'
                    '<div class="tile-body">'
                        'Style-aware dispersion (SIP / Swing) then '
                        '<code>weight = (conviction / total) × 100</code>, bounded to '
                        f'{st.session_state.min_pos_pct*100:.0f}%–{st.session_state.max_pos_pct*100:.0f}% per position.'
                    '</div>'
                '</div>'

            '</div>'
        '</div>'
    )
    st.markdown(method_html, unsafe_allow_html=True)


def _sync_broker_json(json_data, quantity_map: Dict[str, int]) -> Tuple[list, int]:
    """Map curated per-symbol units into a broker order-template JSON.

    Walks each instrument in the template, and where its
    ``instrument.tradingsymbol`` matches a curated holding, writes the holding's
    unit count into ``params.quantity``. Returns the mutated JSON and the number
    of instruments updated.
    """
    updated = 0
    for item in json_data:
        try:
            symbol = item.get("instrument", {}).get("tradingsymbol")
            if symbol and symbol in quantity_map and "params" in item:
                item["params"]["quantity"] = int(quantity_map[symbol])
                updated += 1
        except Exception:
            continue
    return json_data, updated


def _render_broker_sync_tab(portfolio: pd.DataFrame):
    """Tab — Broker JSON Sync: write curated units into broker order templates.

    Reads the LIVE curated portfolio (no CSV re-upload) and maps each holding's
    unit count onto the matching instrument's ``params.quantity`` in every
    uploaded broker template (e.g. Kite ETF.json), producing ready-to-import
    order files. The natural final step of the flow: curate → sync → execute.
    """
    import json as _json

    render_section_header(
        "Broker JSON Sync",
        "Write curated units into broker order templates · curate → sync → execute",
        icon="download",
        accent="cyan",
    )

    # Guard: nothing to sync until a portfolio has been curated.
    if portfolio is None or portfolio.empty or "symbol" not in portfolio.columns or "units" not in portfolio.columns:
        render_interpretation_card(
            title="NO CURATED PORTFOLIO",
            body=(
                "Run an analysis first — the sync uses the live curated portfolio "
                "directly, so there is nothing to map onto your broker templates yet."
            ),
            color="warning",
        )
        return

    # Build the symbol → units map from the live portfolio (units ≥ 0, integer).
    qty_map: Dict[str, int] = {
        str(sym): int(u)
        for sym, u in zip(portfolio["symbol"], portfolio["units"].fillna(0))
    }
    tradable = sum(1 for u in qty_map.values() if u > 0)

    # ── Balanced two-column layout, mirroring the Intelligence tab exactly ─────
    #    col1 = status card + template uploader
    #    col2 = results table + totals line + per-file downloads
    col1, col2 = st.columns([1, 1])

    # Process uploaded templates once, up front, so both columns read the same
    # deterministic result set (status card, results table, download buttons).
    json_files = st.session_state.get("broker_sync_json_uploader")
    results = []  # (fname, payload_or_None, count, error_or_None)
    if json_files:
        for j_file in json_files:
            try:
                j_file.seek(0)
                content = _json.load(j_file)
                updated_json, count = _sync_broker_json(content, qty_map)
                results.append((j_file.name, _json.dumps(updated_json, indent=4), count, None))
            except Exception as e:
                results.append((j_file.name, None, 0, str(e)))
    n_templates = len(results)
    ok = sum(1 for _, p, _, _ in results if p is not None)
    total_updated = sum(c for _, p, c, _ in results if p is not None)

    with col1:
        if n_templates == 0:
            render_interpretation_card(
                title="AWAITING TEMPLATES",
                body=(
                    f"Curated book holds <strong>{tradable}</strong> tradable holding(s). "
                    "Upload one or more broker order-template JSONs (e.g. Kite "
                    "<strong>ETF.json</strong>) to sync their quantities."
                ),
                color="info",
            )
        elif ok == 0:
            render_interpretation_card(
                title="NO FILES SYNCED",
                body=(
                    f"None of the <strong>{n_templates}</strong> uploaded template(s) could be "
                    "processed. Check that each is a valid broker order JSON."
                ),
                color="danger",
            )
        else:
            render_interpretation_card(
                title="READY TO EXPORT",
                body=(
                    f"Synced <strong>{ok}/{n_templates}</strong> template(s) · "
                    f"<strong>{total_updated}</strong> instrument(s) updated from "
                    f"<strong>{tradable}</strong> tradable holding(s). "
                    "Download the import-ready files on the right."
                ),
                color="success",
            )

        st.file_uploader(
            "Upload broker JSON templates",
            type=["json"],
            accept_multiple_files=True,
            help="Your original broker order files (e.g. ETF.json). Each instrument's "
                 "quantity is set from the curated units for its trading symbol.",
            key="broker_sync_json_uploader",
            label_visibility="collapsed",
        )

    with col2:
        if n_templates == 0:
            st.markdown(
                '<div class="intel-table-wrap"><table class="portfolio-table-2col">'
                '<colgroup><col style="width:52%;"><col style="width:23%;">'
                '<col style="width:25%;"></colgroup>'
                '<thead><tr><th class="col-iw-factor">Template</th>'
                '<th class="col-iw-long">Updated</th>'
                '<th class="col-iw-short">Status</th></tr></thead>'
                '<tbody><tr><td colspan="3" '
                'style="text-align:center; color:var(--ink-tertiary); '
                'font-family:var(--data); font-size:0.75rem; padding:var(--sp-6) var(--sp-3);">'
                'No templates uploaded</td>'
                '</tr></tbody></table></div>',
                unsafe_allow_html=True,
            )
        else:
            rows_html = []
            for fname, payload, count, err in results:
                if err is not None:
                    rows_html.append(
                        f'<tr>'
                        f'<td class="iw-label">{html_module.escape(fname)}</td>'
                        f'<td class="iw-long" style="color:var(--rose)">—</td>'
                        f'<td class="iw-short" style="color:var(--rose)">ERROR</td>'
                        f'</tr>'
                    )
                else:
                    c_color = "var(--emerald)" if count > 0 else "var(--ink-secondary)"
                    s_color = "var(--emerald)" if count > 0 else "var(--amber)"
                    s_text = "SYNCED" if count > 0 else "NO MATCH"
                    rows_html.append(
                        f'<tr>'
                        f'<td class="iw-label">{html_module.escape(fname)}</td>'
                        f'<td class="iw-long" style="color:{c_color}">{count}</td>'
                        f'<td class="iw-short" style="color:{s_color}">{s_text}</td>'
                        f'</tr>'
                    )
            st.markdown(
                f'''
                <div class="intel-table-wrap">
                    <table class="portfolio-table-2col">
                        <colgroup>
                            <col style="width:52%;">
                            <col style="width:23%;">
                            <col style="width:25%;">
                        </colgroup>
                        <thead>
                            <tr>
                                <th class="col-iw-factor">Template</th>
                                <th class="col-iw-long">Updated</th>
                                <th class="col-iw-short">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(rows_html)}
                        </tbody>
                    </table>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            for fname, payload, count, err in results:
                if err is None:
                    st.download_button(
                        label=f"Download updated {fname}",
                        data=payload,
                        file_name=f"updated_{fname}",
                        mime="application/json",
                        width="stretch",
                        key=f"broker_sync_dl_{fname}",
                    )

    # ── Full-width METHOD card (Obsidian Quant fidelity) — mirrors the
    #    Intelligence tab's method card: header + pill + lede + tile grid.
    method_html = (
        '<div class="intel-method-card">'
            '<div class="intel-method-header">'
                '<div class="intel-method-title">How the Sync Works</div>'
                '<div class="intel-method-pill">'
                'symbol → units → params.quantity'
                '</div>'
            '</div>'
            '<div class="intel-method-lede">'
                'The Broker Sync closes the loop from a curated book to broker execution. '
                'It reads the <strong>live curated portfolio</strong> directly — no CSV '
                're-upload — and writes each holding\'s unit count into the matching '
                'instrument of every broker order template you upload.'
            '</div>'
            '<div class="intel-method-grid">'

                '<div class="intel-method-tile tile-learns">'
                    '<div class="tile-label">Source</div>'
                    '<div class="tile-body">'
                        'The live curated portfolio in memory — its <code>symbol</code> and '
                        '<code>units</code> columns, exactly as shown in the Portfolio tab. '
                        'Nothing to export or re-import.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-how">'
                    '<div class="tile-label">Mapping</div>'
                    '<div class="tile-body">'
                        'For each instrument in a template, if its '
                        '<code>instrument.tradingsymbol</code> matches a curated holding, '
                        'that holding\'s units are written to <code>params.quantity</code>.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-obj">'
                    '<div class="tile-label">Templates</div>'
                    '<div class="tile-body">'
                        'Standard broker order JSONs (e.g. Kite <code>ETF.json</code>). '
                        'Upload as many as you like; each is synced and offered as a '
                        'separate <code>updated_*</code> download.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-safety">'
                    '<div class="tile-label">Safety</div>'
                    '<div class="tile-body">'
                        'Non-destructive: instruments with no matching holding are left '
                        '<strong>untouched</strong>, and the original files are never '
                        'modified — you download fresh copies.'
                    '</div>'
                '</div>'

            '</div>'
        '</div>'
    )
    st.markdown(method_html, unsafe_allow_html=True)


def _render_analytics_tab(portfolio: pd.DataFrame):
    """Tab — Portfolio Analytics: track the curated book vs a universe-matched
    benchmark (adapted from the SWING Analysis engine, re-themed to Obsidian Quant).

    Anchored to the analysis date (metrics run anchor → today). Shows a normalized
    portfolio-vs-benchmark chart plus risk-adjusted, risk, and benchmark-comparison
    metric cards. Uses the LIVE curated portfolio (no upload); the yfinance fetch is
    cached (see _analytics_series_cached).
    """
    from analytics import RISK_FREE_RATE, resolve_benchmark, compute_metrics
    from charts import create_benchmark_comparison_chart

    universe, selected_index, _regime, _mode = _intel_context()
    bench_ticker, bench_name = resolve_benchmark(universe, selected_index)

    # Guard: needs a curated portfolio with priced units.
    if portfolio is None or portfolio.empty or "symbol" not in portfolio.columns or "units" not in portfolio.columns:
        render_interpretation_card(
            title="NO CURATED PORTFOLIO",
            body=(
                "Run an analysis first — analytics track the live curated portfolio's "
                "performance against the benchmark, so there is nothing to measure yet."
            ),
            color="warning",
        )
        return

    # ── ANCHOR = the analysis date. Metrics run anchor → today; the window is
    #    dictated by the anchor (no user timeframe picker). Handle edge cases. ──
    _sel = st.session_state.get("selected_date")
    anchor_date = _sel if isinstance(_sel, date) else (
        _sel.date() if isinstance(_sel, datetime) else datetime.now().date()
    )
    today = datetime.now().date()

    # Edge: anchor is today or in the future → no forward history to measure.
    if anchor_date >= today:
        render_interpretation_card(
            title="ANCHORED TO TODAY",
            body=(
                f"The analysis date is <strong>{anchor_date.strftime('%d %b %Y')}</strong>, so there "
                "is no forward performance history yet. Analytics measure the curated book from the "
                "analysis date to the present — pick an earlier analysis date (with at least a few "
                "trading days elapsed) to see metrics."
            ),
            color="info",
        )
        return

    _elapsed_days = (today - anchor_date).days
    # Fetch enough calendar days to cover the anchor window (+buffer for alignment);
    # build_return_series then clips precisely to anchor → today.
    days_back = _elapsed_days + 5
    anchor_dt = datetime.combine(anchor_date, datetime.min.time())

    # ── Fetch + compute (CACHED) ───────────────────────────────────────────────
    #  The heavy yfinance fetch is behind _analytics_series_cached, keyed on the
    #  (symbols, units, anchor, benchmark) tuple, so it runs at most ONCE per
    #  unique window and every tab-switch / cosmetic rerun hits cache. Metrics
    #  render immediately on opening the tab — a scoped spinner only shows during
    #  the genuine first (cache-miss) fetch.
    _symbols = tuple(str(s) for s in portfolio["symbol"].tolist())
    _units = tuple(float(u or 0) for u in portfolio["units"].tolist())
    with st.spinner(f"Loading performance history · {bench_name} benchmark…"):
        port_value, port_returns, bench_returns, err, unpriced = _analytics_series_cached(
            _symbols, _units, anchor_dt.isoformat(), days_back, bench_ticker, bench_name,
        )

    if err:
        render_interpretation_card(
            title="DATA UNAVAILABLE",
            body=f"Could not build the performance series: {html_module.escape(err)}",
            color="danger",
        )
        return

    # Surface any held symbols that couldn't be priced/matched — the metrics below
    # exclude them, so the reported performance is for the priced remainder only.
    if unpriced:
        _shown = ", ".join(html_module.escape(s) for s in unpriced[:12])
        _more = f" (+{len(unpriced) - 12} more)" if len(unpriced) > 12 else ""
        render_interpretation_card(
            title="SOME HOLDINGS NOT PRICED",
            body=(
                f"<strong>{len(unpriced)}</strong> held symbol(s) could not be priced and are "
                f"<strong>excluded</strong> from these metrics: {_shown}{_more}. "
                "The performance below reflects only the priced holdings."
            ),
            color="warning",
        )

    # Edge: too few trading days since the anchor to compute meaningful metrics.
    if len(port_returns) < 2:
        render_interpretation_card(
            title="NOT ENOUGH HISTORY YET",
            body=(
                f"Only <strong>{len(port_returns)}</strong> trading day(s) have elapsed since "
                f"<strong>{anchor_date.strftime('%d %b %Y')}</strong>. Risk and benchmark metrics "
                "need at least a few daily returns — check back after more trading days, or use an "
                "earlier analysis date."
            ),
            color="warning",
        )
        return

    m = compute_metrics(port_returns, bench_returns, RISK_FREE_RATE)

    # ── Relative performance: header → anchor-window chip → normalized chart ────
    render_section_header("Relative Performance", f"Portfolio vs {bench_name} · indexed to 100",
                          icon="activity", accent="amber")
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:10px; margin:0 0 10px 0; '
        f'font-family:var(--data); font-size:0.72rem; letter-spacing:0.04em; color:var(--ink-tertiary);">'
        f'<span style="display:inline-flex; align-items:center; gap:6px; padding:4px 12px; '
        f'border:1px solid var(--border-active); border-radius:999px; background:rgba(212,168,83,0.06); '
        f'color:var(--amber); text-transform:uppercase; font-weight:700;">'
        f'Anchored · {anchor_date.strftime("%d %b %Y")} → Today</span>'
        f'<span>{len(port_returns)} trading days · {_elapsed_days} calendar day(s) elapsed</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _bench_series = None
    if bench_returns is not None and len(bench_returns) > 0:
        _bench_series = (1 + bench_returns).cumprod()
    if CHARTS_AVAILABLE and len(port_value) > 0:
        st.markdown('<div class="chart-container regime">', unsafe_allow_html=True)
        fig = create_benchmark_comparison_chart(
            port_value, _bench_series, bench_name, m.get("total_return", 0.0),
        )
        st.plotly_chart(fig, width="stretch", key="analytics_benchmark_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Returns & Risk-Adjusted Performance ────────────────────────────────────
    _section_divider()
    render_section_header("Returns & Risk-Adjusted Performance", "Period, CAGR & efficiency ratios",
                          icon="zap", accent="emerald")
    r1 = st.columns(6)
    with r1[0]:
        v = m.get("total_return", 0)
        render_metric_card("Period Return", f"{v:+.2f}%", f"CAGR: {m.get('cagr', 0):+.1f}%",
                           "success" if v >= 0 else "danger")
    with r1[1]:
        a = m.get("alpha", 0)
        render_metric_card("Alpha", f"{a:+.2f}%", "Excess return",
                           "success" if a > 0 else "danger" if a < 0 else "neutral")
    with r1[2]:
        s = m.get("sharpe", 0)
        render_metric_card("Sharpe", f"{s:.2f}", "Rf = 6.5%",
                           "success" if s > 1 else "warning" if s > 0.5 else "danger")
    with r1[3]:
        so = m.get("sortino", 0)
        render_metric_card("Sortino", f"{so:.2f}", "Downside risk",
                           "success" if so > 1.5 else "warning" if so > 0.5 else "danger")
    with r1[4]:
        ca = m.get("calmar", 0)
        render_metric_card("Calmar", f"{ca:.2f}", "Return / MaxDD",
                           "success" if ca > 1 else "warning" if ca > 0.5 else "danger")
    with r1[5]:
        ir = m.get("info_ratio", 0)
        render_metric_card("Info Ratio", f"{ir:.2f}", "Active return / TE",
                           "success" if ir > 0.5 else "warning" if ir > 0 else "danger")

    # ── Benchmark Comparison ───────────────────────────────────────────────────
    section_gap()
    render_section_header("Benchmark Comparison", f"vs {bench_name}", icon="compass", accent="cyan")
    r3 = st.columns(6)
    with r3[0]:
        br = m.get("benchmark_return", 0)
        render_metric_card("Benchmark", f"{br:+.1f}%", bench_name,
                           "success" if br >= 0 else "danger")
    with r3[1]:
        ex = m.get("total_return", 0) - m.get("benchmark_return", 0)
        render_metric_card("Excess Return", f"{ex:+.1f}%", "vs Benchmark",
                           "success" if ex > 0 else "danger")
    with r3[2]:
        uc = m.get("up_capture", 100)
        render_metric_card("Up Capture", f"{uc:.0f}%", "Bull market",
                           "success" if uc > 100 else "warning")
    with r3[3]:
        dc = m.get("down_capture", 100)
        render_metric_card("Down Capture", f"{dc:.0f}%", "Bear market",
                           "success" if dc < 100 else "danger")
    with r3[4]:
        render_metric_card("Correlation", f"{m.get('correlation', 0):.2f}", "vs Benchmark", "info")
    with r3[5]:
        render_metric_card("R-Squared", f"{m.get('r_squared', 0):.2f}", "Fit quality", "info")

    # ── Risk Metrics ───────────────────────────────────────────────────────────
    section_gap()
    render_section_header("Risk Metrics", "Volatility, drawdown & tail risk", icon="shield", accent="rose")
    r2 = st.columns(6)
    with r2[0]:
        render_metric_card("Volatility", f"{m.get('volatility', 0):.1f}%", "Annualized", "warning")
    with r2[1]:
        mdd = m.get("max_drawdown", 0)
        render_metric_card("Max Drawdown", f"{mdd:.1f}%", "Peak to trough",
                           "danger" if mdd < -20 else "warning" if mdd < -10 else "success")
    with r2[2]:
        render_metric_card("VaR (95%)", f"{m.get('var_95', 0):.2f}%", "Daily at risk", "danger")
    with r2[3]:
        render_metric_card("CVaR (95%)", f"{m.get('cvar_95', 0):.2f}%", "Expected shortfall", "danger")
    with r2[4]:
        b = m.get("beta", 1)
        render_metric_card("Beta", f"{b:.2f}", "Market sensitivity",
                           "warning" if b > 1.2 else "info" if b < 0.8 else "neutral")
    with r2[5]:
        render_metric_card("Tracking Error", f"{m.get('tracking_error', 0):.1f}%", "vs Benchmark", "info")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def _render_header() -> None:
    """Render the main masthead header."""
    render_header(
        title=f"{PRODUCT_NAME}",
        tagline="Conviction-Based Portfolio Curation · All 95 Strategies · Live NSE Data"
    )


def _render_landing_page():
    """Render landing page with system status cards."""
    section_gap()

    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        render_system_card(
            title="PORTFOLIO",
            description="Conviction-based portfolio curation with composite scoring across four technical indicators.",
            specs=[
                ("Signals", "RSI (30%) + Osc (30%) + Z (20%) + MA (20%)"),
                ("Selection", "Top 30 by conviction score"),
                ("Weighting", "(conviction / total) × 100"),
                ("Dispersion", "SIP + Swing modes")
            ],
            card_class="portfolio",
            icon="briefcase"
        )

    with col2:
        render_system_card(
            title="REGIME",
            description="Seven-factor market regime detection with composite scoring for adaptive positioning.",
            specs=[
                ("Regimes", "Strong Bull · Bull · Neutral · Bear"),
                ("Factors", "Momentum · Trend · Breadth · Velocity"),
                ("Output", "Confidence score + mix classification"),
                ("History", "30-day rolling window")
            ],
            card_class="regime",
            icon="compass"
        )

    with col3:
        render_system_card(
            title="STRATEGIES",
            description="Parallel quantitative engines scanning for momentum, reversal, breakout, and pattern signals.",
            specs=[
                ("Categories", "Momentum + Reversal + Breakout"),
                ("Universe", "Nifty 500 + F&O symbols"),
                ("Style", "SIP + Swing trading dispersion"),
                ("Strategies", "95 parallel engines")
            ],
            card_class="strategies",
            icon="layers"
        )

    section_gap()
    
    st.markdown("""
    <div class='landing-prompt'>
        <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
            AWAITING PARAMETERS
        </h4>
        <p>Configure via the <strong>Sidebar</strong>: select <strong>Analysis Date</strong>, <strong>Investment Style</strong>, <strong>Capital</strong>, and <strong>Number of Positions</strong>.<br>
           Execute <strong>Run Analysis</strong> to run all 95 strategies and curate a conviction-based portfolio.<br>
           <span style="color:var(--ink-secondary); font-size:0.85em; margin-top:0.5rem; display:inline-block;">System will detect market regime · Score conviction signals · Optimize weights</span></p>
    </div>
    """, unsafe_allow_html=True)


def _render_intelligence_tab(regime_d: Dict):
    """Tab — per-(universe, index, regime) conviction-weight calibration (4-dim, simplex)."""
    from intelligence import (
        IntelligencePassport, DEFAULT_WEIGHTS, DEFAULT_HORIZON,
        build_harvest, calibrate,
    )

    universe = st.session_state.get("selected_universe") or "default"
    selected_index = st.session_state.get("selected_index")
    regime_name = regime_d.get("regime", "UNKNOWN")
    passport = IntelligencePassport(universe, selected_index, regime_name)

    # Most-recent run outcome (mirrored from Phase 1.5)
    outcome = st.session_state.get("last_intel_outcome") or {}

    col1, col2 = st.columns([1, 1])

    with col1:
        render_section_header("Calibration Status", passport.label, icon="check-circle", accent="emerald")

        if passport.exists():
            m = passport.metrics()
            render_interpretation_card(
                title="CALIBRATED",
                body=(
                    f"Scope: <strong>{passport.label}</strong><br>"
                    f"Last calibrated: <strong>{passport.last_calibrated}</strong><br>"
                    f"Train IR: <strong>{m['train_ir']:+.3f}</strong> "
                    f"(over {m['n_train_dates']} dates)<br>"
                    f"Validation IR: <strong>{m['val_ir']:+.3f}</strong> "
                    f"(over {m['n_val_dates']} dates)<br>"
                    f"Horizon: <strong>{m['horizon']} trading days</strong> · "
                    f"Trials: <strong>{m['n_trials']}</strong>"
                ),
                color="success" if m["val_ir"] > 0 else "warning",
            )
        else:
            # Distinguish "never calibrated" from "tried but failed/skipped"
            status = outcome.get("status")
            reason = outcome.get("reason")
            if status in ("skipped", "failed") and outcome.get("regime") == regime_name:
                render_interpretation_card(
                    title=f"FELL BACK TO DEFAULTS · {status.upper()}",
                    body=(
                        f"Scope: <strong>{passport.label}</strong><br>"
                        f"Reason: <strong>{html_module.escape(reason or 'Unknown')}</strong><br>"
                        f"Conviction scoring is using Pragyam's default weights "
                        f"(RSI 0.30 · OSC 0.30 · Z 0.20 · MA 0.20)."
                    ),
                    color="warning",
                )
            else:
                render_interpretation_card(
                    title="NOT CALIBRATED",
                    body=(
                        f"No passport for <strong>{passport.label}</strong> yet. Conviction scoring "
                        f"uses Pragyam's default weights (RSI 0.30 · OSC 0.30 · Z 0.20 · MA 0.20). "
                        f"Run a calibration to learn weights for this scope."
                    ),
                    color="warning",
                )

        k1, k2 = st.columns(2)
        with k1:
            if st.button("Calibrate", type="primary", use_container_width=True, key="btn_calibrate"):
                hist_window = st.session_state.get("training_data_window", [])
                if not hist_window or len(hist_window) <= DEFAULT_HORIZON + 5:
                    st.error(
                        f"Need at least {DEFAULT_HORIZON + 5} days of history for a {DEFAULT_HORIZON}-day horizon. "
                        "Run an analysis with a longer lookback first."
                    )
                else:
                    with st.spinner(f"Calibrating {passport.label} ({DEFAULT_HORIZON}-day horizon, 100 trials)..."):
                        harvest = build_harvest(hist_window, horizon=DEFAULT_HORIZON)
                        if harvest.empty:
                            st.error("Harvest produced no usable rows. Indicator coverage may be too sparse.")
                        else:
                            result = calibrate(
                                universe, selected_index, regime_name,
                                harvest, n_trials=100, horizon=DEFAULT_HORIZON,
                            )
                            if result is None:
                                st.error(
                                    "Calibration could not produce a usable validation IR. "
                                    "Try a longer lookback or a regime with more historical coverage."
                                )
                            else:
                                st.success(
                                    f"Calibrated {passport.label} · Train IR {result['train_ir']:+.3f} · "
                                    f"Val IR {result['val_ir']:+.3f}"
                                )
                                st.rerun()
        with k2:
            if st.button("Reset", use_container_width=True, key="btn_reset"):
                if passport.exists():
                    passport.delete()
                    st.session_state.last_intel_outcome = None
                    st.rerun()

    with col2:
        render_section_header("Active Weights", "Conviction signal mix", icon="scale")

        weights = passport.get_weights()
        rows_html = []
        labels = [("RSI", "w_rsi"), ("Oscillator", "w_osc"), ("Z-Score", "w_z"), ("MA Alignment", "w_ma")]
        for label, key in labels:
            v = weights.get(key, DEFAULT_WEIGHTS[key])
            d = DEFAULT_WEIGHTS[key]
            delta = v - d
            color = "var(--emerald)" if delta > 0.005 else ("var(--rose)" if delta < -0.005 else "var(--ink-secondary)")
            arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "—")
            rows_html.append(
                f'<tr>'
                f'<td class="iw-label">{html_module.escape(label)}</td>'
                f'<td class="iw-long" style="color:{color}">{v:.3f} '
                f'<span class="iw-delta">{arrow} {abs(delta):.3f}</span></td>'
                f'<td class="iw-short" style="color:var(--ink-secondary)">{d:.3f}</td>'
                f'</tr>'
            )

        st.markdown(
            f'''
            <div class="intel-table-wrap">
                <table class="portfolio-table-2col">
                    <colgroup>
                        <col style="width:40%;">
                        <col style="width:35%;">
                        <col style="width:25%;">
                    </colgroup>
                    <thead>
                        <tr>
                            <th class="col-iw-factor">Signal</th>
                            <th class="col-iw-long">Active</th>
                            <th class="col-iw-short">Default</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows_html)}
                    </tbody>
                </table>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        total = sum(weights.get(k, DEFAULT_WEIGHTS[k]) for k in DEFAULT_WEIGHTS)
        st.markdown(f'<div class="intel-sigma">Σ = {total:.3f}</div>', unsafe_allow_html=True)

    # ── Full-width METHOD card (Obsidian Quant fidelity) ───────────────────
    # Placed after the two-column block so it spans the entire content width.
    # All styling lives in theme.css (.intel-method-*). The 4-tile grid calls
    # out each mechanic of the calibration individually rather than burying
    # it in a paragraph of jargon.
    method_html = (
        '<div class="intel-method-card">'
            '<div class="intel-method-header">'
                '<div class="intel-method-title">Method</div>'
                '<div class="intel-method-pill">'
                f'Optuna TPE · seed 42 · 100 trials · {DEFAULT_HORIZON}-day horizon'
                '</div>'
            '</div>'
            '<div class="intel-method-lede">'
                'The four conviction-signal weights (RSI · Oscillator · Z-Score · MA-alignment) are '
                '<strong>learned per (universe, index, regime)</strong> from historical '
                'signal-to-forward-return evidence — instead of the hard-coded 0.30 / 0.30 / 0.20 / 0.20 '
                'used in Standard mode. Different regimes reward different signals: bull markets favour '
                'momentum (RSI / MA), choppy markets favour mean-reversion (Z-Score). Intelligence Mode '
                'discovers the right mix automatically and stores it as a passport on disk.'
            '</div>'
            '<div class="intel-method-grid">'

                '<div class="intel-method-tile tile-learns">'
                    '<div class="tile-label">What it learns</div>'
                    '<div class="tile-body">'
                        'Four weights on the 4-simplex: '
                        '<code>w_rsi + w_osc + w_z + w_ma = 1</code>, each ≥ 0. '
                        'Parameterised via softmax over four unconstrained scalars, so the optimizer '
                        'sees a smooth, full-support landscape with no boundary degeneracies.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-how">'
                    '<div class="tile-label">How</div>'
                    '<div class="tile-body">'
                        'Bayesian search via <strong>Optuna Tree-structured Parzen Estimator (TPE)</strong> '
                        'with a fixed seed for reproducibility. 100 trials per calibration — typically '
                        '&lt;10 s on a 100-day, ~50-symbol panel thanks to vectorised IC evaluation.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-obj">'
                    '<div class="tile-label">Objective</div>'
                    '<div class="tile-body">'
                        '<strong>Information Ratio</strong> = <code>mean(IC) / std(IC)</code>, '
                        'where IC is the per-date Spearman rank correlation between the weighted '
                        f'conviction score and <strong>{DEFAULT_HORIZON}-day forward returns</strong>. '
                        'Maximises ranked predictive power per unit of cross-date volatility.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-safety">'
                    '<div class="tile-label">Safety rails</div>'
                    '<div class="tile-body">'
                        '<strong>70 / 30 chronological train-val split.</strong> A passport is saved only '
                        'when the held-out val IR is measurable. Calibration falls back to defaults — '
                        'and the Intelligence tab states the reason — when history is too short or the '
                        'harvest is too sparse.'
                    '</div>'
                '</div>'

            '</div>'
        '</div>'
    )
    st.markdown(method_html, unsafe_allow_html=True)


def _render_results(display_capital: float):
    """Render results page with portfolio, regime, and system tabs."""
    portfolio = st.session_state.portfolio
    if portfolio.empty or "value" not in portfolio.columns:
        st.warning("Portfolio is empty. Adjust parameters and re-run.")
        return

    current_df = st.session_state.current_df
    regime_d = st.session_state.regime_result_dict or {}
    training_window = st.session_state.get("training_data_window", [])

    total_value = portfolio["value"].sum()
    cash_remaining = display_capital - total_value

    # Top metrics — logical color coding
    mc1, mc2, mc3, mc4 = st.columns(4)

    # Cash health: <5% = danger, <15% = warning, else = success
    cash_pct = (cash_remaining / display_capital * 100) if display_capital > 0 else 0
    cash_color = "danger" if cash_pct < 5 else ("warning" if cash_pct < 15 else "success")

    # Avg conviction health: <35 = danger, 35-49 = warning, 50-64 = info, >=65 = success
    avg_conv = portfolio.get("conviction_score", pd.Series([50])).mean()
    conv_color = "danger" if avg_conv < 35 else ("warning" if avg_conv < 50 else ("info" if avg_conv < 65 else "success"))

    with mc1:
        render_metric_card("Deployed", f"₹{total_value:,.0f}", f"{total_value / display_capital * 100:.0f}% of capital", "info")
    with mc2:
        render_metric_card("Cash", f"₹{cash_remaining:,.0f}", f"{cash_pct:.1f}% remaining", cash_color)
    with mc3:
        render_metric_card("Positions", str(len(portfolio)), "Curated holdings", "warning")
    with mc4:
        render_metric_card("Avg Conviction", f"{avg_conv:.0f}/100", "Portfolio-wide average", conv_color)

    _section_divider()

    # Tab background pattern
    st.markdown('<div class="tab-bg portfolio"></div>', unsafe_allow_html=True)

    # Tabs
    tabs = ["Portfolio", "Position Guide", "Analytics", "Regime", "Intelligence", "Broker Sync", "System"]
    t_objs = st.tabs(tabs)

    with t_objs[0]:
        _render_portfolio_tab(portfolio, current_df, display_capital)

    with t_objs[1]:
        _render_position_guide_tab(portfolio, current_df)

    with t_objs[2]:
        _render_analytics_tab(portfolio)

    with t_objs[3]:
        _render_regime_tab(regime_d, st.session_state.get("regime_history_series", []), training_window)

    with t_objs[4]:
        _render_intelligence_tab(regime_d)

    with t_objs[5]:
        _render_broker_sync_tab(portfolio)

    with t_objs[6]:
        _render_system_tab(training_window)

    # Footer
    _section_divider()
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist)
    st.markdown(f"""
    <div class="app-footer">
        <div class="content">
            © {now_ist.year} <strong>Pragyam</strong> &nbsp;·&nbsp; @thebullishvalue &nbsp;·&nbsp; v{VERSION} &nbsp;·&nbsp; {now_ist.strftime("%Y-%m-%d %H:%M:%S IST")}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _run_analysis(
    selected_date: datetime,
    investment_style: str,
    capital: float,
    num_positions: int,
    selected_date_display: date,
    symbols_key: str,
    universe: str,
    index: str,
):
    """Execute the 2-phase analysis pipeline."""
    metrics = get_metrics()
    metrics.phases, metrics.errors, metrics.warnings = {}, [], []
    st.session_state.debug_info = []
    st.session_state.regime_history_series = None

    # Resolve the universe to get symbols
    try:
        symbols_list, status_msg = resolve_universe(universe, index)
    except Exception as e:
        st.error(f"Error resolving universe: {e}")
        st.stop()

    if not symbols_list:
        st.error(f"Could not load {index or universe}: {status_msg}")
        st.stop()

    try:
        # Print main header with run details
        from logger_config import generate_run_id
        current_run_id = generate_run_id()  # Fresh ID for each analysis
        run_details = {
            "Analysis Date": selected_date_display,
            "Universe": universe,
            "Index": index if index else "N/A",
            "Symbols": len(symbols_list),
            "Investment Style": investment_style,
            "Capital": f"₹{capital:,.0f}",
            "Positions": num_positions,
            "Run ID": current_run_id[-12:],
            "Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        log.main_header(f"PRAGYAM | Portfolio Intelligence | {VERSION}", run_details)

        # Custom styled progress container (matches Nishkarsh)
        progress_container = st.empty()

        # PHASE 1: DATA FETCHING
        progress_bar(progress_container, 2, "Fetching market data", f"yfinance · {len(symbols_list)} symbols")
        metrics.start_phase("total_execution")
        # Must match _REGIME_LOOKBACK_FILES so the regime card / regime banner /
        # regime history chart / Phase 2 curation all share one cached panel.
        LOOKBACK_FILES = _REGIME_LOOKBACK_FILES

        metrics.start_phase("data_fetching")

        if not symbols_list:
            st.error("Symbol universe empty — select a valid universe.")
            st.stop()

        all_hist = _load_historical_data(selected_date, LOOKBACK_FILES, symbols_key)
        if not all_hist:
            st.error("No historical data loaded. Check universe selection and date range.")
            st.stop()

        metrics.end_phase("data_fetching", success=True, items=len(all_hist))
        metrics.days_count = len(all_hist)

        progress_bar(progress_container, 15, "Data loaded", f"{len(all_hist)} days · {len(symbols_list)} symbols")

        # Regime detection
        progress_bar(progress_container, 20, "Detecting market regime", "7-factor composite scoring")
        regime_result = _detect_regime_cached(selected_date, symbols_key)
        regime_name = regime_result.get("regime", "UNKNOWN")
        confidence = regime_result.get("confidence", 0.0)

        st.session_state.regime_result_dict = regime_result
        st.session_state.suggested_mix = regime_result.get("mix_name", "Chop/Consolidate Mix")
        # Keep the regime-computation markers in sync so the sidebar card's
        # change-detection agrees with what the main flow just computed (otherwise
        # the sidebar can think the card is fresh when it is a run behind).
        st.session_state.regime_date = st.session_state.get("selected_date")
        st.session_state.regime_symbols_key = symbols_key
        st.session_state.training_data_window = all_hist

        if len(all_hist) < 10:
            st.error(f"Insufficient training data: {len(all_hist)} days (need ≥10).")
            metrics.end_phase("data_fetching", success=False, error_msg=f"Insufficient data: {len(all_hist)} days")
            st.stop()

        if not st.session_state.suggested_mix:
            st.error("Market regime could not be determined. Select a valid date.")
            metrics.end_phase("data_fetching", success=False, error_msg="Regime undetermined")
            st.stop()

        st.session_state.current_df = all_hist[-1][1] if all_hist else pd.DataFrame()

        progress_bar(progress_container, 25, "Phase 1 complete", "Data acquisition ready")

        # PHASE 1.5: Intelligence — calibrate weights for the (universe, index, regime)
        # tuple on first encounter; reuse the saved passport on subsequent runs.
        # Calibration outcome is mirrored to st.session_state.last_intel_outcome
        # so the sidebar passport card and result page can show what happened.
        if st.session_state.get("intelligence_mode"):
            from intelligence import (
                IntelligencePassport, build_harvest, calibrate, DEFAULT_HORIZON,
            )
            _universe = st.session_state.get("selected_universe") or "default"
            _selected_index = st.session_state.get("selected_index")
            _passport = IntelligencePassport(_universe, _selected_index, regime_name)
            _outcome: Dict = {
                "universe": _universe,
                "index": _selected_index,
                "regime": regime_name,
                "label": _passport.label,
            }

            if _passport.exists():
                _m = _passport.metrics()
                _outcome.update({
                    "status": "reused",
                    "reason": "Passport already calibrated for this (universe, index, regime).",
                    "train_ir": _m["train_ir"],
                    "val_ir":   _m["val_ir"],
                    "n_train_dates": _m["n_train_dates"],
                    "n_val_dates":   _m["n_val_dates"],
                })
                progress_bar(
                    progress_container, 25, "Intelligence ready",
                    f"{_passport.label} · Val IR {_m['val_ir']:+.3f} · "
                    f"calibrated {_passport.last_calibrated}",
                )
            elif len(all_hist) <= DEFAULT_HORIZON + 5:
                _outcome.update({
                    "status": "skipped",
                    "reason": f"Need >{DEFAULT_HORIZON + 5} days of history (have {len(all_hist)}).",
                })
                progress_bar(
                    progress_container, 25, "Intelligence skipped",
                    f"Need >{DEFAULT_HORIZON + 5} days of history · using default weights",
                )
            else:
                progress_bar(
                    progress_container, 22, "Calibrating intelligence",
                    f"Building signal-return panel · {_passport.label} · "
                    f"{DEFAULT_HORIZON}-day horizon",
                )
                _harvest = build_harvest(all_hist, horizon=DEFAULT_HORIZON)
                if _harvest.empty:
                    _outcome.update({
                        "status": "skipped",
                        "reason": "Harvest produced no usable rows (sparse indicators).",
                    })
                    progress_bar(
                        progress_container, 25, "Intelligence skipped",
                        "Harvest produced no usable rows · using default weights",
                    )
                else:
                    _n_dates = _harvest["date"].nunique()
                    _n_obs = len(_harvest)
                    _best_ir = [float("-inf")]
                    progress_bar(
                        progress_container, 23, "Calibrating intelligence",
                        f"Optuna TPE · {_n_dates} dates · {_n_obs:,} (date, symbol) rows",
                    )

                    def _intel_cb(trial: int, total: int, score: float):
                        if score > _best_ir[0]:
                            _best_ir[0] = score
                        pct = 23 + int((trial / max(total, 1)) * 5)
                        best = _best_ir[0]
                        best_str = f"{best:+.3f}" if best > float("-inf") else "—"
                        progress_bar(
                            progress_container, pct, "Calibrating intelligence",
                            f"Trial {trial}/{total} · best IR {best_str}",
                        )

                    _result = calibrate(
                        _universe, _selected_index, regime_name,
                        _harvest, n_trials=100,
                        horizon=DEFAULT_HORIZON, progress_callback=_intel_cb,
                    )
                    if _result is None:
                        _outcome.update({
                            "status": "failed",
                            "reason": "Validation IR not measurable on the held-out split.",
                        })
                        progress_bar(
                            progress_container, 28, "Intelligence skipped",
                            "Validation IR not measurable · using default weights",
                        )
                    else:
                        _outcome.update({
                            "status": "calibrated",
                            "reason": "Optimized 100 trials with measurable validation IR.",
                            "train_ir": _result["train_ir"],
                            "val_ir":   _result["val_ir"],
                            "n_train_dates": _result["n_train_dates"],
                            "n_val_dates":   _result["n_val_dates"],
                        })
                        progress_bar(
                            progress_container, 28, "Intelligence calibrated",
                            f"Train IR {_result['train_ir']:+.3f} · "
                            f"Val IR {_result['val_ir']:+.3f} · "
                            f"Passport saved for {_passport.label}",
                        )
                        # The sidebar passport card already painted before Phase 1.5
                        # ran, so it shows the pre-calibration state. Persist the
                        # outcome and rerun: run_analysis stays True, Phase 1 hits
                        # the data cache, Phase 1.5 takes the reuse path, Phase 2
                        # curates the portfolio, and the sidebar repaints with the
                        # freshly-saved passport. Adds ~2-4s of cache-hit work but
                        # eliminates the stale-sidebar surprise.
                        st.session_state.last_intel_outcome = _outcome
                        st.rerun()

            st.session_state.last_intel_outcome = _outcome
        else:
            st.session_state.last_intel_outcome = {"status": "disabled", "reason": "Intelligence mode is off."}

        # PHASE 2: CONVICTION-BASED CURATION
        progress_bar(progress_container, 25, "Running strategies", f"95 strategies · {len(symbols_list)} candidates")
        metrics.start_phase("conviction_curation")

        try:
            strategies = discover_strategies()
            strategies_to_run = {name: strategies[name] for name in strategies if name != "System_Curated"}

            if not strategies_to_run:
                st.error("No strategies available.")
                metrics.end_phase("conviction_curation", success=False, error_msg="Empty strategies")
                st.stop()

            # Aggregate holdings
            aggregated_holdings = {}
            progress_bar(progress_container, 35, "Aggregating holdings", f"Processing {len(strategies_to_run)} strategies")

            for name, strategy in strategies_to_run.items():
                try:
                    port = strategy.generate_portfolio(st.session_state.current_df, capital)
                    if port.empty:
                        continue
                    for _, row in port.iterrows():
                        symbol = row["symbol"]
                        price = row["price"]
                        if symbol not in aggregated_holdings:
                            aggregated_holdings[symbol] = {"price": price, "weight": 1.0}
                except Exception:
                    pass

            if not aggregated_holdings:
                st.error("No holdings generated.")
                metrics.end_phase("conviction_curation", success=False, error_msg="Empty holdings")
                st.stop()

            progress_bar(progress_container, 50, "Computing conviction", f"{len(aggregated_holdings)} candidates")

            # Conviction-based weighting with style-aware dispersion
            # SIP: +125% boost / -50% penalty | Swing: +225% boost / -75% penalty
            # Single source of truth for the intelligence context — mirrors what
            # regime.compute_conviction_signals will read internally, so the curated
            # weights and the displayed conviction column can never drift.
            _u, _idx, _, _mode = _intel_context()
            st.session_state.portfolio = compute_conviction_based_weights(
                aggregated_holdings,
                st.session_state.current_df,
                capital,
                num_positions,
                st.session_state.min_pos_pct,
                st.session_state.max_pos_pct,
                apply_dispersion=True,
                investment_style=investment_style,  # Auto-selects dispersion based on style
                universe=_u,
                selected_index=_idx,
                regime_name=regime_name,
                mode=_mode,
            )

            if st.session_state.portfolio.empty:
                st.error("No portfolio generated. Check data quality.")
                metrics.end_phase("conviction_curation", success=False, error_msg="Empty portfolio")
                st.stop()

            progress_bar(progress_container, 85, "Portfolio curated", f"{len(st.session_state.portfolio)} positions")

            # End conviction_curation phase tracking
            metrics.end_phase("conviction_curation", success=True)

            # Update metrics counters
            metrics.symbols_count = len(aggregated_holdings)
            metrics.strategies_count = len(strategies_to_run)
            metrics.portfolios_generated = len(st.session_state.portfolio)

            # Pre-compute regime history
            try:
                st.session_state.regime_history_series = get_regime_history_series(all_hist, window_size=10, step=1)
            except Exception:
                st.session_state.regime_history_series = []

            metrics.end_phase("total_execution", success=True)
            progress_bar(progress_container, 100, "Analysis complete", f"Portfolio: {len(st.session_state.portfolio)} positions ready")

            # Defensive: if conviction_score is all-NaN (shouldn't happen since
            # compute_conviction_signals fillna's to 50, but guard anyway so the
            # execution-summary line never prints "nan/100").
            _conv = st.session_state.portfolio.get("conviction_score", pd.Series([50])).dropna()
            avg_conviction = _conv.mean() if len(_conv) else 50.0
            top_conviction = _conv.max()  if len(_conv) else 50.0

            log.summary("Execution Summary", {
                "Run ID": current_run_id[-12:],
                "Strategies Run": len(strategies_to_run),
                "Candidate Symbols": len(aggregated_holdings),
                "Positions Selected": len(st.session_state.portfolio),
                "Avg Conviction": f"{avg_conviction:.1f}/100",
                "Top Conviction": f"{top_conviction:.0f}/100",
                "Status": "SUCCESS",
            })
            metrics.print_summary(log)
            
            # Clear progress container after short delay
            import time
            time.sleep(1.5)
            progress_container.empty()
            
            st.toast("Analysis Complete!")

        except Exception as e:
            metrics.end_phase("total_execution", success=False, error_msg=str(e))
            st.error(f"Analysis failed: {e}")
            progress_container.empty()

    except Exception as e:
        st.error(f"Initialization failed: {e}")


def _render_footer() -> None:
    """Render the app footer with copyright and version info."""
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    st.markdown(
        f'<div class="app-footer">'
        f'<div class="content">'
        f'© {ist_now.year} <strong>{PRODUCT_NAME}</strong> &nbsp;·&nbsp; {COMPANY} &nbsp;·&nbsp; v{VERSION} &nbsp;·&nbsp; {ist_now.strftime("%Y-%m-%d %H:%M:%S IST")}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point."""
    _init_session_state()

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align:center;padding:0.5rem 0 0.75rem 0;">
            <div style="font-family:var(--display);font-size:1.35rem;font-weight:700;color:var(--amber);letter-spacing:0.04em;">PRAGYAM</div>
            <div style="font-family:var(--data);color:var(--ink-tertiary);font-size:0.6rem;margin-top:0.1rem;letter-spacing:0.06em;text-transform:uppercase;">प्रज्ञम | Portfolio Intelligence</div>
        </div>
        <hr style="margin: 0.5rem 0; opacity: 0.1;">
        """,
            unsafe_allow_html=True,
        )

        # 1. Analysis Date
        st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
        selected_date = st.date_input(
            "Reference Date",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            help="Select the snapshot date for portfolio curation",
            label_visibility="visible"
        )
        st.session_state.selected_date = selected_date
        selected_date_obj = datetime.combine(selected_date, datetime.min.time())

        # 2. Portfolio Style
        st.markdown('<div class="sidebar-title">Portfolio Style</div>', unsafe_allow_html=True)
        investment_style = st.selectbox(
            "Investment Objective",
            options=["Swing Trading", "SIP Investment"],
            index=0,
            help="Choose between short-term swing or long-term SIP curation",
            label_visibility="visible"
        )

        # 3. Analysis Universe
        st.markdown('<div class="sidebar-title">Analysis Universe</div>', unsafe_allow_html=True)
        universe, selected_index = render_universe_selector()
        st.session_state.selected_universe = universe
        st.session_state.selected_index = selected_index

        # Create symbols key for regime detection
        symbols_key = f"UNIVERSE:{universe}|{selected_index}"
        st.session_state.symbols_key = symbols_key

        # 4. Regime Card
        # NOTE: read the "last regime computation" markers, NOT selected_date as a
        # fallback — selected_date was just overwritten above with the NEW date, so
        # falling back to it would make date_changed always False and freeze the
        # card. When no regime has been computed yet, treat it as needing update.
        previous_date = st.session_state.get("regime_date")
        previous_symbols_key = st.session_state.get("regime_symbols_key")
        date_changed = previous_date != selected_date
        universe_changed = previous_symbols_key != symbols_key

        rd = st.session_state.get("regime_result_dict", {})
        regime_needs_update = not rd or date_changed or universe_changed

        if regime_needs_update:
            with st.spinner("Detecting regime..."):
                rd = _detect_regime_cached(selected_date_obj, symbols_key)
                st.session_state.regime_result_dict = rd
                st.session_state.suggested_mix = rd.get("mix_name", "Chop/Consolidate Mix")
                st.session_state.regime_date = selected_date
                st.session_state.regime_symbols_key = symbols_key
        
        if rd and isinstance(rd, dict):
            regime_name_sb = rd.get("regime", "UNKNOWN")
            color_sb = rd.get("color", "#888888")
            conf_sb = rd.get("confidence", 0.0)
            score_sb = rd.get("composite_score", 0.0)
            st.markdown(f"""
            <div style="background:{color_sb}12; border:1px solid {color_sb}40; border-radius:10px;
                        padding:12px; margin:var(--sp-6) 0 var(--sp-3) 0;">
                <div style="color:var(--ink-tertiary); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; margin-bottom:4px; font-family:var(--data);">Market Regime</div>
                <div style="color:{color_sb}; font-size:1.25rem; font-weight:700; line-height:1.2; font-family:var(--display); display:flex; align-items:center; gap:8px;">
                    {get_icon(rd.get('icon', ''), size=20, stroke_width=1.8)} {regime_name_sb.replace('_', ' ')}
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
                    <span style="color:var(--ink-tertiary); font-size:0.75rem; font-family:var(--data);">Score {score_sb:+.2f}</span>
                    <span style="color:{color_sb}; font-weight:700; font-size:0.8rem; font-family:var(--data);">{conf_sb:.0%} confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 5. Portfolio Parameters
        st.markdown('<div class="sidebar-title">Portfolio Parameters</div>', unsafe_allow_html=True)
        capital = st.number_input(
            "Capital (₹)",
            min_value=1000,
            max_value=100000000,
            value=2500000,
            step=1000,
            help="Total capital to allocate",
            label_visibility="visible"
        )
        st.session_state["capital"] = capital

        num_positions = st.slider(
            "Number of Positions",
            min_value=5,
            max_value=100,
            value=30,
            step=5,
            help="Maximum portfolio positions"
        )
        st.session_state.min_pos_pct = 1.0 / 100
        st.session_state.max_pos_pct = 10.0 / 100

        # 6. Run Button
        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

        if run_clicked:
            st.session_state["run_params"] = {
                "selected_date_obj": selected_date_obj,
                "investment_style": investment_style,
                "capital": capital,
                "num_positions": num_positions,
                "selected_date": selected_date,
                "symbols_key": symbols_key,
                "universe": universe,
                "index": selected_index,
            }
            st.session_state["run_analysis"] = True
            st.rerun()

        # ── Model Passport (Sanket-style) ──────────────────────────────────────
        # Mirrors Sanket's sidebar passport: profile state · trained-on label ·
        # regime · train/val IR · last-updated timestamp · import / export / reset.
        # Located below the Run button so it surfaces the freshly-saved passport
        # when Phase 1.5 has just calibrated and reran. Rendered as a standard
        # titled section (no divider) so its gap matches every other section.
        st.markdown('<div class="sidebar-title">Model Passport</div>', unsafe_allow_html=True)

        st.session_state.intelligence_mode = st.toggle(
            "Intelligence Mode",
            value=bool(st.session_state.get("intelligence_mode", True)),
            help="Use per-(universe, index, regime) calibrated conviction weights. "
                 "When OFF, Pragyam uses the canonical 0.30 / 0.30 / 0.20 / 0.20 weights.",
            key="passport_intel_toggle",
        )

        from intelligence import IntelligencePassport, DEFAULT_WEIGHTS, PASSPORT_VERSION
        _pp_universe = universe or "default"
        _pp_index = selected_index
        _pp_regime = (rd.get("regime", "UNKNOWN") if rd else "UNKNOWN")
        _pp_passport = IntelligencePassport(_pp_universe, _pp_index, _pp_regime)

        if not st.session_state.intelligence_mode:
            profile_label = "Default · Off"
            train_str = val_str = updated = "—"
            cal_label_disp = (_pp_index or _pp_universe or "—")
            train_color = val_color = "var(--ink-secondary)"
            card_class = ""  # neutral
        elif _pp_passport.exists():
            _m = _pp_passport.metrics()
            train_v = _m.get("train_ir") or 0.0
            val_v   = _m.get("val_ir") or 0.0
            train_str = f"{train_v:+.3f}"
            val_str   = f"{val_v:+.3f}"
            updated   = _pp_passport.last_calibrated
            train_color = "var(--emerald)" if train_v > 0 else "var(--rose)"
            val_color   = "var(--emerald)" if val_v   > 0 else "var(--rose)"
            cal_label_disp = (_pp_index or _pp_universe or "—")
            profile_label = "Calibrated"
            card_class = "success" if (val_v > 0 and train_v > 0) else "warning"
        else:
            profile_label = "Default"
            train_str = val_str = updated = "—"
            cal_label_disp = (_pp_index or _pp_universe or "—")
            train_color = val_color = "var(--ink-secondary)"
            card_class = ""  # neutral

        def _trim(s, n=22):
            s = str(s)
            return s if len(s) <= n else s[: n - 1] + "…"
        cal_label_disp = _trim(cal_label_disp)
        regime_disp = _trim(_pp_regime.replace("_", " "), 22)

        st.markdown(f"""
        <div class="metric-card {card_class}" style="
                min-height:auto;
                padding:0.85rem 0.95rem;
                margin-bottom:0.7rem;
                animation:none;">
            <h4 style="margin:0 0 0.3rem 0;">Profile</h4>
            <h2 style="font-size:1.05rem; margin:0 0 0.7rem 0; letter-spacing:-0.01em;">{profile_label}</h2>
            <div style="display:flex; flex-direction:column; gap:0.32rem;
                        padding-top:0.55rem;
                        border-top:1px solid rgba(255,255,255,0.06);">
                <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.62rem;">
                    <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Trained on</span>
                    <span style="color:var(--ink-secondary); font-weight:500; max-width:62%; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{html_module.escape(cal_label_disp)}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.62rem;">
                    <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Regime</span>
                    <span style="color:var(--ink-secondary); font-weight:500;">{html_module.escape(regime_disp)}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                    <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Train IR</span>
                    <span style="color:{train_color}; font-weight:600;">{train_str}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                    <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Val IR</span>
                    <span style="color:{val_color}; font-weight:600;">{val_str}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.6rem;">
                    <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Updated</span>
                    <span style="color:var(--ink-secondary);">{html_module.escape(updated)}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Import · Export · Reset (parity with Sanket's passport controls)
        with st.expander("↑ Import Profile", expanded=False):
            uploaded = st.file_uploader(
                " ", type=["json"], label_visibility="collapsed",
                key="passport_uploader",
            )
            if uploaded is not None:
                try:
                    import json as _json
                    payload = _json.load(uploaded)
                    if not isinstance(payload, dict):
                        raise ValueError("file is not a JSON object")
                    weights = payload.get("weights", payload)
                    if not all(k in weights for k in DEFAULT_WEIGHTS):
                        raise ValueError("missing one of w_rsi / w_osc / w_z / w_ma")
                    _pp_passport.save(
                        weights={k: float(weights[k]) for k in DEFAULT_WEIGHTS},
                        train_ir=float(payload.get("train_ir", 0.0)),
                        val_ir  =float(payload.get("val_ir",   0.0)),
                        n_train_dates=int(payload.get("n_train_dates", 0)),
                        n_val_dates  =int(payload.get("n_val_dates",   0)),
                        n_trials=int(payload.get("n_trials", 0)),
                        horizon =int(payload.get("horizon", 10)),
                    )
                    st.toast("Profile imported.", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

        # Export the active passport (or defaults wrapped in passport shape)
        if _pp_passport.exists():
            _export_payload = _pp_passport.data
            _export_universe = _pp_universe
            _export_index    = _pp_index
        else:
            _export_payload = {
                "universe": _pp_universe,
                "selected_index": _pp_index,
                "regime": _pp_regime,
                "weights": DEFAULT_WEIGHTS,
                "engine_version": PASSPORT_VERSION,
                "is_calibrated": False,
            }
            _export_universe = _pp_universe
            _export_index    = _pp_index
        _export_slug_parts = [
            re.sub(r"[^a-z0-9]+", "_", (_export_universe or "default").lower()).strip("_"),
            re.sub(r"[^a-z0-9]+", "_", (_export_index or "all").lower()).strip("_"),
            re.sub(r"[^a-z0-9]+", "_", _pp_regime.lower()).strip("_"),
            datetime.now().strftime("%Y%m%d"),
        ]
        _export_filename = "pragyam_profile_" + "__".join(_export_slug_parts) + ".json"

        import json as _json
        st.download_button(
            "↓ Export Profile",
            data=_json.dumps(_export_payload, indent=2, default=str),
            file_name=_export_filename,
            mime="application/json",
            use_container_width=True,
            key="passport_export",
        )
        if st.button("↺ Reset to Defaults", use_container_width=True, key="passport_reset"):
            if _pp_passport.exists():
                _pp_passport.delete()
                st.session_state.last_intel_outcome = None
            st.rerun()

        st.markdown('<hr style="margin: 3.00rem 0; opacity: 0.05;">', unsafe_allow_html=True)


        # Show current universe info
        try:
            symbols_list, status_msg = resolve_universe(universe, selected_index)
            rows = [
                '<div class="system-spec">',
                '<div class="spec-row"><span class="spec-label">Version</span><span class="spec-value">' + VERSION + '</span></div>',
                '<div class="spec-row"><span class="spec-label">Universe</span><span class="spec-value">' + universe + '</span></div>',
            ]
            if selected_index:
                rows.append('<div class="spec-row"><span class="spec-label">Index</span><span class="spec-value">' + selected_index + '</span></div>')
            num_symbols = len(symbols_list) if symbols_list is not None else 0
            rows.append('<div class="spec-row"><span class="spec-label">Symbols</span><span class="spec-value">' + str(num_symbols) + '</span></div>')
            rows.append('<div class="spec-row"><span class="spec-label">Data</span><span class="spec-value">yfinance</span></div>')
            rows.append('</div>')
            st.markdown(''.join(rows), unsafe_allow_html=True)
        except Exception:
            rows = [
                '<div class="system-spec">',
                '<div class="spec-row"><span class="spec-label">Version</span><span class="spec-value">' + VERSION + '</span></div>',
                '<div class="spec-row"><span class="spec-label">System</span><span class="spec-value">Conviction-Based</span></div>',
                '<div class="spec-row"><span class="spec-label">Data</span><span class="spec-value">yfinance</span></div>',
                '</div>',
            ]
            st.markdown(''.join(rows), unsafe_allow_html=True)


    # Main content area
    # ─── Show progress bar in main area (outside sidebar) when running analysis ───
    if st.session_state.get("run_analysis") and st.session_state.get("run_params"):
        params = st.session_state["run_params"]
        _run_analysis(
            params["selected_date_obj"], params["investment_style"],
            params["capital"], params["num_positions"], params["selected_date"],
            params["symbols_key"], params["universe"], params["index"],
        )
        # Clear the flag after analysis completes
        st.session_state.pop("run_analysis", None)
        st.session_state.pop("run_params", None)

    if st.session_state.portfolio is None and not st.session_state.get("run_analysis"):
        _render_header()
        _render_landing_page()
        _render_footer()
    elif st.session_state.portfolio is not None:
        # Get capital from session state or default
        display_capital = st.session_state.get("capital", 2500000)
        _render_results(display_capital)


if __name__ == "__main__":
    main()
