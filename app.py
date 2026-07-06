"""
PRAGYAM — Portfolio Intelligence (Streamlit App)
══════════════════════════════════════════════════════════════════════════════

Conviction-based portfolio curation with 95 quantitative strategies.

Architecture:
  regime.py         → MarketRegimeDetector (fixed 8-factor), compute_conviction_signals
  portfolio.py      → compute_conviction_based_weights() (value-area tie-break)
  backdata.py       → generate_historical_data(), compute_volume_profile()
  intelligence.py   → conviction-weight calibration (passports)
  analytics.py      → portfolio-vs-benchmark performance metrics (Analytics tab)
  charts.py         → Plotly chart builders
  strategies.py     → 95 BaseStrategy implementations

Conviction blend: 6 signals — RSI · Oscillator · Z-Score · MA-alignment · Value
Area (VAP, from the volume profile) · Strategy Endorsement (cross-sectional
rank of 95-strategy top-quartile votes). Five of the six weights are
regime-calibrated in Intelligence mode (5-simplex over RSI/OSC/Z/MA/VAP,
scaled to fill the remaining mass); Strategy Endorsement stays fixed since it
has no historical values to calibrate against. Even default fallback (1/6 x6)
otherwise.

Pipeline:
  Phase 1:   Data fetching + regime detection (fixed 8-factor weights)
  Phase 1.5: Conviction-weight calibration (Intelligence mode)
  Phase 2:   Conviction-based portfolio curation

Result tabs: Portfolio · Position Guide · Analytics (curated book vs benchmark) ·
Regime · Intelligence · Broker Sync (curated units → broker JSONs) · System.

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
    render_system_card,
    section_gap,
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

VERSION = "v10.0.1"
PRODUCT_NAME = "Pragyam"
COMPANY = "@thebullishvalue"

st.set_page_config(
    page_title="PRAGYAM | Portfolio Intelligence",
    page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
    layout="wide",
    # Start EXPANDED: the landing page explicitly instructs "Configure via
    # the Sidebar", so a first-time visitor should see the sidebar controls
    # immediately rather than discover they're collapsed (see
    # AUDIT_DIRECTIVES.md C5.5).
    initial_sidebar_state="expanded",
)

# Load Obsidian Quant Terminal CSS
inject_css()


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
        # Effective bounds actually applied by the last curation run (may
        # differ from the nominal min/max_pos_pct above when num_positions
        # made them infeasible — see portfolio.compute_conviction_based_weights).
        "min_pos_pct_eff": 0.01,
        "max_pos_pct_eff": 0.10,
        "intelligence_mode": True,  # Use calibrated weights when a passport exists; falls back to defaults otherwise.
        "selected_universe": None,
        "selected_index": None,
        # Frozen (universe, index, regime, mode, anchor_date) the CURRENT
        # portfolio was curated under — see _intel_context()'s docstring.
        "run_context": None,
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
    """(universe, selected_index, regime_name, mode) for scoring/display.

    Once a portfolio has been curated, returns the FROZEN context that
    portfolio was actually curated under (st.session_state.run_context) —
    not live sidebar state. Without this, changing the sidebar's universe or
    date after a run (which immediately recomputes the sidebar regime card
    and mutates selected_universe/selected_index) would re-score the OLD
    curated portfolio under the NEW scope's passport, and the Analytics tab
    would resolve the NEW universe's benchmark against the OLD book — the
    displayed conviction/weights could no longer reconcile with what was
    actually curated (see AUDIT_DIRECTIVES.md A12). Falls back to live
    sidebar state only when no run has completed yet (pre-run / landing page).
    """
    run_ctx = st.session_state.get("run_context")
    if run_ctx is not None and st.session_state.get("portfolio") is not None:
        return (
            run_ctx["universe"], run_ctx["selected_index"],
            run_ctx["regime_name"], run_ctx["mode"],
        )
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


def _log_intel_outcome(outcome: Dict) -> None:
    """Mirror the Phase 1.5 intelligence outcome to the terminal.

    The progress bar shows one Intelligence milestone per run
    (Ready/Calibrated/Skipped); the console trace must carry the same fact —
    especially the skip/fail REASON, which is the single most diagnostically
    useful line a run produces and previously never reached the terminal.
    """
    log.section("Intelligence", phase="PHASE 1.5")
    status = (outcome.get("status") or "unknown").upper()
    if outcome.get("label"):
        log.item("Scope", outcome["label"])
    log.item("Status", status)
    if outcome.get("status") in ("calibrated", "reused"):
        log.item("Train IR", f"{outcome.get('train_ir', float('nan')):+.3f}")
        log.item("Val IR", f"{outcome.get('val_ir', float('nan')):+.3f}")
    if outcome.get("reason"):
        log.detail(outcome["reason"])


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

# Estimation panel used ONLY by Phase 1.5 calibration — sized from the
# statistics, not from the display. The paired beats-default gate needs
# MIN_PAIRED_VAL_DATES (=8) non-overlapping validation dates at horizon 10
# under a 50/50 split, i.e. >= intelligence.min_calibration_dates() = 142
# usable dates INSIDE the target regime family. The regime family typically
# covers only a fraction of any trailing window, so the window must be a
# multiple of that: at a ~40% family share, 142 / 0.4 + horizon ≈ 365 → 375
# trading days (~18 months). The 100-day _REGIME_LOOKBACK_FILES panel was
# structurally incapable of EVER calibrating: 90 harvest dates → 45
# validation dates → at most 5 non-overlapping paired dates < 8, so every
# run failed the gate before a single Optuna trial ran. Kept separate from
# _REGIME_LOOKBACK_FILES so the regime card / chart / curation stay on the
# fast 100-day panel; this longer panel is fetched (and cached for the
# session) only when a scope actually needs calibrating — reused passports
# never pay for it.
_CALIBRATION_LOOKBACK_FILES = 375


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
    bench_returns, err, unpriced). Compute stays in analytics.py; caching lives here (the
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

    # Volume-dependent-signal coverage warning: FX pairs and some futures
    # report zero/NaN volume on Yahoo, so osc_signal/vap_signal/zscore_signal
    # are structurally absent for those symbols (compute_conviction_signals
    # renormalizes weights over whatever signals ARE available, so scores
    # still span the full range — see AUDIT_DIRECTIVES.md A14 — but the
    # conviction read is based on fewer independent signals, which is worth
    # surfacing explicitly rather than presenting with the same confidence
    # as a fully-covered universe).
    if "signals_available" in portfolio_with_signals.columns and len(portfolio_with_signals) > 0:
        _low_coverage = (portfolio_with_signals["signals_available"] <= 3).mean()
        if _low_coverage > 0.5:
            render_interpretation_card(
                title="LIMITED SIGNAL COVERAGE",
                body=(
                    f"<strong>{_low_coverage:.0%}</strong> of this book's holdings have 3 or fewer of "
                    "the six conviction signals available — likely because this universe reports zero "
                    "or missing volume (common for FX pairs and some futures), so the "
                    "volume-dependent signals (Oscillator, Z-Score, Value Area) are structurally "
                    "absent. Conviction weights are renormalized over the available signals, but the "
                    "read relies on fewer independent inputs than a fully-covered universe."
                ),
                color="warning",
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
        "Real-time indicator alignment — RSI · Oscillator · Z-Score · MA Alignment · Value Area · Strategy",
        icon="activity",
        accent="cyan",
    )

    if CHARTS_AVAILABLE and not portfolio_with_signals.empty:
        st.markdown('<div class="chart-container portfolio">', unsafe_allow_html=True)
        fig_conv = create_conviction_heatmap(portfolio_with_signals)
        st.plotly_chart(fig_conv, width='stretch', key="tab1_conviction_heatmap")
        st.markdown('</div>', unsafe_allow_html=True)
        _blend_desc = "regime-calibrated weights" if mode == "Intelligence" else "even default weights (Intelligence Mode is off)"
        st.caption(
            "Green = bullish · Red = bearish · Six signals — RSI · Oscillator · Z-Score · "
            f"MA Alignment · Value Area · Strategy Endorsement · blended by {_blend_desc}"
        )
    elif not portfolio_with_signals.empty:
        conv_cols = [c for c in ["symbol", "rsi_value", "osc_value", "zscore_value", "ma_count", "vap_value", "conviction_score"]
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
        use_container_width=True,
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
            f'<td class="col-vap numeric">{row.get("vap_signal", 0):+.1f}</td>'
            f'<td class="col-strat numeric">{row.get("strat_signal", 0):+.1f}</td>'
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
        .col-symbol {{ width: 13%; }}
        .col-price {{ width: 10%; }}
        .col-signal {{ width: 14%; }}
        .col-conviction {{ width: 8%; }}
        .col-rsi {{ width: 7%; }}
        .col-osc {{ width: 7%; }}
        .col-z {{ width: 7%; }}
        .col-ma {{ width: 7%; }}
        .col-vap {{ width: 7%; }}
        .col-strat {{ width: 7%; }}
        .col-weight {{ width: 10%; }}
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
                    <th class="col-vap numeric">VAP</th>
                    <th class="col-strat numeric">Strat</th>
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

    # The regime detector uses FIXED factor weights (not calibrated); display them
    # so the percentages match what the composite actually used.
    try:
        from regime import FACTOR_WEIGHTS
        _fw = FACTOR_WEIGHTS
    except Exception:
        _fw = {}
    factor_order = [
        ("momentum", "Momentum", "strength"),
        ("trend", "Trend", "quality"),
        ("breadth", "Breadth", "quality"),
        ("velocity", "Velocity", "acceleration"),
        ("extremes", "Extremes", "type"),
        ("volatility", "Volatility", "regime"),
        ("acceptance", "Acceptance", "state"),
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

    # ── ONE flex row: factor scores (left, flex:2) + regime badge (right, flex:1).
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
    method_html = (
        '<div class="intel-method-card">'
            '<div class="intel-method-header">'
                '<div class="intel-method-title">How the Regime Is Read</div>'
                '<div class="intel-method-pill">'
                '8-factor composite · fixed weights'
                '</div>'
            '</div>'
            '<div class="intel-method-lede">'
                'The market regime is a <strong>weighted composite</strong> of eight measured '
                'factors, each scored on a signed <code>[-2, +2]</code> scale (bearish ↔ bullish). '
                'The weighted sum places the market on the regime hierarchy from '
                '<strong>Strong Bull</strong> down to <strong>Crisis</strong>. The eight factor '
                'weights are <strong>fixed</strong> (regime.FACTOR_WEIGHTS) regardless of mode — '
                'the regime detector is not calibrated. Only the six-signal '
                '<strong>conviction blend</strong> is regime-calibrated in Intelligence mode.'
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
                    '<div class="tile-label">Extremes, Volatility &amp; Acceptance</div>'
                    '<div class="tile-body">'
                        'Z-score crowding, Bollinger band-width regime, and the volume-profile '
                        'value distribution (discount vs premium) — stress and mean-reversion context.'
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
    _min_eff = st.session_state.get("min_pos_pct_eff", st.session_state.min_pos_pct)
    _max_eff = st.session_state.get("max_pos_pct_eff", st.session_state.max_pos_pct)
    _min_relaxed = abs(_min_eff - st.session_state.min_pos_pct) > 1e-9
    _max_relaxed = abs(_max_eff - st.session_state.max_pos_pct) > 1e-9
    details = {
        "Version": VERSION,
        "Curation Method": "Conviction-based (6-signal blend)",
        "Weight Formula": "(conviction / total) × 100",
        "Min Position": f"{_min_eff*100:.1f}%" + (" (relaxed)" if _min_relaxed else ""),
        "Max Position": f"{_max_eff*100:.1f}%" + (" (relaxed)" if _max_relaxed else ""),
        "Data Source": "yfinance (NSE)",
        "Lookback Period": f"{len(training_window)} days",
    }
    render_kv_table(details)
    if _min_relaxed or _max_relaxed:
        st.caption(
            f"Bounds relaxed from the nominal {st.session_state.min_pos_pct*100:.0f}%–"
            f"{st.session_state.max_pos_pct*100:.0f}% because the selected position count "
            "made them mathematically infeasible (too few/many positions to satisfy both "
            "the cap and 100% allocation)."
        )

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
                'Every symbol in the universe is scored 0–100 by a six-signal conviction blend, '
                'read against the market regime, then the highest-conviction names are curated into '
                'a bounded, dispersion-weighted book.'
            '</div>'
            '<div class="intel-method-grid">'

                '<div class="intel-method-tile tile-learns">'
                    '<div class="tile-label">Signals</div>'
                    '<div class="tile-body">'
                        'RSI · Oscillator · Z-Score · MA-alignment · Value Area (VAP) · Strategy '
                        'Endorsement (95-strategy vote rank), each in <code>[-2, +2]</code>. Blended by '
                        'regime-calibrated weights (Strategy Endorsement fixed), mapped to '
                        '<code>(raw + 2) / 4 × 100</code>.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-how">'
                    '<div class="tile-label">Regime</div>'
                    '<div class="tile-body">'
                        'An eight-factor composite places the market on the Strong Bull → Crisis '
                        'hierarchy and selects the conviction-weight passport used for scoring.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-obj">'
                    '<div class="tile-label">Selection</div>'
                    '<div class="tile-body">'
                        'Top N by conviction, ties broken by value-area position (a discount to '
                        'accepted value is preferred). No hard threshold — all symbols are eligible.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-safety">'
                    '<div class="tile-label">Weighting</div>'
                    '<div class="tile-body">'
                        'Style-aware power-law dispersion (conviction ** gamma; SIP / Swing) then '
                        '<code>weight = (adjusted / total) × 100</code>, bounded to '
                        f'{_min_eff*100:.0f}%–{_max_eff*100:.0f}% per position.'
                    '</div>'
                '</div>'

            '</div>'
        '</div>'
    )
    st.markdown(method_html, unsafe_allow_html=True)


def _sync_broker_json(json_data, quantity_map: Dict[str, int]) -> Tuple[list, int, int]:
    """Map curated per-symbol units into a broker order-template JSON.

    Walks each instrument in the template, and where its
    ``instrument.tradingsymbol`` matches a curated holding WITH units > 0,
    writes the holding's unit count into ``params.quantity``. A match with
    units == 0 is left untouched rather than zeroing out the template's
    existing quantity — the method card promises non-matching instruments
    are "untouched", and a matched-but-zero-conviction holding silently
    zeroing a possibly-intentional manual quantity was a third, undocumented
    case (see AUDIT_DIRECTIVES.md B8). Returns (mutated JSON, instruments
    updated, instruments matched-but-skipped-for-zero-units).
    """
    updated = 0
    skipped_zero = 0
    for item in json_data:
        try:
            symbol = item.get("instrument", {}).get("tradingsymbol")
            if symbol and symbol in quantity_map and "params" in item:
                qty = int(quantity_map[symbol])
                if qty > 0:
                    item["params"]["quantity"] = qty
                    updated += 1
                else:
                    skipped_zero += 1
        except Exception:
            continue
    return json_data, updated, skipped_zero


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
    results = []  # (fname, payload_or_None, updated_count, skipped_zero_count, error_or_None)
    if json_files:
        for j_file in json_files:
            try:
                j_file.seek(0)
                content = _json.load(j_file)
                updated_json, count, skipped_zero = _sync_broker_json(content, qty_map)
                results.append((j_file.name, _json.dumps(updated_json, indent=4), count, skipped_zero, None))
            except Exception as e:
                results.append((j_file.name, None, 0, 0, str(e)))
    n_templates = len(results)
    ok = sum(1 for _, p, _, _, _ in results if p is not None)
    total_updated = sum(c for _, p, c, _, _ in results if p is not None)
    total_skipped_zero = sum(s for _, p, _, s, _ in results if p is not None)

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
            _skip_note = (
                f" <strong>{total_skipped_zero}</strong> matched instrument(s) left untouched "
                "(curated at 0 units)."
                if total_skipped_zero > 0 else ""
            )
            render_interpretation_card(
                title="READY TO EXPORT",
                body=(
                    f"Synced <strong>{ok}/{n_templates}</strong> template(s) · "
                    f"<strong>{total_updated}</strong> instrument(s) updated from "
                    f"<strong>{tradable}</strong> tradable holding(s).{_skip_note} "
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
            for fname, payload, count, skipped_zero, err in results:
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
                    # Distinguish "matched nothing at all" from "matched, but
                    # every match was a 0-unit holding left untouched" — the
                    # old NO MATCH label conflated both (see AUDIT_DIRECTIVES.md B8).
                    if count > 0:
                        s_text = "SYNCED"
                    elif skipped_zero > 0:
                        s_text = f"SKIPPED ({skipped_zero} @ 0 units)"
                    else:
                        s_text = "NO MATCH"
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

            for fname, payload, count, skipped_zero, err in results:
                if err is None:
                    st.download_button(
                        label=f"Download updated {fname}",
                        data=payload,
                        file_name=f"updated_{fname}",
                        mime="application/json",
                        use_container_width=True,
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
                        'Non-destructive: instruments with no matching holding, or matching a '
                        'holding curated at <strong>0 units</strong>, are left '
                        '<strong>untouched</strong> rather than zeroed out — and the original '
                        'files are never modified, you download fresh copies.'
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
    from analytics import resolve_benchmark, resolve_risk_free_rate, compute_metrics
    from charts import create_benchmark_comparison_chart

    universe, selected_index, _regime, _mode = _intel_context()
    bench_ticker, bench_name = resolve_benchmark(universe, selected_index)
    RISK_FREE_RATE = resolve_risk_free_rate(bench_ticker)

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

    # ── ANCHOR = the analysis date THIS PORTFOLIO was curated under (frozen in
    #    run_context — see _intel_context's docstring), NOT the live sidebar
    #    date picker. Browsing the sidebar to a different date after a run must
    #    not silently re-anchor the already-curated book's performance window.
    #    Metrics run anchor -> today; the window is dictated by the anchor (no
    #    user timeframe picker). Handle edge cases. ──
    _run_ctx = st.session_state.get("run_context") or {}
    _sel = _run_ctx.get("anchor_date") or st.session_state.get("selected_date")
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
    # CAGR/Alpha/Calmar are all functions of an annualized return — annualizing
    # a sub-quarter window (< 60 trading days) linearly extrapolates a few
    # days of noise into a full year and is statistically indefensible (CFA
    # Institute GIPS guidance advises against annualizing periods under a
    # year). analytics.compute_metrics flags this via cagr_meaningful; hide
    # those specific cards rather than print a number that reads as precise
    # but isn't (see AUDIT_DIRECTIVES.md A18).
    _cagr_ok = m.get("cagr_meaningful", True)
    _section_divider()
    render_section_header(
        "Returns & Risk-Adjusted Performance",
        "Period, CAGR & efficiency ratios" if _cagr_ok else "Period return & efficiency ratios (CAGR hidden — window too short to annualize)",
        icon="zap", accent="emerald",
    )

    v = m.get("total_return", 0)
    _period_sub = f"CAGR: {m.get('cagr', 0):+.1f}%" if _cagr_ok else f"{len(port_returns)} trading days"
    s = m.get("sharpe", 0)
    so = m.get("sortino", 0)
    ir = m.get("info_ratio", 0)

    # (label, value, subtext, color) for each card that's always shown, in
    # order; CAGR-dependent cards (Alpha, Calmar) are spliced in only when
    # cagr_ok, so the column count and card set change together instead of
    # via error-prone manual index arithmetic.
    cards = [("Period Return", f"{v:+.2f}%", _period_sub, "success" if v >= 0 else "danger")]
    if _cagr_ok:
        a = m.get("alpha", 0)
        cards.append(("Alpha", f"{a:+.2f}%", "Excess return",
                      "success" if a > 0 else "danger" if a < 0 else "neutral"))
    cards.append(("Sharpe", f"{s:.2f}", f"Rf = {RISK_FREE_RATE*100:.1f}%",
                  "success" if s > 1 else "warning" if s > 0.5 else "danger"))
    cards.append(("Sortino", f"{so:.2f}", "Downside risk",
                  "success" if so > 1.5 else "warning" if so > 0.5 else "danger"))
    if _cagr_ok:
        ca = m.get("calmar", 0)
        cards.append(("Calmar", f"{ca:.2f}", "Return / MaxDD",
                      "success" if ca > 1 else "warning" if ca > 0.5 else "danger"))
    cards.append(("Info Ratio", f"{ir:.2f}", "Active return / TE",
                  "success" if ir > 0.5 else "warning" if ir > 0 else "danger"))

    r1 = st.columns(len(cards))
    for col, (label, value, sub, color) in zip(r1, cards):
        with col:
            render_metric_card(label, value, sub, color)

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
            description="Conviction-based portfolio curation with composite scoring across six technical signals.",
            specs=[
                ("Signals", "RSI + Osc + Z + MA + VAP + Strat"),
                ("Selection", "Top N by conviction · value-area tie-break"),
                ("Weighting", "(conviction / total) × 100"),
                ("Dispersion", "SIP + Swing modes")
            ],
            card_class="portfolio",
            icon="briefcase"
        )

    with col2:
        render_system_card(
            title="REGIME",
            description="Eight-factor market regime detection with fixed composite weights for consistent, reproducible classification.",
            specs=[
                ("Regimes", "Strong Bull · Bull · Weak Bull · Chop · Weak Bear · Bear · Crisis"),
                ("Factors", "Momentum · Trend · Breadth · Acceptance"),
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
                ("Universe", "Any selected asset class"),
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
    """Tab — per-(universe, index, regime) conviction-weight calibration (5-of-6-signal
    simplex; the sixth, Strategy Endorsement, is fixed — see intelligence.calibrate).
    The regime detector itself is NOT calibrated (fixed 8-factor weights)."""
    from intelligence import (
        IntelligencePassport, DEFAULT_WEIGHTS, DEFAULT_HORIZON,
        build_harvest, calibrate, regime_labels_from_series,
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
                        f"(RSI · OSC · Z · MA · VAP · Strategy, each 0.167 (1/6))."
                    ),
                    color="warning",
                )
            else:
                render_interpretation_card(
                    title="NOT CALIBRATED",
                    body=(
                        f"No passport for <strong>{passport.label}</strong> yet. Conviction scoring "
                        f"uses Pragyam's default weights (RSI · OSC · Z · MA · VAP · Strategy, each 0.167 (1/6)). "
                        f"Run a calibration to learn weights for this scope."
                    ),
                    color="warning",
                )

        k1, k2 = st.columns(2)
        with k1:
            if st.button("Calibrate", type="primary", use_container_width=True, key="btn_calibrate"):
                hist_window = st.session_state.get("training_data_window", [])
                if regime_name == "UNKNOWN":
                    # A passport keyed "UNKNOWN" (regime detection failed or
                    # ran on too little history) is a meaningless scope — it
                    # can never be looked up again under a real regime label,
                    # so calibrating it just burns 100 Optuna trials on a
                    # passport nothing will ever read (see
                    # AUDIT_DIRECTIVES.md D2).
                    st.error(
                        "Cannot calibrate: the current regime is UNKNOWN (detection failed or "
                        "there's too little history). A passport keyed to UNKNOWN would never be "
                        "reused under a real regime label. Pick a date/universe with a valid "
                        "regime reading first."
                    )
                elif not hist_window or len(hist_window) <= DEFAULT_HORIZON + 5:
                    st.error(
                        f"Need at least {DEFAULT_HORIZON + 5} days of history for a {DEFAULT_HORIZON}-day horizon. "
                        "Run an analysis with a longer lookback first."
                    )
                else:
                    with st.spinner(
                        f"Calibrating {passport.label} · {_CALIBRATION_LOOKBACK_FILES}-day "
                        f"estimation panel · {DEFAULT_HORIZON}-day horizon · 100 trials..."
                    ):
                        # Same estimation-panel contract as Phase 1.5: the
                        # paired beats-default gate needs >=
                        # min_calibration_dates() in-family dates, which the
                        # 100-day run panel (training_data_window) can never
                        # supply — harvesting from it made this button a
                        # guaranteed fail-fast. Fall back to the run panel
                        # only if the estimation fetch itself comes up short.
                        _anchor = (st.session_state.get("run_context") or {}).get(
                            "anchor_date"
                        ) or st.session_state.get("selected_date")
                        _end_dt = (
                            datetime.combine(_anchor, datetime.min.time())
                            if _anchor else datetime.now()
                        )
                        _cal_hist = _load_historical_data(
                            _end_dt, _CALIBRATION_LOOKBACK_FILES,
                            f"UNIVERSE:{universe}|{selected_index}",
                        )
                        if len(_cal_hist) > len(hist_window):
                            hist_for_harvest = _cal_hist
                            try:
                                _series = get_regime_history_series(_cal_hist, window_size=10, step=1)
                            except Exception:
                                _series = []
                            regime_labels = regime_labels_from_series(_series, len(_cal_hist), window_size=10)
                        else:
                            # Reuse the regime series computed during the last
                            # Run Analysis when it covers this exact window;
                            # otherwise compute it fresh so this manual
                            # calibration is also regime-conditioned (see
                            # AUDIT_DIRECTIVES.md A2).
                            hist_for_harvest = hist_window
                            _cached_series = st.session_state.get("regime_history_series")
                            if _cached_series:
                                regime_labels = regime_labels_from_series(_cached_series, len(hist_window), window_size=10)
                            else:
                                from intelligence import regime_labels_for_window
                                regime_labels = regime_labels_for_window(hist_window, window_size=10)
                        harvest = build_harvest(hist_for_harvest, horizon=DEFAULT_HORIZON, regime_labels=regime_labels)
                        if harvest.empty:
                            st.error("Harvest produced no usable rows. Indicator coverage may be too sparse.")
                        else:
                            result = calibrate(
                                universe, selected_index, regime_name,
                                harvest, n_trials=100, horizon=DEFAULT_HORIZON,
                            )
                            if not result.get("success"):
                                st.error(result.get("reason", "Calibration failed for an unknown reason."))
                            else:
                                st.success(
                                    f"Calibrated {passport.label} · Train IR {result['train_ir']:+.3f} · "
                                    f"Val IR {result['val_ir']:+.3f} (beat default {result['default_val_ir']:+.3f})"
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
        labels = [("RSI", "w_rsi"), ("Oscillator", "w_osc"), ("Z-Score", "w_z"),
                  ("MA Alignment", "w_ma"), ("Value Area", "w_vap"),
                  ("Strategy Endorsement", "w_strat")]
        for label, key in labels:
            v = weights.get(key, DEFAULT_WEIGHTS[key])
            d = DEFAULT_WEIGHTS[key]
            delta = v - d
            color = "var(--emerald)" if delta > 0.005 else ("var(--rose)" if delta < -0.005 else "var(--ink-secondary)")
            arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "—")
            # w_strat is never searched by the calibrator (no historical values
            # to evaluate — see intelligence._signals_from_row's docstring), so
            # it always sits at its default with a flat delta; label that so
            # it doesn't read as "the calibrator chose not to move this".
            label_disp = f"{label} <span class=\"iw-fixed\">(fixed)</span>" if key == "w_strat" else html_module.escape(label)
            rows_html.append(
                f'<tr>'
                f'<td class="iw-label">{label_disp}</td>'
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
                'Five of the six conviction-signal weights (RSI · Oscillator · Z-Score · MA-alignment · '
                'Value Area) are <strong>learned per (universe, index, regime)</strong> from historical '
                'signal-to-forward-return evidence — instead of the even 1/6 x6 fallback used in Standard '
                'mode. The sixth, Strategy Endorsement, stays fixed (see the tile below) since it has no '
                'historical values to evaluate. Different regimes reward different signals: bull markets '
                'favour momentum (RSI / MA), choppy markets favour mean-reversion (Z-Score / Value Area). '
                'The right conviction mix is discovered automatically and stored as a passport on disk — '
                'but only when it demonstrably beats doing nothing (see Safety rails). '
                '(The market-regime detector itself uses fixed 8-factor weights and is not calibrated.)'
            '</div>'
            '<div class="intel-method-grid">'

                '<div class="intel-method-tile tile-learns">'
                    '<div class="tile-label">What it learns</div>'
                    '<div class="tile-body">'
                        'Conviction: five weights on the 5-simplex '
                        '<code>w_rsi + w_osc + w_z + w_ma + w_vap = (1 - w_strat)</code>, each ≥ 0, via '
                        'softmax over five unconstrained scalars — a smooth, full-support landscape with '
                        'no boundary degeneracies. <strong>w_strat</strong> (Strategy Endorsement — the '
                        'cross-sectional rank of how many of the 95 strategies picked a symbol) is held '
                        'FIXED: it only exists for the live day a run executes on, so there is no '
                        'historical value to evaluate an IC against without re-running all 95 strategies '
                        'on every historical day. (The regime detector uses fixed factor weights and is '
                        'not calibrated.)'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-how">'
                    '<div class="tile-label">How</div>'
                    '<div class="tile-body">'
                        'Bayesian search via <strong>Optuna Tree-structured Parzen Estimator (TPE)</strong> '
                        'with a fixed seed for reproducibility, over dates restricted to this passport\'s '
                        'own regime family (Bull / Chop / Bear Mix) — a BEAR passport never learns from '
                        'BULL dates. 100 trials per calibration.'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-obj">'
                    '<div class="tile-label">Objective</div>'
                    '<div class="tile-body">'
                        '<strong>Information Ratio</strong> = <code>mean(IC) / std(IC)</code>, the '
                        'per-date Spearman correlation between the weighted conviction score and '
                        f'<strong>{DEFAULT_HORIZON}-day forward returns</strong>, computed on '
                        '<strong>non-overlapping</strong> dates only (adjacent overlapping windows share '
                        f'{DEFAULT_HORIZON}-1 of their {DEFAULT_HORIZON} forward-return days, which '
                        'inflates a naive IR).'
                    '</div>'
                '</div>'

                '<div class="intel-method-tile tile-safety">'
                    '<div class="tile-label">Safety rails</div>'
                    '<div class="tile-body">'
                        '<strong>50/50 embargoed train-val split</strong> (the last horizon train dates are '
                        'dropped so no label window overlaps validation). A passport is saved only when '
                        'the learned weights beat the default baseline by a '
                        '<strong>statistically significant margin</strong> — a paired significance test on '
                        'validation-date IC differences, not a raw IR comparison — and the deployed weights '
                        'are <strong>shrunk toward the default</strong> in proportion to how strongly that '
                        'edge is corroborated out-of-sample. Falls back to defaults, with the specific '
                        'reason stated, when history is too short, the regime hasn\'t occurred often enough, '
                        'or the edge isn\'t significant.'
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
    _render_footer()


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
    # Reset the run clock: the tracker is per-SESSION (see metrics.get_metrics),
    # so without this the summary's "Total Duration" reports time since the
    # session's first run, not this run's wall time.
    import time as _time
    metrics.start_time, metrics.end_time = _time.time(), 0.0
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
        #
        # Single progress bar contract: `progress_container` is the ONE
        # progress surface for the whole run. Every milestone below renders
        # into it with a STRICTLY NON-DECREASING percentage — the bands are
        # Phase 1 (Data & Regime) 0-20, Phase 1.5 (Intelligence) 20-35,
        # Phase 2 (Strategies & Curation) 35-100 — so the bar can never move
        # backwards regardless of which Phase 1.5 branch executes. Labels are
        # Title Case; subs carry the load-bearing datum for that milestone.
        progress_bar(progress_container, 2, "Fetching Market Data", f"yfinance · {len(symbols_list)} symbols")
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

        progress_bar(progress_container, 14, "Data Loaded", f"{len(all_hist)} days · {len(symbols_list)} symbols")

        # Regime detection — pass intelligence context so the 8 factor weights are
        # the learned ones (Intelligence mode) or the shared defaults (Standard).
        progress_bar(progress_container, 16, "Detecting Market Regime", "8-factor composite scoring")
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

        # Mirror the Phase 1 milestone to the terminal so the console trace
        # carries the same checkpoints the progress bar showed.
        log.section("Data & Regime", phase="PHASE 1")
        log.item("Historical Panel", f"{len(all_hist)} trading days · {len(symbols_list)} symbols")
        log.item("Market Regime", f"{regime_name.replace('_', ' ')} · {confidence:.0%} confidence")

        progress_bar(
            progress_container, 20, "Phase 1 Complete",
            f"{regime_name.replace('_', ' ')} regime · {confidence:.0%} confidence",
        )

        # Regime history series — computed ONCE here (moved up from its old
        # position after Phase 2) so Phase 1.5 can condition calibration on
        # the regime actually in effect at each historical date (see
        # AUDIT_DIRECTIVES.md A2: a passport keyed "BEAR" was previously
        # estimated on the ENTIRE trailing window regardless of regime, since
        # the regime series wasn't available until after calibration ran).
        # Cached in session_state so the Regime tab's chart reuses this exact
        # computation instead of recomputing it.
        try:
            _regime_series_for_harvest = get_regime_history_series(all_hist, window_size=10, step=1)
        except Exception:
            _regime_series_for_harvest = []
        st.session_state.regime_history_series = _regime_series_for_harvest

        # PHASE 1.5: Intelligence — calibrate weights for the (universe, index, regime)
        # tuple on first encounter; reuse the saved passport on subsequent runs.
        # Calibration outcome is mirrored to st.session_state.last_intel_outcome
        # so the sidebar passport card and result page can show what happened.
        if st.session_state.get("intelligence_mode"):
            from intelligence import (
                IntelligencePassport, build_harvest, calibrate, DEFAULT_HORIZON,
                regime_labels_from_series,
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
                    progress_container, 35, "Intelligence Ready",
                    f"{_passport.label} · Val IR {_m['val_ir']:+.3f} · "
                    f"Calibrated {_passport.last_calibrated}",
                )
            elif regime_name == "UNKNOWN":
                # A passport keyed "UNKNOWN" is a meaningless scope — it can
                # never be looked up again under a real regime label once
                # detection succeeds, so calibrating it would just burn 100
                # Optuna trials on a passport nothing will ever read (see
                # AUDIT_DIRECTIVES.md D2).
                _outcome.update({
                    "status": "skipped",
                    "reason": "Regime is UNKNOWN (detection failed or insufficient history) — "
                              "a passport keyed to UNKNOWN would never be reused.",
                })
                progress_bar(
                    progress_container, 35, "Intelligence Skipped",
                    "Regime UNKNOWN · Using default weights",
                )
            elif len(all_hist) <= DEFAULT_HORIZON + 5:
                _outcome.update({
                    "status": "skipped",
                    "reason": f"Need >{DEFAULT_HORIZON + 5} days of history (have {len(all_hist)}).",
                })
                progress_bar(
                    progress_container, 35, "Intelligence Skipped",
                    f"Need >{DEFAULT_HORIZON + 5} days of history · Using default weights",
                )
            else:
                # Estimation panel: fetched at _CALIBRATION_LOOKBACK_FILES, NOT
                # the run's _REGIME_LOOKBACK_FILES display panel. The paired
                # beats-default gate needs >= intelligence.min_calibration_dates()
                # (142 at the defaults) usable dates INSIDE the target regime
                # family — a 100-day panel can never supply that (90 harvest
                # dates → at most 5 non-overlapping paired validation dates vs
                # the 8 required, structurally unreachable). Cached like every
                # other panel and fetched only on this branch, so scopes with
                # an existing passport never pay for it.
                progress_bar(
                    progress_container, 22, "Building Estimation Panel",
                    f"{_CALIBRATION_LOOKBACK_FILES} trading days · {_passport.label}",
                )
                _cal_hist = _load_historical_data(
                    selected_date, _CALIBRATION_LOOKBACK_FILES, symbols_key
                )
                if len(_cal_hist) > len(all_hist):
                    # Regime series over the estimation panel — the run panel's
                    # cached series only labels the trailing 100 days.
                    try:
                        _cal_series = get_regime_history_series(_cal_hist, window_size=10, step=1)
                    except Exception:
                        _cal_series = []
                else:
                    # Estimation fetch failed or added nothing beyond the run
                    # panel — degrade to the run panel; calibrate()'s
                    # reachability check reports the shortfall honestly
                    # instead of hard-failing Phase 1.5 here.
                    _cal_hist = all_hist
                    _cal_series = _regime_series_for_harvest
                progress_bar(
                    progress_container, 26, "Building Signal-Return Panel",
                    f"{_passport.label} · {DEFAULT_HORIZON}-day horizon",
                )
                # Tag each harvested date with the regime family in effect at
                # that date, so calibrate() conditions on regime (A2) rather
                # than learning one unconditional weight set for the whole
                # lookback and labeling it with whatever regime is current now.
                _regime_labels = regime_labels_from_series(
                    _cal_series, len(_cal_hist), window_size=10
                )
                _harvest = build_harvest(_cal_hist, horizon=DEFAULT_HORIZON, regime_labels=_regime_labels)
                if _harvest.empty:
                    _outcome.update({
                        "status": "skipped",
                        "reason": "Harvest produced no usable rows (sparse indicators).",
                    })
                    progress_bar(
                        progress_container, 35, "Intelligence Skipped",
                        "Harvest produced no usable rows · Using default weights",
                    )
                else:
                    _n_dates = _harvest["date"].nunique()
                    _n_obs = len(_harvest)
                    _best_ir = [float("-inf")]
                    progress_bar(
                        progress_container, 28, "Calibrating Intelligence",
                        f"Optuna TPE · {_n_dates} dates · {_n_obs:,} (date, symbol) rows",
                    )

                    def _intel_cb(trial: int, total: int, score: float):
                        if score > _best_ir[0]:
                            _best_ir[0] = score
                        # Trials sweep 28 → 34, keeping the outcome milestone
                        # (35) strictly ahead of every trial update.
                        pct = 28 + int((trial / max(total, 1)) * 6)
                        best = _best_ir[0]
                        best_str = f"{best:+.3f}" if best > float("-inf") else "—"
                        progress_bar(
                            progress_container, pct, "Calibrating Intelligence",
                            f"Trial {trial}/{total} · Best IR {best_str}",
                        )

                    _result = calibrate(
                        _universe, _selected_index, regime_name,
                        _harvest, n_trials=100,
                        horizon=DEFAULT_HORIZON, progress_callback=_intel_cb,
                    )
                    if not _result.get("success"):
                        _reason = _result.get("reason", "Calibration failed for an unknown reason.")
                        _outcome.update({
                            "status": "failed",
                            "reason": _reason,
                        })
                        progress_bar(
                            progress_container, 35, "Intelligence Skipped",
                            f"{_reason} · Using default weights",
                        )
                    else:
                        _outcome.update({
                            "status": "calibrated",
                            "reason": _result.get("reason", "Optimized weights beat the default baseline out-of-sample."),
                            "train_ir": _result["train_ir"],
                            "val_ir":   _result["val_ir"],
                            "n_train_dates": _result["n_train_dates"],
                            "n_val_dates":   _result["n_val_dates"],
                        })
                        progress_bar(
                            progress_container, 35, "Intelligence Calibrated",
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
                        _log_intel_outcome(_outcome)
                        st.rerun()

            st.session_state.last_intel_outcome = _outcome
            _log_intel_outcome(_outcome)
        else:
            _outcome = {"status": "disabled", "reason": "Intelligence Mode is off — using default weights."}
            st.session_state.last_intel_outcome = _outcome
            _log_intel_outcome(_outcome)

        # PHASE 2: CONVICTION-BASED CURATION
        # The strategy count is not known until discovery runs, so this first
        # milestone claims only what is true at this point (the previous label
        # hardcoded "95 strategies" before any strategy had been discovered).
        progress_bar(progress_container, 36, "Discovering Strategies", f"{len(symbols_list)} symbols in scope")
        metrics.start_phase("conviction_curation")

        try:
            strategies = discover_strategies()
            strategies_to_run = {name: strategies[name] for name in strategies if name != "System_Curated"}

            if not strategies_to_run:
                st.error("No strategies available.")
                metrics.end_phase("conviction_curation", success=False, error_msg="Empty strategies")
                st.stop()

            log.section("Conviction Curation", phase="PHASE 2")
            log.item("Strategies", f"{len(strategies_to_run)} discovered")

            # Aggregate holdings + strategy endorsement votes.
            #
            # Every strategy's _allocate_portfolio (strategies.py) returns ALL
            # input rows (weights merely clipped to [1%,10%], never filtered),
            # so the union of "candidates" across 95 strategies was previously
            # ~= the whole data-valid universe: the strategy layer contributed
            # zero selectivity to the final book (see AUDIT_DIRECTIVES.md A4).
            # To make the layer real, each strategy's TOP QUARTILE by its own
            # weightage_pct is treated as that strategy's actual "conviction
            # pick" — an endorsement — rather than every row it returns.
            # strat_votes (how many of the 95 strategies picked a symbol in
            # their own top quartile) becomes a sixth conviction signal
            # (see regime.compute_conviction_signals / intelligence.py).
            aggregated_holdings = {}
            strat_votes: Dict[str, int] = {}
            failed_strategies = 0
            _n_strats = len(strategies_to_run)
            progress_bar(progress_container, 38, "Running Strategies", f"{_n_strats} strategies · {len(symbols_list)} symbols")

            for _si, (name, strategy) in enumerate(strategies_to_run.items()):
                try:
                    port = strategy.generate_portfolio(st.session_state.current_df, capital)
                    if port.empty:
                        continue
                    # Top quartile by this strategy's own weighting (min 1 pick
                    # so a tiny candidate set still casts a vote).
                    n_pick = max(1, len(port) // 4)
                    top_picks = port.nlargest(n_pick, "weightage_pct")
                    top_syms = set(top_picks["symbol"])
                    for _, row in port.iterrows():
                        symbol = row["symbol"]
                        price = row["price"]
                        if symbol not in aggregated_holdings:
                            aggregated_holdings[symbol] = {"price": price, "weight": 1.0}
                    for sym in top_syms:
                        strat_votes[sym] = strat_votes.get(sym, 0) + 1
                except Exception as e:
                    failed_strategies += 1
                    try:
                        log.warning(f"Strategy '{name}' failed: {type(e).__name__}: {e}")
                    except Exception:
                        pass
                # Live progress across the strategy loop — the longest Phase 2
                # stretch. Sweeps 38 → 65 (every 10th strategy plus the last),
                # staying strictly below the 68% conviction milestone.
                if _si % 10 == 9 or _si == _n_strats - 1:
                    progress_bar(
                        progress_container, 38 + int(((_si + 1) / _n_strats) * 27),
                        "Running Strategies",
                        f"Strategy {_si + 1}/{_n_strats} · {len(aggregated_holdings)} candidates",
                    )

            if failed_strategies:
                metrics.add_warning(f"{failed_strategies}/{len(strategies_to_run)} strategies failed to generate a portfolio")

            if not aggregated_holdings:
                st.error("No holdings generated.")
                metrics.end_phase("conviction_curation", success=False, error_msg="Empty holdings")
                st.stop()

            # Attach vote counts to current_df so compute_conviction_signals can
            # read them like any other cross-sectional indicator column. This is
            # the SAME current_df object read by the result tabs on re-render
            # (st.session_state.current_df), so strat_signal stays consistent
            # across the whole session for this run, not just this call.
            st.session_state.current_df = st.session_state.current_df.copy()
            st.session_state.current_df["strat_votes"] = (
                st.session_state.current_df["symbol"].map(strat_votes).fillna(0).astype(int)
            )

            log.item("Candidates", f"{len(aggregated_holdings)} aggregated from {_n_strats} strategies")
            progress_bar(progress_container, 68, "Computing Conviction", f"{len(aggregated_holdings)} candidates · 6-signal blend")

            # Conviction-based weighting with style-aware dispersion
            # SIP: conviction**2.5 | Swing: conviction**4.5 (continuous power-law
            # concentration — see portfolio.py module docstring)
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

            # Effective bounds may differ from the nominal 1%/10% when
            # num_positions makes them infeasible (e.g. 5 positions can't all
            # sit under a 10% cap) — see portfolio.compute_conviction_based_weights.
            # Surface what was ACTUALLY applied for the System tab / any UI that
            # states the bounds, instead of always printing the nominal request.
            st.session_state.min_pos_pct_eff = st.session_state.portfolio.attrs.get(
                "min_pos_pct_eff", st.session_state.min_pos_pct
            )
            st.session_state.max_pos_pct_eff = st.session_state.portfolio.attrs.get(
                "max_pos_pct_eff", st.session_state.max_pos_pct
            )

            # Freeze the scope this portfolio was curated under. Result tabs
            # (_intel_context) read this instead of live sidebar state, so
            # browsing the sidebar after a run (which recomputes the regime
            # card and mutates selected_universe/index) can never re-score an
            # already-curated portfolio under a different scope's passport, or
            # resolve the Analytics benchmark against the wrong universe (see
            # AUDIT_DIRECTIVES.md A12).
            st.session_state.run_context = {
                "universe": _u,
                "selected_index": _idx,
                "regime_name": regime_name,
                "mode": _mode,
                "anchor_date": selected_date_display,
            }

            log.item("Positions", f"{len(st.session_state.portfolio)} curated")
            progress_bar(progress_container, 90, "Portfolio Curated", f"{len(st.session_state.portfolio)} positions")

            # End conviction_curation phase tracking
            metrics.end_phase("conviction_curation", success=True)

            # Update metrics counters
            metrics.symbols_count = len(aggregated_holdings)
            metrics.strategies_count = len(strategies_to_run)
            metrics.portfolios_generated = len(st.session_state.portfolio)

            # Regime history was already computed and cached before Phase 1.5
            # (st.session_state.regime_history_series) so it could condition
            # the calibration harvest on regime — no need to recompute it here.

            metrics.end_phase("total_execution", success=True)
            progress_bar(progress_container, 100, "Analysis Complete", f"{len(st.session_state.portfolio)} positions ready")

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

            # Clear the progress card immediately — the toast below plus the
            # rendered result page are feedback enough; a blocking sleep here
            # just makes every single run 1.5s slower for no functional
            # benefit (see AUDIT_DIRECTIVES.md B7).
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


# _render_footer (defined above) is the single source of truth for the app
# footer — the result-page footer previously duplicated the same markup
# inline with a slightly different timestamp construction (see
# AUDIT_DIRECTIVES.md C5.2); _render_results calls _render_footer() instead.


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

        # Lazy regime card: auto-compute WITHOUT a spinner/blocking fetch only
        # when this exact (date, universe) combo is the one the last
        # completed Run Analysis used — that combo is guaranteed already in
        # Streamlit's cache (@st.cache_data(ttl=3600) on _load_historical_data
        # / _detect_regime_cached), so the call below resolves instantly. Any
        # OTHER combo (the user browsing to a new date/universe without
        # having run it yet) is a probable cache MISS: auto-computing there
        # used to trigger a full synchronous yfinance multi-symbol download
        # inside the sidebar's render path — 10-30s of a frozen sidebar just
        # for looking around (see AUDIT_DIRECTIVES.md C4). Show a simple
        # "awaiting first run" state instead — no manual refresh control:
        # clicking Run Analysis always computes and stores the regime as part
        # of Phase 1, so the card self-resolves on the next rerun with no
        # user action beyond the button they were already going to click.
        _last_run_ctx = st.session_state.get("run_context")
        _likely_cached = (
            _last_run_ctx is not None
            and _last_run_ctx.get("anchor_date") == selected_date
            and _last_run_ctx.get("universe") == universe
            and _last_run_ctx.get("selected_index") == selected_index
        )

        if regime_needs_update and _likely_cached:
            rd = _detect_regime_cached(selected_date_obj, symbols_key)
            st.session_state.regime_result_dict = rd
            st.session_state.suggested_mix = rd.get("mix_name", "Chop/Consolidate Mix")
            st.session_state.regime_date = selected_date
            st.session_state.regime_symbols_key = symbols_key
            regime_needs_update = False

        if rd and isinstance(rd, dict) and not regime_needs_update:
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
        elif regime_needs_update:
            st.markdown("""
            <div style="background:rgba(148,163,184,0.06); border:1px solid rgba(148,163,184,0.18);
                        border-radius:10px; padding:12px; margin:var(--sp-6) 0 var(--sp-3) 0;">
                <div style="color:var(--ink-tertiary); font-size:0.7rem; text-transform:uppercase;
                            letter-spacing:0.5px; font-weight:600; margin-bottom:4px; font-family:var(--data);">Market Regime</div>
                <div style="color:var(--ink-tertiary); font-size:0.8rem; font-family:var(--data);">
                    Run Analysis to detect the market regime for this date and universe.
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
            help="Use calibrated conviction weights learned from forward returns — five of "
                 "the six conviction signals per (universe, index, regime); Strategy "
                 "Endorsement stays fixed (no historical values to calibrate against). "
                 "When OFF, Pragyam uses the even 1/6 x6 conviction weights. (The regime "
                 "detector always uses fixed factor weights — it is not calibrated.)",
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

                    # Reject defaults-shape exports outright: the sidebar's own
                    # Export button writes is_calibrated=False + zero IRs when
                    # no real passport exists for the scope (see the Export
                    # Profile section below) — re-importing that file must not
                    # silently mark the scope "Calibrated" with meaningless
                    # IR=0.000 (see AUDIT_DIRECTIVES.md A16).
                    if payload.get("is_calibrated") is False:
                        raise ValueError(
                            "this file contains default (uncalibrated) weights, exported when no "
                            "passport existed for its scope — nothing to import"
                        )

                    weights_raw = payload.get("weights", payload)
                    if not all(k in weights_raw for k in DEFAULT_WEIGHTS):
                        raise ValueError("missing one of " + " / ".join(DEFAULT_WEIGHTS.keys()))

                    # Numeric + simplex validation: a malformed/adversarial
                    # file (e.g. {"w_rsi": 37, ...}) previously imported with
                    # only a key-presence check, silently corrupting live
                    # conviction scoring for this scope.
                    weights_num: Dict[str, float] = {}
                    for k in DEFAULT_WEIGHTS:
                        try:
                            v = float(weights_raw[k])
                        except (TypeError, ValueError):
                            raise ValueError(f"{k} is not a number")
                        if not (0.0 <= v <= 1.0):
                            raise ValueError(f"{k}={v} is outside the valid [0, 1] range")
                        weights_num[k] = v
                    total = sum(weights_num.values())
                    if abs(total - 1.0) > 0.02:
                        raise ValueError(f"weights sum to {total:.4f}, expected ~1.0 (a simplex)")
                    # Renormalize to exactly 1.0 (absorb float/rounding drift
                    # from the source file rather than rejecting on it).
                    weights_num = {k: v / total for k, v in weights_num.items()}

                    _pp_passport.save(
                        weights=weights_num,
                        train_ir=float(payload.get("train_ir", 0.0)),
                        val_ir  =float(payload.get("val_ir",   0.0)),
                        n_train_dates=int(payload.get("n_train_dates", 0)),
                        n_val_dates  =int(payload.get("n_val_dates",   0)),
                        n_trials=int(payload.get("n_trials", 0)),
                        horizon =int(payload.get("horizon", 10)),
                        is_calibrated=True,
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
        # The sidebar painted BEFORE this run executed, so its regime card and
        # passport card still show pre-run state ("Run Analysis to detect...")
        # — the freshly-computed regime sits in session_state but nothing
        # repaints the sidebar in this script run. Rerun once (the same idiom
        # Phase 1.5 uses after a successful calibration): the flags above are
        # already popped so this cannot loop, every panel the repaint touches
        # is now cached (regime/data/portfolio all in session_state or
        # st.cache_data), the sidebar repaints from fresh state, and the
        # result page renders via st.session_state.portfolio.
        st.rerun()

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
