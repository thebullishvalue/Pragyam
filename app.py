"""
PRAGYAM (प्रज्ञम) — Portfolio Intelligence  |  A @thebullishvalue Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conviction-based portfolio curation with 80+ quantitative strategies.

Architecture:
  regime.py         → MarketRegimeDetector, compute_conviction_signals
  portfolio.py      → compute_conviction_based_weights()
  backdata.py       → generate_historical_data()
  charts.py         → Plotly chart builders
  strategies.py     → 80+ BaseStrategy implementations

Pipeline (2 phases):
  Phase 1: Data fetching + regime detection
  Phase 2: Conviction-based portfolio curation (ALL strategies)

Version: 7.0.5
Author: @thebullishvalue
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import warnings
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Imports ────────────────────────────────────────────────────────────────────
from logger_config import console, get_console
log = get_console()

from metrics import get_metrics
from regime import (
    MarketRegimeDetector,
    REGIME_COLORS,
    REGIME_ICONS,
    REGIME_DESCRIPTIONS,
    get_regime_history_series,
    compute_conviction_signals,
)
from strategies import BaseStrategy, discover_strategies
from backdata import (
    generate_historical_data,
    load_symbols_from_file,
    MAX_INDICATOR_PERIOD,
    SYMBOLS_UNIVERSE,
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

VERSION = "v7.0.5"
PRODUCT_NAME = "Pragyam"
COMPANY = "@thebullishvalue"

st.set_page_config(
    page_title="PRAGYAM | Portfolio Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load CSS
_css_path = os.path.join(os.path.dirname(__file__), "style.css")
try:
    with open(_css_path, encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass


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
        "debug_info": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _load_historical_data(end_date: datetime, lookback_files: int) -> List[Tuple[datetime, pd.DataFrame]]:
    """Fetch and cache historical indicator snapshots from yfinance."""
    try:
        return generate_historical_data(
            symbols_to_process=SYMBOLS_UNIVERSE,
            start_date=end_date - timedelta(days=int((lookback_files + MAX_INDICATOR_PERIOD) * 1.5) + 30),
            end_date=end_date,
        )
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _detect_regime_cached(end_date: datetime) -> Dict:
    """Detect market regime and return as serializable dict."""
    detector = MarketRegimeDetector()
    window_days = int(MAX_INDICATOR_PERIOD * 1.5) + 30
    try:
        hist = generate_historical_data(
            symbols_to_process=SYMBOLS_UNIVERSE,
            start_date=end_date - timedelta(days=window_days),
            end_date=end_date,
        )
        if len(hist) < 5:
            return {
                "regime": "UNKNOWN",
                "mix_name": "Chop/Consolidate Mix",
                "confidence": 0.30,
                "composite_score": 0.0,
                "explanation": "Insufficient data for regime classification.",
                "color": REGIME_COLORS["UNKNOWN"],
                "icon": "❓",
                "description": "",
            }
        result = detector.detect(hist, analysis_date=end_date)
        return result.to_dict()
    except Exception as e:
        return {
            "regime": "UNKNOWN",
            "mix_name": "Chop/Consolidate Mix",
            "confidence": 0.30,
            "composite_score": 0.0,
            "explanation": f"Regime detection error: {e}",
            "color": "#6b7280",
            "icon": "❓",
            "description": "",
        }


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


def _metric_card(label: str, value: str, sub: str = "", cls: str = "neutral") -> str:
    """Generate metric card HTML."""
    sub_html = f"<div class='sub-metric'>{sub}</div>" if sub else ""
    return f"<div class='metric-card {cls}'><h4>{label}</h4><h2>{value}</h2>{sub_html}</div>"


def _styled_table(df: pd.DataFrame, format_dict: Dict[str, str] = None) -> str:
    """Render styled HTML table from DataFrame."""
    df_s = df.copy()
    rename = {}
    for col in df_s.columns:
        f = col.replace("_", " ").replace("pct", "%").replace("dd", "DD").replace("hhi", "HHI").title()
        f = f.replace("Cagr", "CAGR").replace("Dd", "DD").replace("Hhi", "HHI")
        rename[col] = f
    df_s = df_s.rename(columns=rename)
    styled = df_s.style
    if format_dict:
        new_fmt = {rename.get(k, k): v for k, v in format_dict.items()}
        styled = styled.format(new_fmt, na_rep="—")
    return styled.set_table_attributes('class="stMarkdown table"').hide(axis="index").to_html()


# ══════════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_portfolio_tab(portfolio: pd.DataFrame, current_df: pd.DataFrame, capital: float):
    """Tab 1 — Curated portfolio with conviction signal overlay."""
    st.markdown(_section_header(
        "Curated Portfolio Holdings",
        f"{len(portfolio)} positions · conviction-based curation"
    ), unsafe_allow_html=True)

    portfolio_with_signals = compute_conviction_signals(portfolio, current_df)

    # Portfolio table
    display_df = portfolio[["symbol", "price", "units", "weightage_pct", "value"]].rename(columns={
        "symbol": "Symbol",
        "price": "Price (₹)",
        "units": "Units",
        "weightage_pct": "Weight %",
        "value": "Value (₹)",
    })
    styled = display_df.style.format({
        "Price (₹)": "{:,.2f}",
        "Value (₹)": "{:,.2f}",
        "Units": "{:,.0f}",
        "Weight %": "{:.2f}%",
    }).set_table_attributes('class="stMarkdown table"').hide(axis="index")
    st.markdown(styled.to_html(), unsafe_allow_html=True)

    # Conviction Signal Heatmap
    _section_divider()
    st.markdown(_section_header(
        "Conviction Signals",
        "Real-time indicator alignment — RSI · Oscillator · Z-Score · MA Alignment"
    ), unsafe_allow_html=True)

    if CHARTS_AVAILABLE and not portfolio_with_signals.empty:
        fig_conv = create_conviction_heatmap(portfolio_with_signals)
        st.plotly_chart(fig_conv, width='stretch', key="tab1_conviction_heatmap")
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


def _render_position_guide_tab(portfolio: pd.DataFrame, current_df: pd.DataFrame):
    """Tab — Position Guide with entry conditions and conviction signals."""
    st.markdown(_section_header(
        "Position Guide",
        "Entry conditions and conviction summary for all holdings"
    ), unsafe_allow_html=True)

    portfolio_with_signals = compute_conviction_signals(portfolio, current_df)

    if "rsi_signal" not in portfolio_with_signals.columns:
        st.info("Position guide signals unavailable.")
        return

    guide_rows = []
    for _, row in portfolio_with_signals.iterrows():
        score = row.get("conviction_score", 50)
        if score >= 65:
            signal_text, signal_cls = "Strong Buy", "🟢"
        elif score >= 50:
            signal_text, signal_cls = "Buy", "🟩"
        elif score >= 35:
            signal_text, signal_cls = "Hold", "🟡"
        else:
            signal_text, signal_cls = "Caution", "🔴"

        # Format values to 2 decimal places
        rsi_val = row.get("rsi_value")
        osc_val = row.get("osc_value")
        z_val = row.get("zscore_value")

        guide_rows.append({
            "Symbol": row["symbol"],
            "Weight": f"{row['weightage_pct']:.2f}%",
            "RSI": f"{rsi_val:.2f}" if rsi_val is not None and not pd.isna(rsi_val) else "—",
            "Osc": f"{osc_val:.2f}" if osc_val is not None and not pd.isna(osc_val) else "—",
            "Z": f"{z_val:.2f}" if z_val is not None and not pd.isna(z_val) else "—",
            "MA": f"{int(row['ma_count'])}/5" if pd.notna(row.get("ma_count")) else "—",
            "Conviction": f"{int(score)}/100",
            "Signal": f"{signal_cls} {signal_text}",
            "conviction_score_raw": float(score),  # Store raw score for counting
        })

    if guide_rows:
        gdf = pd.DataFrame(guide_rows).drop(columns=["conviction_score_raw"])
        st.markdown(_styled_table(gdf), unsafe_allow_html=True)
        st.caption("MA = aligned moving averages out of 5 conditions.")

    _section_divider()

    # Summary Statistics
    st.markdown(_section_header("Signal Distribution", "Portfolio conviction breakdown"), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    # Count based on raw conviction score thresholds (matches the signal assignment logic)
    scores = [row["conviction_score_raw"] for row in guide_rows]
    strong_buy = sum(1 for s in scores if s >= 65)
    buy = sum(1 for s in scores if 50 <= s < 65)
    hold = sum(1 for s in scores if 35 <= s < 50)
    caution = sum(1 for s in scores if s < 35)

    with c1:
        st.markdown(_metric_card("Strong Buy", str(strong_buy), "High conviction (≥65)", "success"), unsafe_allow_html=True)
    with c2:
        st.markdown(_metric_card("Buy", str(buy), "Moderate conviction (50-64)", "primary"), unsafe_allow_html=True)
    with c3:
        st.markdown(_metric_card("Hold", str(hold), "Neutral (35-49)", "warning"), unsafe_allow_html=True)
    with c4:
        st.markdown(_metric_card("Caution", str(caution), "Low conviction (<35)", "danger"), unsafe_allow_html=True)


def _render_regime_tab(regime_result: Dict, regime_series: List, training_data: List = None):
    """Tab 2 — Market regime analysis."""
    st.markdown(_section_header(
        "Regime Intelligence",
        "7-factor composite scoring · 30-day history"
    ), unsafe_allow_html=True)

    if not regime_result:
        st.info("Run analysis to populate regime data.")
        return

    regime_name = regime_result.get("regime", "UNKNOWN")
    mix_name = regime_result.get("mix_name", "—")
    confidence = regime_result.get("confidence", 0.0)
    score = regime_result.get("composite_score", 0.0)
    color = regime_result.get("color", "#888888")
    icon = regime_result.get("icon", "❓")
    description = regime_result.get("description", "")
    explanation = regime_result.get("explanation", "")
    factors_raw = regime_result.get("factors", {})

    # Current Regime Banner
    st.markdown(_section_header("Current Market Regime", "10-day indicator window"), unsafe_allow_html=True)

    col_badge, col_factors = st.columns([1, 2])

    with col_badge:
        st.markdown(f"""
        <div class="regime-badge" style="border-color: {color}; background: {color}18;">
            <div class="regime-icon">{icon}</div>
            <div class="regime-name" style="color: {color};">{regime_name.replace('_', ' ')}</div>
            <div class="regime-sub">{mix_name}</div>
            <div class="regime-score">Score: {score:+.2f}</div>
            <div class="regime-conf">
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{confidence*100:.0f}%; background:{color};"></div>
                </div>
                <span style="color:{color}; font-size:0.85rem; font-weight:700;">{confidence:.0%} confidence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_factors:
        st.markdown("**Factor Scores** (−2 bearish → +2 bullish):")
        factor_order = [
            ("momentum", "Momentum (30%)", "strength"),
            ("trend", "Trend (25%)", "quality"),
            ("breadth", "Breadth (15%)", "quality"),
            ("velocity", "Velocity (15%)", "acceleration"),
            ("extremes", "Extremes (10%)", "type"),
            ("volatility", "Volatility (5%)", "regime"),
        ]
        for fkey, fname, label_key in factor_order:
            fd = factors_raw.get(fkey, {})
            fs = fd.get("score", 0.0)
            fl = fd.get(label_key, "—")
            pct = max(0, min(100, int((fs + 2.0) / 4.0 * 100.0)))
            bar_color = "#10b981" if fs >= 0.5 else ("#ef4444" if fs <= -0.5 else "#f59e0b")
            st.markdown(f"""
            <div style="margin: 5px 0;">
              <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:3px;">
                <span style="color:#EAEAEA; font-weight:600;">{fname}</span>
                <span style="color:#888;">{fl} <span style="color:{bar_color}; font-weight:700;">({fs:+.1f})</span></span>
              </div>
              <div style="height:6px; background:#2A2A2A; border-radius:3px;">
                <div style="width:{pct}%; height:100%; background:{bar_color}; border-radius:3px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Regime History
    regime_series_to_use = regime_series
    if regime_series_to_use is None and training_data and len(training_data) >= 10:
        with st.spinner("Computing regime history…"):
            regime_series_to_use = get_regime_history_series(training_data, window_size=10, step=1)
        st.session_state.regime_history_series = regime_series_to_use

    if regime_series_to_use and len(regime_series_to_use) > 0:
        _section_divider()
        st.markdown(_section_header("Regime Score History", "Rolling 10-day composite"), unsafe_allow_html=True)

        regimes_seq = [r.regime for r in regime_series_to_use]
        transitions = sum(1 for i in range(1, len(regimes_seq)) if regimes_seq[i] != regimes_seq[i-1])
        last_regime = regimes_seq[-1] if regimes_seq else "—"
        prev_regime = regimes_seq[-2] if len(regimes_seq) > 1 else "—"

        if CHARTS_AVAILABLE:
            fig_rh = create_regime_history_chart(regime_series_to_use)
            st.plotly_chart(fig_rh, width='stretch', key="tab2_regime_history")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_metric_card("Transitions", str(transitions), "Over analysis window", "info"), unsafe_allow_html=True)
        with c2:
            st.markdown(_metric_card("Current", last_regime.replace("_", " "), "Latest", "primary"), unsafe_allow_html=True)
        with c3:
            st.markdown(_metric_card("Prior", prev_regime.replace("_", " "), "Previous", "neutral"), unsafe_allow_html=True)


def _render_system_tab(training_window: List):
    """Tab 3 — System information."""
    st.markdown(_section_header("System & Execution Details", "Configuration and metadata"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_metric_card("Analysis Date", str(st.session_state.get("selected_date", "N/A")), "Portfolio date", "primary"), unsafe_allow_html=True)
    with c2:
        portfolio_val = st.session_state.get("portfolio")
        num_positions = len(portfolio_val) if portfolio_val is not None and not portfolio_val.empty else 0
        st.markdown(_metric_card("Positions", str(num_positions), "Current holdings", "info"), unsafe_allow_html=True)

    _section_divider()

    st.markdown(_section_header("Configuration", "System settings"), unsafe_allow_html=True)

    details = {
        "Version": VERSION,
        "Curation Method": "Conviction-Based (All 80+ Strategies)",
        "Weight Formula": "(conviction / total) × 100",
        "Min Position": f"{st.session_state.min_pos_pct*100:.1f}%",
        "Max Position": f"{st.session_state.max_pos_pct*100:.1f}%",
        "Data Source": "yfinance (NSE)",
        "Lookback Period": f"{len(training_window)} days",
    }

    details_df = pd.DataFrame([{"Setting": k, "Value": v} for k, v in details.items()])
    st.markdown(_styled_table(details_df), unsafe_allow_html=True)

    _section_divider()

    st.markdown(_section_header("Technical Information", "Conviction scoring and weighting"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class='metric-card primary'>
            <h4>📊 Conviction Scoring</h4>
            <h2 style='font-size:1.1rem; font-weight:600;'>0-100 Range</h2>
            <div class='sub-metric'>
                <strong>Signals:</strong> RSI (30%) · Oscillator (30%)<br>
                Z-Score (20%) · MA Alignment (20%)<br><br>
                <strong>Formula:</strong> (raw + 2) / 4 × 100
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='metric-card success'>
            <h4>⚖️ Portfolio Weighting</h4>
            <h2 style='font-size:1.1rem; font-weight:600;'>(conviction / total) × 100</h2>
            <div class='sub-metric'>
                <strong>Bounds:</strong> Min 1% · Max 10%<br>
                <strong>Selection:</strong> Top 30 by conviction<br><br>
                <strong>No threshold:</strong> All symbols eligible
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def _render_landing_page():
    """Render landing page when no portfolio is available."""
    st.markdown(f"""
    <div class="premium-header">
        <h1>PRAGYAM | Portfolio Intelligence</h1>
        <div class="tagline">Conviction-Based Curation · All 95 Strategies · Live NSE Data</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box welcome'>
        <h4>Welcome to Pragyam</h4>
        <p>
            Institutional-grade portfolio curation for Indian markets.
            Pure conviction-based approach: all 95 strategies run → every candidate scored → top 30 selected.
        </p>
        <strong>To begin:</strong>
        <ol style="margin-left:20px; margin-top:10px;">
            <li>Select <strong>Analysis Date</strong> in the sidebar.</li>
            <li>Choose <strong>Investment Style</strong>.</li>
            <li>Set <strong>Capital</strong> and <strong>Number of Positions</strong>.</li>
            <li>Click <strong>Run Analysis</strong>.</li>
        </ol>
        <p style="margin-top:1rem; font-weight:600; color:var(--primary-color);">
            Pure conviction scoring — no strategy bias, no historical performance chasing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _section_divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='metric-card info'><h4>CONVICTION-BASED</h4><h2>4 Signals</h2>
        <div class='sub-metric'>RSI · Oscillator · Z-Score · MA</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card success'><h4>ALL 95 STRATEGIES</h4><h2>No Filtering</h2>
        <div class='sub-metric'>Maximum diversification</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card primary'><h4>SIMPLE FORMULA</h4><h2>Transparent</h2>
        <div class='sub-metric'>weight = (conviction / total) × 100</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card warning'><h4>LIVE SIGNALS</h4><h2>Real-time</h2>
        <div class='sub-metric'>Per-position conviction scoring</div></div>""", unsafe_allow_html=True)

    _section_divider()

    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown(f"""
    <div style="text-align:center; padding:1rem 0; color:var(--text-muted); font-size:0.75rem;">
        <span>© 2026 Pragyam</span>
        <span style="margin:0 0.5rem; color:var(--border-light);">|</span>
        <span>@thebullishvalue</span>
        <span style="margin:0 0.5rem; color:var(--border-light);">|</span>
        <span>{VERSION}</span>
        <span style="margin:0 0.5rem; color:var(--border-light);">|</span>
        <span>{now_ist}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_results(capital: float):
    """Render results page with portfolio, regime, and system tabs."""
    portfolio = st.session_state.portfolio
    if portfolio.empty or "value" not in portfolio.columns:
        st.warning("Portfolio is empty. Adjust parameters and re-run.")
        return

    current_df = st.session_state.current_df
    regime_d = st.session_state.regime_result_dict or {}
    training_window = st.session_state.get("training_data_window", [])

    total_value = portfolio["value"].sum()
    cash_remaining = capital - total_value

    # Top metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"<div class='metric-card primary'><h4>Deployed</h4><h2>₹{total_value:,.0f}</h2></div>", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"<div class='metric-card neutral'><h4>Cash</h4><h2>₹{cash_remaining:,.0f}</h2></div>", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"<div class='metric-card info'><h4>Positions</h4><h2>{len(portfolio)}</h2></div>", unsafe_allow_html=True)
    with mc4:
        avg_conv = portfolio.get("conviction_score", pd.Series([50])).mean()
        st.markdown(f"<div class='metric-card success'><h4>Avg Conviction</h4><h2>{avg_conv:.0f}/100</h2></div>", unsafe_allow_html=True)

    _section_divider()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Position Guide", "Regime", "System"])

    with tab1:
        _render_portfolio_tab(portfolio, current_df, capital)

    with tab2:
        _render_position_guide_tab(portfolio, current_df)

    with tab3:
        _render_regime_tab(regime_d, st.session_state.get("regime_history_series", []), training_window)

    with tab4:
        _render_system_tab(training_window)

    # Footer
    _section_divider()
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown(f"""
    <div style="text-align:center; padding:1rem 0; color:var(--text-muted); font-size:0.75rem;">
        <span>© 2026 Pragyam</span>
        <span style="margin:0 0.5rem; color:var(--border-light);">|</span>
        <span>@thebullishvalue</span>
        <span style="margin:0 0.5rem; color:var(--border-light);">|</span>
        <span>{VERSION}</span>
        <span style="margin:0 0.5rem; color:var(--border-light);">|</span>
        <span>{now_ist}</span>
    </div>
    """, unsafe_allow_html=True)


def _run_analysis(
    selected_date: datetime,
    investment_style: str,
    capital: float,
    num_positions: int,
    selected_date_display: datetime.date,
):
    """Execute the 2-phase analysis pipeline."""
    metrics = get_metrics()
    metrics.phases, metrics.errors, metrics.warnings = {}, [], []
    st.session_state.debug_info = []
    st.session_state.regime_history_series = None

    try:
        # Print main header with run details
        from logger_config import generate_run_id
        current_run_id = generate_run_id()  # Fresh ID for each analysis
        run_details = {
            "Analysis Date": str(selected_date_display),
            "Investment Style": investment_style,
            "Capital": f"₹{capital:,.0f}",
            "Positions": str(num_positions),
            "Run ID": current_run_id[-12:],
            "Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        log.main_header(f"PRAGYAM | Portfolio Intelligence | {VERSION}", run_details)

        # PHASE 1: DATA FETCHING
        progress_bar = st.progress(0, text="Initializing…")
        status_text = st.empty()
        sub_progress = st.progress(0, text="")

        metrics.start_phase("total_execution")
        LOOKBACK_FILES = 100

        progress_bar.progress(0.05, text="Fetching market data…")
        status_text.text(f"Downloading {len(SYMBOLS_UNIVERSE)} symbols from yfinance…")
        log.section("Data Fetching", "Phase 1")
        metrics.start_phase("data_fetching")

        if not SYMBOLS_UNIVERSE:
            st.error("Symbol universe empty — check symbols.txt.")
            st.stop()

        all_hist = _load_historical_data(selected_date, LOOKBACK_FILES)
        if not all_hist:
            st.error("No historical data loaded. Check symbols.txt and date range.")
            st.stop()

        metrics.end_phase("data_fetching", success=True, items=len(all_hist))
        metrics.days_count = len(all_hist)

        progress_bar.progress(0.20, text="Data loaded.")
        status_text.text("Detecting market regime…")
        log.checkpoint(f"Loaded {len(all_hist)} days for {len(SYMBOLS_UNIVERSE)} symbols")

        # Regime detection
        log.detail("Detecting market regime…")
        regime_result = _detect_regime_cached(selected_date)
        regime_name = regime_result.get("regime", "UNKNOWN")
        confidence = regime_result.get("confidence", 0.0)
        log.checkpoint(f"Regime detected: {regime_name.replace('_', ' ')} ({confidence:.0%} confidence)")

        st.session_state.regime_result_dict = regime_result
        st.session_state.suggested_mix = regime_result.get("mix_name", "Chop/Consolidate Mix")
        st.session_state.training_data_window = all_hist

        if len(all_hist) < 10:
            st.error(f"Insufficient training data: {len(all_hist)} days (need ≥10).")
            log.failure("Phase 1", f"Insufficient training data: {len(all_hist)} days")
            metrics.end_phase("data_fetching", success=False, error_msg=f"Insufficient data: {len(all_hist)} days")
            st.stop()

        if not st.session_state.suggested_mix:
            st.error("Market regime could not be determined. Select a valid date.")
            log.failure("Phase 1", "Market regime undetermined")
            metrics.end_phase("data_fetching", success=False, error_msg="Regime undetermined")
            st.stop()

        log.checkpoint(f"Validated {len(all_hist)} days of training data")

        st.session_state.current_df = all_hist[-1][1] if all_hist else pd.DataFrame()
        log.checkpoint("Data ready for portfolio curation")

        # PHASE 2: CONVICTION-BASED CURATION
        progress_bar.progress(0.25, text="Curating portfolio…")
        status_text.text(f"Running {len(discover_strategies())} strategies…")
        sub_progress.progress(0, text="Initializing conviction scoring…")
        log.section("Conviction-Based Curation", "Phase 2")
        metrics.start_phase("conviction_curation")

        try:
            log.detail("Loading strategies from registry…")
            strategies = discover_strategies()
            strategies_to_run = {name: strategies[name] for name in strategies if name != "System_Curated"}

            if not strategies_to_run:
                st.error("No strategies available.")
                metrics.end_phase("conviction_curation", success=False, error_msg="Empty strategies")
                log.failure("Phase 2", "No strategies available")
                st.stop()

            log.checkpoint(f"Loaded {len(strategies_to_run)} strategies (excluded System_Curated)")
            log.success(f"Running ALL {len(strategies_to_run)} strategies")

            # Aggregate holdings
            log.detail("Aggregating holdings across all strategies…")
            aggregated_holdings = {}
            sub_progress.progress(30, text=f"Aggregating from {len(strategies_to_run)} strategies…")

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
                log.failure("Phase 2", "No holdings generated from strategies")
                st.stop()

            log.checkpoint(f"Aggregated {len(aggregated_holdings)} unique candidate symbols")

            sub_progress.progress(60, text=f"Computing conviction for {len(aggregated_holdings)} symbols…")
            log.detail("Computing conviction scores and applying style dispersion…")

            # Conviction-based weighting with style-aware dispersion
            # SIP: +125% boost / -50% penalty | Swing: +225% boost / -75% penalty
            st.session_state.portfolio = compute_conviction_based_weights(
                aggregated_holdings,
                st.session_state.current_df,
                capital,
                num_positions,
                st.session_state.min_pos_pct,
                st.session_state.max_pos_pct,
                apply_dispersion=True,
                investment_style=investment_style,  # Auto-selects dispersion based on style
            )

            if st.session_state.portfolio.empty:
                st.error("No portfolio generated. Check data quality.")
                metrics.end_phase("conviction_curation", success=False, error_msg="Empty portfolio")
                log.failure("Phase 2", "No portfolio generated after conviction weighting")
                st.stop()

            log.checkpoint(f"Curated {len(st.session_state.portfolio)} positions (avg conviction: {st.session_state.portfolio.get('conviction_score', pd.Series([50])).mean():.1f}/100)")
            log.success(f"Final portfolio: {len(st.session_state.portfolio)} positions ready")
            log.detail(f"Total deployed value: ₹{st.session_state.portfolio['value'].sum():,.0f}")
            sub_progress.progress(100, text="✓ Complete")
            sub_progress.empty()

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
            progress_bar.progress(1.0, text="Complete!")
            status_text.text(f"Portfolio: {len(st.session_state.portfolio)} positions ready")

            log.section("ANALYSIS COMPLETE", "DONE")
            avg_conviction = st.session_state.portfolio.get("conviction_score", pd.Series([50])).mean()
            top_conviction = st.session_state.portfolio.get("conviction_score", pd.Series([50])).max()

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
            progress_bar.empty()
            status_text.empty()
            st.toast("Analysis Complete!", icon="✅")

        except Exception as e:
            metrics.end_phase("total_execution", success=False, error_msg=str(e))
            log.failure("Analysis", str(e))
            st.error(f"Analysis failed: {e}")
            progress_bar.empty()
            status_text.empty()

    except Exception as e:
        log.failure("Initialization", str(e))
        st.error(f"Initialization failed: {e}")


def main():
    """Main application entry point."""
    _init_session_state()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0; margin-bottom:1rem;">
            <div style="font-size:1.75rem; font-weight:800; color:#FFC300;">PRAGYAM</div>
            <div style="color:#888; font-size:0.75rem; margin-top:0.25rem;">प्रज्ञम | Portfolio Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">📅 Analysis Date</div>', unsafe_allow_html=True)
        selected_date = st.date_input(
            "Select Date",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            help="Select the date for portfolio curation",
        )
        
        # Update regime display when date changes
        previous_date = st.session_state.get("regime_date", st.session_state.get("selected_date"))
        date_changed = previous_date != selected_date
        st.session_state.selected_date = selected_date
        selected_date_obj = datetime.combine(selected_date, datetime.min.time())
        
        # Auto-detect regime when date changes or if not yet detected for this date
        rd = st.session_state.get("regime_result_dict", {})
        regime_needs_update = not rd or date_changed
        
        if regime_needs_update:
            with st.spinner("Detecting regime..."):
                rd = _detect_regime_cached(selected_date_obj)
                st.session_state.regime_result_dict = rd
                st.session_state.suggested_mix = rd.get("mix_name", "Chop/Consolidate Mix")
                # Store the date for which regime was detected to detect future changes
                st.session_state.regime_date = selected_date
        if rd and isinstance(rd, dict):
            regime_name_sb = rd.get("regime", "UNKNOWN")
            color_sb = rd.get("color", "#888888")
            conf_sb = rd.get("confidence", 0.0)
            score_sb = rd.get("composite_score", 0.0)
            st.markdown(f"""
            <div style="background:{color_sb}12; border:1px solid {color_sb}40; border-radius:10px;
                        padding:12px; margin:10px 0 20px 0;">
                <div style="color:#888; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; margin-bottom:4px;">Market Regime</div>
                <div style="color:{color_sb}; font-size:1.1rem; font-weight:700; line-height:1.2;">
                    {rd.get('icon', '')} {regime_name_sb.replace('_', ' ')}
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
                    <span style="color:#888; font-size:0.75rem;">Score {score_sb:+.2f}</span>
                    <span style="color:{color_sb}; font-weight:700; font-size:0.8rem;">{conf_sb:.0%} confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">💼 Portfolio Style</div>', unsafe_allow_html=True)
        investment_style = st.selectbox(
            "Investment Style",
            options=["Swing Trading", "SIP Investment"],
            index=0,
            help="Primary investment objective",
        )

        st.markdown('<div class="sidebar-title">⚙️ Portfolio Parameters</div>', unsafe_allow_html=True)
        capital = st.number_input(
            "Capital (₹)",
            min_value=1000,
            max_value=100000000,
            value=2500000,
            step=1000,
            help="Total capital to allocate"
        )
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

        _section_divider()

        run_clicked = st.button("Run Analysis", type="primary", width='stretch')

        if run_clicked:
            _run_analysis(
                selected_date_obj, investment_style, capital,
                num_positions, selected_date,
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size:0.8rem; margin:0; color:var(--text-muted); line-height:1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Conviction-Based Curation<br>
                <strong>Data:</strong> Live yfinance (NSE)
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    if st.session_state.portfolio is None:
        _render_landing_page()
    else:
        _render_results(capital)


if __name__ == "__main__":
    main()
