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
from ui.theme import inject_css, VERSION, PRODUCT_NAME, COMPANY, progress_bar
from ui.components import (
    render_header,
    render_section_header,
    render_metric_card,
    render_info_box,
    render_system_card,
    section_gap,
    render_conviction_signal,
    render_warning_box,
    render_chart_skeleton,
    render_collapsible_section,
    render_collapsible_section_close,
    render_theme_toggle,
    render_export_button_row,
    render_interpretation_card,
)
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

VERSION = "v7.0.5"
PRODUCT_NAME = "Pragyam"
COMPANY = "@thebullishvalue"

st.set_page_config(
    page_title="PRAGYAM | Portfolio Intelligence",
    page_icon="📈",
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
        "debug_info": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


@st.cache_data(ttl=3600, show_spinner=False)
def _detect_regime_cached(end_date: datetime, symbols_key: str) -> Dict:
    """Detect market regime and return as serializable dict."""
    # Resolve symbols from the cache key
    try:
        if symbols_key.startswith("UNIVERSE:"):
            universe_name, index = symbols_key.replace("UNIVERSE:", "", 1).split("|", 1)
            index = index if index != "None" else None
            symbols_list, _ = resolve_universe(universe_name, index)
        else:
            symbols_list = get_default_universe()
    except Exception as e:
        return {
            "regime": "UNKNOWN",
            "mix_name": "Chop/Consolidate Mix",
            "confidence": 0.30,
            "composite_score": 0.0,
            "explanation": f"Error resolving universe: {e}",
            "color": REGIME_COLORS["UNKNOWN"],
            "icon": "❓",
            "description": "",
        }
    
    detector = MarketRegimeDetector()
    window_days = int(MAX_INDICATOR_PERIOD * 1.5) + 30
    try:
        hist = generate_historical_data(
            symbols_to_process=symbols_list,
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

    portfolio_with_signals = compute_conviction_signals(portfolio, current_df)

    # Portfolio table — Custom HTML with inline CSS via st_html
    import html as html_module
    
    table_rows = []
    for _, row in portfolio.iterrows():
        symbol_escaped = html_module.escape(str(row["symbol"]))
        table_rows.append(
            f'<tr>'
            f'<td class="symbol">{symbol_escaped}</td>'
            f'<td class="numeric currency">&#8377;{row["price"]:,.2f}</td>'
            f'<td class="numeric">{row["units"]:,.0f}</td>'
            f'<td class="numeric percentage">{row["weightage_pct"]:.2f}%</td>'
            f'<td class="numeric currency">&#8377;{row["value"]:,.2f}</td>'
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
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th class="numeric">Price (&#8377;)</th>
                    <th class="numeric">Units</th>
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
    st.components.v1.html(table_html, height=table_height)

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


def _render_position_guide_tab(portfolio: pd.DataFrame, current_df: pd.DataFrame):
    """Tab — Position Guide with entry conditions and conviction signals."""
    portfolio_with_signals = compute_conviction_signals(portfolio, current_df)

    if "rsi_signal" not in portfolio_with_signals.columns:
        st.info("Position guide signals unavailable.")
        return

    # ── Signal Distribution ──────────────────────────────────────────────
    render_section_header("Signal Distribution", "Portfolio conviction breakdown", icon="target")

    c1, c2, c3, c4 = st.columns(4)

    scores = []
    for _, row in portfolio_with_signals.iterrows():
        score = row.get("conviction_score", 50)
        scores.append(float(score))

    strong_buy = sum(1 for s in scores if s >= 65)
    buy = sum(1 for s in scores if 50 <= s < 65)
    hold = sum(1 for s in scores if 35 <= s < 50)
    caution = sum(1 for s in scores if s < 35)

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

    # Build single unified table sorted by conviction score
    import html as html_module

    # Tier colors mapping
    tier_colors = {
        'Strong Buy': ('#34D399', '#6EE7B7', 'rgba(52, 211, 153, 0.15)', 'rgba(52, 211, 153, 0.3)'),
        'Buy': ('#6EE7B7', '#A7F3D0', 'rgba(52, 211, 153, 0.1)', 'rgba(52, 211, 153, 0.2)'),
        'Hold': ('#D4A853', '#E8C478', 'rgba(212, 168, 83, 0.15)', 'rgba(212, 168, 83, 0.3)'),
        'Caution': ('#FB7185', '#FDA4AF', 'rgba(251, 113, 133, 0.15)', 'rgba(251, 113, 133, 0.3)'),
    }

    # Sort by conviction score descending
    sorted_df = portfolio_with_signals.sort_values('conviction_score', ascending=False).reset_index(drop=True)

    table_rows = []
    for _, row in sorted_df.iterrows():
        score = row.get("conviction_score", 50)
        rsi_val = row.get("rsi_value")
        osc_val = row.get("osc_value")
        z_val = row.get("zscore_value")
        ma_count = row.get("ma_count", 0)
        price = row.get("price", 0)
        weight = row.get("weightage_pct", 0)

        # Determine tier
        if score >= 65:
            tier_name = "Strong Buy"
            emoji = "🟢"
        elif score >= 50:
            tier_name = "Buy"
            emoji = "🟩"
        elif score >= 35:
            tier_name = "Hold"
            emoji = "🟡"
        else:
            tier_name = "Caution"
            emoji = "🔴"

        primary_color, bright_color, bg_color, border_color = tier_colors[tier_name]

        # Format signal values
        rsi_str = f"{rsi_val:.2f}" if rsi_val is not None and not pd.isna(rsi_val) else "—"
        osc_str = f"{osc_val:.2f}" if osc_val is not None and not pd.isna(osc_val) else "—"
        z_str = f"{z_val:.2f}" if z_val is not None and not pd.isna(z_val) else "—"
        ma_str = f"{int(ma_count)}/5" if pd.notna(ma_count) else "—"

        symbol_escaped = html_module.escape(str(row["symbol"]))

        table_rows.append(
            f'<tr>'
            f'<td class="symbol">{symbol_escaped}</td>'
            f'<td class="numeric currency">&#8377;{price:,.2f}</td>'
            f'<td class="numeric"><span style="color: {bright_color};">{emoji} {tier_name}</span></td>'
            f'<td class="numeric" style="color: {bright_color}; font-weight: 600;">{int(score)}</td>'
            f'<td class="numeric">{rsi_str}</td>'
            f'<td class="numeric">{osc_str}</td>'
            f'<td class="numeric">{z_str}</td>'
            f'<td class="numeric">{ma_str}</td>'
            f'<td class="numeric percentage">{weight:.2f}%</td>'
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
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th class="numeric">Price (&#8377;)</th>
                    <th class="numeric">Signal</th>
                    <th class="numeric">Conviction</th>
                    <th class="numeric">RSI</th>
                    <th class="numeric">Osc</th>
                    <th class="numeric">Z</th>
                    <th class="numeric">MA</th>
                    <th class="numeric">Weight %</th>
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

    table_height = max(280, 220 + len(sorted_df) * 42)
    st.components.v1.html(table_html, height=table_height)


def _render_regime_tab(regime_result: Dict, regime_series: List, training_data: List = None):
    """Tab 2 — Market regime analysis."""
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
    render_section_header("Current Market Regime", "10-day indicator window", icon="eye")

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
        render_section_header("Regime Score History", "Rolling 10-day composite", icon="activity", accent="emerald")

        regimes_seq = [r.regime for r in regime_series_to_use]
        transitions = sum(1 for i in range(1, len(regimes_seq)) if regimes_seq[i] != regimes_seq[i-1])
        last_regime = regimes_seq[-1] if regimes_seq else "—"
        prev_regime = regimes_seq[-2] if len(regimes_seq) > 1 else "—"

        if CHARTS_AVAILABLE:
            st.markdown('<div class="chart-container regime">', unsafe_allow_html=True)
            fig_rh = create_regime_history_chart(regime_series_to_use)
            st.plotly_chart(fig_rh, width='stretch', key="tab2_regime_history")
            st.markdown('</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        # Map regime names to semantic metric card colors
        # Matches actual regimes from regime.py: STRONG_BULL, BULL, WEAK_BULL,
        # CHOP, WEAK_BEAR, BEAR, CRISIS, UNKNOWN
        def regime_color(regime: str) -> str:
            r = regime.upper().replace("-", "_")
            if r in ("STRONG_BULL", "BULL", "WEAK_BULL"):
                return "success"   # emerald — bullish
            elif r in ("BEAR", "CRISIS"):
                return "danger"    # rose — bearish/crisis
            elif r == "WEAK_BEAR":
                return "warning"   # amber — cautionary
            elif r in ("CHOP", "UNKNOWN"):
                return "info"      # cyan — neutral/choppy
            return "neutral"       # slate — fallback

        with c1:
            render_metric_card("Transitions", str(transitions), "Over analysis window", "info")
        with c2:
            render_metric_card("Current", last_regime.replace("_", " "), "Latest", regime_color(last_regime))
        with c3:
            render_metric_card("Prior", prev_regime.replace("_", " "), "Previous", regime_color(prev_regime))


def _render_system_tab(training_window: List):
    """Tab 3 — System information."""
    render_section_header("System & Execution Details", "Configuration and metadata", icon="cpu")

    c1, c2 = st.columns(2)
    with c1:
        render_metric_card("Analysis Date", str(st.session_state.get("selected_date", "N/A")), "Portfolio date", "info")
    with c2:
        portfolio_val = st.session_state.get("portfolio")
        num_positions = len(portfolio_val) if portfolio_val is not None and not portfolio_val.empty else 0
        render_metric_card("Positions", str(num_positions), "Current holdings", "info")

    _section_divider()

    render_section_header("Configuration", "System settings", icon="settings")

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
    st.dataframe(details_df, width='stretch', hide_index=True)

    _section_divider()

    render_section_header("Technical Information", "Conviction scoring and weighting", icon="target")

    c1, c2 = st.columns(2)

    with c1:
        render_metric_card(
            "📊 Conviction Scoring",
            "0-100 Range",
            "Signals: RSI (30%) · Oscillator (30%)<br>Z-Score (20%) · MA Alignment (20%)<br><br>Formula: (raw + 2) / 4 × 100",
            "info"
        )

    with c2:
        render_metric_card(
            "⚖️ Portfolio Weighting",
            "(conviction / total) × 100",
            "Bounds: Min 1% · Max 10%<br>Selection: Top 30 by conviction<br><br>No threshold: All symbols eligible",
            "success"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def _render_header() -> None:
    """Render the main masthead header (matches Nishkarsh design)."""
    render_header(
        title=f"{PRODUCT_NAME}",
        tagline="Conviction-Based Portfolio Curation · All 95 Strategies · Live NSE Data"
    )


def _render_landing_page():
    """Render landing page when no portfolio is available.
    
    Exact Nishkarsh design thesis: 3 system cards in a row, followed by landing prompt.
    Adapted for Pragyam's portfolio intelligence features.
    """
    # Three system cards — Portfolio, Regime, Strategies
    section_gap()

    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        st.markdown("""
        <div class='system-card portfolio'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
                PORTFOLIO
            </h3>
            <p>Conviction-based portfolio curation with composite scoring across four technical indicators for precision selection.</p>
            <div class='spec'>
                <span>Signals:</span> RSI (30%) + Osc (30%) + Z (20%) + MA (20%)<br>
                <span>Selection:</span> Top 30 by conviction score<br>
                <span>Weighting:</span> (conviction / total) × 100<br>
                <span>Dispersion:</span> SIP + Swing modes
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='system-card regime'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
                REGIME
            </h3>
            <p>Seven-factor market regime detection with composite scoring for adaptive portfolio positioning and risk management.</p>
            <div class='spec'>
                <span>Regimes:</span> Strong Bull · Bull · Neutral · Bear<br>
                <span>Factors:</span> Momentum · Trend · Breadth · Velocity<br>
                <span>Output:</span> Confidence score + mix classification<br>
                <span>History:</span> 30-day rolling window
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='system-card strategies'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                STRATEGIES
            </h3>
            <p>Ninety-five quantitative strategies running in parallel across momentum, reversal, breakout, and pattern recognition.</p>
            <div class='spec'>
                <span>Categories:</span> Momentum + Reversal + Breakout<br>
                <span>Universe:</span> Nifty 500 + F&O symbols<br>
                <span>Style:</span> SIP + Swing trading dispersion<br>
                <span>Strategies:</span> 95 parallel engines
            </div>
        </div>
        """, unsafe_allow_html=True)

    section_gap()
    
    # Landing prompt
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

    # Top metrics — logical color coding
    mc1, mc2, mc3, mc4 = st.columns(4)

    # Cash health: <5% = danger, <15% = warning, else = success
    cash_pct = (cash_remaining / capital * 100) if capital > 0 else 0
    cash_color = "danger" if cash_pct < 5 else ("warning" if cash_pct < 15 else "success")

    # Avg conviction health: <35 = danger, 35-49 = warning, 50-64 = info, >=65 = success
    avg_conv = portfolio.get("conviction_score", pd.Series([50])).mean()
    conv_color = "danger" if avg_conv < 35 else ("warning" if avg_conv < 50 else ("info" if avg_conv < 65 else "success"))

    with mc1:
        render_metric_card("Deployed", f"₹{total_value:,.0f}", f"{total_value/capital*100:.0f}% of capital", "info")
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
    selected_date_display: datetime.date,
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
            "Analysis Date": str(selected_date_display),
            "Universe": universe,
            "Index": index if index else "N/A",
            "Symbols": str(len(symbols_list)),
            "Investment Style": investment_style,
            "Capital": f"₹{capital:,.0f}",
            "Positions": str(num_positions),
            "Run ID": current_run_id[-12:],
            "Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        log.main_header(f"PRAGYAM | Portfolio Intelligence | {VERSION}", run_details)

        # Custom styled progress container (matches Nishkarsh)
        progress_container = st.empty()

        # PHASE 1: DATA FETCHING
        progress_bar(progress_container, 2, "Fetching market data", f"yfinance · {len(symbols_list)} symbols")
        metrics.start_phase("total_execution")
        LOOKBACK_FILES = 100

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
            
            # Clear progress container after short delay
            import time
            time.sleep(1.5)
            progress_container.empty()
            
            st.toast("Analysis Complete!", icon="✅")

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
        <div style="text-align:center;padding:0.75rem 0 1rem 0;">
            <div style="font-family:var(--display);font-size:1.5rem;font-weight:700;color:var(--amber);letter-spacing:0.06em;">PRAGYAM</div>
            <div style="font-family:var(--data);color:var(--ink-tertiary);font-size:0.65rem;margin-top:0.2rem;letter-spacing:0.08em;text-transform:uppercase;">प्रज्ञम | Portfolio Intelligence</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
        selected_date = st.date_input(
            "Select Date",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            help="Select the date for portfolio curation",
        )

        st.markdown('<div class="sidebar-title">Analysis Universe</div>', unsafe_allow_html=True)
        universe, selected_index = render_universe_selector()
        st.session_state.selected_universe = universe
        st.session_state.selected_index = selected_index

        # Create a cache key for the universe
        symbols_key = f"UNIVERSE:{universe}|{selected_index}"
        st.session_state.symbols_key = symbols_key

        # Check if date or universe changed to trigger regime update
        previous_date = st.session_state.get("regime_date", st.session_state.get("selected_date"))
        previous_symbols_key = st.session_state.get("regime_symbols_key", "")
        date_changed = previous_date != selected_date
        universe_changed = previous_symbols_key != symbols_key
        st.session_state.selected_date = selected_date
        selected_date_obj = datetime.combine(selected_date, datetime.min.time())

        # Auto-detect regime when date changes, universe changes, or if not yet detected
        rd = st.session_state.get("regime_result_dict", {})
        regime_needs_update = not rd or date_changed or universe_changed

        if regime_needs_update:
            with st.spinner("Detecting regime..."):
                rd = _detect_regime_cached(selected_date_obj, symbols_key)
                st.session_state.regime_result_dict = rd
                st.session_state.suggested_mix = rd.get("mix_name", "Chop/Consolidate Mix")
                # Store the date AND universe for which regime was detected
                st.session_state.regime_date = selected_date
                st.session_state.regime_symbols_key = symbols_key
        if rd and isinstance(rd, dict):
            regime_name_sb = rd.get("regime", "UNKNOWN")
            color_sb = rd.get("color", "#888888")
            conf_sb = rd.get("confidence", 0.0)
            score_sb = rd.get("composite_score", 0.0)
            st.markdown(f"""
            <div style="background:{color_sb}12; border:1px solid {color_sb}40; border-radius:10px;
                        padding:12px; margin:10px 0 20px 0;">
                <div style="color:var(--ink-tertiary); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; margin-bottom:4px; font-family:var(--data);">Market Regime</div>
                <div style="color:{color_sb}; font-size:1.1rem; font-weight:700; line-height:1.2; font-family:var(--display);">
                    {rd.get('icon', '')} {regime_name_sb.replace('_', ' ')}
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
                    <span style="color:var(--ink-tertiary); font-size:0.75rem; font-family:var(--data);">Score {score_sb:+.2f}</span>
                    <span style="color:{color_sb}; font-weight:700; font-size:0.8rem; font-family:var(--data);">{conf_sb:.0%} confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">Portfolio Style</div>', unsafe_allow_html=True)
        investment_style = st.selectbox(
            "Investment Style",
            options=["Swing Trading", "SIP Investment"],
            index=0,
            help="Primary investment objective",
        )

        st.markdown('<div class="sidebar-title">Portfolio Parameters</div>', unsafe_allow_html=True)
        capital = st.number_input(
            "Capital (₹)",
            min_value=1000,
            max_value=100000000,
            value=2500000,
            step=1000,
            help="Total capital to allocate"
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

        _section_divider()

        run_clicked = st.button("Run Analysis", type="primary", width='stretch')

        if run_clicked:
            # Store params in session state so they persist across rerun
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

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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
            rows.append('<div class="spec-row"><span class="spec-label">Symbols</span><span class="spec-value">' + str(len(symbols_list)) + '</span></div>')
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
