"""
PRAGYAM (प्रज्ञम) - Portfolio Intelligence | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Walk-forward portfolio curation with regime-aware strategy allocation.
Multi-strategy backtesting and capital optimization engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import io
import base64
from scipy import stats
from sklearn.preprocessing import StandardScaler
import time
import warnings

# --- Suppress known NumPy warnings during backtest warm-up ---
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
# --- End suppression ---

# --- Import Unified Chart Components ---
try:
    from charts import (
        COLORS, get_chart_layout, get_axis_config,
        create_equity_drawdown_chart, create_rolling_metrics_chart,
        create_correlation_heatmap, create_tier_sharpe_heatmap,
        create_risk_return_scatter, create_factor_radar,
        create_weight_evolution_chart, create_signal_heatmap
    )
    UNIFIED_CHARTS_AVAILABLE = True
except ImportError:
    UNIFIED_CHARTS_AVAILABLE = False
    # Fallback color scheme
    COLORS = {
        'primary': '#FFC300', 'success': '#10b981', 'danger': '#ef4444',
        'warning': '#f59e0b', 'info': '#06b6d4', 'muted': '#888888',
        'card': '#1A1A1A', 'border': '#2A2A2A', 'text': '#EAEAEA'
    }

# --- Import Strategies from strategies.py ---
try:
    from strategies import (
        BaseStrategy, PRStrategy, CL1Strategy, CL2Strategy, CL3Strategy, MOM1Strategy, MOM2Strategy,
        MomentumMasters, VolatilitySurfer, AdaptiveVolBreakout, VolReversalHarvester, AlphaSurge,
        ReturnPyramid, MomentumCascade, AlphaVortex, SurgeSentinel, VelocityVortex, BreakoutAlphaHunter,
        ExtremeMomentumBlitz, HyperAlphaIgniter, VelocityApocalypse, QuantumMomentumLeap,
        NebulaMomentumStorm, ResonanceEcho
    )
except ImportError:
    st.error("Fatal Error: `strategies.py` not found. Please ensure it's in the same directory.")
    st.stop()

# --- Import Live Data Generation from backdata.py ---
try:
    from backdata import (
        generate_historical_data, 
        load_symbols_from_file, 
        MAX_INDICATOR_PERIOD,
        SYMBOLS_UNIVERSE
    )
except ImportError:
    st.error("Fatal Error: `backdata.py` not found. Please ensure it's in the same directory.")
    st.stop()

# --- Import Unified Backtest Engine for Dynamic Strategy Selection ---
try:
    from backtest_engine import (
        UnifiedBacktestEngine,
        DynamicPortfolioStylesGenerator,
        PerformanceMetrics
    )
    DYNAMIC_SELECTION_AVAILABLE = True
except ImportError:
    DYNAMIC_SELECTION_AVAILABLE = False
    logging.warning("backtest_engine.py not found. Using static strategy selection.")


# --- System Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('system.log'), logging.StreamHandler()])
st.set_page_config(page_title="PRAGYAM | Portfolio Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# --- Constants ---
VERSION = "v3.1.0"  # Unified UI/UX + Hedge Fund Grade Analytics
PRODUCT_NAME = "Pragyam"
COMPANY = "Hemrek Capital"

# --- CSS Styling (Hemrek Capital Design System) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        
        --success-green: #10b981;
        --danger-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        --neutral: #888888;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main, [data-testid="stSidebar"] {
        background-color: var(--background-color);
        color: var(--text-primary);
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    .block-container {
        padding-top: 3.5rem;
        max-width: 90%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Sidebar toggle button - always visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        width: 40px !important;
        height: 40px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important;
        transform: scale(1.05);
    }
    
    [data-testid="collapsedControl"] svg {
        stroke: var(--primary-color) !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] svg {
        stroke: var(--primary-color) !important;
    }
    
    button[kind="header"] {
        z-index: 999999 !important;
    }
    
    [data-testid="stSidebar"] { 
        background: var(--secondary-background-color); 
        border-right: 1px solid var(--border-color); 
    }
    
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.50px;
        position: relative;
    }
    
    .premium-header .tagline {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 0.25rem;
        font-weight: 400;
        position: relative;
    }
    
    .metric-card {
        background-color: var(--bg-card);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        border-color: var(--border-light);
    }
    
    .metric-card h4 {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h2 {
        color: var(--text-primary);
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    
    .metric-card .sub-metric {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    .metric-card.white h2 { color: var(--text-primary); }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .info-box {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-left: 0px solid var(--primary-color);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
    }
    
    .info-box h4 {
        color: var(--primary-color);
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 700;
    }

    /* Buttons */
    .stButton>button {
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        font-weight: 700;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6);
        background: var(--primary-color);
        color: #1A1A1A;
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }

    /* Download Links */
    .download-link {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        text-decoration: none;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .download-link:hover {
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6);
        background: var(--primary-color);
        color: #1A1A1A;
        transform: translateY(-2px);
    }
    
    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        background: var(--bg-card);
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .stMarkdown table th,
    .stMarkdown table td {
        text-align: left !important;
        padding: 12px 10px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .stMarkdown table th {
        background-color: var(--bg-elevated);
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    
    .stMarkdown table tr:last-child td {
        border-bottom: none;
    }
    
    .stMarkdown table tr:hover {
        background-color: var(--bg-elevated);
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted);
        border-bottom: 2px solid transparent;
        transition: color 0.3s, border-bottom 0.3s;
        background: transparent;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-color);
        background: transparent !important;
    }
    .stPlotlyChart, .stDataFrame {
        border-radius: 12px;
        background-color: var(--secondary-background-color);
        padding: 10px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1);
    }
    h2 {
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 10px;
    }
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%);
        margin: 1.5rem 0;
    }
    
    .sidebar-title { 
        font-size: 0.75rem; 
        font-weight: 700; 
        color: var(--primary-color); 
        text-transform: uppercase; 
        letter-spacing: 1px; 
        margin-bottom: 0.75rem; 
    }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if 'performance' not in st.session_state: st.session_state.performance = None
if 'portfolio' not in st.session_state: st.session_state.portfolio = None
if 'current_df' not in st.session_state: st.session_state.current_df = None
if 'selected_date' not in st.session_state: st.session_state.selected_date = None
if 'suggested_mix' not in st.session_state: st.session_state.suggested_mix = None
if 'regime_display' not in st.session_state: st.session_state.regime_display = None # For sidebar display
if 'min_pos_pct' not in st.session_state: st.session_state.min_pos_pct = 1.0
if 'max_pos_pct' not in st.session_state: st.session_state.max_pos_pct = 10.0

# --- Base Classes and Utilities ---
def fix_csv_export(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()

def create_export_link(data_bytes, filename):
    """Create downloadable CSV link"""
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">Download Portfolio CSV</a>'
    return href

# =========================================================================
# --- Live Data Loading Function ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(end_date: datetime, lookback_files: int) -> List[Tuple[datetime, pd.DataFrame]]:
    """
    Fetches and processes all historical data on-the-fly using the
    backdata.py module.
    """
    logging.info(f"--- START: Live Data Generation (End Date: {end_date.date()}, Training Lookback: {lookback_files}) ---")
    
    total_days_to_fetch = int((lookback_files + MAX_INDICATOR_PERIOD) * 12)
    fetch_start_date = end_date - timedelta(days=total_days_to_fetch)
    
    logging.info(f"Calculated fetch start date: {fetch_start_date.date()} (Total days: {total_days_to_fetch})")

    try:
        live_data = generate_historical_data(
            symbols_to_process=SYMBOLS_UNIVERSE,
            start_date=fetch_start_date,
            end_date=end_date
        )
        
        if not live_data:
            logging.warning("Live data generation returned no data.")
            return []
            
        logging.info(f"--- SUCCESS: Live Data Generation. {len(live_data)} total valid days generated. ---")
        return live_data
        
    except Exception as e:
        logging.error(f"Error during load_historical_data: {e}")
        st.error(f"Failed to fetch or process live data: {e}")
        return []

# =========================================================================


# --- Core Backtesting & Curation Engine (Optimized) ---
def compute_portfolio_return(portfolio: pd.DataFrame, next_prices: pd.DataFrame) -> float:
    if portfolio.empty or 'value' not in portfolio.columns or portfolio['value'].sum() == 0: return 0.0
    merged = portfolio.merge(next_prices[['symbol', 'price']], on='symbol', how='inner', suffixes=('_prev', '_next'))
    if merged.empty: return 0.0
    returns = (merged['price_next'] - merged['price_prev']) / merged['price_prev']
    return np.average(returns, weights=merged['value'])

def calculate_advanced_metrics(returns_with_dates: List[Dict]) -> Tuple[Dict, float]:
    """
    Calculate comprehensive risk-adjusted performance metrics.
    
    Mathematical Framework:
    - CAGR: Compound Annual Growth Rate via geometric mean
    - Sharpe: Excess return per unit of total volatility (annualized)
    - Sortino: Excess return per unit of downside deviation
    - Calmar: CAGR / |Max Drawdown| - recovery efficiency metric
    - Kelly: f* = p - q/b where p=win_rate, q=1-p, b=avg_win/avg_loss
    
    Uses proper time-weighted annualization factor.
    """
    default_metrics = {
        'total_return': 0, 'annual_return': 0, 'volatility': 0, 
        'sharpe': 0, 'sortino': 0, 'max_drawdown': 0, 'calmar': 0, 
        'win_rate': 0, 'kelly_criterion': 0, 'omega_ratio': 1.0,
        'tail_ratio': 1.0, 'gain_to_pain': 0, 'profit_factor': 1.0
    }
    if len(returns_with_dates) < 2: 
        return default_metrics, 52

    returns_df = pd.DataFrame(returns_with_dates).sort_values('date').set_index('date')
    time_deltas = returns_df.index.to_series().diff().dt.days
    avg_period_days = time_deltas.mean()
    periods_per_year = 365.25 / avg_period_days if pd.notna(avg_period_days) and avg_period_days > 0 else 52

    returns = returns_df['return']
    n_periods = len(returns)
    
    # Total Return (geometric)
    total_return = (1 + returns).prod() - 1
    
    # CAGR: Correct annualization formula
    # CAGR = (Final/Initial)^(1/years) - 1 = (1 + total_return)^(periods_per_year/n_periods) - 1
    years = n_periods / periods_per_year
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = 0
    
    # Volatility (annualized standard deviation)
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    
    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe = annual_return / volatility if volatility > 0.001 else 0
    sharpe = np.clip(sharpe, -10, 10)

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) >= 2:
        downside_vol = downside_returns.std(ddof=1) * np.sqrt(periods_per_year)
        sortino = annual_return / downside_vol if downside_vol > 0.001 else (annual_return * 10 if annual_return > 0 else 0)
    else:
        sortino = annual_return * 10 if annual_return > 0 else 0
    sortino = np.clip(sortino, -20, 20)

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown_series = (cumulative / running_max) - 1
    max_drawdown = drawdown_series.min()
    
    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown < -0.001 else (annual_return * 10 if annual_return > 0 else 0)
    calmar = np.clip(calmar, -20, 20)
    
    # Win Rate
    win_rate = (returns > 0).mean()

    # Win/Loss Statistics
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = gains.mean() if len(gains) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    total_gains = gains.sum() if len(gains) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0
    
    # Kelly Criterion: f* = W - (1-W)/R where W=win_rate, R=avg_win/avg_loss
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0.0001 else 0
    kelly = (win_rate - ((1 - win_rate) / win_loss_ratio)) if win_loss_ratio > 0 else 0
    kelly = np.clip(kelly, -1, 1)
    
    # Omega Ratio: ∫(gains) / ∫(losses) above/below threshold=0
    omega_ratio = total_gains / total_losses if total_losses > 0.0001 else (total_gains * 10 if total_gains > 0 else 1.0)
    omega_ratio = np.clip(omega_ratio, 0, 50)
    
    # Profit Factor: Sum(gains) / Sum(losses)
    profit_factor = total_gains / total_losses if total_losses > 0.0001 else (10.0 if total_gains > 0 else 1.0)
    profit_factor = np.clip(profit_factor, 0, 50)
    
    # Tail Ratio: 95th percentile / |5th percentile|
    upper_tail = np.percentile(returns, 95) if len(returns) >= 20 else returns.max()
    lower_tail = abs(np.percentile(returns, 5)) if len(returns) >= 20 else abs(returns.min())
    tail_ratio = upper_tail / lower_tail if lower_tail > 0.0001 else (10.0 if upper_tail > 0 else 1.0)
    tail_ratio = np.clip(tail_ratio, 0, 20)
    
    # Gain-to-Pain Ratio: Total return / Sum(abs(negative returns))
    pain = abs(losses.sum()) if len(losses) > 0 else 0
    gain_to_pain = returns.sum() / pain if pain > 0.0001 else (returns.sum() * 10 if returns.sum() > 0 else 0)
    gain_to_pain = np.clip(gain_to_pain, -20, 20)

    metrics = {
        'total_return': total_return, 
        'annual_return': annual_return, 
        'volatility': volatility, 
        'sharpe': sharpe, 
        'sortino': sortino, 
        'max_drawdown': max_drawdown, 
        'calmar': calmar, 
        'win_rate': win_rate, 
        'kelly_criterion': kelly,
        'omega_ratio': omega_ratio,
        'tail_ratio': tail_ratio,
        'gain_to_pain': gain_to_pain,
        'profit_factor': profit_factor
    }
    return metrics, periods_per_year

def calculate_strategy_weights(performance: Dict) -> Dict[str, float]:
    strat_names = list(performance['strategy'].keys())
    if not strat_names:
        return {}

    sharpe_values = np.array([performance['strategy'][name].get('sharpe', 0) + 2 for name in strat_names])

    if sharpe_values.size == 0:
        return {name: 1.0 / len(strat_names) for name in strat_names} if strat_names else {}

    # Stabilize the exp calculation to prevent overflow
    stable_sharpes = sharpe_values - np.max(sharpe_values)
    exp_sharpes = np.exp(stable_sharpes)
    total_score = np.sum(exp_sharpes)

    if total_score == 0 or not np.isfinite(total_score):
        return {name: 1.0 / len(strat_names) for name in strat_names}

    weights = exp_sharpes / total_score
    return {name: weights[i] for i, name in enumerate(strat_names)}

def _calculate_performance_on_window(window_data: List[Tuple[datetime, pd.DataFrame]], strategies: Dict[str, BaseStrategy], training_capital: float) -> Dict:
    performance = {name: {'returns': []} for name in strategies}
    subset_performance = {name: {} for name in strategies}
    for i in range(len(window_data) - 1):
        date, df = window_data[i]
        next_date, next_df = window_data[i+1]
        for name, strategy in strategies.items():
            try:
                portfolio = strategy.generate_portfolio(df, training_capital)
                if portfolio.empty: continue
                performance[name]['returns'].append({'return': compute_portfolio_return(portfolio, next_df), 'date': next_date})
                n, tier_size = len(portfolio), 10
                num_tiers = n // tier_size
                if num_tiers == 0: continue
                for j in range(num_tiers):
                    tier_name = f'tier_{j+1}'
                    if tier_name not in subset_performance[name]: subset_performance[name][tier_name] = []
                    sub_df = portfolio.iloc[j*tier_size : (j+1)*tier_size]
                    if not sub_df.empty:
                        sub_ret = compute_portfolio_return(sub_df, next_df)
                        subset_performance[name][tier_name].append({'return': sub_ret, 'date': next_date})
            except Exception as e: logging.error(f"Window Calc Error ({name}, {date}): {e}")
    final_performance = {name: {'metrics': calculate_advanced_metrics(perf['returns'])[0], 'sharpe': calculate_advanced_metrics(perf['returns'])[0]['sharpe']} for name, perf in performance.items()}
    final_sub_performance = {name: {sub: calculate_advanced_metrics(sub_perf)[0]['sharpe'] for sub, sub_perf in data.items() if sub_perf} for name, data in subset_performance.items()}
    return {'strategy': final_performance, 'subset': final_sub_performance}

def evaluate_historical_performance(_strategies: Dict[str, BaseStrategy], historical_data: List[Tuple[datetime, pd.DataFrame]]) -> Dict:
    MIN_TRAIN_FILES = 2
    TRAINING_CAPITAL = 2500000.0
    if len(historical_data) < MIN_TRAIN_FILES + 1:
        st.error(f"Not enough historical data for the selected period. Need at least {MIN_TRAIN_FILES + 1} files to run a backtest.")
        return {}

    all_names = list(_strategies.keys()) + ['System_Curated']
    oos_perf = {name: {'returns': []} for name in all_names}
    weight_entropies = []
    strategy_weights_history = []
    subset_weights_history = []

    progress_bar = st.progress(0, text="Initializing backtest...")
    total_steps = len(historical_data) - MIN_TRAIN_FILES - 1
    
    if total_steps <= 0:
        st.error(f"Not enough data for a single backtest step. Need at least {MIN_TRAIN_FILES + 2} days of data.")
        progress_bar.empty()
        return {}


    for i in range(MIN_TRAIN_FILES, len(historical_data) - 1):
        train_window = historical_data[:i]
        test_date, test_df = historical_data[i]
        next_date, next_df = historical_data[i+1]

        progress_text = f"Processing period {i - MIN_TRAIN_FILES + 1}/{total_steps}: Training on {len(train_window)} files..."
        progress_bar.progress((i - MIN_TRAIN_FILES + 1) / total_steps, text=progress_text)
        logging.info(f"Backtest Step {i - MIN_TRAIN_FILES + 1}/{total_steps}: Training on {len(train_window)} files (until {train_window[-1][0].date()})")

        in_sample_perf = _calculate_performance_on_window(train_window, _strategies, TRAINING_CAPITAL)

        try:
            logging.info(f"  - STARTING: Curating out-of-sample portfolio for {test_date.date()}")
            curated_port, strategy_weights, subset_weights = curate_final_portfolio(_strategies, in_sample_perf, test_df, TRAINING_CAPITAL, 30, 1.0, 10.0)
            
            strategy_weights_history.append({'date': test_date, **strategy_weights})
            subset_weights_history.append({'date': test_date, **subset_weights})

            if curated_port.empty:
                logging.warning(f"  - No curated portfolio generated for {test_date.date()}. Appending 0 return.")
                oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
            else:
                oos_perf['System_Curated']['returns'].append({'return': compute_portfolio_return(curated_port, next_df), 'date': next_date})
                logging.info(f"  - COMPLETED: Curating out-of-sample portfolio for {test_date.date()}")

                weights = curated_port['weightage_pct'] / 100
                entropy = -np.sum(weights * np.log2(weights))
                weight_entropies.append(entropy)

        except Exception as e:
            logging.error(f"OOS Curation Error ({test_date.date()}): {e}")
            oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})

        for name, strategy in _strategies.items():
            try:
                logging.info(f"  - STARTING: OOS portfolio for {name} on {test_date.date()}")
                portfolio = strategy.generate_portfolio(test_df, TRAINING_CAPITAL)
                oos_perf[name]['returns'].append({'return': compute_portfolio_return(portfolio, next_df), 'date': next_date})
                logging.info(f"  - COMPLETED: OOS portfolio for {name} on {test_date.date()}")
            except Exception as e:
                logging.error(f"OOS Strategy Error ({name}, {test_date.date()}): {e}")
                oos_perf[name]['returns'].append({'return': 0, 'date': next_date})

    progress_bar.empty()
    final_oos_perf = {name: {**data, 'metrics': calculate_advanced_metrics(data['returns'])[0]} for name, data in oos_perf.items()}

    if weight_entropies:
        final_oos_perf['System_Curated']['metrics']['avg_weight_entropy'] = np.mean(weight_entropies)

    full_history_subset_perf = _calculate_performance_on_window(historical_data, _strategies, TRAINING_CAPITAL)['subset']
    return {
        'strategy': final_oos_perf, 
        'subset': full_history_subset_perf,
        'strategy_weights_history': strategy_weights_history,
        'subset_weights_history': subset_weights_history
    }


def curate_final_portfolio(strategies: Dict[str, BaseStrategy], performance: Dict, current_df: pd.DataFrame, sip_amount: float, num_positions: int, min_pos_pct: float, max_pos_pct: float) -> Tuple[pd.DataFrame, Dict, Dict]:
    strategy_weights = calculate_strategy_weights(performance)
    subset_weights = {}
    for name in strategies:
        sub_perfs = performance.get('subset', {}).get(name, {})
        tier_names = sorted(sub_perfs.keys())
        if not tier_names:
            subset_weights[name] = {}
            continue

        tier_sharpes = np.array([sub_perfs.get(tier, 1.0 - (int(tier.split('_')[1]) * 0.05)) + 2 for tier in tier_names])
        
        if tier_sharpes.size == 0:
            subset_weights[name] = {}
            continue

        stable_sharpes = tier_sharpes - np.max(tier_sharpes)
        exp_sharpes = np.exp(stable_sharpes)
        total_exp = np.sum(exp_sharpes)

        if total_exp > 0 and np.isfinite(total_exp):
            subset_weights[name] = {tier: exp_sharpes[i] / total_exp for i, tier in enumerate(tier_names)}
        else:
            equal_weight = 1.0 / len(tier_names) if tier_names else 0
            subset_weights[name] = {tier: equal_weight for tier in tier_names}

    aggregated_holdings = {}
    for name, strategy in strategies.items():
        port = strategy.generate_portfolio(current_df, sip_amount)
        if port.empty: continue
        n, tier_size = len(port), 10
        num_tiers = n // tier_size
        if num_tiers == 0: continue
        for j in range(num_tiers):
            tier_name = f'tier_{j+1}'
            if tier_name not in subset_weights.get(name, {}): continue
            sub_df = port.iloc[j*tier_size:(j+1)*tier_size]
            tier_weight = subset_weights[name][tier_name]
            for _, row in sub_df.iterrows():
                symbol, price, weight_pct = row['symbol'], row['price'], row['weightage_pct']
                final_weight = (weight_pct / 100) * tier_weight * strategy_weights.get(name, 0)
                if symbol in aggregated_holdings: aggregated_holdings[symbol]['weight'] += final_weight
                else: aggregated_holdings[symbol] = {'price': price, 'weight': final_weight}
    if not aggregated_holdings: 
        return pd.DataFrame(), {}, {}
        
    final_port = pd.DataFrame([{'symbol': s, **d} for s, d in aggregated_holdings.items()]).sort_values('weight', ascending=False).head(num_positions)
    total_weight = final_port['weight'].sum()
    final_port['weightage_pct'] = final_port['weight'] * 100 / total_weight
    final_port['weightage_pct'] = final_port['weightage_pct'].clip(lower=min_pos_pct, upper=max_pos_pct)
    final_port['weightage_pct'] = (final_port['weightage_pct'] / final_port['weightage_pct'].sum()) * 100
    final_port['units'] = np.floor((sip_amount * final_port['weightage_pct'] / 100) / final_port['price'])
    final_port['value'] = final_port['units'] * final_port['price']
    
    final_port_df = final_port.sort_values('weightage_pct', ascending=False).reset_index(drop=True)
    return final_port_df, strategy_weights, subset_weights

# --- NEW: Production-Grade Market Regime Detection System (v2 - Corrected Logic) ---
class MarketRegimeDetectorV2:
    """
    Institutional-grade market regime detection (v2) with corrected scoring and
    classification logic.
    """
    
    def __init__(self):
        self.regime_thresholds = {
            'CRISIS': {'score': -1.0, 'confidence': 0.85},
            'BEAR': {'score': -0.5, 'confidence': 0.75},
            'WEAK_BEAR': {'score': -0.1, 'confidence': 0.65},
            'CHOP': {'score': 0.1, 'confidence': 0.60},
            'WEAK_BULL': {'score': 0.5, 'confidence': 0.65},
            'BULL': {'score': 1.0, 'confidence': 0.75},
            'STRONG_BULL': {'score': 1.5, 'confidence': 0.85},
        }
    
    def detect_regime(self, historical_data: list) -> Tuple[str, str, float, Dict]:
        if len(historical_data) < 10:
            return "INSUFFICIENT_DATA", "🐂 Bull Market Mix", 0.3, {}
        
        analysis_window = historical_data[-10:]
        latest_date, latest_df = analysis_window[-1]
        
        metrics = {
            'momentum': self._analyze_momentum_regime(analysis_window),
            'trend': self._analyze_trend_quality(analysis_window),
            'breadth': self._analyze_market_breadth(latest_df),
            'volatility': self._analyze_volatility_regime(analysis_window),
            'extremes': self._analyze_statistical_extremes(latest_df),
            'correlation': self._analyze_correlation_regime(latest_df),
            'velocity': self._analyze_velocity(analysis_window)
        }
        
        regime_score = self._calculate_composite_score(metrics)
        regime_name, confidence = self._classify_regime(regime_score, metrics)
        mix_name = self._map_regime_to_mix(regime_name)
        explanation = self._generate_explanation(regime_name, confidence, metrics, regime_score)
        
        return regime_name, mix_name, confidence, {
            'score': regime_score,
            'metrics': metrics,
            'explanation': explanation,
            'analysis_date': latest_date.strftime('%Y-%m-%d')
        }

    def _analyze_momentum_regime(self, window: list) -> Dict:
        rsi_values = [df['rsi latest'].mean() for _, df in window]
        osc_values = [df['osc latest'].mean() for _, df in window]
        
        current_rsi = rsi_values[-1]
        rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
        current_osc = osc_values[-1]
        osc_trend = np.polyfit(range(len(osc_values)), osc_values, 1)[0]
        
        if current_rsi > 65 and rsi_trend > 0.5:
            strength, score = 'STRONG_BULLISH', 2.0
        elif current_rsi > 55 and rsi_trend >= 0:
            strength, score = 'BULLISH', 1.0
        elif current_rsi < 35 and rsi_trend < -0.5:
            strength, score = 'STRONG_BEARISH', -2.0
        elif current_rsi < 45 and rsi_trend <= 0:
            strength, score = 'BEARISH', -1.0
        else:
            strength, score = 'NEUTRAL', 0.0
            
        return {'strength': strength, 'score': score, 'current_rsi': current_rsi, 'rsi_trend': rsi_trend, 'current_osc': current_osc, 'osc_trend': osc_trend}

    def _analyze_trend_quality(self, window: list) -> Dict:
        above_ma200_pct = [(df['price'] > df['ma200 latest']).mean() for _, df in window]
        ma_alignment = [(df['ma90 latest'] > df['ma200 latest']).mean() for _, df in window]
        
        current_above_200 = above_ma200_pct[-1]
        current_alignment = ma_alignment[-1]
        trend_consistency = np.polyfit(range(len(above_ma200_pct)), above_ma200_pct, 1)[0]
        
        if current_above_200 > 0.75 and current_alignment > 0.70 and trend_consistency >= 0:
            quality, score = 'STRONG_UPTREND', 2.0
        elif current_above_200 > 0.60 and current_alignment > 0.55:
            quality, score = 'UPTREND', 1.0
        elif current_above_200 < 0.30 and current_alignment < 0.30 and trend_consistency < 0:
            quality, score = 'STRONG_DOWNTREND', -2.0
        elif current_above_200 < 0.45 and current_alignment < 0.45:
            quality, score = 'DOWNTREND', -1.0
        else:
            quality, score = 'TRENDLESS', 0.0
            
        return {'quality': quality, 'score': score, 'above_200dma': current_above_200, 'ma_alignment': current_alignment, 'trend_consistency': trend_consistency}

    def _analyze_market_breadth(self, df: pd.DataFrame) -> Dict:
        rsi_bullish = (df['rsi latest'] > 50).mean()
        osc_positive = (df['osc latest'] > 0).mean()
        rsi_weak = (df['rsi latest'] < 40).mean()
        osc_oversold = (df['osc latest'] < -50).mean()
        divergence = abs(rsi_bullish - osc_positive)
        
        if rsi_bullish > 0.70 and osc_positive > 0.60 and divergence < 0.15:
            quality, score = 'STRONG_BROAD', 2.0
        elif rsi_bullish > 0.55 and osc_positive > 0.45:
            quality, score = 'HEALTHY', 1.0
        elif rsi_weak > 0.60 and osc_oversold > 0.50:
            quality, score = 'CAPITULATION', -2.0
        elif rsi_weak > 0.45 and osc_oversold > 0.35:
            quality, score = 'WEAK', -1.0
        elif divergence > 0.25:
            quality, score = 'DIVERGENT', -0.5
        else:
            quality, score = 'MIXED', 0.0
            
        return {'quality': quality, 'score': score, 'rsi_bullish_pct': rsi_bullish, 'osc_positive_pct': osc_positive, 'divergence': divergence}

    def _analyze_volatility_regime(self, window: list) -> Dict:
        bb_widths = [((4 * df['dev20 latest']) / (df['ma20 latest'] + 1e-6)).mean() for _, df in window]
        current_bbw = bb_widths[-1]
        vol_trend = np.polyfit(range(len(bb_widths)), bb_widths, 1)[0]
        
        if current_bbw < 0.08 and vol_trend < 0:
            regime, score = 'SQUEEZE', 0.5 
        elif current_bbw > 0.15 and vol_trend > 0:
            regime, score = 'PANIC', -1.0 
        elif current_bbw > 0.12:
            regime, score = 'ELEVATED', -0.5
        else:
            regime, score = 'NORMAL', 0.0
            
        return {'regime': regime, 'score': score, 'current_bbw': current_bbw, 'vol_trend': vol_trend}

    def _analyze_statistical_extremes(self, df: pd.DataFrame) -> Dict:
        extreme_oversold = (df['zscore latest'] < -2.0).mean()
        extreme_overbought = (df['zscore latest'] > 2.0).mean()
        
        if extreme_oversold > 0.40:
            extreme_type, score = 'DEEPLY_OVERSOLD', 1.5 
        elif extreme_overbought > 0.40:
            extreme_type, score = 'DEEPLY_OVERBOUGHT', -1.5
        elif extreme_oversold > 0.20:
            extreme_type, score = 'OVERSOLD', 0.75
        elif extreme_overbought > 0.20:
            extreme_type, score = 'OVERBOUGHT', -0.75
        else:
            extreme_type, score = 'NORMAL', 0.0
            
        return {'type': extreme_type, 'score': score, 'zscore_extreme_oversold_pct': extreme_oversold, 'zscore_extreme_overbought_pct': extreme_overbought}

    def _analyze_correlation_regime(self, df: pd.DataFrame) -> Dict:
        """
        Analyze cross-sectional correlation structure.
        
        High correlation (herding) often precedes market stress.
        Low correlation indicates stock-picking environment.
        
        Mathematical approach: Compute pairwise correlation proxy via 
        indicator agreement across the cross-section.
        """
        # Cross-sectional correlation proxy via indicator agreement
        # When indicators agree across stocks, correlation is high
        rsi_median = df['rsi latest'].median()
        osc_median = df['osc latest'].median()
        
        # Fraction of stocks on same side of median (herding measure)
        rsi_above = (df['rsi latest'] > rsi_median).mean()
        rsi_agreement = max(rsi_above, 1 - rsi_above)  # Closer to 1 = more agreement
        
        osc_above = (df['osc latest'] > osc_median).mean()
        osc_agreement = max(osc_above, 1 - osc_above)
        
        # Cross-indicator agreement (both oversold or both overbought)
        both_oversold = ((df['rsi latest'] < 40) & (df['osc latest'] < -30)).mean()
        both_overbought = ((df['rsi latest'] > 60) & (df['osc latest'] > 30)).mean()
        indicator_agreement = both_oversold + both_overbought
        
        # Dispersion as inverse correlation proxy
        rsi_dispersion = df['rsi latest'].std() / 50  # Normalized
        osc_dispersion = df['osc latest'].std() / 100
        avg_dispersion = (rsi_dispersion + osc_dispersion) / 2
        
        # Combined correlation score (0 = dispersed, 1 = correlated)
        correlation_score = (rsi_agreement + osc_agreement) / 2 * (1 - avg_dispersion) + indicator_agreement * 0.3
        correlation_score = np.clip(correlation_score, 0, 1)
        
        if correlation_score > 0.7:
            regime, score = 'HIGH_CORRELATION', -0.5  # High corr often precedes stress
        elif correlation_score < 0.4:
            regime, score = 'LOW_CORRELATION', 0.5  # Good for stock picking
        else:
            regime, score = 'NORMAL', 0.0
            
        return {
            'regime': regime, 
            'score': score, 
            'correlation_score': correlation_score,
            'dispersion': avg_dispersion,
            'indicator_agreement': indicator_agreement
        }

    def _analyze_velocity(self, window: list) -> Dict:
        """
        Analyze momentum velocity and acceleration.
        
        Velocity: First derivative of RSI (rate of change)
        Acceleration: Second derivative (rate of change of velocity)
        
        Positive acceleration with positive velocity = strengthening momentum
        Negative acceleration with positive velocity = momentum fading
        """
        if len(window) < 5: 
            return {'acceleration': 'UNKNOWN', 'score': 0.0, 'avg_velocity': 0.0, 'acceleration_value': 0.0}
        
        recent_rsis = np.array([w[1]['rsi latest'].mean() for w in window[-5:]])
        
        # Velocity: First differences (first derivative)
        velocity = np.diff(recent_rsis)  # 4 values
        avg_velocity = np.mean(velocity)
        current_velocity = velocity[-1]
        
        # Acceleration: Second differences (second derivative)
        acceleration_values = np.diff(velocity)  # 3 values
        avg_acceleration = np.mean(acceleration_values)
        current_acceleration = acceleration_values[-1]
        
        # Classification based on velocity and acceleration
        if avg_velocity > 1.5 and current_acceleration > 0:
            velocity_regime, score = 'ACCELERATING_UP', 1.5
        elif avg_velocity > 1.0 and current_acceleration >= 0:
            velocity_regime, score = 'RISING_FAST', 1.0
        elif avg_velocity > 0.5:
            velocity_regime, score = 'RISING', 0.5
        elif avg_velocity < -1.5 and current_acceleration < 0:
            velocity_regime, score = 'ACCELERATING_DOWN', -1.5
        elif avg_velocity < -1.0 and current_acceleration <= 0:
            velocity_regime, score = 'FALLING_FAST', -1.0
        elif avg_velocity < -0.5:
            velocity_regime, score = 'FALLING', -0.5
        elif abs(avg_velocity) < 0.5 and abs(current_acceleration) > 0.5:
            # Momentum building from stable base
            if current_acceleration > 0:
                velocity_regime, score = 'COILING_UP', 0.3
            else:
                velocity_regime, score = 'COILING_DOWN', -0.3
        else:
            velocity_regime, score = 'STABLE', 0.0
            
        return {
            'acceleration': velocity_regime, 
            'score': score, 
            'avg_velocity': avg_velocity,
            'current_velocity': current_velocity,
            'acceleration_value': current_acceleration
        }

    def _calculate_composite_score(self, metrics: Dict) -> float:
        weights = { 'momentum': 0.30, 'trend': 0.25, 'breadth': 0.15, 'volatility': 0.05, 'extremes': 0.10, 'correlation': 0.0, 'velocity': 0.15 }
        return sum(metrics[factor]['score'] * weight for factor, weight in weights.items())
    
    def _classify_regime(self, score: float, metrics: Dict) -> Tuple[str, float]:
        if metrics['volatility']['regime'] == 'PANIC' and score < -0.5 and metrics['breadth']['quality'] == 'CAPITULATION':
            return 'CRISIS', 0.90
            
        sorted_thresholds = sorted(self.regime_thresholds.items(), key=lambda item: item[1]['score'])
        
        for regime, threshold in reversed(sorted_thresholds):
            if score >= threshold['score']:
                confidence = threshold['confidence'] * 0.75 if metrics['breadth']['quality'] == 'DIVERGENT' else threshold['confidence']
                return regime, confidence

        return 'CRISIS', 0.85
    
    def _map_regime_to_mix(self, regime: str) -> str:
        mapping = {
            'STRONG_BULL': 'Bull Market Mix', 'BULL': 'Bull Market Mix',
            'WEAK_BULL': 'Chop/Consolidate Mix', 'CHOP': 'Chop/Consolidate Mix',
            'WEAK_BEAR': 'Chop/Consolidate Mix', 'BEAR': 'Bear Market Mix',
            'CRISIS': 'Bear Market Mix'
        }
        return mapping.get(regime, 'Chop/Consolidate Mix')
    
    def _generate_explanation(self, regime: str, confidence: float, metrics: Dict, score: float) -> str:
        lines = [f"**Detected Regime:** {regime} (Score: {score:.2f}, Confidence: {confidence:.0%})", ""]
        rationales = {
            'STRONG_BULL': "Strong upward momentum with broad participation. Favor momentum strategies.",
            'BULL': "Positive trend with healthy breadth. Conditions support growth strategies.",
            'WEAK_BULL': "Uptrend showing signs of fatigue or divergence. Rotate to defensive positions.",
            'CHOP': "No clear directional bias. Favors mean reversion and relative value strategies.",
            'WEAK_BEAR': "Downtrend developing. Begin defensive positioning.",
            'BEAR': "Established downtrend with weak breadth. Favor defensive strategies.",
            'CRISIS': "Severe market stress. Focus on capital preservation and oversold opportunities."
        }
        lines.append(f"**Rationale:** {rationales.get(regime, 'Market conditions unclear.')}")
        if metrics['breadth']['quality'] == 'DIVERGENT':
            lines.append("⚠️ **Warning:** Breadth divergence detected - narrow leadership may not be sustainable.")
        lines.append("\n**Key Factors:**")
        lines.append(f"• **Momentum:** {metrics['momentum']['strength']} (RSI: {metrics['momentum']['current_rsi']:.1f})")
        lines.append(f"• **Trend:** {metrics['trend']['quality']} ({metrics['trend']['above_200dma']:.0%} > 200DMA)")
        lines.append(f"• **Breadth:** {metrics['breadth']['quality']} ({metrics['breadth']['rsi_bullish_pct']:.0%} bullish)")
        lines.append(f"• **Volatility:** {metrics['volatility']['regime']} (BBW: {metrics['volatility']['current_bbw']:.3f})")
        if metrics['extremes']['type'] != 'NORMAL':
            lines.append(f"• **Extremes:** {metrics['extremes']['type']} detected")
        return "\n".join(lines)

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_mix_suggestion_v3(end_date: datetime) -> Tuple[str, str, float, Dict]:
    detector = MarketRegimeDetectorV2()
    regime_days_to_fetch = int(MAX_INDICATOR_PERIOD * 1.5) + 30 
    fetch_start_date = end_date - timedelta(days=regime_days_to_fetch)
    
    try:
        historical_data = generate_historical_data(
            symbols_to_process=SYMBOLS_UNIVERSE,
            start_date=fetch_start_date,
            end_date=end_date
        )
        
        if len(historical_data) < 10:
            return (
                "Bull Market Mix",
                "⚠️ Insufficient historical data (< 10 periods). Defaulting to Bull Mix.",
                0.30, {}
            )
            
        regime_name, mix_name, confidence, details = detector.detect_regime(historical_data)
        return mix_name, details['explanation'], confidence, details

    except Exception as e:
        logging.error(f"Error in get_market_mix_suggestion_v3: {e}")
        return (
            "Bull Market Mix",
            f"⚠️ Error during regime detection: {e}. Defaulting to Bull Mix.",
            0.30, {}
        )


# --- UI & Visualization Functions ---
def plot_weight_evolution(weight_history: List[Dict], title: str, y_axis_title: str):
    if not weight_history:
        st.warning(f"No data available for {title}.")
        return

    df = pd.DataFrame(weight_history)
    if 'date' not in df.columns: return

    id_vars = ['date']
    value_vars = [col for col in df.columns if col not in id_vars]
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Category', value_name='Weight')

    fig = px.area(df_melted, x='date', y='Weight', color='Category', title=title,
                  labels={'Weight': y_axis_title, 'date': 'Date', 'Category': 'Category'})
    fig.update_layout(template='plotly_dark', yaxis_tickformat=".0%")
    st.plotly_chart(fig, width='stretch')

def display_performance_metrics(performance: Dict):
    """Performance Analytics - Clean institutional layout."""
    if not performance:
        st.warning("Performance data not available. Please run an analysis.")
        return

    # Extract System_Curated metrics
    curated_data = performance.get('strategy', {}).get('System_Curated', {})
    curated_metrics = curated_data.get('metrics', {})
    curated_returns = curated_data.get('returns', [])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RETURNS & RISK-ADJUSTED METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    ann_ret = curated_metrics.get('annual_return', 0)
    total_ret = curated_metrics.get('total_return', 0)
    volatility = curated_metrics.get('volatility', 0)
    max_dd = curated_metrics.get('max_drawdown', 0)
    sharpe = curated_metrics.get('sharpe', 0)
    sortino = curated_metrics.get('sortino', 0)
    
    col1.metric("CAGR", f"{ann_ret:.1%}")
    col2.metric("Total Return", f"{total_ret:.1%}")
    col3.metric("Volatility", f"{volatility:.1%}")
    col4.metric("Max Drawdown", f"{max_dd:.1%}")
    col5.metric("Sharpe", f"{sharpe:.2f}")
    col6.metric("Sortino", f"{sortino:.2f}")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EQUITY CURVE & DRAWDOWN
    # ═══════════════════════════════════════════════════════════════════════════
    if curated_returns:
        df_returns = pd.DataFrame(curated_returns).sort_values('date')
        
        if UNIFIED_CHARTS_AVAILABLE:
            fig = create_equity_drawdown_chart(df_returns, date_col='date', return_col='return')
        else:
            # Fallback chart creation with proper y-axis range
            df_returns['equity'] = (1 + df_returns['return']).cumprod()
            df_returns['peak'] = df_returns['equity'].expanding().max()
            df_returns['drawdown'] = (df_returns['equity'] / df_returns['peak']) - 1
            
            equity_min = df_returns['equity'].min()
            equity_max = df_returns['equity'].max()
            y_padding = (equity_max - equity_min) * 0.1
            y_min = max(0.8, equity_min - y_padding)
            y_max = equity_max + y_padding
            
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.12,
                row_heights=[0.7, 0.3]
            )
            
            # Baseline for fill
            fig.add_trace(go.Scatter(
                x=df_returns['date'], y=[y_min] * len(df_returns), 
                mode='lines', name='_baseline', showlegend=False,
                line=dict(color='rgba(0,0,0,0)', width=0)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_returns['date'], y=df_returns['equity'], 
                mode='lines', name='Portfolio',
                line=dict(color=COLORS['primary'], width=2.5),
                fill='tonexty', fillcolor=f'rgba(255, 195, 0, 0.15)'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_returns['date'], y=df_returns['peak'], 
                mode='lines', name='High Water Mark',
                line=dict(color=COLORS['muted'], width=1.5, dash='dot')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_returns['date'], y=df_returns['drawdown'], 
                mode='lines', name='Drawdown',
                fill='tozeroy',
                line=dict(color=COLORS['danger'], width=1.5),
                fillcolor='rgba(239, 68, 68, 0.35)'
            ), row=2, col=1)
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor=COLORS['card'],
                height=480,
                showlegend=True,
                legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
                font=dict(family='Inter', color=COLORS['text']),
                margin=dict(l=60, r=20, t=40, b=40)
            )
            fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'])
            fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'], title_text="Portfolio Value", row=1, col=1, range=[y_min, y_max])
            fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'], title_text="Drawdown", tickformat='.0%', row=2, col=1)
        
        st.plotly_chart(fig, width="stretch")

    # ═══════════════════════════════════════════════════════════════════════════
    # ADDITIONAL METRICS ROW
    # ═══════════════════════════════════════════════════════════════════════════
    calmar = curated_metrics.get('calmar', 0)
    omega = curated_metrics.get('omega_ratio', 1)
    win_rate = curated_metrics.get('win_rate', 0)
    profit_factor = curated_metrics.get('profit_factor', 1)
    tail_ratio = curated_metrics.get('tail_ratio', 1)
    gain_to_pain = curated_metrics.get('gain_to_pain', 0)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Calmar", f"{calmar:.2f}")
    col2.metric("Omega", f"{omega:.2f}")
    col3.metric("Win Rate", f"{win_rate:.0%}")
    col4.metric("Profit Factor", f"{profit_factor:.2f}")
    col5.metric("Tail Ratio", f"{tail_ratio:.2f}")
    col6.metric("Gain/Pain", f"{gain_to_pain:.2f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # ROLLING SHARPE/SORTINO
    # ═══════════════════════════════════════════════════════════════════════════
    if curated_returns and len(curated_returns) >= 5:
        st.markdown("##### Rolling Risk-Adjusted Performance")
        df_returns = pd.DataFrame(curated_returns).sort_values('date')
        window_size = max(3, len(df_returns) // 5)
        
        if UNIFIED_CHARTS_AVAILABLE:
            fig_rolling = create_rolling_metrics_chart(df_returns, window=window_size, date_col='date', return_col='return')
        else:
            rolling_mean = df_returns['return'].rolling(window=window_size).mean()
            rolling_std = df_returns['return'].rolling(window=window_size).std()
            rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(52)
            
            downside_returns = df_returns['return'].apply(lambda x: x if x < 0 else 0)
            rolling_downside = downside_returns.rolling(window=window_size).std()
            rolling_sortino = (rolling_mean / rolling_downside.replace(0, np.nan)) * np.sqrt(52)
            
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(
                x=df_returns['date'], y=rolling_sharpe,
                mode='lines', name=f'Sharpe ({window_size}w)',
                line=dict(color=COLORS['primary'], width=2)
            ))
            fig_rolling.add_trace(go.Scatter(
                x=df_returns['date'], y=rolling_sortino,
                mode='lines', name=f'Sortino ({window_size}w)',
                line=dict(color=COLORS['success'], width=2)
            ))
            
            fig_rolling.add_hline(y=0, line_dash="dash", line_color=COLORS['muted'], line_width=1)
            fig_rolling.add_hline(y=1, line_dash="dot", line_color=COLORS['success'], line_width=1)
            fig_rolling.add_hline(y=2, line_dash="dot", line_color=COLORS['warning'], line_width=1)
            
            fig_rolling.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor=COLORS['card'],
                yaxis_title="Ratio",
                height=320,
                font=dict(family='Inter', color=COLORS['text']),
                legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
                margin=dict(l=60, r=20, t=40, b=40)
            )
            fig_rolling.update_xaxes(showgrid=True, gridcolor=COLORS['border'])
            fig_rolling.update_yaxes(showgrid=True, gridcolor=COLORS['border'], zeroline=True, zerolinecolor=COLORS['muted'])
        
        st.plotly_chart(fig_rolling, width="stretch")

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("##### Strategy Attribution")
    
    strategy_data = []
    for name, perf in performance.get('strategy', {}).items():
        metrics = perf.get('metrics', {})
        strategy_data.append({
            'Strategy': name,
            'CAGR': metrics.get('annual_return', 0),
            'Vol': metrics.get('volatility', 0),
            'Sharpe': metrics.get('sharpe', 0),
            'Sortino': metrics.get('sortino', 0),
            'Max DD': metrics.get('max_drawdown', 0),
            'Win Rate': metrics.get('win_rate', 0)
        })
    
    if strategy_data:
        df_strategies = pd.DataFrame(strategy_data)
        df_strategies = df_strategies.sort_values('Sharpe', ascending=False)
        
        df_display = df_strategies.copy()
        df_display['CAGR'] = df_display['CAGR'].apply(lambda x: f"{x:.1%}")
        df_display['Vol'] = df_display['Vol'].apply(lambda x: f"{x:.1%}")
        df_display['Sharpe'] = df_display['Sharpe'].apply(lambda x: f"{x:.2f}")
        df_display['Sortino'] = df_display['Sortino'].apply(lambda x: f"{x:.2f}")
        df_display['Max DD'] = df_display['Max DD'].apply(lambda x: f"{x:.1%}")
        df_display['Win Rate'] = df_display['Win Rate'].apply(lambda x: f"{x:.0%}")
        
        st.dataframe(df_display, width="stretch", hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # CORRELATION MATRIX
    # ═══════════════════════════════════════════════════════════════════════════
    returns_df = pd.DataFrame()
    for name, perf in performance.get('strategy', {}).items():
        if perf.get('returns'):
            df_raw = pd.DataFrame(perf['returns'])
            df = df_raw.drop_duplicates(subset='date', keep='last').set_index('date')
            returns_df[name] = df['return']

    if not returns_df.empty and len(returns_df.columns) > 1:
        st.markdown("##### Strategy Correlation")
        corr_matrix = returns_df.corr()
        
        if UNIFIED_CHARTS_AVAILABLE:
            fig_corr = create_correlation_heatmap(corr_matrix, title="")
        else:
            colorscale = [
                [0.0, '#3b82f6'], [0.25, '#60a5fa'],
                [0.5, COLORS['muted']],
                [0.75, '#f87171'], [1.0, '#ef4444']
            ]
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=colorscale,
                zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont=dict(size=10, color='white'),
                colorbar=dict(title='ρ', tickfont=dict(color=COLORS['muted']))
            ))
            fig_corr.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor=COLORS['card'],
                height=max(300, len(corr_matrix) * 30),
                font=dict(family='Inter', color=COLORS['text']),
                margin=dict(l=100, r=40, t=20, b=40)
            )
        
        st.plotly_chart(fig_corr, width="stretch")
        
        # Diversification metric - single line
        if 'System_Curated' in corr_matrix.columns:
            other_strategies = [c for c in corr_matrix.columns if c != 'System_Curated']
            if other_strategies:
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                st.caption(f"Average pairwise correlation: {avg_corr:.2f}")
    
    # Strategy Weight Evolution (if available)
    plot_weight_evolution(
        performance.get('strategy_weights_history', []),
        title="",
        y_axis_title="Weight"
    )


def create_subset_heatmap(subset_perf: Dict, strategy_options: list):
    """Create tier Sharpe heatmap with unified styling."""
    if not subset_perf: 
        return

    if UNIFIED_CHARTS_AVAILABLE:
        fig = create_tier_sharpe_heatmap(subset_perf, strategy_options)
        if fig:
            st.plotly_chart(fig, width="stretch")
            
            # Add insights
            all_values = []
            for strat in strategy_options:
                if strat in subset_perf:
                    all_values.extend([v for v in subset_perf[strat].values() if not np.isnan(v)])
            
            if all_values:
                col1, col2, col3 = st.columns(3)
                with col1:
                    best_tier = max(range(1, 11), key=lambda t: np.nanmean([
                        subset_perf.get(s, {}).get(f'tier_{t}', np.nan) for s in strategy_options
                    ]))
                    st.markdown(f"""
                    <div class="metric-card info">
                        <h4>Best Performing Tier</h4>
                        <h2>Tier {best_tier}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    avg_sharpe = np.nanmean(all_values)
                    st.markdown(f"""
                    <div class="metric-card {'success' if avg_sharpe > 0 else 'danger'}">
                        <h4>Average Tier Sharpe</h4>
                        <h2>{avg_sharpe:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    tier_dispersion = np.nanstd(all_values)
                    st.markdown(f"""
                    <div class="metric-card neutral">
                        <h4>Tier Dispersion</h4>
                        <h2>{tier_dispersion:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
        return

    # Fallback implementation
    heatmap_data = {}
    max_tier_num = 0
    for strat in strategy_options:
        if strat in subset_perf and subset_perf[strat]:
            tier_nums = [int(tier.split('_')[1]) for tier in subset_perf[strat].keys()]
            if tier_nums:
                max_tier_num = max(max_tier_num, max(tier_nums))

    if max_tier_num == 0:
        st.warning("No subset data available to display.", icon="⚠️")
        return

    for strat in strategy_options:
        row = [subset_perf.get(strat, {}).get(f'tier_{i+1}', np.nan) for i in range(max_tier_num)]
        heatmap_data[strat] = row

    df = pd.DataFrame(heatmap_data).transpose()
    df.columns = [f'Tier {i+1}' for i in range(df.shape[1])]

    fig = px.imshow(df, text_auto=".2f", aspect="auto",
                    color_continuous_scale='RdYlGn',
                    labels=dict(x="10-Stock Tier", y="Strategy", color="Sharpe Ratio"),
                    title="<b>Sharpe Ratio by 10-Stock Tier</b>")
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, width='stretch')

def display_subset_weight_evolution(subset_weights_history: List[Dict], strategies: List[str]):
    st.markdown("---")
    st.subheader("Subset Tier Weight Evolution")

    if not subset_weights_history:
        st.warning("No subset weight history available.")
        return
        
    selected_strategy = st.selectbox("Select Strategy to View Tier Weights", options=strategies)

    if selected_strategy:
        strategy_tier_history = []
        for record in subset_weights_history:
            date = record['date']
            tier_weights = record.get(selected_strategy, {})
            if tier_weights:
                row = {'date': date, **tier_weights}
                strategy_tier_history.append(row)
        
        plot_weight_evolution(
            strategy_tier_history,
            title=f"Tier Weight Evolution for {selected_strategy}",
            y_axis_title="Tier Weight"
        )


def create_conviction_heatmap(strategies, current_df):
    all_signals = []
    for name, s in strategies.items():
        port = s.generate_portfolio(current_df.copy())
        
        if port.empty:
            continue

        if 'composite_score' not in port.columns:
            port['composite_score'] = port['weightage_pct']
            
        for _, row in port.head(20).iterrows():
            all_signals.append({'symbol': row['symbol'], 'strategy': name, 'conviction': row['composite_score']})

    if not all_signals: return go.Figure()

    df = pd.DataFrame(all_signals)
    heatmap_df = df.pivot(index='symbol', columns='strategy', values='conviction').fillna(0)

    fig = px.imshow(heatmap_df, text_auto=".2f", aspect="auto",
                    color_continuous_scale='RdBu',
                    labels=dict(x="Strategy", y="Symbol", color="Conviction Score"),
                    title="<b>Strategy Conviction Scores (Top Symbols)</b>")
    fig.update_layout(template='plotly_dark', height=600)
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC STRATEGY SELECTION ENGINE v2.1
# ═══════════════════════════════════════════════════════════════════════════

if 'dynamic_strategies_cache' not in st.session_state:
    st.session_state.dynamic_strategies_cache = None

# Configure module logger
_dss_logger = logging.getLogger("Pragyam.DynamicSelection")


def _compute_backtest_metrics(daily_values: List[float], periods_per_year: float = 252.0) -> Dict[str, float]:
    """
    Compute performance metrics from daily portfolio values.
    Returns realistic, unbounded metrics for proper comparison.
    """
    result = {
        'total_return': 0.0,
        'ann_return': 0.0,
        'volatility': 0.0,
        'sharpe': 0.0,
        'sortino': 0.0,
        'calmar': 0.0,
        'max_dd': 0.0,
        'win_rate': 0.0
    }
    
    if len(daily_values) < 5:
        return result
    
    values = np.array(daily_values, dtype=np.float64)
    
    # Validate data
    if np.any(values <= 0) or np.any(~np.isfinite(values)):
        return result
    
    initial = values[0]
    final = values[-1]
    n_days = len(values)
    
    # Total Return
    total_return = (final - initial) / initial
    result['total_return'] = total_return
    
    # Daily Returns
    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    
    if len(daily_returns) < 3:
        return result
    
    # Annualized Return (CAGR)
    years = n_days / periods_per_year
    if years > 0 and final > 0 and initial > 0:
        ann_return = (final / initial) ** (1.0 / years) - 1.0
    else:
        ann_return = 0.0
    result['ann_return'] = ann_return
    
    # Volatility (annualized)
    daily_vol = np.std(daily_returns, ddof=1)
    volatility = daily_vol * np.sqrt(periods_per_year)
    result['volatility'] = volatility
    
    # Sharpe Ratio
    if volatility > 0.001:
        sharpe = ann_return / volatility
    else:
        sharpe = 0.0
    result['sharpe'] = sharpe
    
    # Sortino Ratio (downside deviation)
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) >= 2:
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(periods_per_year)
        if downside_vol > 0.001:
            sortino = ann_return / downside_vol
        else:
            sortino = ann_return * 100 if ann_return > 0 else 0  # Very low downside = good
    else:
        # No negative days - excellent performance
        sortino = ann_return * 100 if ann_return > 0 else 0
    result['sortino'] = sortino
    
    # Maximum Drawdown
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_dd = np.min(drawdowns)
    result['max_dd'] = max_dd
    
    # Calmar Ratio (annualized return / max drawdown)
    if max_dd < -0.001:  # At least 0.1% drawdown
        calmar = ann_return / abs(max_dd)
    else:
        # No meaningful drawdown
        calmar = ann_return * 100 if ann_return > 0 else 0
    result['calmar'] = calmar
    
    # Win Rate
    win_rate = np.mean(daily_returns > 0)
    result['win_rate'] = win_rate
    
    return result


def _run_dynamic_strategy_selection(
    historical_data: List[Tuple[datetime, pd.DataFrame]], 
    all_strategies: Dict[str, BaseStrategy],
    selected_style: str,
    progress_bar=None,
    status_text=None
) -> Tuple[Optional[List[str]], Dict[str, Dict]]:
    """
    Backtest all strategies and select top 4 based on performance.
    
    Selection Criteria:
    - SIP Investment: Top 4 by Calmar Ratio (drawdown recovery)
    - Swing Trading: Top 4 by Sortino Ratio (risk-adjusted returns)
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────
    
    is_sip = "SIP" in selected_style
    metric_key = 'calmar' if is_sip else 'sortino'
    metric_label = "Calmar" if is_sip else "Sortino"
    
    _dss_logger.info("=" * 70)
    _dss_logger.info("DYNAMIC STRATEGY SELECTION")
    _dss_logger.info("=" * 70)
    _dss_logger.info(f"Investment Style: {selected_style}")
    _dss_logger.info(f"Selection Metric: {metric_label} Ratio")
    
    # Validation
    if not DYNAMIC_SELECTION_AVAILABLE:
        _dss_logger.warning("backtest_engine.py not available - using static selection")
        return None, {}
    
    if not historical_data or len(historical_data) < 10:
        _dss_logger.warning(f"Insufficient data ({len(historical_data) if historical_data else 0} days) - using static selection")
        return None, {}
    
    # Extract date range
    date_start = historical_data[0][0]
    date_end = historical_data[-1][0]
    n_days = len(historical_data)
    capital = 10_000_000
    
    _dss_logger.info(f"Backtest Period: {date_start.strftime('%Y-%m-%d')} to {date_end.strftime('%Y-%m-%d')} ({n_days} days)")
    _dss_logger.info(f"Strategies to evaluate: {len(all_strategies)}")
    _dss_logger.info("-" * 70)
    
    if status_text:
        status_text.text(f"Building price matrix for {n_days} days...")
    
    # ─────────────────────────────────────────────────────────────────────
    # BUILD PRICE MATRIX
    # ─────────────────────────────────────────────────────────────────────
    
    # Collect all symbols
    all_symbols = set()
    for _, df in historical_data:
        all_symbols.update(df['symbol'].tolist())
    all_symbols = sorted(all_symbols)
    
    # Build price matrix with forward-fill
    price_matrix = {}
    for symbol in all_symbols:
        prices = []
        last_valid = np.nan
        for _, df in historical_data:
            sym_df = df[df['symbol'] == symbol]
            if not sym_df.empty and 'price' in sym_df.columns:
                price = sym_df['price'].iloc[0]
                if pd.notna(price) and price > 0:
                    last_valid = price
            prices.append(last_valid)
        price_matrix[symbol] = prices
    
    _dss_logger.info(f"Price matrix: {len(all_symbols)} symbols × {n_days} days")
    
    # ─────────────────────────────────────────────────────────────────────
    # BACKTEST EACH STRATEGY
    # ─────────────────────────────────────────────────────────────────────
    
    _dss_logger.info("-" * 70)
    _dss_logger.info("BACKTESTING STRATEGIES")
    _dss_logger.info("-" * 70)
    
    results = {}
    valid_strategies = []
    
    for idx, (name, strategy) in enumerate(all_strategies.items()):
        
        if progress_bar:
            pct = 0.25 + (idx / len(all_strategies)) * 0.35
            progress_bar.progress(pct, text=f"Backtesting: {name}")
        
        if status_text:
            status_text.text(f"Testing: {name} ({idx+1}/{len(all_strategies)})")
        
        try:
            # Generate portfolio on first day
            first_df = historical_data[0][1].copy()
            port_df = strategy.generate_portfolio(first_df, capital)
            
            if port_df is None or port_df.empty or 'units' not in port_df.columns:
                _dss_logger.debug(f"  {name}: No portfolio generated - SKIP")
                results[name] = {'status': 'skip', 'reason': 'No portfolio'}
                continue
            
            # Build holdings
            holdings = {}
            for _, row in port_df.iterrows():
                sym = row['symbol']
                units = row.get('units', 0)
                if units > 0 and sym in price_matrix:
                    first_price = price_matrix[sym][0]
                    if pd.notna(first_price) and first_price > 0:
                        holdings[sym] = units
            
            if len(holdings) == 0:
                _dss_logger.debug(f"  {name}: No valid holdings - SKIP")
                results[name] = {'status': 'skip', 'reason': 'No holdings'}
                continue
            
            # Calculate initial investment
            initial_investment = sum(
                units * price_matrix[sym][0]
                for sym, units in holdings.items()
            )
            cash = capital - initial_investment
            
            # Track daily values
            daily_values = []
            for day_idx in range(n_days):
                port_value = sum(
                    units * price_matrix[sym][day_idx]
                    for sym, units in holdings.items()
                    if pd.notna(price_matrix[sym][day_idx])
                )
                daily_values.append(port_value + cash)
            
            # Validate daily values
            if len(daily_values) < 10 or daily_values[0] <= 0:
                _dss_logger.debug(f"  {name}: Invalid daily values - SKIP")
                results[name] = {'status': 'skip', 'reason': 'Invalid values'}
                continue
            
            # Compute metrics
            metrics = _compute_backtest_metrics(daily_values)
            
            total_ret = metrics['total_return']
            max_dd = metrics['max_dd']
            sharpe = metrics['sharpe']
            sortino = metrics['sortino']
            calmar = metrics['calmar']
            score = metrics[metric_key]
            
            # Validate score
            if not np.isfinite(score):
                _dss_logger.debug(f"  {name}: Invalid {metric_key} ({score}) - SKIP")
                results[name] = {'status': 'skip', 'reason': f'Invalid {metric_key}'}
                continue
            
            # Store results
            results[name] = {
                'status': 'ok',
                'metrics': metrics,
                'score': score,
                'positions': len(holdings)
            }
            valid_strategies.append((name, score, metrics))
            
            # Log result
            _dss_logger.info(
                f"  {name:<28} │ Ret: {total_ret:>+6.1%} │ MaxDD: {max_dd:>+6.1%} │ "
                f"Sharpe: {sharpe:>+5.2f} │ Sortino: {sortino:>+6.2f} │ Calmar: {calmar:>+6.2f} │ Pos: {len(holdings)}"
            )
            
        except Exception as e:
            _dss_logger.error(f"  {name}: Error - {str(e)[:50]}")
            results[name] = {'status': 'error', 'reason': str(e)}
            continue
    
    # ─────────────────────────────────────────────────────────────────────
    # SELECT TOP 4
    # ─────────────────────────────────────────────────────────────────────
    
    _dss_logger.info("-" * 70)
    _dss_logger.info(f"SELECTION BY {metric_label.upper()} RATIO ({selected_style})")
    _dss_logger.info("-" * 70)
    
    if len(valid_strategies) < 4:
        _dss_logger.warning(f"Only {len(valid_strategies)} valid strategies (need 4) - using static selection")
        return None, results
    
    # Sort by score
    valid_strategies.sort(key=lambda x: x[1], reverse=True)
    
    # Select top 4
    top_4 = valid_strategies[:4]
    selected = [name for name, _, _ in top_4]
    
    # Log rankings
    for rank, (name, score, metrics) in enumerate(valid_strategies, 1):
        marker = ">>>" if rank <= 4 else "   "
        status = "[SELECTED]" if rank <= 4 else ""
        ret = metrics['total_return']
        _dss_logger.info(f"  {marker} #{rank:<2} {name:<28} │ {metric_label}: {score:>+7.2f} │ Return: {ret:>+6.1%} {status}")
    
    _dss_logger.info("-" * 70)
    _dss_logger.info(f"SELECTED: {', '.join(selected)}")
    _dss_logger.info("=" * 70)
    
    if status_text:
        status_text.text(f"Selected: {', '.join(selected)}")
    
    return selected, results


# --- Main Application ---
def main():
    strategies = {
        'PRStrategy': PRStrategy(),
        'CL1Strategy': CL1Strategy(),
        'CL2Strategy': CL2Strategy(),
        'CL3Strategy': CL3Strategy(),
        'MOM1Strategy': MOM1Strategy(),
        'MOM2Strategy': MOM2Strategy(),
        'MomentumMasters': MomentumMasters(),
        'VolatilitySurfer': VolatilitySurfer(),
        'AdaptiveVolBreakout': AdaptiveVolBreakout(),
        'VolReversalHarvester': VolReversalHarvester(),
        'AlphaSurge': AlphaSurge(),
        'ReturnPyramid': ReturnPyramid(),
        'MomentumCascade': MomentumCascade(),
        'AlphaVortex': AlphaVortex(),
        'SurgeSentinel': SurgeSentinel(),
        'VelocityVortex': VelocityVortex(),
        'BreakoutAlphaHunter': BreakoutAlphaHunter(),
        'ExtremeMomentumBlitz': ExtremeMomentumBlitz(),
        'HyperAlphaIgniter': HyperAlphaIgniter(),
        'VelocityApocalypse': VelocityApocalypse(),
        'QuantumMomentumLeap': QuantumMomentumLeap(),
        'NebulaMomentumStorm': NebulaMomentumStorm(),
        'ResonanceEcho': ResonanceEcho(),
        'DivergenceMirage': DivergenceMirage(),
        'FractalWhisper': FractalWhisper(),
        'InterferenceWave': InterferenceWave(),
        'ShadowPuppet': ShadowPuppet(),
        'EntangledMomentum': EntangledMomentum(),
        'ButterflyChaos': ButterflyChaos(),
        'SynapseFiring': SynapseFiring(),
        'HolographicMomentum': HolographicMomentum(),
        'WormholeTemporal': WormholeTemporal(),
        'SymbioticAlpha': SymbioticAlpha(),
        'PhononVibe': PhononVibe(),
        'HorizonEvent': HorizonEvent(),
        'EscherLoop': EscherLoop(),
        'MicrowaveCosmic': MicrowaveCosmic(),
        'SingularityMomentum': SingularityMomentum(),
        'MultiverseAlpha': MultiverseAlpha(),
        'EternalReturnCycle': EternalReturnCycle(),
        'DivineMomentumOracle': DivineMomentumOracle(),
        'CelestialAlphaForge': CelestialAlphaForge(),
        'InfiniteMomentumLoop': InfiniteMomentumLoop(),
        'GodParticleSurge': GodParticleSurge(),
        'NirvanaMomentumWave': NirvanaMomentumWave(),
        'PantheonAlphaRealm': PantheonAlphaRealm(),
        'ZenithMomentumPeak': ZenithMomentumPeak(),
        'OmniscienceReturn': OmniscienceReturn(),
        'ApotheosisMomentum': ApotheosisMomentum(),
        'TranscendentAlpha': TranscendentAlpha(),
        'TurnaroundSniper': TurnaroundSniper(),
        'MomentumAccelerator': MomentumAccelerator(),
        'VolatilityRegimeTrader': VolatilityRegimeTrader(),
        'CrossSectionalAlpha': CrossSectionalAlpha(),
        'DualMomentum': DualMomentum(),
        'AdaptiveZScoreEngine': AdaptiveZScoreEngine(),
        'MomentumDecayModel': MomentumDecayModel(),
        'InformationRatioOptimizer': InformationRatioOptimizer(),
        'BayesianMomentumUpdater': BayesianMomentumUpdater(),
        'RelativeStrengthRotator': RelativeStrengthRotator(),
        'VolatilityAdjustedValue': VolatilityAdjustedValue(),
        'NonlinearMomentumBlender': NonlinearMomentumBlender(),
        'EntropyWeightedSelector': EntropyWeightedSelector(),
        'KalmanFilterMomentum': KalmanFilterMomentum(),
        'MeanVarianceOptimizer': MeanVarianceOptimizer(),
        'RegimeSwitchingStrategy': RegimeSwitchingStrategy(),
        'FractalMomentumStrategy': FractalMomentumStrategy(),
        'CopulaBlendStrategy': CopulaBlendStrategy(),
        'WaveletDenoiser': WaveletDenoiser(),
        'GradientBoostBlender': GradientBoostBlender(),
        'AttentionMechanism': AttentionMechanism(),
        'EnsembleVotingStrategy': EnsembleVotingStrategy(),
        'OptimalTransportBlender': OptimalTransportBlender(),
        'StochasticDominance': StochasticDominance(),
        'MaximumEntropyStrategy': MaximumEntropyStrategy(),
        'HiddenMarkovModel': HiddenMarkovModel(),
        'QuantileRegressionStrategy': QuantileRegressionStrategy(),
        'MutualInformationBlender': MutualInformationBlender(),
        'GameTheoreticStrategy': GameTheoreticStrategy(),
        'ReinforcementLearningInspired': ReinforcementLearningInspired(),
        'SpectralClusteringStrategy': SpectralClusteringStrategy(),
        'CausalInferenceStrategy': CausalInferenceStrategy(),
        'BootstrapConfidenceStrategy': BootstrapConfidenceStrategy(),
        'KernelDensityStrategy': KernelDensityStrategy(),
        'SurvivalAnalysisStrategy': SurvivalAnalysisStrategy(),
        'PrincipalComponentStrategy': PrincipalComponentStrategy(),
        'FactorMomentumStrategy': FactorMomentumStrategy(),
        'ElasticNetBlender': ElasticNetBlender(),
        'RobustRegressionStrategy': RobustRegressionStrategy(),
        'ConvexOptimizationStrategy': ConvexOptimizationStrategy(),
        'MonteCarloStrategy': MonteCarloStrategy(),
        'VariationalInferenceStrategy': VariationalInferenceStrategy(),
        'NeuralNetworkInspired': NeuralNetworkInspired(),
        'GraphNeuralInspired': GraphNeuralInspired(),
        'ContrastiveLearningStrategy': ContrastiveLearningStrategy(),
    }

    # Fallback static PORTFOLIO_STYLES (used if dynamic selection fails)
    PORTFOLIO_STYLES = {
        "Swing Trading": {
            "description": "Short-term (3-21 day) holds to capture rapid momentum and volatility.",
            "mixes": {
                "Bull Market Mix": {
                    "strategies": ['GameTheoreticStrategy', 'NebulaMomentumStorm', 'VolatilitySurfer', 'CelestialAlphaForge'],
                    "rationale": "Dynamically selected based on highest Sortino Ratio from backtest results."
                },
                
                "Bear Market Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Dynamically selected based on highest Sortino Ratio from backtest results."
                },
                
                "Chop/Consolidate Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Dynamically selected based on highest Sortino Ratio from backtest results."
                }
            }
        },
        
        "SIP Investment": {
            "description": "Systematic long-term (3-12+ months) wealth accumulation. Focus on consistency and drawdown protection.",
            "mixes": {
                "Bull Market Mix": {
                    "strategies": ['GameTheoreticStrategy', 'MomentumAccelerator', 'VolatilitySurfer', 'DivineMomentumOracle'],
                    "rationale": "Dynamically selected based on highest Calmar Ratio from backtest results."
                },
                
                "Bear Market Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Dynamically selected based on highest Calmar Ratio from backtest results."
                },
                
                "Chop/Consolidate Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Dynamically selected based on highest Calmar Ratio from backtest results."
                }
            }
        }
    }
    
    def update_regime_suggestion():
        """
        Called when the analysis date changes. Fetches *just enough*
        data to run the regime model and updates the session state.
        """
        selected_date_obj = st.session_state.get('analysis_date_str') 
        if not selected_date_obj:
            return
            
        selected_date = datetime.combine(selected_date_obj, datetime.min.time())
        
        toast_msg = f"Fetching regime data for {selected_date.date()}..."
        st.toast(toast_msg, icon="🧠")
        
        mix_name, explanation, confidence, details = get_market_mix_suggestion_v3(selected_date)
        
        st.session_state.suggested_mix = mix_name
        
        # --- NEW: Store detailed regime info for sidebar display ---
        st.session_state.regime_display = {
            'mix': mix_name,
            'confidence': confidence,
            'explanation': explanation
        }


    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">PRAGYAM</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">प्रज्ञम | Portfolio Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">📅 Analysis Configuration</div>', unsafe_allow_html=True)
        
        today = datetime.now()
        selected_date_str = st.date_input(
            "Select Analysis Date",
            value=today,
            min_value=today - timedelta(days=5*365),
            max_value=today,
            help="Choose a date to run the portfolio curation.",
            key='analysis_date_str',
            on_change=update_regime_suggestion 
        )

        # --- NEW: Dynamic Market Regime Info Card ---
        # Trigger initial calculation if needed
        if st.session_state.suggested_mix is None:
             update_regime_suggestion()

        if st.session_state.regime_display:
            data = st.session_state.regime_display
            # Using HTML/CSS to blend with the existing sidebar UI (metric-card/info-box style)
            st.markdown(f"""
            <div style="background-color: var(--secondary-background-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; margin: 10px 0 20px 0; border-left: 0px solid var(--primary-color); box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 4px;">Market Regime</div>
                <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 700; line-height: 1.2;">{data['mix']}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                    <span style="color: var(--text-muted); font-size: 0.8rem;">Confidence</span>
                    <span style="color: var(--primary-color); font-weight: 600; font-size: 0.8rem;">{data['confidence']:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        # --- END NEW CARD ---
        
        st.markdown('<div class="sidebar-title">💼 Portfolio Style</div>', unsafe_allow_html=True)

        options_list = list(PORTFOLIO_STYLES.keys())
        default_index = 0 
        if "SIP Investment" in options_list:
            default_index = options_list.index("SIP Investment")

        selected_main_branch = st.selectbox(
            "1. Select Investment Style",
            options=options_list,
            index=default_index,
            help="Choose your primary investment objective (e.g., short-term trading or long-term investing)."
        )
        
        mix_options = list(PORTFOLIO_STYLES[selected_main_branch]["mixes"].keys())
        
        st.markdown('<div class="sidebar-title">⚙️ Portfolio Parameters</div>', unsafe_allow_html=True)
        capital = st.number_input("Capital (₹)", 1000, 100000000, 2500000, 1000, help="Total capital to allocate")
        num_positions = st.slider("Number of Positions", 5, 100, 30, 5, help="Maximum positions in the final portfolio")

        if st.button("Run Analysis", width='stretch', type="primary"):
            
            lookback_files = 25
            
            selected_date_obj = st.session_state.get('analysis_date_str')
            if not selected_date_obj:
                st.error("Analysis date is missing. Please select a date.")
                st.stop()
                
            selected_date_dt = datetime.combine(selected_date_obj, datetime.min.time())

            # --- Create Progress Tracking UI ---
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()

            total_days_to_fetch = int((lookback_files + MAX_INDICATOR_PERIOD) * 1.5) + 30
            fetch_start_date = selected_date_dt - timedelta(days=total_days_to_fetch)
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 1: DATA FETCHING
            # ═══════════════════════════════════════════════════════════════════
            progress_bar.progress(0.05, text="Fetching market data...")
            status_text.text(f"Downloading {len(SYMBOLS_UNIVERSE)} symbols")
            
            logging.info("=" * 70)
            logging.info("PRAGYAM ANALYSIS ENGINE v2.1")
            logging.info("=" * 70)
            logging.info(f"[PHASE 1/4] DATA FETCHING")
            logging.info(f"  Symbols: {len(SYMBOLS_UNIVERSE)}")
            logging.info(f"  Period: {fetch_start_date.date()} to {selected_date_dt.date()}")
            
            all_historical_data = load_historical_data(selected_date_dt, lookback_files)
            
            if not all_historical_data:
                progress_bar.empty()
                status_text.empty()
                st.error("Application cannot start: No historical data could be loaded or generated.")
                st.stop()

            progress_bar.progress(0.20, text="Data loaded. Preparing...")
            logging.info(f"  Result: {len(all_historical_data)} trading days loaded")

            current_date, current_df = all_historical_data[-1]
            training_data = all_historical_data[:-1]
            
            if len(training_data) > lookback_files:
                training_data_window = training_data[-lookback_files:]
            else:
                training_data_window = training_data
            
            training_data_window_with_current = training_data_window + [(current_date, current_df)]
            
            st.session_state.current_df = current_df
            st.session_state.selected_date = current_date.strftime('%Y-%m-%d')
            
            if len(training_data_window_with_current) < 10:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Not enough training data loaded ({len(training_data_window_with_current)} days). Need at least 10. Check date range or lookback period.")
                st.stop()
                
            if not st.session_state.suggested_mix:
                progress_bar.empty()
                status_text.empty()
                st.error("Market regime could not be determined. Please select a valid date. Analysis cannot run.")
                st.stop()
                
            final_mix_to_use = st.session_state.suggested_mix
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 2: DYNAMIC STRATEGY SELECTION
            # ═══════════════════════════════════════════════════════════════════
            progress_bar.progress(0.25, text="Running strategy selection...")
            status_text.text(f"Analyzing {len(strategies)} strategies...")
            
            logging.info("-" * 70)
            logging.info(f"[PHASE 2/4] DYNAMIC STRATEGY SELECTION")
            logging.info(f"  Investment Style: {selected_main_branch}")
            logging.info(f"  Market Regime: {final_mix_to_use}")
            
            dynamic_strategies, strategy_metrics = _run_dynamic_strategy_selection(
                training_data_window_with_current, 
                strategies, 
                selected_main_branch,
                progress_bar=progress_bar,
                status_text=status_text
            )
            
            # Determine which strategies to use
            if dynamic_strategies and len(dynamic_strategies) >= 4:
                style_strategies = dynamic_strategies
                selection_mode = "DYNAMIC"
                logging.info(f"  Mode: DYNAMIC - Selected {len(dynamic_strategies)} strategies")
                st.toast(f"Selected: {', '.join(style_strategies[:2])}...", icon="✅")
            else:
                style_strategies = PORTFOLIO_STYLES[selected_main_branch]["mixes"][final_mix_to_use]['strategies']
                selection_mode = "STATIC"
                logging.info(f"  Mode: STATIC (fallback) - Using predefined strategies")
                st.toast(f"Using default strategies", icon="ℹ️")
            
            # Filter to only available strategies
            strategies_to_run = {name: strategies[name] for name in style_strategies if name in strategies}
            
            if not strategies_to_run:
                progress_bar.empty()
                status_text.empty()
                st.error(f"None of the selected strategies are available: {style_strategies}")
                st.stop()
            
            logging.info(f"  Strategies for execution: {list(strategies_to_run.keys())}")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 3: WALK-FORWARD EVALUATION
            # ═══════════════════════════════════════════════════════════════════
            progress_bar.progress(0.65, text="Running walk-forward evaluation...")
            status_text.text(f"Evaluating {len(strategies_to_run)} strategies...")
            
            logging.info("-" * 70)
            logging.info(f"[PHASE 3/4] WALK-FORWARD EVALUATION")
            logging.info(f"  Strategies: {list(strategies_to_run.keys())}")
            logging.info(f"  Window: {len(training_data_window_with_current)} days")
            
            st.session_state.performance = evaluate_historical_performance(strategies_to_run, training_data_window_with_current)
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 4: PORTFOLIO CURATION
            # ═══════════════════════════════════════════════════════════════════
            if st.session_state.performance:
                progress_bar.progress(0.90, text="Curating final portfolio...")
                status_text.text("Optimizing position weights...")
                
                logging.info("-" * 70)
                logging.info(f"[PHASE 4/4] PORTFOLIO CURATION")
                logging.info(f"  Capital: ₹{capital:,}")
                logging.info(f"  Max Positions: {num_positions}")
                
                st.session_state.portfolio, _, _ = curate_final_portfolio(
                    strategies_to_run,
                    st.session_state.performance,
                    st.session_state.current_df,
                    capital,
                    num_positions,
                    st.session_state.min_pos_pct,
                    st.session_state.max_pos_pct
                )
                
                progress_bar.progress(1.0, text="Complete!")
                status_text.text(f"Portfolio: {len(st.session_state.portfolio)} positions")
                
                logging.info(f"  Result: {len(st.session_state.portfolio)} positions curated")
                logging.info("=" * 70)
                logging.info(f"ANALYSIS COMPLETE | Mode: {selection_mode} | Strategies: {len(strategies_to_run)} | Positions: {len(st.session_state.portfolio)}")
                logging.info("=" * 70)
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.toast("Analysis Complete!", icon="✅")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Walk-Forward Curation<br> 
                <strong>Data:</strong> Live Generated
            </p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.portfolio is None or st.session_state.performance is None:
        # Show header only on landing page
        st.markdown(f"""
        <div class="premium-header">
            <h1>PRAGYAM | Portfolio Intelligence</h1>
            <div class="tagline">Walk-Forward Curation with Regime-Aware Strategy Allocation</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box welcome'>
            <h4>Welcome to the Pragyam Curation System</h4>
            <p>
                This platform uses a walk-forward engine to backtest and curate a final portfolio
                based on a dynamic mix of quantitative strategies.
            </p>
            <strong>To begin, please follow these steps:</strong>
            <ol style="margin-left: 20px; margin-top: 10px;">
                <li>Select your desired <strong>Analysis Date</strong> in the sidebar.</li>
                <li>Choose your <strong>Investment Style</strong> (e.g., SIP Investment, Swing Trading).</li>
                <li>Adjust your <strong>Capital</strong> and desired <strong>Number of Positions</strong>.</li>
                <li>Click the <strong>Run Analysis</strong> button to start the curation.</li>
            </ol>
            <p style="margin-top: 1rem; font-weight: 600; color: var(--primary-color);">
                The system will automatically detect the market regime and select the optimal strategy mix for you.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card info'>
                <h4>REGIME-AWARE</h4>
                <h2>Auto-Detects</h2>
                <div class='sub-metric'>Bull, Bear, or Chop Mix</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card success'>
                <h4>DYNAMIC</h4>
                <h2>Strategy Curation</h2>
                <div class='sub-metric'>Weights strategies by performance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card primary'>
                <h4>WALK-FORWARD</h4>
                <h2>Robust Backtesting</h2>
                <div class='sub-metric'>Avoids lookahead bias</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        total_value = st.session_state.portfolio['value'].sum()
        cash_remaining = capital - total_value

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h4>Cash Utilized</h4><h2>{total_value:,.2f}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h4>Cash Remaining</h4><h2>{cash_remaining:,.2f}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h4>Positions</h4><h2>{len(st.session_state.portfolio)}</h2></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["**Portfolio**", "**Performance**", "**Strategy Deep Dive**"])

        with tab1:
            st.header("Curated Portfolio Holdings")
            display_df = st.session_state.portfolio[['symbol', 'price', 'units', 'weightage_pct', 'value']]
            
            styled_df = display_df.style.format({
                'price': '{:,.2f}', 
                'value': '{:,.2f}', 
                'units': '{:,.0f}',
                'weightage_pct': '{:.2f}%'
            }).set_table_attributes(
                'class="stMarkdown table"' 
            ).hide(
                axis="index"
            )
            
            st.markdown(styled_df.to_html(), unsafe_allow_html=True)
            
            portfolio_df = st.session_state.portfolio
            first_cols = ['symbol', 'price', 'units']
            other_cols = [col for col in portfolio_df.columns if col not in first_cols]
            new_order = first_cols + other_cols
            download_df = portfolio_df[new_order]

            csv_bytes = fix_csv_export(download_df)
            st.markdown(
                create_export_link(csv_bytes, f"curated_portfolio_{datetime.now().strftime('%Y%m%d')}.csv"), 
                unsafe_allow_html=True
            )

        with tab2:
            display_performance_metrics(st.session_state.performance)

        with tab3:
            st.header("Strategy Deep Dive")
            
            strategies_in_performance = [k for k in st.session_state.performance.get('strategy', {}).keys() if k != 'System_Curated']
            
            if not strategies_in_performance:
                st.warning("No individual strategy data available for deep dive analysis.")
            else:
                # ═══════════════════════════════════════════════════════════════════════════
                # TIER SHARPE HEATMAP
                # ═══════════════════════════════════════════════════════════════════════════
                subset_perf = st.session_state.performance.get('subset', {})
                
                if subset_perf:
                    st.markdown("##### Sharpe by Position Tier")
                    
                    heatmap_data = {}
                    max_tier_num = 0
                    for strat in strategies_in_performance:
                        if strat in subset_perf and subset_perf[strat]:
                            tier_nums = [int(tier.split('_')[1]) for tier in subset_perf[strat].keys()]
                            if tier_nums:
                                max_tier_num = max(max_tier_num, max(tier_nums))

                    if max_tier_num > 0:
                        for strat in strategies_in_performance:
                            row = [subset_perf.get(strat, {}).get(f'tier_{i+1}', np.nan) for i in range(max_tier_num)]
                            heatmap_data[strat] = row

                        df_heatmap = pd.DataFrame(heatmap_data).transpose()
                        df_heatmap.columns = [f'T{i+1}' for i in range(df_heatmap.shape[1])]
                        
                        df_heatmap['Avg'] = df_heatmap.mean(axis=1)
                        df_heatmap = df_heatmap.sort_values('Avg', ascending=False)
                        avg_sharpe = df_heatmap['Avg']
                        df_heatmap = df_heatmap.drop('Avg', axis=1)

                        fig_tier = px.imshow(
                            df_heatmap, 
                            text_auto=".2f", 
                            aspect="auto",
                            color_continuous_scale='RdYlGn',
                            color_continuous_midpoint=0
                        )
                        fig_tier.update_layout(
                            template='plotly_dark',
                            height=max(350, len(strategies_in_performance) * 30),
                            margin=dict(l=120, r=20, t=20, b=40),
                            coloraxis_colorbar=dict(title="Sharpe")
                        )
                        st.plotly_chart(fig_tier, width="stretch")
                        
                        # Tier insights - compact
                        tier_means = df_heatmap.mean(axis=0)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Best Tier", tier_means.idxmax(), f"{tier_means.max():.2f}")
                        col2.metric("Worst Tier", tier_means.idxmin(), f"{tier_means.min():.2f}")
                        col3.metric("Dispersion", f"{tier_means.std():.2f}")
                
                st.markdown("---")
                
                # ═══════════════════════════════════════════════════════════════════════════
                # RISK-RETURN SCATTER
                # ═══════════════════════════════════════════════════════════════════════════
                st.markdown("##### Risk-Return Profile")
                
                scatter_data = []
                for name in strategies_in_performance:
                    metrics = st.session_state.performance.get('strategy', {}).get(name, {}).get('metrics', {})
                    if metrics:
                        scatter_data.append({
                            'Strategy': name,
                            'Volatility': metrics.get('volatility', 0),
                            'CAGR': metrics.get('annual_return', 0),
                            'Sharpe': metrics.get('sharpe', 0),
                            'Max DD': metrics.get('max_drawdown', 0)
                        })
                
                if scatter_data:
                    if UNIFIED_CHARTS_AVAILABLE:
                        fig_scatter = create_risk_return_scatter(scatter_data)
                    else:
                        df_scatter = pd.DataFrame(scatter_data)
                        df_scatter['Vol_pct'] = df_scatter['Volatility'] * 100
                        df_scatter['CAGR_pct'] = df_scatter['CAGR'] * 100
                        df_scatter['Size'] = np.abs(df_scatter['Max DD']) * 100 + 5
                        
                        fig_scatter = go.Figure()
                        
                        fig_scatter.add_trace(go.Scatter(
                            x=df_scatter['Vol_pct'],
                            y=df_scatter['CAGR_pct'],
                            mode='markers+text',
                            marker=dict(
                                size=df_scatter['Size'],
                                color=df_scatter['Sharpe'],
                                colorscale='RdYlGn',
                                cmin=-1, cmax=2,
                                showscale=True,
                                colorbar=dict(title='Sharpe', tickfont=dict(color=COLORS['muted'])),
                                line=dict(width=1, color=COLORS['border'])
                            ),
                            text=df_scatter['Strategy'].apply(lambda x: x[:10]),
                            textposition='top center',
                            textfont=dict(size=9, color=COLORS['muted']),
                            customdata=df_scatter[['Strategy', 'Max DD']].values,
                            hovertemplate='<b>%{customdata[0]}</b><br>CAGR: %{y:.1f}%<br>Vol: %{x:.1f}%<extra></extra>'
                        ))
                        
                        if len(df_scatter) > 2:
                            max_sharpe_idx = df_scatter['Sharpe'].idxmax()
                            tangent_vol = df_scatter.loc[max_sharpe_idx, 'Vol_pct']
                            tangent_ret = df_scatter.loc[max_sharpe_idx, 'CAGR_pct']
                            
                            fig_scatter.add_trace(go.Scatter(
                                x=[0, tangent_vol * 1.8], y=[0, tangent_ret * 1.8],
                                mode='lines', name='CML',
                                line=dict(color=COLORS['muted'], dash='dash', width=1.5),
                                showlegend=False
                            ))
                            
                            fig_scatter.add_trace(go.Scatter(
                                x=[tangent_vol], y=[tangent_ret],
                                mode='markers', name='Optimal',
                                marker=dict(size=15, color=COLORS['primary'], symbol='star',
                                           line=dict(width=2, color=COLORS['text'])),
                                showlegend=False
                            ))
                        
                        fig_scatter.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor=COLORS['card'],
                            height=420,
                            font=dict(family='Inter', color=COLORS['text']),
                            margin=dict(l=60, r=20, t=20, b=50),
                            xaxis_title="Volatility (%)",
                            yaxis_title="CAGR (%)"
                        )
                        fig_scatter.update_xaxes(showgrid=True, gridcolor=COLORS['border'])
                        fig_scatter.update_yaxes(showgrid=True, gridcolor=COLORS['border'])
                    
                    st.plotly_chart(fig_scatter, width="stretch")
                    st.caption("Bubble size = Max Drawdown magnitude | Color = Sharpe ratio | ⭐ = Optimal")
                
                st.markdown("---")
                
                # ═══════════════════════════════════════════════════════════════════════════
                # FACTOR RADAR
                # ═══════════════════════════════════════════════════════════════════════════
                st.markdown("##### Factor Fingerprint")
                
                factor_data = []
                for name in strategies_in_performance:
                    metrics = st.session_state.performance.get('strategy', {}).get(name, {}).get('metrics', {})
                    if metrics:
                        factor_data.append({
                            'Strategy': name,
                            'Return Factor': min(max(metrics.get('annual_return', 0) / 0.30, -1), 1),
                            'Risk Control': min(max(-metrics.get('max_drawdown', -0.20) / 0.20, 0), 1),
                            'Consistency': metrics.get('win_rate', 0.5),
                            'Efficiency': min(max(metrics.get('sharpe', 0) / 2, -1), 1),
                            'Tail Risk': min(max(metrics.get('tail_ratio', 1), 0), 2) / 2
                        })
                
                if factor_data and len(factor_data) > 0:
                    if UNIFIED_CHARTS_AVAILABLE:
                        fig_radar = create_factor_radar(factor_data, max_strategies=4)
                    else:
                        df_factors = pd.DataFrame(factor_data)
                        top_strats = df_factors.nlargest(min(4, len(df_factors)), 'Efficiency')
                        categories = ['Return Factor', 'Risk Control', 'Consistency', 'Efficiency', 'Tail Risk']
                        
                        fig_radar = go.Figure()
                        palette = [COLORS['primary'], COLORS['success'], COLORS['info'], COLORS['warning']]
                        
                        for idx, (_, row) in enumerate(top_strats.iterrows()):
                            values = [row[cat] for cat in categories]
                            values.append(values[0])
                            color = palette[idx % len(palette)]
                            
                            fig_radar.add_trace(go.Scatterpolar(
                                r=values,
                                theta=categories + [categories[0]],
                                fill='toself',
                                name=row['Strategy'][:15],
                                line_color=color,
                                fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                                opacity=0.8
                            ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True, range=[0, 1],
                                    showticklabels=True,
                                    tickfont=dict(size=9, color=COLORS['muted']),
                                    gridcolor=COLORS['border']
                                ),
                                angularaxis=dict(
                                    tickfont=dict(size=11, color=COLORS['text']),
                                    gridcolor=COLORS['border']
                                ),
                                bgcolor='rgba(0,0,0,0)'
                            ),
                            showlegend=True,
                            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center', font=dict(size=10)),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family='Inter', color=COLORS['text']),
                            height=400,
                            margin=dict(l=60, r=60, t=40, b=60)
                        )
                    
                    st.plotly_chart(fig_radar, width="stretch")
                
                # ═══════════════════════════════════════════════════════════════════════════
                # TIER WEIGHT EVOLUTION
                # ═══════════════════════════════════════════════════════════════════════════
                st.markdown("---")
                st.markdown("##### Tier Allocation History")
                
                display_subset_weight_evolution(
                    st.session_state.performance.get('subset_weights_history', []),
                    strategies_in_performance
                )
                
                # ═══════════════════════════════════════════════════════════════════════════
                # CONVICTION ANALYSIS
                # ═══════════════════════════════════════════════════════════════════════════
                st.markdown("---")
                st.markdown("##### Cross-Strategy Conviction")
                
                strategies_for_heatmap = {name: strategies[name] for name in strategies_in_performance if name in strategies}
                
                if strategies_for_heatmap and st.session_state.current_df is not None:
                    heatmap_fig = create_conviction_heatmap(strategies_for_heatmap, st.session_state.current_df)
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, width="stretch")
                    
                    # Signal agreement analysis - compact
                    signal_counts = {}
                    for name, s in strategies_for_heatmap.items():
                        try:
                            port = s.generate_portfolio(st.session_state.current_df.copy())
                            if not port.empty:
                                for symbol in port.head(10)['symbol']:
                                    signal_counts[symbol] = signal_counts.get(symbol, 0) + 1
                        except:
                            pass
                    
                    if signal_counts:
                        sorted_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)
                        top_consensus = sorted_signals[:5]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**High Consensus Picks** (agreed by multiple strategies):")
                            for symbol, count in top_consensus:
                                agreement_pct = count / len(strategies_for_heatmap) * 100
                                st.markdown(f"• **{symbol}**: {count}/{len(strategies_for_heatmap)} strategies ({agreement_pct:.0f}%)")
                        
                        with col2:
                            consensus_threshold = len(strategies_for_heatmap) / 2
                            high_conviction = [s for s, c in sorted_signals if c >= consensus_threshold]
                            st.metric("High Conviction Symbols", len(high_conviction), 
                                     help=f"Symbols appearing in ≥{consensus_threshold:.0f} strategy portfolios")
                            
                            avg_agreement = np.mean([c for _, c in sorted_signals]) / len(strategies_for_heatmap)
                            st.metric("Average Signal Agreement", f"{avg_agreement:.1%}",
                                     help="Mean fraction of strategies agreeing on each symbol")
                
                # ═══════════════════════════════════════════════════════════════════════════
                # SELECTION SUMMARY (Adaptive Rank-Based)
                # ═══════════════════════════════════════════════════════════════════════════
                st.markdown("---")
                st.markdown("##### Adaptive Selection Ranking")
                
                if strategies_in_performance:
                    summary_data = []
                    for name in strategies_in_performance:
                        metrics = st.session_state.performance.get('strategy', {}).get(name, {}).get('metrics', {})
                        subset = st.session_state.performance.get('subset', {}).get(name, {})
                        
                        tier1_sharpe = subset.get('tier_1', np.nan) if subset else np.nan
                        
                        summary_data.append({
                            'Strategy': name,
                            'Sharpe': metrics.get('sharpe', 0),
                            'Sortino': metrics.get('sortino', 0),
                            'Calmar': metrics.get('calmar', 0),
                            'Max DD': metrics.get('max_drawdown', 0),
                            'Win Rate': metrics.get('win_rate', 0),
                            'T1 Sharpe': tier1_sharpe
                        })
                    
                    df_summary = pd.DataFrame(summary_data)
                    
                    # Adaptive rank-based scoring (no fixed thresholds)
                    # Each metric ranked 0-1, weighted by cross-sectional dispersion
                    rank_metrics = ['Sharpe', 'Sortino', 'Calmar', 'Win Rate']
                    
                    for col in rank_metrics:
                        df_summary[f'{col}_Rank'] = df_summary[col].rank(pct=True)
                    
                    # Max DD: reverse rank (less negative = better)
                    df_summary['DD_Rank'] = df_summary['Max DD'].rank(pct=True, ascending=False)
                    
                    # Compute dispersion-weighted score
                    rank_cols = [c for c in df_summary.columns if c.endswith('_Rank')]
                    dispersions = {col: df_summary[col].std() for col in rank_cols}
                    total_disp = sum(dispersions.values()) or 1
                    
                    # Adaptive weights from dispersion (higher dispersion = more discriminating)
                    weights = {col: disp / total_disp for col, disp in dispersions.items()}
                    
                    df_summary['Score'] = sum(df_summary[col] * w for col, w in weights.items())
                    df_summary = df_summary.sort_values('Score', ascending=False)
                    
                    # Display table
                    df_display = df_summary[['Strategy', 'Sharpe', 'Sortino', 'Calmar', 'Max DD', 'Win Rate', 'T1 Sharpe', 'Score']].copy()
                    df_display['Sharpe'] = df_display['Sharpe'].apply(lambda x: f"{x:.2f}")
                    df_display['Sortino'] = df_display['Sortino'].apply(lambda x: f"{x:.2f}")
                    df_display['Calmar'] = df_display['Calmar'].apply(lambda x: f"{x:.2f}")
                    df_display['Max DD'] = df_display['Max DD'].apply(lambda x: f"{x:.1%}")
                    df_display['Win Rate'] = df_display['Win Rate'].apply(lambda x: f"{x:.0%}")
                    df_display['T1 Sharpe'] = df_display['T1 Sharpe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                    df_display['Score'] = df_display['Score'].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(df_display, width="stretch", hide_index=True)
                    st.caption("Score = Dispersion-weighted rank composite (metrics with higher cross-sectional variance get more weight)")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Dynamic footer with IST time (timezone-aware)
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")

if __name__ == "__main__":
    main()
