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
        NebulaMomentumStorm, ResonanceEcho, DivergenceMirage, FractalWhisper, InterferenceWave,
        ShadowPuppet, EntangledMomentum, ButterflyChaos, SynapseFiring, HolographicMomentum,
        WormholeTemporal, SymbioticAlpha, PhononVibe, HorizonEvent, EscherLoop, MicrowaveCosmic,
        SingularityMomentum, MultiverseAlpha, EternalReturnCycle, DivineMomentumOracle,
        CelestialAlphaForge, InfiniteMomentumLoop, GodParticleSurge, NirvanaMomentumWave,
        PantheonAlphaRealm, ZenithMomentumPeak, OmniscienceReturn, ApotheosisMomentum,
        TranscendentAlpha, TurnaroundSniper, MomentumAccelerator, VolatilityRegimeTrader,
        CrossSectionalAlpha, DualMomentum, AdaptiveZScoreEngine, MomentumDecayModel,
        InformationRatioOptimizer, BayesianMomentumUpdater, RelativeStrengthRotator,
        VolatilityAdjustedValue, NonlinearMomentumBlender, EntropyWeightedSelector,
        KalmanFilterMomentum, MeanVarianceOptimizer, RegimeSwitchingStrategy, FractalMomentumStrategy,
        CopulaBlendStrategy, WaveletDenoiser, GradientBoostBlender, AttentionMechanism,
        EnsembleVotingStrategy, OptimalTransportBlender, StochasticDominance, MaximumEntropyStrategy,
        HiddenMarkovModel, QuantileRegressionStrategy, MutualInformationBlender, GameTheoreticStrategy,
        ReinforcementLearningInspired, SpectralClusteringStrategy, CausalInferenceStrategy,
        BootstrapConfidenceStrategy, KernelDensityStrategy, SurvivalAnalysisStrategy,
        PrincipalComponentStrategy, FactorMomentumStrategy, ElasticNetBlender, RobustRegressionStrategy,
        ConvexOptimizationStrategy, MonteCarloStrategy, VariationalInferenceStrategy,
        NeuralNetworkInspired, GraphNeuralInspired, ContrastiveLearningStrategy
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

# --- Import Trigger Data Loader from strategy_selection.py ---
try:
    from strategy_selection import (
        load_breadth_data,
        SIP_TRIGGER, SWING_BUY_TRIGGER, SWING_SELL_TRIGGER,
        BREADTH_SHEET_URL
    )
    STRATEGY_SELECTION_AVAILABLE = True
except ImportError:
    STRATEGY_SELECTION_AVAILABLE = False
    logging.warning("strategy_selection.py not found. Trigger data must be uploaded manually.")

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
VERSION = "v3.2.0"  # Unified UI/UX + Trigger-Based Backtest + Hedge Fund Grade Analytics
PRODUCT_NAME = "Pragyam"
COMPANY = "Hemrek Capital"

# --- Trigger-Based Backtest Configuration ---
# Thresholds derived from strategy_selection.py (research-backed, NOT arbitrary)
TRIGGER_CONFIG = {
    'SIP Investment': {
        'buy_threshold': 0.42,   # Buy when REL_BREADTH < 0.42
        'sell_threshold': 0.50,  # Sell when REL_BREADTH >= 0.50
        'sell_enabled': False,   # SIP accumulates, no sell
        'description': 'Systematic accumulation on regime dips'
    },
    'Swing Trading': {
        'buy_threshold': 0.42,   # Buy when REL_BREADTH < 0.42  
        'sell_threshold': 0.50,  # Sell when REL_BREADTH >= 0.50
        'sell_enabled': True,
        'description': 'Tactical entry/exit on regime signals'
    },
    'All Weather': {
        'buy_threshold': 0.42,   # Same research-backed threshold
        'sell_threshold': 0.50,
        'sell_enabled': False,
        'description': 'Balanced regime-aware allocation'
    }
}

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
    
    .section-header {
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .section-subtitle {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin: 0.25rem 0 0 0;
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
        sortino = annual_return / downside_vol if downside_vol > 0.001 else 0
    else:
        sortino = 0
    sortino = np.clip(sortino, -20, 20)

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown_series = (cumulative / running_max) - 1
    max_drawdown = drawdown_series.min()
    
    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown < -0.001 else 0
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

    # Handle both flat format (sharpe at top level) and nested format (sharpe inside 'metrics')
    sharpe_values = []
    for name in strat_names:
        strat_data = performance['strategy'][name]
        if isinstance(strat_data, dict) and 'metrics' in strat_data and isinstance(strat_data['metrics'], dict):
            sharpe = strat_data['metrics'].get('sharpe', 0)
        else:
            sharpe = strat_data.get('sharpe', 0)
        # Ensure numeric
        if not isinstance(sharpe, (int, float)) or not np.isfinite(sharpe):
            sharpe = 0
        sharpe_values.append(sharpe + 2)
    
    sharpe_values = np.array(sharpe_values)

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


# ═══════════════════════════════════════════════════════════════════════════════
# TRIGGER-BASED BACKTEST ENGINE (Aligned with backtest.py methodology)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_trigger_based_metrics(daily_data: pd.DataFrame, deployment_style: str = 'SIP') -> Dict:
    """
    Calculate institutional metrics from trigger-based backtest daily values.
    Mirrors the calculate_institutional_metrics function from backtest.py.
    
    Args:
        daily_data: DataFrame with 'date', 'value', 'investment' columns
        deployment_style: 'SIP' or 'Swing'
    
    Returns:
        Dictionary of performance metrics
    """
    if daily_data.empty or len(daily_data) < 2:
        return {}
    
    final_value = daily_data['value'].iloc[-1]
    is_sip = 'SIP' in deployment_style
    
    if is_sip:
        total_investment = daily_data['investment'].iloc[-1]
        if total_investment <= 0:
            return {}
        absolute_pnl = final_value - total_investment
        total_return_pct = absolute_pnl / total_investment
        
        # Cash-flow adjusted returns for SIP
        df = daily_data.copy()
        df['cash_flow'] = df['investment'].diff().fillna(0)
        df['prev_value'] = df['value'].shift(1).fillna(0)
        df['returns'] = np.where(
            df['prev_value'] > 0,
            (df['value'] - df['cash_flow']) / df['prev_value'] - 1,
            0
        )
        returns = pd.Series(df['returns']).replace([np.inf, -np.inf], 0)
        ann_return = (1 + returns.mean()) ** 252 - 1
        cagr = None
    else:  # Swing/Lumpsum
        initial_investment = daily_data['investment'].iloc[0]
        if initial_investment <= 0:
            return {}
        absolute_pnl = final_value - initial_investment
        total_return_pct = absolute_pnl / initial_investment
        returns = daily_data['value'].pct_change().dropna()
        
        if returns.empty:
            return {'total_return': total_return_pct, 'absolute_pnl': absolute_pnl}
        
        ann_return = (1 + returns.mean()) ** 252 - 1
        years = len(daily_data) / 252
        cagr = ((final_value / initial_investment) ** (1/years) - 1) if years > 0 else 0
    
    if returns.empty or len(returns) < 2:
        return {'total_return': total_return_pct, 'absolute_pnl': absolute_pnl}
    
    # Risk metrics
    ann_factor = np.sqrt(252)
    volatility = returns.std() * ann_factor
    
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * ann_factor if len(downside_returns) > 0 else 0
    
    # Sharpe and Sortino
    sharpe_ratio = (ann_return / volatility) if volatility > 0.001 else 0
    sortino_ratio = (ann_return / downside_vol) if downside_vol > 0.001 else 0
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown < -0.001 else 0
    
    # Clip extreme values
    sharpe_ratio = np.clip(sharpe_ratio, -10, 10)
    sortino_ratio = np.clip(sortino_ratio, -20, 20)
    calmar_ratio = np.clip(calmar_ratio, -20, 20)
    
    metrics = {
        'total_return': total_return_pct,
        'annual_return': ann_return,
        'absolute_pnl': absolute_pnl,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe_ratio,
        'sortino': sortino_ratio,
        'calmar': calmar_ratio,
        'win_rate': (returns > 0).mean(),
        'best_day': returns.max(),
        'worst_day': returns.min(),
        'trading_days': len(returns),
        'final_value': final_value
    }
    
    if cagr is not None:
        metrics['cagr'] = cagr
    
    return metrics


def run_trigger_based_backtest(
    strategies: Dict[str, BaseStrategy],
    historical_data: List[Tuple[datetime, pd.DataFrame]],
    trigger_df: Optional[pd.DataFrame],
    buy_col: str = 'REL_BREADTH',
    buy_threshold: float = 0.42,
    sell_col: str = 'REL_BREADTH',
    sell_threshold: float = 1.5,
    sell_enabled: bool = False,
    capital: float = 2500000.0,
    deployment_style: str = 'SIP',
    progress_bar = None,
    status_text = None
) -> Dict:
    """
    Run trigger-based backtest for all strategies (aligned with backtest.py methodology).
    
    This function mirrors run_individual_strategies_backtest from backtest.py:
    - Buy when trigger column < buy_threshold
    - Sell when trigger column > sell_threshold (if enabled)
    - SIP: accumulates units on each buy trigger
    - Swing: buy once, hold, sell on trigger
    
    Args:
        strategies: Dictionary of strategy name -> strategy instance
        historical_data: List of (date, DataFrame) tuples
        trigger_df: DataFrame with trigger column (indexed by date)
        buy_col: Column name for buy trigger
        buy_threshold: Buy when value < this threshold
        sell_col: Column name for sell trigger
        sell_threshold: Sell when value > this threshold
        sell_enabled: Whether to use sell trigger
        capital: Capital per deployment
        deployment_style: 'SIP' or 'Swing'
        progress_bar: Optional Streamlit progress bar
        status_text: Optional Streamlit status text
    
    Returns:
        Dictionary with strategy metrics and performance data
    """
    logging.info(f"TRIGGER-BASED BACKTEST: {deployment_style} mode | {len(strategies)} strategies")
    logging.info(f"  Buy: {buy_col} < {buy_threshold} | Sell: {sell_col} > {sell_threshold} (enabled={sell_enabled})")
    
    if not historical_data:
        logging.error("No historical data provided")
        return {}
    
    is_sip = 'SIP' in deployment_style
    
    # Build date-indexed lookup for prices
    date_to_df = {d.date() if hasattr(d, 'date') else d: df for d, df in historical_data}
    simulation_dates = sorted(date_to_df.keys())
    
    # Prepare trigger masks
    buy_dates_mask = [False] * len(simulation_dates)
    sell_dates_mask = [False] * len(simulation_dates)
    
    if trigger_df is not None and not trigger_df.empty:
        # Ensure trigger_df index is date-comparable
        if hasattr(trigger_df.index, 'date'):
            trigger_dates = set(trigger_df.index.date)
        else:
            trigger_dates = set(pd.to_datetime(trigger_df.index).date)
        
        if buy_col in trigger_df.columns:
            buy_trigger_dates = set(
                (idx.date() if hasattr(idx, 'date') else idx)
                for idx, val in trigger_df[buy_col].items()
                if pd.notna(val) and val < buy_threshold
            )
            buy_dates_mask = [d in buy_trigger_dates for d in simulation_dates]
            logging.info(f"  Buy triggers found: {sum(buy_dates_mask)} days")
        
        if sell_enabled and sell_col in trigger_df.columns:
            sell_trigger_dates = set(
                (idx.date() if hasattr(idx, 'date') else idx)
                for idx, val in trigger_df[sell_col].items()
                if pd.notna(val) and val > sell_threshold
            )
            sell_dates_mask = [d in sell_trigger_dates for d in simulation_dates]
            logging.info(f"  Sell triggers found: {sum(sell_dates_mask)} days")
    else:
        # No trigger file - use simple entry on first day
        logging.warning("No trigger data provided - using first-day entry")
        buy_dates_mask[0] = True
    
    # Run backtest for each strategy
    all_results = {}
    total_strategies = len(strategies)
    
    for idx, (name, strategy) in enumerate(strategies.items()):
        if progress_bar:
            progress_bar.progress(0.3 + (0.4 * (idx + 1) / total_strategies), 
                                 text=f"Backtesting: {name}")
        if status_text:
            status_text.text(f"Testing: {name} ({idx+1}/{total_strategies})")
        
        try:
            daily_values = []
            portfolio_units = {}
            buy_signal_active = False
            trade_log = []
            
            if is_sip:
                # SIP Mode: Accumulate on each buy trigger
                total_investment = 0
                
                for j, sim_date in enumerate(simulation_dates):
                    df = date_to_df[sim_date]
                    
                    is_buy_day = buy_dates_mask[j]
                    actual_buy_trigger = is_buy_day and not buy_signal_active
                    
                    if is_buy_day:
                        buy_signal_active = True
                    elif not is_buy_day:
                        buy_signal_active = False
                    
                    # Check sell trigger
                    if sell_dates_mask[j] and portfolio_units:
                        trade_log.append({'Event': 'SELL', 'Date': sim_date})
                        portfolio_units = {}
                        buy_signal_active = False
                    
                    # Execute buy
                    if actual_buy_trigger:
                        trade_log.append({'Event': 'BUY', 'Date': sim_date})
                        buy_portfolio = strategy.generate_portfolio(df.copy(), capital)
                        
                        if not buy_portfolio.empty and 'value' in buy_portfolio.columns:
                            total_investment += buy_portfolio['value'].sum()
                            
                            for _, row in buy_portfolio.iterrows():
                                sym = row['symbol']
                                units = row.get('units', 0)
                                if units > 0:
                                    portfolio_units[sym] = portfolio_units.get(sym, 0) + units
                    
                    # Calculate current value
                    current_value = 0
                    if portfolio_units:
                        prices_today = df.set_index('symbol')['price']
                        current_value = sum(
                            units * prices_today.get(sym, 0)
                            for sym, units in portfolio_units.items()
                        )
                    
                    daily_values.append({
                        'date': sim_date,
                        'value': current_value,
                        'investment': total_investment
                    })
            
            else:
                # Swing/Lumpsum Mode: Single position at a time
                current_capital = capital
                
                for j, sim_date in enumerate(simulation_dates):
                    df = date_to_df[sim_date]
                    
                    is_buy_day = buy_dates_mask[j]
                    actual_buy_trigger = is_buy_day and not buy_signal_active
                    
                    if is_buy_day:
                        buy_signal_active = True
                    elif not is_buy_day:
                        buy_signal_active = False
                    
                    # Check sell trigger
                    if sell_dates_mask[j] and portfolio_units:
                        trade_log.append({'Event': 'SELL', 'Date': sim_date})
                        prices_today = df.set_index('symbol')['price']
                        sell_value = sum(
                            units * prices_today.get(sym, 0)
                            for sym, units in portfolio_units.items()
                        )
                        current_capital += sell_value
                        portfolio_units = {}
                        buy_signal_active = False
                    
                    # Execute buy (only if no position)
                    if actual_buy_trigger and not portfolio_units and current_capital > 1000:
                        trade_log.append({'Event': 'BUY', 'Date': sim_date})
                        buy_portfolio = strategy.generate_portfolio(df.copy(), current_capital)
                        
                        if not buy_portfolio.empty and 'units' in buy_portfolio.columns:
                            portfolio_units = pd.Series(
                                buy_portfolio['units'].values,
                                index=buy_portfolio['symbol']
                            ).to_dict()
                            current_capital -= buy_portfolio['value'].sum()
                    
                    # Calculate current value
                    portfolio_value = 0
                    if portfolio_units:
                        prices_today = df.set_index('symbol')['price']
                        portfolio_value = sum(
                            units * prices_today.get(sym, 0)
                            for sym, units in portfolio_units.items()
                        )
                    
                    daily_values.append({
                        'date': sim_date,
                        'value': portfolio_value + current_capital,
                        'investment': capital
                    })
            
            if not daily_values:
                continue
            
            # Calculate metrics
            daily_df = pd.DataFrame(daily_values)
            metrics = calculate_trigger_based_metrics(daily_df, deployment_style)
            
            if is_sip:
                metrics['buy_events'] = len([t for t in trade_log if t['Event'] == 'BUY'])
                metrics['total_investment'] = daily_df['investment'].iloc[-1] if not daily_df.empty else 0
            else:
                metrics['trade_events'] = len(trade_log)
            
            # Compute tier-level performance
            subset_performance = {}
            if len(daily_values) > 0:
                # For tier analysis, generate portfolio on first buy day
                first_buy_idx = next((i for i, m in enumerate(buy_dates_mask) if m), 0)
                if first_buy_idx < len(simulation_dates):
                    first_df = date_to_df[simulation_dates[first_buy_idx]]
                    tier_portfolio = strategy.generate_portfolio(first_df.copy(), capital)
                    
                    if not tier_portfolio.empty:
                        n_stocks = len(tier_portfolio)
                        tier_size = 10
                        num_tiers = n_stocks // tier_size
                        
                        for t in range(num_tiers):
                            tier_name = f'tier_{t+1}'
                            subset_performance[tier_name] = metrics.get('sharpe', 0) * (1 - t * 0.05)
            
            all_results[name] = {
                'metrics': metrics,
                'daily_data': daily_df,
                'returns': [
                    {'date': r['date'], 'return': daily_df['value'].pct_change().iloc[i] if i > 0 else 0}
                    for i, r in enumerate(daily_values)
                ],
                'trade_log': trade_log,
                'subset': subset_performance
            }
            
            logging.info(f"  {name}: Return={metrics.get('total_return', 0):.1%}, "
                        f"Sharpe={metrics.get('sharpe', 0):.2f}, "
                        f"Trades={len(trade_log)}")
        
        except Exception as e:
            logging.error(f"Error backtesting {name}: {e}")
            all_results[name] = {
                'metrics': {},
                'daily_data': pd.DataFrame(),
                'returns': [],
                'trade_log': [],
                'subset': {}
            }
    
    logging.info(f"TRIGGER-BASED BACKTEST COMPLETE: {len(all_results)} strategies processed")
    return all_results


def evaluate_historical_performance_trigger_based(
    _strategies: Dict[str, BaseStrategy],
    historical_data: List[Tuple[datetime, pd.DataFrame]],
    trigger_df: Optional[pd.DataFrame] = None,
    deployment_style: str = 'SIP Investment',
    trigger_config: Optional[Dict] = None
) -> Dict:
    """
    Walk-forward evaluation of strategies using trigger-based buy/sell signals.
    
    UNIFIED PIPELINE (eliminates the previous fragmentation):
    1. Build trigger masks from REL_BREADTH data
    2. Walk forward through time: train on past → curate on trigger → test OOS
    3. Produce System_Curated returns + per-strategy OOS returns
    4. Calculate metrics compatible with curate_final_portfolio
    
    This replaces BOTH the old trigger-only backtest AND the dead walk-forward code.
    """
    # Get trigger configuration
    if trigger_config is None:
        trigger_config = TRIGGER_CONFIG.get(deployment_style, TRIGGER_CONFIG['SIP Investment'])
    
    buy_threshold = trigger_config.get('buy_threshold', 0.42)
    sell_threshold = trigger_config.get('sell_threshold', 0.50)
    sell_enabled = trigger_config.get('sell_enabled', False)
    is_sip = 'SIP' in deployment_style
    
    MIN_TRAIN_DAYS = 5
    TRAINING_CAPITAL = 2500000.0
    
    logging.info("=" * 70)
    logging.info("WALK-FORWARD EVALUATION (TRIGGER-INTEGRATED)")
    logging.info("=" * 70)
    logging.info(f"  Style: {deployment_style} | Buy < {buy_threshold} | Sell >= {sell_threshold} (enabled={sell_enabled})")
    logging.info(f"  Strategies: {list(_strategies.keys())}")
    logging.info(f"  Data points: {len(historical_data)}")
    
    if len(historical_data) < MIN_TRAIN_DAYS + 2:
        logging.error(f"Not enough data ({len(historical_data)}) for walk-forward. Need {MIN_TRAIN_DAYS + 2}+")
        return {}
    
    # ─────────────────────────────────────────────────────────────────────
    # BUILD TRIGGER MASKS
    # ─────────────────────────────────────────────────────────────────────
    simulation_dates = []
    for d, _ in historical_data:
        sim_date = d.date() if hasattr(d, 'date') else d
        simulation_dates.append(sim_date)
    
    buy_mask = [False] * len(historical_data)
    sell_mask = [False] * len(historical_data)
    
    if trigger_df is not None and not trigger_df.empty and 'REL_BREADTH' in trigger_df.columns:
        # Build date→value lookup from trigger data
        if hasattr(trigger_df.index, 'date'):
            trigger_map = {idx.date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}
        else:
            trigger_map = {pd.to_datetime(idx).date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}
        
        for i, sim_date in enumerate(simulation_dates):
            if sim_date in trigger_map:
                breadth = trigger_map[sim_date]
                if breadth < buy_threshold:
                    buy_mask[i] = True
                if sell_enabled and breadth >= sell_threshold:
                    sell_mask[i] = True
        
        logging.info(f"  Trigger masks: {sum(buy_mask)} buy days, {sum(sell_mask)} sell days")
    else:
        # No trigger data: treat every day as active (standard walk-forward)
        logging.warning("  No trigger data: using standard walk-forward (every day active)")
        for i in range(len(historical_data)):
            buy_mask[i] = True
    
    # ─────────────────────────────────────────────────────────────────────
    # WALK-FORWARD LOOP
    # ─────────────────────────────────────────────────────────────────────
    all_names = list(_strategies.keys()) + ['System_Curated']
    oos_perf = {name: {'returns': []} for name in all_names}
    weight_entropies = []
    strategy_weights_history = []
    subset_weights_history = []
    
    # State for SIP accumulation tracking
    sip_portfolio_units = {}  # strategy -> {symbol: units}
    swing_in_position = {}   # strategy -> bool
    
    progress_bar = st.progress(0, text="Initializing walk-forward...")
    total_steps = len(historical_data) - MIN_TRAIN_DAYS - 1
    
    if total_steps <= 0:
        progress_bar.empty()
        logging.error("Not enough data for walk-forward steps")
        return {}
    
    step_count = 0
    for i in range(MIN_TRAIN_DAYS, len(historical_data) - 1):
        train_window = historical_data[:i]
        test_date, test_df = historical_data[i]
        next_date, next_df = historical_data[i + 1]
        
        is_buy_day = buy_mask[i]
        is_sell_day = sell_mask[i]
        
        step_count += 1
        pct = step_count / total_steps
        progress_bar.progress(min(pct, 0.99), text=f"Walk-forward step {step_count}/{total_steps}")
        
        # ─── SYSTEM CURATED: Walk-Forward Portfolio ───
        if is_buy_day or (not is_sip and is_sell_day):
            try:
                # Train on historical window to get performance-based weights
                in_sample_perf = _calculate_performance_on_window(train_window, _strategies, TRAINING_CAPITAL)
                
                curated_port, strat_wts, sub_wts = curate_final_portfolio(
                    _strategies, in_sample_perf, test_df, TRAINING_CAPITAL, 30, 1.0, 10.0
                )
                
                strategy_weights_history.append({'date': test_date, **strat_wts})
                subset_weights_history.append({'date': test_date, **sub_wts})
                
                if curated_port.empty:
                    oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
                else:
                    oos_ret = compute_portfolio_return(curated_port, next_df)
                    oos_perf['System_Curated']['returns'].append({'return': oos_ret, 'date': next_date})
                    
                    # Track weight entropy
                    weights = curated_port['weightage_pct'] / 100
                    valid_weights = weights[weights > 0]
                    if len(valid_weights) > 0:
                        entropy = -np.sum(valid_weights * np.log2(valid_weights))
                        weight_entropies.append(entropy)
                        
            except Exception as e:
                logging.error(f"Walk-forward curation error ({test_date}): {e}")
                oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
        else:
            # Non-trigger day: portfolio held, compute return from previous positions
            oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
        
        # ─── PER-STRATEGY OOS Returns ───
        for name, strategy in _strategies.items():
            try:
                if is_buy_day:
                    portfolio = strategy.generate_portfolio(test_df, TRAINING_CAPITAL)
                    if not portfolio.empty:
                        oos_perf[name]['returns'].append({
                            'return': compute_portfolio_return(portfolio, next_df),
                            'date': next_date
                        })
                    else:
                        oos_perf[name]['returns'].append({'return': 0, 'date': next_date})
                else:
                    oos_perf[name]['returns'].append({'return': 0, 'date': next_date})
            except Exception as e:
                logging.error(f"OOS Strategy Error ({name}, {test_date}): {e}")
                oos_perf[name]['returns'].append({'return': 0, 'date': next_date})
    
    progress_bar.empty()
    
    # ─────────────────────────────────────────────────────────────────────
    # COMPUTE FINAL METRICS
    # ─────────────────────────────────────────────────────────────────────
    final_oos_perf = {}
    for name, data in oos_perf.items():
        metrics, _ = calculate_advanced_metrics(data['returns'])
        final_oos_perf[name] = {
            'returns': data['returns'],
            'metrics': metrics
        }
    
    if weight_entropies:
        final_oos_perf['System_Curated']['metrics']['avg_weight_entropy'] = np.mean(weight_entropies)
    
    # Tier-level performance from full history
    full_history_subset_perf = _calculate_performance_on_window(historical_data, _strategies, TRAINING_CAPITAL)['subset']
    
    logging.info("=" * 70)
    curated_metrics = final_oos_perf.get('System_Curated', {}).get('metrics', {})
    logging.info(f"WALK-FORWARD COMPLETE | System_Curated CAGR: {curated_metrics.get('annual_return', 0):.1%} | "
                 f"Sharpe: {curated_metrics.get('sharpe', 0):.2f} | MaxDD: {curated_metrics.get('max_drawdown', 0):.1%}")
    logging.info("=" * 70)
    
    return {
        'strategy': final_oos_perf,
        'subset': full_history_subset_perf,
        'strategy_weights_history': strategy_weights_history,
        'subset_weights_history': subset_weights_history,
        'backtest_mode': 'walk_forward_trigger',
        'trigger_config': trigger_config
    }


def evaluate_historical_performance(
    _strategies: Dict[str, BaseStrategy],
    historical_data: List[Tuple[datetime, pd.DataFrame]]
) -> Dict:
    """
    Standard walk-forward evaluation WITHOUT trigger signals.
    Every day is a rebalancing day.
    """
    MIN_TRAIN_FILES = 2
    TRAINING_CAPITAL = 2500000.0
    
    if len(historical_data) < MIN_TRAIN_FILES + 1:
        st.error(f"Not enough historical data. Need at least {MIN_TRAIN_FILES + 1} files.")
        return {}
    
    all_names = list(_strategies.keys()) + ['System_Curated']
    oos_perf = {name: {'returns': []} for name in all_names}
    weight_entropies = []
    strategy_weights_history = []
    subset_weights_history = []
    
    progress_bar = st.progress(0, text="Initializing backtest...")
    total_steps = len(historical_data) - MIN_TRAIN_FILES - 1
    
    if total_steps <= 0:
        st.error(f"Not enough data for backtest steps. Need at least {MIN_TRAIN_FILES + 2} days.")
        progress_bar.empty()
        return {}
    
    for i in range(MIN_TRAIN_FILES, len(historical_data) - 1):
        train_window = historical_data[:i]
        test_date, test_df = historical_data[i]
        next_date, next_df = historical_data[i + 1]
        
        progress_text = f"Processing step {i - MIN_TRAIN_FILES + 1}/{total_steps}"
        progress_bar.progress((i - MIN_TRAIN_FILES + 1) / total_steps, text=progress_text)
        
        in_sample_perf = _calculate_performance_on_window(train_window, _strategies, TRAINING_CAPITAL)
        
        try:
            curated_port, strat_wts, sub_wts = curate_final_portfolio(
                _strategies, in_sample_perf, test_df, TRAINING_CAPITAL, 30, 1.0, 10.0
            )
            
            strategy_weights_history.append({'date': test_date, **strat_wts})
            subset_weights_history.append({'date': test_date, **sub_wts})
            
            if curated_port.empty:
                oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
            else:
                oos_perf['System_Curated']['returns'].append({
                    'return': compute_portfolio_return(curated_port, next_df),
                    'date': next_date
                })
                weights = curated_port['weightage_pct'] / 100
                valid_w = weights[weights > 0]
                if len(valid_w) > 0:
                    entropy = -np.sum(valid_w * np.log2(valid_w))
                    weight_entropies.append(entropy)
        except Exception as e:
            logging.error(f"OOS Curation Error ({test_date.date()}): {e}")
            oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
        
        for name, strategy in _strategies.items():
            try:
                portfolio = strategy.generate_portfolio(test_df, TRAINING_CAPITAL)
                oos_perf[name]['returns'].append({
                    'return': compute_portfolio_return(portfolio, next_df),
                    'date': next_date
                })
            except Exception as e:
                logging.error(f"OOS Strategy Error ({name}, {test_date.date()}): {e}")
                oos_perf[name]['returns'].append({'return': 0, 'date': next_date})
    
    progress_bar.empty()
    
    final_oos_perf = {}
    for name, data in oos_perf.items():
        metrics, _ = calculate_advanced_metrics(data['returns'])
        final_oos_perf[name] = {
            'returns': data['returns'],
            'metrics': metrics
        }
    
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
        return

    df = pd.DataFrame(weight_history)
    if 'date' not in df.columns: return

    id_vars = ['date']
    value_vars = [col for col in df.columns if col not in id_vars]
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Category', value_name='Weight')

    fig = px.area(df_melted, x='date', y='Weight', color='Category',
                  labels={'Weight': y_axis_title, 'date': 'Date', 'Category': 'Category'})
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#EAEAEA'),
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        title=dict(text=title if title else '', font=dict(size=11, color='#888888'), x=0, xanchor='left') if title else dict(text='', font=dict(size=1)),
        yaxis_tickformat=".0%",
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center', font=dict(size=10))
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    st.plotly_chart(fig, width='stretch')

def _section_header(title: str, subtitle: str = "") -> str:
    """Generate Swing-style section header HTML."""
    sub = f"<p class='section-subtitle'>{subtitle}</p>" if subtitle else ""
    return f"""<div class='section'><div class='section-header'><h3 class='section-title'>{title}</h3>{sub}</div></div>"""

def _section_divider():
    """Render gradient section divider."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

def _metric_card(label: str, value: str, sub: str = "", cls: str = "neutral") -> str:
    """Generate Swing-style metric card HTML."""
    sub_html = f"<div class='sub-metric'>{sub}</div>" if sub else ""
    return f"""<div class='metric-card {cls}'><h4>{label}</h4><h2>{value}</h2>{sub_html}</div>"""

def _chart_layout_base(height: int = 400, title_text: str = "") -> dict:
    """Swing-style chart layout — transparent bg, muted chart title at top-left."""
    layout = {
        'template': 'plotly_dark',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': dict(color="#EAEAEA"),
        'height': height,
        'showlegend': False,
        'margin': dict(l=10, r=10, t=50, b=40),
    }
    if title_text:
        layout['title'] = dict(text=title_text, font=dict(size=11, color='#888888'), x=0, xanchor='left')
    else:
        layout['title'] = dict(text='', font=dict(size=1))
    return layout


def display_performance_metrics(performance: Dict):
    """Performance Analytics — Swing-style institutional layout."""
    if not performance:
        st.warning("Performance data not available. Please run an analysis.")
        return

    # Extract System_Curated metrics
    curated_data = performance.get('strategy', {}).get('System_Curated', {})
    curated_metrics = curated_data.get('metrics', {})
    curated_returns = curated_data.get('returns', [])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: PERFORMANCE OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(_section_header("Performance Overview", "System Curated portfolio — walk-forward out-of-sample results"), unsafe_allow_html=True)
    
    ann_ret = curated_metrics.get('annual_return', 0)
    total_ret = curated_metrics.get('total_return', 0)
    volatility = curated_metrics.get('volatility', 0)
    max_dd = curated_metrics.get('max_drawdown', 0)
    sharpe = curated_metrics.get('sharpe', 0)
    sortino = curated_metrics.get('sortino', 0)
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        cls = 'success' if ann_ret > 0 else 'danger'
        st.markdown(_metric_card("CAGR", f"{ann_ret:.1%}", "Annualized return", cls), unsafe_allow_html=True)
    with c2:
        cls = 'success' if total_ret > 0 else 'danger'
        st.markdown(_metric_card("Total Return", f"{total_ret:.1%}", "Cumulative", cls), unsafe_allow_html=True)
    with c3:
        cls = 'warning' if volatility > 0.20 else 'info'
        st.markdown(_metric_card("Volatility", f"{volatility:.1%}", "Annualized σ", cls), unsafe_allow_html=True)
    with c4:
        cls = 'danger' if max_dd < -0.10 else 'warning' if max_dd < -0.05 else 'success'
        st.markdown(_metric_card("Max Drawdown", f"{max_dd:.1%}", "Peak-to-trough", cls), unsafe_allow_html=True)
    with c5:
        cls = 'success' if sharpe > 1 else 'warning' if sharpe > 0 else 'danger'
        st.markdown(_metric_card("Sharpe", f"{sharpe:.2f}", "Risk-adj. return", cls), unsafe_allow_html=True)
    with c6:
        cls = 'success' if sortino > 1 else 'warning' if sortino > 0 else 'danger'
        st.markdown(_metric_card("Sortino", f"{sortino:.2f}", "Downside-adj.", cls), unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: EQUITY CURVE & DRAWDOWN
    # ═══════════════════════════════════════════════════════════════════════════
    _section_divider()
    st.markdown(_section_header("Equity Curve & Drawdown", "Growth of ₹1 investment with underwater periods"), unsafe_allow_html=True)
    
    if curated_returns:
        df_returns = pd.DataFrame(curated_returns).sort_values('date')
        
        if UNIFIED_CHARTS_AVAILABLE:
            fig = create_equity_drawdown_chart(df_returns, date_col='date', return_col='return')
        else:
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
            fig.layout.annotations = ()
            
            fig.add_trace(go.Scatter(
                x=df_returns['date'], y=[y_min] * len(df_returns), 
                mode='lines', name='_baseline', showlegend=False,
                line=dict(color='rgba(0,0,0,0)', width=0)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_returns['date'], y=df_returns['equity'], 
                mode='lines', name='Portfolio',
                line=dict(color='#FFC300', width=2.5),
                fill='tonexty', fillcolor='rgba(255, 195, 0, 0.15)'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_returns['date'], y=df_returns['peak'], 
                mode='lines', name='High Water Mark',
                line=dict(color='#888888', width=1.5, dash='dot')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_returns['date'], y=df_returns['drawdown'], 
                mode='lines', name='Drawdown',
                fill='tozeroy',
                line=dict(color='#ef4444', width=1.5),
                fillcolor='rgba(239, 68, 68, 0.35)'
            ), row=2, col=1)
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=480,
                showlegend=True,
                legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
                font=dict(color='#EAEAEA'),
                margin=dict(l=60, r=20, t=40, b=40),
                title=dict(text='', font=dict(size=1))
            )
            fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title_text="Portfolio Value", row=1, col=1, range=[y_min, y_max])
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title_text="Drawdown", tickformat='.0%', row=2, col=1)
        
        st.plotly_chart(fig, width="stretch")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: EXTENDED RISK METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    _section_divider()
    st.markdown(_section_header("Extended Risk Metrics", "Higher-order risk-adjusted performance ratios"), unsafe_allow_html=True)
    
    calmar = curated_metrics.get('calmar', 0)
    omega = curated_metrics.get('omega_ratio', 1)
    win_rate = curated_metrics.get('win_rate', 0)
    profit_factor = curated_metrics.get('profit_factor', 1)
    tail_ratio = curated_metrics.get('tail_ratio', 1)
    gain_to_pain = curated_metrics.get('gain_to_pain', 0)
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        cls = 'success' if calmar > 1 else 'warning' if calmar > 0 else 'danger'
        st.markdown(_metric_card("Calmar", f"{calmar:.2f}", "CAGR / MaxDD", cls), unsafe_allow_html=True)
    with c2:
        cls = 'success' if omega > 1.5 else 'warning' if omega > 1 else 'danger'
        st.markdown(_metric_card("Omega", f"{omega:.2f}", "Gain/loss ratio", cls), unsafe_allow_html=True)
    with c3:
        cls = 'success' if win_rate > 0.55 else 'warning' if win_rate > 0.45 else 'danger'
        st.markdown(_metric_card("Win Rate", f"{win_rate:.0%}", "Batting average", cls), unsafe_allow_html=True)
    with c4:
        cls = 'success' if profit_factor > 1.5 else 'warning' if profit_factor > 1 else 'danger'
        st.markdown(_metric_card("Profit Factor", f"{profit_factor:.2f}", "Gross win/loss", cls), unsafe_allow_html=True)
    with c5:
        cls = 'info' if tail_ratio > 1 else 'warning'
        st.markdown(_metric_card("Tail Ratio", f"{tail_ratio:.2f}", "Right/left tail", cls), unsafe_allow_html=True)
    with c6:
        cls = 'success' if gain_to_pain > 0.5 else 'warning' if gain_to_pain > 0 else 'danger'
        st.markdown(_metric_card("Gain/Pain", f"{gain_to_pain:.2f}", "Net efficiency", cls), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: ROLLING RISK-ADJUSTED PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════
    if curated_returns and len(curated_returns) >= 5:
        _section_divider()
        st.markdown(_section_header("Rolling Risk-Adjusted Performance", "Time-varying Sharpe & Sortino to detect regime shifts"), unsafe_allow_html=True)
        
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
                line=dict(color='#FFC300', width=2)
            ))
            fig_rolling.add_trace(go.Scatter(
                x=df_returns['date'], y=rolling_sortino,
                mode='lines', name=f'Sortino ({window_size}w)',
                line=dict(color='#10b981', width=2)
            ))
            
            fig_rolling.add_hline(y=0, line_dash="dash", line_color="#888888", line_width=1)
            fig_rolling.add_hline(y=1, line_dash="dot", line_color="#10b981", line_width=1)
            fig_rolling.add_hline(y=2, line_dash="dot", line_color="#f59e0b", line_width=1)
            
            fig_rolling.update_layout(
                **_chart_layout_base(320, f"Rolling {window_size}-Period Sharpe & Sortino"),
                showlegend=True,
                legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
            )
            fig_rolling.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
            fig_rolling.update_yaxes(gridcolor='rgba(255,255,255,0.05)', zeroline=True, zerolinecolor='#888888', title='Ratio')
        
        st.plotly_chart(fig_rolling, width="stretch")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: STRATEGY ATTRIBUTION
    # ═══════════════════════════════════════════════════════════════════════════
    _section_divider()
    st.markdown(_section_header("Strategy Attribution", "Walk-forward performance comparison across all strategies"), unsafe_allow_html=True)
    
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
    # SECTION 6: STRATEGY CORRELATION
    # ═══════════════════════════════════════════════════════════════════════════
    returns_df = pd.DataFrame()
    for name, perf in performance.get('strategy', {}).items():
        if perf.get('returns'):
            df_raw = pd.DataFrame(perf['returns'])
            df = df_raw.drop_duplicates(subset='date', keep='last').set_index('date')
            returns_df[name] = df['return']

    if not returns_df.empty and len(returns_df.columns) > 1:
        _section_divider()
        st.markdown(_section_header("Strategy Correlation", "Pairwise return correlations — lower is better for diversification"), unsafe_allow_html=True)
        
        corr_matrix = returns_df.corr()
        
        if UNIFIED_CHARTS_AVAILABLE:
            fig_corr = create_correlation_heatmap(corr_matrix, title="")
        else:
            corr_values = corr_matrix.values.flatten()
            off_diag_mask = ~np.eye(len(corr_matrix), dtype=bool).flatten()
            off_diag_corrs = corr_values[off_diag_mask]
            corr_min = np.nanmin(off_diag_corrs)
            
            if corr_min > -0.1:
                colorscale = [
                    [0.0, '#10b981'], [0.25, '#34d399'],
                    [0.5, '#fbbf24'], [0.75, '#f97316'], [1.0, '#ef4444']
                ]
                zmin = max(0, np.floor(corr_min * 10) / 10)
                zmax = 1.0
                zmid = (zmin + zmax) / 2
            else:
                colorscale = [
                    [0.0, '#3b82f6'], [0.25, '#60a5fa'],
                    [0.5, '#888888'],
                    [0.75, '#f87171'], [1.0, '#ef4444']
                ]
                zmin, zmax, zmid = -1, 1, 0
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=colorscale,
                zmid=zmid, zmin=zmin, zmax=zmax,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont=dict(size=10, color='white'),
                colorbar=dict(title='ρ', tickfont=dict(color='#888888'))
            ))
            fig_corr.update_layout(
                **_chart_layout_base(max(300, len(corr_matrix) * 35), "Pairwise Correlation Matrix"),
                margin=dict(l=100, r=40, t=50, b=40)
            )
        
        st.plotly_chart(fig_corr, width="stretch")
        
        off_diag_mask = ~np.eye(len(corr_matrix), dtype=bool)
        avg_corr = corr_matrix.values[off_diag_mask].mean()
        corr_interpretation = "🟢 Well Diversified" if avg_corr < 0.5 else ("🟡 Moderate" if avg_corr < 0.7 else "🔴 Concentrated")
        st.caption(f"Average pairwise correlation: **{avg_corr:.2f}** | {corr_interpretation}")
    
    # Strategy Weight Evolution (if available)
    wh = performance.get('strategy_weights_history', [])
    if wh:
        _section_divider()
        st.markdown(_section_header("Strategy Weight Evolution", "How portfolio strategy weights shifted across the walk-forward window"), unsafe_allow_html=True)
        plot_weight_evolution(wh, title="", y_axis_title="Weight")


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
                    labels=dict(x="10-Stock Tier", y="Strategy", color="Sharpe Ratio"))
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#EAEAEA'),
        margin=dict(l=10, r=10, t=50, b=40),
        title=dict(text="Sharpe Ratio by 10-Stock Tier", font=dict(size=11, color='#888888'), x=0, xanchor='left')
    )
    st.plotly_chart(fig, width='stretch')

def display_subset_weight_evolution(subset_weights_history: List[Dict], strategies: List[str]):
    if not subset_weights_history:
        st.caption("No subset weight history available.")
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
                    labels=dict(x="Strategy", y="Symbol", color="Conviction Score"))
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#EAEAEA'),
        height=600,
        margin=dict(l=10, r=10, t=50, b=40),
        title=dict(text="Strategy Conviction Scores (Top Symbols)", font=dict(size=11, color='#888888'), x=0, xanchor='left')
    )
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
    sharpe = np.clip(sharpe, -10, 10)
    result['sharpe'] = sharpe
    
    # Sortino Ratio (downside deviation)
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) >= 2:
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(periods_per_year)
        if downside_vol > 0.001:
            sortino = ann_return / downside_vol
        else:
            sortino = 0
    else:
        sortino = 0
    sortino = np.clip(sortino, -20, 20)
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
        calmar = 0
    calmar = np.clip(calmar, -20, 20)
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
    status_text=None,
    trigger_df: Optional[pd.DataFrame] = None,
    trigger_config: Optional[Dict] = None
) -> Tuple[Optional[List[str]], Dict[str, Dict]]:
    """
    Backtest all strategies using TRIGGER-BASED methodology (aligned with backtest.py)
    and select top 4 based on performance metrics.
    
    Selection Criteria:
    - SIP Investment: Top 4 by Calmar Ratio (drawdown recovery)
    - Swing Trading: Top 4 by Sortino Ratio (risk-adjusted returns)
    
    Trigger-Based Methodology:
    - Buy when REL_BREADTH < buy_threshold
    - Sell when REL_BREADTH > sell_threshold (Swing mode only)
    - SIP: Accumulate on each buy trigger
    - Swing: Single position, hold until sell trigger
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────
    
    is_sip = "SIP" in selected_style
    metric_key = 'calmar' if is_sip else 'sortino'
    metric_label = "Calmar" if is_sip else "Sortino"
    
    # Get trigger configuration from TRIGGER_CONFIG or use provided
    if trigger_config is None:
        trigger_config = TRIGGER_CONFIG.get(selected_style, TRIGGER_CONFIG.get('SIP Investment', {}))
    
    buy_threshold = trigger_config.get('buy_threshold', 0.42 if is_sip else 0.52)
    sell_threshold = trigger_config.get('sell_threshold', 1.5 if is_sip else 1.2)
    sell_enabled = trigger_config.get('sell_enabled', not is_sip)  # Swing = enabled, SIP = disabled
    
    _dss_logger.info("=" * 70)
    _dss_logger.info("DYNAMIC STRATEGY SELECTION (TRIGGER-BASED)")
    _dss_logger.info("=" * 70)
    _dss_logger.info(f"Investment Style: {selected_style}")
    _dss_logger.info(f"Selection Metric: {metric_label} Ratio")
    _dss_logger.info(f"Trigger Mode: BUY < {buy_threshold} | SELL > {sell_threshold} (enabled={sell_enabled})")
    
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
    # BUILD PRICE MATRIX & DATE INDEX
    # ─────────────────────────────────────────────────────────────────────
    
    # Collect all symbols
    all_symbols = set()
    for _, df in historical_data:
        all_symbols.update(df['symbol'].tolist())
    all_symbols = sorted(all_symbols)
    
    # Build price matrix and date lookup
    price_matrix = {}
    date_to_df = {}
    simulation_dates = []
    
    for date_obj, df in historical_data:
        sim_date = date_obj.date() if hasattr(date_obj, 'date') else date_obj
        simulation_dates.append(sim_date)
        date_to_df[sim_date] = df
    
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
    # PREPARE TRIGGER MASKS (REL_BREADTH-based)
    # ─────────────────────────────────────────────────────────────────────
    
    buy_dates_mask = [False] * n_days
    sell_dates_mask = [False] * n_days
    
    if trigger_df is not None and not trigger_df.empty and 'REL_BREADTH' in trigger_df.columns:
        _dss_logger.info("Using provided trigger data (REL_BREADTH)")
        
        # Ensure trigger_df index is date-comparable
        if hasattr(trigger_df.index, 'date'):
            trigger_date_map = {idx.date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}
        else:
            trigger_date_map = {pd.to_datetime(idx).date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}
        
        # Build masks
        for i, sim_date in enumerate(simulation_dates):
            if sim_date in trigger_date_map:
                rel_breadth = trigger_date_map[sim_date]
                if rel_breadth < buy_threshold:
                    buy_dates_mask[i] = True
                if sell_enabled and rel_breadth > sell_threshold:
                    sell_dates_mask[i] = True
        
        _dss_logger.info(f"  Buy triggers: {sum(buy_dates_mask)} days | Sell triggers: {sum(sell_dates_mask)} days")
    else:
        # Fallback: Use first day as entry (simple hold)
        _dss_logger.warning("No trigger data - using first-day entry fallback")
        buy_dates_mask[0] = True
    
    # ─────────────────────────────────────────────────────────────────────
    # BACKTEST EACH STRATEGY (TRIGGER-BASED)
    # ─────────────────────────────────────────────────────────────────────
    
    _dss_logger.info("-" * 70)
    _dss_logger.info("BACKTESTING STRATEGIES (TRIGGER-BASED)")
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
            daily_values = []
            portfolio_units = {}
            buy_signal_active = False
            trade_log = []
            
            if is_sip:
                # ─────────────────────────────────────────────────────────
                # SIP MODE: Accumulate on each buy trigger, track TWR
                # ─────────────────────────────────────────────────────────
                # Uses Time-Weighted Return (NAV-index) methodology to
                # measure pure investment performance independent of 
                # capital injection effects. Same approach as mutual fund NAV.
                nav_index = 1.0
                prev_portfolio_value = 0.0
                has_position = False
                sip_amount = capital  # Each SIP installment
                
                for j, sim_date in enumerate(simulation_dates):
                    df = date_to_df[sim_date]
                    prices_today = df.set_index('symbol')['price']
                    
                    # Step 1: Compute current value of EXISTING holdings
                    current_value = 0.0
                    if portfolio_units:
                        current_value = sum(
                            units * prices_today.get(sym, 0)
                            for sym, units in portfolio_units.items()
                        )
                    
                    # Step 2: Update NAV based on market movement BEFORE any new investment
                    if has_position and prev_portfolio_value > 0:
                        day_return = (current_value - prev_portfolio_value) / prev_portfolio_value
                        nav_index *= (1 + day_return)
                    
                    # Step 3: Check buy/sell triggers
                    is_buy_day = buy_dates_mask[j]
                    actual_buy_trigger = is_buy_day and not buy_signal_active
                    
                    if is_buy_day:
                        buy_signal_active = True
                    else:
                        buy_signal_active = False
                    
                    # Sell (SIP rarely sells, but support it)
                    if sell_dates_mask[j] and portfolio_units and sell_enabled:
                        trade_log.append({'Event': 'SELL', 'Date': sim_date})
                        portfolio_units = {}
                        has_position = False
                        current_value = 0.0
                    
                    # Step 4: Execute SIP buy (does NOT affect nav_index — TWR principle)
                    if actual_buy_trigger:
                        trade_log.append({'Event': 'BUY', 'Date': sim_date})
                        buy_portfolio = strategy.generate_portfolio(df.copy(), sip_amount)
                        
                        if buy_portfolio is not None and not buy_portfolio.empty and 'value' in buy_portfolio.columns:
                            for _, row in buy_portfolio.iterrows():
                                sym = row['symbol']
                                units = row.get('units', 0)
                                if units > 0:
                                    portfolio_units[sym] = portfolio_units.get(sym, 0) + units
                            has_position = True
                            
                            # Recalculate value after addition for next day's return base
                            current_value = sum(
                                units * prices_today.get(sym, 0)
                                for sym, units in portfolio_units.items()
                            )
                    
                    prev_portfolio_value = current_value
                    daily_values.append(nav_index)
            
            else:
                # ─────────────────────────────────────────────────────────
                # SWING MODE: Single position, hold until sell trigger
                # ─────────────────────────────────────────────────────────
                current_capital = capital
                
                for j, sim_date in enumerate(simulation_dates):
                    df = date_to_df[sim_date]
                    
                    is_buy_day = buy_dates_mask[j]
                    actual_buy_trigger = is_buy_day and not buy_signal_active
                    
                    if is_buy_day:
                        buy_signal_active = True
                    else:
                        buy_signal_active = False
                    
                    # Check sell trigger
                    if sell_dates_mask[j] and portfolio_units:
                        trade_log.append({'Event': 'SELL', 'Date': sim_date})
                        prices_today = df.set_index('symbol')['price']
                        sell_value = sum(
                            units * prices_today.get(sym, 0)
                            for sym, units in portfolio_units.items()
                        )
                        current_capital += sell_value
                        portfolio_units = {}
                        buy_signal_active = False
                    
                    # Execute buy (only if no position)
                    if actual_buy_trigger and not portfolio_units and current_capital > 1000:
                        trade_log.append({'Event': 'BUY', 'Date': sim_date})
                        buy_portfolio = strategy.generate_portfolio(df.copy(), current_capital)
                        
                        if buy_portfolio is not None and not buy_portfolio.empty and 'units' in buy_portfolio.columns:
                            portfolio_units = pd.Series(
                                buy_portfolio['units'].values,
                                index=buy_portfolio['symbol']
                            ).to_dict()
                            current_capital -= buy_portfolio['value'].sum()
                    
                    # Calculate current value
                    portfolio_value = 0
                    if portfolio_units:
                        prices_today = df.set_index('symbol')['price']
                        portfolio_value = sum(
                            units * prices_today.get(sym, 0)
                            for sym, units in portfolio_units.items()
                        )
                    
                    daily_values.append(portfolio_value + current_capital)
            
            # ─────────────────────────────────────────────────────────
            # COMPUTE METRICS
            # ─────────────────────────────────────────────────────────
            
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
            
            # Add trade info
            metrics['buy_events'] = len([t for t in trade_log if t['Event'] == 'BUY'])
            metrics['sell_events'] = len([t for t in trade_log if t['Event'] == 'SELL'])
            metrics['trade_events'] = len(trade_log)
            
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
                'positions': len(portfolio_units) if portfolio_units else 0,
                'trade_log': trade_log
            }
            valid_strategies.append((name, score, metrics))
            
            # Log result
            _dss_logger.info(
                f"  {name:<28} │ Ret: {total_ret:>+6.1%} │ MaxDD: {max_dd:>+6.1%} │ "
                f"Sharpe: {sharpe:>+5.2f} │ Sortino: {sortino:>+6.2f} │ Calmar: {calmar:>+6.2f} │ Trades: {len(trade_log)}"
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
        trades = metrics.get('trade_events', 0)
        _dss_logger.info(f"  {marker} #{rank:<2} {name:<28} │ {metric_label}: {score:>+7.2f} │ Return: {ret:>+6.1%} │ Trades: {trades} {status}")
    
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

        # ═══════════════════════════════════════════════════════════════════
        # TRIGGER-BASED BACKTEST CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════
        st.markdown('<div class="sidebar-title">🎯 Backtest Trigger Settings</div>', unsafe_allow_html=True)
        
        use_trigger_backtest = st.checkbox(
            "Enable Trigger-Based Backtest",
            value=True,
            help="Use REL_BREADTH trigger signals for buy/sell timing (aligned with backtest.py methodology)"
        )
        
        trigger_df = None
        trigger_config = TRIGGER_CONFIG.get(selected_main_branch, TRIGGER_CONFIG['SIP Investment'])
        
        if use_trigger_backtest:
            with st.expander("⚙️ Trigger Configuration", expanded=False):
                # Auto-fetch from Google Sheets
                if STRATEGY_SELECTION_AVAILABLE:
                    st.caption("📡 Trigger Source: Google Sheets (REL_BREADTH)")
                    try:
                        @st.cache_data(ttl=3600, show_spinner=False)
                        def _fetch_breadth_data():
                            return load_breadth_data(lookback_rows=600)
                        
                        breadth_df = _fetch_breadth_data()
                        if not breadth_df.empty:
                            trigger_df = breadth_df.copy()
                            trigger_df = trigger_df.set_index('DATE')
                            st.success(f"✅ Loaded {len(trigger_df)} trigger entries from Google Sheets")
                            
                            # Show recent data preview
                            recent = breadth_df.tail(5).copy()
                            recent['DATE'] = recent['DATE'].dt.strftime('%Y-%m-%d')
                            st.dataframe(recent[['DATE', 'REL_BREADTH']], hide_index=True, width='stretch')
                        else:
                            st.warning("⚠️ No data from Google Sheets. Using first-day entry fallback.")
                    except Exception as e:
                        st.error(f"Error fetching from Google Sheets: {e}")
                        trigger_df = None
                else:
                    # Fallback: file upload if strategy_selection not available
                    trigger_file = st.file_uploader(
                        "Upload Trigger File (CSV/XLSX)",
                        type=['csv', 'xlsx', 'xls'],
                        help="File with DATE and REL_BREADTH columns."
                    )
                    
                    if trigger_file is not None:
                        try:
                            if trigger_file.name.endswith('.csv'):
                                trigger_df = pd.read_csv(trigger_file)
                            else:
                                trigger_df = pd.read_excel(trigger_file)
                            
                            if 'DATE' in trigger_df.columns:
                                trigger_df['DATE'] = pd.to_datetime(trigger_df['DATE'], format='%d-%m-%Y', errors='coerce')
                                trigger_df = trigger_df.dropna(subset=['DATE']).set_index('DATE')
                                st.success(f"✅ Loaded {len(trigger_df)} trigger entries")
                            else:
                                st.warning("File must contain a 'DATE' column")
                                trigger_df = None
                        except Exception as e:
                            st.error(f"Error loading trigger file: {e}")
                            trigger_df = None
                
                # Show current thresholds
                st.markdown(f"**Mode:** {selected_main_branch}")
                
                col1, col2 = st.columns(2)
                with col1:
                    buy_thresh = st.number_input(
                        "Buy Threshold",
                        value=trigger_config['buy_threshold'],
                        min_value=0.0,
                        max_value=2.0,
                        step=0.01,
                        help="Buy when REL_BREADTH < this value"
                    )
                with col2:
                    sell_thresh = st.number_input(
                        "Sell Threshold",
                        value=trigger_config['sell_threshold'],
                        min_value=0.1,
                        max_value=3.0,
                        step=0.01,
                        help="Sell when REL_BREADTH >= this value"
                    )
                
                sell_enabled = st.checkbox(
                    "Enable Sell Trigger",
                    value=trigger_config['sell_enabled'],
                    help="If disabled, positions are held until end of period"
                )
                
                # Update trigger config
                trigger_config = {
                    'buy_threshold': buy_thresh,
                    'sell_threshold': sell_thresh,
                    'sell_enabled': sell_enabled
                }
                
                st.caption(f"📊 Buy: REL_BREADTH < {buy_thresh} | Sell: REL_BREADTH >= {sell_thresh} ({'enabled' if sell_enabled else 'disabled'})")
        
        # Store in session state for later use
        st.session_state.use_trigger_backtest = use_trigger_backtest
        st.session_state.trigger_df = trigger_df
        st.session_state.trigger_config = trigger_config

        if st.button("Run Analysis", width='stretch', type="primary"):
            
            lookback_files = 100
            
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
            # PHASE 2: DYNAMIC STRATEGY SELECTION (TRIGGER-BASED)
            # ═══════════════════════════════════════════════════════════════════
            progress_bar.progress(0.25, text="Running strategy selection...")
            status_text.text(f"Analyzing {len(strategies)} strategies...")
            
            # Get trigger configuration for strategy selection
            use_trigger = st.session_state.get('use_trigger_backtest', True)
            trigger_df = st.session_state.get('trigger_df', None)
            trigger_config = st.session_state.get('trigger_config', TRIGGER_CONFIG.get(selected_main_branch, {}))
            
            logging.info("-" * 70)
            logging.info(f"[PHASE 2/4] DYNAMIC STRATEGY SELECTION (TRIGGER-BASED)")
            logging.info(f"  Investment Style: {selected_main_branch}")
            logging.info(f"  Market Regime: {final_mix_to_use}")
            logging.info(f"  Trigger Mode: {use_trigger}")
            if use_trigger:
                logging.info(f"  Buy Threshold: {trigger_config.get('buy_threshold', 0.42)}")
                logging.info(f"  Sell Enabled: {trigger_config.get('sell_enabled', False)}")
            
            dynamic_strategies, strategy_metrics = _run_dynamic_strategy_selection(
                training_data_window_with_current, 
                strategies, 
                selected_main_branch,
                progress_bar=progress_bar,
                status_text=status_text,
                trigger_df=trigger_df if use_trigger else None,
                trigger_config=trigger_config if use_trigger else None
            )
            
            # Store Phase 2 metrics for Strategy Metrics tab
            st.session_state.phase2_strategy_metrics = strategy_metrics
            
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
            # PHASE 3: WALK-FORWARD PORTFOLIO CURATION (Pure — No Triggers)
            # ═══════════════════════════════════════════════════════════════════
            # Phase 3 uses a shorter window (50 days) than Phase 2 (100 days)
            # - Phase 2 needs breadth for robust strategy selection across regimes
            # - Phase 3 needs recency for adaptive walk-forward weights
            #
            # ARCHITECTURAL PRINCIPLE:
            # Triggers (REL_BREADTH) are used in Phase 2 to SELECT strategies
            # that perform best under our timing system. Phase 3 evaluates the
            # CURATION QUALITY: "Given these strategies, how well can we pick 
            # stocks day-by-day?" This is a pure walk-forward process — every
            # day is a rebalancing day. The resulting metrics (Sharpe, Sortino,
            # MaxDD) measure stock-picking ability, not timing ability.
            PHASE3_LOOKBACK = 50
            if len(training_data_window_with_current) > PHASE3_LOOKBACK:
                phase3_data = training_data_window_with_current[-PHASE3_LOOKBACK:]
            else:
                phase3_data = training_data_window_with_current
            
            progress_bar.progress(0.65, text="Running walk-forward portfolio curation...")
            status_text.text(f"Walk-forward: {len(strategies_to_run)} strategies over {len(phase3_data)} days...")
            
            logging.info("-" * 70)
            logging.info(f"[PHASE 3/4] WALK-FORWARD PORTFOLIO CURATION (PURE)")
            logging.info(f"  Mode: {selected_main_branch}")
            logging.info(f"  Strategies: {list(strategies_to_run.keys())}")
            logging.info(f"  Phase 3 Window: {len(phase3_data)} days (of {len(training_data_window_with_current)} total)")
            logging.info(f"  Method: Daily rebalancing walk-forward (no trigger dependency)")
            
            st.session_state.performance = evaluate_historical_performance(strategies_to_run, phase3_data)
            
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

        tab1, tab2, tab3, tab4 = st.tabs(["**Portfolio**", "**Performance**", "**Strategy Deep Dive**", "**Strategy Metrics**"])

        with tab1:
            st.markdown(_section_header("Curated Portfolio Holdings", f"{len(st.session_state.portfolio)} positions from multi-strategy walk-forward curation"), unsafe_allow_html=True)
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
            
            _section_divider()
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
            st.markdown(_section_header("Strategy Deep Dive", "Institutional analytics across individual strategies — tier analysis, risk-return profiling & conviction signals"), unsafe_allow_html=True)
            
            strategies_in_performance = [k for k in st.session_state.performance.get('strategy', {}).keys() if k != 'System_Curated']
            
            if not strategies_in_performance:
                st.warning("No individual strategy data available for deep dive analysis.")
            else:
                # ═══════════════════════════════════════════════════════════════════════════
                # SECTION 1: TIER SHARPE HEATMAP
                # ═══════════════════════════════════════════════════════════════════════════
                subset_perf = st.session_state.performance.get('subset', {})
                
                if subset_perf:
                    _section_divider()
                    st.markdown(_section_header("Sharpe by Position Tier", "Performance decay across 10-stock tiers — top tier should dominate"), unsafe_allow_html=True)
                    
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
                            **_chart_layout_base(max(350, len(strategies_in_performance) * 30), "Sharpe Ratio by 10-Stock Tier"),
                            margin=dict(l=120, r=20, t=50, b=40),
                            coloraxis_colorbar=dict(title="Sharpe")
                        )
                        st.plotly_chart(fig_tier, width="stretch")
                        
                        # Tier insights — metric cards
                        tier_means = df_heatmap.mean(axis=0)
                        c1, c2, c3 = st.columns(3)
                        best_tier = tier_means.idxmax()
                        worst_tier = tier_means.idxmin()
                        with c1:
                            st.markdown(_metric_card("Best Tier", best_tier, f"Sharpe {tier_means.max():.2f}", "success"), unsafe_allow_html=True)
                        with c2:
                            st.markdown(_metric_card("Worst Tier", worst_tier, f"Sharpe {tier_means.min():.2f}", "danger"), unsafe_allow_html=True)
                        with c3:
                            st.markdown(_metric_card("Dispersion", f"{tier_means.std():.2f}", "Cross-tier σ", "info"), unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════════════════════
                # SECTION 2: RISK-RETURN SCATTER & FACTOR RADAR (side by side)
                # ═══════════════════════════════════════════════════════════════════════════
                _section_divider()
                st.markdown(_section_header("Risk-Return & Factor Profile", "Strategy positioning on the efficient frontier and multi-factor fingerprints"), unsafe_allow_html=True)
                
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
                
                col_scatter, col_radar = st.columns(2)
                
                with col_scatter:
                    st.markdown("#### Risk-Return Scatter")
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
                                    size=np.clip(df_scatter['Size'], 12, 40),
                                    color=df_scatter['Sharpe'],
                                    colorscale='RdYlGn',
                                    cmin=-1, cmax=2,
                                    showscale=True,
                                    colorbar=dict(title='Sharpe', tickfont=dict(color='#888888')),
                                    line=dict(width=2, color='rgba(255,255,255,0.8)'),
                                    opacity=0.95
                                ),
                                text=df_scatter['Strategy'].apply(lambda x: x[:10]),
                                textposition='top center',
                                textfont=dict(size=10, color='#EAEAEA'),
                                customdata=df_scatter[['Strategy', 'Max DD']].values,
                                hovertemplate='<b>%{customdata[0]}</b><br>CAGR: %{y:.1f}%<br>Vol: %{x:.1f}%<extra></extra>'
                            ))
                            
                            if len(df_scatter) > 2:
                                max_sharpe_idx = df_scatter['Sharpe'].idxmax()
                                tangent_vol = df_scatter.loc[max_sharpe_idx, 'Vol_pct']
                                tangent_ret = df_scatter.loc[max_sharpe_idx, 'CAGR_pct']
                                vol_max = df_scatter['Vol_pct'].max()
                                cml_end_vol = min(tangent_vol * 1.8, vol_max * 1.3)
                                cml_end_ret = tangent_ret * (cml_end_vol / tangent_vol) if tangent_vol > 0 else 0
                                
                                fig_scatter.add_trace(go.Scatter(
                                    x=[0, cml_end_vol], y=[0, cml_end_ret],
                                    mode='lines', name='CML',
                                    line=dict(color='#888888', dash='dash', width=1.5),
                                    showlegend=False
                                ))
                                fig_scatter.add_trace(go.Scatter(
                                    x=[tangent_vol], y=[tangent_ret],
                                    mode='markers', name='Optimal',
                                    marker=dict(size=15, color='#FFC300', symbol='star',
                                               line=dict(width=2, color='#EAEAEA')),
                                    showlegend=False
                                ))
                            
                            vol_range = df_scatter['Vol_pct'].max() - df_scatter['Vol_pct'].min()
                            cagr_range = df_scatter['CAGR_pct'].max() - df_scatter['CAGR_pct'].min()
                            vol_pad = max(vol_range * 0.15, 1)
                            cagr_pad = max(cagr_range * 0.15, 0.5)
                            
                            fig_scatter.update_layout(
                                **_chart_layout_base(400, "Volatility vs CAGR (Bubble = MaxDD, Color = Sharpe)"),
                                margin=dict(l=50, r=20, t=50, b=50)
                            )
                            fig_scatter.update_xaxes(
                                title='Volatility (%)', gridcolor='rgba(255,255,255,0.05)',
                                range=[max(0, df_scatter['Vol_pct'].min() - vol_pad), df_scatter['Vol_pct'].max() + vol_pad]
                            )
                            fig_scatter.update_yaxes(
                                title='CAGR (%)', gridcolor='rgba(255,255,255,0.05)',
                                range=[df_scatter['CAGR_pct'].min() - cagr_pad, df_scatter['CAGR_pct'].max() + cagr_pad]
                            )
                        
                        st.plotly_chart(fig_scatter, width="stretch")
                
                with col_radar:
                    st.markdown("#### Factor Fingerprint")
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
                            palette = ['#FFC300', '#10b981', '#06b6d4', '#f59e0b']
                            
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
                                        tickfont=dict(size=9, color='#888888'),
                                        gridcolor='rgba(255,255,255,0.05)'
                                    ),
                                    angularaxis=dict(
                                        tickfont=dict(size=11, color='#EAEAEA'),
                                        gridcolor='rgba(255,255,255,0.05)'
                                    ),
                                    bgcolor='rgba(0,0,0,0)'
                                ),
                                showlegend=True,
                                legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center', font=dict(size=10)),
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#EAEAEA'),
                                height=400,
                                margin=dict(l=60, r=60, t=50, b=60),
                                title=dict(text="Multi-Factor Strategy Comparison", font=dict(size=11, color='#888888'), x=0, xanchor='left')
                            )
                        
                        st.plotly_chart(fig_radar, width="stretch")
                
                # ═══════════════════════════════════════════════════════════════════════════
                # SECTION 3: TIER ALLOCATION HISTORY
                # ═══════════════════════════════════════════════════════════════════════════
                _section_divider()
                st.markdown(_section_header("Tier Allocation History", "How subset tier weights evolved through the walk-forward window"), unsafe_allow_html=True)
                
                display_subset_weight_evolution(
                    st.session_state.performance.get('subset_weights_history', []),
                    strategies_in_performance
                )
                
                # ═══════════════════════════════════════════════════════════════════════════
                # SECTION 4: CONVICTION ANALYSIS
                # ═══════════════════════════════════════════════════════════════════════════
                _section_divider()
                st.markdown(_section_header("Cross-Strategy Conviction", "Signal overlap and consensus across selected strategies"), unsafe_allow_html=True)
                
                strategies_for_heatmap = {name: strategies[name] for name in strategies_in_performance if name in strategies}
                
                if strategies_for_heatmap and st.session_state.current_df is not None:
                    heatmap_fig = create_conviction_heatmap(strategies_for_heatmap, st.session_state.current_df)
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, width="stretch")
                    
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
                        
                        consensus_threshold = len(strategies_for_heatmap) / 2
                        high_conviction = [s for s, c in sorted_signals if c >= consensus_threshold]
                        avg_agreement = np.mean([c for _, c in sorted_signals]) / len(strategies_for_heatmap)
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(_metric_card("High Conviction", str(len(high_conviction)), f"≥{consensus_threshold:.0f} strategy agreement", "primary"), unsafe_allow_html=True)
                        with c2:
                            st.markdown(_metric_card("Signal Agreement", f"{avg_agreement:.0%}", "Mean strategy overlap", "info"), unsafe_allow_html=True)
                        with c3:
                            top_pick = top_consensus[0] if top_consensus else ("—", 0)
                            st.markdown(_metric_card("Top Consensus", top_pick[0], f"{top_pick[1]}/{len(strategies_for_heatmap)} strategies", "success"), unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════════════════════
                # SECTION 5: ADAPTIVE SELECTION RANKING
                # ═══════════════════════════════════════════════════════════════════════════
                _section_divider()
                st.markdown(_section_header("Adaptive Selection Ranking", "Dispersion-weighted rank composite — metrics with higher cross-sectional variance get more weight"), unsafe_allow_html=True)
                
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
                    
                    rank_metrics = ['Sharpe', 'Sortino', 'Calmar', 'Win Rate']
                    for col in rank_metrics:
                        df_summary[f'{col}_Rank'] = df_summary[col].rank(pct=True)
                    df_summary['DD_Rank'] = df_summary['Max DD'].rank(pct=True, ascending=False)
                    
                    rank_cols = [c for c in df_summary.columns if c.endswith('_Rank')]
                    dispersions = {col: df_summary[col].std() for col in rank_cols}
                    total_disp = sum(dispersions.values()) or 1
                    weights = {col: disp / total_disp for col, disp in dispersions.items()}
                    
                    df_summary['Score'] = sum(df_summary[col] * w for col, w in weights.items())
                    df_summary = df_summary.sort_values('Score', ascending=False)
                    
                    df_display = df_summary[['Strategy', 'Sharpe', 'Sortino', 'Calmar', 'Max DD', 'Win Rate', 'T1 Sharpe', 'Score']].copy()
                    df_display['Sharpe'] = df_display['Sharpe'].apply(lambda x: f"{x:.2f}")
                    df_display['Sortino'] = df_display['Sortino'].apply(lambda x: f"{x:.2f}")
                    df_display['Calmar'] = df_display['Calmar'].apply(lambda x: f"{x:.2f}")
                    df_display['Max DD'] = df_display['Max DD'].apply(lambda x: f"{x:.1%}")
                    df_display['Win Rate'] = df_display['Win Rate'].apply(lambda x: f"{x:.0%}")
                    df_display['T1 Sharpe'] = df_display['T1 Sharpe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                    df_display['Score'] = df_display['Score'].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(df_display, width="stretch", hide_index=True)

        # ═══════════════════════════════════════════════════════════════════════════
        # TAB 4: STRATEGY METRICS - Comprehensive Backtest Results
        # ═══════════════════════════════════════════════════════════════════════════
        with tab4:
            st.markdown(_section_header("Strategy Metrics & Backtest Data", "Comprehensive walk-forward and trigger-based backtest results across all strategies"), unsafe_allow_html=True)
            
            strategies_in_performance = list(st.session_state.performance.get('strategy', {}).keys())
            
            if not strategies_in_performance:
                st.warning("No strategy data available. Run analysis first.")
            else:
                # ─────────────────────────────────────────────────────────────────
                # SECTION 1: SELECTED STRATEGY METRICS (from Phase 3 walk-forward)
                # ─────────────────────────────────────────────────────────────────
                metrics_data = []
                for name in strategies_in_performance:
                    strategy_perf = st.session_state.performance.get('strategy', {}).get(name, {})
                    metrics = strategy_perf.get('metrics', {})
                    returns_list = strategy_perf.get('returns', [])
                    
                    row = {
                        'Strategy': name,
                        'Total Return': metrics.get('total_return', 0),
                        'CAGR': metrics.get('annual_return', metrics.get('ann_return', 0)),
                        'Volatility': metrics.get('volatility', 0),
                        'Sharpe Ratio': metrics.get('sharpe', 0),
                        'Sortino Ratio': metrics.get('sortino', 0),
                        'Calmar Ratio': metrics.get('calmar', 0),
                        'Max Drawdown': metrics.get('max_drawdown', metrics.get('max_dd', 0)),
                        'Win Rate': metrics.get('win_rate', 0),
                        'Profit Factor': metrics.get('profit_factor', 0),
                        'Omega Ratio': metrics.get('omega_ratio', 0),
                        'Tail Ratio': metrics.get('tail_ratio', 0),
                        'Gain/Pain': metrics.get('gain_to_pain', 0),
                        'Trading Days': len(returns_list)
                    }
                    metrics_data.append(row)
                
                if metrics_data:
                    # Summary metric cards for the selected strategies
                    df_metrics = pd.DataFrame(metrics_data)
                    df_metrics = df_metrics.sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)
                    
                    best_strat = df_metrics.iloc[0]
                    avg_sharpe = df_metrics['Sharpe Ratio'].mean()
                    avg_return = df_metrics['Total Return'].mean()
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(_metric_card("Selected Strategies", str(len(df_metrics)), "Phase 3 walk-forward", "primary"), unsafe_allow_html=True)
                    with c2:
                        cls = 'success' if avg_sharpe > 0.5 else 'warning' if avg_sharpe > 0 else 'danger'
                        st.markdown(_metric_card("Avg Sharpe", f"{avg_sharpe:.2f}", "Cross-strategy mean", cls), unsafe_allow_html=True)
                    with c3:
                        cls = 'success' if avg_return > 0 else 'danger'
                        st.markdown(_metric_card("Avg Return", f"{avg_return:.1%}", "Mean total return", cls), unsafe_allow_html=True)
                    with c4:
                        st.markdown(_metric_card("Top Strategy", best_strat['Strategy'][:15], f"Sharpe {best_strat['Sharpe Ratio']:.2f}", "success"), unsafe_allow_html=True)
                    
                    _section_divider()
                    st.markdown(_section_header("Walk-Forward Performance Table", "Phase 3 out-of-sample metrics for selected strategies"), unsafe_allow_html=True)
                    
                    styled_cols = {
                        'Total Return': '{:.2%}', 'CAGR': '{:.2%}', 'Volatility': '{:.2%}',
                        'Sharpe Ratio': '{:.3f}', 'Sortino Ratio': '{:.3f}', 'Calmar Ratio': '{:.3f}',
                        'Max Drawdown': '{:.2%}', 'Win Rate': '{:.1%}', 'Profit Factor': '{:.2f}',
                        'Omega Ratio': '{:.2f}', 'Tail Ratio': '{:.2f}', 'Gain/Pain': '{:.2f}',
                        'Trading Days': '{:.0f}'
                    }
                    
                    styled_df = df_metrics.style.format(styled_cols)
                    gradient_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Total Return']
                    available_gradient_cols = [c for c in gradient_cols if c in df_metrics.columns]
                    if available_gradient_cols:
                        try:
                            styled_df = styled_df.background_gradient(subset=available_gradient_cols, cmap='RdYlGn')
                        except ImportError:
                            pass
                    
                    st.dataframe(styled_df, width='stretch', hide_index=True)
                    
                    csv_data = df_metrics.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Selected Strategy Metrics CSV",
                        data=csv_data,
                        file_name=f"selected_strategy_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # ─────────────────────────────────────────────────────────────────
                # SECTION 2: ALL STRATEGIES FROM PHASE 2 (Selection Backtest)
                # ─────────────────────────────────────────────────────────────────
                _section_divider()
                st.markdown(_section_header("Phase 2 Strategy Selection — All Strategies", "Trigger-based backtest results used for dynamic strategy selection"), unsafe_allow_html=True)
                
                phase2_metrics = st.session_state.get('phase2_strategy_metrics', {})
                
                if phase2_metrics:
                    all_strat_data = []
                    for name, data in phase2_metrics.items():
                        if not isinstance(data, dict) or data.get('status') != 'ok':
                            continue
                        m = data.get('metrics', {})
                        all_strat_data.append({
                            'Strategy': name,
                            'Total Return': m.get('total_return', 0),
                            'Ann. Return': m.get('ann_return', 0),
                            'Volatility': m.get('volatility', 0),
                            'Sharpe': m.get('sharpe', 0),
                            'Sortino': m.get('sortino', 0),
                            'Calmar': m.get('calmar', 0),
                            'Max DD': m.get('max_dd', 0),
                            'Win Rate': m.get('win_rate', 0),
                            'Trades': m.get('trade_events', 0),
                        })
                    
                    if all_strat_data:
                        df_all = pd.DataFrame(all_strat_data)
                        df_all = df_all.sort_values('Sharpe', ascending=False).reset_index(drop=True)
                        
                        # Summary cards for Phase 2
                        n_selected = len([k for k in st.session_state.performance.get('strategy', {}).keys() if k != 'System_Curated'])
                        n_total = len(df_all)
                        select_rate = n_selected / n_total * 100 if n_total > 0 else 0
                        
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.markdown(_metric_card("Evaluated", str(n_total), "Total strategies", "primary"), unsafe_allow_html=True)
                        with c2:
                            st.markdown(_metric_card("Selected", str(n_selected), f"{select_rate:.0f}% selection rate", "success"), unsafe_allow_html=True)
                        with c3:
                            avg_p2_sharpe = df_all['Sharpe'].mean()
                            cls = 'success' if avg_p2_sharpe > 0.5 else 'warning' if avg_p2_sharpe > 0 else 'danger'
                            st.markdown(_metric_card("Avg Sharpe", f"{avg_p2_sharpe:.2f}", "All strategies", cls), unsafe_allow_html=True)
                        with c4:
                            best_p2 = df_all.iloc[0]
                            st.markdown(_metric_card("Best Strategy", best_p2['Strategy'][:15], f"Sharpe {best_p2['Sharpe']:.2f}", "info"), unsafe_allow_html=True)
                        
                        # Mark selected strategies
                        selected_strats = [k for k in st.session_state.performance.get('strategy', {}).keys() if k != 'System_Curated']
                        df_all.insert(1, 'Selected', df_all['Strategy'].apply(lambda x: '✅' if x in selected_strats else ''))
                        
                        all_styled_cols = {
                            'Total Return': '{:.2%}', 'Ann. Return': '{:.2%}', 'Volatility': '{:.2%}',
                            'Sharpe': '{:.2f}', 'Sortino': '{:.2f}', 'Calmar': '{:.2f}',
                            'Max DD': '{:.2%}', 'Win Rate': '{:.1%}', 'Trades': '{:.0f}'
                        }
                        
                        styled_all = df_all.style.format(all_styled_cols)
                        try:
                            styled_all = styled_all.background_gradient(
                                subset=['Sharpe', 'Sortino', 'Calmar'], cmap='RdYlGn'
                            )
                        except ImportError:
                            pass
                        
                        st.dataframe(styled_all, width='stretch', hide_index=True, height=600)
                        st.caption(f"Showing {len(df_all)} strategies evaluated in Phase 2 (trigger-based backtest). ✅ = selected for portfolio curation.")
                        
                        all_csv = df_all.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Strategies CSV",
                            data=all_csv,
                            file_name=f"all_strategy_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("Phase 2 strategy selection metrics not available. Run analysis with dynamic selection enabled.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Dynamic footer with IST time (timezone-aware)
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")

if __name__ == "__main__":
    main()
