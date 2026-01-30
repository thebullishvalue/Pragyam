"""
PRAGYAM (प्रज्ञम) - Portfolio Intelligence | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Walk-forward portfolio curation with regime-aware strategy allocation.
Multi-strategy backtesting and capital optimization engine.

v2.0.0: Dynamic Strategy Selection
- Strategies now selected based on backtest performance
- SIP Mode: Top 4 strategies by Calmar Ratio
- Swing Mode: Top 4 strategies by Sortino Ratio
- Unified data pipeline with shared resources
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


# --- Import Strategies from strategies.py ---
try:
    from strategies import (
        BaseStrategy, GameTheoreticStrategy, MomentumAccelerator, VolatilitySurfer, 
        DivineMomentumOracle, AdaptiveVolBreakout, NebulaMomentumStorm, CelestialAlphaForge
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
        DataCacheManager,
        get_dynamic_portfolio_styles,
        initialize_backtest_engine
    )
    BACKTEST_ENGINE_AVAILABLE = True
except ImportError:
    BACKTEST_ENGINE_AVAILABLE = False
    logging.warning("backtest_engine.py not found. Using static strategy selection.")


# --- System Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('system.log'), logging.StreamHandler()])
st.set_page_config(page_title="PRAGYAM | Portfolio Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# --- Constants ---
VERSION = "v2.0.0"  # Updated for dynamic strategy selection
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
    
    /* Typography */
    h1, h2, h3, h4 {
        color: var(--text-primary);
        font-weight: 700;
    }
    
    h1 { font-size: 2rem; }
    h2 { font-size: 1.5rem; border-bottom: 2px solid var(--border-color); padding-bottom: 0.5rem; }
    h3 { font-size: 1.25rem; }
    
    /* Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.15);
    }
    
    /* Buttons */
    .stButton > button {
        background: transparent;
        border: 2px solid var(--primary-color);
        color: var(--primary-color);
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--primary-color);
        color: var(--background-color);
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.4);
    }
    
    .stButton > button[kind="primary"] {
        background: var(--primary-color);
        color: var(--background-color);
    }
    
    /* Sidebar */
    .sidebar-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 1.5rem 0 0.75rem 0;
    }
    
    .section-divider {
        border-top: 1px solid var(--border-color);
        margin: 1.5rem 0;
    }
    
    .info-box {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Dynamic Strategy Badge */
    .strategy-badge {
        display: inline-block;
        background: rgba(var(--primary-rgb), 0.15);
        border: 1px solid var(--primary-color);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: var(--primary-color);
        margin: 2px;
    }
    
    .dynamic-selection-indicator {
        background: linear-gradient(135deg, rgba(var(--primary-rgb), 0.1), rgba(var(--primary-rgb), 0.05));
        border-left: 3px solid var(--primary-color);
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Tables */
    .stDataFrame {
        background: var(--bg-card);
        border-radius: 8px;
    }
    
    /* Plotly Charts */
    .stPlotlyChart {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'performance' not in st.session_state: st.session_state.performance = None
if 'portfolio' not in st.session_state: st.session_state.portfolio = None
if 'current_df' not in st.session_state: st.session_state.current_df = None
if 'selected_date' not in st.session_state: st.session_state.selected_date = None
if 'min_pos_pct' not in st.session_state: st.session_state.min_pos_pct = 1.0
if 'max_pos_pct' not in st.session_state: st.session_state.max_pos_pct = 10.0
if 'suggested_mix' not in st.session_state: st.session_state.suggested_mix = None
if 'regime_display' not in st.session_state: st.session_state.regime_display = None

# --- NEW: Dynamic Strategy Selection State ---
if 'dynamic_portfolio_styles' not in st.session_state: st.session_state.dynamic_portfolio_styles = None
if 'strategy_leaderboard' not in st.session_state: st.session_state.strategy_leaderboard = None
if 'backtest_engine' not in st.session_state: st.session_state.backtest_engine = None
if 'backtest_complete' not in st.session_state: st.session_state.backtest_complete = False


# --- Fallback Static PORTFOLIO_STYLES (used if backtest engine unavailable) ---
STATIC_PORTFOLIO_STYLES = {
    "Swing Trading": {
        "description": "Short-term (3-21 day) holds to capture rapid momentum and volatility.",
        "mixes": {
            "Bull Market Mix": {
                "strategies": ['GameTheoreticStrategy', 'NebulaMomentumStorm', 'VolatilitySurfer', 'CelestialAlphaForge'],
                "rationale": "Static selection (backtest engine not available)"
            },
            "Bear Market Mix": {
                "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                "rationale": "Static selection (backtest engine not available)"
            },
            "Chop/Consolidate Mix": {
                "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                "rationale": "Static selection (backtest engine not available)"
            }
        }
    },
    "SIP Investment": {
        "description": "Systematic long-term (3-12+ months) wealth accumulation. Focus on consistency and drawdown protection.",
        "mixes": {
            "Bull Market Mix": {
                "strategies": ['GameTheoreticStrategy', 'MomentumAccelerator', 'VolatilitySurfer', 'DivineMomentumOracle'],
                "rationale": "Static selection (backtest engine not available)"
            },
            "Bear Market Mix": {
                "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                "rationale": "Static selection (backtest engine not available)"
            },
            "Chop/Consolidate Mix": {
                "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                "rationale": "Static selection (backtest engine not available)"
            }
        }
    }
}


# ============================================================================
# DYNAMIC STRATEGY SELECTION FUNCTIONS
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_shared_data_cache():
    """Get shared data cache manager for resource optimization."""
    if BACKTEST_ENGINE_AVAILABLE:
        return DataCacheManager()
    return None


def run_dynamic_strategy_selection(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    capital: float = 10_000_000,
    n_strategies: int = 4
) -> Optional[Dict]:
    """
    Run the backtest engine to dynamically select strategies.
    
    Args:
        symbols: List of stock symbols
        start_date: Backtest start date
        end_date: Backtest end date
        capital: Backtest capital
        n_strategies: Number of strategies per mix
        
    Returns:
        Dynamic PORTFOLIO_STYLES dictionary or None if failed
    """
    if not BACKTEST_ENGINE_AVAILABLE:
        st.warning("⚠️ Backtest engine not available. Using static strategy selection.")
        return None
    
    progress_placeholder = st.empty()
    
    def update_progress(p, msg):
        progress_placeholder.progress(p, text=msg)
    
    try:
        # Initialize engine with shared cache
        engine = UnifiedBacktestEngine(capital=capital)
        
        # Load data using shared resources
        update_progress(0.1, "Loading market data...")
        historical_data = engine.load_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        if not historical_data:
            st.error("Failed to load historical data for strategy selection.")
            progress_placeholder.empty()
            return None
        
        # Load all strategies
        update_progress(0.2, "Loading strategies...")
        engine.load_strategies()
        
        # Create generator
        generator = DynamicPortfolioStylesGenerator(engine)
        
        # Run comprehensive backtest
        update_progress(0.3, "Running SIP backtest...")
        sip_results, swing_results = generator.run_comprehensive_backtest(
            progress_callback=lambda p, m: update_progress(0.3 + p * 0.5, m)
        )
        
        # Generate portfolio styles
        update_progress(0.85, "Selecting top strategies...")
        portfolio_styles = generator.generate_portfolio_styles(n_strategies=n_strategies)
        
        # Store leaderboard for display
        st.session_state.strategy_leaderboard = {
            'sip': generator.get_strategy_leaderboard('sip'),
            'swing': generator.get_strategy_leaderboard('swing')
        }
        
        # Store engine for later use
        st.session_state.backtest_engine = engine
        
        update_progress(1.0, "Strategy selection complete!")
        time.sleep(0.5)
        progress_placeholder.empty()
        
        return portfolio_styles
        
    except Exception as e:
        progress_placeholder.empty()
        st.error(f"Strategy selection failed: {e}")
        logging.error(f"Dynamic strategy selection error: {e}")
        return None


def get_active_portfolio_styles() -> Dict:
    """
    Get the active PORTFOLIO_STYLES - either dynamic or static.
    
    Returns:
        PORTFOLIO_STYLES dictionary
    """
    if st.session_state.dynamic_portfolio_styles is not None:
        return st.session_state.dynamic_portfolio_styles
    return STATIC_PORTFOLIO_STYLES


# ============================================================================
# EXISTING PRAGYAM FUNCTIONS (Preserved from original)
# ============================================================================

def fix_csv_export(df: pd.DataFrame) -> bytes:
    """Export DataFrame to CSV bytes with proper encoding."""
    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()


@st.cache_data(ttl=1800, show_spinner=False)
def load_historical_data(selected_date: datetime, lookback: int) -> List[Tuple[datetime, pd.DataFrame]]:
    """
    Load historical data using backdata.py's generate_historical_data.
    Uses caching to optimize API calls.
    """
    total_days = int((lookback + MAX_INDICATOR_PERIOD) * 1.5) + 30
    fetch_start = selected_date - timedelta(days=total_days)
    
    all_data = generate_historical_data(SYMBOLS_UNIVERSE, fetch_start, selected_date)
    
    if not all_data:
        return []
    
    # Filter to only include dates up to selected_date
    filtered = [(d, df) for d, df in all_data if d <= selected_date]
    return sorted(filtered, key=lambda x: x[0])


def compute_portfolio_return(portfolio: pd.DataFrame, next_prices: pd.DataFrame) -> float:
    """Compute portfolio return given next period prices."""
    if portfolio.empty or 'value' not in portfolio.columns or portfolio['value'].sum() == 0:
        return 0.0
    merged = portfolio.merge(next_prices[['symbol', 'price']], on='symbol', how='inner', suffixes=('_prev', '_next'))
    if merged.empty:
        return 0.0
    returns = (merged['price_next'] - merged['price_prev']) / merged['price_prev']
    return np.average(returns, weights=merged['value'])


def calculate_advanced_metrics(returns_with_dates: List[Dict]) -> Tuple[Dict, float]:
    """Calculate comprehensive performance metrics from returns."""
    default_metrics = {
        'total_return': 0, 'annual_return': 0, 'volatility': 0, 'sharpe': 0,
        'sortino': 0, 'max_drawdown': 0, 'calmar': 0, 'win_rate': 0, 'kelly_criterion': 0
    }
    if len(returns_with_dates) < 2:
        return default_metrics, 52

    returns_df = pd.DataFrame(returns_with_dates).sort_values('date').set_index('date')
    time_deltas = returns_df.index.to_series().diff().dt.days
    avg_period_days = time_deltas.mean()
    periods_per_year = 365.25 / avg_period_days if pd.notna(avg_period_days) and avg_period_days > 0 else 1

    returns = returns_df['return']
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / volatility if volatility != 0 else 0

    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(periods_per_year) if not downside_returns.empty else 0
    sortino = annual_return / downside_vol if downside_vol != 0 else 0

    cumulative = (1 + returns).cumprod()
    max_drawdown = (cumulative / cumulative.expanding(min_periods=1).max() - 1).min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    win_rate = (returns > 0).mean()

    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0

    metrics = {
        'total_return': total_return, 'annual_return': annual_return, 'volatility': volatility,
        'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_drawdown, 'calmar': calmar,
        'win_rate': win_rate, 'kelly_criterion': kelly
    }
    return metrics, periods_per_year


def calculate_strategy_weights(performance: Dict) -> Dict[str, float]:
    """Calculate strategy weights based on Sharpe ratios."""
    strat_names = list(performance['strategy'].keys())
    if not strat_names:
        return {}

    sharpe_values = np.array([performance['strategy'][name].get('sharpe', 0) + 2 for name in strat_names])

    if sharpe_values.size == 0:
        return {name: 1.0 / len(strat_names) for name in strat_names} if strat_names else {}

    stable_sharpes = sharpe_values - np.max(sharpe_values)
    exp_sharpes = np.exp(stable_sharpes)
    total_score = np.sum(exp_sharpes)

    if total_score == 0 or not np.isfinite(total_score):
        return {name: 1.0 / len(strat_names) for name in strat_names}

    weights = exp_sharpes / total_score
    return {name: weights[i] for i, name in enumerate(strat_names)}


def _calculate_performance_on_window(
    window_data: List[Tuple[datetime, pd.DataFrame]],
    strategies: Dict[str, BaseStrategy],
    training_capital: float
) -> Dict:
    """Calculate performance metrics on a training window."""
    performance = {name: {'returns': []} for name in strategies}
    subset_performance = {name: {} for name in strategies}
    
    for i in range(len(window_data) - 1):
        date, df = window_data[i]
        next_date, next_df = window_data[i+1]
        
        for name, strategy in strategies.items():
            try:
                portfolio = strategy.generate_portfolio(df, training_capital)
                if portfolio.empty:
                    continue
                    
                performance[name]['returns'].append({
                    'return': compute_portfolio_return(portfolio, next_df),
                    'date': next_date
                })
                
                n, tier_size = len(portfolio), 10
                num_tiers = n // tier_size
                if num_tiers == 0:
                    continue
                    
                for j in range(num_tiers):
                    tier_name = f'tier_{j+1}'
                    if tier_name not in subset_performance[name]:
                        subset_performance[name][tier_name] = []
                    sub_df = portfolio.iloc[j*tier_size : (j+1)*tier_size]
                    if not sub_df.empty:
                        sub_ret = compute_portfolio_return(sub_df, next_df)
                        subset_performance[name][tier_name].append({'return': sub_ret, 'date': next_date})
                        
            except Exception as e:
                logging.error(f"Window Calc Error ({name}, {date}): {e}")
    
    final_performance = {
        name: {
            'metrics': calculate_advanced_metrics(perf['returns'])[0],
            'sharpe': calculate_advanced_metrics(perf['returns'])[0]['sharpe']
        }
        for name, perf in performance.items()
    }
    
    final_sub_performance = {
        name: {sub: calculate_advanced_metrics(sub_perf)[0]['sharpe'] for sub, sub_perf in data.items() if sub_perf}
        for name, data in subset_performance.items()
    }
    
    return {'strategy': final_performance, 'subset': final_sub_performance}


def evaluate_historical_performance(
    _strategies: Dict[str, BaseStrategy],
    historical_data: List[Tuple[datetime, pd.DataFrame]]
) -> Dict:
    """Evaluate historical performance using walk-forward analysis."""
    MIN_TRAIN_FILES = 2
    TRAINING_CAPITAL = 2500000.0
    
    if len(historical_data) < MIN_TRAIN_FILES + 1:
        st.error(f"Not enough historical data. Need at least {MIN_TRAIN_FILES + 1} files.")
        return {}

    all_names = list(_strategies.keys()) + ['System_Curated']
    all_returns = {name: [] for name in all_names}
    strategy_weights_history = []
    subset_weights_history = []
    
    progress_bar = st.progress(0, text="Initializing backtest...")
    
    if len(historical_data) <= MIN_TRAIN_FILES + 1:
        st.error(f"Not enough data for backtest. Need at least {MIN_TRAIN_FILES + 2} days.")
        return {}

    for i in range(MIN_TRAIN_FILES, len(historical_data) - 1):
        progress = (i - MIN_TRAIN_FILES + 1) / (len(historical_data) - MIN_TRAIN_FILES - 1)
        progress_bar.progress(progress, text=f"Walk-forward step {i - MIN_TRAIN_FILES + 1}/{len(historical_data) - MIN_TRAIN_FILES - 1}")
        
        train_window = historical_data[:i]
        test_date, test_df = historical_data[i]
        next_date, next_df = historical_data[i + 1]

        in_sample_perf = _calculate_performance_on_window(train_window, _strategies, TRAINING_CAPITAL)

        try:
            curated_port, strategy_weights, subset_weights = curate_final_portfolio(
                _strategies, in_sample_perf, test_df, TRAINING_CAPITAL, 30, 1.0, 10.0
            )
            strategy_weights_history.append({'date': test_date, **strategy_weights})
            subset_weights_history.append({'date': test_date, **subset_weights})
        except Exception as e:
            logging.error(f"Curate portfolio error at {test_date}: {e}")
            curated_port = pd.DataFrame()

        if not curated_port.empty:
            curated_ret = compute_portfolio_return(curated_port, next_df)
            all_returns['System_Curated'].append({'return': curated_ret, 'date': next_date})

        for name, strategy in _strategies.items():
            try:
                test_port = strategy.generate_portfolio(test_df, TRAINING_CAPITAL)
                if not test_port.empty:
                    strat_ret = compute_portfolio_return(test_port, next_df)
                    all_returns[name].append({'return': strat_ret, 'date': next_date})
            except Exception as e:
                logging.error(f"Strategy {name} error at {test_date}: {e}")

    progress_bar.empty()
    
    full_history_subset_perf = _calculate_performance_on_window(historical_data, _strategies, TRAINING_CAPITAL)['subset']
    
    return {
        'strategy': {name: {'metrics': calculate_advanced_metrics(rets)[0], 'sharpe': calculate_advanced_metrics(rets)[0]['sharpe']} for name, rets in all_returns.items() if rets},
        'subset': full_history_subset_perf,
        'strategy_weights': strategy_weights_history,
        'subset_weights': subset_weights_history
    }


def curate_final_portfolio(
    strategies: Dict[str, BaseStrategy],
    performance: Dict,
    current_df: pd.DataFrame,
    sip_amount: float,
    num_positions: int,
    min_pos_pct: float,
    max_pos_pct: float
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Curate final portfolio using weighted strategy combination."""
    strategy_weights = calculate_strategy_weights(performance)
    
    subset_weights = {}
    for name in strategies:
        sub_sharpes = performance.get('subset', {}).get(name, {})
        if sub_sharpes:
            max_sharpe = max(sub_sharpes.values()) if sub_sharpes else 0
            min_sharpe = min(sub_sharpes.values()) if sub_sharpes else 0
            range_sharpe = max_sharpe - min_sharpe
            if range_sharpe > 0:
                normalized = {k: (v - min_sharpe) / range_sharpe for k, v in sub_sharpes.items()}
            else:
                normalized = {k: 1.0 for k in sub_sharpes}
            total_norm = sum(normalized.values())
            subset_weights[name] = {k: v / total_norm for k, v in normalized.items()} if total_norm > 0 else {}

    all_portfolios = []
    for name, strategy in strategies.items():
        try:
            strat_weight = strategy_weights.get(name, 1.0 / len(strategies))
            strat_capital = sip_amount * strat_weight
            port = strategy.generate_portfolio(current_df, strat_capital)
            if not port.empty:
                port['source_strategy'] = name
                all_portfolios.append(port)
        except Exception as e:
            logging.error(f"Portfolio generation error ({name}): {e}")

    if not all_portfolios:
        return pd.DataFrame(), strategy_weights, subset_weights

    combined = pd.concat(all_portfolios, ignore_index=True)
    aggregated = combined.groupby('symbol').agg({
        'price': 'first',
        'value': 'sum',
        'units': 'sum',
        'source_strategy': lambda x: ', '.join(sorted(set(x)))
    }).reset_index()

    aggregated['weightage'] = aggregated['value'] / aggregated['value'].sum()
    aggregated = aggregated.sort_values('weightage', ascending=False).head(num_positions)

    # Apply position limits
    min_w = min_pos_pct / 100
    max_w = max_pos_pct / 100
    for _ in range(10):
        aggregated['weightage'] = aggregated['weightage'].clip(lower=min_w, upper=max_w)
        total_w = aggregated['weightage'].sum()
        if total_w > 0:
            aggregated['weightage'] = aggregated['weightage'] / total_w
        if abs(aggregated['weightage'].sum() - 1.0) < 1e-6:
            break

    aggregated['weightage_pct'] = aggregated['weightage'] * 100
    aggregated['units'] = np.floor((sip_amount * aggregated['weightage']) / aggregated['price'])
    aggregated['value'] = aggregated['units'] * aggregated['price']

    return aggregated[['symbol', 'price', 'weightage_pct', 'units', 'value', 'source_strategy']].reset_index(drop=True), strategy_weights, subset_weights


def get_market_mix_suggestion_v3(selected_date: datetime) -> Tuple[str, str, float, Dict]:
    """
    Enhanced market regime detection.
    Returns suggested mix, explanation, confidence, and details.
    """
    try:
        # Fetch minimal data for regime detection
        lookback_days = 60
        fetch_start = selected_date - timedelta(days=int(lookback_days * 1.5) + MAX_INDICATOR_PERIOD)
        
        historical_data = generate_historical_data(SYMBOLS_UNIVERSE[:20], fetch_start, selected_date)
        
        if not historical_data or len(historical_data) < 10:
            return "Bull Market Mix", "Insufficient data for regime detection. Defaulting to Bull.", 0.5, {}
        
        # Get recent price data
        recent_data = historical_data[-20:]
        
        # Calculate market breadth and trend
        up_days = 0
        total_days = 0
        avg_returns = []
        
        for i in range(1, len(recent_data)):
            prev_df = recent_data[i-1][1]
            curr_df = recent_data[i][1]
            
            merged = prev_df.merge(curr_df, on='symbol', suffixes=('_prev', '_curr'))
            if not merged.empty:
                returns = (merged['price_curr'] - merged['price_prev']) / merged['price_prev']
                avg_return = returns.mean()
                avg_returns.append(avg_return)
                if avg_return > 0:
                    up_days += 1
                total_days += 1
        
        if total_days == 0:
            return "Chop/Consolidate Mix", "No valid trading data.", 0.5, {}
        
        breadth = up_days / total_days
        trend = np.mean(avg_returns) if avg_returns else 0
        volatility = np.std(avg_returns) if len(avg_returns) > 1 else 0
        
        # Determine regime
        if breadth > 0.6 and trend > 0.001:
            regime = "Bull Market Mix"
            confidence = min(0.9, 0.5 + breadth * 0.5)
            explanation = f"Strong upward momentum with {breadth:.0%} positive days."
        elif breadth < 0.4 or trend < -0.001:
            regime = "Bear Market Mix"
            confidence = min(0.9, 0.5 + (1 - breadth) * 0.5)
            explanation = f"Negative trend with {breadth:.0%} positive days."
        else:
            regime = "Chop/Consolidate Mix"
            confidence = 0.6
            explanation = f"Mixed signals with {breadth:.0%} positive days. Consolidation likely."
        
        details = {
            'breadth': breadth,
            'trend': trend,
            'volatility': volatility,
            'up_days': up_days,
            'total_days': total_days
        }
        
        return regime, explanation, confidence, details
        
    except Exception as e:
        logging.error(f"Regime detection error: {e}")
        return "Bull Market Mix", f"Regime detection failed: {e}. Using default.", 0.5, {}


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_strategy_leaderboard():
    """Display the strategy selection leaderboard."""
    if st.session_state.strategy_leaderboard is None:
        return
    
    st.subheader("📊 Strategy Selection Leaderboard")
    
    tab1, tab2 = st.tabs(["SIP (Calmar)", "Swing (Sortino)"])
    
    with tab1:
        sip_df = st.session_state.strategy_leaderboard.get('sip')
        if sip_df is not None and not sip_df.empty:
            st.markdown("*Strategies ranked by Calmar Ratio (best drawdown recovery)*")
            
            # Format the display
            display_df = sip_df.copy()
            format_dict = {
                'Total Return': '{:.2%}',
                'CAGR': '{:.2%}',
                'Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'Sortino Ratio': '{:.2f}',
                'Max Drawdown': '{:.2%}',
                'Calmar Ratio': '{:.2f}',
                'Win Rate': '{:.2%}'
            }
            
            st.dataframe(
                display_df.style.format(format_dict).background_gradient(
                    subset=['Calmar Ratio'], cmap='RdYlGn'
                ),
                use_container_width=True
            )
    
    with tab2:
        swing_df = st.session_state.strategy_leaderboard.get('swing')
        if swing_df is not None and not swing_df.empty:
            st.markdown("*Strategies ranked by Sortino Ratio (best risk-adjusted returns)*")
            
            display_df = swing_df.copy()
            st.dataframe(
                display_df.style.format(format_dict).background_gradient(
                    subset=['Sortino Ratio'], cmap='RdYlGn'
                ),
                use_container_width=True
            )


def display_dynamic_strategy_info(portfolio_styles: Dict, selected_style: str, selected_mix: str):
    """Display information about dynamically selected strategies."""
    if portfolio_styles is None:
        return
    
    style_data = portfolio_styles.get(selected_style, {})
    mix_data = style_data.get('mixes', {}).get(selected_mix, {})
    
    strategies = mix_data.get('strategies', [])
    rationale = mix_data.get('rationale', '')
    
    st.markdown(f"""
    <div class="dynamic-selection-indicator">
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">
            🎯 DYNAMICALLY SELECTED STRATEGIES
        </div>
        <div style="margin: 8px 0;">
            {''.join([f'<span class="strategy-badge">{s}</span>' for s in strategies])}
        </div>
        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 8px;">
            {rationale}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Initialize all strategies
    strategies = {
        'VolatilitySurfer': VolatilitySurfer(),
        'GameTheoreticStrategy': GameTheoreticStrategy(),
        'CelestialAlphaForge': CelestialAlphaForge(),
        'MomentumAccelerator': MomentumAccelerator(),
        'NebulaMomentumStorm': NebulaMomentumStorm(),
        'AdaptiveVolBreakout': AdaptiveVolBreakout(),
        'DivineMomentumOracle': DivineMomentumOracle(),
    }
    
    def update_regime_suggestion():
        """Update regime suggestion when date changes."""
        selected_date_obj = st.session_state.get('analysis_date_str')
        if not selected_date_obj:
            return
        
        selected_date = datetime.combine(selected_date_obj, datetime.min.time())
        st.toast(f"Analyzing market regime for {selected_date.date()}...", icon="🧠")
        
        mix_name, explanation, confidence, details = get_market_mix_suggestion_v3(selected_date)
        
        st.session_state.suggested_mix = mix_name
        st.session_state.regime_display = {
            'mix': mix_name,
            'confidence': confidence,
            'explanation': explanation
        }

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">PRAGYAM</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">प्रज्ञम | Portfolio Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # --- Dynamic Strategy Selection Toggle ---
        st.markdown('<div class="sidebar-title">🧠 Strategy Selection Mode</div>', unsafe_allow_html=True)
        
        use_dynamic = st.toggle(
            "Dynamic Selection",
            value=BACKTEST_ENGINE_AVAILABLE,
            help="Enable to automatically select best strategies based on backtest performance",
            disabled=not BACKTEST_ENGINE_AVAILABLE
        )
        
        if use_dynamic and BACKTEST_ENGINE_AVAILABLE:
            st.markdown("""
            <div class="info-box" style="border-left: 3px solid var(--success-green);">
                <div style="font-size: 0.75rem; color: var(--success-green);">✓ DYNAMIC MODE</div>
                <div style="font-size: 0.7rem; color: var(--text-muted);">
                    SIP: Top 4 by Calmar<br>
                    Swing: Top 4 by Sortino
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="border-left: 3px solid var(--warning-amber);">
                <div style="font-size: 0.75rem; color: var(--warning-amber);">⚠ STATIC MODE</div>
                <div style="font-size: 0.7rem; color: var(--text-muted);">
                    Using predefined strategy selections
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # --- Date Selection ---
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
        
        # Trigger initial calculation if needed
        if st.session_state.suggested_mix is None:
            update_regime_suggestion()
        
        # Display regime info
        if st.session_state.regime_display:
            data = st.session_state.regime_display
            st.markdown(f"""
            <div style="background-color: var(--secondary-background-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; margin: 10px 0 20px 0;">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 4px;">Market Regime</div>
                <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 700; line-height: 1.2;">{data['mix']}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                    <span style="color: var(--text-muted); font-size: 0.8rem;">Confidence</span>
                    <span style="color: var(--primary-color); font-weight: 600; font-size: 0.8rem;">{data['confidence']:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # --- Portfolio Style Selection ---
        st.markdown('<div class="sidebar-title">💼 Portfolio Style</div>', unsafe_allow_html=True)
        
        PORTFOLIO_STYLES = get_active_portfolio_styles()
        
        options_list = list(PORTFOLIO_STYLES.keys())
        default_index = 0
        if "SIP Investment" in options_list:
            default_index = options_list.index("SIP Investment")
        
        selected_main_branch = st.selectbox(
            "1. Select Investment Style",
            options=options_list,
            index=default_index,
            help="Choose your primary investment objective."
        )
        
        # --- Parameters ---
        st.markdown('<div class="sidebar-title">⚙️ Portfolio Parameters</div>', unsafe_allow_html=True)
        capital = st.number_input("Capital (₹)", 1000, 100000000, 2500000, 1000, help="Total capital to allocate")
        num_positions = st.slider("Number of Positions", 5, 100, 30, 5, help="Maximum positions in final portfolio")
        
        # --- Run Button ---
        if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
            
            lookback_files = 25
            selected_date_obj = st.session_state.get('analysis_date_str')
            
            if not selected_date_obj:
                st.error("Analysis date is missing. Please select a date.")
                st.stop()
            
            selected_date_dt = datetime.combine(selected_date_obj, datetime.min.time())
            
            # --- DYNAMIC STRATEGY SELECTION ---
            if use_dynamic and BACKTEST_ENGINE_AVAILABLE and st.session_state.dynamic_portfolio_styles is None:
                st.toast("Running dynamic strategy selection...", icon="🧠")
                
                # Run backtest for last 365 days
                backtest_start = selected_date_dt - timedelta(days=365)
                
                dynamic_styles = run_dynamic_strategy_selection(
                    symbols=SYMBOLS_UNIVERSE,
                    start_date=backtest_start,
                    end_date=selected_date_dt,
                    capital=float(capital),
                    n_strategies=4
                )
                
                if dynamic_styles:
                    st.session_state.dynamic_portfolio_styles = dynamic_styles
                    st.session_state.backtest_complete = True
                    st.toast("Dynamic strategies selected!", icon="✅")
                    PORTFOLIO_STYLES = dynamic_styles
            
            # --- LOAD HISTORICAL DATA ---
            total_days_to_fetch = int((lookback_files + MAX_INDICATOR_PERIOD) * 1.5) + 30
            fetch_start_date = selected_date_dt - timedelta(days=total_days_to_fetch)
            st.toast(f"Fetching data for {len(SYMBOLS_UNIVERSE)} symbols...", icon="⏳")
            
            all_historical_data = load_historical_data(selected_date_dt, lookback_files)
            
            if not all_historical_data:
                st.error("No historical data could be loaded.")
                st.stop()
            
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
                st.error(f"Not enough training data ({len(training_data_window_with_current)} days). Need at least 10.")
                st.stop()
            
            if not st.session_state.suggested_mix:
                st.error("Market regime could not be determined.")
                st.stop()
            
            final_mix_to_use = st.session_state.suggested_mix
            
            # Get strategies for the selected style and mix
            PORTFOLIO_STYLES = get_active_portfolio_styles()
            style_strategies = PORTFOLIO_STYLES[selected_main_branch]["mixes"][final_mix_to_use]['strategies']
            
            # Filter to only available strategies
            available_strategies = {name: strategies[name] for name in style_strategies if name in strategies}
            
            if not available_strategies:
                st.error(f"None of the selected strategies are available: {style_strategies}")
                st.stop()
            
            strategies_to_run = available_strategies
            
            # Run evaluation
            st.session_state.performance = evaluate_historical_performance(
                strategies_to_run, training_data_window_with_current
            )
            
            if st.session_state.performance:
                st.session_state.portfolio, _, _ = curate_final_portfolio(
                    strategies_to_run,
                    st.session_state.performance,
                    st.session_state.current_df,
                    capital,
                    num_positions,
                    st.session_state.min_pos_pct,
                    st.session_state.max_pos_pct
                )
                st.toast("Analysis Complete!", icon="✅")
        
        # --- Footer ---
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Walk-Forward + Dynamic Selection<br>
                <strong>Data:</strong> Live Generated
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- MAIN CONTENT AREA ---
    if st.session_state.portfolio is None or st.session_state.performance is None:
        # Landing page
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
                <span style="color: #FFC300;">PRAGYAM</span>
            </h1>
            <p style="color: #888888; font-size: 1.2rem; margin-bottom: 2rem;">
                प्रज्ञम | Institutional Portfolio Intelligence
            </p>
            <div style="max-width: 600px; margin: 0 auto; text-align: left;">
                <div class="metric-card" style="margin-bottom: 1rem;">
                    <h3 style="color: #FFC300; margin-bottom: 0.5rem;">🧠 Dynamic Strategy Selection</h3>
                    <p style="color: #888888; font-size: 0.9rem;">
                        Strategies are now selected automatically based on backtest performance.
                        SIP mode uses Calmar Ratio, Swing mode uses Sortino Ratio.
                    </p>
                </div>
                <div class="metric-card" style="margin-bottom: 1rem;">
                    <h3 style="color: #FFC300; margin-bottom: 0.5rem;">📊 Walk-Forward Analysis</h3>
                    <p style="color: #888888; font-size: 0.9rem;">
                        Out-of-sample testing with rolling windows for robust performance estimation.
                    </p>
                </div>
                <div class="metric-card">
                    <h3 style="color: #FFC300; margin-bottom: 0.5rem;">⚡ Live Data Integration</h3>
                    <p style="color: #888888; font-size: 0.9rem;">
                        Real-time data fetching with intelligent caching and resource optimization.
                    </p>
                </div>
            </div>
            <p style="color: #666666; font-size: 0.8rem; margin-top: 3rem;">
                Click the menu (☰) to configure and run analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Results display
        st.title(f"📈 Portfolio Analysis - {st.session_state.selected_date}")
        
        # Show dynamic strategy info if available
        if st.session_state.dynamic_portfolio_styles:
            PORTFOLIO_STYLES = st.session_state.dynamic_portfolio_styles
            selected_style = list(PORTFOLIO_STYLES.keys())[0]  # Get current selection
            display_dynamic_strategy_info(
                PORTFOLIO_STYLES,
                selected_style,
                st.session_state.suggested_mix
            )
        
        # Show strategy leaderboard
        if st.session_state.strategy_leaderboard:
            with st.expander("📊 Strategy Selection Leaderboard", expanded=False):
                display_strategy_leaderboard()
        
        # Performance metrics
        if st.session_state.performance:
            perf = st.session_state.performance
            curated_metrics = perf.get('strategy', {}).get('System_Curated', {}).get('metrics', {})
            
            st.subheader("📊 Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{curated_metrics.get('total_return', 0):.2%}")
            col2.metric("Sharpe Ratio", f"{curated_metrics.get('sharpe', 0):.2f}")
            col3.metric("Sortino Ratio", f"{curated_metrics.get('sortino', 0):.2f}")
            col4.metric("Max Drawdown", f"{curated_metrics.get('max_drawdown', 0):.2%}")
        
        # Portfolio display
        if st.session_state.portfolio is not None and not st.session_state.portfolio.empty:
            st.subheader("📋 Curated Portfolio")
            
            portfolio_df = st.session_state.portfolio.copy()
            portfolio_df['weightage_pct'] = portfolio_df['weightage_pct'].apply(lambda x: f"{x:.2f}%")
            portfolio_df['price'] = portfolio_df['price'].apply(lambda x: f"₹{x:,.2f}")
            portfolio_df['value'] = portfolio_df['value'].apply(lambda x: f"₹{x:,.2f}")
            
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Download button
            csv = fix_csv_export(st.session_state.portfolio)
            st.download_button(
                label="⬇️ Download Portfolio (CSV)",
                data=csv,
                file_name=f"pragyam_portfolio_{st.session_state.selected_date}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
