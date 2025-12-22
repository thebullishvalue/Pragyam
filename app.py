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
from typing import List, Dict, Tuple
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


# --- System Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('system.log'), logging.StreamHandler()])
st.set_page_config(page_title="Pragyam : Quantitative Portfolio Curation System", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
VERSION = "v1.1.0 - Curation Engine"

# --- CSS Styling ---
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
    
    .block-container {
        padding-top: 1rem;
        max-width: 1400px;
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
        margin-top: 2.5rem;
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
        font-size: 2.50rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.50px;
        position: relative;
    }
    
    .premium-header .tagline {
        color: var(--text-muted);
        font-size: 1rem;
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
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h2 {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    
    .metric-card .sub-metric {
        font-size: 0.8rem;
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
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted);
        border-bottom: 2px solid transparent;
        transition: color 0.3s, border-bottom 0.3s;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-color);
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
        margin: 1rem 0;
    }
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
    default_metrics = {'total_return': 0, 'annual_return': 0, 'volatility': 0, 'sharpe': 0, 'sortino': 0, 'max_drawdown': 0, 'calmar': 0, 'win_rate': 0, 'kelly_criterion': 0}
    if len(returns_with_dates) < 2: return default_metrics, 52

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

    metrics = {'total_return': total_return, 'annual_return': annual_return, 'volatility': volatility, 'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_drawdown, 'calmar': calmar, 'win_rate': win_rate, 'kelly_criterion': kelly}
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
        avg_std = (df['rsi latest'].std() / 100 + df['osc latest'].std() / 100 + df['zscore latest'].std() / 5) / 3
        
        if avg_std < 0.15:
            regime, score = 'HIGH_CORRELATION', -0.5 
        elif avg_std > 0.30:
            regime, score = 'LOW_CORRELATION', 0.5 
        else:
            regime, score = 'NORMAL', 0.0
            
        return {'regime': regime, 'score': score, 'dispersion': avg_std}

    def _analyze_velocity(self, window: list) -> Dict:
        if len(window) < 5: return {'acceleration': 'UNKNOWN', 'score': 0.0}
        
        recent_rsis = [w[1]['rsi latest'].mean() for w in window[-5:]]
        rsi_changes = np.diff(recent_rsis)
        avg_velocity = np.mean(rsi_changes)
        acceleration = rsi_changes[-1] - rsi_changes[0]
        
        if avg_velocity > 2 and acceleration > 0:
            velocity, score = 'ACCELERATING_UP', 1.0
        elif avg_velocity > 1:
            velocity, score = 'RISING', 0.5
        elif avg_velocity < -2 and acceleration < 0:
            velocity, score = 'ACCELERATING_DOWN', -1.0
        elif avg_velocity < -1:
            velocity, score = 'FALLING', -0.5
        else:
            velocity, score = 'STABLE', 0.0
            
        return {'acceleration': velocity, 'score': score, 'avg_velocity': avg_velocity}

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
    if not performance:
        st.warning("Performance data not available. Please run an analysis.")
        return

    st.header("Out-of-Sample Performance Analysis")

    st.subheader("Cumulative Performance")
    curated_metrics = performance.get('strategy', {}).get('System_Curated', {}).get('metrics', {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Annual Return", f"{curated_metrics.get('annual_return', 0):.2%}", help="The geometric average return on an annualized basis. Higher is better.")
    col2.metric("Total Return", f"{curated_metrics.get('total_return', 0):.2%}", help="The total compounded return over the entire backtest period. Higher is better.")
    col3.metric("Sharpe Ratio", f"{curated_metrics.get('sharpe', 0):.2f}", help="Measures return per unit of total risk (volatility). Good > 1, Excellent > 2.")
    col4.metric("Calmar Ratio", f"{curated_metrics.get('calmar', 0):.2f}", help="Annualized return divided by the max drawdown. Good > 1, Excellent > 3.")
    col5.metric("Win Rate", f"{curated_metrics.get('win_rate', 0):.2%}", help="The percentage of periods with a positive return. Good > 50%.")

    st.subheader("Risk & System Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Volatility", f"{curated_metrics.get('volatility', 0):.2%}", help="The annualized standard deviation of returns (risk). Lower is better.")
    col2.metric("Max Drawdown", f"{curated_metrics.get('max_drawdown', 0):.2%}", help="The largest peak-to-trough decline in value. Closer to 0 is better.")
    col3.metric("Sortino Ratio", f"{curated_metrics.get('sortino', 0):.2f}", help="Measures return per unit of downside risk. Good > 1, Excellent > 2.")
    col4.metric("Kelly Criterion", f"{curated_metrics.get('kelly_criterion', 0):.2%}", help="Theoretical optimal fraction of capital to allocate. Prone to estimation error; use with caution.")
    
    avg_entropy = curated_metrics.get('avg_weight_entropy')
    if avg_entropy is not None:
        st.metric("Average Weight Entropy", f"{avg_entropy:.3f}", help="Measures portfolio diversification. Higher values indicate a more diversified, less concentrated portfolio.")

    st.markdown("---")

    st.subheader("Growth of Investment (Equity Curve)")
    equity_curves = []
    for name, perf in performance.get('strategy', {}).items():
        if perf.get('returns'):
            df = pd.DataFrame(perf['returns']).sort_values('date')
            df['equity'] = (1 + df['return']).cumprod()
            df['strategy'] = name
            equity_curves.append(df)

    if equity_curves:
        full_equity_df = pd.concat(equity_curves)
        fig_equity = px.line(full_equity_df, x='date', y='equity', color='strategy',
                             title="Growth of ₹1 Investment Over Time",
                             labels={'equity': 'Growth of ₹1', 'date': 'Date', 'strategy': 'Strategy'})
        fig_equity.update_layout(template='plotly_dark', legend_title_text='Strategy')
        st.plotly_chart(fig_equity, width='stretch')

    plot_weight_evolution(
        performance.get('strategy_weights_history', []),
        title="Strategy Weight Evolution Over Time",
        y_axis_title="Strategy Weight"
    )

    st.subheader("Rolling Sharpe Ratio (3-Period Window)")
    df_list = []
    for name, perf in performance.get('strategy', {}).items():
        if perf.get('returns'):
            df = pd.DataFrame(perf['returns']).sort_values('date')
            rolling_mean = df['return'].rolling(window=3).mean()
            rolling_std = df['return'].rolling(window=3).std()
            periods_per_year = 52 
            df['rolling_sharpe'] = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(periods_per_year)
            df['strategy'] = name
            df_list.append(df)

    if df_list:
        full_df = pd.concat(df_list)
        fig_sharpe = px.line(full_df, x='date', y='rolling_sharpe', color='strategy', title="Strategy Rolling Sharpe Ratio (3-Period)")
        fig_sharpe.update_layout(template='plotly_dark', legend_title_text='Strategy', yaxis_title="Sharpe Ratio")
        st.plotly_chart(fig_sharpe, width='stretch')

    st.subheader("Strategy Correlation Matrix")
    returns_df = pd.DataFrame()
    for name, perf in performance.get('strategy', {}).items():
        if perf.get('returns'):
            df_raw = pd.DataFrame(perf['returns'])
            df = df_raw.drop_duplicates(subset='date', keep='last').set_index('date')
            if len(df_raw) > len(df):
                logging.warning(f"Removed {len(df_raw) - len(df)} duplicate date entries for strategy '{name}'.")
            returns_df[name] = df['return']

    corr_matrix = returns_df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Portland', aspect="auto")
    fig_corr.update_layout(title="Correlation of Strategy Returns", template='plotly_dark')
    st.plotly_chart(fig_corr, width='stretch')


def create_subset_heatmap(subset_perf: Dict, strategy_options: list):
    if not subset_perf: return

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

# --- Main Application ---
def main():
    strategies = {
        'VolatilitySurfer': VolatilitySurfer(),
        'GameTheoreticStrategy': GameTheoreticStrategy(),
        'CelestialAlphaForge': CelestialAlphaForge(),
        'MomentumAccelerator': MomentumAccelerator(),
        'NebulaMomentumStorm': NebulaMomentumStorm(),
        'AdaptiveVolBreakout': AdaptiveVolBreakout(),
        'DivineMomentumOracle': DivineMomentumOracle(),
    }

    PORTFOLIO_STYLES = {
        "Swing Trading": {
            "description": "Short-term (3-21 day) holds to capture rapid momentum and volatility.",
            "mixes": {
                "Bull Market Mix": {
                    "strategies": ['GameTheoreticStrategy', 'NebulaMomentumStorm', 'VolatilitySurfer', 'CelestialAlphaForge'],
                    "rationale": "Active when Breadth < 0.42. Selected based on highest Calmar Ratio (Top: GameTheoretic @ 5.11). These strategies recover fastest from the deep drawdowns typical of this entry zone."
                },
                
                "Bear Market Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Active when Breadth > 0.52. Selected based on highest Sortino Ratio. These strategies maximize upside volatility (momentum) during stable market trends."
                },
                
                "Chop/Consolidate Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Active during transition (0.42-0.52). Selected based on highest Sortino Ratio (Top: GameTheoretic @ 2.73) to ensure consistent returns without excessive volatility."
                }
            }
        },
        
        "SIP Investment": {
            "description": "Systematic long-term (3-12+ months) wealth accumulation. Focus on consistency and drawdown protection.",
            "mixes": {
                "Bull Market Mix": {
                    "strategies": ['GameTheoreticStrategy', 'MomentumAccelerator', 'VolatilitySurfer', 'DivineMomentumOracle'],
                    "rationale": "Selected based on highest Sharpe Ratio to ensure steady compounding and maximize risk-adjusted returns."
                },
                
                "Bear Market Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Selected based on highest Sortino Ratio. In expensive markets, these strategies provide the best defense by minimizing volatility per unit of return."
                },
                
                "Chop/Consolidate Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Selected based on highest Sortino Ratio for consistency."
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
        st.markdown("# Configuration")
        st.markdown("### Analysis Configuration")
        
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
        
        st.markdown("### Portfolio Style Selection")

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
        
        st.markdown("### Portfolio Parameters")
        capital = st.number_input("Capital (₹)", 1000, 100000000, 2500000, 1000, help="Total capital to allocate")
        num_positions = st.slider("Number of Positions", 5, 100, 30, 5, help="Maximum positions in the final portfolio")

        if st.button("Run Analysis", width='stretch', type="primary"):
            
            lookback_files = 25
            
            selected_date_obj = st.session_state.get('analysis_date_str')
            if not selected_date_obj:
                st.error("Analysis date is missing. Please select a date.")
                st.stop()
                
            selected_date_dt = datetime.combine(selected_date_obj, datetime.min.time())

            total_days_to_fetch = int((lookback_files + MAX_INDICATOR_PERIOD) * 1.5) + 30
            fetch_start_date = selected_date_dt - timedelta(days=total_days_to_fetch)
            toast_msg = f"Fetching live data for {len(SYMBOLS_UNIVERSE)} symbols from {fetch_start_date.date()} to {selected_date_dt.date()}..."
            st.toast(toast_msg, icon="⏳")

            all_historical_data = load_historical_data(selected_date_dt, lookback_files)
            
            if not all_historical_data:
                st.error("Application cannot start: No historical data could be loaded or generated.")
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
                st.error(f"Not enough training data loaded ({len(training_data_window_with_current)} days). Need at least 10. Check date range or lookback period.")
                st.stop()
                
            if not st.session_state.suggested_mix:
                st.error("Market regime could not be determined. Please select a valid date. Analysis cannot run.")
                st.stop()
                
            final_mix_to_use = st.session_state.suggested_mix 
            
            style_strategies = PORTFOLIO_STYLES[selected_main_branch]["mixes"][final_mix_to_use]['strategies']
            strategies_to_run = {name: strategies[name] for name in style_strategies}

            st.session_state.performance = evaluate_historical_performance(strategies_to_run, training_data_window_with_current)
            
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
                st.success("✅ Analysis Complete!")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Platform Info")
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.85rem; margin: 0; color: var(--text-muted); line-height: 1.6;'>
                <strong>Version:</strong> v1.0.0 - Pragyam<br>
                <strong>Engine:</strong> Walk-Forward Curation<br> 
                <strong>Data:</strong> Live Generated
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="premium-header">
        <h1>Pragyam : Quantitative Portfolio Curation System</h1>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.portfolio is None or st.session_state.performance is None:
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
            st.header("Strategy & Subset Analysis")
            strategies_in_performance = [k for k in st.session_state.performance.get('strategy', {}).keys() if k != 'System_Curated']

            create_subset_heatmap(st.session_state.performance.get('subset'), strategies_in_performance)
            
            display_subset_weight_evolution(
                st.session_state.performance.get('subset_weights_history', []),
                strategies_in_performance
            )

            st.markdown("---")
            st.subheader(f"Conviction Heatmap (for {st.session_state.selected_date})")
            strategies_for_heatmap = {name: strategies[name] for name in strategies_in_performance}
            heatmap_fig = create_conviction_heatmap(strategies_for_heatmap, st.session_state.current_df)
            st.plotly_chart(heatmap_fig, width='stretch')

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© {datetime.now().year} Pragyam | Hemrek Capital | {VERSION} | Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S IST')}")

if __name__ == "__main__":
    main()
