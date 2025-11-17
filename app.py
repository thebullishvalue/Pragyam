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
from scipy import stats
from sklearn.preprocessing import StandardScaler
import time # Import time for toast
import warnings  # Add this import

# --- Suppress known NumPy warnings during backtest warm-up ---
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
# --- End suppression ---


# --- Import Strategies from strategies.py ---
# Ensure strategies.py is in the same directory or accessible in the path
try:
    from strategies import (
        BaseStrategy, PRStrategy, CL1Strategy, CL2Strategy, CL3Strategy, 
        MOM1Strategy, MOM2Strategy, MomentumMasters, VolatilitySurfer
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
st.set_page_config(page_title="Pragati : Quantitative Portfolio Curation System", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

# "Glowy Matte" Gold/Yellow Inspired Dashboard CSS
st.markdown("""
<style>
    :root {
        --primary-color: #FFC300; /* Vibrant Gold/Yellow */
        --background-color: #0F0F0F; /* Near Black */
        --secondary-background-color: #1A1A1A; /* Charcoal */
        --text-color: #EAEAEA; /* Light Grey */
        --text-color-darker: #888888; /* Grey */
        --primary-rgb: 255, 195, 0; /* RGB for Gold/Yellow */
    }
    .main, [data-testid="stSidebar"] {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stApp > header { background-color: transparent; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color-darker);
        border-bottom: 2px solid transparent;
        transition: color 0.3s, border-bottom 0.3s;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-color);
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.2);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
        text-shadow: 0 0 10px rgba(var(--primary-rgb), 0.4);
    }
    h2 {
        border-bottom: 2px solid var(--secondary-background-color);
        padding-bottom: 10px;
    }
    .stPlotlyChart, .stDataFrame {
        border-radius: 8px;
        background-color: var(--secondary-background-color);
        padding: 10px;
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.15);
    }
    .stButton>button {
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6);
        background: var(--primary-color);
        color: #1A1A1A; /* Dark text on hover for contrast */
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if 'performance' not in st.session_state: st.session_state.performance = None
if 'portfolio' not in st.session_state: st.session_state.portfolio = None
if 'current_df' not in st.session_state: st.session_state.current_df = None
if 'selected_date' not in st.session_state: st.session_state.selected_date = None
if 'suggested_mix' not in st.session_state: st.session_state.suggested_mix = None # For auto-selection
# Set fixed position size limits
if 'min_pos_pct' not in st.session_state: st.session_state.min_pos_pct = 1.0
if 'max_pos_pct' not in st.session_state: st.session_state.max_pos_pct = 10.0

# --- Base Classes and Utilities ---
def fix_csv_export(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()

# =========================================================================
# --- Live Data Loading Function ---
@st.cache_data(ttl=3600, show_spinner=False) # Cache live data for 1 hour to speed up re-runs
def load_historical_data(end_date: datetime, lookback_files: int) -> List[Tuple[datetime, pd.DataFrame]]:
    """
    Fetches and processes all historical data on-the-fly using the
    backdata.py module. This is the main data engine for the app.
    
    Args:
        end_date: The final "analysis date" selected by the user.
        lookback_files: The number of trading days *prior* to end_date to use for training.
        
    Returns:
        A list of tuples in the format Pragati expects: [(date, DataFrame), ...]
        Or an empty list if data fetching fails.
    """
    logging.info(f"--- START: Live Data Generation (End Date: {end_date.date()}, Training Lookback: {lookback_files}) ---")
    
    # We need data for the training period + enough data for the indicators (e.g., 200-day MA)
    # 1.5x buffer for non-trading days, + 30 days extra buffer
    total_days_to_fetch = int((lookback_files + MAX_INDICATOR_PERIOD) * 1.5) + 30
    fetch_start_date = end_date - timedelta(days=total_days_to_fetch)
    
    logging.info(f"Calculated fetch start date: {fetch_start_date.date()} (Total days: {total_days_to_fetch})")

    # Use a toast for the loading message
    # toast_msg = f"Fetching live data for {len(SYMBOLS_UNIVERSE)} symbols from {fetch_start_date.date()} to {end_date.date()}..."
    # st.toast(toast_msg, icon="⏳") # <-- MOVED out of cached function
    
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
    TRAINING_CAPITAL = 2500000.0  # Set to a robust value for the given universe
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
    
    # Handle the edge case where there aren't enough steps to loop
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
            
            # Store weight history
            strategy_weights_history.append({'date': test_date, **strategy_weights})
            subset_weights_history.append({'date': test_date, **subset_weights})

            # --- FIX: Check for empty portfolio before accessing columns ---
            if curated_port.empty:
                logging.warning(f"  - No curated portfolio generated for {test_date.date()}. Appending 0 return.")
                oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
            else:
                oos_perf['System_Curated']['returns'].append({'return': compute_portfolio_return(curated_port, next_df), 'date': next_date})
                logging.info(f"  - COMPLETED: Curating out-of-sample portfolio for {test_date.date()}")

                weights = curated_port['weightage_pct'] / 100
                entropy = -np.sum(weights * np.log2(weights))
                weight_entropies.append(entropy)
            # --- END FIX ---

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

        # Stabilize the exp calculation to prevent overflow
        stable_sharpes = tier_sharpes - np.max(tier_sharpes)
        exp_sharpes = np.exp(stable_sharpes)
        total_exp = np.sum(exp_sharpes)

        if total_exp > 0 and np.isfinite(total_exp):
            subset_weights[name] = {tier: exp_sharpes[i] / total_exp for i, tier in enumerate(tier_names)}
        else: # Fallback to equal weighting
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
        return pd.DataFrame(), {}, {} # Return empty objects if no holdings
        
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
        # --- GRANULARITY V3: Increased sensitivity to reduce "CHOP" classification. ---
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
        """Track momentum evolution over time"""
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
        """Assess trend quality and persistence"""
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
        """Analyze participation and breadth indicators"""
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
        """Classify volatility regime and its implications"""
        bb_widths = [((4 * df['dev20 latest']) / (df['ma20 latest'] + 1e-6)).mean() for _, df in window]
        current_bbw = bb_widths[-1]
        vol_trend = np.polyfit(range(len(bb_widths)), bb_widths, 1)[0]
        
        # --- GRANULARITY V3: Panic is now a bearish signal ---
        if current_bbw < 0.08 and vol_trend < 0:
            regime, score = 'SQUEEZE', 0.5  # Slightly positive, anticipates breakout
        elif current_bbw > 0.15 and vol_trend > 0:
            regime, score = 'PANIC', -1.0  # Bearish signal
        elif current_bbw > 0.12:
            regime, score = 'ELEVATED', -0.5
        else:
            regime, score = 'NORMAL', 0.0
            
        return {'regime': regime, 'score': score, 'current_bbw': current_bbw, 'vol_trend': vol_trend}

    def _analyze_statistical_extremes(self, df: pd.DataFrame) -> Dict:
        """Detect statistical extreme conditions"""
        extreme_oversold = (df['zscore latest'] < -2.0).mean()
        extreme_overbought = (df['zscore latest'] > 2.0).mean()
        
        # --- FIX: Inverted scoring logic ---
        if extreme_oversold > 0.40:
            extreme_type, score = 'DEEPLY_OVERSOLD', 1.5  # Contrarian BULLISH signal
        elif extreme_overbought > 0.40:
            extreme_type, score = 'DEEPLY_OVERBOUGHT', -1.5 # Contrarian BEARISH signal
        elif extreme_oversold > 0.20:
            extreme_type, score = 'OVERSOLD', 0.75
        elif extreme_overbought > 0.20:
            extreme_type, score = 'OVERBOUGHT', -0.75
        else:
            extreme_type, score = 'NORMAL', 0.0
            
        return {'type': extreme_type, 'score': score, 'zscore_extreme_oversold_pct': extreme_oversold, 'zscore_extreme_overbought_pct': extreme_overbought}

    def _analyze_correlation_regime(self, df: pd.DataFrame) -> Dict:
        """Analyze cross-sectional correlation (systemic risk)"""
        avg_std = (df['rsi latest'].std() / 100 + df['osc latest'].std() / 100 + df['zscore latest'].std() / 5) / 3
        
        if avg_std < 0.15:
            regime, score = 'HIGH_CORRELATION', -0.5  # Risk-off
        elif avg_std > 0.30:
            regime, score = 'LOW_CORRELATION', 0.5   # Risk-on, stock picking
        else:
            regime, score = 'NORMAL', 0.0
            
        return {'regime': regime, 'score': score, 'dispersion': avg_std}

    def _analyze_velocity(self, window: list) -> Dict:
        """Measure rate of change in key metrics"""
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
        # --- GRANULARITY V3: Increased weight on momentum and velocity, removed correlation ---
        weights = { 'momentum': 0.30, 'trend': 0.25, 'breadth': 0.15, 'volatility': 0.05, 'extremes': 0.10, 'correlation': 0.0, 'velocity': 0.15 }
        return sum(metrics[factor]['score'] * weight for factor, weight in weights.items())
    
    def _classify_regime(self, score: float, metrics: Dict) -> Tuple[str, float]:
        """Map composite score to regime with confidence"""
        if metrics['volatility']['regime'] == 'PANIC' and score < -0.5 and metrics['breadth']['quality'] == 'CAPITULATION':
            return 'CRISIS', 0.90
            
        # --- FIX: Iterate from most bearish to most bullish to prevent incorrect classification ---
        sorted_thresholds = sorted(self.regime_thresholds.items(), key=lambda item: item[1]['score'])
        
        for regime, threshold in reversed(sorted_thresholds):
            if score >= threshold['score']:
                # Downgrade confidence if breadth is divergent
                confidence = threshold['confidence'] * 0.75 if metrics['breadth']['quality'] == 'DIVERGENT' else threshold['confidence']
                return regime, confidence

        return 'CRISIS', 0.85 # Fallback
    
    def _map_regime_to_mix(self, regime: str) -> str:
        mapping = {
            'STRONG_BULL': '🐂 Bull Market Mix', 'BULL': '🐂 Bull Market Mix',
            'WEAK_BULL': '📊 Chop/Consolidate Mix', 'CHOP': '📊 Chop/Consolidate Mix',
            'WEAK_BEAR': '📊 Chop/Consolidate Mix', 'BEAR': '🐻 Bear Market Mix',
            'CRISIS': '🐻 Bear Market Mix'
        }
        return mapping.get(regime, '📊 Chop/Consolidate Mix')
    
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

@st.cache_data(ttl=3600, show_spinner=False) # Cache regime detection for 1 hour
def get_market_mix_suggestion_v3(end_date: datetime) -> Tuple[str, str, float, Dict]:
    """
    Fetches *just enough* data to run the regime detection model.
    This is designed to be a lightweight call.
    """
    detector = MarketRegimeDetectorV2()
    
    # Fetch data for regime detection: 200-day MA + 10-day lookback
    # (200 * 1.5) + 30 buffer = 330 days
    regime_days_to_fetch = int(MAX_INDICATOR_PERIOD * 1.5) + 30 
    fetch_start_date = end_date - timedelta(days=regime_days_to_fetch)
    
    # Use a toast for the loading message
    # toast_msg = f"Fetching regime data for {end_date.date()}..."
    # st.toast(toast_msg, icon="🧠") # <-- MOVED out of cached function
    
    try:
        historical_data = generate_historical_data(
            symbols_to_process=SYMBOLS_UNIVERSE,
            start_date=fetch_start_date,
            end_date=end_date
        )
        
        if len(historical_data) < 10:
            return (
                "🐂 Bull Market Mix",
                "⚠️ Insufficient historical data (< 10 periods). Defaulting to Bull Mix.",
                0.30, {}
            )
            
        regime_name, mix_name, confidence, details = detector.detect_regime(historical_data)
        
        return mix_name, details['explanation'], confidence, details

    except Exception as e:
        logging.error(f"Error in get_market_mix_suggestion_v3: {e}")
        return (
            "🐂 Bull Market Mix",
            f"⚠️ Error during regime detection: {e}. Defaulting to Bull Mix.",
            0.30, {}
        )


# --- UI & Visualization Functions ---
def plot_weight_evolution(weight_history: List[Dict], title: str, y_axis_title: str):
    """Generic function to plot weight evolution over time."""
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
    st.plotly_chart(fig, use_container_width=True)

def display_performance_metrics(performance: Dict):
    """
    Renders the full performance dashboard for the backtest results.
    This includes cumulative metrics, equity curve, risk metrics, and advanced analytics.
    """
    if not performance:
        st.warning("Performance data not available. Please run an analysis.")
        return

    st.header("Out-of-Sample Performance Analysis")

    # Section for key cumulative performance indicators
    st.subheader("Cumulative Performance")
    curated_metrics = performance.get('strategy', {}).get('System_Curated', {}).get('metrics', {})

    # Display metrics in 5 columns for better layout
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Annual Return", f"{curated_metrics.get('annual_return', 0):.2%}",
                help="The geometric average return on an annualized basis. Higher is better.")
    col2.metric("Total Return", f"{curated_metrics.get('total_return', 0):.2%}",
                help="The total compounded return over the entire backtest period. Higher is better.")
    col3.metric("Sharpe Ratio", f"{curated_metrics.get('sharpe', 0):.2f}",
                help="Measures return per unit of total risk (volatility). Good > 1, Excellent > 2.")
    col4.metric("Calmar Ratio", f"{curated_metrics.get('calmar', 0):.2f}",
                help="Annualized return divided by the max drawdown. Good > 1, Excellent > 3.")
    col5.metric("Win Rate", f"{curated_metrics.get('win_rate', 0):.2%}",
                help="The percentage of periods with a positive return. Good > 50%.")

    # Section for Risk and other System-level metrics
    st.subheader("Risk & System Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Volatility", f"{curated_metrics.get('volatility', 0):.2%}",
                help="The annualized standard deviation of returns (risk). Lower is better.")
    col2.metric("Max Drawdown", f"{curated_metrics.get('max_drawdown', 0):.2%}",
                help="The largest peak-to-trough decline in value. Closer to 0 is better.")
    col3.metric("Sortino Ratio", f"{curated_metrics.get('sortino', 0):.2f}",
                help="Measures return per unit of downside risk. Good > 1, Excellent > 2.")
    col4.metric("Kelly Criterion", f"{curated_metrics.get('kelly_criterion', 0):.2%}",
                help="Theoretical optimal fraction of capital to allocate. Prone to estimation error; use with caution.")
    
    avg_entropy = curated_metrics.get('avg_weight_entropy')
    if avg_entropy is not None:
        st.metric("Average Weight Entropy", f"{avg_entropy:.3f}",
                  help="Measures portfolio diversification. Higher values indicate a more diversified, less concentrated portfolio.")

    st.markdown("---")

    # Section for the Equity Curve visualization
    st.subheader("Growth of Investment (Equity Curve)")
    equity_curves = []
    for name, perf in performance.get('strategy', {}).items():
        if perf.get('returns'):
            df = pd.DataFrame(perf['returns']).sort_values('date')
            df['equity'] = (1 + df['return']).cumprod() # Calculate cumulative growth
            df['strategy'] = name
            equity_curves.append(df)

    if equity_curves:
        full_equity_df = pd.concat(equity_curves)
        fig_equity = px.line(full_equity_df, x='date', y='equity', color='strategy',
                             title="Growth of ₹1 Investment Over Time",
                             labels={'equity': 'Growth of ₹1', 'date': 'Date', 'strategy': 'Strategy'})
        fig_equity.update_layout(template='plotly_dark', legend_title_text='Strategy')
        st.plotly_chart(fig_equity, use_container_width=True)

    # --- NEW: Strategy Weight Evolution Chart ---
    plot_weight_evolution(
        performance.get('strategy_weights_history', []),
        title="Strategy Weight Evolution Over Time",
        y_axis_title="Strategy Weight"
    )

    # Section for the Rolling Sharpe Ratio analysis
    st.subheader("Rolling Sharpe Ratio (3-Period Window)")
    df_list = []
    for name, perf in performance.get('strategy', {}).items():
        if perf.get('returns'):
            df = pd.DataFrame(perf['returns']).sort_values('date')
            # Calculate rolling metrics over a 3-period window
            rolling_mean = df['return'].rolling(window=3).mean()
            rolling_std = df['return'].rolling(window=3).std()
            periods_per_year = 52 # Assuming weekly data for annualization
            df['rolling_sharpe'] = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(periods_per_year)
            df['strategy'] = name
            df_list.append(df)

    if df_list:
        full_df = pd.concat(df_list)
        fig_sharpe = px.line(full_df, x='date', y='rolling_sharpe', color='strategy', title="Strategy Rolling Sharpe Ratio (3-Period)")
        fig_sharpe.update_layout(template='plotly_dark', legend_title_text='Strategy', yaxis_title="Sharpe Ratio")
        st.plotly_chart(fig_sharpe, use_container_width=True)

    # Section for the Strategy Correlation Matrix
    st.subheader("Strategy Correlation Matrix")
    returns_df = pd.DataFrame()
    for name, perf in performance.get('strategy', {}).items():
        if perf.get('returns'):
            # --- FIX: START ---
            # Create DataFrame from list of dicts
            df_raw = pd.DataFrame(perf['returns'])
            
            # Drop potential duplicate dates before setting index
            # This directly prevents the "duplicate labels" issue
            df = df_raw.drop_duplicates(subset='date', keep='last').set_index('date')
            
            # Log if we actually dropped anything
            if len(df_raw) > len(df):
                logging.warning(f"Removed {len(df_raw) - len(df)} duplicate date entries for strategy '{name}'.")

            returns_df[name] = df['return']
            # --- FIX: END ---

    corr_matrix = returns_df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Portland', aspect="auto")
    fig_corr.update_layout(title="Correlation of Strategy Returns", template='plotly_dark')
    st.plotly_chart(fig_corr, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)

def display_subset_weight_evolution(subset_weights_history: List[Dict], strategies: List[str]):
    """Displays an interactive chart for subset weight evolution."""
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
        'PR_v1': PRStrategy(), 
        'CL_v1': CL1Strategy(), 
        'CL2': CL2Strategy(), 
        'CL3': CL3Strategy(),
        'MomentumMasters': MomentumMasters(),
        'VolatilitySurfer': VolatilitySurfer(),
        'MOM_v1': MOM1Strategy(),
        'MOM_v2': MOM2Strategy() 
    }
    
    PORTFOLIO_STYLES = {
        "⚡ Swing Trading": {
            "description": "Short-term (3-21 day) holds to capture rapid momentum and volatility.",
            "mixes": {
                "🐂 Bull Market Mix": {
                    "strategies": ['CL2', 'CL3', 'VolatilitySurfer', 'PR_v1'],
                    "rationale": "Counter-intuitive smooth-trend specialists. CL2/CL3 dominate bull swings (14.4-14.6% returns) by avoiding momentum whipsaw in strong trends. VolatilitySurfer provides superior risk management (Calmar 141.93). PR_v1 captures pullbacks within uptrend. MomentumMasters deliberately EXCLUDED - performs worst in bull swings (9.24% vs 14.55% for CL2) due to premature stop-outs."
                },
                
                "🐻 Bear Market Mix": {
                    "strategies": ['VolatilitySurfer', 'MOM_v1', 'MomentumMasters'],
                    "rationale": "Volatility-first defense with measured aggression. VolatilitySurfer dominates bear swings (-1.21% loss vs -4.71% average) through superior drawdown control (-4.61% vs -5.54% average). MOM_v1 provides 38% win rate (best in bears) through adaptive positioning. MomentumMasters limited to 20% - surprisingly resilient in bears (-2.53%) despite failing in bulls."
                },
                
                "📊 Chop/Consolidate Mix": {
                    "strategies": ['VolatilitySurfer', 'MomentumMasters', 'MOM_v1'],
                    "rationale": "Range masters with breakout validation. VolatilitySurfer dominates chop swings (12.53% vs 7.96% for worst) with exceptional Calmar (20.42). MomentumMasters surprisingly effective in chop (11.60%, #2) - volatility enables quick pivots. MOM_v1 stabilizes with consistent middle performance (10.59%). CL strategies completely excluded - fail in choppy conditions (bottom 4 positions consistently)."
                }
            }
        },
        
        "💰 SIP Investment": {
            "description": "Systematic long-term (3-12+ months) wealth accumulation. Focus on consistency and drawdown protection.",
            "mixes": {
                "🐂 Bull Market Mix": {
                    "strategies": ['CL_v1', 'CL2', 'VolatilitySurfer', 'MOM_v1'],
                    "rationale": "Regime-specific reversion: CL strategies excel. CL_v1/CL2 deliver highest returns in bull SIPs (13.12% vs 11.57% for MomentumMasters) through low-volatility trend capture. VolatilitySurfer provides best risk-adjusted returns (Calmar 13.19) despite lower absolute performance. MOM_v1 stabilizes with adaptive allocation. MomentumMasters deliberately EXCLUDED despite 65% win rate - absolute returns lag (11.57%, dead last)."
                },
                
                "🐻 Bear Market Mix": {
                    "strategies": ['VolatilitySurfer', 'MomentumMasters', 'MOM_v1'],
                    "rationale": "Damage control with VolatilitySurfer anchor. VolatilitySurfer loses least in bear SIPs (-4.12% vs -5.86% average) with shallowest drawdown (-6.07% vs -8.97% average). MomentumMasters at 25% for measured participation (-4.80%, #2 overall). MOM_v1 provides adaptive defense (37% win rate, highest in bears). All CL strategies excluded - catastrophic bear SIP performance (bottom 4 positions, losses -5.54% to -5.86%)."
                },
                
                "📊 Chop/Consolidate Mix": {
                    "strategies": ['VolatilitySurfer', 'MomentumMasters', 'MOM_v1'],
                    "rationale": "Range extraction specialists. VolatilitySurfer dominates chop SIPs (9.95% vs 4.33% for worst) with superior Calmar (12.17). MomentumMasters surprisingly effective (#2, 8.91%) through volatility-enabled range navigation. MOM_v1 provides steady extraction (7.92%, #3). CL strategies systematically fail in chop (4.33-5.39%, bottom 4) - excluded entirely."
                }
            }
        },
        
        "🌍 All Weather": {
            "description": "Balanced, long-term (6-18 months) portfolio for all market regimes. Dynamic regime-aware allocation.",
            "mixes": {
                "🐂 Bull Market Mix": {
                    "strategies": ['CL_v1', 'CL2', 'VolatilitySurfer', 'MOM_v1', 'PR_v1'],
                    "rationale": "Bull-optimized multi-factor ensemble. CL_v1/CL2 (combined 40%) capture smooth bull trends with highest returns (13.12% in SIP, 14.4-14.6% in Swing). VolatilitySurfer (25%) provides risk management with best Calmar ratios across bull regimes (13.19 SIP, 141.93 Swing). MOM_v1 (20%) stabilizes through adaptive factors. PR_v1 (15%) captures tactical pullbacks. MomentumMasters excluded - consistently worst in both bull SIP (11.57%, #8) and bull Swing (9.24%, #8)."
                },
                
                "🐻 Bear Market Mix": {
                    "strategies": ['VolatilitySurfer', 'MOM_v1', 'MomentumMasters', 'CL_v1'],
                    "rationale": "Drawdown minimization with VolatilitySurfer leadership. VolatilitySurfer (40%) anchors with best bear performance across SIP (-4.12%, #1) and Swing (-1.21%, #1) while maintaining shallowest drawdowns. MOM_v1 (25%) provides highest bear win rates (37-38%) through adaptive defensive positioning. MomentumMasters (20%) contributes measured participation - surprisingly resilient in bears. CL_v1 (15%) limited to crisis monitoring - most CL strategies fail bears (bottom 4 positions)."
                },
                
                "📊 Chop/Consolidate Mix": {
                    "strategies": ['VolatilitySurfer', 'MomentumMasters', 'MOM_v1', 'MOM_v2', 'PR_v1'],
                    "rationale": "Range dominance architecture. VolatilitySurfer (35%) leads with exceptional chop performance (9.95% SIP #1, 12.53% Swing #1) and Calmar ratios (12.17 SIP, 20.42 Swing). MomentumMasters (25%) surprisingly effective in chop (#2 in both SIP 8.91% and Swing 11.60%) - high volatility enables range pivots. MOM_v1 (20%) provides consistent middle ground. MOM_v2 (10%) adds statistical arbitrage. PR_v1 (10%) tactical fades. All CL strategies excluded - systematic chop failure (bottom 4 in both SIP and Swing)."
                }
            }
        }
    }
    
    # --- NEW: `on_change` callback for the date input ---
    def update_regime_suggestion():
        """
        Called when the analysis date changes. Fetches *just enough*
        data to run the regime model and updates the session state.
        """
        selected_date_obj = st.session_state.get('analysis_date_str') # This is a datetime.date object
        if not selected_date_obj:
            return
            
        # --- FIX: Convert datetime.date to datetime.datetime ---
        # The strptime line was incorrect as we already have a date object
        selected_date = datetime.combine(selected_date_obj, datetime.min.time())
        
        # --- FIX: Moved st.toast out of cached function ---
        toast_msg = f"Fetching regime data for {selected_date.date()}..."
        st.toast(toast_msg, icon="🧠")
        # --- END FIX ---
        
        # This function is cached, so it's fast on repeated calls
        mix_name, explanation, confidence, details = get_market_mix_suggestion_v3(selected_date)
        
        st.session_state.suggested_mix = mix_name


    with st.sidebar:
        st.markdown("# ⚙️ Configuration")
        st.markdown("### Analysis Configuration")
        
        today = datetime.now()
        selected_date_str = st.date_input(
            "Select Analysis Date",
            value=today,
            min_value=today - timedelta(days=5*365), # 5 years back
            max_value=today,
            help="Choose a date to run the portfolio curation.",
            key='analysis_date_str', # Add a key to access the value
            on_change=update_regime_suggestion # Set the callback
        )
        
        st.markdown("### Portfolio Style Selection")

        selected_main_branch = st.selectbox(
            "1. Select Investment Style",
            options=list(PORTFOLIO_STYLES.keys()),
            help="Choose your primary investment objective (e.g., short-term trading or long-term investing)."
        )
        
        st.markdown("### Market Condition Mix")
        
        mix_options = list(PORTFOLIO_STYLES[selected_main_branch]["mixes"].keys())
        
        # --- UPDATED: No more dropdown, just display the suggested mix ---
        if st.session_state.suggested_mix:
             pass # Removed the st.info message
        else:
             # Run it once on the first load
             update_regime_suggestion()

        
        st.markdown("### Training Data Selection")
        lookback_files = st.number_input(
            "Lookback Files for Training",
            min_value=10,
            max_value=1000, # Set a reasonable max
            value=25, # Default to 25
            step=5,
            help=f"Number of historical files (days) to use for training, prior to the analysis date."
        )

        st.markdown("### Portfolio Parameters")
        capital = st.number_input("Capital (₹)", 1000, 100000000, 2500000, 1000, help="Total capital to allocate")
        num_positions = st.slider("Number of Positions", 5, 100, 30, 5, help="Maximum positions in the final portfolio")

        st.markdown("### Risk Management")
        st.markdown(f"**Min Position Size:** `{st.session_state.min_pos_pct:.1f}%`")
        st.markdown(f"**Max Position Size:** `{st.session_state.max_pos_pct:.1f}%`")

        if st.button("🚀 Run Analysis", width='stretch', type="primary"):
            
            # --- 1. Load Main Data for Backtest ---
            # This is the main data-loading call, triggered by the button
            
            # --- FIX: Get date object from state and convert to datetime ---
            selected_date_obj = st.session_state.get('analysis_date_str')
            if not selected_date_obj:
                st.error("Analysis date is missing. Please select a date.")
                st.stop()
                
            selected_date_dt = datetime.combine(selected_date_obj, datetime.min.time())

            # --- FIX: Moved st.toast out of cached function ---
            # Re-calculate fetch_start_date components for the toast message
            total_days_to_fetch = int((lookback_files + MAX_INDICATOR_PERIOD) * 1.5) + 30
            fetch_start_date = selected_date_dt - timedelta(days=total_days_to_fetch)
            toast_msg = f"Fetching live data for {len(SYMBOLS_UNIVERSE)} symbols from {fetch_start_date.date()} to {selected_date_dt.date()}..."
            st.toast(toast_msg, icon="⏳")
            # --- END FIX ---

            all_historical_data = load_historical_data(selected_date_dt, lookback_files)
            
            if not all_historical_data:
                st.error("Application cannot start: No historical data could be loaded or generated.")
                st.stop() # Stop execution if no data

            # --- 2. Filter data for training and current day ---
            # all_historical_data is already pre-filtered up to selected_date_str
            # The last item is the "current" data frame
            current_date, current_df = all_historical_data[-1]
            
            # The rest is the training data
            training_data = all_historical_data[:-1]
            
            # Apply the user's lookback filter
            if len(training_data) > lookback_files:
                training_data_window = training_data[-lookback_files:]
            else:
                training_data_window = training_data # Use all available if less than requested
            
            # Add the "current" day to the end of the training window
            # The backtester needs it to calculate the *final* portfolio
            training_data_window_with_current = training_data_window + [(current_date, current_df)]
            
            st.session_state.current_df = current_df
            st.session_state.selected_date = current_date.strftime('%Y-%m-%d')
            
            if len(training_data_window_with_current) < 10:
                st.error(f"Not enough training data loaded ({len(training_data_window_with_current)} days). Need at least 10. Check date range or lookback period.")
                st.stop()
                
            # --- 3. Determine Strategies to Run ---
            if not st.session_state.suggested_mix:
                st.error("Market regime could not be determined. Please select a valid date. Analysis cannot run.")
                st.stop()
                
            final_mix_to_use = st.session_state.suggested_mix 
            
            style_strategies = PORTFOLIO_STYLES[selected_main_branch]["mixes"][final_mix_to_use]['strategies']
            strategies_to_run = {name: strategies[name] for name in style_strategies}

            # --- 4. Run the Backtest & Curation ---
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

    st.title("Pragati : Quantitative Portfolio Curation System")
    # st.subheader(f"Displaying analysis for date: {st.session_state.selected_date}" if st.session_state.selected_date else "Please run an analysis")

    if st.session_state.portfolio is not None:
        total_value = st.session_state.portfolio['value'].sum()
        cash_remaining = capital - total_value

        # Using metric cards for the header stats
        col1, col2, col3 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><h4>Total Invested</h4><h2>{total_value:,.2f}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h4>Positions</h4><h2>{len(st.session_state.portfolio)}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h4>Cash Remaining</h4><h2>{cash_remaining:,.2f}</h2></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h4>Max Position</h4><h2>{st.session_state.portfolio['weightage_pct'].max():.2f}%</h2></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["**📈 Portfolio**", "**📊 Performance**", "**🎯 Strategy Deep Dive**"])

    with tab1:
        st.header("Curated Portfolio Holdings")
        if st.session_state.portfolio is not None:
            display_df = st.session_state.portfolio[['symbol', 'price', 'units', 'weightage_pct', 'value']]
            st.dataframe(display_df.style.format({'price': '{:,.2f}', 'value': '{:,.2f}', 'units': '{:,.2f}', 'weightage_pct': '{:.2f}%'}))
            
            # Prepare DataFrame for CSV export with specific column order
            portfolio_df = st.session_state.portfolio
            first_cols = ['symbol', 'price', 'units']
            other_cols = [col for col in portfolio_df.columns if col not in first_cols]
            new_order = first_cols + other_cols
            download_df = portfolio_df[new_order]

            csv = fix_csv_export(download_df)
            st.download_button("📥 Download Portfolio CSV", csv, f"curated_portfolio_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", width='stretch')
        else: st.info("👈 Run analysis from the sidebar to generate a portfolio.")

    with tab2:
        if st.session_state.performance:
            display_performance_metrics(st.session_state.performance)
        else: st.info("👈 Run analysis to view performance metrics.")

    with tab3:
        st.header("Strategy & Subset Analysis")
        if st.session_state.performance and st.session_state.current_df is not None:
            # Get the list of strategies that were actually run in the analysis
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
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else: st.info("👈 Run analysis to view strategy details.")

if __name__ == "__main__":
    main()
