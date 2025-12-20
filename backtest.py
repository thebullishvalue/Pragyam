"""
Backtest Engine - Institutional Analytics & Optimization Suite

A hedge-fund-grade backtesting platform with comprehensive performance attribution,
risk analytics, and portfolio holdings evaluation.

This version integrates the Lookback Period Optimizer, allowing for both
full backtest simulations and sensitivity analysis to find robust strategy
parameters.

V21 Changes:
- ML-AWARE: Updated all backtesting functions to correctly train 'MLStrategy'.
- UPDATED STYLES: Synced PORTFOLIO_STYLES with pragati.py.
- DUAL-MODE OPERATION: Switch between 'Backtest Simulation' and 'Lookback Sensitivity Analysis'.

V22 Changes:
- DYNAMIC ML TRAINING: Patched strategies to train dynamically inside simulation loops.

V23 Changes:
- MULTI-RANGE SELECTION: Added ability to select multiple non-continuous date ranges.

V24 Changes:
- INTELLIGENT REGIME STITCHING: Fixed P&L distortion across non-continuous ranges.
  1. Detects time gaps > 7 days between data points.
  2. Forces liquidation (cash settlement) at the end of each regime chunk.
  3. Sanitizes return metrics to prevent artificial spikes on regime boundaries.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib.util
import os
import logging
from typing import List, Tuple, Dict

# --- Page Configuration ---
st.set_page_config(page_title="Backtest Engine - Analytics & Optimization", layout="wide", page_icon="⚙️")

# --- Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("BacktestEngine")

# --- Module Loader ---
@st.cache_resource
def load_pragati_as_module():
    pragati_path = "pragati.py"
    if not os.path.exists(pragati_path):
        st.error(f"Fatal Error: 'pragati.py' not found.")
        return None
    original_set_page_config = st.set_page_config
    original_markdown = st.markdown
    def do_nothing(*args, **kwargs): pass
    try:
        # Suppress pragati's own page config and markdown calls
        st.set_page_config = do_nothing
        st.markdown = do_nothing
        spec = importlib.util.spec_from_file_location("pragati", pragati_path)
        pragati_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pragati_module)
        return pragati_module
    except Exception as e:
        st.error(f"Failed to load 'pragati.py': {e}")
        return None
    finally:
        # Restore Streamlit's original functions
        st.set_page_config = original_set_page_config
        st.markdown = original_markdown

pragati = load_pragati_as_module()
if not pragati:
    st.stop()

# --- Global Configurations & Styles ---
PORTFOLIO_STYLES = {
    "⚡ Swing Trading": {
        "description": "Short-term (3-21 day) holds to capture rapid momentum and volatility and momentum.",
        "mixes": {
            "🐂 Bull Market Mix": {
                "strategies": ['VolatilitySurfer', 'NebulaMomentumStorm', 'MultiverseAlpha', 'ResonanceEcho'],
                "rationale": "Counter-intuitive smooth-trend specialists. CL2/CL3 dominate bull swings."
            },
            "🐻 Bear Market Mix": {
                "strategies": ['AdaptiveVolBreakout', 'WormholeTemporal', 'AdaptiveVolBreakout', 'MultiverseAlpha'],
                "rationale": "Volatility-first defense with measured aggression."
            },
            "📊 Chop/Consolidate Mix": {
                "strategies": ['AdaptiveVolBreakout', 'WormholeTemporal', 'AdaptiveVolBreakout', 'MultiverseAlpha'],
                "rationale": "Range masters with breakout validation."
            }
        }
    },
    "💰 SIP Investment": {
        "description": "Systematic long-term (3-12+ months) wealth accumulation. Focus on consistency and drawdown protection.",
        "mixes": {
            "🐂 Bull Market Mix": {
                "strategies": ['VolatilitySurfer', 'NebulaMomentumStorm', 'MultiverseAlpha', 'ResonanceEcho'],
                "rationale": "Regime-specific reversion: CL strategies excel."
            },
            "🐻 Bear Market Mix": {
                "strategies": ['DivineMomentumOracle', 'AdaptiveVolBreakout', 'VolatilitySurfer', 'PantheonAlphaRealm'],
                "rationale": "Damage control with VolatilitySurfer anchor."
            },
            "📊 Chop/Consolidate Mix": {
                "strategies": ['DivineMomentumOracle', 'AdaptiveVolBreakout', 'VolatilitySurfer', 'PantheonAlphaRealm'],
                "rationale": "Range extraction specialists."
            }
        }
    },
    "🌍 All Weather": {
        "description": "Balanced, long-term (6-18 months) portfolio for all market regimes. Dynamic regime-aware allocation.",
        "mixes": {
            "🐂 Bull Market Mix": {
                "strategies": ['AdaptiveVolBreakout', 'VolatilitySurfer'],
                "rationale": "Bull-optimized multi-factor ensemble."
            },
            "🐻 Bear Market Mix": {
                "strategies": ['AdaptiveVolBreakout', 'VolatilitySurfer'],
                "rationale": "Drawdown minimization with VolatilitySurfer anchor."
            },
            "📊 Chop/Consolidate Mix": {
                "strategies": ['AdaptiveVolBreakout', 'VolatilitySurfer'],
                "rationale": "Range dominance architecture."
            }
        }
    }
}

# --- Enhanced UI Theme ---
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
    .status-dashboard { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 20px; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.15); }
    .status-title { font-weight: bold; font-size: 1.2rem; color: var(--primary-color); }
    .status-text { font-family: monospace; font-size: 1rem; color: var(--text-color); }
</style>
""", unsafe_allow_html=True)


# --- Utility Functions ---
@st.cache_data
def load_external_trigger_file(uploaded_file):
    if uploaded_file is None: return None
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        if 'DATE' not in df.columns:
            st.error("External file must contain a 'DATE' column.")
            return None
        df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y', errors='coerce')
        df.dropna(subset=['DATE'], inplace=True)
        df = df.set_index('DATE')
        logger.info(f"Loaded external trigger file with {len(df)} entries.")
        return df
    except Exception as e:
        st.error(f"Failed to process external trigger file: {e}")
        return None

def calculate_ad_ratio(df: pd.DataFrame) -> float:
    if 'rsi latest' not in df.columns: return 1.0
    advancing = df[df['rsi latest'] > 50].shape[0]
    declining = df[df['rsi latest'] < 50].shape[0]
    return advancing / declining if declining > 0 else advancing

def filter_data_by_ranges(data_list, date_ranges):
    """
    Filters the historical data list based on multiple date ranges.
    """
    filtered_data = []
    seen_dates = set()
    sorted_ranges = sorted(date_ranges, key=lambda x: x[0])
    
    for d, df in data_list:
        d_date = d.date()
        in_range = False
        for start, end in sorted_ranges:
            if start <= d_date <= end:
                in_range = True
                break
        
        if in_range and d_date not in seen_dates:
            filtered_data.append((d, df))
            seen_dates.add(d_date)
            
    return filtered_data

def identify_regime_gaps(simulation_data):
    """
    Identifies indices in simulation_data where a significant gap (regime change) occurs immediately after.
    Returns a set of indices where forced liquidation should happen.
    """
    liquidation_indices = set()
    if not simulation_data:
        return liquidation_indices
    
    # Threshold for a "gap" in days (e.g., > 7 days implies a week+ gap, likely regime change)
    GAP_THRESHOLD_DAYS = 7
    
    for i in range(len(simulation_data) - 1):
        current_date = simulation_data[i][0]
        next_date = simulation_data[i+1][0]
        delta = (next_date - current_date).days
        
        if delta > GAP_THRESHOLD_DAYS:
            liquidation_indices.add(i)
            
    # Always liquidate at the very end of the simulation
    liquidation_indices.add(len(simulation_data) - 1)
    
    return liquidation_indices

# --- INSTITUTIONAL PERFORMANCE METRICS ---
def calculate_institutional_metrics(daily_data: pd.DataFrame, risk_free_rate: float = 0.0, deployment_style='SIP'):
    if daily_data.empty: return {}
    final_value = daily_data['value'].iloc[-1]
    
    # V24: Enhanced Metrics with Gap Sanitization
    df = daily_data.copy()
    
    # Calculate date difference to identify gaps
    df['date_diff'] = df['date'].diff().dt.days.fillna(1)
    
    # Calculate returns normally
    df['cash_flow'] = df['investment'].diff().fillna(0)
    df['prev_value'] = df['value'].shift(1).fillna(0)
    df['returns'] = (df['value'] - df['cash_flow']) / df['prev_value'] - 1
    
    # Sanitize: If prev_value is 0, return is 0
    df.loc[df['prev_value'] == 0, 'returns'] = 0
    
    # V24 SANITIZATION: If gap is > 7 days, force return to 0.0
    # This prevents the "24% best day" bug when stitching regimes.
    df.loc[df['date_diff'] > 7, 'returns'] = 0.0
    
    # Handle infinite values
    returns = df['returns'].replace([np.inf, -np.inf], 0)
    
    if deployment_style == 'SIP (Systematic Investment)':
        total_investment = daily_data['investment'].iloc[-1]
        absolute_pnl = final_value - total_investment
        total_return_pct = absolute_pnl / total_investment if total_investment > 0 else 0
        ann_return = (1 + returns.mean()) ** 252 - 1
        cagr = None
    else: # Lumpsum
        initial_investment = daily_data['investment'].iloc[0]
        absolute_pnl = final_value - initial_investment
        total_return_pct = absolute_pnl / initial_investment if initial_investment > 0 else 0
        
        if returns.empty: return {'Total P&L %': total_return_pct, 'Absolute P&L': absolute_pnl}
        
        ann_return = (1 + returns.mean()) ** 252 - 1
        
        # Calculate CAGR correctly considering only active trading periods
        # We estimate years by counting actual trading days / 252
        years = len(daily_data) / 252
        cagr = ((final_value / initial_investment) ** (1/years) - 1) if years > 0 and initial_investment > 0 else 0

    if returns.empty: return {'Total P&L %': total_return_pct, 'Absolute P&L': absolute_pnl}
    
    ann_factor = np.sqrt(252)
    volatility = returns.std() * ann_factor
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * ann_factor if not downside_returns.empty else 0
    excess_returns = returns - (risk_free_rate / 252)
    
    sharpe_ratio = (excess_returns.mean() / returns.std() * ann_factor) if returns.std() > 0 else 0
    sortino_ratio = (excess_returns.mean() / downside_vol * ann_factor) if downside_vol > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / running_max) - 1
    max_drawdown = drawdown.min()
    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    metrics = {
        'Total Return %': total_return_pct, 'Annualized Return %': ann_return, 'Absolute P&L': absolute_pnl,
        'Volatility % (Ann.)': volatility, 'Max Drawdown %': max_drawdown, 'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio, 'Calmar Ratio': calmar_ratio, 'Win Rate %': (returns > 0).mean(),
        'Best Day %': returns.max(), 'Worst Day %': returns.min(), 'Trading Days': len(returns),
        'Current Value': final_value
    }
    if cagr is not None: metrics['CAGR %'] = cagr
    return metrics

# --- CORE BACKTESTING ENGINES ---
def run_pragati_system_backtest(date_ranges, buy_trigger_col, buy_trigger_threshold,
                                sell_trigger_enabled, sell_trigger_col, sell_trigger_threshold,
                                capital_amount, deployment_style, lookback_period, selected_style, 
                                _status_placeholder, _external_trigger_df):
    
    range_str = ", ".join([f"{s} to {e}" for s, e in date_ranges])
    logger.info(f"STARTING: Pragati System Backtest | Mode: {deployment_style} | Ranges: {range_str}")
    
    all_historical_data = pragati.load_historical_data()
    simulation_data = filter_data_by_ranges(all_historical_data, date_ranges)
    
    if not simulation_data:
        st.error("No historical data found for the selected date ranges.")
        return None
    
    # V24: Identify gaps for forced liquidation
    liquidation_indices = identify_regime_gaps(simulation_data)

    daily_values, portfolio_units, total_investment, current_capital, buy_signal_active = [], {}, 0, capital_amount, False
    trade_log = []

    buy_dates_mask = [False] * len(simulation_data)
    if _external_trigger_df is not None and buy_trigger_col in _external_trigger_df.columns:
        external_buy_dates = _external_trigger_df[_external_trigger_df[buy_trigger_col] < buy_trigger_threshold].index.date
        buy_dates_mask = [d.date() in external_buy_dates for d, _ in simulation_data]
    
    sell_dates_mask = [False] * len(simulation_data)
    if sell_trigger_enabled and _external_trigger_df is not None and sell_trigger_col in _external_trigger_df.columns:
        external_sell_dates = _external_trigger_df[_external_trigger_df[sell_trigger_col] > sell_trigger_threshold].index.date
        sell_dates_mask = [d.date() in external_sell_dates for d, _ in simulation_data]

    for i, (current_date, current_df) in enumerate(simulation_data):
        _status_placeholder.markdown(f"<div class='status-dashboard'><p class='status-title'>Executing Pragati System Backtest</p><p class='status-text'>Processing: {current_date.strftime('%Y-%m-%d')} | Day {i+1}/{len(simulation_data)}</p></div>", unsafe_allow_html=True)
        
        # 1. Normal Sell Trigger Check
        if sell_dates_mask[i] and portfolio_units:
            trade_log.append({'Event': 'SELL', 'Date': current_date})
            prices_today = current_df.set_index('symbol')['price']; sell_value = sum(units * prices_today.get(symbol, 0) for symbol, units in portfolio_units.items())
            if deployment_style == 'Lumpsum': current_capital += sell_value
            portfolio_units, buy_signal_active = {}, False
        
        # 2. Buy Logic
        is_buy_day = buy_dates_mask[i]; actual_buy_trigger = is_buy_day and not buy_signal_active
        if is_buy_day: buy_signal_active = True
        elif not is_buy_day: buy_signal_active = False
        can_buy = (deployment_style == 'SIP (Systematic Investment)') or (deployment_style == 'Lumpsum' and not portfolio_units)
        
        if actual_buy_trigger and can_buy:
            amount_to_deploy = capital_amount if deployment_style == 'SIP (Systematic Investment)' else current_capital
            if amount_to_deploy > 1000:
                regime_detection_window = [d for d in all_historical_data if d[0] < current_date]
                mix_name, _, _, _ = pragati.get_market_mix_suggestion_v3(regime_detection_window)
                style_info = PORTFOLIO_STYLES[selected_style]["mixes"][mix_name]
                strategies_to_run = {cls.__name__: cls() for cls in pragati.BaseStrategy.__subclasses__() if cls.__name__ in style_info['strategies']}
                training_window = regime_detection_window[-lookback_period:]

                if 'MLStrategy' in strategies_to_run:
                    try:
                        strategies_to_run['MLStrategy'].train(training_window)
                    except Exception as e:
                        logger.error(f"MLStrategy live training failed: {e}")

                performance = pragati.evaluate_historical_performance(strategies_to_run, training_window)
                curated_port, _, _ = pragati.curate_final_portfolio(
                    strategies=strategies_to_run, performance=performance, current_df=current_df, 
                    sip_amount=amount_to_deploy, num_positions=30, min_pos_pct=1.0, max_pos_pct=10.0
                )
                if not curated_port.empty:
                    trade_log.append({'Event': 'BUY', 'Date': current_date})
                    amount_invested = curated_port['value'].sum()
                    if deployment_style == 'Lumpsum':
                        portfolio_units = pd.Series(curated_port.units.values, index=curated_port.symbol).to_dict(); current_capital -= amount_invested
                    else: # SIP
                        for _, row in curated_port.iterrows(): portfolio_units[row['symbol']] = portfolio_units.get(row['symbol'], 0) + row['units']
                        total_investment += amount_invested
        
        # 3. Calculate Portfolio Value for Today
        portfolio_value = 0
        if portfolio_units: 
            prices_today = current_df.set_index('symbol')['price']
            portfolio_value = sum(units * prices_today.get(symbol, 0) for symbol, units in portfolio_units.items())
        
        # 4. Log Daily Value
        if deployment_style == 'SIP (Systematic Investment)': 
            daily_values.append({'date': current_date, 'value': portfolio_value, 'investment': total_investment})
        else: 
            daily_values.append({'date': current_date, 'value': portfolio_value + current_capital, 'investment': capital_amount})
            
        # 5. V24: FORCE LIQUIDATION AT END OF REGIME
        # If this index is a liquidation point (end of a range), we sell everything to cash.
        # This ensures the gap jump doesn't distort P&L.
        if i in liquidation_indices and portfolio_units:
            logger.info(f"End of regime detected at {current_date.date()}. Force liquidating.")
            # We already calculated value above. Now convert units to cash for the NEXT iteration.
            prices_today = current_df.set_index('symbol')['price']
            liquidated_cash = sum(units * prices_today.get(symbol, 0) for symbol, units in portfolio_units.items())
            
            if deployment_style == 'Lumpsum':
                current_capital += liquidated_cash
            else:
                # For SIP, we technically "hold" the cash value implicitly in the 'value' tracker for the next step,
                # effectively exiting the market.
                pass 
                
            portfolio_units = {} # Clear positions
            buy_signal_active = False # Reset signal

    if not daily_values: return None
    daily_df = pd.DataFrame(daily_values)
    metrics = calculate_institutional_metrics(daily_df, deployment_style=deployment_style)

    if deployment_style == 'SIP (Systematic Investment)':
        metrics['Buy Events'] = len([t for t in trade_log if t['Event'] == 'BUY'])
        if not daily_df.empty: metrics['Total Investment'] = daily_df['investment'].iloc[-1]
    else: # Lumpsum
        metrics['Trade Events'] = len(trade_log)
        
    return {"metrics": metrics, "daily_data": daily_df}

def run_individual_strategies_backtest(date_ranges, buy_trigger_col, buy_trigger_threshold,
                                       sell_trigger_enabled, sell_trigger_col, sell_trigger_threshold,
                                       capital_amount, deployment_style,
                                       _status_placeholder, _external_trigger_df):
    
    range_str = ", ".join([f"{s} to {e}" for s, e in date_ranges])
    logger.info(f"STARTING: Individual Strategy Analysis | Mode: {deployment_style} | Ranges: {range_str}")
    
    all_historical_data = pragati.load_historical_data()
    strategies = {cls.__name__: cls() for cls in pragati.BaseStrategy.__subclasses__()}
    simulation_data = filter_data_by_ranges(all_historical_data, date_ranges)
    
    if not simulation_data:
        st.error("No historical data found for the selected date ranges.")
        return {}
    
    # V24: Identify gaps
    liquidation_indices = identify_regime_gaps(simulation_data)
        
    all_results = {}

    buy_dates_mask = [False] * len(simulation_data)
    if _external_trigger_df is not None and buy_trigger_col in _external_trigger_df.columns:
        external_buy_dates = _external_trigger_df[_external_trigger_df[buy_trigger_col] < buy_trigger_threshold].index.date
        buy_dates_mask = [d.date() in external_buy_dates for d, _ in simulation_data]
    
    sell_dates_mask = [False] * len(simulation_data)
    if sell_trigger_enabled and _external_trigger_df is not None and sell_trigger_col in _external_trigger_df.columns:
        external_sell_dates = _external_trigger_df[_external_trigger_df[sell_trigger_col] > sell_trigger_threshold].index.date
        sell_dates_mask = [d.date() in external_sell_dates for d, _ in simulation_data]

    for i, (name, strategy) in enumerate(strategies.items()):
        daily_values, portfolio_units, buy_signal_active, trade_log = [], {}, False, []
        
        if deployment_style == 'SIP (Systematic Investment)':
            total_investment = 0
            for j, (date, df) in enumerate(simulation_data):
                _status_placeholder.markdown(f"<div class='status-dashboard'><p class='status-title'>Executing Individual Backtest</p><p class='status-text'>Processing: {name} ({i+1}/{len(strategies)}) | {date.strftime('%Y-%m-%d')}</p></div>", unsafe_allow_html=True)
                
                if name == 'MLStrategy':
                    training_window = [d for d in all_historical_data if d[0].date() < date.date()]
                    if len(training_window) < 3:
                        current_value = sum(units * df.set_index('symbol')['price'].get(symbol, 0) for symbol, units in portfolio_units.items()) if portfolio_units else 0
                        daily_values.append({'date': date, 'value': current_value, 'investment': total_investment})
                        continue 
                    
                    try:
                        strategy.train(training_window[-60:])
                    except Exception:
                        current_value = sum(units * df.set_index('symbol')['price'].get(symbol, 0) for symbol, units in portfolio_units.items()) if portfolio_units else 0
                        daily_values.append({'date': date, 'value': current_value, 'investment': total_investment})
                        continue

                # 1. Normal Sell
                if sell_dates_mask[j] and portfolio_units: 
                    trade_log.append({'Event': 'SELL'})
                    portfolio_units, buy_signal_active = {}, False
                
                # 2. Buy
                is_buy_day = buy_dates_mask[j]; actual_buy_trigger = is_buy_day and not buy_signal_active
                if is_buy_day: buy_signal_active = True
                elif not is_buy_day: buy_signal_active = False
                
                if actual_buy_trigger:
                    trade_log.append({'Event': 'BUY'})
                    buy_portfolio = strategy.generate_portfolio(df, capital_amount); total_investment += buy_portfolio['value'].sum()
                    for _, row in buy_portfolio.iterrows(): portfolio_units[row['symbol']] = portfolio_units.get(row['symbol'], 0) + row['units']
                
                # 3. Value
                current_value = 0
                if portfolio_units: 
                    prices_today = df.set_index('symbol')['price']
                    current_value = sum(units * prices_today.get(symbol, 0) for symbol, units in portfolio_units.items())
                daily_values.append({'date': date, 'value': current_value, 'investment': total_investment})
                
                # 4. V24 Force Liquidation
                if j in liquidation_indices and portfolio_units:
                    portfolio_units = {}
                    buy_signal_active = False

        else: # Lumpsum
            current_capital = capital_amount
            for j, (date, df) in enumerate(simulation_data):
                _status_placeholder.markdown(f"<div class='status-dashboard'><p class='status-title'>Executing Individual Backtest</p><p class='status-text'>Processing: {name} ({i+1}/{len(strategies)}) | {date.strftime('%Y-%m-%d')}</p></div>", unsafe_allow_html=True)
                
                if name == 'MLStrategy':
                    training_window = [d for d in all_historical_data if d[0].date() < date.date()]
                    if len(training_window) < 3:
                        portfolio_value = sum(units * df.set_index('symbol')['price'].get(symbol, 0) for symbol, units in portfolio_units.items()) if portfolio_units else 0
                        daily_values.append({'date': date, 'value': portfolio_value + current_capital, 'investment': capital_amount})
                        continue
                    try:
                        strategy.train(training_window[-60:])
                    except Exception:
                        portfolio_value = sum(units * df.set_index('symbol')['price'].get(symbol, 0) for symbol, units in portfolio_units.items()) if portfolio_units else 0
                        daily_values.append({'date': date, 'value': portfolio_value + current_capital, 'investment': capital_amount})
                        continue

                # 1. Normal Sell
                if sell_dates_mask[j] and portfolio_units:
                    trade_log.append({'Event': 'SELL'})
                    prices_today = df.set_index('symbol')['price']; current_capital += sum(units * prices_today.get(symbol, 0) for symbol, units in portfolio_units.items())
                    portfolio_units, buy_signal_active = {}, False
                
                # 2. Buy
                is_buy_day = buy_dates_mask[j]; actual_buy_trigger = is_buy_day and not buy_signal_active
                if is_buy_day: buy_signal_active = True
                elif not is_buy_day: buy_signal_active = False
                
                if actual_buy_trigger and not portfolio_units and current_capital > 1000:
                    trade_log.append({'Event': 'BUY'})
                    buy_portfolio = strategy.generate_portfolio(df, current_capital)
                    if not buy_portfolio.empty: portfolio_units = pd.Series(buy_portfolio.units.values, index=buy_portfolio.symbol).to_dict(); current_capital -= buy_portfolio['value'].sum()
                
                # 3. Value
                portfolio_value = 0
                if portfolio_units: 
                    prices_today = df.set_index('symbol')['price']
                    portfolio_value = sum(units * prices_today.get(symbol, 0) for symbol, units in portfolio_units.items())
                daily_values.append({'date': date, 'value': portfolio_value + current_capital, 'investment': capital_amount})
                
                # 4. V24 Force Liquidation
                if j in liquidation_indices and portfolio_units:
                    prices_today = df.set_index('symbol')['price']
                    current_capital += sum(units * prices_today.get(symbol, 0) for symbol, units in portfolio_units.items())
                    portfolio_units = {}
                    buy_signal_active = False
        
        if not daily_values: continue
        daily_df = pd.DataFrame(daily_values)
        metrics = calculate_institutional_metrics(daily_df, deployment_style=deployment_style)
        
        if deployment_style == 'SIP (Systematic Investment)':
            metrics['Buy Events'] = len([t for t in trade_log if t['Event'] == 'BUY'])
            if not daily_df.empty: metrics['Total Investment'] = daily_df['investment'].iloc[-1]
        else: # Lumpsum
            metrics['Trade Events'] = len(trade_log)

        all_results[name] = {"metrics": metrics, "daily_data": daily_df}

    logger.info("COMPLETED: Individual Strategy Analysis")
    return all_results

# --- SENSITIVITY ANALYSIS ENGINE ---
def run_lookback_sensitivity_analysis(full_historical_data, lookback_range, analysis_dates, performance_metric,
                                      historical_data_map, latest_prices_df, selected_style, analysis_mode, 
                                      static_strategies_to_test=None):
    results = []; progress_bar = st.progress(0, text="Initializing...")
    all_strategies = {cls.__name__: cls() for cls in pragati.BaseStrategy.__subclasses__()}
    for i, lookback in enumerate(lookback_range):
        progress_bar.progress((i + 1) / len(lookback_range), text=f"Testing lookback {lookback}...")
        lookback_performances = []
        for date_str in analysis_dates:
            analysis_date = datetime.strptime(date_str, '%Y-%m-%d')
            data_before_analysis = [d for d in full_historical_data if d[0] < analysis_date]
            if len(data_before_analysis) < lookback: continue
            training_window = data_before_analysis[-lookback:]
            
            strategies_to_test = static_strategies_to_test
            if analysis_mode == "A/D Ratio Threshold":
                regime_detection_window = [d for d in full_historical_data if d[0] <= analysis_date]
                mix_name, _, _, _ = pragati.get_market_mix_suggestion_v3(regime_detection_window)
                style_info = PORTFOLIO_STYLES[selected_style]["mixes"].get(mix_name)
                if not style_info: continue
                # Create new instances for each test
                strategies_to_test = {name: all_strategies[name].__class__() for name in style_info['strategies']}
            
            if strategies_to_test is None: continue

            # --- V21 ML-Aware FIX ---
            # Train the MLStrategy instance *before* passing it to the backtester and curator
            if 'MLStrategy' in strategies_to_test:
                logger.info(f"Sensitivity: Training MLStrategy for {date_str} on {len(training_window)} periods...")
                try:
                    strategies_to_test['MLStrategy'].train(training_window)
                    if strategies_to_test['MLStrategy'].pipeline is None:
                        logger.warning(f"MLStrategy training failed for {date_str}, will be excluded.")
                except Exception as e:
                    logger.error(f"MLStrategy sensitivity training failed: {e}")
            # --- End V21 ML-Aware FIX ---

            performance = pragati.evaluate_historical_performance(strategies_to_test, training_window)
            metric_value = 0
            if performance_metric == 'portfolio_return':
                curated_port, _, _ = pragati.curate_final_portfolio(
                    strategies=strategies_to_test, performance=performance, current_df=historical_data_map.get(date_str),
                    sip_amount=100000, num_positions=30, min_pos_pct=1.0, max_pos_pct=10.0
                )
                if not curated_port.empty: metric_value = pragati.compute_portfolio_return(curated_port, latest_prices_df)
            else:
                metric_value = performance.get('strategy', {}).get('System_Curated', {}).get('metrics', {}).get(performance_metric, 0)
            lookback_performances.append(metric_value)
        if lookback_performances: results.append({'lookback_period': lookback, 'performance': np.mean(lookback_performances)})
    progress_bar.empty(); return pd.DataFrame(results)

# --- VISUALIZATION FUNCTIONS ---
def plot_comprehensive_performance(daily_data, title):
    df = daily_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # V24: Use masked returns if available, otherwise calculate, but ensure gap handling
    if 'returns' not in df.columns:
        df['date_diff'] = df.index.to_series().diff().dt.days.fillna(1)
        df['returns'] = df['value'].pct_change()
        df.loc[df['date_diff'] > 7, 'returns'] = 0.0 # Sanitize visual returns too
    
    returns = df['returns'].replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty: 
        return go.Figure().update_layout(title_text="No trading activity to display.", template='plotly_dark')

    fig = make_subplots(rows=2, cols=2, subplot_titles=('Portfolio Value', 'Drawdown (%)', 'Daily Returns Distribution', 'Rolling 30D Sharpe'))
    fig.add_trace(go.Scatter(x=df.index, y=df['value'], name='Portfolio', line=dict(color='#FFC300')), row=1, col=1)
    if df['investment'].nunique() > 1: fig.add_trace(go.Scatter(x=df.index, y=df['investment'], name='Investment', line=dict(color='#888888', dash='dash')), row=1, col=1)
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown = ((cumulative / running_max) - 1) * 100
    
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name='Drawdown', fill='tozeroy', line=dict(color='#FF5B5B')), row=1, col=2)
    fig.add_trace(go.Histogram(x=returns * 100, name='Returns', marker_color='#FFC300'), row=2, col=1)
    
    if len(returns) >= 30:
        rolling_sharpe = returns.rolling(30).mean() / returns.rolling(30).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='30D Sharpe', line=dict(color='#00BCD4')), row=2, col=2)
    
    fig.update_layout(title=f'<b>{title}</b>', template='plotly_dark', height=700, showlegend=False)
    return fig

def plot_sensitivity_results(results_df: pd.DataFrame, metric_name: str):
    if results_df.empty: st.warning("No sensitivity results to plot."); return
    metric_label = metric_name.replace('_', ' ').capitalize()
    fig = go.Figure(go.Scatter(x=results_df['lookback_period'], y=results_df['performance'], mode='lines+markers', name=metric_label))
    best_lookback = results_df.loc[results_df['performance'].idxmax()]
    fig.add_annotation(x=best_lookback['lookback_period'], y=best_lookback['performance'], text=f"Peak at {best_lookback['lookback_period']}", showarrow=True)
    fig.update_layout(title=f'Performance vs. Lookback Period ({metric_label})', xaxis_title='Lookback Period (Days)', yaxis_title=f'Average {metric_label}', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Optimal lookback period based on {metric_label} is **{best_lookback['lookback_period']} days**.")

# --- MAIN APPLICATION ---
def main():
    st.title("Backtest Engine - Analytics & Optimization")
    st.sidebar.title("⚙️ Controls")
    analysis_mode = st.sidebar.radio("Select Analysis Mode", ["Backtest Simulation", "Lookback Sensitivity Analysis"])
    all_historical_data = pragati.load_historical_data()
    if not all_historical_data: st.error("No historical data found."); return
    historical_data_map = {d.strftime('%Y-%m-%d'): df for d, df in all_historical_data}
    available_dates_str = sorted(historical_data_map.keys(), reverse=True)
    
    # Helper to get date objects
    all_dates_dt = sorted([d.date() for d, _ in all_historical_data])

    if analysis_mode == "Backtest Simulation":
        st.sidebar.header("Select Simulation Type")
        simulation_type = st.sidebar.radio("Simulation Type", ["Pragati System", "Individual Strategy"])
        st.sidebar.header("💰 Capital Deployment")
        deployment_style = st.sidebar.radio("Deployment Style", ["SIP (Systematic Investment)", "Lumpsum"])
        capital = st.sidebar.number_input("Amount (₹)", value=100000 if deployment_style == 'SIP (Systematic Investment)' else 5000000)
        
        st.sidebar.header("📅 Execution Parameters")
        
        # --- V23: Multiple Date Range Selection ---
        date_mode = st.sidebar.radio("Date Selection Mode", ["Single Continuous Range", "Multiple Custom Ranges"])
        date_ranges = []
        
        if date_mode == "Single Continuous Range":
            start_date = st.sidebar.date_input("Start Date", all_dates_dt[0])
            end_date = st.sidebar.date_input("End Date", all_dates_dt[-1])
            if start_date <= end_date:
                date_ranges = [(start_date, end_date)]
            else:
                st.sidebar.error("Start date must be before End date.")
        else:
            st.sidebar.markdown("### Define Ranges")
            st.sidebar.info("Add ranges to simulate specific regimes (e.g., Bull or Bear) stitched together.")
            
            # Default to two example ranges if session state is empty
            default_ranges = pd.DataFrame([
                {"Start Date": all_dates_dt[0], "End Date": all_dates_dt[min(30, len(all_dates_dt)-1)]},
                {"Start Date": all_dates_dt[max(0, len(all_dates_dt)-31)], "End Date": all_dates_dt[-1]}
            ])
            
            edited_df = st.sidebar.data_editor(
                default_ranges, 
                num_rows="dynamic",
                column_config={
                    "Start Date": st.column_config.DateColumn("Start Date", required=True),
                    "End Date": st.column_config.DateColumn("End Date", required=True)
                },
                hide_index=True,
                use_container_width=True
            )
            
            for _, row in edited_df.iterrows():
                if row['Start Date'] and row['End Date'] and row['Start Date'] <= row['End Date']:
                    date_ranges.append((row['Start Date'], row['End Date']))
        
        if simulation_type == "Pragati System":
            lookback_period = st.sidebar.number_input("Strategy Lookback (days)", min_value=10, max_value=100, value=45)
            selected_style = st.sidebar.selectbox("Investment Style", list(PORTFOLIO_STYLES.keys()))
        else:
            lookback_period, selected_style = None, None

        st.sidebar.header("🎯 Trigger Conditions (External File)"); uploaded_file = st.sidebar.file_uploader("Upload Trigger File (CSV/XLSX)", type=['csv', 'xlsx']); external_df = load_external_trigger_file(uploaded_file)
        buy_col, sell_col, buy_thresh, sell_thresh, sell_enabled = None, None, 0.0, 0.0, False
        if external_df is not None:
            numeric_cols = external_df.select_dtypes(include=np.number).columns.tolist()
            buy_col = st.sidebar.selectbox("Buy Column", numeric_cols); buy_thresh = st.sidebar.number_input(f"Buy when {buy_col} is BELOW", value=0.8)
            sell_enabled = st.sidebar.checkbox("Enable Sell Trigger")
            if sell_enabled: sell_col = st.sidebar.selectbox("Sell Column", numeric_cols, index=min(1, len(numeric_cols)-1)); sell_thresh = st.sidebar.number_input(f"Sell when {sell_col} is ABOVE", value=1.5)
        else: st.sidebar.warning("Upload a trigger file to run the backtest.")

        if st.sidebar.button("🚀 Execute Backtest", use_container_width=True, type="primary", disabled=(external_df is None or not date_ranges)):
            st.session_state.deployment_style = deployment_style
            status_placeholder = st.empty()
            common_params = {'date_ranges': date_ranges, 'buy_trigger_col': buy_col, 'buy_trigger_threshold': buy_thresh,
                             'sell_trigger_enabled': sell_enabled, 'sell_trigger_col': sell_col, 'sell_trigger_threshold': sell_thresh,
                             'capital_amount': capital, 'deployment_style': deployment_style, '_status_placeholder': status_placeholder, '_external_trigger_df': external_df}
            
            if simulation_type == "Pragati System":
                results = run_pragati_system_backtest(**common_params, lookback_period=lookback_period, selected_style=selected_style)
            else:
                results = run_individual_strategies_backtest(**common_params)
            
            st.session_state.backtest_results = results; st.session_state.simulation_type = simulation_type
            status_placeholder.empty(); st.success("Backtest execution complete!")

        if 'backtest_results' in st.session_state and st.session_state.backtest_results:
            results = st.session_state.backtest_results
            if st.session_state.simulation_type == "Pragati System":
                st.header("🎯 Pragati System Performance Summary")
                metrics = results['metrics']
                
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)

                if st.session_state.deployment_style == 'SIP (Systematic Investment)':
                    with col1: st.markdown(f"<div class='metric-card'><h4>Total Return</h4><h2>{metrics.get('Total Return %', 0):.2%}</h2></div>", unsafe_allow_html=True)
                    with col2: st.markdown(f"<div class='metric-card'><h4>Total Invested</h4><h2>₹{metrics.get('Total Investment', 0):,.0f}</h2></div>", unsafe_allow_html=True)
                    with col3: st.markdown(f"<div class='metric-card'><h4>Current Value</h4><h2>₹{metrics.get('Current Value', 0):,.0f}</h2></div>", unsafe_allow_html=True)
                    st.markdown("---")
                    with col4: st.markdown(f"<div class='metric-card'><h4>Sharpe Ratio</h4><h2>{metrics.get('Sharpe Ratio', 0):.2f}</h2></div>", unsafe_allow_html=True)
                    with col5: st.markdown(f"<div class='metric-card'><h4>Max Drawdown</h4><h2>{metrics.get('Max Drawdown %', 0):.2%}</h2></div>", unsafe_allow_html=True)
                    with col6: st.markdown(f"<div class='metric-card'><h4>Buy Events</h4><h2>{metrics.get('Buy Events', 0)}</h2></div>", unsafe_allow_html=True)
                else: # Lumpsum
                    with col1: st.markdown(f"<div class='metric-card'><h4>Total Return</h4><h2>{metrics.get('Total Return %', 0):.2%}</h2></div>", unsafe_allow_html=True)
                    with col2: st.markdown(f"<div class='metric-card'><h4>Current Value</h4><h2>₹{metrics.get('Current Value', 0):,.0f}</h2></div>", unsafe_allow_html=True)
                    with col3: st.markdown(f"<div class='metric-card'><h4>Sharpe Ratio</h4><h2>{metrics.get('Sharpe Ratio', 0):.2f}</h2></div>", unsafe_allow_html=True)
                    st.markdown("---")
                    with col4: st.markdown(f"<div class='metric-card'><h4>Calmar Ratio</h4><h2>{metrics.get('Calmar Ratio', 0):.2f}</h2></div>", unsafe_allow_html=True)
                    with col5: st.markdown(f"<div class='metric-card'><h4>Max Drawdown</h4><h2>{metrics.get('Max Drawdown %', 0):.2%}</h2></div>", unsafe_allow_html=True)
                    with col6: st.markdown(f"<div class='metric-card'><h4>Trade Events</h4><h2>{metrics.get('Trade Events', 0)}</h2></div>", unsafe_allow_html=True)

                st.plotly_chart(plot_comprehensive_performance(results['daily_data'], "Pragati System Performance"), use_container_width=True)
                
                st.header("📊 Complete Performance Metrics")
                metrics_df = pd.DataFrame([metrics]).transpose()
                metrics_df.columns = ['Value']
                
                formatted_values = []
                for metric, value in metrics_df['Value'].items():
                    if pd.isna(value):
                        formatted_values.append('N/A')
                        continue
                    if '%' in metric:
                        formatted_values.append(f'{value:.2%}')
                    elif 'P&L' in metric or 'Investment' in metric or 'Value' in metric:
                        formatted_values.append(f'₹{value:,.0f}')
                    elif 'Ratio' in metric:
                         formatted_values.append(f'{value:.2f}')
                    elif 'Events' in metric or 'Days' in metric:
                         formatted_values.append(f'{int(value):,}')
                    elif isinstance(value, float):
                        formatted_values.append(f'{value:.2f}')
                    else:
                         formatted_values.append(value)
                
                metrics_df['Value'] = formatted_values
                st.dataframe(metrics_df, use_container_width=True)

            else: # Individual Strategy Results
                st.header("📊 Individual Strategy Performance Leaderboard")
                leaderboard_data = {name: res['metrics'] for name, res in results.items() if res.get('metrics')}
                if not leaderboard_data: st.warning("No strategies generated any valid results."); return
                leaderboard_df = pd.DataFrame(leaderboard_data).transpose().sort_values("Sharpe Ratio", ascending=False)
                
                formatter = {
                    'Total Return %': '{:.2%}', 'Annualized Return %': '{:.2%}', 'Absolute P&L': '₹{:,.0f}',
                    'CAGR %': '{:.2%}', 'Volatility % (Ann.)': '{:.2%}', 'Max Drawdown %': '{:.2%}',
                    'Sharpe Ratio': '{:.2f}', 'Sortino Ratio': '{:.2f}', 'Calmar Ratio': '{:.2f}', 
                    'Win Rate %': '{:.2%}', 'Best Day %': '{:.2%}', 'Worst Day %': '{:.2%}', 
                    'Trading Days': '{:,.0f}', 'Total Investment': '₹{:,.0f}', 'Buy Events': '{:,.0f}', 
                    'Trade Events': '{:,.0f}', 'Current Value': '₹{:,.0f}'
                }
                
                st.dataframe(leaderboard_df.style.format(formatter).background_gradient(subset=['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Total Return %'], cmap='RdYlGn'), use_container_width=True)
                st.header("🔬 Strategy Deep Dive"); selected_strategy = st.selectbox("Select Strategy", options=leaderboard_df.index)
                strategy_data = results[selected_strategy]; st.plotly_chart(plot_comprehensive_performance(strategy_data['daily_data'], f"Performance: {selected_strategy}"), use_container_width=True)
                with st.expander("View All Metrics for Selected Strategy"): 
                    st.json({k: (f"{v:.2%}" if "%" in k else f"{v:.2f}") for k, v in strategy_data['metrics'].items()})

    elif analysis_mode == "Lookback Sensitivity Analysis":
        st.sidebar.header("🔬 Analysis Configuration")
        sensitivity_analysis_mode = st.sidebar.radio("Select Sensitivity Mode", ["A/D Ratio Threshold", "Target Analysis Date"])
        selected_style = st.sidebar.selectbox("1. Select Investment Style", options=list(PORTFOLIO_STYLES.keys()))
        all_strategies = {cls.__name__: cls() for cls in pragati.BaseStrategy.__subclasses__()}

        analysis_dates = []
        static_strategies_to_test = None

        if sensitivity_analysis_mode == "Target Analysis Date":
            analysis_date_str = st.sidebar.selectbox("Select Analysis Date", available_dates_str, help="The date on which to generate the portfolio.")
            analysis_dates = [analysis_date_str]
            selected_mix = st.sidebar.selectbox("2. Select Market Mix", options=list(PORTFOLIO_STYLES[selected_style]["mixes"].keys()))
            mix_info = PORTFOLIO_STYLES[selected_style]["mixes"][selected_mix]
            # Create new instances for the test
            static_strategies_to_test = {name: all_strategies[name].__class__() for name in mix_info['strategies']}
        else: # A/D Ratio Threshold
            st.sidebar.markdown("### A/D Ratio Trigger"); ad_file = st.sidebar.file_uploader("Upload A/D Ratio File", type=['xlsx', 'xls']); ad_threshold = st.sidebar.number_input("A/D Ratio Threshold", value=0.7, help="Find dates where A/D Ratio was BELOW this value.")
            if ad_file:
                ad_df = pd.read_excel(ad_file)
                if 'AD_RATIO' in ad_df.columns and 'DATE' in ad_df.columns:
                    ad_df['DATE_STR'] = pd.to_datetime(ad_df['DATE'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
                    available_dates_set = {d for d in available_dates_str}
                    target_dates_df = ad_df[ad_df['AD_RATIO'] < ad_threshold]
                    analysis_dates = sorted([d for d in target_dates_df['DATE_STR'].tolist() if d in available_dates_set])
                    st.sidebar.success(f"Found {len(analysis_dates)} matching historical dates.")
                else: st.sidebar.error("File must contain 'AD_RATIO' and 'DATE' columns.")
        
        st.sidebar.markdown("### Lookback Range"); min_lookback = st.sidebar.number_input("Min Lookback", 10, 100, 20, 5); max_lookback = st.sidebar.number_input("Max Lookback", min_lookback + 5, 200, 100, 5); step = st.sidebar.number_input("Step", 1, 20, 5)
        metric_to_optimize = st.sidebar.selectbox("Performance Metric", ['sharpe', 'calmar', 'annual_return', 'sortino', 'portfolio_return'])
        
        if st.sidebar.button("🚀 Run Sensitivity Analysis", use_container_width=True, type="primary", disabled=(not analysis_dates)):
            lookback_range = range(min_lookback, max_lookback + 1, step)
            latest_prices_df = all_historical_data[-1][1] if all_historical_data else None
            results = run_lookback_sensitivity_analysis(
                full_historical_data=all_historical_data,
                lookback_range=lookback_range,
                analysis_dates=analysis_dates,
                performance_metric=metric_to_optimize,
                historical_data_map=historical_data_map,
                latest_prices_df=latest_prices_df,
                selected_style=selected_style,
                analysis_mode=sensitivity_analysis_mode,
                static_strategies_to_test=static_strategies_to_test
            )
            st.session_state.sensitivity_results = results
            st.session_state.metric_name = metric_to_optimize
            st.success("Sensitivity analysis complete!")
        
        if 'sensitivity_results' in st.session_state:
            st.header("Sensitivity Analysis Results")
            plot_sensitivity_results(st.session_state.sensitivity_results, st.session_state.metric_name)
            with st.expander("View Raw Data"): st.dataframe(st.session_state.sensitivity_results)

if __name__ == "__main__":
    main()
