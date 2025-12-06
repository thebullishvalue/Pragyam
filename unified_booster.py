import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import datetime
import logging
from typing import List, Dict, Optional

# --- CONFIGURATION ---
MACRO_SYMBOLS = {
    "India 10Y": "10YINY.B",
    "India 02Y": "2YINY.B",
    "US 30Y": "30YUSY.B",
    "US 10Y": "10YUSY.B",
    "US 05Y": "5YUSY.B",
    "US 02Y": "2YUSY.B",
    "UK 30Y": "30YUKY.B",
    "UK 10Y": "10YUKY.B",
    "UK 05Y": "5YUKY.B",
    "UK 02Y": "2YUKY.B",
    "EU (DE) 30Y": "30YDEY.B",
    "EU (DE) 10Y": "10YDEY.B",
    "EU (DE) 05Y": "5YDEY.B",
    "EU (DE) 02Y": "2YDEY.B",
    "China 10Y": "10YCNY.B",
    "China 02Y": "2YCNY.B",
    "Japan 30Y": "30YJPY.B",
    "Japan 10Y": "10YJPY.B",
    "Japan 02Y": "2YJPY.B",
    "Singapore 10Y": "10YSGY.B",
}

# Global cache to prevent re-fetching macro data for every symbol in the portfolio
_MACRO_DATA_CACHE = None
_MACRO_CACHE_DATE = None

# --- UTILITY FUNCTIONS ---

def sigmoid(x, scale=1.0):
    """Sigmoid transformation bounded [-1, 1]"""
    return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0

def zscore_clipped(series, window, clip=3.0):
    """Rolling Z-Score with clipping"""
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0)

def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

# --- DATA FETCHING ---

def fetch_macro_data_cached(start_date, end_date):
    """Fetches macro data once and caches it for the session/run."""
    global _MACRO_DATA_CACHE, _MACRO_CACHE_DATE
    
    # Simple cache validity check (same day)
    today_str = datetime.date.today().isoformat()
    if _MACRO_DATA_CACHE is not None and _MACRO_CACHE_DATE == today_str:
        return _MACRO_DATA_CACHE

    logging.info("Fetching Macro Data from Stooq...")
    try:
        macro_tickers = list(MACRO_SYMBOLS.values())
        # Stooq often fails if we request too many at once or too frequently, 
        # so we fetch carefully.
        macro_df = web.DataReader(macro_tickers, "stooq", start=start_date, end=end_date)
        
        if isinstance(macro_df.columns, pd.MultiIndex):
            if 'Close' in macro_df.columns.get_level_values(0):
                macro_df = macro_df['Close']
            elif 'Value' in macro_df.columns.get_level_values(0):
                macro_df = macro_df['Value']
        
        # Ensure it's sorted by date ascending
        macro_df = macro_df.sort_index()
        
        _MACRO_DATA_CACHE = macro_df
        _MACRO_CACHE_DATE = today_str
        return macro_df
    except Exception as e:
        logging.error(f"Error fetching macro data: {e}")
        return pd.DataFrame()

def fetch_data_for_booster(target_ticker, days_back=200):
    """Fetches Target + Cached Macro Data and joins them."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back + 150) # Buffer for rolling windows
    
    try:
        # 1. Fetch Target (yfinance)
        # We need to handle potential .NS suffix issues
        if not target_ticker.endswith('.NS') and not target_ticker.endswith('.BO'):
             # Try appending .NS if not present, assuming NSE for this system context
             ticker_to_fetch = f"{target_ticker}.NS"
        else:
             ticker_to_fetch = target_ticker

        target_df = yf.download(ticker_to_fetch, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if target_df.empty:
            # Fallback: try without .NS if it failed
            target_df = yf.download(target_ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if target_df.empty:
            return None

        # Flatten columns if multi-index
        if isinstance(target_df.columns, pd.MultiIndex):
            target_df.columns = target_df.columns.get_level_values(0)
            
        # Standardize columns
        target_df = target_df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
        
        # 2. Get Macro Data (Cached)
        macro_df = fetch_macro_data_cached(start_date, end_date)
        
        if macro_df.empty:
             return target_df # Return just target if macro fails

        # 3. Join
        # Ensure timezone naive
        if target_df.index.tz is not None:
            target_df.index = target_df.index.tz_localize(None)
        if macro_df.index.tz is not None:
             macro_df.index = macro_df.index.tz_localize(None)

        combined = target_df.join(macro_df, how='left').ffill()
        return combined

    except Exception as e:
        logging.error(f"Booster Data Fetch Error for {target_ticker}: {e}")
        return None

# --- INDICATOR CALCULATIONS (From indicator.py) ---

def calculate_msf(df, length=20, roc_len=14, clip=3.0):
    close = df['Close']
    
    # 1. Momentum
    roc_raw = close.pct_change(roc_len)
    roc_z = zscore_clipped(roc_raw, length, clip)
    momentum_norm = sigmoid(roc_z, 1.5)
    
    # 2. Microstructure
    intrabar_dir = (df['High'] + df['Low']) / 2 - df['Open']
    vol_ma = df['Volume'].rolling(length).mean()
    vol_ratio = (df['Volume'] / vol_ma).fillna(1.0)
    
    vw_direction = (intrabar_dir * vol_ratio).rolling(length).mean()
    price_change_imp = close.diff(5)
    vw_impact = (price_change_imp * vol_ratio).rolling(length).mean()
    
    micro_raw = vw_direction - vw_impact
    micro_z = zscore_clipped(micro_raw, length, clip)
    micro_norm = sigmoid(micro_z, 1.5)
    
    # 3. Composite Trend
    trend_fast = close.rolling(5).mean()
    trend_slow = close.rolling(length).mean()
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, length, clip)
    
    mom_accel_raw = close.diff(5).diff(5)
    mom_accel_z = zscore_clipped(mom_accel_raw, length, clip)
    
    atr = calculate_atr(df, 14)
    vol_adj_mom_raw = close.diff(5) / atr
    vol_adj_mom_z = zscore_clipped(vol_adj_mom_raw, length, clip)
    
    mean_rev_z = zscore_clipped(close - trend_slow, length, clip)
    
    composite_trend_z = (trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z) / np.sqrt(4.0)
    composite_trend_norm = sigmoid(composite_trend_z, 1.5)
    
    # 4. Flow
    typical_price = (df['High'] + df['Low'] + close) / 3
    mf = typical_price * df['Volume']
    mf_pos = np.where(close > close.shift(1), mf, 0)
    mf_neg = np.where(close < close.shift(1), mf, 0)
    
    mf_pos_smooth = pd.Series(mf_pos, index=df.index).rolling(length).mean()
    mf_neg_smooth = pd.Series(mf_neg, index=df.index).rolling(length).mean()
    mf_total = mf_pos_smooth + mf_neg_smooth
    
    accum_ratio = mf_pos_smooth / mf_total.replace(0, np.nan)
    accum_ratio = accum_ratio.fillna(0.5)
    accum_norm = 2.0 * (accum_ratio - 0.5)
    
    # 5. Regime
    pct_change = close.pct_change()
    threshold = 0.0033
    regime_signals = np.select([pct_change > threshold, pct_change < -threshold], [1, -1], default=0)
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length).mean()
    regime_z = zscore_clipped(regime_raw, length, clip)
    regime_norm = sigmoid(regime_z, 1.5)
    
    # Final MSF
    osc_momentum = momentum_norm
    osc_structure = (micro_norm + composite_trend_norm) / np.sqrt(2.0)
    osc_flow = (accum_norm + regime_norm) / np.sqrt(2.0)
    
    msf_raw = (osc_momentum + osc_structure + osc_flow) / np.sqrt(3.0)
    msf_signal = sigmoid(msf_raw * np.sqrt(3.0), 1.0)
    
    return msf_signal

def calculate_mmr(df, length=20, num_vars=5):
    macro_cols = [c for c in df.columns if c in MACRO_SYMBOLS.values()]
    target = df['Close']
    
    if not macro_cols:
        # If no macro data available, return neutral
        return pd.Series(0, index=df.index), 0
    
    correlations = df[macro_cols].corrwith(target).abs().sort_values(ascending=False)
    top_drivers = correlations.head(num_vars).index.tolist()
    
    preds = []
    r2_sum = 0
    y_mean = target.rolling(length).mean()
    y_std = target.rolling(length).std()
    
    # Vectorized loop over drivers
    for ticker in top_drivers:
        x = df[ticker]
        x_mean = x.rolling(length).mean()
        x_std = x.rolling(length).std()
        roll_corr = x.rolling(length).corr(target)
        
        # Handle cases where std is 0
        slope = roll_corr * (y_std / x_std)
        intercept = y_mean - (slope * x_mean)
        
        pred = (slope * x) + intercept
        r2 = roll_corr ** 2
        
        preds.append(pred * r2)
        r2_sum += r2

    r2_sum = r2_sum.replace(0, np.nan)
    
    if len(preds) == 0:
         return pd.Series(0, index=df.index), 0

    y_predicted = sum(preds) / r2_sum
    
    deviation = target - y_predicted
    mmr_z = zscore_clipped(deviation, length, 3.0)
    mmr_signal = sigmoid(mmr_z, 1.5)
    
    avg_r2 = sum([r**2 for r in [df[t].rolling(length).corr(target) for t in top_drivers]]) / r2_sum
    mmr_quality = np.sqrt(avg_r2.fillna(0))
    
    return mmr_signal, mmr_quality

def calculate_unified_signal(df):
    """
    Calculates the Unified Oscillator and checks for the Lime Color Circle condition.
    """
    length = 20
    roc_len = 14
    regime_sensitivity = 1.5
    base_weight = 0.5
    
    # 1. Component Calculation
    df['MSF'] = calculate_msf(df, length, roc_len)
    df['MMR'], df['MMR_Quality'] = calculate_mmr(df, length)
    
    # 2. Adaptive Weighting
    msf_clarity = df['MSF'].abs()
    mmr_clarity = df['MMR'].abs()
    msf_clarity_scaled = msf_clarity.pow(regime_sensitivity)
    mmr_clarity_scaled = (mmr_clarity * df['MMR_Quality']).pow(regime_sensitivity)
    clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001
    
    msf_w_adaptive = msf_clarity_scaled / clarity_sum
    mmr_w_adaptive = mmr_clarity_scaled / clarity_sum
    
    msf_w_final = 0.5 * base_weight + 0.5 * msf_w_adaptive
    mmr_w_final = 0.5 * (1.0 - base_weight) + 0.5 * mmr_w_adaptive
    w_sum = msf_w_final + mmr_w_final
    msf_w_norm = msf_w_final / w_sum
    mmr_w_norm = mmr_w_final / w_sum
    
    unified_signal = (msf_w_norm * df['MSF']) + (mmr_w_norm * df['MMR'])
    
    agreement = df['MSF'] * df['MMR'] 
    agree_strength = agreement.abs()
    multiplier = np.where(agreement > 0, 1.0 + 0.2 * agree_strength, 1.0 - 0.1 * agree_strength)
    
    df['Unified'] = (unified_signal * multiplier).clip(-1.0, 1.0)
    df['Unified_Osc'] = df['Unified'] * 10
    
    # 3. Lime Circle Logic (Confirmed Buy)
    # Strong Agreement (> 0.3) + Oversold (< -5)
    strong_agreement = agreement > 0.3
    df['Buy_Signal'] = strong_agreement & (df['Unified_Osc'] < -5)
    
    return df

# --- MAIN EXPORT FUNCTION ---

def boost_portfolio_with_unified_signals(
    portfolio_df: pd.DataFrame,
    symbols: List[str],
    boost_multiplier: float = 1.15,
    max_boost_weight: float = 0.15,
    lookback_days: int = 200
) -> pd.DataFrame:
    """
    Analyzes each symbol in the portfolio using the Unified Market Analysis indicator.
    If a symbol has a 'Lime Circle' (Confirmed Buy) on the analysis date, its weight is boosted.
    
    Note: This fetch logic is separate from the backtest engine because it requires 
    historical context (rolling windows) and external macro data which isn't present 
    in the daily snapshots passed to the main strategies.
    """
    
    if portfolio_df.empty:
        return portfolio_df

    logging.info("--- Starting Unified Booster Analysis ---")
    
    boosted_df = portfolio_df.copy()
    boost_indices = []

    # Prepare cache
    # We fetch macro data once here to cover the required range for all symbols
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=lookback_days + 150)
    fetch_macro_data_cached(start_date, end_date)

    for idx, row in boosted_df.iterrows():
        symbol = row['symbol']
        
        # 1. Fetch Data
        df = fetch_data_for_booster(symbol, days_back=lookback_days)
        
        if df is None or df.empty or len(df) < 50:
            continue
            
        # 2. Run Indicator Logic
        try:
            df_analyzed = calculate_unified_signal(df)
            
            # 3. Check Signal on Latest Date
            # We look at the very last available row (Analysis Date)
            latest_signal = df_analyzed['Buy_Signal'].iloc[-1]
            latest_unified = df_analyzed['Unified_Osc'].iloc[-1]
            
            if latest_signal:
                boost_indices.append(idx)
                logging.info(f"🟢 BOOST TRIGGER: {symbol} | Unified Osc: {latest_unified:.2f} | Condition: Lime Circle")
            
        except Exception as e:
            logging.error(f"Error calculating booster for {symbol}: {e}")
            continue

    # 4. Apply Boost
    if boost_indices:
        original_total_weight = boosted_df['weightage_pct'].sum()
        
        for idx in boost_indices:
            current_weight = boosted_df.at[idx, 'weightage_pct']
            new_weight = current_weight * boost_multiplier
            
            # Cap at max_boost_weight (converted to percentage)
            max_pct = max_boost_weight * 100
            if new_weight > max_pct:
                new_weight = max_pct
                
            boosted_df.at[idx, 'weightage_pct'] = new_weight
            
        # 5. Re-normalize to original total weight (usually 100%)
        # This ensures we don't end up with >100% allocation
        new_total_weight = boosted_df['weightage_pct'].sum()
        boosted_df['weightage_pct'] = (boosted_df['weightage_pct'] / new_total_weight) * original_total_weight
        
        # Recalculate units and value
        if 'price' in boosted_df.columns and 'value' in boosted_df.columns:
            total_value = boosted_df['value'].sum()
            boosted_df['units'] = np.floor((total_value * boosted_df['weightage_pct'] / 100) / boosted_df['price'])
            boosted_df['value'] = boosted_df['units'] * boosted_df['price']

        logging.info(f"--- Booster Applied to {len(boost_indices)} positions ---")
    else:
        logging.info("--- No Boost Signals Detected ---")

    return boosted_df
