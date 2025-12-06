import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import datetime
import logging
from typing import List, Dict, Optional

# --- CONFIGURATION ---

# 1. Symbols fetched from Stooq (Existing)
MACRO_SYMBOLS_STOOQ = {
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

# 2. Symbols fetched from Yahoo Finance (New)
MACRO_SYMBOLS_YF = {
    "SGDINR": "SGDINR=X",
    "USDINR": "USDINR=X",
    "GBPINR": "GBPINR=X",
    "EURINR": "EURINR=X",
    "JPYINR": "JPYINR=X",
    "GOLD": "GC=F",       # Gold Futures
    "SILVER": "SI=F",     # Silver Futures
    "USOIL": "CL=F",      # WTI Crude Oil
    "UKOIL": "BZ=F",      # Brent Crude Oil
    "DXY": "DX-Y.NYB"     # US Dollar Index
}

# Combine all for lookup logic
ALL_MACRO_SYMBOLS = {**MACRO_SYMBOLS_STOOQ, **MACRO_SYMBOLS_YF}

# Global cache to prevent re-fetching macro data
_MACRO_DATA_CACHE = None
_MACRO_CACHE_KEY = None

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

def calculate_rsi_series(series, length=14):
    """Calculates RSI for a pandas Series (e.g., an Oscillator)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=length, min_periods=1).mean()
    avg_loss = loss.rolling(window=length, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# --- DATA FETCHING ---

def fetch_macro_data_cached(start_date, end_date):
    """Fetches macro data from Stooq AND Yahoo Finance, merges, and caches it."""
    global _MACRO_DATA_CACHE, _MACRO_CACHE_KEY
    
    # Cache key based on date range
    cache_key = f"{start_date}_{end_date}"
    
    if _MACRO_DATA_CACHE is not None and _MACRO_CACHE_KEY == cache_key:
        return _MACRO_DATA_CACHE

    logging.info(f"Fetching Macro Data ({start_date} to {end_date})...")
    
    combined_macro_df = pd.DataFrame()

    try:
        # --- 1. Fetch from Stooq ---
        stooq_tickers = list(MACRO_SYMBOLS_STOOQ.values())
        if stooq_tickers:
            stooq_df = web.DataReader(stooq_tickers, "stooq", start=start_date, end=end_date)
            
            if isinstance(stooq_df.columns, pd.MultiIndex):
                if 'Close' in stooq_df.columns.get_level_values(0):
                    stooq_df = stooq_df['Close']
                elif 'Value' in stooq_df.columns.get_level_values(0):
                    stooq_df = stooq_df['Value']
            
            # Ensure index is timezone-naive
            if stooq_df.index.tz is not None:
                stooq_df.index = stooq_df.index.tz_localize(None)
            
            combined_macro_df = stooq_df.sort_index()

        # --- 2. Fetch from Yahoo Finance ---
        yf_tickers = list(MACRO_SYMBOLS_YF.values())
        if yf_tickers:
            # Add 1 day buffer for yfinance inclusive/exclusive handling
            yf_end = end_date + datetime.timedelta(days=1) if isinstance(end_date, datetime.date) else end_date
            
            yf_data = yf.download(yf_tickers, start=start_date, end=yf_end, progress=False, auto_adjust=False)
            
            if not yf_data.empty:
                # Extract Close prices
                if 'Close' in yf_data.columns:
                     yf_close = yf_data['Close']
                else:
                     yf_close = yf_data # Fallback if single series

                # Ensure index is timezone-naive
                if yf_close.index.tz is not None:
                    yf_close.index = yf_close.index.tz_localize(None)
                
                yf_close = yf_close.sort_index()

                # --- 3. Merge ---
                # Join Stooq and YF data on Date index
                if combined_macro_df.empty:
                    combined_macro_df = yf_close
                else:
                    combined_macro_df = combined_macro_df.join(yf_close, how='outer')

        # Forward fill to handle different holidays/trading hours
        combined_macro_df = combined_macro_df.sort_index().ffill()
        
        if combined_macro_df.empty:
            logging.warning("⚠️ Macro data fetch returned EMPTY results. MMR will be 0.")
        else:
            logging.info(f"✅ Macro Data Fetched: {combined_macro_df.shape[0]} rows, {combined_macro_df.shape[1]} cols")

        _MACRO_DATA_CACHE = combined_macro_df
        _MACRO_CACHE_KEY = cache_key
        return combined_macro_df

    except Exception as e:
        logging.error(f"❌ Error fetching macro data: {e}")
        return pd.DataFrame()

def fetch_data_for_booster(target_ticker, analysis_date, days_back=200):
    """Fetches Target + Cached Macro Data and joins them."""
    # Convert analysis_date to date object if it's datetime
    if isinstance(analysis_date, datetime.datetime):
        end_date_date = analysis_date.date()
    else:
        end_date_date = analysis_date

    start_date = end_date_date - datetime.timedelta(days=days_back + 150)
    
    try:
        # 1. Fetch Target (yfinance)
        ticker_to_fetch = f"{target_ticker}.NS" if not (target_ticker.endswith('.NS') or target_ticker.endswith('.BO')) else target_ticker
        
        # Note: yf.download end date is exclusive, so add 1 day to include analysis_date
        yf_end = end_date_date + datetime.timedelta(days=1)
        
        target_df = yf.download(ticker_to_fetch, start=start_date, end=yf_end, progress=False, auto_adjust=False)
        
        if target_df.empty:
            # Fallback
            target_df = yf.download(target_ticker, start=start_date, end=yf_end, progress=False, auto_adjust=False)
        
        if target_df.empty:
            logging.debug(f"No data found for {target_ticker}")
            return None

        if isinstance(target_df.columns, pd.MultiIndex):
            target_df.columns = target_df.columns.get_level_values(0)
            
        target_df = target_df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
        
        # 2. Get Macro Data
        macro_df = fetch_macro_data_cached(start_date, end_date_date)
        
        # 3. Join
        if target_df.index.tz is not None:
            target_df.index = target_df.index.tz_localize(None)
        
        if not macro_df.empty:
            # Ensure macro is same timezone (naive) before join
             if macro_df.index.tz is not None:
                 macro_df.index = macro_df.index.tz_localize(None)
             
             # Join left to keep target dates
             combined = target_df.join(macro_df, how='left').ffill()
             return combined
        else:
             return target_df # Proceed without macro (MMR will be 0)

    except Exception as e:
        logging.error(f"Booster Data Fetch Error for {target_ticker}: {e}")
        return None

# --- INDICATOR CALCULATIONS ---

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
    
    return msf_signal, micro_norm

def calculate_mmr(df, length=20, num_vars=5):
    # Use ALL_MACRO_SYMBOLS to find relevant columns
    macro_cols = [c for c in df.columns if c in ALL_MACRO_SYMBOLS.values()]
    target = df['Close']
    
    if not macro_cols:
        return pd.Series(0, index=df.index), 0
    
    correlations = df[macro_cols].corrwith(target).abs().sort_values(ascending=False)
    top_drivers = correlations.head(num_vars).index.tolist()
    
    preds = []
    r2_sum = 0
    y_mean = target.rolling(length).mean()
    y_std = target.rolling(length).std()
    
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
    Calculates the Unified Oscillator and checks for the 3 Tiers of Buy Signals.
    """
    length = 20
    roc_len = 14
    regime_sensitivity = 1.5
    base_weight = 0.5
    
    # 1. Component Calculation
    df['MSF'], df['Micro'] = calculate_msf(df, length, roc_len)
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
    
    # --- TIERED BUY SIGNAL LOGIC ---
    
    # 1. Calc Bollinger Bands & RSI on Oscillator (needed for Tier 3)
    bb_len = 20
    bb_mult = 2.0
    bb_basis = df['Unified_Osc'].rolling(bb_len).mean()
    bb_dev = df['Unified_Osc'].rolling(bb_len).std()
    bb_lower = bb_basis - (bb_mult * bb_dev)
    
    # RSI on Oscillator
    rsi_osc = calculate_rsi_series(df['Unified_Osc'], 14)
    
    # 2. Define Signals
    
    # Tier 1: Confirmed Bullish (Lime Circle)
    # Strong Agreement (> 0.3) + Oversold (< -5)
    strong_agreement = agreement > 0.3
    df['Tier1_Signal'] = strong_agreement & (df['Unified_Osc'] < -5)
    
    # Tier 2: Bullish Divergence
    # Oscillator Rising + Price Falling + Oscillator < -5
    osc_rising = df['Unified_Osc'] > df['Unified_Osc'].shift(1)
    price_falling = df['Close'] < df['Close'].shift(1)
    df['Tier2_Signal'] = osc_rising & price_falling & (df['Unified_Osc'] < -5)
    
    # Tier 3: Is Oversold
    # Note: User likely meant 'Oversold' as 'Overbought' is usually a sell signal.
    # Logic: Below lower Bollinger Band AND RSI < 40 (standard Pine Script logic)
    df['Tier3_Signal'] = (df['Unified_Osc'] < bb_lower) & (rsi_osc < 40)
    
    return df

# --- MAIN EXPORT FUNCTION ---

def boost_portfolio_with_unified_signals(
    portfolio_df: pd.DataFrame,
    symbols: List[str],
    boost_multiplier: float = 1.15,
    max_boost_weight: float = 0.15,
    lookback_days: int = 200,
    analysis_date: Optional[datetime.datetime] = None
) -> pd.DataFrame:
    """
    Analyzes each symbol using tiered Unified Market Analysis.
    Tiers:
      1. Confirmed Bullish (Lime Circle): Full Boost
      2. Bullish Divergence: Medium Boost
      3. Oversold: Minor Boost
    """
    
    if portfolio_df.empty:
        return portfolio_df

    # Default to today if no date provided
    if analysis_date is None:
        analysis_date = datetime.datetime.now()
    
    if isinstance(analysis_date, datetime.date) and not isinstance(analysis_date, datetime.datetime):
        analysis_date = datetime.datetime.combine(analysis_date, datetime.datetime.min.time())

    logging.info(f"--- Unified Booster: Analysis for {analysis_date.date()} ---")
    
    boosted_df = portfolio_df.copy()
    boost_updates = [] # Store (index, multiplier_to_apply)

    # Prefetch macro
    end_date = analysis_date.date()
    start_date = end_date - datetime.timedelta(days=lookback_days + 150)
    fetch_macro_data_cached(start_date, end_date)

    for idx, row in boosted_df.iterrows():
        symbol = row['symbol']
        
        # 1. Fetch Data
        df = fetch_data_for_booster(symbol, analysis_date, days_back=lookback_days)
        
        if df is None or df.empty or len(df) < 50:
            continue
            
        # 2. Run Indicator Logic
        try:
            df_analyzed = calculate_unified_signal(df)
            
            # Find closest date
            try:
                target_idx = df_analyzed.index.get_indexer([analysis_date], method='pad')[0]
                latest_row = df_analyzed.iloc[target_idx]
                latest_date_in_df = df_analyzed.index[target_idx]
            except:
                latest_row = df_analyzed.iloc[-1]
                latest_date_in_df = df_analyzed.index[-1]

            days_diff = (analysis_date - latest_date_in_df).days
            if days_diff > 5:
                logging.warning(f"   [{symbol}] Data outdated by {days_diff} days. Skipping.")
                continue

            # 3. Check Tiers (Hierarchy)
            tier1 = latest_row['Tier1_Signal']
            tier2 = latest_row['Tier2_Signal']
            tier3 = latest_row['Tier3_Signal']
            
            osc_val = latest_row['Unified_Osc']

            applied_mult = 1.0
            trigger_reason = ""
            
            # Scale the boost factor based on multiplier input
            # Example: If boost=1.15 (15% increase)
            # Tier 1: 15% increase
            # Tier 2: 10% increase (approx 2/3)
            # Tier 3: 5% increase (approx 1/3)
            
            boost_delta = boost_multiplier - 1.0 # e.g. 0.15
            
            if tier1:
                # Full Boost
                applied_mult = boost_multiplier
                trigger_reason = "TIER 1 (Confirmed Bullish)"
            elif tier2:
                # Medium Boost (66% of delta)
                applied_mult = 1.0 + (boost_delta * 0.66)
                trigger_reason = "TIER 2 (Bullish Divergence)"
            elif tier3:
                # Minor Boost (33% of delta)
                applied_mult = 1.0 + (boost_delta * 0.33)
                trigger_reason = "TIER 3 (Oversold)"
            
            # Debug Log for near-misses or hits
            if applied_mult > 1.0:
                logging.info(f"🚀 {symbol}: {trigger_reason} on {latest_date_in_df.date()} | Osc: {osc_val:.2f} | Boost: {applied_mult:.3f}x")
                boost_updates.append((idx, applied_mult))
            elif osc_val < -2.0:
                 logging.info(f"   [{symbol}] No Signal | Osc: {osc_val:.2f} (Tier1:{tier1} Tier2:{tier2} Tier3:{tier3})")

        except Exception as e:
            logging.error(f"Error calculating booster for {symbol}: {e}")
            continue

    # 4. Apply Boost
    if boost_updates:
        original_total_weight = boosted_df['weightage_pct'].sum()
        
        for idx, mult in boost_updates:
            current_weight = boosted_df.at[idx, 'weightage_pct']
            new_weight = current_weight * mult
            
            # Cap at max_boost_weight
            max_pct = max_boost_weight * 100
            if new_weight > max_pct:
                new_weight = max_pct
                
            boosted_df.at[idx, 'weightage_pct'] = new_weight
            
        # 5. Re-normalize
        new_total_weight = boosted_df['weightage_pct'].sum()
        if new_total_weight > 0:
            boosted_df['weightage_pct'] = (boosted_df['weightage_pct'] / new_total_weight) * original_total_weight
        
        # Recalculate units/value
        if 'price' in boosted_df.columns and 'value' in boosted_df.columns:
            total_value = boosted_df['value'].sum()
            boosted_df['units'] = np.floor((total_value * boosted_df['weightage_pct'] / 100) / boosted_df['price'])
            boosted_df['value'] = boosted_df['units'] * boosted_df['price']

        logging.info(f"--- Booster Applied to {len(boost_updates)} positions ---")
    else:
        logging.info("--- No Boost Signals Detected ---")

    return boosted_df
