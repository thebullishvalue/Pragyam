import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import zipfile
import shutil
from typing import List, Tuple, Dict, Any
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Suppress yfinance warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration: Macro Symbols (Unified Oscillator) ---
MACRO_SYMBOLS_STOOQ = {
    "India 10Y": "10YINY.B", "India 02Y": "2YINY.B",
    "US 30Y": "30YUSY.B", "US 10Y": "10YUSY.B", "US 05Y": "5YUSY.B", "US 02Y": "2YUSY.B",
    "UK 30Y": "30YUKY.B", "UK 10Y": "10YUKY.B", "UK 05Y": "5YUKY.B", "UK 02Y": "2YUKY.B",
    "EU (DE) 30Y": "30YDEY.B", "EU (DE) 10Y": "10YDEY.B", "EU (DE) 05Y": "5YDEY.B", "EU (DE) 02Y": "2YDEY.B",
    "China 10Y": "10YCNY.B", "China 02Y": "2YCNY.B",
    "Japan 30Y": "30YJPY.B", "Japan 10Y": "10YJPY.B", "Japan 02Y": "2YJPY.B",
    "Singapore 10Y": "10YSGY.B",
}

MACRO_SYMBOLS_YF = {
    "SGDINR": "SGDINR=X", "USDINR": "USDINR=X", "GBPINR": "GBPINR=X",
    "EURINR": "EURINR=X", "JPYINR": "JPYINR=X",
    "GOLD": "GC=F", "SILVER": "SI=F", "USOIL": "CL=F", "UKOIL": "BZ=F", "DXY": "DX-Y.NYB"
}

ALL_MACRO_SYMBOLS = {**MACRO_SYMBOLS_STOOQ, **MACRO_SYMBOLS_YF}

# --- Helper Math Functions (Unified Oscillator) ---
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
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

# --- Macro Data Fetching (Optimized) ---
def fetch_macro_data(start_date, end_date):
    """Fetches and merges macro data once for the session."""
    logging.info(f"Fetching Macro Data ({start_date} to {end_date})...")
    combined_macro_df = pd.DataFrame()
    
    try:
        # 1. Stooq
        stooq_tickers = list(MACRO_SYMBOLS_STOOQ.values())
        if stooq_tickers:
            try:
                stooq_df = web.DataReader(stooq_tickers, "stooq", start=start_date, end=end_date)
                if isinstance(stooq_df.columns, pd.MultiIndex):
                    if 'Close' in stooq_df.columns.get_level_values(0):
                        stooq_df = stooq_df['Close']
                    elif 'Value' in stooq_df.columns.get_level_values(0):
                        stooq_df = stooq_df['Value']
                # Ensure timezone-naive
                if stooq_df.index.tz is not None:
                    stooq_df.index = stooq_df.index.tz_localize(None)
                combined_macro_df = stooq_df.sort_index()
            except Exception as e:
                logging.warning(f"Stooq fetch failed: {e}")

        # 2. Yahoo Finance
        yf_tickers = list(MACRO_SYMBOLS_YF.values())
        if yf_tickers:
            yf_end = end_date + timedelta(days=1)
            try:
                yf_data = yf.download(yf_tickers, start=start_date, end=yf_end, progress=False, auto_adjust=False)
                if not yf_data.empty:
                    yf_close = yf_data['Close'] if 'Close' in yf_data.columns else yf_data
                    if yf_close.index.tz is not None:
                        yf_close.index = yf_close.index.tz_localize(None)
                    yf_close = yf_close.sort_index()
                    
                    if combined_macro_df.empty:
                        combined_macro_df = yf_close
                    else:
                        combined_macro_df = combined_macro_df.join(yf_close, how='outer')
            except Exception as e:
                logging.warning(f"YF Macro fetch failed: {e}")

        # Forward fill and clean
        combined_macro_df = combined_macro_df.sort_index().ffill()
        return combined_macro_df

    except Exception as e:
        logging.error(f"Critical error fetching macro data: {e}")
        return pd.DataFrame()

# --- REPLACEMENT: Unified Oscillator Class ---
class UnifiedOscillator:
    """
    Calculates the 'Unified Oscillator' (MSF + MMR) optimized for backdata.
    Replaces the legacy LiquidityOscillator.
    """
    def __init__(self, macro_df: pd.DataFrame = None, length: int = 20):
        self.length = length
        self.macro_df = macro_df
        # If macro_df is provided, ensure index is timezone naive for joining
        if self.macro_df is not None and not self.macro_df.empty:
            if self.macro_df.index.tz is not None:
                self.macro_df.index = self.macro_df.index.tz_localize(None)

    def calculate_msf(self, df: pd.DataFrame, length=20, roc_len=14, clip=3.0):
        # Expects lower case columns from backdata pipeline
        close = df['close']
        
        # 1. Momentum
        roc_raw = close.pct_change(roc_len)
        roc_z = zscore_clipped(roc_raw, length, clip)
        momentum_norm = sigmoid(roc_z, 1.5)
        
        # 2. Microstructure
        intrabar_dir = (df['high'] + df['low']) / 2 - df['open']
        vol_ma = df['volume'].rolling(length).mean()
        vol_ratio = (df['volume'] / vol_ma).fillna(1.0)
        
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
        typical_price = (df['high'] + df['low'] + close) / 3
        mf = typical_price * df['volume']
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
        return sigmoid(msf_raw * np.sqrt(3.0), 1.0)

    def calculate_mmr(self, df: pd.DataFrame, length=20, num_vars=5):
        # Uses columns joined from macro_df
        if self.macro_df is None or self.macro_df.empty:
            return pd.Series(0, index=df.index), 0.0

        target = df['close']
        # Identify macro columns present in the joined df
        macro_cols = [c for c in df.columns if c in ALL_MACRO_SYMBOLS.values()]
        
        if not macro_cols:
            return pd.Series(0, index=df.index), 0.0
        
        # Performance Opt: Calculate correlation on the whole series at once to find top drivers
        correlations = df[macro_cols].corrwith(target).abs().sort_values(ascending=False)
        top_drivers = correlations.head(num_vars).index.tolist()
        
        preds = []
        r2_sum = 0
        y_mean = target.rolling(length).mean()
        y_std = target.rolling(length).std()
        
        # Vectorized rolling regression prediction
        for ticker in top_drivers:
            x = df[ticker]
            x_mean = x.rolling(length).mean()
            x_std = x.rolling(length).std()
            roll_corr = x.rolling(length).corr(target)
            
            slope = roll_corr * (y_std / x_std)
            intercept = y_mean - (slope * x_mean)
            
            pred = (slope * x) + intercept
            r2 = roll_corr ** 2
            
            preds.append(pred * r2)
            r2_sum += r2

        if r2_sum is 0 or len(preds) == 0:
             return pd.Series(0, index=df.index), 0.0
             
        r2_sum = r2_sum.replace(0, np.nan)
        y_predicted = sum(preds) / r2_sum
        
        deviation = target - y_predicted
        mmr_z = zscore_clipped(deviation, length, 3.0)
        mmr_signal = sigmoid(mmr_z, 1.5)
        
        # Avg R2 for quality
        avg_r2 = sum([r**2 for r in [df[t].rolling(length).corr(target) for t in top_drivers]]) / r2_sum
        mmr_quality = np.sqrt(avg_r2.fillna(0))
        
        return mmr_signal, mmr_quality

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Main calculation method.
        Joins macro data -> Calculates MSF & MMR -> Returns Unified Oscillator Series.
        """
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(data.columns):
            return pd.Series(dtype=float)

        df = data.copy()
        
        # Ensure TZ-naive index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Join with pre-fetched macro data if available
        if self.macro_df is not None and not self.macro_df.empty:
            df = df.join(self.macro_df, how='left').ffill()

        # 1. MSF
        msf = self.calculate_msf(df, self.length)
        
        # 2. MMR
        mmr, mmr_quality = self.calculate_mmr(df, self.length)
        
        # 3. Unified Combination
        regime_sensitivity = 1.5
        base_weight = 0.5
        
        msf_clarity = msf.abs().pow(regime_sensitivity)
        mmr_clarity = (mmr.abs() * mmr_quality).pow(regime_sensitivity)
        clarity_sum = msf_clarity + mmr_clarity + 0.001
        
        msf_w = (0.5 * base_weight) + (0.5 * (msf_clarity / clarity_sum))
        mmr_w = (0.5 * (1.0 - base_weight)) + (0.5 * (mmr_clarity / clarity_sum))
        
        w_sum = msf_w + mmr_w
        unified_signal = (msf_w / w_sum * msf) + (mmr_w / w_sum * mmr)
        
        agreement = msf * mmr
        multiplier = np.where(agreement > 0, 1.0 + 0.2 * agreement.abs(), 1.0 - 0.1 * agreement.abs())
        
        # 4. Final Output (Scaled to -10 to 10 range as per booster)
        unified_osc = (unified_signal * multiplier).clip(-1.0, 1.0) * 10
        
        return unified_osc.rename('unified_oscillator')

# --- Helper & Data Fetching Functions (Unchanged) ---
def resample_data(df, rule='W-FRI'):
    """Resamples daily OHLCV data to a different timeframe."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    logic = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return df.resample(rule).apply(logic).dropna()

def calculate_rsi(data, period=14):
    """Calculates Relative Strength Index (RSI)."""
    if data.empty or 'close' not in data.columns or len(data) < period:
        return pd.Series(index=data.index, dtype=float)
    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi[avg_loss == 0] = 100.0
    return rsi

# --- Core Logic for Pre-Calculating All Indicators ---
def calculate_all_indicators(symbol_data, oscillator_calculator):
    """
    Calculates all indicators for a single symbol's *entire* history
    and returns a single DataFrame.
    """
    daily_data = symbol_data.copy()
    if daily_data.empty:
        return None

    weekly_data = resample_data(daily_data, 'W-FRI')
    
    all_results_df = pd.DataFrame(index=daily_data.index)
    all_results_df['price'] = daily_data['close']
    all_results_df['% change'] = daily_data['close'].pct_change()

    timeframes = {'latest': daily_data, 'weekly': weekly_data}
    
    for tf_name, df in timeframes.items():
        if len(df) < 2:
            continue
        
        # Use the passed UnifiedOscillator
        osc = oscillator_calculator.calculate(df)
        
        if not osc.dropna().empty:
            all_results_df[f'osc {tf_name}'] = osc
            all_results_df[f'9ema osc {tf_name}'] = osc.ewm(span=9).mean()
            all_results_df[f'21ema osc {tf_name}'] = osc.ewm(span=21).mean()

            if len(osc.dropna()) >= 20:
                osc_sma20 = osc.rolling(window=20).mean()
                osc_std20 = osc.rolling(window=20).std()
                safe_std20 = osc_std20.replace(0, pd.NA)
                all_results_df[f'zscore {tf_name}'] = (osc - osc_sma20) / safe_std20

        rsi_series = calculate_rsi(df)
        if rsi_series is not None and not rsi_series.dropna().empty:
            all_results_df[f'rsi {tf_name}'] = rsi_series

        for period in [20, 90, 200]:
            if len(df) >= period:
                all_results_df[f'ma{period} {tf_name}'] = df['close'].rolling(window=period).mean()
                if period == 20:
                    all_results_df[f'dev{period} {tf_name}'] = df['close'].rolling(window=period).std()

    all_results_df = all_results_df.reindex(daily_data.index)
    
    weekly_cols = [col for col in all_results_df.columns if 'weekly' in col]
    all_results_df[weekly_cols] = all_results_df[weekly_cols].ffill()
    
    return all_results_df

# --- Load symbols from file ---
def load_symbols_from_file(filepath: str = "symbols.txt") -> List[str]:
    """Loads a list of symbols from a text file."""
    if not os.path.exists(filepath):
        logging.error(f"Symbol file not found at: {filepath}")
        try:
            st.error(f"Symbol file not found: {filepath}")
        except Exception:
            pass
        return []
    
    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        logging.info(f"Loaded {len(symbols)} symbols from {filepath}")
        return symbols
    except Exception as e:
        logging.error(f"Error reading symbol file {filepath}: {e}")
        try:
            st.error(f"Error reading symbol file: {e}")
        except Exception:
            pass
        return []

# Load the fixed universe
SYMBOLS_UNIVERSE = load_symbols_from_file()

COLUMN_ORDER = [
    'date', 'symbol', 'price', 'rsi latest', 'rsi weekly',
    '% change', 'osc latest', 'osc weekly',
    '9ema osc latest', '9ema osc weekly',
    '21ema osc latest', '21ema osc weekly',
    'zscore latest', 'zscore weekly',
    'ma20 latest', 'ma90 latest', 'ma200 latest',
    'ma20 weekly', 'ma90 weekly', 'ma200 weekly',
    'dev20 latest', 'dev20 weekly'
]

INDICATOR_PERIODS = [20, 90, 200]
MAX_INDICATOR_PERIOD = max(INDICATOR_PERIODS)


def generate_historical_data(
    symbols_to_process: List[str], 
    start_date: datetime, 
    end_date: datetime
) -> List[Tuple[datetime, pd.DataFrame]]:
    """
    Generates historical indicator snapshots with the Unified Oscillator.
    """
    
    if not symbols_to_process:
        try:
            st.error("Error: No symbols provided to generate_historical_data.")
        except Exception:
            logging.error("Error: No symbols provided to generate_historical_data.")
        return []
        
    # 1. --- Download Data ---
    try:
        logging.info(f"--- yfinance: Attempting to download {len(symbols_to_process)} symbols...")
        all_data = yf.download(
            symbols_to_process, 
            start=start_date, 
            end=end_date + timedelta(days=1), 
            progress=False
        )
    except Exception as e:
         logging.error(f"yf.download failed: {e}")
         all_data = pd.DataFrame()

    if all_data.empty or 'Close' not in all_data or all_data['Close'].dropna(how='all').empty:
        logging.error("yf.download returned an empty dataframe or all-NaN Close data.")
        try:
            st.error("Could not download any data. Check symbols or date range.")
        except Exception:
            print("ERROR: Could not download any data. Check symbols or date range.")
        return []
    
    # --- Clean up failed tickers ---
    if len(symbols_to_process) > 1:
        valid_tickers = all_data['Close'].dropna(how='all', axis=1).columns
        invalid_tickers = [s for s in symbols_to_process if s not in valid_tickers]
        
        if invalid_tickers:
            warning_msg = f"Failed to download data for: {', '.join(invalid_tickers)}. They will be skipped."
            logging.warning(warning_msg)
            try:
                st.warning(warning_msg)
            except Exception:
                pass
                
            all_data = all_data.loc[:, (slice(None), valid_tickers)]
            symbols_to_process = list(valid_tickers)
            
            if not symbols_to_process:
                logging.error("No valid tickers remaining after download.")
                return []
    
    logging.info(f"yf.download successful. Data shape: {all_data.shape}. Valid tickers: {len(symbols_to_process)}")
    all_data.columns.names = ['Indicator', 'Symbol']

    # 2. --- PRE-FETCH MACRO DATA (Optimization) ---
    # We fetch macro data once here and pass it to the oscillator logic.
    # We need a slightly larger buffer for macro data to ensure rolling windows can calculate at start_date
    macro_start = start_date - timedelta(days=365) 
    macro_df = fetch_macro_data(macro_start, end_date)
    
    # Instantiate the Unified Oscillator with the cached macro data
    oscillator_calculator = UnifiedOscillator(macro_df=macro_df, length=20)
    
    # 3. --- Pre-calculate all indicators for all symbols ---
    ticker_indicator_cache = {}
    
    # Progress bar if running in Streamlit
    try:
        progress = st.progress(0, text="Calculating Unified Oscillators...")
    except:
        progress = None

    total_sym = len(symbols_to_process)
    
    for i, ticker in enumerate(symbols_to_process):
        try:
            if progress: progress.progress((i+1)/total_sym, text=f"Processing {ticker}...")

            if len(symbols_to_process) > 1:
                symbol_df = all_data.xs(ticker, level='Symbol', axis=1).copy()
            else:
                symbol_df = all_data.copy()
                
            symbol_df.columns = [col.lower() for col in symbol_df.columns]
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in symbol_df.columns:
                    symbol_df[col] = pd.to_numeric(symbol_df[col], errors='coerce')
            
            symbol_df = symbol_df.dropna(subset=['close', 'volume'])
            symbol_df.name = ticker
            
            if not symbol_df.empty:
                indicators_df = calculate_all_indicators(symbol_df, oscillator_calculator)
                ticker_indicator_cache[ticker] = indicators_df
                
        except (pd.errors.DataError, KeyError, IndexError) as e:
            logging.warning(f"Skipping {ticker} due to calculation error: {e}")
            continue

    if progress: progress.empty()

    # 4. --- Generate Daily Snapshots in Memory ---
    pragati_data_list: List[Tuple[datetime, pd.DataFrame]] = []
    date_range = all_data.index.normalize().unique()

    for snapshot_date in date_range:
        if snapshot_date < (start_date + timedelta(days=MAX_INDICATOR_PERIOD)) or snapshot_date > end_date:
            continue

        daily_results: List[Dict[str, Any]] = []
        for ticker in symbols_to_process:
            if ticker not in ticker_indicator_cache:
                continue
            
            full_indicator_df = ticker_indicator_cache[ticker]
            
            if snapshot_date not in full_indicator_df.index:
                continue
                
            try:
                indicator_row = full_indicator_df.loc[snapshot_date]
                if indicator_row.isnull().all() or pd.isna(indicator_row.get('price')):
                    continue 

                indicators = indicator_row.to_dict()
                indicators['symbol'] = ticker.replace('.NS', '')
                indicators['date'] = snapshot_date.strftime('%dth %b')
                indicators['% change'] = indicators['% change'] * 100
                
                daily_results.append(indicators)
            except KeyError:
                continue
        
        if daily_results:
            final_df = pd.DataFrame(daily_results)
            for col in COLUMN_ORDER:
                if col not in final_df.columns:
                    final_df[col] = pd.NA
            
            final_df = final_df[COLUMN_ORDER]
            pragati_data_list.append((snapshot_date, final_df))

    return pragati_data_list


# --- Main Application UI and Logic ---
def main():
    st.set_page_config(
        page_title="Unified Snapshot Generator",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("📊 Unified Oscillator Snapshot Generator")
    st.markdown("Replaced Liquidity Oscillator with **Unified Oscillator** (MSF + MMR).")

    with st.sidebar:
        st.header("1. Select Date Range")
        today = datetime.now()
        default_start = today - timedelta(days=MAX_INDICATOR_PERIOD + 90)
        start_date = st.date_input("Start Date", default_start)
        end_date = st.date_input("End Date", today)

        st.header("2. Ticker Universe")
        if SYMBOLS_UNIVERSE:
            st.info(f"Using fixed universe from `symbols.txt` ({len(SYMBOLS_UNIVERSE)} tickers).")
            with st.expander("View Tickers"):
                st.dataframe(SYMBOLS_UNIVERSE, use_container_width=True)
        else:
            st.error("`symbols.txt` not found or is empty. Cannot proceed.")
        
        st.header("3. Generate")
        process_button = st.button("Generate Snapshots", type="primary", use_container_width=True)

    if process_button:
        if start_date > end_date:
            st.error("Error: Start date cannot be after end date.")
        elif not SYMBOLS_UNIVERSE:
            st.error("Error: Cannot generate data. `symbols.txt` is missing or empty.")
        else:
            symbols_to_process = SYMBOLS_UNIVERSE
            
            fetch_start_date = start_date - timedelta(days=int(MAX_INDICATOR_PERIOD * 1.5) + 30)
            
            with st.spinner(f"Generating unified data from {fetch_start_date.date()} to {end_date.date()}..."):
                all_generated_data = generate_historical_data(
                    symbols_to_process, 
                    fetch_start_date, 
                    end_date
                )
            
            if not all_generated_data:
                st.error("Failed to generate any data.")
                return

            # Filter final range
            all_generated_data = [
                (date, df) for date, df in all_generated_data 
                if date.date() >= start_date and date.date() <= end_date
            ]
            
            if not all_generated_data:
                st.warning("Data was fetched, but no valid trading days found in the selected range.")
                return

            base_dir = "data"
            reports_dir = os.path.join(base_dir, "historical")
            zip_dir = os.path.join(base_dir, "zip")

            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            os.makedirs(reports_dir)
            os.makedirs(zip_dir)

            st.info("Saving daily snapshots to 'data/historical' folder...")
            progress_bar = st.progress(0)
            last_day_df = pd.DataFrame()

            if all_generated_data:
                for i, (snapshot_date, final_df) in enumerate(all_generated_data):
                    if not final_df.empty:
                        last_day_df = final_df
                        filename = os.path.join(reports_dir, f"{snapshot_date.strftime('%Y-%m-%d')}.csv")
                        final_df.to_csv(filename, index=False, float_format='%.2f')
                    
                    progress_bar.progress((i + 1) / len(all_generated_data))
            
                zip_file_name_only = f"indicator_reports_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.zip"
                zip_full_path = os.path.join(zip_dir, zip_file_name_only)
                
                with zipfile.ZipFile(zip_full_path, 'w') as zipf:
                    for root, _, files in os.walk(reports_dir):
                        for file in files:
                            zipf.write(os.path.join(root, file), os.path.join(os.path.basename(root), file))

                st.success("✅ Snapshots and Zip file generated successfully!")
                
                st.subheader(f"Data for {end_date.strftime('%Y-%m-%d')} (Last Day)")
                if not last_day_df.empty:
                    st.dataframe(last_day_df[COLUMN_ORDER].round(2))
                
                with open(zip_full_path, "rb") as fp:
                    st.download_button(
                        label="⬇️ Download All Reports (.zip)",
                        data=fp,
                        file_name=zip_file_name_only,
                        mime="application/zip"
                    )
            else:
                st.warning("No data was generated for the selected date range.")

if __name__ == "__main__":
    main()
