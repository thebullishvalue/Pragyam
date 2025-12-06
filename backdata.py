import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import zipfile
import shutil
from typing import List, Tuple, Dict, Any
import logging

# --- Setup Logging ---
# Note: This basicConfig will also apply when imported by pragati.py
# This is fine, as pragati.py sets its own handlers.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Suppress yfinance warnings for cleaner output ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- UTILITY FUNCTIONS FOR UNIFIED OSCILLATOR ---
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
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

# --- UnifiedOscillator Class (Replaces LiquidityOscillator) ---
class UnifiedOscillator:
    """
    Calculates the 'Unified Oscillator' combining Market Structure, Flow, 
    Momentum (MSF) and Macro Risk (MMR).
    
    NOTE: Outputs values scaled to [-100, 100] to maintain compatibility 
    with existing strategies that expect this range.
    """
    def __init__(self, length: int = 20, roc_len: int = 14):
        self.length = length
        self.roc_len = roc_len

    def _calculate_msf(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculates Market Structure Flow (MSF) signal."""
        close = df['close']
        length = self.length
        clip = 3.0
        
        # 1. Momentum
        roc_raw = close.pct_change(self.roc_len)
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
        vol_adj_mom_raw = close.diff(5) / atr.replace(0, np.nan)
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
        msf_signal = sigmoid(msf_raw * np.sqrt(3.0), 1.0)
        
        return msf_signal, micro_norm

    def _calculate_mmr(self, df: pd.DataFrame) -> Tuple[pd.Series, float]:
        """Calculates Macro Market Risk (MMR). Defaults to 0 if macro cols missing."""
        # Note: In backdata.py context, we usually don't have macro columns.
        # This function is kept for logic completeness but is optimized to return 0 fast.
        return pd.Series(0.0, index=df.index), 0.0

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Main calculation entry point.
        Expects lower-case columns: 'open', 'high', 'low', 'close', 'volume'
        """
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(data.columns):
            return pd.Series(dtype=float)

        df = data.copy()
        
        # 1. Component Calculation
        msf_signal, _ = self._calculate_msf(df)
        mmr_signal, mmr_quality = self._calculate_mmr(df)
        
        # 2. Adaptive Weighting Logic
        regime_sensitivity = 1.5
        base_weight = 0.5
        
        msf_clarity = msf_signal.abs()
        mmr_clarity = mmr_signal.abs()
        
        msf_clarity_scaled = msf_clarity.pow(regime_sensitivity)
        mmr_clarity_scaled = (mmr_clarity * mmr_quality).pow(regime_sensitivity)
        clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001
        
        msf_w_adaptive = msf_clarity_scaled / clarity_sum
        mmr_w_adaptive = mmr_clarity_scaled / clarity_sum
        
        msf_w_final = 0.5 * base_weight + 0.5 * msf_w_adaptive
        mmr_w_final = 0.5 * (1.0 - base_weight) + 0.5 * mmr_w_adaptive
        w_sum = msf_w_final + mmr_w_final
        
        msf_w_norm = msf_w_final / w_sum
        mmr_w_norm = mmr_w_final / w_sum
        
        unified_signal = (msf_w_norm * msf_signal) + (mmr_w_norm * mmr_signal)
        
        agreement = msf_signal * mmr_signal 
        agree_strength = agreement.abs()
        multiplier = np.where(agreement > 0, 1.0 + 0.2 * agree_strength, 1.0 - 0.1 * agree_strength)
        
        final_signal = (unified_signal * multiplier).clip(-1.0, 1.0)
        
        # SCALING: Multiply by 100 to map [-1, 1] to [-100, 100]
        # This ensures compatibility with strategies expecting 'LiquidityOscillator' ranges.
        return final_signal * 100.0

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
        
        # Using the new Unified Oscillator logic here
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


# --- *** NEW: Refactored Core Generation Logic *** ---

# --- Load symbols from file ---
def load_symbols_from_file(filepath: str = "symbols.txt") -> List[str]:
    """
    Loads a list of symbols from a text file.
    """
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

# Define the column order here so it can be used by the generator
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

# --- NEW: Export max indicator period ---
INDICATOR_PERIODS = [20, 90, 200]
MAX_INDICATOR_PERIOD = max(INDICATOR_PERIODS)


def generate_historical_data(
    symbols_to_process: List[str], 
    start_date: datetime, 
    end_date: datetime
) -> List[Tuple[datetime, pd.DataFrame]]:
    """
    Generates historical indicator snapshots for a list of symbols
    and returns them in the format required by Pragati.
    
    Args:
        symbols_to_process: List of stock ticker symbols.
        start_date: The beginning of the date range for data download.
        end_date: The end of the date range for snapshot generation.
        
    Returns:
        A list of tuples: [(date, DataFrame), (date, DataFrame), ...]
    """
    
    if not symbols_to_process:
        try:
            st.error("Error: No symbols provided to generate_historical_data.")
        except Exception:
            logging.error("Error: No symbols provided to generate_historical_data.")
        return []
        
    # 1. --- Download Data ---
    
    try:
        # Spinner is now handled in pragati.py
        logging.info(f"--- yfinance: Attempting to download {len(symbols_to_process)} symbols...")
        all_data = yf.download(
            symbols_to_process, 
            start=start_date, 
            end=end_date + timedelta(days=1), # yf is end-exclusive
            progress=False
        )
    except Exception as e:
         logging.error(f"yf.download failed: {e}")
         all_data = pd.DataFrame() # Ensure all_data is a DataFrame

    if all_data.empty or all_data['Close'].dropna(how='all').empty:
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
                print(warning_msg)
                
            all_data = all_data.loc[:, (slice(None), valid_tickers)]
            symbols_to_process = list(valid_tickers)
            
            if not symbols_to_process:
                logging.error("No valid tickers remaining after download.")
                return []
    
    logging.info(f"yf.download successful. Data shape: {all_data.shape}. Valid tickers: {len(symbols_to_process)}")

    all_data.columns.names = ['Indicator', 'Symbol']
    
    # --- REPLACED: Use UnifiedOscillator instead of LiquidityOscillator ---
    oscillator_calculator = UnifiedOscillator(length=20, roc_len=14)
    
    # 2. --- Pre-calculate all indicators for all symbols ---
    ticker_indicator_cache = {}
    for i, ticker in enumerate(symbols_to_process):
        try:
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
                
        except (pd.errors.DataError, KeyError, IndexError):
            try:
                st.warning(f"⚠️ Skipping {ticker} due to a data quality error during indicator calculation.")
            except Exception:
                print(f"⚠️ Skipping {ticker} due to a data quality error during indicator calculation.")
            continue

    # 3. --- Generate Daily Snapshots in Memory ---
    pragati_data_list: List[Tuple[datetime, pd.DataFrame]] = []
    # Use the index of the downloaded data as the authoritative date range
    date_range = all_data.index.normalize().unique()

    for snapshot_date in date_range:
        # --- NEW: Only start generating snapshots *after* the indicator period
        # We also only care about dates *within* the requested range (end_date)
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
                    continue # Skip if all data is NaN or price is NaN

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
# This remains so the app can be run standalone
def main():
    # --- PAGE CONFIG MOVED HERE ---
    st.set_page_config(
        page_title="Indicator Snapshot Generator (Optimized)",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("📊 Daily Indicator Snapshot Generator (Optimized)")

    with st.sidebar:
        st.header("1. Select Date Range")
        today = datetime.now()
        # --- UPDATED: Default start date to be far enough back for indicators
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
            
            # --- UPDATED: Calculate fetch_start_date for standalone run ---
            fetch_start_date = start_date - timedelta(days=int(MAX_INDICATOR_PERIOD * 1.5) + 30)
            
            with st.spinner(f"Generating historical data from {fetch_start_date.date()} to {end_date.date()}..."):
                all_generated_data = generate_historical_data(
                    symbols_to_process, 
                    fetch_start_date, # Pass the earlier date for indicator warmup
                    end_date
                )
            
            if not all_generated_data:
                st.error("Failed to generate any data.")
                return

            # --- Filter the generated data to *only* the user's requested date range
            all_generated_data = [
                (date, df) for date, df in all_generated_data 
                if date.date() >= start_date and date.date() <= end_date
            ]
            
            if not all_generated_data:
                st.warning("Data was fetched, but no valid trading days found in the selected Start/End range.")
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

                st.success("✅ Snapshots and Zip file generated successfully in the 'data' folder!")
                
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
