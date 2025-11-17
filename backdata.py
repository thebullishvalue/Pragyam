import streamlit as st
import yfinance as yf
import pandas as pd
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


# --- Standalone LiquidityOscillator Class (Unchanged) ---
class LiquidityOscillator:
    """
    Calculates the 'Liquidity Oscillator' indicator.
    """
    def __init__(self, length: int = 20, impact_window: int = 3):
        if length <= 0 or impact_window <= 0:
            raise ValueError("Length and impact_window must be positive integers.")
        self.length = length
        self.impact_window = impact_window

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(data.columns):
            return pd.Series(dtype=float)

        df = data.copy()
        df['spread'] = (df['high'] + df['low']) / 2 - df['open']
        df['vol_ma'] = df['volume'].rolling(window=self.length).mean()
        safe_vol_ma = df['vol_ma'].replace(0, pd.NA)
        df['vwap_spread'] = (df['spread'] * df['volume'] / safe_vol_ma).rolling(window=self.length).mean()
        close_shifted = df['close'].shift(self.impact_window)
        df['price_impact'] = ((df['close'] - close_shifted) * df['volume'] / safe_vol_ma).rolling(window=self.length).mean()
        df['liquidity_score'] = df['vwap_spread'] - df['price_impact']
        df['source_value'] = df['close'] + df['liquidity_score']
        df['lowest_value'] = df['source_value'].rolling(window=self.length).min()
        df['highest_value'] = df['source_value'].rolling(window=self.length).max()
        range_value = df['highest_value'] - df['lowest_value']
        safe_range_value = range_value.replace(0, pd.NA)
        oscillator = 200 * (df['source_value'] - df['lowest_value']) / safe_range_value - 100
        return oscillator.rename('liquidity_oscillator')

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
    # The start_date and end_date are now calculated *before* calling this function.
    
    try:
        # Spinner is now handled in pragati.py
        logging.info(f"--- yfinance: Attempting to download {len(symbols_to_process)} symbols...")
        all_data = yf.download(
            symbols_to_process, 
            start=start_date, 
            end=end_date + timedelta(days=1), # yf is end-exclusive
            progress=False
            # --- REMOVED 'raise_errors' and 'repair' ---
            # --- These arguments are not supported in your yfinance version ---
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
    oscillator_calculator = LiquidityOscillator(length=20, impact_window=3)
    
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
