# ============================================================================
# Unified Market Analysis (UMA) - Complete Implementation for Pragyam System
# ============================================================================
#
# This module implements the full Unified Market Analysis indicator logic
# from Pine Script, providing a non-invasive weight boosting layer based on
# buy signals (lime circle conditions).
#
# DATA SOURCE: 
# 1. ETFs: yfinance (Yahoo Finance)
# 2. Macros: Stooq.com (via pandas_datareader)
# 
# Architecture:
# 1. MSF (Momentum Structure Flow) - Internal price dynamics
#    - Momentum Component (ROC-based)
#    - Market Microstructure Component
#    - Volatility Regime (Confidence Bands)
#    - Composite Trend
#    - Accumulation/Distribution
#    - Regime Counter
#    - RSI Component
#
# 2. MMR (Macro Multiple Regression) - External macro drivers
#    - Bond yields from major economies
#    - Currency pairs (INR crosses)
#    - Commodities (Gold, Silver, Oil)
#    - DXY (Dollar Index)
#
# 3. Signal Integration
#    - Adaptive weighting based on signal clarity
#    - Agreement multiplier for confirmation
#
# Performance Optimizations:
# - LRU caching for data fetches
# - Vectorized numpy operations (no row-wise apply)
# - Batch data fetching
# - Pre-computed indicator cache
# - Lazy macro data loading
#
# Statistical Foundation:
# - Orthogonal information sources (endogenous vs exogenous)
# - Proper normalization throughout (z-scores, sigmoid transforms)
# - Variance preservation when combining signals
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Set, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from functools import lru_cache
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time

warnings.filterwarnings('ignore')

# --- Import Data Libraries ---
try:
    import yfinance as yf
    import pandas_datareader.data as web
    DATA_LIBS_AVAILABLE = True
except ImportError:
    DATA_LIBS_AVAILABLE = False
    logging.warning("yfinance or pandas_datareader not available - UMA Booster will be disabled")


# ============================================================================
# SECTION 1: CONFIGURATION & PARAMETERS
# ============================================================================

@dataclass
class UMAParameters:
    """Configuration parameters matching Pine Script defaults"""
    
    # Core Settings
    length: int = 20  # Lookback Period
    roc_length: int = 14  # ROC Length
    
    # Statistical Settings
    confidence_level: float = 0.95  # For confidence bands
    zscore_clip: float = 3.0  # Z-Score Clipping threshold
    
    # Signal Integration Settings
    msf_weight_base: float = 0.5  # Base weight for MSF (internal dynamics)
    use_adaptive_weights: bool = True  # Adjust weights based on signal clarity
    regime_sensitivity: float = 1.5  # Sensitivity for adaptive weighting
    
    # Oscillator Settings
    bb_length: int = 20  # Bollinger Band Length
    bb_mult: float = 2.0  # BB Standard Deviation multiplier
    
    # RSI Settings
    rsi_length: int = 14
    rsi_lower: int = 40  # Oversold threshold
    rsi_upper: int = 70  # Overbought threshold
    
    # Macro Regression Settings
    regression_length: int = 20
    correlation_lookback: int = 1000
    num_macro_vars: int = 5  # Number of macro variables in model
    
    # Buy Signal Thresholds
    unified_osc_oversold: float = -5.0  # Unified oscillator oversold level
    agreement_threshold: float = 0.3  # Strong agreement threshold
    
    # Performance Settings
    cache_ttl_seconds: int = 3600  # Cache TTL (1 hour)
    max_parallel_fetches: int = 5  # Max parallel data fetches
    min_data_points: int = 200  # Minimum data points required


# ============================================================================
# SECTION 2: DATA CACHING LAYER
# ============================================================================

class DataCache:
    """
    In-memory cache for fetched data with TTL support.
    Prevents redundant API calls within the same session.
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl_seconds
    
    def _get_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cache value with current timestamp"""
        self._cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cached data"""
        self._cache.clear()


# Global cache instance
_data_cache = DataCache(ttl_seconds=3600)


# ============================================================================
# SECTION 3: UTILITY FUNCTIONS (Vectorized Operations)
# ============================================================================

class StatisticalUtils:
    """
    Statistical utility functions matching Pine Script implementations.
    All functions use vectorized numpy operations for performance.
    """
    
    @staticmethod
    def zscore_clipped(series: pd.Series, length: int, clip_threshold: float = 3.0) -> pd.Series:
        """
        Robust Z-Score with clipping (vectorized).
        """
        mean_val = series.rolling(window=length, min_periods=1).mean()
        std_val = series.rolling(window=length, min_periods=1).std()
        
        # Vectorized division with zero handling
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_z = np.where(std_val > 0, (series - mean_val) / std_val, 0.0)
        
        # Vectorized clipping
        return pd.Series(np.clip(raw_z, -clip_threshold, clip_threshold), index=series.index)
    
    @staticmethod
    def sigmoid(z: pd.Series, scale: float = 1.5) -> pd.Series:
        """
        Sigmoid transformation (vectorized).
        """
        # Use numpy for vectorized exp
        z_arr = z.values
        result = 2.0 / (1.0 + np.exp(-z_arr / scale)) - 1.0
        return pd.Series(result, index=z.index)
    
    @staticmethod
    def minmax_normalize(series: pd.Series, length: int) -> pd.Series:
        """
        Min-max normalization (vectorized).
        """
        src_max = series.rolling(window=length, min_periods=1).max()
        src_min = series.rolling(window=length, min_periods=1).min()
        
        range_val = src_max - src_min
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = np.where(range_val > 0, 2.0 * (series - src_min) / range_val - 1.0, 0.0)
        
        return pd.Series(normalized, index=series.index)
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def stdev(series: pd.Series, period: int) -> pd.Series:
        """Standard Deviation"""
        return series.rolling(window=period, min_periods=1).std()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI calculation (vectorized).
        """
        delta = series.diff()
        
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
            rsi_val = 100.0 - (100.0 / (1.0 + rs))
        
        return pd.Series(rsi_val, index=series.index).fillna(50.0)
    
    @staticmethod
    def roc(series: pd.Series, period: int) -> pd.Series:
        """
        Rate of Change (vectorized).
        """
        shifted = series.shift(period)
        with np.errstate(divide='ignore', invalid='ignore'):
            roc_val = np.where(shifted > 0, (series - shifted) / shifted * 100, 0.0)
        return pd.Series(roc_val, index=series.index).fillna(0.0)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range (vectorized).
        """
        prev_close = close.shift(1)
        
        # True Range components (vectorized)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        # Max of all three
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        
        return pd.Series(true_range, index=close.index).ewm(span=period, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def correlation(series1: pd.Series, series2: pd.Series, period: int) -> pd.Series:
        """Rolling correlation"""
        return series1.rolling(window=period, min_periods=period//2).corr(series2)


# ============================================================================
# SECTION 4: MARKET DATA FETCHER (YFinance + Stooq)
# ============================================================================

class MarketDataFetcher:
    """
    Data fetching layer using yfinance for ETFs and Stooq (via pandas_datareader) for Macros.
    
    Performance optimizations:
    - LRU caching for repeated requests
    - Batch fetching with parallel execution
    - Graceful degradation on errors
    """
    
    # Macro data mappings for Stooq
    # Format: 'KEY': {'ticker': 'STOOQ_SYMBOL', 'name': 'Display Name'}
    # Note: Stooq yield tickers usually follow the format 10YUSY.B (10 Year US Yield Bond)
    MACRO_MAPPINGS = {
        # Bonds
        'US10YT=X': {'ticker': '10YUSY.B', 'name': 'United States 10-Year'},
        'US2YT=X':  {'ticker': '2YUSY.B', 'name': 'United States 2-Year'},
        'US30YT=X': {'ticker': '30YUSY.B', 'name': 'United States 30-Year'},
        'JP10Y':    {'ticker': '10YJPY.B', 'name': 'Japan 10Y'},
        'CN10Y':    {'ticker': '10YCNY.B', 'name': 'China 10Y'},
        'EU10Y':    {'ticker': '10YDEY.B', 'name': 'Germany 10Y (EU Proxy)'},
        'GB10Y':    {'ticker': '10YUKY.B', 'name': 'U.K. 10Y'},
        'IN10Y':    {'ticker': '10YINY.B', 'name': 'India 10Y'},
        
        # Currencies (Stooq tickers usually standard pairs)
        'USDINR':   {'ticker': 'USDINR', 'name': 'USD/INR'},
        'EURINR':   {'ticker': 'EURINR', 'name': 'EUR/INR'},
        'GBPINR':   {'ticker': 'GBPINR', 'name': 'GBP/INR'},
        'JPYINR':   {'ticker': 'JPYINR', 'name': 'JPY/INR'},
        
        # Commodities
        'GOLD':     {'ticker': 'XAUUSD', 'name': 'Gold Spot'},
        'SILVER':   {'ticker': 'XAGUSD', 'name': 'Silver Spot'},
        'OIL':      {'ticker': 'CL.F',   'name': 'WTI Crude Oil Futures'},
        
        # Indices
        'DXY':      {'ticker': 'USD_I',  'name': 'US Dollar Index (Stooq)'},
    }
    
    DISPLAY_NAMES = {k: v['name'] for k, v in MACRO_MAPPINGS.items()}
    
    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or _data_cache
    
    def fetch_etf_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch ETF data from yfinance with caching.
        Appends .NS for NSE symbols if not present.
        """
        if not DATA_LIBS_AVAILABLE:
            return None
        
        # Clean and format symbol for Yahoo Finance
        clean_symbol = symbol.strip().upper().replace(' ', '')
        if not clean_symbol.endswith('.NS') and not clean_symbol.endswith('.BO'):
            # Default to NSE
            yf_symbol = f"{clean_symbol}.NS"
        else:
            yf_symbol = clean_symbol
            
        # Check cache first
        cache_key = f"etf_{yf_symbol}_{start_date.date()}_{end_date.date()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logging.debug(f"Cache hit for {yf_symbol}")
            return cached
        
        try:
            # Fetch data using yfinance
            # auto_adjust=True gives us adjusted close as 'Close'
            df = yf.download(yf_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if df is None or df.empty:
                # Try without suffix just in case it's a global ticker
                if yf_symbol != clean_symbol:
                    logging.debug(f"Retrying {clean_symbol} without .NS suffix")
                    df = yf.download(clean_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    if df is None or df.empty:
                        return None
                else:
                    return None
            
            # Standardize column names (lowercase)
            # yfinance might return MultiIndex columns if multiple tickers (not the case here)
            # but we should be safe.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1) # Drop Ticker level if present
            
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                # Fallback: if 'adj close' is present but 'close' isn't (due to auto_adjust=False), rename it
                if 'adj close' in df.columns and 'close' not in df.columns:
                    df.rename(columns={'adj close': 'close'}, inplace=True)
                
                # If still missing
                if not all(col in df.columns for col in required):
                    logging.warning(f"Missing columns for {yf_symbol}. Found: {df.columns}")
                    return None
            
            # Cache the result
            self.cache.set(cache_key, df)
            
            logging.debug(f"Fetched {len(df)} bars for {yf_symbol}")
            return df
            
        except Exception as e:
            logging.debug(f"yfinance fetch failed for {yf_symbol}: {e}")
            return None
    
    def fetch_macro_data(self, var_name: str, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
        """
        Fetch macro variable data from Stooq via pandas_datareader.
        """
        if not DATA_LIBS_AVAILABLE:
            return None
        
        # Check cache
        cache_key = f"macro_stooq_{var_name}_{start_date.date()}_{end_date.date()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        mapping = self.MACRO_MAPPINGS.get(var_name)
        if not mapping:
            return None
        
        stooq_ticker = mapping['ticker']
        
        try:
            # Stooq fetch using pandas_datareader
            df = web.DataReader(stooq_ticker, 'stooq', start=start_date, end=end_date)
            
            if df is None or df.empty:
                return None
            
            # Stooq returns data with Date index, newest first usually.
            # We need to sort index ascending
            df = df.sort_index()
            
            # Get close price series
            # Stooq columns are usually Capitalized (Open, High, Low, Close)
            close_col = 'Close' if 'Close' in df.columns else 'close'
            if close_col not in df.columns:
                # Some stooq data might just have a value column?
                # Usually it's OHLC
                return None
                
            series = df[close_col]
            
            # Fill NaNs (forward fill then backward fill)
            series = series.fillna(method='ffill').fillna(method='bfill')
            
            # Cache result
            self.cache.set(cache_key, series)
            
            return series
            
        except Exception as e:
            logging.debug(f"Stooq fetch failed for {var_name} ({stooq_ticker}): {e}")
            return None
    
    def fetch_macro_data_batch(self, var_names: List[str], start_date: datetime, 
                                end_date: datetime, max_workers: int = 5) -> Dict[str, pd.Series]:
        """
        Fetch multiple macro variables in parallel.
        """
        results = {}
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_var = {
                executor.submit(self.fetch_macro_data, var, start_date, end_date): var 
                for var in var_names
            }
            
            for future in as_completed(future_to_var):
                var_name = future_to_var[future]
                try:
                    data = future.result()
                    if data is not None and len(data) > 0:
                        results[var_name] = data
                except Exception as e:
                    logging.debug(f"Parallel fetch failed for {var_name}: {e}")
        
        return results


# ============================================================================
# SECTION 5: MSF - MOMENTUM STRUCTURE FLOW (Vectorized)
# ============================================================================

class MomentumStructureFlow:
    """
    MSF (Momentum Structure Flow) - Analyzes internal price dynamics.
    Fully vectorized implementation for performance.
    """
    
    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all MSF components (vectorized).
        """
        results = {}
        length = self.params.length
        zscore_clip = self.params.zscore_clip
        
        # Ensure column names are lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df['volume']
        
        # === 3.1 Momentum Component (ROC-based) - Vectorized ===
        roc_raw = self.stats.roc(close, self.params.roc_length)
        roc_z = self.stats.zscore_clipped(roc_raw, length, zscore_clip)
        momentum_norm = self.stats.sigmoid(roc_z, 1.5)
        results['momentum_norm'] = momentum_norm
        
        # === 3.2 Market Microstructure Component - Vectorized ===
        intrabar_direction = (high + low) / 2 - open_price
        vol_ma = self.stats.sma(volume, length)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
        vol_ratio = pd.Series(vol_ratio, index=df.index)
        
        vw_direction = self.stats.sma(intrabar_direction * vol_ratio, length)
        price_change_impact = close - close.shift(5)
        vw_impact = self.stats.sma(price_change_impact * vol_ratio, length)
        
        microstructure_raw = vw_direction - vw_impact
        microstructure_z = self.stats.zscore_clipped(microstructure_raw, length, zscore_clip)
        microstructure_norm = self.stats.sigmoid(microstructure_z, 1.5)
        results['microstructure_norm'] = microstructure_norm
        
        # === 3.3 Volatility Regime - Vectorized ===
        price_mean = self.stats.sma(close, length)
        price_stdev = self.stats.stdev(close, length)
        conf_mult = 1.96 if self.params.confidence_level >= 0.95 else 1.645
        
        upper_bound = price_mean + conf_mult * price_stdev
        lower_bound = price_mean - conf_mult * price_stdev
        band_width = upper_bound - lower_bound
        
        with np.errstate(divide='ignore', invalid='ignore'):
            price_position = np.where(band_width > 0, (close - lower_bound) / band_width * 2 - 1, 0.0)
        price_position = pd.Series(np.clip(price_position, -1.5, 1.5), index=df.index)
        
        results['price_position'] = price_position
        results['upper_bound'] = upper_bound
        results['lower_bound'] = lower_bound
        
        # === 3.4 Composite Trend - Vectorized ===
        trend_fast = self.stats.sma(close, 5)
        trend_slow = self.stats.sma(close, length)
        trend_diff_z = self.stats.zscore_clipped(trend_fast - trend_slow, length, zscore_clip)
        
        price_change_5 = close.diff(5)
        momentum_accel_raw = price_change_5.diff(5)
        momentum_accel_z = self.stats.zscore_clipped(momentum_accel_raw, length, zscore_clip)
        
        atr_val = self.stats.atr(high, low, close, 14)
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_adj_mom_raw = np.where(atr_val > 0, price_change_5 / atr_val, 0.0)
        vol_adj_mom_raw = pd.Series(vol_adj_mom_raw, index=df.index)
        vol_adj_mom_z = self.stats.zscore_clipped(vol_adj_mom_raw, length, zscore_clip)
        
        mean_reversion_z = self.stats.zscore_clipped(close - price_mean, length, zscore_clip)
        
        # Variance-preserving combination
        composite_trend_z = (trend_diff_z + momentum_accel_z + vol_adj_mom_z + mean_reversion_z) / np.sqrt(4.0)
        composite_trend_norm = self.stats.sigmoid(composite_trend_z, 1.5)
        results['composite_trend_norm'] = composite_trend_norm
        
        # === 3.5 Accumulation/Distribution - Vectorized ===
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        close_up = close > close.shift(1)
        mf_positive = money_flow.where(close_up, 0)
        mf_negative = money_flow.where(~close_up, 0)
        
        mf_pos_smooth = self.stats.sma(mf_positive, length)
        mf_neg_smooth = self.stats.sma(mf_negative, length)
        mf_total = mf_pos_smooth + mf_neg_smooth
        
        with np.errstate(divide='ignore', invalid='ignore'):
            accum_ratio = np.where(mf_total > 0, mf_pos_smooth / mf_total, 0.5)
        accum_norm = pd.Series(2.0 * (accum_ratio - 0.5), index=df.index)
        results['accum_norm'] = accum_norm
        
        # === 3.6 Regime Counter - Vectorized ===
        pct_change = close.pct_change() * 100
        threshold_pct = 0.33
        
        # Vectorized regime counting using cumsum
        up_signal = (pct_change > threshold_pct).astype(int)
        down_signal = (pct_change < -threshold_pct).astype(int)
        regime_count = (up_signal - down_signal).cumsum()
        
        regime_raw = regime_count - self.stats.sma(regime_count, length)
        regime_z = self.stats.zscore_clipped(regime_raw, length, zscore_clip)
        regime_norm = self.stats.sigmoid(regime_z, 1.5)
        results['regime_norm'] = regime_norm
        
        # === 3.7 RSI Component - Vectorized ===
        rsi_value = self.stats.rsi(close, self.params.rsi_length)
        rsi_norm = (rsi_value - 50) / 50
        results['rsi_value'] = rsi_value
        results['rsi_norm'] = rsi_norm
        
        # === 3.8 MSF Composite Signal - Vectorized ===
        osc_momentum = momentum_norm
        osc_structure = (microstructure_norm + composite_trend_norm) / np.sqrt(2.0)
        osc_flow = (accum_norm + regime_norm) / np.sqrt(2.0)
        
        msf_raw = (osc_momentum + osc_structure + osc_flow) / np.sqrt(3.0)
        msf_signal = self.stats.sigmoid(msf_raw * np.sqrt(3.0), 1.0)
        results['msf_signal'] = msf_signal
        results['msf_clarity'] = msf_signal.abs()
        
        return results


# ============================================================================
# SECTION 6: MMR - MACRO MULTIPLE REGRESSION
# ============================================================================

class MacroMultipleRegression:
    """
    MMR (Macro Multiple Regression) - Analyzes external macro drivers.
    Uses MarketDataFetcher (Stooq) for data fetching.
    """
    
    def __init__(self, params: UMAParameters, fetcher: Optional[MarketDataFetcher] = None):
        self.params = params
        self.stats = StatisticalUtils()
        self.fetcher = fetcher or MarketDataFetcher()
    
    def fetch_macro_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.Series]:
        """
        Fetch all macro data in parallel batches.
        """
        var_names = list(self.fetcher.MACRO_MAPPINGS.keys())
        return self.fetcher.fetch_macro_data_batch(
            var_names, 
            start_date, 
            end_date,
            max_workers=self.params.max_parallel_fetches
        )
    
    def calculate(self, df: pd.DataFrame, macro_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Calculate MMR signal based on macro regression (vectorized).
        """
        results = {}
        
        if not macro_data:
            # Return neutral signal if no macro data
            results['mmr_signal'] = pd.Series(0.0, index=df.index)
            results['mmr_clarity'] = pd.Series(0.0, index=df.index)
            results['model_r2'] = 0.0
            results['top_drivers'] = []
            return results
        
        target = df['close'] if 'close' in df.columns else df['Close']
        
        # Align macro data to target index (vectorized)
        aligned_macro = {}
        for name, series in macro_data.items():
            # Reindex aligned with target, filling missing values
            aligned = series.reindex(target.index).ffill()
            valid_count = aligned.notna().sum()
            if valid_count > self.params.correlation_lookback * 0.5:
                aligned_macro[name] = aligned
        
        if not aligned_macro:
            results['mmr_signal'] = pd.Series(0.0, index=df.index)
            results['mmr_clarity'] = pd.Series(0.0, index=df.index)
            results['model_r2'] = 0.0
            results['top_drivers'] = []
            return results
        
        # Calculate correlations (vectorized)
        correlations = {}
        for name, series in aligned_macro.items():
            corr = self.stats.correlation(target, series, self.params.correlation_lookback)
            valid_corr = corr.dropna()
            if len(valid_corr) > 0:
                correlations[name] = valid_corr.iloc[-1]
        
        if not correlations:
            results['mmr_signal'] = pd.Series(0.0, index=df.index)
            results['mmr_clarity'] = pd.Series(0.0, index=df.index)
            results['model_r2'] = 0.0
            results['top_drivers'] = []
            return results
        
        # Sort by absolute correlation and select top N
        sorted_vars = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        top_vars = sorted_vars[:self.params.num_macro_vars]
        
        results['top_drivers'] = [
            {
                'name': name, 
                'display': self.fetcher.DISPLAY_NAMES.get(name, name), 
                'correlation': corr
            }
            for name, corr in top_vars
        ]
        
        # Build regression predictions (vectorized)
        predictions = []
        weights = []
        
        for name, corr in top_vars:
            if name not in aligned_macro:
                continue
            
            x = aligned_macro[name]
            pred, r2 = self._regression_predict_vectorized(x, target, self.params.regression_length)
            
            predictions.append(pred)
            weights.append(r2)
        
        if not predictions:
            results['mmr_signal'] = pd.Series(0.0, index=df.index)
            results['mmr_clarity'] = pd.Series(0.0, index=df.index)
            results['model_r2'] = 0.0
            return results
        
        # Weighted average prediction (vectorized)
        total_weight = sum(weights)
        if total_weight > 0:
            y_predicted = sum(p * w for p, w in zip(predictions, weights)) / total_weight
            model_r2 = sum(w * w for w in weights) / total_weight
        else:
            y_predicted = predictions[0]
            model_r2 = weights[0] if weights else 0
        
        results['model_r2'] = model_r2
        results['y_predicted'] = y_predicted
        
        # Deviation from fair value
        deviation = target - y_predicted
        deviation_z = self.stats.zscore_clipped(deviation, self.params.length, self.params.zscore_clip)
        mmr_signal = self.stats.sigmoid(deviation_z, 1.5)
        
        results['mmr_signal'] = mmr_signal
        results['mmr_clarity'] = mmr_signal.abs()
        
        return results
    
    def _regression_predict_vectorized(self, x: pd.Series, y: pd.Series, length: int) -> Tuple[pd.Series, float]:
        """
        Vectorized linear regression prediction.
        """
        x_mean = self.stats.sma(x, length)
        y_mean = self.stats.sma(y, length)
        x_std = self.stats.stdev(x, length)
        y_std = self.stats.stdev(y, length)
        
        corr = self.stats.correlation(x, y, length)
        
        # Vectorized slope calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = np.where(x_std > 0, corr * (y_std / x_std), 0.0)
        slope = pd.Series(slope, index=x.index).fillna(0)
        
        intercept = y_mean - slope * x_mean
        prediction = x * slope + intercept
        
        # R² from most recent correlation
        recent_corr = corr.dropna()
        r2 = recent_corr.iloc[-1] ** 2 if len(recent_corr) > 0 else 0
        
        return prediction, r2


# ============================================================================
# SECTION 7: SIGNAL INTEGRATION
# ============================================================================

class SignalIntegrator:
    """
    Integrates MSF and MMR signals (vectorized).
    """
    
    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()
    
    def integrate(self, 
                  msf_signal: pd.Series, 
                  msf_clarity: pd.Series,
                  mmr_signal: pd.Series,
                  mmr_clarity: pd.Series,
                  mmr_quality: float) -> Dict[str, pd.Series]:
        """
        Integrate MSF and MMR signals (vectorized).
        """
        results = {}
        
        # === Adaptive Weight Calculation (Vectorized) ===
        msf_clarity_scaled = np.power(msf_clarity.values, self.params.regime_sensitivity)
        mmr_clarity_scaled = np.power(mmr_clarity.values * np.sqrt(mmr_quality), self.params.regime_sensitivity)
        
        clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001
        
        msf_weight_adaptive = msf_clarity_scaled / clarity_sum
        mmr_weight_adaptive = mmr_clarity_scaled / clarity_sum
        
        if self.params.use_adaptive_weights:
            msf_weight_final = 0.5 * self.params.msf_weight_base + 0.5 * msf_weight_adaptive
            mmr_weight_final = 0.5 * (1.0 - self.params.msf_weight_base) + 0.5 * mmr_weight_adaptive
        else:
            msf_weight_final = np.full(len(msf_signal), self.params.msf_weight_base)
            mmr_weight_final = np.full(len(msf_signal), 1.0 - self.params.msf_weight_base)
        
        weight_sum = msf_weight_final + mmr_weight_final
        msf_weight_norm = pd.Series(msf_weight_final / weight_sum, index=msf_signal.index)
        mmr_weight_norm = pd.Series(mmr_weight_final / weight_sum, index=msf_signal.index)
        
        results['msf_weight'] = msf_weight_norm
        results['mmr_weight'] = mmr_weight_norm
        
        # === Combined Signal (Vectorized) ===
        unified_signal = msf_weight_norm * msf_signal + mmr_weight_norm * mmr_signal
        
        # === Agreement Analysis (Vectorized) ===
        signal_agreement = msf_signal * mmr_signal
        agreement_strength = signal_agreement.abs()
        
        results['signal_agreement'] = signal_agreement
        
        # Agreement multiplier (vectorized)
        agreement_multiplier = np.where(
            signal_agreement > 0,
            1.0 + 0.2 * agreement_strength,
            1.0 - 0.1 * agreement_strength
        )
        
        unified_final = pd.Series(
            np.clip(unified_signal * agreement_multiplier, -1.0, 1.0),
            index=msf_signal.index
        )
        
        results['unified_signal'] = unified_final
        results['unified_osc'] = unified_final * 10.0
        results['msf_osc'] = msf_signal * 10.0
        results['mmr_osc'] = mmr_signal * 10.0
        
        return results


# ============================================================================
# SECTION 8: BUY SIGNAL DETECTION
# ============================================================================

class BuySignalDetector:
    """
    Detects buy signals (vectorized).
    """
    
    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()
    
    def detect(self, 
               unified_osc: pd.Series,
               signal_agreement: pd.Series,
               rsi_value: pd.Series) -> Dict[str, pd.Series]:
        """
        Detect buy and sell signals (vectorized).
        """
        results = {}
        
        # Bollinger Bands on unified oscillator
        bb_basis = self.stats.sma(unified_osc, self.params.bb_length)
        bb_dev = self.stats.stdev(unified_osc, self.params.bb_length)
        bb_upper = bb_basis + self.params.bb_mult * bb_dev
        bb_lower = bb_basis - self.params.bb_mult * bb_dev
        
        results['bb_upper'] = bb_upper
        results['bb_lower'] = bb_lower
        results['bb_basis'] = bb_basis
        
        # RSI of oscillator
        rsi_osc = self.stats.rsi(unified_osc, self.params.rsi_length)
        results['rsi_osc'] = rsi_osc
        
        # Vectorized condition checks
        is_oversold = (unified_osc < bb_lower) & (rsi_osc < self.params.rsi_lower)
        is_overbought = (unified_osc > bb_upper) & (rsi_osc > self.params.rsi_upper)
        
        results['is_oversold'] = is_oversold
        results['is_overbought'] = is_overbought
        
        # Strong agreement
        strong_agreement = signal_agreement > self.params.agreement_threshold
        results['strong_agreement'] = strong_agreement
        
        # Buy signal (lime circle condition)
        buy_signal = is_oversold & strong_agreement & (unified_osc < self.params.unified_osc_oversold)
        results['buy_signal'] = buy_signal
        
        # Sell signal
        sell_signal = is_overbought & strong_agreement & (unified_osc > -self.params.unified_osc_oversold)
        results['sell_signal'] = sell_signal
        
        # Soft buy signal (simpler criteria for boosting)
        soft_buy_signal = strong_agreement & (unified_osc < self.params.unified_osc_oversold)
        results['soft_buy_signal'] = soft_buy_signal
        
        # Divergence detection
        osc_rising = unified_osc > unified_osc.shift(1)
        rsi_falling = rsi_value < rsi_value.shift(1)
        bullish_divergence = osc_rising & rsi_falling & (unified_osc < -5)
        results['bullish_divergence'] = bullish_divergence
        
        return results


# ============================================================================
# SECTION 9: MAIN UMA CALCULATOR
# ============================================================================

class UnifiedMarketAnalysis:
    """
    Complete Unified Market Analysis calculator.
    """
    
    def __init__(self, params: Optional[UMAParameters] = None):
        self.params = params or UMAParameters()
        self.fetcher = MarketDataFetcher()
        self.msf = MomentumStructureFlow(self.params)
        self.mmr = MacroMultipleRegression(self.params, self.fetcher)
        self.integrator = SignalIntegrator(self.params)
        self.detector = BuySignalDetector(self.params)
        
        # Cache for macro data (fetched once per session)
        self._macro_data_cache: Optional[Dict[str, pd.Series]] = None
        self._macro_cache_dates: Optional[Tuple[datetime, datetime]] = None
    
    def _get_macro_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.Series]:
        """
        Get macro data with lazy loading and caching.
        """
        # Check if we have valid cached data
        if (self._macro_data_cache is not None and 
            self._macro_cache_dates is not None and
            self._macro_cache_dates[0] <= start_date and 
            self._macro_cache_dates[1] >= end_date):
            return self._macro_data_cache
        
        # Fetch fresh data
        self._macro_data_cache = self.mmr.fetch_macro_data(start_date, end_date)
        self._macro_cache_dates = (start_date, end_date)
        
        return self._macro_data_cache
    
    def calculate(self, 
                  df: pd.DataFrame, 
                  macro_data: Optional[Dict[str, pd.Series]] = None,
                  skip_macro: bool = False) -> Dict[str, Any]:
        """
        Calculate complete UMA analysis.
        
        Args:
            df: OHLCV DataFrame
            macro_data: Pre-fetched macro data (optional)
            skip_macro: If True, skip macro regression for faster calculation
        """
        results = {}
        
        # Standardize column names
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # === Calculate MSF ===
        msf_results = self.msf.calculate(df)
        results['msf'] = msf_results
        
        # === Calculate MMR (with optional skip for performance) ===
        if skip_macro:
            mmr_results = {
                'mmr_signal': pd.Series(0.0, index=df.index),
                'mmr_clarity': pd.Series(0.0, index=df.index),
                'model_r2': 0.0,
                'top_drivers': []
            }
        else:
            if macro_data is None:
                if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex):
                    start_date = df.index[0].to_pydatetime()
                    end_date = df.index[-1].to_pydatetime()
                    macro_data = self._get_macro_data(start_date, end_date)
                else:
                    macro_data = {}
            
            mmr_results = self.mmr.calculate(df, macro_data)
        
        results['mmr'] = mmr_results
        
        # === Integrate Signals ===
        integrated = self.integrator.integrate(
            msf_results['msf_signal'],
            msf_results['msf_clarity'],
            mmr_results['mmr_signal'],
            mmr_results['mmr_clarity'],
            mmr_results['model_r2']
        )
        results['integrated'] = integrated
        
        # === Detect Buy/Sell Signals ===
        signals = self.detector.detect(
            integrated['unified_osc'],
            integrated['signal_agreement'],
            msf_results['rsi_value']
        )
        results['signals'] = signals
        
        # === Summary ===
        results['unified_osc'] = integrated['unified_osc']
        results['buy_signal'] = signals['buy_signal']
        results['soft_buy_signal'] = signals['soft_buy_signal']
        results['sell_signal'] = signals['sell_signal']
        
        return results
    
    def has_buy_signal(self, df: pd.DataFrame, macro_data: Optional[Dict] = None, 
                       use_soft_signal: bool = True) -> bool:
        """
        Quick check if latest bar has a buy signal.
        """
        try:
            results = self.calculate(df, macro_data)
            
            signal_key = 'soft_buy_signal' if use_soft_signal else 'buy_signal'
            signal = results.get(signal_key)
            
            if signal is not None and len(signal) > 0:
                return bool(signal.iloc[-1])
            
            return False
            
        except Exception as e:
            logging.debug(f"Error detecting buy signal: {e}")
            return False


# ============================================================================
# SECTION 10: PORTFOLIO BOOSTER
# ============================================================================

class UMAPortfolioBooster:
    """
    Portfolio weight booster based on UMA signals.
    Optimized for batch processing with caching.
    """
    
    def __init__(self,
                 lookback_days: int = 200,
                 boost_multiplier: float = 1.15,
                 max_boost_weight: float = 0.15,
                 params: Optional[UMAParameters] = None,
                 skip_macro: bool = False):
        """
        Args:
            lookback_days: Days of historical data to fetch
            boost_multiplier: Weight multiplier for buy signals (1.15 = 15% boost)
            max_boost_weight: Maximum weight after boost (0.15 = 15%)
            params: UMA parameters
            skip_macro: Skip macro regression for faster processing
        """
        self.lookback_days = lookback_days
        self.boost_multiplier = boost_multiplier
        self.max_boost_weight = max_boost_weight
        self.params = params or UMAParameters()
        self.skip_macro = skip_macro
        
        self.fetcher = MarketDataFetcher()
        self.uma = UnifiedMarketAnalysis(self.params)
        
        logging.info(f"UMA Booster initialized: boost={boost_multiplier}x, max_weight={max_boost_weight}, skip_macro={skip_macro}")
    
    def _fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch ETF data using MarketDataFetcher (yfinance).
        """
        if not DATA_LIBS_AVAILABLE:
            return None
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        return self.fetcher.fetch_etf_data(symbol, start_date, end_date)
    
    def get_buy_signals(self, symbols: List[str]) -> Set[str]:
        """
        Detect buy signals for a list of symbols.
        Uses batch processing for efficiency.
        """
        if not DATA_LIBS_AVAILABLE:
            logging.warning("yfinance/pandas_datareader not available - no buy signals generated")
            return set()
        
        buy_signals = set()
        
        # Pre-fetch macro data once (if not skipping)
        macro_data = None
        if not self.skip_macro:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            # Use threading to ensure macro fetch doesn't block if slow
            try:
                macro_data = self.uma._get_macro_data(start_date, end_date)
            except Exception as e:
                logging.error(f"Failed to fetch macro data: {e}. Proceeding without macro component.")
                macro_data = {}
        
        # Process symbols
        for symbol in symbols:
            try:
                df = self._fetch_symbol_data(symbol)
                
                if df is None or len(df) < self.params.min_data_points:
                    logging.debug(f"Insufficient data for {symbol}")
                    continue
                
                if self.uma.has_buy_signal(df, macro_data, use_soft_signal=True):
                    clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
                    buy_signals.add(clean_symbol)
                    logging.info(f"✅ UMA Buy signal: {symbol}")
                    
            except Exception as e:
                logging.debug(f"Error processing {symbol}: {e}")
                continue
        
        logging.info(f"UMA Booster: {len(buy_signals)} buy signals from {len(symbols)} symbols")
        return buy_signals
    
    def apply_boost(self, portfolio_df: pd.DataFrame, buy_signals: Set[str]) -> pd.DataFrame:
        """
        Apply weight boost to symbols with buy signals.
        """
        if portfolio_df.empty or not buy_signals:
            return portfolio_df
        
        boosted_df = portfolio_df.copy()
        original_total = boosted_df['weightage_pct'].sum()
        
        # Vectorized symbol matching
        # Handle variations like "ICII", "ICII.NS"
        symbols = boosted_df['symbol'].str.replace('.NS', '', regex=False)
        has_signal = symbols.isin(buy_signals) | boosted_df['symbol'].isin(buy_signals)
        
        # Apply boost
        current_weights = boosted_df['weightage_pct'].values.copy()
        boosted_weights = np.where(
            has_signal,
            np.minimum(current_weights * self.boost_multiplier, self.max_boost_weight * 100),
            current_weights
        )
        
        boosted_df['weightage_pct'] = boosted_weights
        
        # Log boosted symbols
        for idx in np.where(has_signal)[0]:
            symbol = boosted_df.iloc[idx]['symbol']
            old_w = current_weights[idx]
            new_w = boosted_weights[idx]
            if new_w > old_w:
                boost_pct = (new_w / old_w - 1) * 100
                logging.info(f"UMA Boosted {symbol}: {old_w:.2f}% → {new_w:.2f}% (+{boost_pct:.1f}%)")
        
        # Renormalize weights
        new_total = boosted_df['weightage_pct'].sum()
        if new_total > 0:
            boosted_df['weightage_pct'] = (boosted_df['weightage_pct'] / new_total) * original_total
        
        # Recalculate units and values if present
        if 'price' in boosted_df.columns and 'value' in boosted_df.columns:
            sip_amount = portfolio_df['value'].sum()
            boosted_df['units'] = np.floor(
                (sip_amount * boosted_df['weightage_pct'] / 100) / boosted_df['price']
            )
            boosted_df['value'] = boosted_df['units'] * boosted_df['price']
        
        return boosted_df


# ============================================================================
# SECTION 11: CONVENIENCE FUNCTIONS
# ============================================================================

def boost_portfolio_with_uma(
    portfolio_df: pd.DataFrame,
    symbols: List[str],
    boost_multiplier: float = 1.15,
    max_boost_weight: float = 0.15,
    lookback_days: int = 200,
    skip_macro: bool = False
) -> pd.DataFrame:
    """
    Main integration point for Pragyam system.
    
    Args:
        portfolio_df: Portfolio DataFrame
        symbols: All symbols to check for signals
        boost_multiplier: Weight multiplier (1.15 = 15% boost)
        max_boost_weight: Maximum weight cap (0.15 = 15%)
        lookback_days: Historical data lookback
        skip_macro: Skip macro regression for faster processing
        
    Returns:
        Portfolio with boosted weights
    """
    try:
        booster = UMAPortfolioBooster(
            lookback_days=lookback_days,
            boost_multiplier=boost_multiplier,
            max_boost_weight=max_boost_weight,
            skip_macro=skip_macro
        )
        
        buy_signals = booster.get_buy_signals(symbols)
        boosted_portfolio = booster.apply_boost(portfolio_df, buy_signals)
        
        n_boosted = len([s for s in portfolio_df['symbol'] 
                        if s.replace('.NS', '') in buy_signals or s in buy_signals])
        logging.info(f"UMA Booster: Applied to {n_boosted}/{len(portfolio_df)} positions")
        
        return boosted_portfolio
        
    except Exception as e:
        logging.error(f"UMA Booster failed: {e} - returning original portfolio")
        return portfolio_df


def get_uma_analysis(symbol: str, lookback_days: int = 200, skip_macro: bool = False) -> Optional[Dict]:
    """
    Get complete UMA analysis for a single symbol.
    """
    if not DATA_LIBS_AVAILABLE:
        logging.error("yfinance required for UMA analysis")
        return None
    
    try:
        fetcher = MarketDataFetcher()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = fetcher.fetch_etf_data(symbol, start_date, end_date)
        
        if df is None or df.empty:
            logging.error(f"No data for {symbol}")
            return None
        
        uma = UnifiedMarketAnalysis()
        results = uma.calculate(df, skip_macro=skip_macro)
        
        results['symbol'] = symbol
        results['analysis_date'] = datetime.now().strftime('%Y-%m-%d')
        results['data_points'] = len(df)
        
        return results
        
    except Exception as e:
        logging.error(f"UMA analysis failed for {symbol}: {e}")
        return None


def clear_uma_cache():
    """Clear all cached UMA data."""
    global _data_cache
    _data_cache.clear()
    logging.info("UMA cache cleared")


# ============================================================================
# SECTION 12: TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Unified Market Analysis (UMA) - YFinance + Stooq Implementation Test")
    print("=" * 80)
    
    if not DATA_LIBS_AVAILABLE:
        print("❌ Libraries not installed. Install with: pip install yfinance pandas_datareader")
        exit(1)
    
    print("✅ Libraries available")
    print()
    
    # Test symbols
    test_symbols = [
        'ICII',        # Should resolve to ICII.NS
        'GOLDIETF',    # Should resolve to GOLDIETF.NS
        'NIFTYBEES.NS' # Already has suffix
    ]
    
    print(f"Testing with symbols: {test_symbols}")
    print()
    
    # Initialize booster
    # We set skip_macro=False to test Stooq connectivity
    booster = UMAPortfolioBooster(
        lookback_days=300,
        boost_multiplier=1.15,
        max_boost_weight=0.15,
        skip_macro=False 
    )
    
    # Get buy signals
    print("Detecting UMA buy signals (fetching Macros from Stooq + ETFs from YF)...")
    start_time = time.time()
    buy_signals = booster.get_buy_signals(test_symbols)
    elapsed = time.time() - start_time
    
    print()
    print(f"Buy signals detected: {len(buy_signals)} (in {elapsed:.2f}s)")
    for signal in buy_signals:
        print(f"  🟢 {signal}")
    
    # Test portfolio boosting
    print()
    print("-" * 40)
    print("Testing Portfolio Boost")
    print("-" * 40)
    
    test_portfolio = pd.DataFrame({
        'symbol': ['ICII', 'ICNI', 'BANKIETF', 'GOLDIETF'],
        'price': [100.0, 150.0, 200.0, 50.0],
        'weightage_pct': [25.0, 25.0, 25.0, 25.0],
        'units': [250, 166, 125, 500],
        'value': [25000, 24900, 25000, 25000]
    })
    
    print("\nOriginal Portfolio:")
    print(test_portfolio[['symbol', 'weightage_pct']].to_string(index=False))
    
    boosted = booster.apply_boost(test_portfolio, buy_signals)
    
    print("\nBoosted Portfolio:")
    print(boosted[['symbol', 'weightage_pct']].to_string(index=False))
    
    print()
    print("=" * 80)
    print("UMA Test Complete!")
    print("=" * 80)


# ============================================================================
# ALIAS FOR app.py COMPATIBILITY
# ============================================================================

def boost_portfolio_with_unified_signals(
    portfolio_df,
    symbols,
    boost_multiplier=1.15,
    max_boost_weight=0.15,
    lookback_days=100
):
    """
    Alias for boost_portfolio_with_uma() - matches app.py expected interface.
    """
    return boost_portfolio_with_uma(
        portfolio_df=portfolio_df,
        symbols=symbols,
        boost_multiplier=boost_multiplier,
        max_boost_weight=max_boost_weight,
        lookback_days=lookback_days,
        skip_macro=False
    )
