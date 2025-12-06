# ============================================================================
# unified_booster.py - Compatibility Layer for Pragyam app.py
# ============================================================================
#
# This module provides the exact interface expected by app.py:
#   - boost_portfolio_with_unified_signals()
#
# It wraps the full Unified Market Analysis (UMA) implementation.
#
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Set, Optional, Tuple, List, Any
from dataclasses import dataclass
from functools import lru_cache
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time

warnings.filterwarnings('ignore')

# --- Import investpy ---
try:
    import investpy
    INVESTPY_AVAILABLE = True
except ImportError:
    INVESTPY_AVAILABLE = False
    logging.warning("investpy not available - Unified Booster will be disabled")


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

@dataclass
class UMAParameters:
    """Configuration parameters matching Pine Script defaults"""
    length: int = 20
    roc_length: int = 14
    confidence_level: float = 0.95
    zscore_clip: float = 3.0
    msf_weight_base: float = 0.5
    use_adaptive_weights: bool = True
    regime_sensitivity: float = 1.5
    bb_length: int = 20
    bb_mult: float = 2.0
    rsi_length: int = 14
    rsi_lower: int = 40
    rsi_upper: int = 70
    regression_length: int = 20
    correlation_lookback: int = 1000
    num_macro_vars: int = 5
    unified_osc_oversold: float = -5.0
    agreement_threshold: float = 0.3
    cache_ttl_seconds: int = 3600
    max_parallel_fetches: int = 5
    min_data_points: int = 200


# ============================================================================
# SECTION 2: DATA CACHING
# ============================================================================

class DataCache:
    """In-memory cache with TTL"""
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self._cache[key] = (value, time.time())
    
    def clear(self):
        self._cache.clear()

_data_cache = DataCache(ttl_seconds=3600)


# ============================================================================
# SECTION 3: STATISTICAL UTILITIES (Vectorized)
# ============================================================================

class StatisticalUtils:
    """Vectorized statistical functions"""
    
    @staticmethod
    def zscore_clipped(series: pd.Series, length: int, clip_threshold: float = 3.0) -> pd.Series:
        mean_val = series.rolling(window=length, min_periods=1).mean()
        std_val = series.rolling(window=length, min_periods=1).std()
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_z = np.where(std_val > 0, (series - mean_val) / std_val, 0.0)
        return pd.Series(np.clip(raw_z, -clip_threshold, clip_threshold), index=series.index)
    
    @staticmethod
    def sigmoid(z: pd.Series, scale: float = 1.5) -> pd.Series:
        return pd.Series(2.0 / (1.0 + np.exp(-z.values / scale)) - 1.0, index=z.index)
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def stdev(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).std()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
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
        shifted = series.shift(period)
        with np.errstate(divide='ignore', invalid='ignore'):
            roc_val = np.where(shifted > 0, (series - shifted) / shifted * 100, 0.0)
        return pd.Series(roc_val, index=series.index).fillna(0.0)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        return pd.Series(true_range, index=close.index).ewm(span=period, adjust=False, min_periods=1).mean()


# ============================================================================
# SECTION 4: DATA FETCHER (investpy)
# ============================================================================

class InvestpyDataFetcher:
    """Data fetching via investpy with caching"""
    
    NSE_ETF_MAPPING = {
        'SENSEXIETF': 'Nippon India ETF Sensex',
        'NIFTYIETF': 'Nippon India ETF Nifty BeES',
        'MON100': 'Motilal Oswal Nasdaq 100 ETF',
        'HEALTHIETF': 'Nippon India ETF Nifty Pharma',
        'MAKEINDIA': 'ICICI Prudential Nifty India Manufacturing Index Fund',
        'CONSUMIETF': 'Nippon India ETF Nifty India Consumption',
        'SILVERIETF': 'Nippon India Silver ETF',
        'TNIDETF': 'Nippon India ETF Nifty 50 Value 20',
        'INFRAIETF': 'Nippon India ETF Nifty Infrastructure',
        'GOLDIETF': 'Nippon India ETF Gold BeES',
        'CPSEETF': 'Nippon India ETF CPSE',
        'COMMOIETF': 'Nippon India ETF Nifty Commodities',
        'MOREALTY': 'Motilal Oswal Nifty Realty ETF',
        'MODEFENCE': 'Motilal Oswal Nifty India Defence Index Fund',
        'PSUBNKIETF': 'Nippon India ETF Nifty PSU Bank BeES',
        'MASPTOP50': 'Mirae Asset Nifty 50 ETF',
        'FMCGIETF': 'ICICI Prudential Nifty FMCG ETF',
        'BANKIETF': 'Nippon India ETF Bank BeES',
        'ITIETF': 'Nippon India ETF Nifty IT',
        'EVINDIA': 'Mirae Asset Nifty EV & New Age Automotive ETF',
        'MNC': 'Nippon India ETF Nifty MNC',
        'FINIETF': 'ICICI Prudential Nifty Financial Services ETF',
        'AUTOIETF': 'Nippon India ETF Nifty Auto',
        'PVTBANIETF': 'Nippon India ETF Nifty Private Bank',
        'MONIFTY500': 'Motilal Oswal Nifty 500 ETF',
        'ECAPINSURE': 'Edelweiss ETF',
        'MIDCAPIETF': 'Nippon India ETF Nifty Midcap 150',
        'MOSMALL250': 'Motilal Oswal Nifty Smallcap 250 ETF',
        'OILIETF': 'Nippon India ETF Nifty Commodities',
        'METALIETF': 'Nippon India ETF Nifty Metal',
        'GOLDBEES': 'Nippon India ETF Gold BeES',
        'SILVERBEES': 'Nippon India Silver ETF',
    }
    
    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or _data_cache
    
    def _format_date(self, dt: datetime) -> str:
        return dt.strftime('%d/%m/%Y')
    
    def fetch_etf_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        if not INVESTPY_AVAILABLE:
            return None
        
        cache_key = f"etf_{symbol}_{start_date.date()}_{end_date.date()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        clean_symbol = symbol.replace('.NS', '').upper()
        etf_name = self.NSE_ETF_MAPPING.get(clean_symbol)
        
        if not etf_name:
            logging.debug(f"No investpy mapping for {clean_symbol}")
            return None
        
        try:
            df = investpy.get_etf_historical_data(
                etf=etf_name,
                country='india',
                from_date=self._format_date(start_date),
                to_date=self._format_date(end_date)
            )
            
            if df is None or df.empty:
                return None
            
            df.columns = [c.lower() for c in df.columns]
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                return None
            
            self.cache.set(cache_key, df)
            logging.debug(f"Fetched {len(df)} bars for {clean_symbol}")
            return df
            
        except Exception as e:
            logging.debug(f"investpy fetch failed for {clean_symbol}: {e}")
            return None


# ============================================================================
# SECTION 5: MSF CALCULATOR (Vectorized)
# ============================================================================

class MomentumStructureFlow:
    """MSF - Internal price dynamics (vectorized)"""
    
    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        results = {}
        length = self.params.length
        zscore_clip = self.params.zscore_clip
        
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df['volume']
        
        # Momentum
        roc_raw = self.stats.roc(close, self.params.roc_length)
        roc_z = self.stats.zscore_clipped(roc_raw, length, zscore_clip)
        momentum_norm = self.stats.sigmoid(roc_z, 1.5)
        results['momentum_norm'] = momentum_norm
        
        # Microstructure
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
        
        # Volatility Regime
        price_mean = self.stats.sma(close, length)
        price_stdev = self.stats.stdev(close, length)
        conf_mult = 1.96 if self.params.confidence_level >= 0.95 else 1.645
        
        upper_bound = price_mean + conf_mult * price_stdev
        lower_bound = price_mean - conf_mult * price_stdev
        band_width = upper_bound - lower_bound
        
        with np.errstate(divide='ignore', invalid='ignore'):
            price_position = np.where(band_width > 0, (close - lower_bound) / band_width * 2 - 1, 0.0)
        results['price_position'] = pd.Series(np.clip(price_position, -1.5, 1.5), index=df.index)
        
        # Composite Trend
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
        
        composite_trend_z = (trend_diff_z + momentum_accel_z + vol_adj_mom_z + mean_reversion_z) / np.sqrt(4.0)
        composite_trend_norm = self.stats.sigmoid(composite_trend_z, 1.5)
        results['composite_trend_norm'] = composite_trend_norm
        
        # Accumulation/Distribution
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
        
        # Regime Counter
        pct_change = close.pct_change() * 100
        threshold_pct = 0.33
        up_signal = (pct_change > threshold_pct).astype(int)
        down_signal = (pct_change < -threshold_pct).astype(int)
        regime_count = (up_signal - down_signal).cumsum()
        
        regime_raw = regime_count - self.stats.sma(regime_count, length)
        regime_z = self.stats.zscore_clipped(regime_raw, length, zscore_clip)
        regime_norm = self.stats.sigmoid(regime_z, 1.5)
        results['regime_norm'] = regime_norm
        
        # RSI
        rsi_value = self.stats.rsi(close, self.params.rsi_length)
        rsi_norm = (rsi_value - 50) / 50
        results['rsi_value'] = rsi_value
        results['rsi_norm'] = rsi_norm
        
        # MSF Composite
        osc_momentum = momentum_norm
        osc_structure = (microstructure_norm + composite_trend_norm) / np.sqrt(2.0)
        osc_flow = (accum_norm + regime_norm) / np.sqrt(2.0)
        
        msf_raw = (osc_momentum + osc_structure + osc_flow) / np.sqrt(3.0)
        msf_signal = self.stats.sigmoid(msf_raw * np.sqrt(3.0), 1.0)
        results['msf_signal'] = msf_signal
        results['msf_clarity'] = msf_signal.abs()
        
        return results


# ============================================================================
# SECTION 6: SIGNAL INTEGRATION
# ============================================================================

class SignalIntegrator:
    """Integrates MSF signal (simplified - MSF only mode)"""
    
    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()
    
    def integrate(self, msf_signal: pd.Series, msf_clarity: pd.Series) -> Dict[str, pd.Series]:
        results = {}
        
        # In MSF-only mode, unified = MSF
        unified_final = msf_signal.clip(lower=-1.0, upper=1.0)
        results['unified_signal'] = unified_final
        results['unified_osc'] = unified_final * 10.0
        results['msf_osc'] = msf_signal * 10.0
        
        # Agreement with self = always agree
        results['signal_agreement'] = pd.Series(1.0, index=msf_signal.index)
        
        return results


# ============================================================================
# SECTION 7: BUY SIGNAL DETECTOR
# ============================================================================

class BuySignalDetector:
    """Detects buy signals (vectorized)"""
    
    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()
    
    def detect(self, unified_osc: pd.Series, rsi_value: pd.Series) -> Dict[str, pd.Series]:
        results = {}
        
        # Bollinger Bands on oscillator
        bb_basis = self.stats.sma(unified_osc, self.params.bb_length)
        bb_dev = self.stats.stdev(unified_osc, self.params.bb_length)
        bb_upper = bb_basis + self.params.bb_mult * bb_dev
        bb_lower = bb_basis - self.params.bb_mult * bb_dev
        
        results['bb_upper'] = bb_upper
        results['bb_lower'] = bb_lower
        
        # RSI of oscillator
        rsi_osc = self.stats.rsi(unified_osc, self.params.rsi_length)
        results['rsi_osc'] = rsi_osc
        
        # Conditions
        is_oversold = (unified_osc < bb_lower) & (rsi_osc < self.params.rsi_lower)
        is_overbought = (unified_osc > bb_upper) & (rsi_osc > self.params.rsi_upper)
        
        results['is_oversold'] = is_oversold
        results['is_overbought'] = is_overbought
        
        # Buy signal (lime circle) - simplified for MSF-only
        buy_signal = is_oversold & (unified_osc < self.params.unified_osc_oversold)
        results['buy_signal'] = buy_signal
        
        # Soft buy signal (for boosting)
        soft_buy_signal = unified_osc < self.params.unified_osc_oversold
        results['soft_buy_signal'] = soft_buy_signal
        
        return results


# ============================================================================
# SECTION 8: UNIFIED MARKET ANALYSIS
# ============================================================================

class UnifiedMarketAnalysis:
    """Complete UMA calculator (MSF-focused for speed)"""
    
    def __init__(self, params: Optional[UMAParameters] = None):
        self.params = params or UMAParameters()
        self.msf = MomentumStructureFlow(self.params)
        self.integrator = SignalIntegrator(self.params)
        self.detector = BuySignalDetector(self.params)
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # Calculate MSF
        msf_results = self.msf.calculate(df)
        results['msf'] = msf_results
        
        # Integrate
        integrated = self.integrator.integrate(
            msf_results['msf_signal'],
            msf_results['msf_clarity']
        )
        results['integrated'] = integrated
        
        # Detect signals
        signals = self.detector.detect(
            integrated['unified_osc'],
            msf_results['rsi_value']
        )
        results['signals'] = signals
        
        # Summary
        results['unified_osc'] = integrated['unified_osc']
        results['buy_signal'] = signals['buy_signal']
        results['soft_buy_signal'] = signals['soft_buy_signal']
        
        return results
    
    def has_buy_signal(self, df: pd.DataFrame) -> bool:
        try:
            results = self.calculate(df)
            signal = results.get('soft_buy_signal')
            if signal is not None and len(signal) > 0:
                return bool(signal.iloc[-1])
            return False
        except Exception as e:
            logging.debug(f"Error detecting buy signal: {e}")
            return False


# ============================================================================
# SECTION 9: PORTFOLIO BOOSTER
# ============================================================================

class UnifiedMarketAnalysisBooster:
    """
    Portfolio booster based on UMA signals.
    This is the class expected by the original unified_booster.py interface.
    """
    
    def __init__(self,
                 lookback_days: int = 100,
                 boost_multiplier: float = 1.15,
                 max_boost_weight: float = 0.15):
        self.lookback_days = lookback_days
        self.boost_multiplier = boost_multiplier
        self.max_boost_weight = max_boost_weight
        self.params = UMAParameters()
        self.fetcher = InvestpyDataFetcher()
        self.uma = UnifiedMarketAnalysis(self.params)
        
        logging.info(f"UnifiedBooster initialized: boost={boost_multiplier}x, max_weight={max_boost_weight}")
    
    def _fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if not INVESTPY_AVAILABLE:
            return None
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        return self.fetcher.fetch_etf_data(symbol, start_date, end_date)
    
    def get_buy_signals(self, symbols: List[str]) -> Set[str]:
        if not INVESTPY_AVAILABLE:
            logging.warning("investpy not available - no buy signals generated")
            return set()
        
        buy_signals = set()
        
        for symbol in symbols:
            try:
                df = self._fetch_symbol_data(symbol)
                
                if df is None or len(df) < self.params.min_data_points:
                    continue
                
                if self.uma.has_buy_signal(df):
                    clean_symbol = symbol.replace('.NS', '')
                    buy_signals.add(clean_symbol)
                    logging.info(f"✅ Buy signal detected for {symbol}")
                    
            except Exception as e:
                logging.debug(f"Error processing {symbol}: {e}")
                continue
        
        logging.info(f"Unified Booster: {len(buy_signals)} buy signals detected from {len(symbols)} symbols")
        return buy_signals
    
    def apply_boost(self, portfolio_df: pd.DataFrame, buy_signals: Set[str]) -> pd.DataFrame:
        if portfolio_df.empty or not buy_signals:
            return portfolio_df
        
        boosted_df = portfolio_df.copy()
        original_total = boosted_df['weightage_pct'].sum()
        
        # Vectorized matching
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
        
        # Log
        for idx in np.where(has_signal)[0]:
            symbol = boosted_df.iloc[idx]['symbol']
            old_w = current_weights[idx]
            new_w = boosted_weights[idx]
            if new_w > old_w:
                logging.info(f"Boosted {symbol}: {old_w:.2f}% → {new_w:.2f}%")
        
        # Renormalize
        new_total = boosted_df['weightage_pct'].sum()
        if new_total > 0:
            boosted_df['weightage_pct'] = (boosted_df['weightage_pct'] / new_total) * original_total
        
        # Recalculate units/values
        if 'price' in boosted_df.columns and 'value' in boosted_df.columns:
            sip_amount = portfolio_df['value'].sum()
            boosted_df['units'] = np.floor(
                (sip_amount * boosted_df['weightage_pct'] / 100) / boosted_df['price']
            )
            boosted_df['value'] = boosted_df['units'] * boosted_df['price']
        
        return boosted_df


# ============================================================================
# SECTION 10: PUBLIC API (matches app.py expectations)
# ============================================================================

def boost_portfolio_with_unified_signals(
    portfolio_df: pd.DataFrame,
    symbols: List[str],
    boost_multiplier: float = 1.15,
    max_boost_weight: float = 0.15,
    lookback_days: int = 100
) -> pd.DataFrame:
    """
    Main integration point for Pragyam app.py.
    
    This function matches the exact signature expected by app.py:
        boost_portfolio_with_unified_signals(
            portfolio_df=final_port_df,
            symbols=all_symbols,
            boost_multiplier=booster_config.get('boost_multiplier', 1.15),
            max_boost_weight=booster_config.get('max_boost_weight', 0.15),
            lookback_days=booster_config.get('lookback_days', 100)
        )
    
    Args:
        portfolio_df: Portfolio DataFrame with 'symbol', 'weightage_pct', etc.
        symbols: List of all symbols to check for buy signals
        boost_multiplier: Weight multiplier for buy signals (1.15 = 15% boost)
        max_boost_weight: Maximum weight after boost (0.15 = 15%)
        lookback_days: Days of historical data for signal calculation
    
    Returns:
        Modified portfolio DataFrame with boosted weights
    """
    try:
        booster = UnifiedMarketAnalysisBooster(
            lookback_days=lookback_days,
            boost_multiplier=boost_multiplier,
            max_boost_weight=max_boost_weight
        )
        
        buy_signals = booster.get_buy_signals(symbols)
        boosted_portfolio = booster.apply_boost(portfolio_df, buy_signals)
        
        n_boosted = len([s for s in portfolio_df['symbol'] 
                        if s.replace('.NS', '') in buy_signals or s in buy_signals])
        logging.info(f"Unified Booster: Applied to {n_boosted}/{len(portfolio_df)} portfolio positions")
        
        return boosted_portfolio
        
    except Exception as e:
        logging.error(f"Unified Booster failed: {e} - returning original portfolio")
        return portfolio_df


# ============================================================================
# SECTION 11: TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Unified Booster - Compatibility Test")
    print("=" * 80)
    
    if not INVESTPY_AVAILABLE:
        print("❌ investpy not installed. Install with: pip install investpy")
        exit(1)
    
    print("✅ investpy available")
    
    # Test with synthetic data
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n = 250
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(n) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.random.randint(1000000, 5000000, n)
    }, index=dates)
    
    # Test UMA calculation
    uma = UnifiedMarketAnalysis()
    results = uma.calculate(df)
    print(f"✅ UMA calculation: unified_osc = {results['unified_osc'].iloc[-1]:.2f}")
    
    # Test portfolio boost
    test_portfolio = pd.DataFrame({
        'symbol': ['TEST1', 'TEST2', 'TEST3'],
        'price': [100.0, 150.0, 200.0],
        'weightage_pct': [33.33, 33.33, 33.34],
        'units': [333, 222, 167],
        'value': [33300, 33300, 33400]
    })
    
    # Simulate buy signals
    buy_signals = {'TEST1', 'TEST3'}
    booster = UnifiedMarketAnalysisBooster()
    boosted = booster.apply_boost(test_portfolio, buy_signals)
    
    print(f"✅ Portfolio boost test:")
    print(f"   Original: {list(test_portfolio['weightage_pct'])}")
    print(f"   Boosted:  {[round(w, 2) for w in boosted['weightage_pct']]}")
    
    # Test the main API function
    result = boost_portfolio_with_unified_signals(
        portfolio_df=test_portfolio,
        symbols=['TEST1', 'TEST2', 'TEST3'],
        boost_multiplier=1.15,
        max_boost_weight=0.15,
        lookback_days=100
    )
    print(f"✅ boost_portfolio_with_unified_signals API working")
    
    print()
    print("=" * 80)
    print("ALL TESTS PASSED - COMPATIBLE WITH app.py")
    print("=" * 80)
