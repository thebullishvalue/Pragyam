# ============================================================================
# Unified Market Analysis (UMA) - Complete Implementation for Pragyam System
# BACKEND: INVESTPY (Investing.com)
# ============================================================================
#
# This module implements the full Unified Market Analysis indicator logic
# utilizing investpy for data retrieval instead of yfinance.
#
# Architecture:
# 1. MSF (Momentum Structure Flow) - Internal price dynamics
# 2. MMR (Macro Multiple Regression) - External macro drivers
# 3. Signal Integration
#
# Note on Investpy:
# investpy retrieves data from Investing.com. Ensure you have internet access
# and are not blocked by Cloudflare protections often used by the site.
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Set, Optional, Tuple, List
from dataclasses import dataclass
import warnings
import time

warnings.filterwarnings('ignore')

# Try to import investpy for data fetching
try:
    import investpy
    INVESTPY_AVAILABLE = True
except ImportError:
    INVESTPY_AVAILABLE = False
    logging.warning("investpy not available - UMA Booster will be disabled. Run: pip install investpy")

# ============================================================================
# SECTION 1: CONFIGURATION & PARAMETERS
# ============================================================================

@dataclass
class UMAParameters:
    """Configuration parameters matching Pine Script defaults"""
    # Core Settings
    length: int = 20
    roc_length: int = 14

    # Statistical Settings
    confidence_level: float = 0.95
    zscore_clip: float = 3.0

    # Signal Integration Settings
    msf_weight_base: float = 0.5
    use_adaptive_weights: bool = True
    regime_sensitivity: float = 1.5

    # Oscillator Settings
    bb_length: int = 20
    bb_mult: float = 2.0

    # RSI Settings
    rsi_length: int = 14
    rsi_lower: int = 40
    rsi_upper: int = 70

    # Macro Regression Settings
    regression_length: int = 20
    correlation_lookback: int = 1000
    num_macro_vars: int = 5

    # Buy Signal Thresholds
    unified_osc_oversold: float = -5.0
    agreement_threshold: float = 0.3

# ============================================================================
# SECTION 2: UTILITY FUNCTIONS (Statistical Foundations)
# ============================================================================

class StatisticalUtils:
    """
    Statistical utility functions matching Pine Script implementations.
    All functions are designed for proper normalization and scale independence.
    """

    @staticmethod
    def zscore_clipped(series: pd.Series, length: int, clip_threshold: float = 3.0) -> pd.Series:
        mean_val = series.rolling(window=length).mean()
        std_val = series.rolling(window=length).std()
        std_val = std_val.replace(0, np.nan)
        raw_z = (series - mean_val) / std_val
        raw_z = raw_z.fillna(0)
        return raw_z.clip(lower=-clip_threshold, upper=clip_threshold)

    @staticmethod
    def sigmoid(z: pd.Series, scale: float = 1.5) -> pd.Series:
        return 2.0 / (1.0 + np.exp(-z / scale)) - 1.0

    @staticmethod
    def minmax_normalize(series: pd.Series, length: int) -> pd.Series:
        src_max = series.rolling(window=length).max()
        src_min = series.rolling(window=length).min()
        range_val = src_max - src_min
        range_val = range_val.replace(0, np.nan)
        normalized = 2.0 * (series - src_min) / range_val - 1.0
        return normalized.fillna(0)

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def stdev(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).std()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(100)

    @staticmethod
    def roc(series: pd.Series, period: int) -> pd.Series:
        shifted = series.shift(period)
        return ((series - shifted) / shifted.replace(0, np.nan) * 100).fillna(0)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.ewm(span=period, adjust=False).mean()

    @staticmethod
    def correlation(series1: pd.Series, series2: pd.Series, period: int) -> pd.Series:
        return series1.rolling(window=period).corr(series2)

# ============================================================================
# SECTION 3: MSF - MOMENTUM STRUCTURE FLOW (Internal Dynamics)
# ============================================================================

class MomentumStructureFlow:
    """
    MSF (Momentum Structure Flow) - Analyzes internal price dynamics.
    Calculations remain identical to original implementation.
    """

    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        results = {}
        length = self.params.length
        zscore_clip = self.params.zscore_clip
        
        # Ensure column names are lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # === 3.1 Momentum Component (ROC-based) ===
        roc_raw = self.stats.roc(df['close'], self.params.roc_length)
        roc_z = self.stats.zscore_clipped(roc_raw, length, zscore_clip)
        momentum_norm = self.stats.sigmoid(roc_z, 1.5)
        results['momentum_norm'] = momentum_norm
        
        # === 3.2 Market Microstructure Component ===
        intrabar_direction = (df['high'] + df['low']) / 2 - df['open']
        vol_ma = self.stats.sma(df['volume'], length)
        vol_ratio = (df['volume'] / vol_ma.replace(0, np.nan)).fillna(1.0)
        vw_direction = self.stats.sma(intrabar_direction * vol_ratio, length)
        price_change_impact = df['close'] - df['close'].shift(5)
        vw_impact = self.stats.sma(price_change_impact * vol_ratio, length)
        
        microstructure_raw = vw_direction - vw_impact
        microstructure_z = self.stats.zscore_clipped(microstructure_raw, length, zscore_clip)
        microstructure_norm = self.stats.sigmoid(microstructure_z, 1.5)
        results['microstructure_norm'] = microstructure_norm
        
        # === 3.3 Volatility Regime ===
        price_mean = self.stats.sma(df['close'], length)
        price_stdev = self.stats.stdev(df['close'], length)
        conf_mult = 1.96 if self.params.confidence_level >= 0.95 else 1.645
        
        upper_bound = price_mean + conf_mult * price_stdev
        lower_bound = price_mean - conf_mult * price_stdev
        band_width = upper_bound - lower_bound
        
        price_position = ((df['close'] - lower_bound) / band_width.replace(0, np.nan) * 2 - 1).fillna(0)
        price_position_clipped = price_position.clip(lower=-1.5, upper=1.5)
        results['price_position'] = price_position_clipped
        results['upper_bound'] = upper_bound
        results['lower_bound'] = lower_bound
        
        # === 3.4 Composite Trend ===
        trend_fast = self.stats.sma(df['close'], 5)
        trend_slow = self.stats.sma(df['close'], length)
        trend_diff_z = self.stats.zscore_clipped(trend_fast - trend_slow, length, zscore_clip)
        
        price_change_5 = df['close'].diff(5)
        momentum_accel_raw = price_change_5.diff(5)
        momentum_accel_z = self.stats.zscore_clipped(momentum_accel_raw, length, zscore_clip)
        
        atr_val = self.stats.atr(df['high'], df['low'], df['close'], 14)
        vol_adj_mom_raw = (price_change_5 / atr_val.replace(0, np.nan)).fillna(0)
        vol_adj_mom_z = self.stats.zscore_clipped(vol_adj_mom_raw, length, zscore_clip)
        
        mean_reversion_z = self.stats.zscore_clipped(df['close'] - price_mean, length, zscore_clip)
        
        composite_trend_z = (trend_diff_z + momentum_accel_z + vol_adj_mom_z + mean_reversion_z) / np.sqrt(4.0)
        composite_trend_norm = self.stats.sigmoid(composite_trend_z, 1.5)
        results['composite_trend_norm'] = composite_trend_norm
        
        # === 3.5 Accumulation/Distribution ===
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        close_up = df['close'] > df['close'].shift(1)
        mf_positive = money_flow.where(close_up, 0)
        mf_negative = money_flow.where(~close_up, 0)
        
        mf_pos_smooth = self.stats.sma(mf_positive, length)
        mf_neg_smooth = self.stats.sma(mf_negative, length)
        mf_total = mf_pos_smooth + mf_neg_smooth
        
        accum_ratio = (mf_pos_smooth / mf_total.replace(0, np.nan)).fillna(0.5)
        accum_norm = 2.0 * (accum_ratio - 0.5)
        results['accum_norm'] = accum_norm
        
        # === 3.6 Regime Counter ===
        pct_change = df['close'].pct_change() * 100
        threshold_pct = 0.33
        regime_count = pd.Series(0.0, index=df.index)
        count = 0.0
        
        # Optimization: Vectorize if possible, but keeping iterative for logic preservation
        for i in range(len(pct_change)):
            if pd.isna(pct_change.iloc[i]): continue
            if pct_change.iloc[i] > threshold_pct: count += 1
            elif pct_change.iloc[i] < -threshold_pct: count -= 1
            regime_count.iloc[i] = count
        
        regime_raw = regime_count - self.stats.sma(regime_count, length)
        regime_z = self.stats.zscore_clipped(regime_raw, length, zscore_clip)
        regime_norm = self.stats.sigmoid(regime_z, 1.5)
        results['regime_norm'] = regime_norm
        
        # === 3.7 RSI Component ===
        rsi_value = self.stats.rsi(df['close'], self.params.rsi_length)
        rsi_norm = (rsi_value - 50) / 50
        results['rsi_value'] = rsi_value
        results['rsi_norm'] = rsi_norm
        
        # === 3.8 MSF Composite Signal ===
        osc_momentum = momentum_norm
        osc_structure = (microstructure_norm + composite_trend_norm) / np.sqrt(2.0)
        osc_flow = (accum_norm + regime_norm) / np.sqrt(2.0)
        
        msf_raw = (osc_momentum + osc_structure + osc_flow) / np.sqrt(3.0)
        msf_signal = self.stats.sigmoid(msf_raw * np.sqrt(3.0), 1.0)
        results['msf_signal'] = msf_signal
        results['msf_clarity'] = msf_signal.abs()
        
        return results

# ============================================================================
# SECTION 4: MMR - MACRO MULTIPLE REGRESSION (External Drivers)
# ============================================================================

class MacroMultipleRegression:
    """
    MMR (Macro Multiple Regression) - Analyzes external macro drivers.
    Adapted to use investpy search parameters.
    """

    # Investpy requires specific Name, Country, and Type (stock, bond, commodity, etc)
    MACRO_CONFIG = {
        'US10Y': {'name': 'U.S. 10Y', 'country': 'united states', 'type': 'bond'},
        'US02Y': {'name': 'U.S. 2Y', 'country': 'united states', 'type': 'bond'},
        'US30Y': {'name': 'U.S. 30Y', 'country': 'united states', 'type': 'bond'},
        'JP10Y': {'name': 'Japan 10Y', 'country': 'japan', 'type': 'bond'},
        'JP02Y': {'name': 'Japan 2Y', 'country': 'japan', 'type': 'bond'},
        'CN10Y': {'name': 'China 10Y', 'country': 'china', 'type': 'bond'},
        'CN02Y': {'name': 'China 2Y', 'country': 'china', 'type': 'bond'},
        'EU10Y': {'name': 'Germany 10Y', 'country': 'germany', 'type': 'bond'},
        'EU02Y': {'name': 'Germany 2Y', 'country': 'germany', 'type': 'bond'},
        'GB10Y': {'name': 'U.K. 10Y', 'country': 'united kingdom', 'type': 'bond'},
        'GB02Y': {'name': 'U.K. 2Y', 'country': 'united kingdom', 'type': 'bond'},
        'IN10Y': {'name': 'India 10Y', 'country': 'india', 'type': 'bond'},
        'IN02Y': {'name': 'India 2Y', 'country': 'india', 'type': 'bond'},
        'DXY': {'name': 'US Dollar Index', 'country': 'united states', 'type': 'index'},
        'GOLD': {'name': 'Gold', 'country': None, 'type': 'commodity'},
        'SILVER': {'name': 'Silver', 'country': None, 'type': 'commodity'},
        'OIL': {'name': 'Crude Oil', 'country': None, 'type': 'commodity'},
        'USDINR': {'name': 'USD/INR', 'country': None, 'type': 'currency_cross'},
        'EURINR': {'name': 'EUR/INR', 'country': None, 'type': 'currency_cross'},
        'GBPINR': {'name': 'GBP/INR', 'country': None, 'type': 'currency_cross'},
        'JPYINR': {'name': 'JPY/INR', 'country': None, 'type': 'currency_cross'},
    }

    DISPLAY_NAMES = {k: k for k in MACRO_CONFIG.keys()}

    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()

    def fetch_macro_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.Series]:
        """
        Fetch macro data using investpy.
        Dates must be converted to 'dd/mm/yyyy' string format.
        """
        if not INVESTPY_AVAILABLE:
            logging.warning("investpy not available - skipping macro data fetch")
            return {}
        
        macro_data = {}
        
        # Convert dates to investpy format (dd/mm/yyyy)
        from_date = start_date.strftime('%d/%m/%Y')
        to_date = end_date.strftime('%d/%m/%Y')
        
        for key, config in self.MACRO_CONFIG.items():
            try:
                data = None
                name = config['name']
                country = config['country']
                asset_type = config['type']
                
                # Investpy separates functions by asset class
                if asset_type == 'bond':
                    data = investpy.get_bond_historical_data(bond=name, country=country, from_date=from_date, to_date=to_date)
                elif asset_type == 'index':
                    data = investpy.get_index_historical_data(index=name, country=country, from_date=from_date, to_date=to_date)
                elif asset_type == 'commodity':
                    data = investpy.get_commodity_historical_data(commodity=name, from_date=from_date, to_date=to_date)
                elif asset_type == 'currency_cross':
                    data = investpy.get_currency_cross_historical_data(currency_cross=name, from_date=from_date, to_date=to_date)
                
                if data is not None and not data.empty:
                    # Investpy returns capitalized 'Close'. We normalize to 'close' later if needed, 
                    # but here we just need the Series.
                    macro_data[key] = data['Close']
                    logging.debug(f"Fetched {key}: {len(data)} bars")
                    # Rate limit kindness
                    time.sleep(0.5) 
                
            except Exception as e:
                logging.debug(f"Failed to fetch {key} via investpy: {e}")
                continue
        
        return macro_data

    def calculate(self, df: pd.DataFrame, macro_data: Dict[str, pd.Series]) -> Dict[str, any]:
        """
        Calculate MMR signal based on macro regression.
        (Logic remains identical to original, only data source changed)
        """
        results = {}
        
        if not macro_data:
            results['mmr_signal'] = pd.Series(0.0, index=df.index)
            results['mmr_clarity'] = pd.Series(0.0, index=df.index)
            results['model_r2'] = 0.0
            results['top_drivers'] = []
            return results
        
        target = df['close'] if 'close' in df.columns else df['Close']
        
        # Align all data to target index
        aligned_macro = {}
        for name, series in macro_data.items():
            # Reindex to target and forward fill
            # Note: investpy index is usually datetime, but verify timezone compatibility if needed
            aligned = series.reindex(target.index).ffill()
            if aligned.notna().sum() > self.params.correlation_lookback * 0.5:
                aligned_macro[name] = aligned
        
        if not aligned_macro:
            results['mmr_signal'] = pd.Series(0.0, index=df.index)
            results['mmr_clarity'] = pd.Series(0.0, index=df.index)
            results['model_r2'] = 0.0
            results['top_drivers'] = []
            return results
        
        # Calculate correlations
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
        
        # Sort and select top drivers
        sorted_vars = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        top_vars = sorted_vars[:self.params.num_macro_vars]
        
        results['top_drivers'] = [
            {'name': name, 'display': self.DISPLAY_NAMES.get(name, name), 'correlation': corr}
            for name, corr in top_vars
        ]
        
        # Build regression
        predictions = []
        weights = []
        
        for name, corr in top_vars:
            if name not in aligned_macro: continue 
            pred, r2 = self._regression_predict(aligned_macro[name], target, self.params.regression_length)
            predictions.append(pred)
            weights.append(r2)
        
        if not predictions:
            results['mmr_signal'] = pd.Series(0.0, index=df.index)
            results['mmr_clarity'] = pd.Series(0.0, index=df.index)
            results['model_r2'] = 0.0
            return results
        
        total_weight = sum(weights)
        if total_weight > 0:
            y_predicted = sum(p * w for p, w in zip(predictions, weights)) / total_weight
            model_r2 = sum(w * w for w in weights) / total_weight
        else:
            y_predicted = predictions[0]
            model_r2 = weights[0] if weights else 0
        
        results['model_r2'] = model_r2
        results['y_predicted'] = y_predicted
        
        deviation = target - y_predicted
        deviation_z = self.stats.zscore_clipped(deviation, self.params.length, self.params.zscore_clip)
        mmr_signal = self.stats.sigmoid(deviation_z, 1.5)
        
        results['mmr_signal'] = mmr_signal
        results['mmr_clarity'] = mmr_signal.abs()
        
        return results

    def _regression_predict(self, x: pd.Series, y: pd.Series, length: int) -> Tuple[pd.Series, float]:
        x_mean = self.stats.sma(x, length)
        y_mean = self.stats.sma(y, length)
        x_std = self.stats.stdev(x, length)
        y_std = self.stats.stdev(y, length)
        
        corr = self.stats.correlation(x, y, length)
        slope = corr * (y_std / x_std.replace(0, np.nan))
        slope = slope.fillna(0)
        intercept = y_mean - slope * x_mean
        prediction = x * slope + intercept
        
        recent_corr = corr.dropna()
        r2 = recent_corr.iloc[-1] ** 2 if len(recent_corr) > 0 else 0
        return prediction, r2

# ============================================================================
# SECTION 5: SIGNAL INTEGRATION
# ============================================================================

class SignalIntegrator:
    """Integrates MSF and MMR signals into unified output."""
    
    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()

    def integrate(self, msf_signal, msf_clarity, mmr_signal, mmr_clarity, mmr_quality):
        results = {}
        
        msf_clarity_scaled = msf_clarity ** self.params.regime_sensitivity
        mmr_clarity_scaled = (mmr_clarity * np.sqrt(mmr_quality)) ** self.params.regime_sensitivity
        clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001
        
        msf_weight_adaptive = msf_clarity_scaled / clarity_sum
        mmr_weight_adaptive = mmr_clarity_scaled / clarity_sum
        
        if self.params.use_adaptive_weights:
            msf_weight_final = 0.5 * self.params.msf_weight_base + 0.5 * msf_weight_adaptive
            mmr_weight_final = 0.5 * (1.0 - self.params.msf_weight_base) + 0.5 * mmr_weight_adaptive
        else:
            msf_weight_final = pd.Series(self.params.msf_weight_base, index=msf_signal.index)
            mmr_weight_final = pd.Series(1.0 - self.params.msf_weight_base, index=msf_signal.index)
        
        weight_sum = msf_weight_final + mmr_weight_final
        msf_weight_norm = msf_weight_final / weight_sum
        mmr_weight_norm = mmr_weight_final / weight_sum
        
        results['msf_weight'] = msf_weight_norm
        results['mmr_weight'] = mmr_weight_norm
        
        unified_signal = msf_weight_norm * msf_signal + mmr_weight_norm * mmr_signal
        
        signal_agreement = msf_signal * mmr_signal
        agreement_strength = signal_agreement.abs()
        results['signal_agreement'] = signal_agreement
        
        agreement_multiplier = pd.Series(1.0, index=msf_signal.index)
        agreement_multiplier = agreement_multiplier.where(signal_agreement <= 0, 1.0 + 0.2 * agreement_strength)
        agreement_multiplier = agreement_multiplier.where(signal_agreement >= 0, 1.0 - 0.1 * agreement_strength)
        
        unified_final = (unified_signal * agreement_multiplier).clip(lower=-1.0, upper=1.0)
        results['unified_signal'] = unified_final
        results['unified_osc'] = unified_final * 10.0
        results['msf_osc'] = msf_signal * 10.0
        results['mmr_osc'] = mmr_signal * 10.0
        
        return results

# ============================================================================
# SECTION 6: BUY SIGNAL DETECTION
# ============================================================================

class BuySignalDetector:
    """Detects buy signals based on Unified Market Analysis criteria."""

    def __init__(self, params: UMAParameters):
        self.params = params
        self.stats = StatisticalUtils()

    def detect(self, unified_osc, signal_agreement, rsi_value):
        results = {}
        
        bb_basis = self.stats.sma(unified_osc, self.params.bb_length)
        bb_dev = self.stats.stdev(unified_osc, self.params.bb_length)
        bb_upper = bb_basis + self.params.bb_mult * bb_dev
        bb_lower = bb_basis - self.params.bb_mult * bb_dev
        
        results['bb_upper'] = bb_upper
        results['bb_lower'] = bb_lower
        
        rsi_osc = self.stats.rsi(unified_osc, self.params.rsi_length)
        results['rsi_osc'] = rsi_osc
        
        is_oversold = (unified_osc < bb_lower) & (rsi_osc < self.params.rsi_lower)
        is_overbought = (unified_osc > bb_upper) & (rsi_osc > self.params.rsi_upper)
        results['is_oversold'] = is_oversold
        results['is_overbought'] = is_overbought
        
        strong_agreement = signal_agreement > self.params.agreement_threshold
        results['strong_agreement'] = strong_agreement
        
        buy_signal = is_oversold & strong_agreement & (unified_osc < self.params.unified_osc_oversold)
        results['buy_signal'] = buy_signal
        
        sell_signal = is_overbought & strong_agreement & (unified_osc > -self.params.unified_osc_oversold)
        results['sell_signal'] = sell_signal
        
        soft_buy_signal = strong_agreement & (unified_osc < self.params.unified_osc_oversold)
        results['soft_buy_signal'] = soft_buy_signal
        
        return results

# ============================================================================
# SECTION 7: MAIN UMA CALCULATOR
# ============================================================================

class UnifiedMarketAnalysis:
    """Complete Unified Market Analysis calculator."""

    def __init__(self, params: Optional[UMAParameters] = None):
        self.params = params or UMAParameters()
        self.msf = MomentumStructureFlow(self.params)
        self.mmr = MacroMultipleRegression(self.params)
        self.integrator = SignalIntegrator(self.params)
        self.detector = BuySignalDetector(self.params)

    def calculate(self, df: pd.DataFrame, macro_data: Optional[Dict[str, pd.Series]] = None) -> Dict[str, any]:
        results = {}
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # === Calculate MSF ===
        msf_results = self.msf.calculate(df)
        results['msf'] = msf_results
        
        # === Calculate MMR ===
        if macro_data is None:
            if len(df) > 0:
                end_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
                # Default to 365 days history for macro
                start_date = end_date - timedelta(days=365)
                macro_data = self.mmr.fetch_macro_data(start_date, end_date)
            else:
                macro_data = {}
        
        mmr_results = self.mmr.calculate(df, macro_data)
        results['mmr'] = mmr_results
        
        # === Integrate Signals ===
        integrated = self.integrator.integrate(
            msf_results['msf_signal'], msf_results['msf_clarity'],
            mmr_results['mmr_signal'], mmr_results['mmr_clarity'],
            mmr_results['model_r2']
        )
        results['integrated'] = integrated
        
        # === Detect Buy/Sell Signals ===
        signals = self.detector.detect(
            integrated['unified_osc'], integrated['signal_agreement'], msf_results['rsi_value']
        )
        results['signals'] = signals
        
        # === Summary ===
        results['unified_osc'] = integrated['unified_osc']
        results['buy_signal'] = signals['buy_signal']
        results['soft_buy_signal'] = signals['soft_buy_signal']
        results['sell_signal'] = signals['sell_signal']
        
        return results

    def has_buy_signal(self, df: pd.DataFrame, macro_data: Optional[Dict] = None) -> bool:
        try:
            results = self.calculate(df, macro_data)
            soft_signal = results.get('soft_buy_signal')
            if soft_signal is not None and len(soft_signal) > 0:
                return bool(soft_signal.iloc[-1])
            return False
        except Exception as e:
            logging.error(f"Error detecting buy signal: {e}")
            return False

# ============================================================================
# SECTION 8: PORTFOLIO BOOSTER
# ============================================================================

class UMAPortfolioBooster:
    """
    Portfolio weight booster based on Unified Market Analysis signals.
    Backended by Investpy.
    """

    def __init__(self, lookback_days: int = 200, boost_multiplier: float = 1.15,
                 max_boost_weight: float = 0.15, params: Optional[UMAParameters] = None):
        self.lookback_days = lookback_days
        self.boost_multiplier = boost_multiplier
        self.max_boost_weight = max_boost_weight
        self.params = params or UMAParameters()
        self.uma = UnifiedMarketAnalysis(self.params)
        
        logging.info(f"UMA Booster (Investpy) initialized: boost={boost_multiplier}x")

    def _fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a symbol using investpy.
        Assumes NSE (India) context if suffix is .NS or implicit.
        """
        if not INVESTPY_AVAILABLE:
            return None
        
        try:
            # Investpy takes name ('RELIANCE') and country ('india')
            # It does not want '.NS'
            clean_symbol = symbol.replace('.NS', '')
            country = 'india' # Assuming India based on Pragyam system context
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            from_str = start_date.strftime('%d/%m/%Y')
            to_str = end_date.strftime('%d/%m/%Y')
            
            # Note: This looks up by symbol/ticker, not full company name
            df = investpy.get_stock_historical_data(
                stock=clean_symbol,
                country=country,
                from_date=from_str,
                to_date=to_str
            )
            
            if df is None or df.empty:
                return None
            
            # Standardize
            df.columns = [c.lower() for c in df.columns]
            
            # Investpy index might not be datetime by default in older versions
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            return df
            
        except RuntimeError as re:
            logging.debug(f"Investpy Error (Symbol not found or blocked): {symbol} - {re}")
            return None
        except Exception as e:
            logging.debug(f"Failed to fetch {symbol}: {e}")
            return None

    def get_buy_signals(self, symbols: List[str]) -> Set[str]:
        if not INVESTPY_AVAILABLE:
            logging.warning("investpy not available - no buy signals generated")
            return set()
        
        buy_signals = set()
        
        # Fetch macro data once
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        macro_data = self.uma.mmr.fetch_macro_data(start_date, end_date)
        
        for symbol in symbols:
            try:
                # Rate limit kindness for scraping
                time.sleep(0.5)
                
                df = self._fetch_symbol_data(symbol)
                
                if df is None or len(df) < 100: # Need history
                    continue
                
                if self.uma.has_buy_signal(df, macro_data):
                    clean_symbol = symbol.replace('.NS', '')
                    buy_signals.add(clean_symbol)
                    logging.info(f"✅ UMA Buy signal: {symbol}")
                    
            except Exception as e:
                logging.debug(f"Error processing {symbol}: {e}")
                continue
        
        logging.info(f"UMA Booster: {len(buy_signals)} buy signals from {len(symbols)} symbols")
        return buy_signals

    def apply_boost(self, portfolio_df: pd.DataFrame, buy_signals: Set[str]) -> pd.DataFrame:
        """Apply weight boost to symbols with buy signals."""
        if portfolio_df.empty or not buy_signals:
            return portfolio_df
        
        boosted_df = portfolio_df.copy()
        original_total = boosted_df['weightage_pct'].sum()
        
        for idx, row in boosted_df.iterrows():
            symbol = row['symbol']
            
            # Check various formats (RELIANCE, RELIANCE.NS)
            clean_sym = symbol.replace('.NS', '')
            has_signal = clean_sym in buy_signals
            
            if has_signal:
                current_weight = row['weightage_pct']
                boosted_weight = current_weight * self.boost_multiplier
                boosted_weight = min(boosted_weight, self.max_boost_weight * 100)
                boosted_df.at[idx, 'weightage_pct'] = boosted_weight
                
                boost_pct = (boosted_weight / current_weight - 1) * 100
                logging.info(f"UMA Boosted {symbol}: {current_weight:.2f}% -> {boosted_weight:.2f}% (+{boost_pct:.1f}%)")
        
        # Renormalize
        new_total = boosted_df['weightage_pct'].sum()
        if new_total > 0:
            boosted_df['weightage_pct'] = (boosted_df['weightage_pct'] / new_total) * original_total
        
        # Recalculate units/value if columns exist
        if 'price' in boosted_df.columns and 'value' in boosted_df.columns:
            sip_amount = portfolio_df['value'].sum()
            boosted_df['units'] = np.floor(
                (sip_amount * boosted_df['weightage_pct'] / 100) / boosted_df['price']
            )
            boosted_df['value'] = boosted_df['units'] * boosted_df['price']
        
        return boosted_df

# ============================================================================
# SECTION 9: CONVENIENCE & TESTING
# ============================================================================

def get_uma_analysis(symbol: str, lookback_days: int = 200) -> Optional[Dict]:
    if not INVESTPY_AVAILABLE: return None
    try:
        clean_symbol = symbol.replace('.NS', '')
        country = 'india'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        from_str, to_str = start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y')
        
        df = investpy.get_stock_historical_data(stock=clean_symbol, country=country, from_date=from_str, to_date=to_str)
        if df is None or df.empty: return None
        
        df.columns = [c.lower() for c in df.columns]
        
        uma = UnifiedMarketAnalysis()
        results = uma.calculate(df)
        results['symbol'] = symbol
        results['analysis_date'] = datetime.now().strftime('%Y-%m-%d')
        return results
    except Exception as e:
        logging.error(f"UMA analysis failed for {symbol}: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    print("=== UMA (Investpy Backend) Test ===")
    
    if not INVESTPY_AVAILABLE:
        print("investpy not installed.")
        exit(1)

    # Test symbols (Names must match Investing.com names, usually standard tickers work for India)
    # Note: investpy is case/name sensitive. ETF names on Investing.com often differ from tickers.
    # We use standard stocks here for reliability of the test.
    test_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'SBIN.NS']
    
    booster = UMAPortfolioBooster(lookback_days=300)
    
    print("Fetching signals (this may take time due to web scraping)...")
    signals = booster.get_buy_signals(test_symbols)
    
    print("\nBuy Signals Detected:")
    print(signals if signals else "None")
    
    # Simple Portfolio Mockup
    pf = pd.DataFrame({
        'symbol': ['RELIANCE.NS', 'TCS.NS'],
        'weightage_pct': [50.0, 50.0],
        'price': [2500, 3500],
        'value': [50000, 50000]
    })
    
    boosted = booster.apply_boost(pf, signals)
    print("\nBoosted Portfolio:")
    print(boosted[['symbol', 'weightage_pct']])