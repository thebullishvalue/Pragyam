# ============================================================================
# Unified Market Analysis Booster for Pragyam System
# ============================================================================
#
# This module provides a non-invasive weight boosting layer based on the
# Unified Market Analysis indicator’s buy signals (lime circle conditions).
#
# Features:
# - Uses investpy for independent data fetching
# - Gracefully degrades if data unavailable
# - Doesn’t disrupt core Pragyam logic
# - Boosts weights for symbols with active buy signals
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Set, Optional
import warnings

# Suppress investpy warnings
warnings.filterwarnings('ignore')

# Try to import investpy, gracefully handle if not available
try:
    import investpy
    INVESTPY_AVAILABLE = True
except ImportError:
    INVESTPY_AVAILABLE = False
    logging.warning("investpy not available - Unified Booster will be disabled")

# ============================================================================
# CORE INDICATOR CALCULATIONS (from Unified Market Analysis)
# ============================================================================

class UnifiedMarketAnalysisBooster:
    """
    Calculates buy signals from Unified Market Analysis indicator logic.

    Buy Signal = Lime Circle Condition:
    - is_oversold AND strong_agreement

    Where:
    - is_oversold: Unified oscillator < oversold_threshold
    - strong_agreement: MSF and Macro signals both bullish
    """

    def __init__(self, 
                 lookback_days: int = 100,
                 boost_multiplier: float = 1.15,
                 max_boost_weight: float = 0.15):
        """
        Args:
            lookback_days: Days of historical data to fetch (default 100)
            boost_multiplier: Weight multiplier for buy signals (default 1.15 = 15% boost)
            max_boost_weight: Maximum weight after boost (default 0.15 = 15%)
        """
        self.lookback_days = lookback_days
        self.boost_multiplier = boost_multiplier
        self.max_boost_weight = max_boost_weight
        
        # Indicator parameters (matching Pine Script defaults)
        self.oscillator_length = 20
        self.impact_window = 3
        self.smoothing = 9
        self.z_score_length = 20
        self.oversold_threshold = -60.0
        
        logging.info(f"UnifiedBooster initialized: boost={boost_multiplier}x, max_weight={max_boost_weight}")

    def _fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data using investpy for a single symbol.
        
        Args:
            symbol: NSE symbol (e.g., 'SENSEXIETF')
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not INVESTPY_AVAILABLE:
            return None
        
        try:
            # Remove .NS suffix if present
            clean_symbol = symbol.replace('.NS', '')
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Format dates for investpy
            from_date = start_date.strftime('%d/%m/%Y')
            to_date = end_date.strftime('%d/%m/%Y')
            
            # Fetch ETF data from India
            df = investpy.get_etf_historical_data(
                etf=clean_symbol,
                country='india',
                from_date=from_date,
                to_date=to_date
            )
            
            if df is None or df.empty:
                logging.warning(f"investpy returned empty data for {clean_symbol}")
                return None
            
            # Standardize column names to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Ensure we have all required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logging.warning(f"Missing required columns for {clean_symbol}")
                return None
            
            logging.info(f"Successfully fetched {len(df)} bars for {clean_symbol}")
            return df
            
        except Exception as e:
            logging.debug(f"Failed to fetch {symbol} via investpy: {e}")
            return None

    def _calculate_liquidity_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the core liquidity oscillator (matching Pine Script logic).
        
        Formula:
        1. spread = (high + low) / 2 - open
        2. volume_ma = sma(volume, length)
        3. vwap_spread = sma(spread * volume / volume_ma, length)
        4. price_impact = sma((close - close[impact_window]) * volume / volume_ma, length)
        5. liquidity_score = vwap_spread - price_impact
        6. source_value = close + liquidity_score
        7. oscillator = 200 * (source_value - lowest(source_value, length)) / (highest - lowest) - 100
        """
        length = self.oscillator_length
        impact = self.impact_window
        
        # Step 1: Spread
        df['spread'] = (df['high'] + df['low']) / 2 - df['open']
        
        # Step 2: Volume MA
        df['vol_ma'] = df['volume'].rolling(window=length).mean()
        df['vol_ma'] = df['vol_ma'].replace(0, np.nan)  # Avoid division by zero
        
        # Step 3: VWAP Spread
        df['vwap_spread'] = (df['spread'] * df['volume'] / df['vol_ma']).rolling(window=length).mean()
        
        # Step 4: Price Impact
        close_shifted = df['close'].shift(impact)
        df['price_impact'] = ((df['close'] - close_shifted) * df['volume'] / df['vol_ma']).rolling(window=length).mean()
        
        # Step 5: Liquidity Score
        df['liquidity_score'] = df['vwap_spread'] - df['price_impact']
        
        # Step 6: Source Value
        df['source_value'] = df['close'] + df['liquidity_score']
        
        # Step 7: Oscillator
        df['lowest_value'] = df['source_value'].rolling(window=length).min()
        df['highest_value'] = df['source_value'].rolling(window=length).max()
        
        range_val = df['highest_value'] - df['lowest_value']
        range_val = range_val.replace(0, np.nan)
        
        oscillator = 200 * (df['source_value'] - df['lowest_value']) / range_val - 100
        
        return oscillator.fillna(0)

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()

    def _calculate_stdev(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Standard Deviation"""
        return series.rolling(window=period).std()

    def _calculate_z_score(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculate Z-Score normalization.
        z_score = (value - sma) / stdev
        """
        mean = self._calculate_sma(series, length)
        std = self._calculate_stdev(series, length)
        std = std.replace(0, np.nan)  # Avoid division by zero
        return ((series - mean) / std).fillna(0)

    def _calculate_unified_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the unified oscillator (composite of liquidity + momentum).
        
        Components:
        1. Base oscillator (liquidity)
        2. Smoothed oscillator (EMA)
        3. Z-scored oscillator
        
        Final = (base + smoothed + z_scored) / 3
        """
        # Component 1: Base oscillator
        base_osc = self._calculate_liquidity_oscillator(df)
        
        # Component 2: Smoothed oscillator
        smoothed_osc = self._calculate_ema(base_osc, self.smoothing)
        
        # Component 3: Z-scored oscillator
        z_scored_osc = self._calculate_z_score(base_osc, self.z_score_length)
        
        # Combine components
        unified = (base_osc + smoothed_osc + z_scored_osc) / 3.0
        
        return unified

    def _detect_msf_signal(self, df: pd.DataFrame, unified_osc: pd.Series) -> pd.Series:
        """
        Detect Multi-Scale Flow (MSF) signal.
        
        Bullish MSF:
        - Short EMA > Long EMA (crossover)
        - Unified oscillator rising
        """
        short_ema = self._calculate_ema(df['close'], 9)
        long_ema = self._calculate_ema(df['close'], 21)
        
        crossover = short_ema > long_ema
        rising = unified_osc > unified_osc.shift(1)
        
        msf_bullish = crossover & rising
        
        return msf_bullish

    def _detect_macro_signal(self, df: pd.DataFrame, unified_osc: pd.Series) -> pd.Series:
        """
        Detect Macro trend signal.
        
        Bullish Macro:
        - Price > 200 SMA (uptrend)
        - Unified oscillator > 20-period SMA
        """
        sma_200 = self._calculate_sma(df['close'], 200)
        osc_sma = self._calculate_sma(unified_osc, 20)
        
        uptrend = df['close'] > sma_200
        osc_strong = unified_osc > osc_sma
        
        macro_bullish = uptrend & osc_strong
        
        return macro_bullish

    def _detect_buy_signal(self, df: pd.DataFrame) -> bool:
        """
        Detect the buy signal (lime circle condition).
        
        Buy Signal = is_oversold AND strong_agreement
        
        Where:
        - is_oversold: unified_osc < oversold_threshold
        - strong_agreement: MSF bullish AND Macro bullish
        
        Returns:
            True if buy signal active on latest bar
        """
        try:
            # Calculate unified oscillator
            unified_osc = self._calculate_unified_oscillator(df)
            
            # Check oversold condition
            is_oversold = unified_osc < self.oversold_threshold
            
            # Detect MSF and Macro signals
            msf_bullish = self._detect_msf_signal(df, unified_osc)
            macro_bullish = self._detect_macro_signal(df, unified_osc)
            
            # Strong agreement = both MSF and Macro bullish
            strong_agreement = msf_bullish & macro_bullish
            
            # Buy signal = oversold + strong agreement
            buy_signal = is_oversold & strong_agreement
            
            # Return signal status for latest bar
            if len(buy_signal) > 0:
                return bool(buy_signal.iloc[-1])
            else:
                return False
                
        except Exception as e:
            logging.error(f"Error detecting buy signal: {e}")
            return False

    def get_buy_signals(self, symbols: list) -> Set[str]:
        """
        Fetch data and detect buy signals for a list of symbols.
        
        Args:
            symbols: List of NSE symbols (e.g., ['SENSEXIETF.NS', 'NIFTYIETF.NS'])
        
        Returns:
            Set of symbols with active buy signals
        """
        if not INVESTPY_AVAILABLE:
            logging.warning("investpy not available - no buy signals generated")
            return set()
        
        buy_signals = set()
        
        for symbol in symbols:
            try:
                # Fetch data
                df = self._fetch_symbol_data(symbol)
                
                if df is None or len(df) < 200:
                    # Need at least 200 bars for 200 SMA
                    continue
                
                # Detect buy signal
                if self._detect_buy_signal(df):
                    buy_signals.add(symbol)
                    logging.info(f"✅ Buy signal detected for {symbol}")
                
            except Exception as e:
                logging.debug(f"Error processing {symbol}: {e}")
                continue
        
        logging.info(f"Unified Booster: {len(buy_signals)} buy signals detected from {len(symbols)} symbols")
        return buy_signals

    def apply_boost(self, portfolio_df: pd.DataFrame, buy_signals: Set[str]) -> pd.DataFrame:
        """
        Apply weight boost to symbols with buy signals.
        
        Args:
            portfolio_df: Portfolio DataFrame with 'symbol' and 'weightage_pct' columns
            buy_signals: Set of symbols with active buy signals
        
        Returns:
            Modified portfolio DataFrame with boosted weights
        """
        if portfolio_df.empty or not buy_signals:
            return portfolio_df
        
        # Create a copy to avoid modifying original
        boosted_df = portfolio_df.copy()
        
        # Track original total weight
        original_total = boosted_df['weightage_pct'].sum()
        
        # Apply boost to symbols with buy signals
        for idx, row in boosted_df.iterrows():
            symbol = row['symbol']
            
            # Check if symbol has buy signal (with or without .NS suffix)
            has_signal = (
                symbol in buy_signals or 
                f"{symbol}.NS" in buy_signals or
                symbol.replace('.NS', '') in buy_signals
            )
            
            if has_signal:
                # Apply boost
                current_weight = row['weightage_pct']
                boosted_weight = current_weight * self.boost_multiplier
                
                # Cap at max_boost_weight
                boosted_weight = min(boosted_weight, self.max_boost_weight * 100)
                
                boosted_df.at[idx, 'weightage_pct'] = boosted_weight
                
                logging.info(
                    f"Boosted {symbol}: {current_weight:.2f}% → {boosted_weight:.2f}% "
                    f"(+{((boosted_weight/current_weight - 1) * 100):.1f}%)"
                )
        
        # Renormalize to maintain 100% total
        new_total = boosted_df['weightage_pct'].sum()
        if new_total > 0:
            boosted_df['weightage_pct'] = (boosted_df['weightage_pct'] / new_total) * original_total
        
        # Recalculate units and values
        if 'price' in boosted_df.columns:
            # Assume we're working with the same total capital
            sip_amount = boosted_df['value'].sum()
            boosted_df['units'] = np.floor((sip_amount * boosted_df['weightage_pct'] / 100) / boosted_df['price'])
            boosted_df['value'] = boosted_df['units'] * boosted_df['price']
        
        return boosted_df

# ============================================================================
# CONVENIENCE FUNCTIONS FOR INTEGRATION
# ============================================================================

def boost_portfolio_with_unified_signals(
    portfolio_df: pd.DataFrame,
    symbols: list,
    boost_multiplier: float = 1.15,
    max_boost_weight: float = 0.15,
    lookback_days: int = 100
    ) -> pd.DataFrame:
    """
    Convenience function to boost portfolio weights based on Unified Market Analysis buy signals.
    
    This is the main integration point for the Pragyam system.

    Args:
        portfolio_df: Portfolio DataFrame with 'symbol', 'weightage_pct', etc.
        symbols: List of all symbols to check for buy signals
        boost_multiplier: Weight multiplier for buy signals (default 1.15 = 15% boost)
        max_boost_weight: Maximum weight after boost (default 0.15 = 15%)
        lookback_days: Days of historical data to fetch (default 100)

    Returns:
        Modified portfolio DataFrame with boosted weights
    """
    try:
        # Initialize booster
        booster = UnifiedMarketAnalysisBooster(
            lookback_days=lookback_days,
            boost_multiplier=boost_multiplier,
            max_boost_weight=max_boost_weight
        )
        
        # Get buy signals
        buy_signals = booster.get_buy_signals(symbols)
        
        # Apply boost
        boosted_portfolio = booster.apply_boost(portfolio_df, buy_signals)
        
        # Log summary
        n_boosted = len([s for s in portfolio_df['symbol'] if s in buy_signals or f"{s}.NS" in buy_signals])
        logging.info(f"Unified Booster: Applied to {n_boosted}/{len(portfolio_df)} portfolio positions")
        
        return boosted_portfolio
        
    except Exception as e:
        logging.error(f"Unified Booster failed: {e} - returning original portfolio")
        return portfolio_df

# ============================================================================
# TESTING & DIAGNOSTICS
# ============================================================================

if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Unified Market Analysis Booster - Standalone Test")
    print("=" * 80)

    # Check investpy availability
    if not INVESTPY_AVAILABLE:
        print("❌ investpy not installed. Install with: pip install investpy")
        exit(1)

    print("✅ investpy available")
    print()

    # Test with sample symbols
    test_symbols = [
        'SENSEXIETF.NS',
        'NIFTYIETF.NS',
        'BANKIETF.NS'
    ]

    print(f"Testing with {len(test_symbols)} symbols:")
    for s in test_symbols:
        print(f"  - {s}")
    print()

    # Initialize booster
    booster = UnifiedMarketAnalysisBooster(
        lookback_days=100,
        boost_multiplier=1.15,
        max_boost_weight=0.15
    )

    # Get buy signals
    print("Fetching data and detecting buy signals...")
    buy_signals = booster.get_buy_signals(test_symbols)

    print()
    print(f"Buy signals detected: {len(buy_signals)}")
    for signal in buy_signals:
        print(f"  🟢 {signal}")

    # Test boost application
    print()
    print("Testing boost application...")

    test_portfolio = pd.DataFrame({
        'symbol': ['SENSEXIETF.NS', 'NIFTYIETF.NS', 'BANKIETF.NS'],
        'price': [100, 150, 200],
        'weightage_pct': [33.33, 33.33, 33.34],
        'units': [333, 222, 167],
        'value': [33300, 33300, 33400]
    })

    print("\nOriginal Portfolio:")
    print(test_portfolio[['symbol', 'weightage_pct']])

    boosted_portfolio = booster.apply_boost(test_portfolio, buy_signals)

    print("\nBoosted Portfolio:")
    print(boosted_portfolio[['symbol', 'weightage_pct']])

    print()
    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)

