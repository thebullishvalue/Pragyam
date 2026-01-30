"""
Backtest Engine - Unified Integration Module for Pragyam
=========================================================

This module provides institutional-grade backtesting capabilities that integrate
seamlessly with Pragyam's data pipeline and strategy ecosystem.

Key Features:
- Unified data fetching through backdata.py (shared resources)
- Dynamic strategy selection based on performance metrics
- SIP Mode: Top 4 strategies by Calmar Ratio
- Swing Mode: Top 4 strategies by Sortino Ratio
- Performance-optimized with intelligent caching

Author: Hemrek Capital
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("BacktestEngine")


# ============================================================================
# PERFORMANCE METRICS CALCULATOR
# ============================================================================

class PerformanceMetrics:
    """
    Institutional-grade performance metrics calculator.
    Computes all major risk-adjusted return metrics with proper bounds checking.
    """
    
    @staticmethod
    def calculate(
        daily_values: pd.DataFrame,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252.0
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics from daily portfolio values.
        
        Args:
            daily_values: DataFrame with 'date', 'value', 'investment' columns
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Trading periods per year (default 252)
            
        Returns:
            Dictionary of performance metrics
        """
        if daily_values.empty or len(daily_values) < 2:
            return PerformanceMetrics._empty_metrics()
        
        # Extract values
        values = daily_values['value'].values
        initial_value = daily_values['investment'].iloc[0]
        final_value = values[-1]
        
        # Sanity check: values should not be zero or negative
        if initial_value <= 0 or final_value <= 0:
            return PerformanceMetrics._empty_metrics()
        
        # Sanity check: values should not have wild swings (>99% single-day moves indicate data issues)
        values_series = pd.Series(values)
        daily_returns = values_series.pct_change().dropna()
        
        # Filter out invalid returns
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Cap extreme returns at +/- 50% per day (anything beyond indicates data error)
        daily_returns = daily_returns.clip(-0.5, 0.5)
        
        if daily_returns.empty or len(daily_returns) < 2:
            return PerformanceMetrics._empty_metrics()
        
        # Basic metrics - calculate from actual values, not returns
        total_return = (final_value - initial_value) / initial_value
        
        # Time-adjusted metrics
        n_periods = len(daily_values)
        years = n_periods / periods_per_year
        
        # CAGR - use this for annualized return (more stable than compounding daily)
        if years > 0 and initial_value > 0 and final_value > 0:
            cagr = (final_value / initial_value) ** (1 / years) - 1
        else:
            cagr = 0
        
        # Use CAGR as annualized return (more stable)
        ann_return = cagr
        
        # Volatility
        ann_factor = np.sqrt(periods_per_year)
        volatility = daily_returns.std() * ann_factor
        
        # Sharpe Ratio
        excess_return = ann_return - risk_free_rate
        sharpe = excess_return / volatility if volatility > 0.001 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * ann_factor
            sortino = excess_return / downside_std if downside_std > 0.001 else 0
        else:
            sortino = 0
        
        # Maximum Drawdown - calculate from values directly
        cumulative_max = values_series.expanding(min_periods=1).max()
        drawdown_series = (values_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown_series.min()
        
        # Calmar Ratio - annual return / max drawdown
        if max_drawdown < -0.001:  # At least 0.1% drawdown
            calmar = ann_return / abs(max_drawdown)
        else:
            calmar = 0
        
        # Sanity bounds on ratios (realistic range: -10 to +10)
        sharpe = np.clip(sharpe, -10, 10)
        sortino = np.clip(sortino, -10, 10)
        calmar = np.clip(calmar, -10, 10)
        
        # Win Rate
        win_rate = (daily_returns > 0).mean()
        
        # Best/Worst Days
        best_day = daily_returns.max()
        worst_day = daily_returns.min()
        
        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'best_day': best_day,
            'worst_day': worst_day,
            'trading_days': n_periods,
            'final_value': final_value
        }
    
    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        """Return empty metrics structure."""
        return {
            'total_return': 0, 'annualized_return': 0, 'cagr': 0,
            'volatility': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
            'max_drawdown': 0, 'calmar_ratio': 0, 'win_rate': 0,
            'best_day': 0, 'worst_day': 0, 'trading_days': 0, 'final_value': 0
        }


# ============================================================================
# DATA CACHE MANAGER
# ============================================================================

class DataCacheManager:
    """
    Manages shared data caching to avoid redundant API calls.
    Thread-safe singleton pattern for cross-module data sharing.
    """
    
    _instance = None
    _cache: Dict[str, Any] = {}
    _cache_timestamps: Dict[str, datetime] = {}
    _cache_ttl_minutes: int = 30  # Cache TTL in minutes
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            cls._instance._cache_timestamps = {}
        return cls._instance
    
    def _generate_key(self, symbols: List[str], start_date: datetime, end_date: datetime) -> str:
        """Generate unique cache key from parameters."""
        key_str = f"{'-'.join(sorted(symbols))}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Optional[List[Tuple[datetime, pd.DataFrame]]]:
        """Retrieve cached data if valid."""
        key = self._generate_key(symbols, start_date, end_date)
        
        if key in self._cache:
            cached_time = self._cache_timestamps.get(key)
            if cached_time and (datetime.now() - cached_time).total_seconds() < self._cache_ttl_minutes * 60:
                logger.info(f"Cache HIT for key: {key[:8]}...")
                return self._cache[key]
            else:
                # Expired - remove from cache
                self._cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
        
        logger.info(f"Cache MISS for key: {key[:8]}...")
        return None
    
    def set(self, symbols: List[str], start_date: datetime, end_date: datetime, data: List[Tuple[datetime, pd.DataFrame]]):
        """Store data in cache."""
        key = self._generate_key(symbols, start_date, end_date)
        self._cache[key] = data
        self._cache_timestamps[key] = datetime.now()
        logger.info(f"Cached data for key: {key[:8]}... ({len(data)} snapshots)")
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")


# ============================================================================
# UNIFIED BACKTEST ENGINE
# ============================================================================

class UnifiedBacktestEngine:
    """
    Unified backtesting engine that integrates with Pragyam's ecosystem.
    
    Supports both SIP (Systematic Investment) and Swing Trading modes
    with dynamic strategy selection based on performance metrics.
    """
    
    def __init__(
        self,
        capital: float = 10_000_000,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize the backtest engine.
        
        Args:
            capital: Initial capital (default: 1 Crore)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.capital = capital
        self.risk_free_rate = risk_free_rate
        self.cache_manager = DataCacheManager()
        self._historical_data: Optional[List[Tuple[datetime, pd.DataFrame]]] = None
        self._strategies: Dict[str, Any] = {}
        
    def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[datetime, pd.DataFrame]]:
        """
        Load historical data using Pragyam's backdata module.
        Utilizes caching to avoid redundant API calls.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use cached data
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of (date, DataFrame) tuples
        """
        # Import backdata here to avoid circular imports
        try:
            from backdata import generate_historical_data, MAX_INDICATOR_PERIOD
        except ImportError:
            logger.error("backdata.py not found. Cannot load data.")
            return []
        
        # Check cache first
        if use_cache:
            cached_data = self.cache_manager.get(symbols, start_date, end_date)
            if cached_data is not None:
                self._historical_data = cached_data
                return cached_data
        
        # Calculate fetch start date (need extra days for indicator warmup)
        fetch_start = start_date - timedelta(days=int(MAX_INDICATOR_PERIOD * 1.5) + 30)
        
        if progress_callback:
            progress_callback(0.1, "Fetching market data...")
        
        logger.info(f"Fetching data from {fetch_start.date()} to {end_date.date()} for {len(symbols)} symbols")
        
        # Generate historical data
        all_data = generate_historical_data(symbols, fetch_start, end_date)
        
        if not all_data:
            logger.error("Failed to fetch any historical data")
            return []
        
        # Filter to requested date range
        filtered_data = [
            (date, df) for date, df in all_data
            if start_date <= date <= end_date
        ]
        
        if progress_callback:
            progress_callback(0.5, f"Loaded {len(filtered_data)} trading days")
        
        # Cache the data
        if use_cache and filtered_data:
            self.cache_manager.set(symbols, start_date, end_date, filtered_data)
        
        self._historical_data = filtered_data
        return filtered_data
    
    def load_strategies(self, strategy_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load all available strategies from strategies.py.
        
        Args:
            strategy_filter: Optional list of strategy names to include
            
        Returns:
            Dictionary of strategy instances
        """
        try:
            from strategies import BaseStrategy
            import strategies as strategies_module
            import inspect
            
            all_strategies = {}
            
            # Find all BaseStrategy subclasses
            for name, obj in inspect.getmembers(strategies_module, inspect.isclass):
                if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                    if strategy_filter is None or name in strategy_filter:
                        try:
                            all_strategies[name] = obj()
                            logger.debug(f"Loaded strategy: {name}")
                        except Exception as e:
                            logger.warning(f"Could not instantiate strategy {name}: {e}")
            
            self._strategies = all_strategies
            logger.info(f"Loaded {len(all_strategies)} strategies")
            return all_strategies
            
        except ImportError as e:
            logger.error(f"Failed to import strategies module: {e}")
            return {}
    
    def run_backtest(
        self,
        mode: str = 'swing',
        external_trigger_df: Optional[pd.DataFrame] = None,
        buy_col: Optional[str] = None,
        sell_col: Optional[str] = None,
        buy_threshold: float = 0.42,
        sell_threshold: float = 0.52,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Dict]:
        """
        Run backtest on all loaded strategies.
        
        Args:
            mode: 'swing' for lumpsum or 'sip' for systematic investment
            external_trigger_df: DataFrame with trigger signals (optional)
            buy_col: Column name for buy signals
            sell_col: Column name for sell signals
            buy_threshold: Threshold for buy trigger
            sell_threshold: Threshold for sell trigger
            progress_callback: Callback for progress updates
            
        Returns:
            Dictionary with metrics for each strategy
        """
        if not self._historical_data:
            logger.error("No historical data loaded. Call load_data() first.")
            return {}
        
        if not self._strategies:
            logger.error("No strategies loaded. Call load_strategies() first.")
            return {}
        
        all_results = {}
        total_strategies = len(self._strategies)
        
        # Prepare trigger masks
        buy_dates_mask, sell_dates_mask = self._prepare_trigger_masks(
            external_trigger_df, buy_col, sell_col, buy_threshold, sell_threshold
        )
        
        for i, (name, strategy) in enumerate(self._strategies.items()):
            if progress_callback:
                progress_callback(
                    0.5 + (0.4 * (i + 1) / total_strategies),
                    f"Backtesting: {name} ({i+1}/{total_strategies})"
                )
            
            try:
                if mode.lower() == 'sip':
                    metrics, daily_data = self._run_sip_backtest(strategy, name)
                else:
                    metrics, daily_data = self._run_swing_backtest(
                        strategy, name, buy_dates_mask, sell_dates_mask
                    )
                
                all_results[name] = {
                    'metrics': metrics,
                    'daily_data': daily_data
                }
                
            except Exception as e:
                logger.error(f"Error backtesting {name}: {e}")
                all_results[name] = {
                    'metrics': PerformanceMetrics._empty_metrics(),
                    'daily_data': pd.DataFrame()
                }
        
        if progress_callback:
            progress_callback(1.0, "Backtest complete")
        
        return all_results
    
    def _prepare_trigger_masks(
        self,
        trigger_df: Optional[pd.DataFrame],
        buy_col: Optional[str],
        sell_col: Optional[str],
        buy_threshold: float,
        sell_threshold: float
    ) -> Tuple[List[bool], List[bool]]:
        """Prepare buy/sell trigger masks from external trigger DataFrame."""
        
        if trigger_df is None or trigger_df.empty:
            # No external triggers - use simple hold strategy
            return [True] + [False] * (len(self._historical_data) - 1), [False] * len(self._historical_data)
        
        buy_mask = [False] * len(self._historical_data)
        sell_mask = [False] * len(self._historical_data)
        
        if buy_col and buy_col in trigger_df.columns:
            external_buy_dates = trigger_df[trigger_df[buy_col] < buy_threshold].index.date
            buy_mask = [d.date() in external_buy_dates for d, _ in self._historical_data]
        
        if sell_col and sell_col in trigger_df.columns:
            external_sell_dates = trigger_df[trigger_df[sell_col] > sell_threshold].index.date
            sell_mask = [d.date() in external_sell_dates for d, _ in self._historical_data]
        
        return buy_mask, sell_mask
    
    def _run_swing_backtest(
        self,
        strategy,
        name: str,
        buy_mask: List[bool],
        sell_mask: List[bool]
    ) -> Tuple[Dict, pd.DataFrame]:
        """Run swing trading (lumpsum) backtest for a single strategy."""
        
        daily_values = []
        portfolio_units = {}
        buy_signal_active = False
        current_capital = self.capital
        
        for j, (date, df) in enumerate(self._historical_data):
            is_buy_day = buy_mask[j]
            actual_buy_trigger = is_buy_day and not buy_signal_active
            
            if is_buy_day:
                buy_signal_active = True
            elif not is_buy_day:
                buy_signal_active = False
            
            # Sell Logic
            if sell_mask[j] and portfolio_units:
                prices_today = df.set_index('symbol')['price']
                current_capital += sum(
                    units * prices_today.get(symbol, 0)
                    for symbol, units in portfolio_units.items()
                )
                portfolio_units = {}
                buy_signal_active = False
            
            # Buy Logic
            if actual_buy_trigger and not portfolio_units and current_capital > 1000:
                try:
                    buy_portfolio = strategy.generate_portfolio(df, current_capital)
                    if not buy_portfolio.empty and 'units' in buy_portfolio.columns:
                        portfolio_units = pd.Series(
                            buy_portfolio['units'].values,
                            index=buy_portfolio['symbol']
                        ).to_dict()
                        current_capital -= buy_portfolio['value'].sum()
                except Exception as e:
                    logger.debug(f"Portfolio generation failed for {name} on {date}: {e}")
            
            # Valuation
            portfolio_value = 0
            if portfolio_units:
                prices_today = df.set_index('symbol')['price']
                portfolio_value = sum(
                    units * prices_today.get(symbol, 0)
                    for symbol, units in portfolio_units.items()
                )
            
            daily_values.append({
                'date': date,
                'value': portfolio_value + current_capital,
                'investment': self.capital
            })
        
        if not daily_values:
            return PerformanceMetrics._empty_metrics(), pd.DataFrame()
        
        daily_df = pd.DataFrame(daily_values)
        metrics = PerformanceMetrics.calculate(daily_df, self.risk_free_rate)
        
        return metrics, daily_df
    
    def _run_sip_backtest(self, strategy, name: str) -> Tuple[Dict, pd.DataFrame]:
        """Run SIP (systematic investment) backtest for a single strategy."""
        
        # SIP: Invest fixed amount periodically (weekly in this case)
        sip_amount = self.capital / 52  # Weekly SIP
        total_invested = 0
        portfolio_units = {}
        daily_values = []
        last_sip_week = -1
        
        for date, df in self._historical_data:
            week_num = date.isocalendar()[1]
            
            # Weekly SIP
            if week_num != last_sip_week:
                try:
                    buy_portfolio = strategy.generate_portfolio(df, sip_amount)
                    if not buy_portfolio.empty and 'units' in buy_portfolio.columns:
                        for _, row in buy_portfolio.iterrows():
                            symbol = row['symbol']
                            units = row['units']
                            if symbol in portfolio_units:
                                portfolio_units[symbol] += units
                            else:
                                portfolio_units[symbol] = units
                        total_invested += buy_portfolio['value'].sum()
                        last_sip_week = week_num
                except Exception as e:
                    logger.debug(f"SIP generation failed for {name} on {date}: {e}")
            
            # Valuation
            portfolio_value = 0
            if portfolio_units:
                prices_today = df.set_index('symbol')['price']
                portfolio_value = sum(
                    units * prices_today.get(symbol, 0)
                    for symbol, units in portfolio_units.items()
                )
            
            daily_values.append({
                'date': date,
                'value': portfolio_value,
                'investment': total_invested if total_invested > 0 else sip_amount
            })
        
        if not daily_values:
            return PerformanceMetrics._empty_metrics(), pd.DataFrame()
        
        daily_df = pd.DataFrame(daily_values)
        metrics = PerformanceMetrics.calculate(daily_df, self.risk_free_rate)
        
        return metrics, daily_df
    
    def select_top_strategies(
        self,
        results: Dict[str, Dict],
        mode: str,
        n_strategies: int = 4
    ) -> List[str]:
        """
        Select top strategies based on mode-specific metrics.
        
        Args:
            results: Backtest results dictionary
            mode: 'sip' for Calmar-based selection, 'swing' for Sortino-based
            n_strategies: Number of strategies to select
            
        Returns:
            List of top strategy names
        """
        if not results:
            return []
        
        # Determine selection metric based on mode
        if mode.lower() == 'sip':
            metric_key = 'calmar_ratio'
            logger.info("Selecting strategies by Calmar Ratio (SIP mode)")
        else:
            metric_key = 'sortino_ratio'
            logger.info("Selecting strategies by Sortino Ratio (Swing mode)")
        
        # Extract metrics
        strategy_scores = []
        for name, data in results.items():
            metrics = data.get('metrics', {})
            score = metrics.get(metric_key, 0)
            
            # Filter out strategies with invalid/negative scores
            if np.isfinite(score) and score > -100:
                strategy_scores.append((name, score))
        
        # Sort by score descending
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        top_strategies = [name for name, score in strategy_scores[:n_strategies]]
        
        logger.info(f"Selected top {len(top_strategies)} strategies: {top_strategies}")
        for name, score in strategy_scores[:n_strategies]:
            logger.info(f"  {name}: {metric_key}={score:.4f}")
        
        return top_strategies


# ============================================================================
# DYNAMIC PORTFOLIO STYLES GENERATOR
# ============================================================================

class DynamicPortfolioStylesGenerator:
    """
    Generates PORTFOLIO_STYLES dictionary dynamically based on backtest results.
    Replaces hardcoded strategy selections with data-driven choices.
    """
    
    def __init__(self, engine: UnifiedBacktestEngine):
        """
        Initialize with a configured backtest engine.
        
        Args:
            engine: UnifiedBacktestEngine instance with loaded data and strategies
        """
        self.engine = engine
        self._backtest_results: Dict[str, Dict] = {}
        self._sip_results: Dict[str, Dict] = {}
        self._swing_results: Dict[str, Dict] = {}
    
    def run_comprehensive_backtest(
        self,
        external_trigger_df: Optional[pd.DataFrame] = None,
        buy_col: Optional[str] = None,
        sell_col: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[Dict, Dict]:
        """
        Run backtests for both SIP and Swing modes.
        
        Returns:
            Tuple of (sip_results, swing_results)
        """
        # Run SIP backtest
        if progress_callback:
            progress_callback(0.1, "Running SIP backtest...")
        
        self._sip_results = self.engine.run_backtest(
            mode='sip',
            progress_callback=lambda p, m: progress_callback(0.1 + p * 0.4, m) if progress_callback else None
        )
        
        # Run Swing backtest
        if progress_callback:
            progress_callback(0.5, "Running Swing backtest...")
        
        self._swing_results = self.engine.run_backtest(
            mode='swing',
            external_trigger_df=external_trigger_df,
            buy_col=buy_col,
            sell_col=sell_col,
            progress_callback=lambda p, m: progress_callback(0.5 + p * 0.4, m) if progress_callback else None
        )
        
        return self._sip_results, self._swing_results
    
    def generate_portfolio_styles(
        self,
        n_strategies: int = 4
    ) -> Dict[str, Dict]:
        """
        Generate dynamic PORTFOLIO_STYLES based on backtest results.
        
        Args:
            n_strategies: Number of strategies per mix
            
        Returns:
            PORTFOLIO_STYLES dictionary
        """
        # Select strategies for each mode
        sip_strategies = self.engine.select_top_strategies(
            self._sip_results, mode='sip', n_strategies=n_strategies
        )
        
        swing_strategies = self.engine.select_top_strategies(
            self._swing_results, mode='swing', n_strategies=n_strategies
        )
        
        # Build rationale strings with metrics
        sip_rationale = self._build_rationale(self._sip_results, sip_strategies, 'calmar_ratio')
        swing_rationale = self._build_rationale(self._swing_results, swing_strategies, 'sortino_ratio')
        
        # Generate PORTFOLIO_STYLES
        portfolio_styles = {
            "Swing Trading": {
                "description": "Short-term (3-21 day) holds to capture rapid momentum and volatility.",
                "mixes": {
                    "Bull Market Mix": {
                        "strategies": swing_strategies,
                        "rationale": f"Dynamically selected based on highest Sortino Ratio from backtest. {swing_rationale}"
                    },
                    "Bear Market Mix": {
                        "strategies": swing_strategies,
                        "rationale": f"Dynamically selected based on highest Sortino Ratio from backtest. {swing_rationale}"
                    },
                    "Chop/Consolidate Mix": {
                        "strategies": swing_strategies,
                        "rationale": f"Dynamically selected based on highest Sortino Ratio from backtest. {swing_rationale}"
                    }
                }
            },
            "SIP Investment": {
                "description": "Systematic long-term (3-12+ months) wealth accumulation. Focus on consistency and drawdown protection.",
                "mixes": {
                    "Bull Market Mix": {
                        "strategies": sip_strategies,
                        "rationale": f"Dynamically selected based on highest Calmar Ratio from backtest. {sip_rationale}"
                    },
                    "Bear Market Mix": {
                        "strategies": sip_strategies,
                        "rationale": f"Dynamically selected based on highest Calmar Ratio from backtest. {sip_rationale}"
                    },
                    "Chop/Consolidate Mix": {
                        "strategies": sip_strategies,
                        "rationale": f"Dynamically selected based on highest Calmar Ratio from backtest. {sip_rationale}"
                    }
                }
            }
        }
        
        return portfolio_styles
    
    def _build_rationale(
        self,
        results: Dict[str, Dict],
        strategies: List[str],
        metric_key: str
    ) -> str:
        """Build rationale string with metric values."""
        if not results or not strategies:
            return "No backtest data available."
        
        parts = []
        for name in strategies:
            if name in results:
                score = results[name].get('metrics', {}).get(metric_key, 0)
                parts.append(f"{name}: {score:.2f}")
        
        return " | ".join(parts) if parts else "Metrics unavailable."
    
    def get_strategy_leaderboard(self, mode: str = 'swing') -> pd.DataFrame:
        """
        Get a formatted leaderboard of strategy performance.
        
        Args:
            mode: 'sip' or 'swing'
            
        Returns:
            DataFrame with strategy rankings
        """
        results = self._sip_results if mode.lower() == 'sip' else self._swing_results
        
        if not results:
            return pd.DataFrame()
        
        leaderboard_data = []
        for name, data in results.items():
            metrics = data.get('metrics', {})
            leaderboard_data.append({
                'Strategy': name,
                'Total Return': metrics.get('total_return', 0),
                'CAGR': metrics.get('cagr', 0),
                'Volatility': metrics.get('volatility', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Calmar Ratio': metrics.get('calmar_ratio', 0),
                'Win Rate': metrics.get('win_rate', 0)
            })
        
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by primary metric for mode
        sort_col = 'Calmar Ratio' if mode.lower() == 'sip' else 'Sortino Ratio'
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        
        return df


# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================

def initialize_backtest_engine(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    capital: float = 10_000_000,
    progress_callback: Optional[callable] = None
) -> Tuple[UnifiedBacktestEngine, DynamicPortfolioStylesGenerator]:
    """
    Initialize and configure the backtest engine with data.
    
    This is the main entry point for Pragyam integration.
    
    Args:
        symbols: List of stock symbols
        start_date: Backtest start date
        end_date: Backtest end date
        capital: Initial capital
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (engine, generator) ready for use
    """
    engine = UnifiedBacktestEngine(capital=capital)
    
    # Load data
    engine.load_data(symbols, start_date, end_date, progress_callback=progress_callback)
    
    # Load all strategies
    engine.load_strategies()
    
    # Create generator
    generator = DynamicPortfolioStylesGenerator(engine)
    
    return engine, generator


def get_dynamic_portfolio_styles(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    capital: float = 10_000_000,
    external_trigger_df: Optional[pd.DataFrame] = None,
    buy_col: Optional[str] = None,
    sell_col: Optional[str] = None,
    n_strategies: int = 4,
    progress_callback: Optional[callable] = None
) -> Dict[str, Dict]:
    """
    Main entry point: Generate dynamic PORTFOLIO_STYLES based on backtest.
    
    This function replaces the hardcoded PORTFOLIO_STYLES in app.py.
    
    Args:
        symbols: List of stock symbols
        start_date: Backtest start date  
        end_date: Backtest end date
        capital: Initial capital for backtest
        external_trigger_df: Optional trigger signal DataFrame
        buy_col: Buy trigger column name
        sell_col: Sell trigger column name
        n_strategies: Number of strategies to select per mode
        progress_callback: Optional callback for progress updates
        
    Returns:
        PORTFOLIO_STYLES dictionary with dynamically selected strategies
    """
    # Initialize engine
    engine, generator = initialize_backtest_engine(
        symbols, start_date, end_date, capital,
        progress_callback=lambda p, m: progress_callback(p * 0.3, m) if progress_callback else None
    )
    
    # Run comprehensive backtest
    generator.run_comprehensive_backtest(
        external_trigger_df=external_trigger_df,
        buy_col=buy_col,
        sell_col=sell_col,
        progress_callback=lambda p, m: progress_callback(0.3 + p * 0.6, m) if progress_callback else None
    )
    
    # Generate portfolio styles
    if progress_callback:
        progress_callback(0.95, "Generating portfolio styles...")
    
    portfolio_styles = generator.generate_portfolio_styles(n_strategies=n_strategies)
    
    if progress_callback:
        progress_callback(1.0, "Complete")
    
    return portfolio_styles


# ============================================================================
# STREAMLIT UI MODULE (Standalone Operation)
# ============================================================================

def run_streamlit_ui():
    """
    Run standalone Streamlit UI for backtesting.
    Can be used independently or integrated into Pragyam.
    """
    try:
        import streamlit as st
    except ImportError:
        logger.error("Streamlit not installed. Cannot run UI.")
        return
    
    st.set_page_config(
        page_title="Backtest Engine | Pragyam",
        page_icon="⚙️",
        layout="wide"
    )
    
    st.title("⚙️ Unified Backtest Engine")
    st.markdown("*Dynamic Strategy Selection for Pragyam*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        capital = st.number_input(
            "Capital (₹)",
            min_value=100000,
            max_value=100000000,
            value=10000000,
            step=100000
        )
        
        n_strategies = st.slider(
            "Strategies per Mix",
            min_value=2,
            max_value=8,
            value=4
        )
        
        run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    if run_button:
        try:
            from backdata import SYMBOLS_UNIVERSE, MAX_INDICATOR_PERIOD
        except ImportError:
            st.error("Could not import backdata.py")
            return
        
        if not SYMBOLS_UNIVERSE:
            st.error("No symbols found in symbols.txt")
            return
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Progress
        progress_bar = st.progress(0, text="Initializing...")
        
        def update_progress(p, msg):
            progress_bar.progress(p, text=msg)
        
        # Run backtest
        portfolio_styles = get_dynamic_portfolio_styles(
            symbols=SYMBOLS_UNIVERSE,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
            n_strategies=n_strategies,
            progress_callback=update_progress
        )
        
        progress_bar.empty()
        st.success("✅ Backtest Complete!")
        
        # Display results
        st.header("📊 Dynamic Portfolio Styles")
        
        for style_name, style_data in portfolio_styles.items():
            with st.expander(f"**{style_name}**", expanded=True):
                st.markdown(f"*{style_data['description']}*")
                
                for mix_name, mix_data in style_data['mixes'].items():
                    st.subheader(mix_name)
                    st.markdown(f"**Strategies:** `{'`, `'.join(mix_data['strategies'])}`")
                    st.caption(mix_data['rationale'])


if __name__ == "__main__":
    run_streamlit_ui()
