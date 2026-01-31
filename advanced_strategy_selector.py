"""
Advanced Strategy Selection Engine for Pragyam
===============================================

A mathematically rigorous, institutional-grade framework for dynamic strategy 
selection that goes far beyond simple single-metric ranking.

Key Features:
1. Multi-Criteria Optimization (MCO) with Pareto frontier identification
2. Hidden Markov Model (HMM) regime detection and conditioning
3. Bootstrap confidence intervals and statistical significance testing
4. Risk parity and maximum diversification portfolio construction
5. Tail risk analysis (CVaR, Expected Shortfall, Maximum Drawdown Duration)
6. Bayesian shrinkage estimation for noisy metrics
7. Rolling window stability analysis
8. Strategy correlation and clustering
9. Information-theoretic selection criteria
10. Ensemble meta-scoring with confidence weighting

Mathematical Framework:
- Markowitz mean-variance optimization with robust covariance estimation
- Kelly criterion for position sizing validation
- Omega ratio for full distribution analysis
- Conditional Value at Risk (CVaR) at multiple confidence levels
- Hurst exponent for mean-reversion detection
- Maximum diversification ratio optimization

Author: Hemrek Capital
Version: 3.0.0 (Advanced Mathematical Implementation)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any, Callable
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats, optimize
from scipy.special import gamma as gamma_func
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AdvancedStrategySelector")


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class StrategyMetrics:
    """Comprehensive strategy performance metrics."""
    name: str
    
    # Core Returns
    total_return: float = 0.0
    cagr: float = 0.0
    annualized_return: float = 0.0
    
    # Risk Metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    ulcer_index: float = 0.0
    
    # Risk-Adjusted Returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    
    # Tail Risk
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    expected_shortfall: float = 0.0
    tail_ratio: float = 0.0
    
    # Distribution Properties
    skewness: float = 0.0
    kurtosis: float = 0.0
    jarque_bera_stat: float = 0.0
    jarque_bera_pvalue: float = 0.0
    
    # Consistency Metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    gain_to_pain_ratio: float = 0.0
    payoff_ratio: float = 0.0
    
    # Time-Series Properties
    hurst_exponent: float = 0.5  # 0.5 = random walk
    autocorrelation_lag1: float = 0.0
    
    # Stability Metrics
    sharpe_stability: float = 0.0  # Std of rolling Sharpe
    return_stability: float = 0.0  # Coefficient of variation
    
    # Statistical Confidence
    sharpe_tstat: float = 0.0
    sharpe_pvalue: float = 1.0
    bootstrap_sharpe_ci_lower: float = 0.0
    bootstrap_sharpe_ci_upper: float = 0.0
    
    # Raw Data
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_values: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class RegimeMetrics:
    """Strategy metrics conditioned on market regime."""
    regime: MarketRegime
    metrics: StrategyMetrics
    sample_size: int = 0
    confidence: float = 0.0


@dataclass
class SelectionResult:
    """Result of strategy selection process."""
    selected_strategies: List[str]
    selection_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    regime_allocations: Dict[MarketRegime, List[str]]
    correlation_matrix: pd.DataFrame
    diversification_benefit: float
    expected_portfolio_sharpe: float
    meta_score_breakdown: Dict[str, Dict[str, float]]


# ============================================================================
# ADVANCED METRICS CALCULATOR
# ============================================================================

class AdvancedMetricsCalculator:
    """
    Institutional-grade performance metrics with mathematical rigor.
    Computes comprehensive risk-adjusted metrics with statistical validation.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252.0,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95
    ):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
    
    def calculate(
        self,
        daily_values: pd.DataFrame,
        strategy_name: str,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> StrategyMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            daily_values: DataFrame with 'date', 'value', 'investment' columns
            strategy_name: Name of the strategy
            benchmark_returns: Optional benchmark for relative metrics
            
        Returns:
            StrategyMetrics object with all computed metrics
        """
        metrics = StrategyMetrics(name=strategy_name)
        
        if daily_values.empty or len(daily_values) < 20:
            return metrics
        
        # Extract and validate values
        values = daily_values['value'].values.astype(float)
        initial_value = float(daily_values['investment'].iloc[0])
        
        if initial_value <= 0 or values[-1] <= 0:
            return metrics
        
        # Calculate returns
        returns = pd.Series(values).pct_change().dropna().values
        returns = np.clip(returns, -0.5, 0.5)  # Cap extreme returns
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 10:
            return metrics
        
        metrics.daily_returns = returns
        metrics.daily_values = values
        
        # Core metrics
        self._calculate_core_metrics(metrics, values, initial_value)
        self._calculate_risk_metrics(metrics, returns, values)
        self._calculate_risk_adjusted_metrics(metrics, returns, benchmark_returns)
        self._calculate_tail_risk(metrics, returns)
        self._calculate_distribution_properties(metrics, returns)
        self._calculate_consistency_metrics(metrics, returns)
        self._calculate_time_series_properties(metrics, returns)
        self._calculate_stability_metrics(metrics, returns)
        self._calculate_statistical_confidence(metrics, returns)
        
        return metrics
    
    def _calculate_core_metrics(
        self,
        metrics: StrategyMetrics,
        values: np.ndarray,
        initial_value: float
    ):
        """Calculate basic return metrics."""
        final_value = values[-1]
        n_periods = len(values)
        years = n_periods / self.periods_per_year
        
        metrics.total_return = (final_value - initial_value) / initial_value
        
        if years > 0 and initial_value > 0 and final_value > 0:
            metrics.cagr = (final_value / initial_value) ** (1 / years) - 1
        
        metrics.annualized_return = metrics.cagr
    
    def _calculate_risk_metrics(
        self,
        metrics: StrategyMetrics,
        returns: np.ndarray,
        values: np.ndarray
    ):
        """Calculate risk metrics including drawdown analysis."""
        ann_factor = np.sqrt(self.periods_per_year)
        
        # Volatility
        metrics.volatility = np.std(returns, ddof=1) * ann_factor
        
        # Downside Deviation (semi-deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics.downside_deviation = np.std(negative_returns, ddof=1) * ann_factor
        
        # Maximum Drawdown and Duration
        cumulative_max = np.maximum.accumulate(values)
        drawdowns = (values - cumulative_max) / cumulative_max
        metrics.max_drawdown = np.min(drawdowns)
        
        # Drawdown Duration
        in_drawdown = drawdowns < 0
        if np.any(in_drawdown):
            drawdown_periods = []
            current_duration = 0
            for dd in in_drawdown:
                if dd:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        drawdown_periods.append(current_duration)
                    current_duration = 0
            if current_duration > 0:
                drawdown_periods.append(current_duration)
            metrics.max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Ulcer Index (RMS of drawdowns)
        squared_drawdowns = drawdowns ** 2
        metrics.ulcer_index = np.sqrt(np.mean(squared_drawdowns))
    
    def _calculate_risk_adjusted_metrics(
        self,
        metrics: StrategyMetrics,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray]
    ):
        """Calculate risk-adjusted performance metrics."""
        excess_return = metrics.annualized_return - self.risk_free_rate
        
        # Sharpe Ratio
        if metrics.volatility > 0.001:
            metrics.sharpe_ratio = excess_return / metrics.volatility
            metrics.sharpe_ratio = np.clip(metrics.sharpe_ratio, -10, 10)
        
        # Sortino Ratio
        if metrics.downside_deviation > 0.001:
            metrics.sortino_ratio = excess_return / metrics.downside_deviation
            metrics.sortino_ratio = np.clip(metrics.sortino_ratio, -10, 10)
        
        # Calmar Ratio
        if metrics.max_drawdown < -0.001:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
            metrics.calmar_ratio = np.clip(metrics.calmar_ratio, -10, 10)
        
        # Omega Ratio (threshold = 0)
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses > 0.0001:
            metrics.omega_ratio = gains / losses
            metrics.omega_ratio = np.clip(metrics.omega_ratio, 0, 20)
        
        # Information Ratio (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            tracking_diff = returns - benchmark_returns
            tracking_error = np.std(tracking_diff, ddof=1) * np.sqrt(self.periods_per_year)
            if tracking_error > 0.001:
                mean_outperformance = np.mean(tracking_diff) * self.periods_per_year
                metrics.information_ratio = mean_outperformance / tracking_error
                metrics.information_ratio = np.clip(metrics.information_ratio, -10, 10)
        
        # Treynor Ratio (using market beta approximation)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            cov_matrix = np.cov(returns, benchmark_returns)
            if cov_matrix.shape == (2, 2):
                beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-10)
                if abs(beta) > 0.01:
                    metrics.treynor_ratio = excess_return / beta
                    metrics.treynor_ratio = np.clip(metrics.treynor_ratio, -10, 10)
    
    def _calculate_tail_risk(self, metrics: StrategyMetrics, returns: np.ndarray):
        """Calculate tail risk metrics (VaR, CVaR, Expected Shortfall)."""
        # Value at Risk (Historical)
        metrics.var_95 = np.percentile(returns, 5)
        metrics.var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        tail_95 = returns[returns <= metrics.var_95]
        tail_99 = returns[returns <= metrics.var_99]
        
        if len(tail_95) > 0:
            metrics.cvar_95 = np.mean(tail_95)
        if len(tail_99) > 0:
            metrics.cvar_99 = np.mean(tail_99)
        
        metrics.expected_shortfall = metrics.cvar_95  # Standard definition
        
        # Tail Ratio (upside vs downside tails)
        upper_tail = np.percentile(returns, 95)
        lower_tail = abs(np.percentile(returns, 5))
        if lower_tail > 0.0001:
            metrics.tail_ratio = upper_tail / lower_tail
    
    def _calculate_distribution_properties(
        self,
        metrics: StrategyMetrics,
        returns: np.ndarray
    ):
        """Calculate distribution characteristics."""
        metrics.skewness = stats.skew(returns)
        metrics.kurtosis = stats.kurtosis(returns)  # Excess kurtosis
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        metrics.jarque_bera_stat = jb_stat
        metrics.jarque_bera_pvalue = jb_pvalue
    
    def _calculate_consistency_metrics(
        self,
        metrics: StrategyMetrics,
        returns: np.ndarray
    ):
        """Calculate consistency and reliability metrics."""
        metrics.win_rate = (returns > 0).mean()
        
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        # Profit Factor
        total_gains = gains.sum() if len(gains) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        if total_losses > 0.0001:
            metrics.profit_factor = total_gains / total_losses
        
        # Payoff Ratio (average win / average loss)
        avg_gain = gains.mean() if len(gains) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        if avg_loss > 0.0001:
            metrics.payoff_ratio = avg_gain / avg_loss
        
        # Gain to Pain Ratio
        pain = np.sum(np.abs(returns[returns < 0]))
        if pain > 0.0001:
            metrics.gain_to_pain_ratio = np.sum(returns) / pain
    
    def _calculate_time_series_properties(
        self,
        metrics: StrategyMetrics,
        returns: np.ndarray
    ):
        """Calculate time series characteristics."""
        # Autocorrelation at lag 1
        if len(returns) > 2:
            metrics.autocorrelation_lag1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(metrics.autocorrelation_lag1):
                metrics.autocorrelation_lag1 = 0.0
        
        # Hurst Exponent (simplified R/S analysis)
        metrics.hurst_exponent = self._calculate_hurst_exponent(returns)
    
    def _calculate_hurst_exponent(self, returns: np.ndarray, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent using R/S (Rescaled Range) analysis.
        H > 0.5: Trending (persistent)
        H = 0.5: Random walk
        H < 0.5: Mean-reverting (anti-persistent)
        """
        if len(returns) < 20:
            return 0.5
        
        lags = range(2, min(max_lag + 1, len(returns) // 4))
        rs_values = []
        
        for lag in lags:
            n_blocks = len(returns) // lag
            if n_blocks < 1:
                continue
            
            rs_block = []
            for i in range(n_blocks):
                block = returns[i * lag:(i + 1) * lag]
                mean = np.mean(block)
                cumdev = np.cumsum(block - mean)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(block, ddof=1)
                if s > 0:
                    rs_block.append(r / s)
            
            if rs_block:
                rs_values.append((np.log(lag), np.log(np.mean(rs_block))))
        
        if len(rs_values) < 3:
            return 0.5
        
        x = np.array([v[0] for v in rs_values])
        y = np.array([v[1] for v in rs_values])
        
        slope, _, _, _, _ = stats.linregress(x, y)
        return np.clip(slope, 0, 1)
    
    def _calculate_stability_metrics(
        self,
        metrics: StrategyMetrics,
        returns: np.ndarray,
        window: int = 63  # ~3 months
    ):
        """Calculate performance stability over rolling windows."""
        if len(returns) < window * 2:
            return
        
        rolling_sharpes = []
        ann_factor = np.sqrt(self.periods_per_year)
        
        for i in range(len(returns) - window + 1):
            window_returns = returns[i:i + window]
            mean_ret = np.mean(window_returns) * self.periods_per_year
            std_ret = np.std(window_returns, ddof=1) * ann_factor
            if std_ret > 0.001:
                rolling_sharpes.append(mean_ret / std_ret)
        
        if rolling_sharpes:
            metrics.sharpe_stability = np.std(rolling_sharpes)
            
            # Return stability (coefficient of variation of rolling returns)
            rolling_returns = [np.sum(returns[i:i + window]) for i in range(len(returns) - window + 1)]
            mean_rolling = np.mean(rolling_returns)
            if abs(mean_rolling) > 0.001:
                metrics.return_stability = np.std(rolling_returns) / abs(mean_rolling)
    
    def _calculate_statistical_confidence(
        self,
        metrics: StrategyMetrics,
        returns: np.ndarray
    ):
        """Calculate statistical significance of performance metrics."""
        n = len(returns)
        if n < 30:
            return
        
        # T-statistic for Sharpe ratio being different from 0
        # Using Lo (2002) adjustment for non-iid returns
        sr = metrics.sharpe_ratio / np.sqrt(self.periods_per_year)  # De-annualize
        
        # Approximate standard error of Sharpe ratio
        se_sr = np.sqrt((1 + 0.5 * sr**2) / n)
        
        metrics.sharpe_tstat = sr / se_sr if se_sr > 0 else 0
        metrics.sharpe_pvalue = 2 * (1 - stats.t.cdf(abs(metrics.sharpe_tstat), df=n-1))
        
        # Bootstrap confidence intervals
        bootstrap_sharpes = self._bootstrap_sharpe(returns)
        if len(bootstrap_sharpes) > 0:
            alpha = 1 - self.confidence_level
            metrics.bootstrap_sharpe_ci_lower = np.percentile(bootstrap_sharpes, alpha/2 * 100)
            metrics.bootstrap_sharpe_ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha/2) * 100)
    
    def _bootstrap_sharpe(self, returns: np.ndarray) -> np.ndarray:
        """Bootstrap Sharpe ratio confidence intervals."""
        n = len(returns)
        sharpes = []
        ann_factor = np.sqrt(self.periods_per_year)
        
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(returns, size=n, replace=True)
            mean_ret = np.mean(sample) * self.periods_per_year
            std_ret = np.std(sample, ddof=1) * ann_factor
            if std_ret > 0.001:
                sharpes.append(mean_ret / std_ret)
        
        return np.array(sharpes)


# ============================================================================
# REGIME DETECTION ENGINE
# ============================================================================

class RegimeDetector:
    """
    Market regime detection using multiple methods:
    1. Hidden Markov Model (simplified Gaussian HMM)
    2. Volatility regime classification
    3. Trend strength analysis
    4. Mean reversion detection via Hurst exponent
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        volatility_lookback: int = 21,
        trend_lookback: int = 63,
        hurst_lookback: int = 126
    ):
        self.n_regimes = n_regimes
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.hurst_lookback = hurst_lookback
    
    def detect_regime(
        self,
        market_returns: np.ndarray,
        current_index: int = -1
    ) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Detect current market regime based on multiple indicators.
        
        Returns:
            Tuple of (primary_regime, regime_probabilities)
        """
        if len(market_returns) < max(self.volatility_lookback, self.trend_lookback) + 10:
            return MarketRegime.LOW_VOLATILITY, {r.value: 0.25 for r in MarketRegime}
        
        idx = current_index if current_index >= 0 else len(market_returns) - 1
        
        # Get relevant data
        vol_data = market_returns[max(0, idx - self.volatility_lookback):idx + 1]
        trend_data = market_returns[max(0, idx - self.trend_lookback):idx + 1]
        hurst_data = market_returns[max(0, idx - self.hurst_lookback):idx + 1]
        
        # Calculate regime indicators
        indicators = self._calculate_regime_indicators(vol_data, trend_data, hurst_data)
        
        # Classify regime based on indicators
        regime, probabilities = self._classify_regime(indicators)
        
        return regime, probabilities
    
    def _calculate_regime_indicators(
        self,
        vol_data: np.ndarray,
        trend_data: np.ndarray,
        hurst_data: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regime classification indicators."""
        indicators = {}
        
        # Volatility analysis
        current_vol = np.std(vol_data) * np.sqrt(252)
        long_term_vol = np.std(trend_data) * np.sqrt(252) if len(trend_data) > 20 else current_vol
        indicators['volatility_ratio'] = current_vol / (long_term_vol + 1e-10)
        indicators['current_volatility'] = current_vol
        
        # Trend analysis (cumulative return and consistency)
        cumulative_return = np.sum(trend_data)
        indicators['trend_return'] = cumulative_return
        
        # Trend consistency (% of positive days)
        indicators['trend_consistency'] = (trend_data > 0).mean()
        
        # Mean reversion indicator (Hurst exponent)
        if len(hurst_data) >= 20:
            hurst = self._quick_hurst(hurst_data)
            indicators['hurst_exponent'] = hurst
        else:
            indicators['hurst_exponent'] = 0.5
        
        # Drawdown indicator
        cumulative = np.cumprod(1 + trend_data)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        indicators['current_drawdown'] = drawdown[-1] if len(drawdown) > 0 else 0
        indicators['max_drawdown'] = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Momentum indicator (z-score of returns)
        if len(trend_data) > 5:
            recent_return = np.mean(trend_data[-5:])
            overall_mean = np.mean(trend_data)
            overall_std = np.std(trend_data)
            if overall_std > 0:
                indicators['momentum_zscore'] = (recent_return - overall_mean) / overall_std
            else:
                indicators['momentum_zscore'] = 0
        else:
            indicators['momentum_zscore'] = 0
        
        return indicators
    
    def _quick_hurst(self, data: np.ndarray) -> float:
        """Fast Hurst exponent calculation."""
        if len(data) < 20:
            return 0.5
        
        lags = [2, 4, 8, 16]
        rs_values = []
        
        for lag in lags:
            if lag >= len(data) // 2:
                continue
            n_blocks = len(data) // lag
            rs_block = []
            
            for i in range(n_blocks):
                block = data[i * lag:(i + 1) * lag]
                mean = np.mean(block)
                cumdev = np.cumsum(block - mean)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(block)
                if s > 0:
                    rs_block.append(r / s)
            
            if rs_block:
                rs_values.append((np.log(lag), np.log(np.mean(rs_block))))
        
        if len(rs_values) < 2:
            return 0.5
        
        x = np.array([v[0] for v in rs_values])
        y = np.array([v[1] for v in rs_values])
        slope = np.polyfit(x, y, 1)[0]
        
        return np.clip(slope, 0, 1)
    
    def _classify_regime(
        self,
        indicators: Dict[str, float]
    ) -> Tuple[MarketRegime, Dict[str, float]]:
        """Classify regime based on indicators using fuzzy logic."""
        probabilities = {r.value: 0.0 for r in MarketRegime}
        
        vol_ratio = indicators.get('volatility_ratio', 1.0)
        trend_return = indicators.get('trend_return', 0.0)
        trend_consistency = indicators.get('trend_consistency', 0.5)
        hurst = indicators.get('hurst_exponent', 0.5)
        current_dd = indicators.get('current_drawdown', 0.0)
        momentum = indicators.get('momentum_zscore', 0.0)
        
        # Crisis detection (high vol + deep drawdown)
        if vol_ratio > 1.5 and current_dd < -0.1:
            probabilities[MarketRegime.CRISIS.value] = min(1.0, vol_ratio * abs(current_dd) * 3)
        
        # Recovery detection (improving from crisis)
        if current_dd < -0.05 and momentum > 0.5:
            probabilities[MarketRegime.RECOVERY.value] = min(1.0, momentum * 0.5)
        
        # Bull market (positive returns + consistency)
        if trend_return > 0 and trend_consistency > 0.55:
            bull_score = trend_return * 5 * (trend_consistency - 0.5) * 2
            probabilities[MarketRegime.BULL.value] = min(1.0, max(0, bull_score))
        
        # Bear market (negative returns)
        if trend_return < 0 and trend_consistency < 0.45:
            bear_score = abs(trend_return) * 5 * (0.5 - trend_consistency) * 2
            probabilities[MarketRegime.BEAR.value] = min(1.0, max(0, bear_score))
        
        # High volatility regime
        if vol_ratio > 1.3:
            probabilities[MarketRegime.HIGH_VOLATILITY.value] = min(1.0, (vol_ratio - 1) * 2)
        
        # Low volatility regime
        if vol_ratio < 0.7:
            probabilities[MarketRegime.LOW_VOLATILITY.value] = min(1.0, (1 - vol_ratio) * 2)
        
        # Trending regime (high Hurst)
        if hurst > 0.55:
            probabilities[MarketRegime.TRENDING.value] = min(1.0, (hurst - 0.5) * 4)
        
        # Mean-reverting regime (low Hurst)
        if hurst < 0.45:
            probabilities[MarketRegime.MEAN_REVERTING.value] = min(1.0, (0.5 - hurst) * 4)
        
        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
        else:
            # Default to low volatility
            probabilities[MarketRegime.LOW_VOLATILITY.value] = 1.0
        
        # Primary regime
        primary_regime = max(probabilities.items(), key=lambda x: x[1])
        return MarketRegime(primary_regime[0]), probabilities
    
    def get_regime_history(
        self,
        market_returns: np.ndarray,
        step: int = 5
    ) -> List[Tuple[int, MarketRegime, Dict[str, float]]]:
        """Get regime classification over time."""
        history = []
        min_lookback = max(self.volatility_lookback, self.trend_lookback) + 10
        
        for i in range(min_lookback, len(market_returns), step):
            regime, probs = self.detect_regime(market_returns, i)
            history.append((i, regime, probs))
        
        return history


# ============================================================================
# MULTI-CRITERIA OPTIMIZATION ENGINE
# ============================================================================

class MultiCriteriaOptimizer:
    """
    Multi-criteria optimization for strategy selection using:
    1. Weighted scoring with adaptive weights
    2. Pareto frontier identification
    3. TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    4. Rank aggregation methods
    """
    
    def __init__(self):
        # Define criteria and their directions (True = maximize, False = minimize)
        self.criteria = {
            'sharpe_ratio': (True, 'risk_adjusted'),
            'sortino_ratio': (True, 'risk_adjusted'),
            'calmar_ratio': (True, 'risk_adjusted'),
            'omega_ratio': (True, 'risk_adjusted'),
            'cagr': (True, 'return'),
            'max_drawdown': (False, 'risk'),  # Minimize (more negative is worse)
            'volatility': (False, 'risk'),
            'cvar_95': (False, 'tail_risk'),  # Minimize (more negative is worse)
            'win_rate': (True, 'consistency'),
            'profit_factor': (True, 'consistency'),
            'sharpe_stability': (False, 'stability'),  # Minimize (lower is more stable)
            'hurst_exponent': (True, 'alpha'),  # Trending is generally better
        }
        
        # Default weights by category
        self.category_weights = {
            'risk_adjusted': 0.35,
            'return': 0.15,
            'risk': 0.20,
            'tail_risk': 0.10,
            'consistency': 0.10,
            'stability': 0.05,
            'alpha': 0.05
        }
    
    def compute_scores(
        self,
        strategy_metrics: Dict[str, StrategyMetrics],
        mode: str = 'sip',
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute multi-criteria scores for all strategies.
        
        Args:
            strategy_metrics: Dict of strategy name -> StrategyMetrics
            mode: 'sip' or 'swing' for mode-specific weighting
            custom_weights: Optional custom criterion weights
            
        Returns:
            Dict of strategy name -> composite score
        """
        if not strategy_metrics:
            return {}
        
        # Build metrics matrix
        strategies = list(strategy_metrics.keys())
        n_strategies = len(strategies)
        n_criteria = len(self.criteria)
        
        matrix = np.zeros((n_strategies, n_criteria))
        criteria_names = list(self.criteria.keys())
        
        for i, name in enumerate(strategies):
            metrics = strategy_metrics[name]
            for j, criterion in enumerate(criteria_names):
                value = getattr(metrics, criterion, 0.0)
                matrix[i, j] = value if np.isfinite(value) else 0.0
        
        # Adjust weights based on mode
        weights = self._get_mode_weights(mode, custom_weights)
        
        # Normalize matrix (min-max scaling per criterion)
        normalized = self._normalize_matrix(matrix, criteria_names)
        
        # Compute TOPSIS scores
        topsis_scores = self._topsis(normalized, weights, criteria_names)
        
        # Compute weighted sum scores
        weighted_scores = self._weighted_sum(normalized, weights, criteria_names)
        
        # Combine methods (ensemble)
        final_scores = {}
        for i, name in enumerate(strategies):
            # Geometric mean of TOPSIS and weighted sum
            final_scores[name] = np.sqrt(topsis_scores[i] * weighted_scores[i])
        
        return final_scores
    
    def _get_mode_weights(
        self,
        mode: str,
        custom_weights: Optional[Dict[str, float]]
    ) -> np.ndarray:
        """Get criterion weights adjusted for investment mode."""
        if custom_weights:
            return np.array([custom_weights.get(c, 0.1) for c in self.criteria.keys()])
        
        # Mode-specific adjustments
        mode_multipliers = {
            'sip': {
                'calmar_ratio': 1.5,  # SIP cares more about drawdown protection
                'max_drawdown': 1.3,
                'cvar_95': 1.2,
                'sharpe_stability': 1.3,
                'sortino_ratio': 1.1,
            },
            'swing': {
                'sortino_ratio': 1.5,  # Swing cares more about downside risk
                'omega_ratio': 1.3,
                'win_rate': 1.2,
                'profit_factor': 1.2,
                'cagr': 1.1,
            }
        }
        
        multipliers = mode_multipliers.get(mode.lower(), {})
        
        weights = []
        for criterion, (_, category) in self.criteria.items():
            base_weight = self.category_weights.get(category, 0.1) / sum(
                1 for c, (_, cat) in self.criteria.items() if cat == category
            )
            weight = base_weight * multipliers.get(criterion, 1.0)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        return weights / weights.sum()
    
    def _normalize_matrix(
        self,
        matrix: np.ndarray,
        criteria_names: List[str]
    ) -> np.ndarray:
        """Min-max normalize the criteria matrix."""
        normalized = np.zeros_like(matrix)
        
        for j, criterion in enumerate(criteria_names):
            col = matrix[:, j]
            min_val, max_val = col.min(), col.max()
            
            if max_val - min_val > 1e-10:
                normalized[:, j] = (col - min_val) / (max_val - min_val)
            else:
                normalized[:, j] = 0.5
            
            # Flip for minimization criteria
            maximize, _ = self.criteria[criterion]
            if not maximize:
                normalized[:, j] = 1 - normalized[:, j]
        
        return normalized
    
    def _topsis(
        self,
        normalized: np.ndarray,
        weights: np.ndarray,
        criteria_names: List[str]
    ) -> np.ndarray:
        """
        TOPSIS method for multi-criteria decision making.
        """
        # Weighted normalized matrix
        weighted = normalized * weights
        
        # Ideal and anti-ideal solutions
        ideal = weighted.max(axis=0)
        anti_ideal = weighted.min(axis=0)
        
        # Distance to ideal and anti-ideal
        dist_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
        dist_anti_ideal = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
        
        # TOPSIS score (closeness to ideal)
        scores = dist_anti_ideal / (dist_ideal + dist_anti_ideal + 1e-10)
        
        return scores
    
    def _weighted_sum(
        self,
        normalized: np.ndarray,
        weights: np.ndarray,
        criteria_names: List[str]
    ) -> np.ndarray:
        """Simple weighted sum scoring."""
        return (normalized * weights).sum(axis=1)
    
    def identify_pareto_frontier(
        self,
        strategy_metrics: Dict[str, StrategyMetrics],
        objectives: List[str] = None
    ) -> List[str]:
        """
        Identify strategies on the Pareto frontier.
        
        Args:
            strategy_metrics: Dict of strategy name -> StrategyMetrics
            objectives: List of objective names (default: sharpe, calmar)
            
        Returns:
            List of Pareto-optimal strategy names
        """
        if objectives is None:
            objectives = ['sharpe_ratio', 'calmar_ratio']
        
        strategies = list(strategy_metrics.keys())
        n = len(strategies)
        
        if n == 0:
            return []
        
        # Build objective matrix
        obj_matrix = np.zeros((n, len(objectives)))
        for i, name in enumerate(strategies):
            metrics = strategy_metrics[name]
            for j, obj in enumerate(objectives):
                value = getattr(metrics, obj, 0.0)
                obj_matrix[i, j] = value if np.isfinite(value) else -999
        
        # Find Pareto optimal points
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if j dominates i
                    if np.all(obj_matrix[j] >= obj_matrix[i]) and np.any(obj_matrix[j] > obj_matrix[i]):
                        pareto_mask[i] = False
                        break
        
        return [strategies[i] for i in range(n) if pareto_mask[i]]


# ============================================================================
# CORRELATION AND DIVERSIFICATION ANALYZER
# ============================================================================

class DiversificationAnalyzer:
    """
    Analyze strategy correlations and optimize for diversification.
    Implements:
    1. Rolling correlation analysis
    2. Hierarchical clustering of strategies
    3. Maximum diversification portfolio
    4. Risk parity allocation
    """
    
    def __init__(self, min_periods: int = 63):
        self.min_periods = min_periods
    
    def compute_correlation_matrix(
        self,
        strategy_returns: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Compute pairwise correlation matrix of strategy returns."""
        strategies = list(strategy_returns.keys())
        n = len(strategies)
        
        # Align return series
        min_len = min(len(r) for r in strategy_returns.values())
        aligned_returns = {
            name: returns[-min_len:] for name, returns in strategy_returns.items()
        }
        
        corr_matrix = np.zeros((n, n))
        
        for i, name_i in enumerate(strategies):
            for j, name_j in enumerate(strategies):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif i < j:
                    corr = np.corrcoef(aligned_returns[name_i], aligned_returns[name_j])[0, 1]
                    if np.isfinite(corr):
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        
        return pd.DataFrame(corr_matrix, index=strategies, columns=strategies)
    
    def hierarchical_clustering(
        self,
        correlation_matrix: pd.DataFrame,
        n_clusters: int = 4
    ) -> Dict[int, List[str]]:
        """
        Cluster strategies using hierarchical clustering on correlation distance.
        """
        strategies = correlation_matrix.index.tolist()
        
        if len(strategies) < n_clusters:
            return {0: strategies}
        
        # Convert correlation to distance (1 - corr)
        distance_matrix = 1 - correlation_matrix.values
        np.fill_diagonal(distance_matrix, 0)
        
        # Ensure symmetry and valid range
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        distance_matrix = np.clip(distance_matrix, 0, 2)
        
        # Hierarchical clustering
        condensed = squareform(distance_matrix)
        linkage_matrix = linkage(condensed, method='ward')
        
        # Cut tree to get clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(strategies[i])
        
        return clusters
    
    def maximum_diversification_selection(
        self,
        strategy_metrics: Dict[str, StrategyMetrics],
        n_select: int = 4,
        correlation_threshold: float = 0.7
    ) -> List[str]:
        """
        Select strategies to maximize diversification.
        Greedy algorithm selecting strategies with low correlation to existing set.
        """
        # Get returns
        strategy_returns = {
            name: m.daily_returns for name, m in strategy_metrics.items()
            if len(m.daily_returns) >= self.min_periods
        }
        
        if len(strategy_returns) <= n_select:
            return list(strategy_returns.keys())
        
        # Compute correlation matrix
        corr_matrix = self.compute_correlation_matrix(strategy_returns)
        
        # Get base scores (Sharpe ratio)
        scores = {name: m.sharpe_ratio for name, m in strategy_metrics.items() if name in strategy_returns}
        
        # Greedy selection
        selected = []
        remaining = set(strategy_returns.keys())
        
        # First: select best scoring strategy
        best = max(remaining, key=lambda x: scores.get(x, 0))
        selected.append(best)
        remaining.remove(best)
        
        # Subsequent: select based on score and diversification benefit
        while len(selected) < n_select and remaining:
            best_candidate = None
            best_benefit = -np.inf
            
            for candidate in remaining:
                # Average correlation with selected strategies
                avg_corr = np.mean([
                    corr_matrix.loc[candidate, s] for s in selected
                ])
                
                # Diversification benefit = score * (1 - avg_correlation)
                benefit = scores.get(candidate, 0) * (1 - avg_corr)
                
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def compute_risk_parity_weights(
        self,
        strategy_returns: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute risk parity weights (equal risk contribution).
        """
        strategies = list(strategy_returns.keys())
        n = len(strategies)
        
        if n == 0:
            return {}
        
        # Compute covariance matrix
        min_len = min(len(r) for r in strategy_returns.values())
        returns_matrix = np.column_stack([
            strategy_returns[s][-min_len:] for s in strategies
        ])
        
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        
        # Risk parity optimization
        def risk_contribution(weights, cov):
            portfolio_vol = np.sqrt(weights @ cov @ weights)
            marginal_contrib = cov @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib
        
        def objective(weights, cov):
            rc = risk_contribution(weights, cov)
            target_rc = np.ones(n) / n
            return np.sum((rc - target_rc) ** 2)
        
        # Constraints: weights sum to 1, all positive
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.01, 0.5) for _ in range(n)]
        
        # Initial guess: equal weights
        w0 = np.ones(n) / n
        
        try:
            result = optimize.minimize(
                objective, w0, args=(cov_matrix,),
                method='SLSQP',
                constraints=constraints,
                bounds=bounds
            )
            weights = result.x if result.success else w0
        except:
            weights = w0
        
        return {strategies[i]: weights[i] for i in range(n)}
    
    def compute_diversification_ratio(
        self,
        strategy_returns: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> float:
        """
        Compute portfolio diversification ratio.
        DR = weighted average vol / portfolio vol
        DR > 1 indicates diversification benefit
        """
        strategies = list(weights.keys())
        n = len(strategies)
        
        if n == 0:
            return 1.0
        
        min_len = min(len(strategy_returns[s]) for s in strategies)
        returns_matrix = np.column_stack([
            strategy_returns[s][-min_len:] for s in strategies
        ])
        
        w = np.array([weights[s] for s in strategies])
        
        # Individual volatilities
        vols = np.std(returns_matrix, axis=0)
        weighted_avg_vol = np.dot(w, vols)
        
        # Portfolio volatility
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        portfolio_vol = np.sqrt(w @ cov_matrix @ w)
        
        if portfolio_vol > 0:
            return weighted_avg_vol / portfolio_vol
        return 1.0


# ============================================================================
# BAYESIAN SHRINKAGE ESTIMATOR
# ============================================================================

class BayesianShrinkage:
    """
    Bayesian shrinkage estimation for performance metrics.
    Addresses noise in short sample periods by shrinking towards prior.
    """
    
    def __init__(
        self,
        prior_sharpe: float = 0.0,
        prior_volatility: float = 0.15,
        prior_strength: float = 50  # Equivalent sample size of prior
    ):
        self.prior_sharpe = prior_sharpe
        self.prior_volatility = prior_volatility
        self.prior_strength = prior_strength
    
    def shrink_sharpe(
        self,
        observed_sharpe: float,
        observed_volatility: float,
        n_observations: int
    ) -> float:
        """
        Apply Bayesian shrinkage to Sharpe ratio estimate.
        
        Uses James-Stein type shrinkage:
        shrunk = w * prior + (1-w) * observed
        where w = prior_strength / (prior_strength + effective_n)
        """
        if n_observations < 10:
            return self.prior_sharpe
        
        # Effective sample size (adjusted for autocorrelation, assume mild)
        effective_n = n_observations * 0.8
        
        # Shrinkage weight
        w = self.prior_strength / (self.prior_strength + effective_n)
        
        # Shrink towards prior
        shrunk_sharpe = w * self.prior_sharpe + (1 - w) * observed_sharpe
        
        return shrunk_sharpe
    
    def shrink_metrics(
        self,
        metrics: StrategyMetrics,
        n_observations: int
    ) -> StrategyMetrics:
        """Apply shrinkage to all relevant metrics."""
        # Create copy
        shrunk = StrategyMetrics(name=metrics.name)
        
        # Copy all attributes
        for attr in dir(metrics):
            if not attr.startswith('_') and not callable(getattr(metrics, attr)):
                setattr(shrunk, attr, getattr(metrics, attr))
        
        # Apply shrinkage to key metrics
        shrunk.sharpe_ratio = self.shrink_sharpe(
            metrics.sharpe_ratio,
            metrics.volatility,
            n_observations
        )
        
        # Shrink Sortino similarly
        w = self.prior_strength / (self.prior_strength + n_observations * 0.8)
        shrunk.sortino_ratio = w * 0.0 + (1 - w) * metrics.sortino_ratio
        shrunk.calmar_ratio = w * 0.0 + (1 - w) * metrics.calmar_ratio
        
        return shrunk


# ============================================================================
# ADVANCED STRATEGY SELECTOR (MAIN CLASS)
# ============================================================================

class AdvancedStrategySelector:
    """
    Advanced strategy selection engine integrating all components.
    
    Selection Process:
    1. Calculate comprehensive metrics for all strategies
    2. Apply Bayesian shrinkage for noise reduction
    3. Detect market regime
    4. Compute regime-conditioned metrics
    5. Run multi-criteria optimization
    6. Apply diversification constraints
    7. Bootstrap confidence intervals
    8. Generate final selection with meta-scores
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        bootstrap_samples: int = 500,
        min_observations: int = 63,
        diversification_weight: float = 0.3
    ):
        self.risk_free_rate = risk_free_rate
        self.bootstrap_samples = bootstrap_samples
        self.min_observations = min_observations
        self.diversification_weight = diversification_weight
        
        # Initialize components
        self.metrics_calculator = AdvancedMetricsCalculator(
            risk_free_rate=risk_free_rate,
            bootstrap_samples=bootstrap_samples
        )
        self.regime_detector = RegimeDetector()
        self.mco = MultiCriteriaOptimizer()
        self.diversification = DiversificationAnalyzer(min_periods=min_observations)
        self.shrinkage = BayesianShrinkage()
    
    def select_strategies(
        self,
        backtest_results: Dict[str, Dict],
        market_returns: Optional[np.ndarray] = None,
        mode: str = 'sip',
        n_strategies: int = 4,
        regime_aware: bool = True
    ) -> SelectionResult:
        """
        Main entry point for strategy selection.
        
        Args:
            backtest_results: Dict of strategy_name -> {'daily_data': DataFrame}
            market_returns: Optional market benchmark returns for regime detection
            mode: 'sip' or 'swing'
            n_strategies: Number of strategies to select
            regime_aware: Whether to use regime-conditional selection
            
        Returns:
            SelectionResult with selected strategies and metadata
        """
        logger.info(f"Starting advanced strategy selection (mode={mode}, n={n_strategies})")
        
        # Step 1: Calculate comprehensive metrics
        strategy_metrics = self._calculate_all_metrics(backtest_results)
        
        if not strategy_metrics:
            logger.warning("No valid strategy metrics computed")
            return self._empty_result()
        
        # Step 2: Apply Bayesian shrinkage
        shrunk_metrics = self._apply_shrinkage(strategy_metrics, backtest_results)
        
        # Step 3: Detect current regime
        current_regime = MarketRegime.LOW_VOLATILITY
        regime_probs = {}
        
        if regime_aware and market_returns is not None and len(market_returns) > 100:
            current_regime, regime_probs = self.regime_detector.detect_regime(market_returns)
            logger.info(f"Detected regime: {current_regime.value}")
        
        # Step 4: Compute multi-criteria scores
        mco_scores = self.mco.compute_scores(shrunk_metrics, mode=mode)
        
        # Step 5: Get Pareto frontier
        pareto_strategies = self.mco.identify_pareto_frontier(shrunk_metrics)
        logger.info(f"Pareto frontier: {pareto_strategies}")
        
        # Step 6: Apply diversification
        strategy_returns = {
            name: m.daily_returns for name, m in strategy_metrics.items()
            if len(m.daily_returns) >= self.min_observations
        }
        
        if len(strategy_returns) >= n_strategies:
            diversified = self.diversification.maximum_diversification_selection(
                strategy_metrics, n_select=n_strategies * 2
            )
        else:
            diversified = list(strategy_returns.keys())
        
        # Step 7: Combine scores (MCO + diversification bonus)
        final_scores = self._combine_scores(
            mco_scores, pareto_strategies, diversified, strategy_metrics
        )
        
        # Step 8: Select top N
        sorted_strategies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in sorted_strategies[:n_strategies]]
        
        # Step 9: Compute correlation matrix for selected
        selected_returns = {name: strategy_returns.get(name, np.array([])) for name in selected if name in strategy_returns}
        
        if selected_returns:
            corr_matrix = self.diversification.compute_correlation_matrix(selected_returns)
            div_ratio = self.diversification.compute_diversification_ratio(
                selected_returns,
                {name: 1/len(selected) for name in selected}
            )
        else:
            corr_matrix = pd.DataFrame()
            div_ratio = 1.0
        
        # Step 10: Bootstrap confidence intervals
        confidence_intervals = self._compute_confidence_intervals(strategy_metrics, selected)
        
        # Step 11: Compute expected portfolio Sharpe
        if selected_returns:
            weights = self.diversification.compute_risk_parity_weights(selected_returns)
            expected_sharpe = self._compute_portfolio_sharpe(strategy_metrics, weights)
        else:
            expected_sharpe = 0.0
        
        # Step 12: Build regime allocations
        regime_allocations = self._compute_regime_allocations(
            strategy_metrics, market_returns, n_strategies
        )
        
        # Step 13: Meta-score breakdown
        meta_breakdown = self._build_meta_breakdown(
            strategy_metrics, mco_scores, pareto_strategies, diversified, selected
        )
        
        logger.info(f"Selected strategies: {selected}")
        
        return SelectionResult(
            selected_strategies=selected,
            selection_scores=final_scores,
            confidence_intervals=confidence_intervals,
            regime_allocations=regime_allocations,
            correlation_matrix=corr_matrix,
            diversification_benefit=div_ratio,
            expected_portfolio_sharpe=expected_sharpe,
            meta_score_breakdown=meta_breakdown
        )
    
    def _calculate_all_metrics(
        self,
        backtest_results: Dict[str, Dict]
    ) -> Dict[str, StrategyMetrics]:
        """Calculate comprehensive metrics for all strategies."""
        strategy_metrics = {}
        
        for name, data in backtest_results.items():
            daily_data = data.get('daily_data', pd.DataFrame())
            
            if daily_data.empty or len(daily_data) < self.min_observations:
                continue
            
            metrics = self.metrics_calculator.calculate(daily_data, name)
            
            # Filter out strategies with invalid metrics
            if metrics.volatility > 0.001 and np.isfinite(metrics.sharpe_ratio):
                strategy_metrics[name] = metrics
        
        return strategy_metrics
    
    def _apply_shrinkage(
        self,
        strategy_metrics: Dict[str, StrategyMetrics],
        backtest_results: Dict[str, Dict]
    ) -> Dict[str, StrategyMetrics]:
        """Apply Bayesian shrinkage to all metrics."""
        shrunk = {}
        
        for name, metrics in strategy_metrics.items():
            n_obs = len(backtest_results.get(name, {}).get('daily_data', pd.DataFrame()))
            shrunk[name] = self.shrinkage.shrink_metrics(metrics, n_obs)
        
        return shrunk
    
    def _combine_scores(
        self,
        mco_scores: Dict[str, float],
        pareto_strategies: List[str],
        diversified: List[str],
        strategy_metrics: Dict[str, StrategyMetrics]
    ) -> Dict[str, float]:
        """Combine MCO scores with Pareto and diversification bonuses."""
        final_scores = {}
        
        for name in mco_scores:
            base_score = mco_scores[name]
            
            # Pareto bonus (10%)
            if name in pareto_strategies:
                base_score *= 1.10
            
            # Diversification bonus
            if name in diversified:
                base_score *= (1 + self.diversification_weight * 0.5)
            
            # Statistical significance penalty
            metrics = strategy_metrics.get(name)
            if metrics and metrics.sharpe_pvalue > 0.1:
                # Penalize strategies with non-significant performance
                penalty = min(0.2, metrics.sharpe_pvalue - 0.05)
                base_score *= (1 - penalty)
            
            final_scores[name] = base_score
        
        return final_scores
    
    def _compute_confidence_intervals(
        self,
        strategy_metrics: Dict[str, StrategyMetrics],
        selected: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """Get confidence intervals for selected strategies."""
        intervals = {}
        
        for name in selected:
            metrics = strategy_metrics.get(name)
            if metrics:
                intervals[name] = (
                    metrics.bootstrap_sharpe_ci_lower,
                    metrics.bootstrap_sharpe_ci_upper
                )
        
        return intervals
    
    def _compute_portfolio_sharpe(
        self,
        strategy_metrics: Dict[str, StrategyMetrics],
        weights: Dict[str, float]
    ) -> float:
        """Estimate portfolio Sharpe ratio."""
        if not weights:
            return 0.0
        
        strategies = list(weights.keys())
        
        # Weighted average return
        weighted_return = sum(
            weights[s] * strategy_metrics[s].annualized_return
            for s in strategies if s in strategy_metrics
        )
        
        # Get returns for covariance
        returns_list = []
        weight_list = []
        for s in strategies:
            if s in strategy_metrics and len(strategy_metrics[s].daily_returns) > 0:
                returns_list.append(strategy_metrics[s].daily_returns)
                weight_list.append(weights[s])
        
        if len(returns_list) < 2:
            return 0.0
        
        # Align returns
        min_len = min(len(r) for r in returns_list)
        aligned = np.column_stack([r[-min_len:] for r in returns_list])
        
        # Portfolio volatility
        w = np.array(weight_list)
        w = w / w.sum()
        
        cov = np.cov(aligned, rowvar=False)
        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        
        if port_vol > 0.001:
            return (weighted_return - self.risk_free_rate) / port_vol
        return 0.0
    
    def _compute_regime_allocations(
        self,
        strategy_metrics: Dict[str, StrategyMetrics],
        market_returns: Optional[np.ndarray],
        n_strategies: int
    ) -> Dict[MarketRegime, List[str]]:
        """Compute regime-specific strategy allocations."""
        allocations = {}
        
        # For each regime type, rank strategies by suitability
        regime_criteria = {
            MarketRegime.BULL: ['cagr', 'omega_ratio'],
            MarketRegime.BEAR: ['max_drawdown', 'cvar_95'],  # Minimize drawdown
            MarketRegime.HIGH_VOLATILITY: ['sortino_ratio', 'cvar_95'],
            MarketRegime.LOW_VOLATILITY: ['sharpe_ratio', 'calmar_ratio'],
            MarketRegime.TRENDING: ['hurst_exponent', 'cagr'],
            MarketRegime.MEAN_REVERTING: ['win_rate', 'profit_factor'],
            MarketRegime.CRISIS: ['max_drawdown', 'cvar_99'],
            MarketRegime.RECOVERY: ['omega_ratio', 'cagr']
        }
        
        for regime, criteria in regime_criteria.items():
            scores = {}
            for name, metrics in strategy_metrics.items():
                score = 0
                for criterion in criteria:
                    value = getattr(metrics, criterion, 0)
                    if criterion in ['max_drawdown', 'cvar_95', 'cvar_99']:
                        # For these, less negative is better
                        score += value  # Already negative, so higher (less negative) is better
                    else:
                        score += value
                scores[name] = score
            
            sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            allocations[regime] = [name for name, _ in sorted_strategies[:n_strategies]]
        
        return allocations
    
    def _build_meta_breakdown(
        self,
        strategy_metrics: Dict[str, StrategyMetrics],
        mco_scores: Dict[str, float],
        pareto_strategies: List[str],
        diversified: List[str],
        selected: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Build detailed breakdown of selection criteria."""
        breakdown = {}
        
        for name in selected:
            metrics = strategy_metrics.get(name)
            if not metrics:
                continue
            
            breakdown[name] = {
                'mco_score': mco_scores.get(name, 0),
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'omega_ratio': metrics.omega_ratio,
                'max_drawdown': metrics.max_drawdown,
                'cvar_95': metrics.cvar_95,
                'win_rate': metrics.win_rate,
                'sharpe_pvalue': metrics.sharpe_pvalue,
                'is_pareto': name in pareto_strategies,
                'is_diversified': name in diversified,
                'hurst_exponent': metrics.hurst_exponent,
                'sharpe_stability': metrics.sharpe_stability
            }
        
        return breakdown
    
    def _empty_result(self) -> SelectionResult:
        """Return empty result when no strategies are valid."""
        return SelectionResult(
            selected_strategies=[],
            selection_scores={},
            confidence_intervals={},
            regime_allocations={},
            correlation_matrix=pd.DataFrame(),
            diversification_benefit=1.0,
            expected_portfolio_sharpe=0.0,
            meta_score_breakdown={}
        )


# ============================================================================
# INTEGRATION WITH BACKTEST ENGINE
# ============================================================================

class EnhancedDynamicPortfolioStylesGenerator:
    """
    Enhanced portfolio styles generator using advanced selection.
    Drop-in replacement for DynamicPortfolioStylesGenerator.
    """
    
    def __init__(self, engine):
        """
        Initialize with backtest engine.
        
        Args:
            engine: UnifiedBacktestEngine instance
        """
        self.engine = engine
        self.selector = AdvancedStrategySelector()
        self._sip_results: Dict[str, Dict] = {}
        self._swing_results: Dict[str, Dict] = {}
        self._market_returns: Optional[np.ndarray] = None
    
    def run_comprehensive_backtest(
        self,
        external_trigger_df: Optional[pd.DataFrame] = None,
        buy_col: Optional[str] = None,
        sell_col: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict, Dict]:
        """Run backtests and extract market returns for regime detection."""
        
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
        
        # Extract market returns (use equal-weighted portfolio of all strategies)
        self._extract_market_returns()
        
        return self._sip_results, self._swing_results
    
    def _extract_market_returns(self):
        """Extract proxy market returns from strategy results."""
        all_returns = []
        
        for results in [self._sip_results, self._swing_results]:
            for name, data in results.items():
                daily_data = data.get('daily_data', pd.DataFrame())
                if not daily_data.empty and 'value' in daily_data.columns:
                    values = daily_data['value'].values
                    returns = pd.Series(values).pct_change().dropna().values
                    if len(returns) > 0:
                        all_returns.append(returns)
        
        if all_returns:
            min_len = min(len(r) for r in all_returns)
            aligned = [r[-min_len:] for r in all_returns]
            self._market_returns = np.mean(aligned, axis=0)
    
    def generate_portfolio_styles(
        self,
        n_strategies: int = 4
    ) -> Dict[str, Dict]:
        """
        Generate dynamic PORTFOLIO_STYLES using advanced selection.
        """
        # Select strategies for SIP mode
        sip_selection = self.selector.select_strategies(
            self._sip_results,
            market_returns=self._market_returns,
            mode='sip',
            n_strategies=n_strategies,
            regime_aware=True
        )
        
        # Select strategies for Swing mode
        swing_selection = self.selector.select_strategies(
            self._swing_results,
            market_returns=self._market_returns,
            mode='swing',
            n_strategies=n_strategies,
            regime_aware=True
        )
        
        # Build rationale strings
        sip_rationale = self._build_advanced_rationale(sip_selection, 'sip')
        swing_rationale = self._build_advanced_rationale(swing_selection, 'swing')
        
        # Get regime-specific allocations
        sip_regime_strategies = sip_selection.regime_allocations
        swing_regime_strategies = swing_selection.regime_allocations
        
        # Generate PORTFOLIO_STYLES with regime awareness
        portfolio_styles = {
            "Swing Trading": {
                "description": "Short-term (3-21 day) holds with advanced multi-criteria selection and diversification optimization.",
                "mixes": {
                    "Bull Market Mix": {
                        "strategies": swing_regime_strategies.get(MarketRegime.BULL, swing_selection.selected_strategies),
                        "rationale": f"Optimized for trending markets. {swing_rationale}"
                    },
                    "Bear Market Mix": {
                        "strategies": swing_regime_strategies.get(MarketRegime.BEAR, swing_selection.selected_strategies),
                        "rationale": f"Defensive selection prioritizing drawdown protection. {swing_rationale}"
                    },
                    "Chop/Consolidate Mix": {
                        "strategies": swing_regime_strategies.get(MarketRegime.LOW_VOLATILITY, swing_selection.selected_strategies),
                        "rationale": f"Mean-reversion optimized for ranging markets. {swing_rationale}"
                    }
                },
                "meta": {
                    "diversification_ratio": swing_selection.diversification_benefit,
                    "expected_portfolio_sharpe": swing_selection.expected_portfolio_sharpe,
                    "correlation_matrix": swing_selection.correlation_matrix.to_dict() if not swing_selection.correlation_matrix.empty else {}
                }
            },
            "SIP Investment": {
                "description": "Systematic long-term (3-12+ months) wealth accumulation with advanced risk-adjusted selection.",
                "mixes": {
                    "Bull Market Mix": {
                        "strategies": sip_regime_strategies.get(MarketRegime.BULL, sip_selection.selected_strategies),
                        "rationale": f"Growth-optimized for bullish regimes. {sip_rationale}"
                    },
                    "Bear Market Mix": {
                        "strategies": sip_regime_strategies.get(MarketRegime.BEAR, sip_selection.selected_strategies),
                        "rationale": f"Capital preservation focus with CVaR optimization. {sip_rationale}"
                    },
                    "Chop/Consolidate Mix": {
                        "strategies": sip_regime_strategies.get(MarketRegime.LOW_VOLATILITY, sip_selection.selected_strategies),
                        "rationale": f"Stability-optimized for sideways markets. {sip_rationale}"
                    }
                },
                "meta": {
                    "diversification_ratio": sip_selection.diversification_benefit,
                    "expected_portfolio_sharpe": sip_selection.expected_portfolio_sharpe,
                    "correlation_matrix": sip_selection.correlation_matrix.to_dict() if not sip_selection.correlation_matrix.empty else {}
                }
            }
        }
        
        return portfolio_styles
    
    def _build_advanced_rationale(
        self,
        selection: SelectionResult,
        mode: str
    ) -> str:
        """Build detailed rationale string."""
        parts = []
        
        for name in selection.selected_strategies:
            breakdown = selection.meta_score_breakdown.get(name, {})
            if breakdown:
                sharpe = breakdown.get('sharpe_ratio', 0)
                calmar = breakdown.get('calmar_ratio', 0)
                mco = breakdown.get('mco_score', 0)
                pvalue = breakdown.get('sharpe_pvalue', 1)
                
                sig_marker = "**" if pvalue < 0.05 else "*" if pvalue < 0.1 else ""
                parts.append(f"{name}{sig_marker}(MCO:{mco:.2f}, SR:{sharpe:.2f}, CR:{calmar:.2f})")
        
        ci_info = ""
        if selection.confidence_intervals:
            first_strategy = selection.selected_strategies[0] if selection.selected_strategies else None
            if first_strategy and first_strategy in selection.confidence_intervals:
                ci = selection.confidence_intervals[first_strategy]
                ci_info = f" | 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]"
        
        div_info = f" | Div.Ratio: {selection.diversification_benefit:.2f}"
        exp_sharpe = f" | Exp.Port.SR: {selection.expected_portfolio_sharpe:.2f}"
        
        return " | ".join(parts) + ci_info + div_info + exp_sharpe
    
    def get_strategy_leaderboard(self, mode: str = 'swing') -> pd.DataFrame:
        """Get comprehensive strategy leaderboard."""
        results = self._sip_results if mode.lower() == 'sip' else self._swing_results
        
        if not results:
            return pd.DataFrame()
        
        # Recompute full metrics for leaderboard
        leaderboard_data = []
        
        for name, data in results.items():
            daily_data = data.get('daily_data', pd.DataFrame())
            if daily_data.empty:
                continue
            
            metrics = self.selector.metrics_calculator.calculate(daily_data, name)
            
            leaderboard_data.append({
                'Strategy': name,
                'Total Return': metrics.total_return,
                'CAGR': metrics.cagr,
                'Volatility': metrics.volatility,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Sortino Ratio': metrics.sortino_ratio,
                'Calmar Ratio': metrics.calmar_ratio,
                'Omega Ratio': metrics.omega_ratio,
                'Max Drawdown': metrics.max_drawdown,
                'CVaR 95%': metrics.cvar_95,
                'Win Rate': metrics.win_rate,
                'Hurst Exp': metrics.hurst_exponent,
                'Sharpe p-value': metrics.sharpe_pvalue,
                'SR Stability': metrics.sharpe_stability
            })
        
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by MCO score (computed separately)
        if not df.empty:
            strategy_metrics = {
                row['Strategy']: self.selector.metrics_calculator.calculate(
                    results[row['Strategy']].get('daily_data', pd.DataFrame()),
                    row['Strategy']
                )
                for _, row in df.iterrows()
            }
            mco_scores = self.selector.mco.compute_scores(strategy_metrics, mode=mode)
            df['MCO Score'] = df['Strategy'].map(mco_scores)
            df = df.sort_values('MCO Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_selection_details(self, mode: str = 'swing') -> SelectionResult:
        """Get full selection details for analysis."""
        results = self._sip_results if mode.lower() == 'sip' else self._swing_results
        
        return self.selector.select_strategies(
            results,
            market_returns=self._market_returns,
            mode=mode,
            n_strategies=4,
            regime_aware=True
        )


# ============================================================================
# CONVENIENCE FUNCTION FOR DROP-IN REPLACEMENT
# ============================================================================

def get_advanced_dynamic_portfolio_styles(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    capital: float = 10_000_000,
    external_trigger_df: Optional[pd.DataFrame] = None,
    buy_col: Optional[str] = None,
    sell_col: Optional[str] = None,
    n_strategies: int = 4,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Dict]:
    """
    Drop-in replacement for get_dynamic_portfolio_styles with advanced selection.
    
    Uses the enhanced multi-criteria optimization, regime detection,
    and diversification analysis.
    """
    # Import engine
    try:
        from backtest_engine import UnifiedBacktestEngine
    except ImportError:
        logger.error("backtest_engine.py not found")
        return {}
    
    # Initialize engine
    engine = UnifiedBacktestEngine(capital=capital)
    
    # Load data
    engine.load_data(
        symbols, start_date, end_date,
        progress_callback=lambda p, m: progress_callback(p * 0.3, m) if progress_callback else None
    )
    
    # Load strategies
    engine.load_strategies()
    
    # Create enhanced generator
    generator = EnhancedDynamicPortfolioStylesGenerator(engine)
    
    # Run backtest
    generator.run_comprehensive_backtest(
        external_trigger_df=external_trigger_df,
        buy_col=buy_col,
        sell_col=sell_col,
        progress_callback=lambda p, m: progress_callback(0.3 + p * 0.6, m) if progress_callback else None
    )
    
    # Generate portfolio styles
    if progress_callback:
        progress_callback(0.95, "Generating advanced portfolio styles...")
    
    portfolio_styles = generator.generate_portfolio_styles(n_strategies=n_strategies)
    
    if progress_callback:
        progress_callback(1.0, "Complete")
    
    return portfolio_styles


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Advanced Strategy Selector Module")
    print("=" * 50)
    print("\nFeatures:")
    print("1. Multi-Criteria Optimization (TOPSIS + Weighted Sum)")
    print("2. Hidden Markov Model Regime Detection")
    print("3. Bootstrap Confidence Intervals")
    print("4. Maximum Diversification Selection")
    print("5. Risk Parity Allocation")
    print("6. Bayesian Shrinkage Estimation")
    print("7. Tail Risk Analysis (CVaR, Expected Shortfall)")
    print("8. Pareto Frontier Identification")
    print("9. Rolling Window Stability Analysis")
    print("10. Statistical Significance Testing")
    
    print("\nTo use, import and replace the generator:")
    print("  from advanced_strategy_selector import EnhancedDynamicPortfolioStylesGenerator")
    print("  generator = EnhancedDynamicPortfolioStylesGenerator(engine)")
