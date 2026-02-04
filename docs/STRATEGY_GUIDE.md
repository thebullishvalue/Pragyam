# PRAGYAM Strategy Guide

## Overview

Pragyam implements 80+ quantitative strategies organized into distinct categories. Each strategy generates a ranked portfolio of stocks based on its unique alpha-generation logic.

---

## Strategy Categories

### 1. Momentum Strategies

**Core Concept**: Stocks that have performed well recently tend to continue performing well in the near term (momentum persistence).

| Strategy | Signal Logic | Best Market Condition |
|----------|-------------|----------------------|
| `MOM1Strategy` | 12-month momentum with 1-month reversal filter | Trending markets |
| `MOM2Strategy` | Acceleration of momentum (momentum of momentum) | Strong trends |
| `MomentumMasters` | Multi-timeframe momentum consensus | All conditions |
| `MomentumCascade` | Cascading momentum across 3/6/12 month windows | Sustained trends |
| `DualMomentum` | Absolute + relative momentum combination | Risk-aware momentum |

**Parameters**:
- Lookback: 6-12 months typical
- Rebalance: Weekly to monthly
- Position sizing: Momentum score weighted

---

### 2. Mean-Reversion Strategies

**Core Concept**: Extreme price deviations tend to revert to historical averages.

| Strategy | Signal Logic | Best Market Condition |
|----------|-------------|----------------------|
| `AdaptiveZScoreEngine` | Z-score from adaptive moving average | Range-bound markets |
| `KalmanFilterMomentum` | Kalman-filtered price deviation | Noisy markets |
| `BayesianMomentumUpdater` | Bayesian estimate of fair value | Uncertain regimes |

**Parameters**:
- Z-score threshold: ±2.0 standard deviations
- Mean estimation: EMA, Kalman, or Bayesian

---

### 3. Volatility Strategies

**Core Concept**: Exploit volatility patterns - breakouts, compressions, and regime changes.

| Strategy | Signal Logic | Best Market Condition |
|----------|-------------|----------------------|
| `VolatilitySurfer` | Ride volatility expansion phases | Breakout markets |
| `AdaptiveVolBreakout` | ATR-based breakout detection | Consolidation breakouts |
| `VolReversalHarvester` | Fade extreme volatility spikes | Mean-reverting vol |
| `VolatilityRegimeTrader` | HMM-based regime classification | Regime transitions |

**Parameters**:
- ATR period: 14-21 days
- Volatility percentile: 20th/80th thresholds

---

### 4. Factor-Based Strategies

**Core Concept**: Systematic exposure to documented risk factors.

| Strategy | Signal Logic | Best Market Condition |
|----------|-------------|----------------------|
| `VolatilityAdjustedValue` | Value score / volatility | Quality-seeking markets |
| `CrossSectionalAlpha` | Cross-sectional return ranking | Dispersion markets |
| `RelativeStrengthRotator` | Sector/stock relative strength | Rotational markets |
| `FactorMomentumStrategy` | Momentum on factor exposures | Factor-aware allocation |

---

### 5. Advanced/Hybrid Strategies

**Core Concept**: Combine multiple signals using sophisticated mathematical frameworks.

| Strategy | Signal Logic | Best Market Condition |
|----------|-------------|----------------------|
| `HiddenMarkovModel` | HMM state-based allocation | Regime-dependent |
| `WaveletDenoiser` | Wavelet-filtered signals | Noisy environments |
| `CopulaBlendStrategy` | Copula-based correlation modeling | Tail-dependent |
| `EnsembleVotingStrategy` | Multi-strategy voting consensus | Uncertain markets |
| `AttentionMechanism` | Attention-weighted signal aggregation | Complex patterns |

---

## Strategy Selection Framework

### TOPSIS Multi-Criteria Optimization

Pragyam uses TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) to select optimal strategy combinations:

```
Score = √(TOPSIS_score × Weighted_Sum_score)
```

**Criteria Weights**:
| Criterion | Weight | Direction |
|-----------|--------|-----------|
| Sharpe Ratio | 35% | Maximize |
| Sortino Ratio | 25% | Maximize |
| Calmar Ratio | 15% | Maximize |
| Max Drawdown | 10% | Minimize |
| Win Rate | 10% | Maximize |
| Stability | 5% | Minimize |

### Mode-Specific Adjustments

**SIP Mode** (Long-term accumulation):
- Higher weight on Calmar ratio
- Emphasis on drawdown protection
- Lower turnover preference

**Swing Mode** (Active trading):
- Higher weight on Sortino ratio
- Emphasis on win rate
- Accepts higher turnover

---

## Position Sizing

### Tier-Based Allocation

Stocks are allocated to tiers based on conviction:

| Tier | Conviction | Typical Weight |
|------|------------|----------------|
| Tier 1 | Highest (Top 10) | 5-8% each |
| Tier 2 | High (11-20) | 3-5% each |
| Tier 3 | Medium (21-30) | 2-3% each |
| Tier 4 | Lower (31-40) | 1-2% each |
| Tier 5 | Lowest (41-50) | 0.5-1% each |

### Risk Parity Weighting

Within each tier, risk parity ensures equal risk contribution:

```python
def risk_parity_weights(returns):
    cov = returns.cov()
    # Optimize for equal marginal risk contribution
    weights = optimize.minimize(
        risk_contribution_objective,
        initial_weights,
        constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
    )
    return weights
```

---

## Adding Custom Strategies

### Strategy Template

```python
class MyCustomStrategy(BaseStrategy):
    """
    Custom strategy description.
    """
    
    def __init__(self):
        super().__init__(
            name="MyCustomStrategy",
            description="My custom alpha generation logic"
        )
    
    def generate_portfolio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ranked portfolio.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with columns: symbol, score, weight
        """
        # Your alpha logic here
        scores = self._calculate_scores(df)
        
        # Rank and weight
        ranked = scores.sort_values('score', ascending=False)
        ranked['weight'] = self._calculate_weights(ranked)
        
        return ranked[['symbol', 'score', 'weight']]
    
    def _calculate_scores(self, df):
        # Implement your scoring logic
        pass
    
    def _calculate_weights(self, ranked):
        # Implement your weighting logic
        pass
```

### Registration

Add to `strategies.py`:

```python
from my_strategies import MyCustomStrategy

STRATEGY_REGISTRY['MyCustomStrategy'] = MyCustomStrategy
```

---

## Performance Expectations

### Historical Backtests (2020-2025)

| Strategy Category | Avg Sharpe | Avg Max DD | Win Rate |
|-------------------|------------|------------|----------|
| Momentum | 1.2-1.8 | -15% to -25% | 55-60% |
| Mean-Reversion | 0.8-1.2 | -10% to -20% | 50-55% |
| Volatility | 1.0-1.5 | -20% to -30% | 52-58% |
| Factor | 0.9-1.3 | -12% to -22% | 53-57% |
| Hybrid/Advanced | 1.3-2.0 | -10% to -18% | 55-62% |

*Note: Past performance does not guarantee future results.*

---

## Best Practices

1. **Diversification**: Use strategies from multiple categories
2. **Regime Awareness**: Let regime detection adjust weights
3. **Rebalancing**: Weekly rebalancing for active strategies
4. **Risk Limits**: Set maximum drawdown thresholds
5. **Monitoring**: Track rolling Sharpe for strategy degradation

---

*© 2024-2026 Hemrek Capital - Proprietary & Confidential*
