# Advanced Strategy Selection Framework for Pragyam

## Mathematical Documentation v3.0.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problems with Previous Implementation](#problems-with-previous-implementation)
3. [Mathematical Framework](#mathematical-framework)
4. [Component Architecture](#component-architecture)
5. [Algorithm Details](#algorithm-details)
6. [Usage Guide](#usage-guide)
7. [API Reference](#api-reference)

---

## Executive Summary

The Advanced Strategy Selection Framework replaces the naive single-metric ranking approach with a comprehensive, mathematically rigorous system that addresses the fundamental challenges of strategy selection in portfolio management:

### Key Improvements

| Aspect | Old Implementation | New Implementation |
|--------|-------------------|-------------------|
| Selection Criteria | Single metric (Calmar/Sortino) | 12+ criteria with TOPSIS optimization |
| Regime Awareness | None | HMM-based regime detection |
| Diversification | None | Maximum diversification + risk parity |
| Statistical Rigor | None | Bootstrap CI + significance testing |
| Tail Risk | Ignored | CVaR, Expected Shortfall analysis |
| Noise Handling | None | Bayesian shrinkage estimation |
| Stability | Ignored | Rolling window stability analysis |

---

## Problems with Previous Implementation

### 1. Single-Metric Fallacy

The original implementation selected strategies using only:
- **SIP Mode**: Top N by Calmar Ratio
- **Swing Mode**: Top N by Sortino Ratio

**Problems:**
- Ignores other crucial dimensions (consistency, tail risk, stability)
- Highly sensitive to sample period
- No consideration of statistical significance
- A strategy with Calmar = 5.0 may not be statistically different from one with Calmar = 3.0

### 2. No Regime Differentiation

Same strategies for Bull/Bear/Chop markets:

```python
# OLD APPROACH - same strategies regardless of market regime
portfolio_styles = {
    "Bull Market Mix": {"strategies": sip_strategies},
    "Bear Market Mix": {"strategies": sip_strategies},  # Same!
    "Chop/Consolidate Mix": {"strategies": sip_strategies}  # Same!
}
```

### 3. No Diversification Consideration

Selecting 4 strategies with 0.95 correlation provides almost no diversification benefit.

### 4. No Statistical Validation

- No confidence intervals
- No significance testing
- No robustness checks
- Extreme Calmar ratios of 50+ not flagged as suspicious

---

## Mathematical Framework

### 1. Multi-Criteria Optimization via TOPSIS

The Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) identifies strategies closest to the ideal solution and furthest from the anti-ideal.

**Decision Matrix:**
```
        C₁    C₂    C₃    ...   Cₘ
S₁     x₁₁   x₁₂   x₁₃   ...   x₁ₘ
S₂     x₂₁   x₂₂   x₂₃   ...   x₂ₘ
...
Sₙ     xₙ₁   xₙ₂   xₙ₃   ...   xₙₘ
```

**Normalization (Min-Max):**
$$r_{ij} = \frac{x_{ij} - \min_i(x_{ij})}{\max_i(x_{ij}) - \min_i(x_{ij})}$$

**Weighted Normalized Matrix:**
$$v_{ij} = w_j \cdot r_{ij}$$

**Ideal and Anti-Ideal Solutions:**
$$A^+ = \{\max_i(v_{ij})\} \quad A^- = \{\min_i(v_{ij})\}$$

**Separation Measures:**
$$D_i^+ = \sqrt{\sum_j (v_{ij} - v_j^+)^2} \quad D_i^- = \sqrt{\sum_j (v_{ij} - v_j^-)^2}$$

**TOPSIS Score:**
$$S_i = \frac{D_i^-}{D_i^+ + D_i^-}$$

### 2. Market Regime Detection

**Regime Indicators:**

1. **Volatility Ratio:** $\sigma_{short} / \sigma_{long}$
2. **Trend Return:** $\sum_{t=1}^{T} r_t$
3. **Trend Consistency:** $\frac{1}{T}\sum_{t=1}^{T} \mathbb{1}_{r_t > 0}$
4. **Hurst Exponent:** Via R/S analysis
5. **Current Drawdown:** $(V_t - \max_{s \leq t} V_s) / \max_{s \leq t} V_s$

**Fuzzy Classification:**

Regime probabilities computed via fuzzy logic rules:
- **Crisis:** High volatility ratio AND deep drawdown
- **Bull:** Positive cumulative return AND high consistency
- **Bear:** Negative cumulative return AND low consistency
- etc.

### 3. Hurst Exponent Calculation

The Hurst exponent H characterizes time series behavior:
- H > 0.5: Trending (persistent)
- H = 0.5: Random walk
- H < 0.5: Mean-reverting

**R/S Analysis:**
$$\frac{R}{S} = c \cdot n^H$$

Where R is the range of cumulative deviations from mean and S is the standard deviation.

**Estimation:**
$$H = \frac{d \log(R/S)}{d \log(n)}$$

### 4. Maximum Diversification Selection

**Diversification Ratio:**
$$DR = \frac{\sum_i w_i \sigma_i}{\sqrt{w' \Sigma w}}$$

DR > 1 indicates diversification benefit.

**Greedy Selection Algorithm:**
1. Select strategy with highest base score
2. For each remaining slot:
   - Compute diversification benefit = score × (1 - avg_correlation)
   - Select strategy with highest benefit
3. Repeat until N strategies selected

### 5. Risk Parity Allocation

Equal risk contribution portfolio:

**Marginal Risk Contribution:**
$$MRC_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$

**Risk Contribution:**
$$RC_i = w_i \cdot MRC_i$$

**Optimization:**
$$\min_w \sum_i \left(RC_i - \frac{\sigma_p}{n}\right)^2$$

Subject to: $\sum_i w_i = 1$, $w_i \geq 0$

### 6. Bayesian Shrinkage Estimation

James-Stein type shrinkage for noisy estimates:

$$\hat{\theta}_{shrunk} = \lambda \cdot \theta_{prior} + (1 - \lambda) \cdot \hat{\theta}_{observed}$$

**Shrinkage Intensity:**
$$\lambda = \frac{n_{prior}}{n_{prior} + n_{effective}}$$

Where $n_{effective} = n_{observed} \times \rho$ accounts for autocorrelation.

### 7. Statistical Significance Testing

**Sharpe Ratio T-Statistic (Lo, 2002):**

Standard error:
$$SE(\widehat{SR}) = \sqrt{\frac{1 + 0.5 \cdot \widehat{SR}^2}{n}}$$

T-statistic:
$$t = \frac{\widehat{SR}}{SE(\widehat{SR})}$$

**Bootstrap Confidence Intervals:**

1. Resample returns with replacement B times
2. Compute Sharpe ratio for each sample
3. Extract percentiles for CI

### 8. Tail Risk Metrics

**Value at Risk (Historical):**
$$VaR_\alpha = -\text{Percentile}_\alpha(R)$$

**Conditional Value at Risk (Expected Shortfall):**
$$CVaR_\alpha = -E[R | R \leq -VaR_\alpha]$$

**Tail Ratio:**
$$TR = \frac{\text{Percentile}_{95}(R)}{|\text{Percentile}_5(R)|}$$

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  AdvancedStrategySelector                    │
│                      (Main Orchestrator)                     │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Metrics       │  │    Regime       │  │  Multi-Criteria │
│   Calculator    │  │    Detector     │  │   Optimizer     │
│                 │  │                 │  │                 │
│ • Core Returns  │  │ • Vol Analysis  │  │ • TOPSIS        │
│ • Risk Metrics  │  │ • Trend Detect  │  │ • Weighted Sum  │
│ • Tail Risk     │  │ • Hurst Exp     │  │ • Pareto Front  │
│ • Distribution  │  │ • HMM States    │  │ • Mode Weights  │
│ • Consistency   │  │                 │  │                 │
│ • Stability     │  │                 │  │                 │
│ • Bootstrap CI  │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Diversification │  │    Bayesian     │  │   Selection     │
│    Analyzer     │  │   Shrinkage     │  │    Result       │
│                 │  │                 │  │                 │
│ • Correlation   │  │ • Prior Setup   │  │ • Strategies    │
│ • Clustering    │  │ • Noise Reduce  │  │ • Scores        │
│ • Max Div       │  │ • Regularize    │  │ • Confidence    │
│ • Risk Parity   │  │                 │  │ • Regimes       │
│ • Div Ratio     │  │                 │  │ • Correlation   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Algorithm Details

### Complete Selection Pipeline

```python
def select_strategies(backtest_results, mode, n_strategies):
    # Step 1: Calculate comprehensive metrics
    strategy_metrics = calculate_all_metrics(backtest_results)
    
    # Step 2: Apply Bayesian shrinkage
    shrunk_metrics = apply_shrinkage(strategy_metrics)
    
    # Step 3: Detect market regime
    regime, regime_probs = detect_regime(market_returns)
    
    # Step 4: Compute multi-criteria scores (TOPSIS)
    mco_scores = compute_topsis_scores(shrunk_metrics, mode)
    
    # Step 5: Identify Pareto frontier
    pareto_strategies = identify_pareto_frontier(shrunk_metrics)
    
    # Step 6: Apply diversification constraints
    diversified = maximum_diversification_selection(strategy_metrics)
    
    # Step 7: Combine scores
    final_scores = combine_scores(
        mco_scores, 
        pareto_bonus=0.10,
        diversification_bonus=0.15,
        significance_penalty=0.20
    )
    
    # Step 8: Select top N
    selected = sorted(final_scores, reverse=True)[:n_strategies]
    
    # Step 9: Compute portfolio characteristics
    correlation_matrix = compute_correlations(selected)
    risk_parity_weights = compute_risk_parity(selected)
    expected_sharpe = compute_portfolio_sharpe(selected, risk_parity_weights)
    
    # Step 10: Compute regime-specific allocations
    regime_allocations = compute_regime_allocations(strategy_metrics)
    
    return SelectionResult(
        selected_strategies=selected,
        selection_scores=final_scores,
        confidence_intervals=bootstrap_ci,
        regime_allocations=regime_allocations,
        correlation_matrix=correlation_matrix,
        diversification_benefit=div_ratio,
        expected_portfolio_sharpe=expected_sharpe
    )
```

### Mode-Specific Weight Adjustments

**SIP Mode (Long-term wealth accumulation):**
- Calmar Ratio: +50% weight
- Max Drawdown: +30% weight
- CVaR 95%: +20% weight
- Sharpe Stability: +30% weight

**Swing Mode (Short-term trading):**
- Sortino Ratio: +50% weight
- Omega Ratio: +30% weight
- Win Rate: +20% weight
- Profit Factor: +20% weight

---

## Usage Guide

### Basic Usage

```python
from advanced_strategy_selector import (
    AdvancedStrategySelector,
    EnhancedDynamicPortfolioStylesGenerator
)

# Initialize selector
selector = AdvancedStrategySelector(
    risk_free_rate=0.0,
    bootstrap_samples=500,
    min_observations=63,
    diversification_weight=0.3
)

# Run selection
result = selector.select_strategies(
    backtest_results=backtest_results,
    market_returns=market_returns,
    mode='sip',
    n_strategies=4,
    regime_aware=True
)

# Access results
print(f"Selected: {result.selected_strategies}")
print(f"Portfolio Sharpe: {result.expected_portfolio_sharpe:.2f}")
print(f"Diversification Ratio: {result.diversification_benefit:.2f}")
```

### Integration with Existing System

```python
# Option 1: Patch existing module
from backtest_integration_patch import patch_backtest_engine
patch_backtest_engine()

# Then use normally
from backtest_engine import get_dynamic_portfolio_styles
styles = get_dynamic_portfolio_styles(...)

# Option 2: Use enhanced generator directly
from advanced_strategy_selector import EnhancedDynamicPortfolioStylesGenerator

generator = EnhancedDynamicPortfolioStylesGenerator(engine)
generator.run_comprehensive_backtest(...)
styles = generator.generate_portfolio_styles(n_strategies=4)
```

### Accessing Detailed Metrics

```python
# Get full selection details
details = generator.get_selection_details(mode='sip')

# Print confidence intervals
for strategy, (low, high) in details.confidence_intervals.items():
    print(f"{strategy}: Sharpe 95% CI [{low:.2f}, {high:.2f}]")

# Check statistical significance
for strategy, breakdown in details.meta_score_breakdown.items():
    pvalue = breakdown['sharpe_pvalue']
    sig = "**" if pvalue < 0.05 else "*" if pvalue < 0.1 else ""
    print(f"{strategy}{sig}: p={pvalue:.3f}")
```

---

## API Reference

### StrategyMetrics

Comprehensive metrics dataclass:

```python
@dataclass
class StrategyMetrics:
    name: str
    
    # Core Returns
    total_return: float
    cagr: float
    annualized_return: float
    
    # Risk Metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int
    ulcer_index: float
    
    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Tail Risk
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    tail_ratio: float
    
    # Distribution
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    # Consistency
    win_rate: float
    profit_factor: float
    gain_to_pain_ratio: float
    payoff_ratio: float
    
    # Time-Series
    hurst_exponent: float
    autocorrelation_lag1: float
    
    # Stability
    sharpe_stability: float
    return_stability: float
    
    # Statistical Confidence
    sharpe_tstat: float
    sharpe_pvalue: float
    bootstrap_sharpe_ci_lower: float
    bootstrap_sharpe_ci_upper: float
```

### SelectionResult

Result of strategy selection:

```python
@dataclass
class SelectionResult:
    selected_strategies: List[str]
    selection_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    regime_allocations: Dict[MarketRegime, List[str]]
    correlation_matrix: pd.DataFrame
    diversification_benefit: float
    expected_portfolio_sharpe: float
    meta_score_breakdown: Dict[str, Dict[str, float]]
```

### Key Methods

**AdvancedStrategySelector:**
- `select_strategies(backtest_results, market_returns, mode, n_strategies, regime_aware)` → SelectionResult

**AdvancedMetricsCalculator:**
- `calculate(daily_values, strategy_name, benchmark_returns)` → StrategyMetrics

**RegimeDetector:**
- `detect_regime(market_returns, current_index)` → Tuple[MarketRegime, Dict]
- `get_regime_history(market_returns, step)` → List[Tuple]

**MultiCriteriaOptimizer:**
- `compute_scores(strategy_metrics, mode, custom_weights)` → Dict[str, float]
- `identify_pareto_frontier(strategy_metrics, objectives)` → List[str]

**DiversificationAnalyzer:**
- `compute_correlation_matrix(strategy_returns)` → pd.DataFrame
- `hierarchical_clustering(correlation_matrix, n_clusters)` → Dict[int, List[str]]
- `maximum_diversification_selection(strategy_metrics, n_select)` → List[str]
- `compute_risk_parity_weights(strategy_returns)` → Dict[str, float]
- `compute_diversification_ratio(strategy_returns, weights)` → float

---

## Conclusion

This advanced strategy selection framework transforms strategy selection from a naive ranking exercise into a rigorous, multi-dimensional optimization problem. By incorporating regime awareness, diversification constraints, statistical validation, and comprehensive risk analysis, it provides a robust foundation for institutional-grade portfolio construction.

The framework is designed to be:
1. **Mathematically Sound** - Based on established financial theory
2. **Statistically Rigorous** - With proper uncertainty quantification
3. **Practically Useful** - Drop-in replacement for existing code
4. **Extensible** - Easy to add new criteria or methods

---

*Author: Hemrek Capital*  
*Version: 3.0.0*  
*Last Updated: January 2026*
