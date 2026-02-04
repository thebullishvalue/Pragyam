# PRAGYAM Mathematical Framework

## Overview

This document details the quantitative methods underlying Pragyam's portfolio intelligence system. All implementations follow institutional standards with proper statistical rigor.

---

## 1. Performance Metrics

### 1.1 Return Calculations

**Simple Return**:
$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

**Log Return** (used for aggregation):
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Compound Annual Growth Rate (CAGR)**:
$$CAGR = \left(\frac{V_{final}}{V_{initial}}\right)^{\frac{1}{n}} - 1$$

where $n$ is the number of years.

### 1.2 Risk Metrics

**Annualized Volatility**:
$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$

**Downside Deviation** (Sortino denominator):
$$\sigma_d = \sqrt{\frac{\sum_{r_t < \tau}(r_t - \tau)^2}{n}}$$

where $\tau$ is the target return (typically 0).

**Maximum Drawdown**:
$$MDD = \max_{t \in [0,T]} \left(\frac{\max_{s \in [0,t]} P_s - P_t}{\max_{s \in [0,t]} P_s}\right)$$

### 1.3 Risk-Adjusted Returns

**Sharpe Ratio**:
$$SR = \frac{E[R] - R_f}{\sigma}$$

**Sortino Ratio**:
$$Sortino = \frac{E[R] - R_f}{\sigma_d}$$

**Calmar Ratio**:
$$Calmar = \frac{CAGR}{|MDD|}$$

**Omega Ratio**:
$$\Omega(\tau) = \frac{\int_\tau^\infty [1 - F(r)]dr}{\int_{-\infty}^\tau F(r)dr}$$

where $F(r)$ is the cumulative distribution function of returns.

---

## 2. Statistical Significance

### 2.1 Lo (2002) Sharpe Ratio Standard Error

The standard error of the Sharpe ratio, accounting for serial correlation:

$$SE(SR) = \sqrt{\frac{1 + 2\sum_{k=1}^{q}\rho_k(1 - k/q)}{T}}$$

where:
- $\rho_k$ is the autocorrelation at lag $k$
- $q$ is the truncation lag (typically $\min(10, T/4)$)
- $T$ is the sample size

**T-statistic**:
$$t = \frac{SR}{SE(SR)}$$

### 2.2 Bootstrap Confidence Intervals

Block bootstrap preserves serial correlation:

```python
for b in range(B):  # B = 1000 bootstrap samples
    sample = block_bootstrap(returns, block_size)
    sharpe_b = calculate_sharpe(sample)
    bootstrap_sharpes.append(sharpe_b)

CI_lower = percentile(bootstrap_sharpes, 2.5)
CI_upper = percentile(bootstrap_sharpes, 97.5)
```

### 2.3 Jarque-Bera Normality Test

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

where:
- $S$ is skewness
- $K$ is kurtosis
- Under $H_0$: $JB \sim \chi^2(2)$

---

## 3. Tail Risk Analysis

### 3.1 Value at Risk (VaR)

**Historical VaR**:
$$VaR_\alpha = -\text{Percentile}(R, \alpha)$$

Example: $VaR_{95}$ is the 5th percentile of returns (5% worst-case loss).

### 3.2 Conditional VaR (Expected Shortfall)

$$CVaR_\alpha = E[R | R \leq VaR_\alpha]$$

$$CVaR_\alpha = \frac{1}{\alpha}\int_0^\alpha VaR_u \, du$$

### 3.3 Tail Ratio

$$Tail\ Ratio = \frac{Percentile_{95}}{|Percentile_5|}$$

- $> 1$: Positive skew (bigger wins than losses)
- $< 1$: Negative skew (bigger losses than wins)

---

## 4. Multi-Criteria Decision Making

### 4.1 TOPSIS Method

**Step 1**: Normalize decision matrix
$$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^m x_{ij}^2}}$$

**Step 2**: Apply weights
$$v_{ij} = w_j \times r_{ij}$$

**Step 3**: Determine ideal solutions
$$A^+ = \{v_1^+, v_2^+, ..., v_n^+\}$$ (best values)
$$A^- = \{v_1^-, v_2^-, ..., v_n^-\}$$ (worst values)

**Step 4**: Calculate distances
$$D_i^+ = \sqrt{\sum_{j=1}^n (v_{ij} - v_j^+)^2}$$
$$D_i^- = \sqrt{\sum_{j=1}^n (v_{ij} - v_j^-)^2}$$

**Step 5**: Calculate relative closeness
$$C_i = \frac{D_i^-}{D_i^+ + D_i^-}$$

Higher $C_i$ indicates better alternative.

### 4.2 Pareto Frontier

Strategy $A$ dominates strategy $B$ if:
- $A$ is at least as good as $B$ on all criteria
- $A$ is strictly better than $B$ on at least one criterion

The Pareto frontier contains all non-dominated strategies.

---

## 5. Portfolio Construction

### 5.1 Risk Parity Optimization

**Objective**: Equal risk contribution from each asset

$$RC_i = w_i \times \frac{\partial \sigma_p}{\partial w_i} = w_i \times \frac{(\Sigma w)_i}{\sigma_p}$$

**Optimization problem**:
$$\min_w \sum_{i=1}^n \left(RC_i - \frac{\sigma_p}{n}\right)^2$$

subject to: $\sum w_i = 1$, $w_i \geq 0$

### 5.2 Maximum Diversification

**Diversification Ratio**:
$$DR = \frac{w'\sigma}{\sqrt{w'\Sigma w}} = \frac{\text{Weighted Avg Vol}}{\text{Portfolio Vol}}$$

Maximize $DR$ subject to constraints.

### 5.3 Bayesian Shrinkage

**James-Stein Shrinkage for Sharpe Ratio**:
$$\hat{SR}_{shrunk} = \lambda \cdot SR_{prior} + (1-\lambda) \cdot SR_{observed}$$

where:
$$\lambda = \frac{n_{prior}}{n_{prior} + n_{effective}}$$

- $n_{prior}$: Prior strength (typically 50)
- $n_{effective}$: Effective sample size (adjusted for autocorrelation)

---

## 6. Regime Detection

### 6.1 Momentum Regime

**RSI Breadth**:
$$RSI_{breadth} = \frac{\#\{stocks: RSI > 50\}}{N}$$

**Momentum Score**:
$$M = \alpha \cdot RSI_{breadth} + \beta \cdot \frac{\bar{R}_{20}}{\sigma_{20}}$$

### 6.2 Trend Regime

**200-DMA Positioning**:
$$Trend_{quality} = \frac{\#\{stocks: P > MA_{200}\}}{N}$$

**Trend Strength**:
$$TS = \frac{MA_{50} - MA_{200}}{MA_{200}}$$

### 6.3 Volatility Regime

**Bollinger Band Width Percentile**:
$$BBW = \frac{Upper - Lower}{Middle}$$

**Regime Classification**:
- $BBW < P_{20}$: Low volatility (compression)
- $BBW > P_{80}$: High volatility (expansion)
- Otherwise: Normal volatility

### 6.4 Hidden Markov Model

**State Transition**:
$$P(S_t = j | S_{t-1} = i) = a_{ij}$$

**Observation Probability**:
$$P(O_t | S_t = j) = \mathcal{N}(\mu_j, \sigma_j^2)$$

**Viterbi Algorithm** for most likely state sequence.

---

## 7. Time Series Analysis

### 7.1 Hurst Exponent (R/S Analysis)

**Rescaled Range**:
$$R/S = \frac{R(n)}{S(n)}$$

where:
- $R(n) = \max(Y_k) - \min(Y_k)$ (range of cumulative deviations)
- $S(n)$ = standard deviation

**Estimation**:
$$\log(R/S) = H \cdot \log(n) + c$$

**Interpretation**:
- $H > 0.5$: Trending (persistent)
- $H = 0.5$: Random walk
- $H < 0.5$: Mean-reverting

### 7.2 Autocorrelation Structure

**Ljung-Box Test**:
$$Q = n(n+2)\sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k}$$

Under $H_0$ (no autocorrelation): $Q \sim \chi^2(h)$

---

## 8. Implementation Notes

### Numerical Stability

```python
# Avoid division by zero
denominator = max(denominator, 1e-10)

# Log-sum-exp trick for numerical stability
log_sum = max_val + np.log(np.sum(np.exp(values - max_val)))
```

### Optimization Convergence

```python
result = scipy.optimize.minimize(
    objective,
    x0=initial_guess,
    method='SLSQP',
    constraints=constraints,
    bounds=bounds,
    options={'maxiter': 1000, 'ftol': 1e-9}
)
```

### Rolling Window Calculations

```python
# Efficient rolling calculations
rolling_mean = df['returns'].rolling(window=window).mean()
rolling_std = df['returns'].rolling(window=window).std()
rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
```

---

## References

1. Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*.
2. Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management*.
3. Hwang, C., & Yoon, K. (1981). "Multiple Attribute Decision Making: Methods and Applications."
4. Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series."
5. Keating, C., & Shadwick, W. F. (2002). "A Universal Performance Measure." *Journal of Performance Measurement*.

---

*© 2024-2026 Hemrek Capital - Proprietary & Confidential*
