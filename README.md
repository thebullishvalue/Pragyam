# PRAGYAM (प्रज्ञम) v3.0 - Advanced Portfolio Intelligence

<p align="center">
  <strong>Walk-forward portfolio curation with advanced multi-criteria strategy selection</strong>
</p>

<p align="center">
  <em>A Hemrek Capital Product</em>
</p>

---

## 🚀 What's New in v3.0

### Advanced Multi-Criteria Strategy Selection

The v3.0 release introduces a mathematically rigorous strategy selection framework that goes far beyond simple single-metric ranking:

| Feature | v2.0 (Legacy) | v3.0 (Advanced) |
|---------|---------------|-----------------|
| Selection Criteria | Single metric (Calmar/Sortino) | 12+ criteria with TOPSIS optimization |
| Regime Awareness | None | HMM-based market regime detection |
| Diversification | None | Maximum diversification + risk parity |
| Statistical Rigor | None | Bootstrap CI + significance testing |
| Tail Risk | Ignored | CVaR, Expected Shortfall analysis |
| Noise Handling | None | Bayesian shrinkage estimation |

### Key Components

1. **Multi-Criteria Optimization (TOPSIS)** - Combines 12+ metrics using Technique for Order Preference by Similarity to Ideal Solution

2. **Market Regime Detection** - Identifies Bull, Bear, High/Low Volatility, Trending, Mean-Reverting, Crisis, and Recovery regimes

3. **Maximum Diversification Selection** - Selects strategies that maximize portfolio diversification benefit

4. **Risk Parity Allocation** - Equal risk contribution portfolio construction

5. **Bootstrap Confidence Intervals** - 95% CI for Sharpe ratio estimates

6. **Bayesian Shrinkage** - Reduces noise in short backtest periods

---

## 📁 Repository Structure

```
Pragyam-main/
├── app.py                          # Main Streamlit application (v3.0)
├── backtest_engine.py              # Unified backtest engine with advanced selection
├── advanced_strategy_selector.py   # Advanced multi-criteria selection module
├── backtest_integration_patch.py   # Integration utilities
├── strategies.py                   # 80+ trading strategies
├── backdata.py                     # Historical data generation
├── pragati.py                      # Core portfolio logic
├── symbols.txt                     # Stock universe
├── requirements.txt                # Python dependencies
├── ADVANCED_SELECTION_DOCUMENTATION.md  # Mathematical documentation
└── README.md                       # This file
```

---

## 🔧 Installation

```bash
# Clone or download the repository
cd Pragyam-main

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 📊 How It Works

### Strategy Selection Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   Backtest All Strategies                    │
│               (80+ strategies × historical data)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Calculate Comprehensive Metrics                 │
│   • Risk-Adjusted (Sharpe, Sortino, Calmar, Omega)          │
│   • Tail Risk (VaR, CVaR, Expected Shortfall)               │
│   • Distribution (Skewness, Kurtosis, Hurst)                │
│   • Consistency (Win Rate, Profit Factor)                   │
│   • Stability (Rolling Sharpe Std, Return CV)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Apply Bayesian Shrinkage                      │
│            (Reduce noise in short samples)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Detect Market Regime                          │
│     (Bull/Bear/HighVol/LowVol/Trending/MeanReverting)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Multi-Criteria Optimization (TOPSIS)              │
│         Mode-specific weights for SIP vs Swing              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Apply Diversification Constraints                 │
│      (Correlation-aware selection, Risk Parity)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Final Selection (Top 4 Strategies)              │
│         with Confidence Intervals & Regime Allocations      │
└─────────────────────────────────────────────────────────────┘
```

### Mode-Specific Selection

**SIP Mode** (Long-term wealth accumulation):
- Prioritizes Calmar Ratio (+50% weight)
- Higher weight on drawdown protection
- Focus on CVaR and stability metrics

**Swing Mode** (Short-term trading):
- Prioritizes Sortino Ratio (+50% weight)
- Higher weight on Omega Ratio
- Focus on win rate and profit factor

---

## 📈 Usage

### Basic Usage

1. Run the Streamlit app: `streamlit run app.py`
2. Select investment style (SIP or Swing Trading)
3. Choose market regime expectation
4. Generate portfolio

### Programmatic Usage

```python
from advanced_strategy_selector import AdvancedStrategySelector

# Initialize selector
selector = AdvancedStrategySelector(
    risk_free_rate=0.0,
    bootstrap_samples=500,
    diversification_weight=0.3
)

# Run selection
result = selector.select_strategies(
    backtest_results=results,
    market_returns=market_returns,
    mode='sip',
    n_strategies=4,
    regime_aware=True
)

# Access results
print(f"Selected: {result.selected_strategies}")
print(f"Diversification Ratio: {result.diversification_benefit:.2f}")
print(f"Expected Portfolio Sharpe: {result.expected_portfolio_sharpe:.2f}")

# View confidence intervals
for strategy, (low, high) in result.confidence_intervals.items():
    print(f"{strategy}: Sharpe 95% CI [{low:.2f}, {high:.2f}]")
```

---

## 🔬 Mathematical Details

### TOPSIS (Multi-Criteria Optimization)

The TOPSIS method identifies strategies closest to the ideal solution:

1. **Normalize** criteria matrix (min-max scaling)
2. **Weight** by mode-specific importance
3. Calculate distance to **ideal** (best values) and **anti-ideal** (worst values)
4. **Score** = D⁻ / (D⁺ + D⁻)

### Hurst Exponent

Characterizes time series behavior:
- H > 0.5: Trending (persistent)
- H = 0.5: Random walk
- H < 0.5: Mean-reverting

Calculated via R/S (Rescaled Range) analysis.

### Bayesian Shrinkage

Reduces noise in short samples:

```
θ_shrunk = λ × prior + (1-λ) × observed
where λ = n_prior / (n_prior + n_effective)
```

### Risk Parity

Equal risk contribution optimization:

```
minimize Σ(RC_i - σ_p/n)²
subject to: Σw_i = 1, w_i ≥ 0
```

---

## 📋 Strategies Included

The system includes 80+ trading strategies across categories:

- **Momentum**: MomentumMasters, VelocityVortex, AlphaSurge, etc.
- **Volatility**: VolatilitySurfer, AdaptiveVolBreakout, etc.
- **Statistical**: KalmanFilterMomentum, BayesianMomentumUpdater, etc.
- **ML-Inspired**: NeuralNetworkInspired, GraphNeuralInspired, etc.
- **Regime-Based**: RegimeSwitchingStrategy, HiddenMarkovModel, etc.

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

Proprietary - Hemrek Capital

---

## 🤝 Support

For questions or issues, please contact Hemrek Capital.

---

<p align="center">
  <strong>Built with ❤️ by Hemrek Capital</strong>
</p>
