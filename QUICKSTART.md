# Quick Start Guide: Pragyam v3.0

## 🚀 5-Minute Setup

### 1. Install Dependencies

```bash
cd Pragyam-main
pip install -r requirements.txt
```

### 2. Add Your Stock Universe

Edit `symbols.txt` with your stock symbols (one per line):
```
RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
...
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Generate Portfolio

1. Select investment style: **SIP Investment** or **Swing Trading**
2. Choose market regime expectation: **Bull**, **Bear**, or **Chop/Consolidate**
3. Click **Generate Portfolio**
4. View results with diversification metrics

---

## 📊 Understanding the Output

### Selection Scores

Each selected strategy shows:
- **MCO Score**: Multi-Criteria Optimization score (higher = better overall)
- **CI**: 95% confidence interval for Sharpe ratio
- **Div.Ratio**: Portfolio diversification ratio (>1 = diversification benefit)

### Regime Allocations

Different strategies are optimal for different market regimes:
- **Bull Market**: Growth-focused strategies
- **Bear Market**: Defensive, drawdown-protected strategies
- **Sideways**: Mean-reversion optimized strategies

---

## 🔧 Configuration Options

### In `advanced_strategy_selector.py`:

```python
selector = AdvancedStrategySelector(
    risk_free_rate=0.0,      # Annual risk-free rate
    bootstrap_samples=500,   # Samples for confidence intervals
    min_observations=63,     # Minimum backtest days required
    diversification_weight=0.3  # Weight for diversification benefit
)
```

### Mode-Specific Weights

SIP Mode prioritizes:
- Calmar Ratio (drawdown recovery)
- CVaR (tail risk)
- Sharpe Stability (consistency)

Swing Mode prioritizes:
- Sortino Ratio (downside risk)
- Omega Ratio (gain/loss distribution)
- Win Rate (trade success)

---

## 📈 Key Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| Sharpe Ratio | Risk-adjusted return | > 1.0 |
| Sortino Ratio | Downside risk-adjusted | > 1.5 |
| Calmar Ratio | Return / Max Drawdown | > 1.0 |
| Omega Ratio | Gains / Losses | > 1.5 |
| Max Drawdown | Worst peak-to-trough | > -20% |
| CVaR 95% | Expected loss in worst 5% | > -3% |
| Win Rate | % of profitable days | > 52% |
| Hurst Exponent | 0.5=random, >0.5=trending | varies |

---

## 🛠️ Troubleshooting

### "Advanced selection not available"

Ensure `advanced_strategy_selector.py` is in the same directory as `app.py`.

### "Insufficient data"

The system needs at least 20 trading days of historical data. Extend your date range.

### "No strategies passed filters"

Lower the `min_observations` parameter or ensure your strategies generate valid portfolios.

---

## 📚 Further Reading

- `ADVANCED_SELECTION_DOCUMENTATION.md` - Full mathematical framework
- `advanced_strategy_selector.py` - Source code with detailed comments
- `strategies.py` - All 80+ strategy implementations

---

<p align="center">
  <strong>Happy Trading! 📈</strong>
</p>
