# PRAGYAM (प्रज्ञम्)

<div align="center">

![Version](https://img.shields.io/badge/version-3.2.0-gold)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-Proprietary-red)
![Status](https://img.shields.io/badge/status-Production-green)

**Institutional-Grade Portfolio Intelligence System**

*Walk-forward portfolio curation with regime-aware strategy allocation*

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Architecture](#architecture)

---

<img src="https://img.shields.io/badge/Hemrek_Capital-FFC300?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iIzBGMEYwRiIgZD0iTTEyIDJMMiA3bDEwIDUgMTAtNS0xMC01ek0yIDE3bDEwIDUgMTAtNS0xMC01LTEwIDV6Ii8+PC9zdmc+" alt="Hemrek Capital"/>

</div>

---

## Overview

**Pragyam** (Sanskrit: प्रज्ञम् - "Wisdom/Intelligence") is a hedge fund-grade portfolio intelligence platform designed for systematic equity investing in Indian markets. It combines 80+ quantitative strategies with advanced regime detection and dynamic allocation to deliver institutional-quality portfolio construction.

### Key Differentiators

| Feature | Description |
|---------|-------------|
| **Multi-Strategy Engine** | 80+ unique alpha-generating strategies spanning momentum, mean-reversion, volatility, and factor-based approaches |
| **Regime-Aware Allocation** | Real-time market regime detection adjusts strategy weights based on momentum, trend, breadth, and volatility conditions |
| **Dynamic Selection** | TOPSIS multi-criteria optimization with Bayesian shrinkage selects optimal strategy combinations |
| **Hedge Fund Analytics** | Institutional metrics including Sharpe, Sortino, Calmar, Omega, CVaR, and bootstrap confidence intervals |
| **Tier-Based Construction** | Position sizing via conviction tiers with risk parity optimization |

---

## Features

### 📊 Portfolio Intelligence
- **Walk-Forward Backtesting**: Out-of-sample validation with realistic transaction modeling
- **Strategy Attribution**: Decompose returns by strategy, tier, and time period
- **Correlation Analysis**: Inter-strategy correlation monitoring for diversification
- **Weight Evolution**: Track how strategy allocations change through market regimes

### 🎯 Strategy Universe
- **Momentum Strategies**: RSI, MACD, Rate of Change, Acceleration
- **Mean-Reversion**: Bollinger Bands, Z-Score, Kalman Filter
- **Volatility**: ATR-based, Regime-switching, Breakout detection
- **Factor-Based**: Value-momentum blends, Quality scores, Size tilts
- **Advanced**: HMM-based, Wavelet denoising, Copula blending

### 📈 Risk Analytics
- **Core Metrics**: CAGR, Volatility, Maximum Drawdown, Win Rate
- **Risk-Adjusted**: Sharpe, Sortino, Calmar, Omega, Information Ratio
- **Tail Risk**: VaR (95/99), CVaR, Expected Shortfall, Tail Ratio
- **Statistical**: Bootstrap CI, Lo(2002) Sharpe SE, Jarque-Bera normality

### 🔄 Regime Detection
- **Momentum Regime**: RSI breadth, momentum persistence scoring
- **Trend Regime**: 200-DMA positioning, trend quality metrics
- **Volatility Regime**: Bollinger Band Width, ATR percentile ranking
- **Breadth Regime**: Advance-decline analysis, sector rotation signals

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/hemrek/pragyam.git
cd pragyam

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
yfinance>=0.2.28
scipy>=1.11.0
scikit-learn>=1.3.0
```

---

## Quick Start

### Running Locally

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Streamlit Cloud Deployment

1. Push to GitHub repository
2. Connect to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy with `app.py` as the main file

### Basic Usage

1. **Select Analysis Date**: Choose the portfolio construction date
2. **Choose Mode**: SIP (accumulation) or Swing (trading)
3. **Set Lookback**: Historical period for strategy evaluation
4. **Run Analysis**: Generate curated portfolio with full analytics

---

## Architecture

```
pragyam/
├── app.py                      # Main Streamlit application & portfolio curation
├── strategies.py               # 80+ strategy implementations + auto-discovery registry
├── charts.py                   # Unified Plotly visualization components
├── strategy_selection.py       # Trigger-based strategy evaluation (REL_BREADTH)
├── backtest_engine.py          # Walk-forward backtesting framework
├── backdata.py                 # Data fetching & indicator computation
├── style.css                   # Hemrek Capital Design System (dark theme)
├── symbols.txt                 # Universe of tradeable symbols (NSE tickers)
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Ruff, mypy, and package configuration
├── CHANGELOG.md                # Version history
└── docs/
    ├── STRATEGY_GUIDE.md       # Strategy documentation
    ├── MATHEMATICAL_FRAMEWORK.md  # Quantitative methods
    └── PROCESS_ARCHITECTURE.md # System architecture details
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRAGYAM DATA FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Yahoo   │───▶│ Indicator│───▶│  Regime  │───▶│ Strategy │  │
│  │ Finance  │    │  Engine  │    │ Detector │    │ Universe │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                       │         │
│                                                       ▼         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │Dashboard │◀───│ Backtest │◀───│ Portfolio│◀───│  TOPSIS  │  │
│  │  & UI    │    │  Engine  │    │  Builder │    │ Selector │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Symbol Universe

Edit `symbols.txt` to customize the stock universe:

```
RELIANCE.NS
TCS.NS
HDFCBANK.NS
INFY.NS
...
```

### Trigger Configuration

Adjust trigger thresholds for buy/sell signals in `app.py` via `TRIGGER_CONFIG`:

```python
# strategy_selection.py exposes these defaults
SIP_TRIGGER = 0.42          # Accumulate when REL_BREADTH < threshold
SWING_BUY_TRIGGER = 0.42    # Enter when breadth drops below
SWING_SELL_TRIGGER = 0.50   # Exit when breadth rises above
```

---

## Performance Metrics

### Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Sharpe Ratio** | (R - Rf) / σ | Risk-adjusted return; >1 good, >2 excellent |
| **Sortino Ratio** | (R - Rf) / σd | Downside-adjusted; ignores upside volatility |
| **Calmar Ratio** | CAGR / \|MaxDD\| | Drawdown efficiency; >1 indicates resilience |
| **Omega Ratio** | Σ(gains) / Σ(losses) | Full distribution analysis; >1 positive expectancy |
| **Tail Ratio** | P95 / \|P5\| | Skewness measure; >1 indicates positive skew |

### Statistical Tests

- **Lo (2002) Sharpe SE**: Autocorrelation-adjusted standard error
- **Bootstrap CI**: 1000-sample confidence intervals
- **Jarque-Bera**: Normality test for return distribution

---

## API Reference

### Core Components

```python
# Market regime detection
from app import MarketRegimeDetectorV2

detector = MarketRegimeDetectorV2()
regime, mix, confidence, details = detector.detect_regime(data)

# Strategy evaluation with trigger-based selection
from strategy_selection import StrategySelectionEngine

engine = StrategySelectionEngine(strategies, data, breadth_data)
results = engine.evaluate()

# Generate indicator snapshots
from backdata import generate_historical_data

snapshots = generate_historical_data(
    symbols_to_process=["RELIANCE.NS", "TCS.NS"],
    start_date=start,
    end_date=end,
)
```

---

## Support

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Yahoo Finance rate limiting | Add delays between requests or use cached data |
| Memory errors on large universes | Reduce symbol count or increase system memory |
| Slow backtest execution | Enable caching with `@st.cache_data` decorators |

### Contact

- **Technical Support**: tech@hemrekcapital.com
- **Documentation**: docs.hemrekcapital.com/pragyam
- **Issues**: GitHub Issues (for licensed users)

---

## License

**Proprietary Software** - © 2024-2026 Hemrek Capital

This software is licensed exclusively to authorized users. Redistribution, modification, or commercial use without explicit written permission is prohibited.

---

## Changelog

### v3.2.0 (March 2026)
- Code quality audit and refactoring
- Fixed deprecated Streamlit APIs
- Removed dead code and unused imports
- Named loggers across all modules

### v3.1.0 (February 2026)
- Strategy selection framework with REL_BREADTH triggers
- Unified chart styling (Hemrek design system)
- Trigger-based backtesting (SIP & Swing modes)

### v3.0.0 (January 2026)
- Advanced strategy selector with TOPSIS optimization
- Bayesian shrinkage estimation
- Risk parity portfolio construction
- HMM regime detection integration

### v2.0.0 (December 2025)
- 80+ strategy implementations
- Walk-forward backtesting engine
- Regime-aware allocation system

---

<div align="center">

**Built with ❤️ by Hemrek Capital**

*"Wisdom in Every Trade"*

</div>
