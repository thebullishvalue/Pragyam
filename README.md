# PRAGYAM (प्रज्ञम) - Portfolio Intelligence

**A Hemrek Capital Product**

Walk-forward portfolio curation with regime-aware strategy allocation. Multi-strategy backtesting and capital optimization engine for quantitative portfolio construction.

## Features

- **Walk-Forward Engine**: Historical backtesting with rolling window validation
- **Regime Detection**: Automatic Bull/Bear/Chop market classification
- **Multi-Strategy Allocation**: 7 quantitative strategies with dynamic weighting
- **Portfolio Curation**: Position sizing with risk-adjusted optimization
- **Live Data Integration**: Real-time market data fetching

## Strategies

- GameTheoreticStrategy
- MomentumAccelerator
- VolatilitySurfer
- DivineMomentumOracle
- AdaptiveVolBreakout
- NebulaMomentumStorm
- CelestialAlphaForge

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

- Python 3.10+
- streamlit
- pandas
- numpy
- plotly
- scipy
- scikit-learn
- yfinance

## Files

- `app.py` - Main Streamlit application
- `strategies.py` - Strategy implementations
- `backdata.py` - Live data generation
- `backtest.py` - Backtesting engine
- `symbols.txt` - Universe of tradeable symbols

## Version

v1.1.0 - Hemrek Capital Design System
