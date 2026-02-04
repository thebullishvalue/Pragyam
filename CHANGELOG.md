# Changelog

All notable changes to PRAGYAM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.1.0] - 2026-02-04

### Added
- **Strategy Selection Framework** (`strategy_selection.py`)
  - Fetches REL_BREADTH data from Google Sheets (400 rows lookback)
  - SIP Mode: Accumulates portfolio on every trigger (REL_BREADTH < 0.42)
  - Swing Mode: Buy-sell cycles (Buy < 0.42, Sell >= 0.50)
  - MasterPortfolio class tracks cumulative holdings across SIP entries
  - Dispersion-weighted ranking (no fixed formula weights)
- Chart annotations restored ("Growth of ₹1 Investment", "Underwater Curve")

### Changed
- Performance tab redesigned with clean metric layout using st.metric()
- Strategy Deep Dive tab simplified with minimal headers
- Selection scoring uses rank-based adaptive weights
- Equity chart y-axis starts from sensible minimum (not zero)

### Removed
- Fixed threshold selection formulas (0.30×Sharpe + 0.25×Sortino...)
- Verbose section headers and info-boxes

### Fixed
- Plotly `titlefont` deprecation error
- `use_container_width` deprecation warnings
- Equity curve appearing flat due to y-axis starting at zero

---

## [3.0.0] - 2026-01-30

### Added
- Advanced Strategy Selector with TOPSIS multi-criteria optimization
- Bayesian shrinkage estimation for small sample periods
- Risk parity portfolio construction with SLSQP optimization
- Hidden Markov Model (HMM) regime detection integration
- Bootstrap confidence intervals for Sharpe ratio
- Lo (2002) standard error adjustment for serial correlation
- Pareto frontier identification for strategy selection
- Maximum diversification selection algorithm
- Strategy clustering via hierarchical clustering

### Changed
- Complete rewrite of strategy selection logic
- Enhanced backtest engine with walk-forward validation
- Improved regime detection with multi-factor approach

---

## [2.5.0] - 2025-12-15

### Added
- 80+ quantitative strategies implementation
- Tier-based position sizing system
- Regime-aware allocation framework
- Strategy correlation monitoring

### Changed
- Migrated to Streamlit 1.28+ API
- Improved data caching mechanisms
- Enhanced error handling

---

## [2.0.0] - 2025-10-01

### Added
- Walk-forward backtesting engine
- Multi-strategy portfolio construction
- Performance attribution analytics
- Interactive visualizations with Plotly

### Changed
- Complete architecture redesign
- Modular strategy framework

---

## [1.0.0] - 2025-07-01

### Added
- Initial release
- Basic momentum strategies
- Simple backtesting framework
- Streamlit dashboard

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Features to be removed in future versions
- **Removed**: Features removed in this version
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes
