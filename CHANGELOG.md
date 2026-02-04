# Changelog

All notable changes to PRAGYAM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.1.0] - 2026-02-04

### Added
- Unified chart styling module (`charts.py`) following Hemrek Capital design system
- Premium metric cards with semantic color coding (success/warning/danger)
- Factor interpretation guide in Strategy Deep Dive
- Cross-strategy conviction analysis with signal consensus metrics
- Diversification ratio display in correlation analysis

### Changed
- Performance tab completely redesigned with institutional-grade aesthetics
- Strategy Deep Dive tab enhanced with consistent styling
- Replaced deprecated `use_container_width` with `width="stretch"`
- Updated Plotly axis configuration for compatibility

### Fixed
- Plotly `titlefont` deprecation error (replaced with inline styling)
- Chart rendering issues in Streamlit Cloud environment
- Axis configuration conflicts in subplot charts

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
