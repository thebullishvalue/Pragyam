# Changelog

All notable changes to PRAGYAM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.2.0] - 2026-03-16

### Added
- Strategy registry with auto-discovery (`STRATEGY_REGISTRY`, `discover_strategies()` in strategies.py)
- `style.css` — extracted ~360 lines of inline CSS into standalone Hemrek Capital Design System file
- `pyproject.toml` with Ruff linter/formatter and mypy configuration
- `__all__` exports to all modules (strategies.py, backtest_engine.py, backdata.py; fixed stale entry in charts.py)
- `.gitignore` for Python, IDE, OS, and application artifacts

### Changed
- Named loggers across all modules (replaced bare `logging.info/error/warning`)
- Removed `logging.basicConfig` calls that overrode root logger config
- Migrated deprecated Streamlit `use_container_width` → `width='stretch'` across all modules
- Replaced redundant `elif not is_buy_day` with `else` in trigger backtest
- Fixed Sortino ratio formula in `backtest_engine.py` to use proper RMS of downside deviations
- Fixed week numbering collision across year boundaries in SIP backtest
- Widened return clipping from ±50% to ±100% in `PerformanceMetrics`
- Fixed rolling downside calculation in charts.py (was modifying data in-place)
- Removed magic-number metric fallbacks in `strategy_selection.py` (Sortino `*1.5`, Calmar `*10`)
- Eliminated double `calculate_advanced_metrics()` call in `_calculate_performance_on_window`
- Vectorized row-wise `.apply()` calls in CL2Strategy and CL3Strategy with `np.select`/`np.where`
- Replaced 97-line manual strategy instantiation block with `discover_strategies()` auto-registry
- Replaced 25-line manual strategy import block with single `from strategies import discover_strategies`

### Removed
- Dead code: `fix_csv_export` in strategies.py, `get_axis_config` in charts.py
- Unused imports: `time`, `ABC`/`abstractmethod`, `scipy.stats`, `StandardScaler`, `io` (from strategies.py)
- Stale `pragati.py` comment in strategies.py
- `matplotlib` and `openpyxl` from requirements.txt (not imported anywhere)
- `time.sleep(0.5)` UI delay in app.py
- Inline CSS block from app.py (~360 lines, moved to style.css)

### Fixed
- Bare `except:` clause → `except Exception:` in app.py conviction analysis
- `from datetime import timezone` moved from function body to top-level imports
- References to non-existent `pragati.py` and `advanced_strategy_selector.py` in README
- Version mismatch: README badge, requirements.txt header, and CHANGELOG now all say v3.2.0
- Stale `get_axis_config` entry in charts.py `__all__` (function was removed)

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
