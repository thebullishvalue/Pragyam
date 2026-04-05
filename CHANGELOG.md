# Changelog

All notable changes to PRAGYAM (प्रज्ञम) — Portfolio Intelligence will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [7.0.5] - 2026-04-05

### 🧹 Removed

**Dead Code & Stale Files**
- Removed `docs/PROCESS_ARCHITECTURE.md` — described obsolete v6.0.0 4-phase architecture
- Removed `docs/STRATEGY_GUIDE.md` — described TOPSIS optimization removed in v7.0.0
- Removed 11 unused chart functions from `charts.py` (~1,000 lines):
  `create_equity_drawdown_chart`, `create_rolling_metrics_chart`, `create_correlation_heatmap`,
  `create_tier_sharpe_heatmap`, `create_risk_return_scatter`, `create_factor_radar`,
  `create_weight_evolution_chart`, `create_signal_heatmap`, `create_bar_chart`,
  `create_regime_factor_bars`, `create_portfolio_breakdown_chart`
- Removed dead functions from `circuit_breaker.py`:
  `google_sheets_circuit`, `get_yfinance_circuit()`, `protect_with_circuit()`
- Removed unused `import plotly.graph_objects as go` from `app.py`

### ✨ Added

**Enhanced Terminal Logging**
- Added main run header with analysis date, investment style, capital, positions, Run ID, and timestamp
- Added detailed checkpoints for every critical step in Phase 1 and Phase 2
- Per-run unique Run ID generated on each "Run Analysis" click (previously session-scoped)
- Signal Distribution card counts now based on raw conviction scores instead of fragile string parsing

**Position Guide Tab**
- Moved Position Guide section from Portfolio tab into dedicated tab
- Added signal distribution summary with conviction breakdown metrics

**Market Regime Auto-Detection**
- Sidebar regime display now updates automatically when analysis date changes (no "Run Analysis" required)

### 🔧 Changed

- Simplified section headers from `P1: PHASE 1: DATA FETCHING` → `Phase 1: Data Fetching` (eliminated redundancy)
- Removed redundant "Regime Analysis" text section from Regime tab
- Updated `metrics.py` counters to populate correctly (symbols, strategies, portfolios)
- Fixed `conviction_curation` phase timing (previously showed 0.00s)
- Removed redundant `EXECUTION METRICS` header from `metrics.print_summary()`
- Replaced deprecated `use_container_width=True` with `width='stretch'` (Streamlit compatibility)

### 🐛 Fixed

- Investment style selector default index always evaluated to 1 (SIP) — now correctly defaults to 0 (Swing Trading)
- Signal Distribution card counts misclassified positions due to emoji-prefixed signal strings

---

## [7.0.4] - 2026-04-02

### ✨ Added

**Style-Aware Conviction Dispersion**

Different dispersion profiles for SIP vs Swing Trading investment styles:

| Style | Boost | Penalty | Top Pick Advantage | Use Case |
|-------|-------|---------|-------------------|----------|
| **SIP Investment** | +125% (×2.25) | -50% (×0.50) | ~350% more weight | Long-term wealth building |
| **Swing Trading** | +225% (×3.25) | -75% (×0.25) | ~1200% more weight | Active trading, alpha capture |

**Formula:**
```python
# Auto-selected based on investment_style parameter

# SIP Mode (conservative concentration)
if score > median:
    adjusted = score × 2.25  # +125% boost
else:
    adjusted = score × 0.50  # -50% penalty

# Swing Mode (aggressive, 2σ more concentration)
if score > median:
    adjusted = score × 3.25  # +225% boost
else:
    adjusted = score × 0.25  # -75% penalty

weight = (adjusted / Σ all_adjusted) × 100
```

**Effect:**
- SIP: Strong concentration with moderate risk (~350% tilt to top picks)
- Swing: Maximum concentration in best ideas (~1200% tilt) for alpha capture
- Both maintain 30-position diversification with 1-10% bounds

### 🔧 Changed

- `portfolio.py::compute_conviction_based_weights()` now accepts `investment_style` parameter
- Dispersion auto-selects based on style when `dispersion_params=None`

---

## [7.0.3] - 2026-04-02

### 🔧 Changed

**Aggressive Conviction Dispersion**

Maximum concentration in high-conviction picks:
- Symbols with conviction **above median**: **+75% boost** (was +40%)
- Symbols with conviction **at/below median**: **-50% penalty** (was -30%)

**Formula:**
```python
median = median(all_conviction_scores)

if score > median:
    adjusted = score × 1.75  # +75% boost
else:
    adjusted = score × 0.50  # -50% penalty

weight = (adjusted / Σ all_adjusted) × 100
```

**Effect:**
- High-conviction symbols receive ~250% more weight vs linear allocation
- Aggressive concentration in best ideas while maintaining 30-position diversification
- Configurable via `dispersion_params` tuple in `portfolio.py`

---

## [7.0.2] - 2026-04-02

### 🔧 Changed

**Increased Conviction Dispersion**

Stronger concentration in high-conviction picks:
- Symbols with conviction **above median**: **+40% boost** (was +15%)
- Symbols with conviction **at/below median**: **-30% penalty** (was -10%)

**Formula:**
```python
median = median(all_conviction_scores)

if score > median:
    adjusted = score × 1.40  # +40% boost
else:
    adjusted = score × 0.70  # -30% penalty

weight = (adjusted / Σ all_adjusted) × 100
```

**Effect:**
- High-conviction symbols receive ~100% more weight vs linear allocation
- Strong concentration in best ideas while maintaining 30-position diversification
- Configurable via `dispersion_params` tuple in `portfolio.py`

---

## [7.0.1] - 2026-04-02

### ✨ Added

**Conviction Dispersion Weighting**

To concentrate capital in high-conviction picks:
- Symbols with conviction **above median**: **+15% boost**
- Symbols with conviction **at/below median**: **-10% penalty**

**Formula:**
```python
median = median(all_conviction_scores)

if score > median:
    adjusted = score × 1.15  # Boost
else:
    adjusted = score × 0.90  # Penalty

weight = (adjusted / Σ all_adjusted) × 100
```

**Effect:**
- High-conviction symbols receive ~28% more weight vs linear allocation
- Maintains diversification (still 30 positions)
- Preserves bounds (1% min, 10% max)

### 🔧 Changed

- `portfolio.py::compute_conviction_based_weights()` now has `apply_dispersion` parameter (default: `True`)
- Version bumped to 7.0.1 across all files

---

## [7.0.0] - 2026-04-02

### 🎯 Major Changes

**Complete system refactoring to implement pure conviction-based portfolio curation.**

### ✨ Added

- Pure conviction-based portfolio weighting formula: `weight_i = (conviction_score_i / Σ all_conviction_scores) × 100`
- No conviction threshold filter — all symbols eligible for portfolio inclusion
- Top 30 positions selected by conviction score (0-100 range)
- Position bounds: 1% minimum, 10% maximum per position
- 2-phase architecture (Data Fetching + Conviction-Based Curation)

### 🚀 Performance Improvements

- **6-10x faster execution**: 20-40 seconds vs 2-5 minutes (v6.0.0)
- **5x larger candidate pool**: ~200-400 symbols vs ~40-80 symbols
- **Maximum diversification**: All 80+ strategies run (no filtering)
- **30% code reduction**: ~3,500 lines vs ~5,000+ lines

### 🔧 Technical Changes

- Removed walk-forward evaluation (Phase 3)
- Removed strategy selection meta-weighting (Phase 2 old)
- Removed tier-based allocation logic
- Removed SLSQP portfolio optimization
- Unified conviction scoring: single source of truth in `regime.py::compute_conviction_signals()`
- Simplified `walk_forward.py`: 1,308 lines → 95 lines (-93%)
- Simplified `app.py`: 1,608 lines → 815 lines (-49%)

### ❌ Removed

- Walk-forward performance tracking
- Strategy selection competition
- Meta-weighting (6-method competition)
- Tier-based portfolio allocation
- Conviction threshold filter (>50)
- `strategy_selector.py` module
- `backup_refactor_20260328/` directory

### 📝 Documentation

- Updated `README.md` with v7.0.0 architecture
- Added `REFACTORING_SUMMARY.md` with migration guide
- Added `CHANGELOG.md` (this file)

### 🐛 Bug Fixes

- Fixed duplicate conviction scoring logic (now single source of truth)
- Removed dead code and unused imports
- Cleaned up session state variables

---

## [6.0.0] - Previous Version (Walk-Forward with Meta-Weighting)

### Features

- 4-phase architecture (Data Fetching, Strategy Selection, Walk-Forward Evaluation, Portfolio Curation)
- Strategy selection via meta-weighting competition
- Walk-forward performance evaluation
- SLSQP optimization for portfolio weights
- Tier-based allocation
- Conviction threshold filter (>50)

### Known Issues

- Slow execution (2-5 minutes)
- Complex architecture (4 phases)
- Duplicate conviction scoring logic
- Limited candidate pool (~40-80 symbols)

---

## Version History Summary

| Version | Date | Architecture | Execution Time | Key Feature |
|---------|------|--------------|----------------|-------------|
| 7.0.0 | 2026-04-02 | 2 phases | 20-40 sec | Conviction-based curation |
| 6.0.0 | Previous | 4 phases | 2-5 min | Walk-forward evaluation |

---

## Upcoming (Future Versions)

### Recommended Enhancements

- [ ] Optional walk-forward tracking (advanced mode)
- [ ] Conviction threshold slider (user-configurable)
- [ ] Strategy filtering UI (manual selection)
- [ ] Portfolio performance tracking over time
- [ ] Parallel strategy execution
- [ ] Improved caching for strategy outputs
- [ ] Conviction explainability breakdown

---

## Migration Notes

### From v6.0.0 to v7.0.0

**Breaking Changes:**
- Walk-forward evaluation removed — Tab 2 (Performance) now shows methodology explanation
- Strategy selection removed — All 80+ strategies run by default
- Meta-weighting removed — Simple conviction-based formula used instead

**Migration Path:**
If you need walk-forward evaluation:
1. Restore `walk_forward.py` from backup
2. Restore `strategy_selector.py` from backup
3. Re-add imports in `app.py`
4. Re-enable Phase 3 in `_run_analysis()` function

---

**PRAGYAM** — Portfolio Intelligence | @thebullishvalue
