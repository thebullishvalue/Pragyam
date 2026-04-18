# PRAGYAM (प्रज्ञम) — Portfolio Intelligence

**Version:** 7.0.5
**Author:** @thebullishvalue
**License:** Proprietary (See LICENSE file)

Conviction-based portfolio curation for Indian equity markets using 80+ quantitative strategies.

**Latest:** v7.0.5 — Production hardening, dead code removal, and refined terminal logging.

---

## Overview

PRAGYAM uses a **pure conviction-based approach** to portfolio construction:

1. **All 80+ strategies run** — Every strategy generates candidate holdings
2. **Conviction scoring** — Each symbol scored 0-100 using 4 technical signals
3. **Top 30 selection** — Highest conviction scores selected
4. **Simple weighting** — `weight = (conviction / total) × 100`

**Execution time:** ~20-40 seconds (10x faster than v6.0.0)

---

## Features

| Feature | Description |
|---------|-------------|
| **Conviction Scoring** | 4 signals: RSI (30%), Oscillator (30%), Z-Score (20%), MA Alignment (20%) |
| **All Strategies** | 80+ quantitative strategies contribute candidates |
| **Simple Formula** | `weight_i = (conviction_score_i / Σ all_conviction_scores) × 100` |
| **Position Bounds** | 1% minimum, 10% maximum per position |
| **Regime Detection** | 7-factor market regime analysis (display only) |
| **Live Data** | Real-time NSE data via yfinance |

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Pragyam-02.01

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- streamlit>=1.28.0
- pandas>=2.0.0
- numpy>=1.24.0
- plotly>=5.18.0
- yfinance>=0.2.28
- scipy>=1.11.0
- colorama>=0.4.6

**Note:** `scikit-learn` and `matplotlib` removed as dependencies in v7.0.0 (no longer needed).

---

## Usage

### 1. Select Universe

Use the sidebar dropdown to choose your analysis universe:

**Available Universes:**
- **ETF Universe** — 30 NSE ETFs covering major indices and sectors
- **India Indexes** — NIFTY 50, NIFTY 100, NIFTY 500, F&O Stocks, sectoral indices
- **US Indexes** — S&P 500, DOW JONES, NASDAQ 100
- **Commodities** — 24 commodity futures
- **Currency** — 24 currency pairs

Simply select the universe and index (where applicable) from the sidebar dropdown before running analysis.

### 2. Run Application

```bash
streamlit run app.py
```

### 3. Use the Interface

1. **Select Analysis Date** — Choose the date for portfolio curation
2. **Set Investment Style** — Swing Trading or SIP Investment (UI option, legacy feature)
3. **Configure Parameters:**
   - Capital (₹) — Total capital to deploy (default: ₹2,500,000)
   - Number of Positions — Holdings in final portfolio (default: 30, range: 5-100)
   - Min/Max Position Weight — Bounds for individual positions (1%-10% default)
4. **Click "Run Analysis"**

### 4. Review Results

- **Tab 1: Portfolio** — Holdings with conviction signals and position guide
- **Tab 2: Performance** — Methodology explanation (walk-forward removed in v7.0.0)
- **Tab 3: Regime** — Market regime analysis (7-factor composite)
- **Tab 4: System** — Technical details and configuration

---

## Conviction Scoring Formula

### Signal Components

| Signal | Weight | Calculation |
|--------|--------|-------------|
| **RSI** | 30% | >60: +2, >52: +1, <48: -1, <40: -2 |
| **Oscillator** | 30% | >EMA9 & >0: +2, >EMA9: +1, <EMA9 & <0: -2, else: -1 |
| **Z-Score** | 20% | <-2: +2, <-1: +1, >2: -2, >1: -1 |
| **MA Alignment** | 20% | Count of 5 bullish conditions (0-5 scaled to -2 to +2) |

### Composite Score

```python
raw = (RSI_signal × 0.30 +
       OSC_signal × 0.30 +
       Z-Score_signal × 0.20 +
       MA_signal × 0.20)

# Normalize to 0-100 scale
conviction_score = (raw + 2) / 4 × 100
```

### Conviction Dispersion Weighting (v7.0.5)

Style-aware dispersion weighting automatically adjusts based on investment style:

| Style | Boost (Above Median) | Penalty (Below Median) | Top Pick Advantage |
|-------|---------------------|------------------------|-------------------|
| **SIP Investment** | +125% (×2.25) | -50% (×0.50) | ~350% more weight |
| **Swing Trading** | +225% (×3.25) | -75% (×0.25) | ~1200% more weight |

```python
# SIP Mode (conservative concentration)
if conviction_score > median:
    adjusted_score = conviction_score × 2.25  # +125% boost
else:
    adjusted_score = conviction_score × 0.50  # -50% penalty

# Swing Mode (aggressive concentration, 2σ more)
if conviction_score > median:
    adjusted_score = conviction_score × 3.25  # +225% boost
else:
    adjusted_score = conviction_score × 0.25  # -75% penalty

weight_i = (adjusted_conviction_i / Σ all_adjusted_conviction) × 100
```

**Effect:** 
- SIP: Strong concentration in high-conviction picks (~350% tilt)
- Swing: Very aggressive concentration (~1200% tilt) — maximum alpha capture

### Portfolio Weighting

```python
# For top 30 positions by conviction score
weight_i = (adjusted_conviction_i / Σ all_adjusted_conviction) × 100

# Apply bounds: 1% ≤ weight_i ≤ 10%
```

---

## Architecture

```
PRAGYAM v7.0.5 — 2 Phases

┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA FETCHING                                      │
│ → Fetch historical data for all symbols (yfinance)          │
│ → Detect market regime (7-factor composite)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: CONVICTION-BASED CURATION                          │
│ → Run ALL 80+ strategies                                    │
│ → Aggregate all candidate holdings (~200-400 symbols)       │
│ → Compute conviction scores (regime.py)                     │
│ → Select top 30 by conviction                               │
│ → Apply formula: weight = (conviction / total) × 100        │
│ → Apply bounds (1%-10%)                                     │
│ → Calculate units and value                                 │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
Pragyam-Final/
├── app.py                    # Main UI (Streamlit) — ~1300 lines
├── universe.py               # Universe definitions & selection — ~450 lines
├── portfolio.py              # Conviction-based weighting — ~150 lines
├── regime.py                 # Market regime + conviction scoring — ~640 lines
├── strategies.py             # 95 BaseStrategy implementations
├── backdata.py               # Data fetching (yfinance)
├── charts.py                 # Plotly visualizations — ~250 lines
├── circuit_breaker.py        # yfinance rate limiting — ~315 lines
├── logger_config.py          # Console output system — ~280 lines
├── metrics.py                # Execution metrics — ~270 lines
├── style.css                 # UI styling
├── requirements.txt          # Dependencies
└── pyproject.toml            # Project configuration
```

---

## Performance

| Metric | v7.0.0 | v6.0.0 | Improvement |
|--------|--------|--------|-------------|
| Execution Time | 20-40 sec | 2-5 min | **6-10x faster** |
| Code Lines | 3,500 | 5,000+ | **-30%** |
| Phases | 2 | 4 | Simpler |
| Strategies | All 80+ | 4 selected | Maximum diversification |
| Candidate Pool | ~200-400 | ~40-80 | **5x larger** |

---

## Key Changes from v6.0.0

### Removed Features
- ❌ Walk-forward evaluation (Phase 3)
- ❌ Strategy selection (Phase 2 old)
- ❌ Meta-weighting competition
- ❌ Tier-based allocation
- ❌ Conviction threshold filter (>50)

### New Features
- ✅ All 80+ strategies run (no filtering)
- ✅ Pure conviction-based weighting
- ✅ Simple, transparent formula
- ✅ No threshold (all symbols eligible)
- ✅ 10x faster execution

---

## Example Output

```
Execution Summary
─────────────────────────────────────────
Run ID:             20260402_143022
Strategies Run:     83
Candidate Symbols:  287
Positions Selected: 30
Avg Conviction:     62.3/100
Top Conviction:     78/100
Status:             SUCCESS
─────────────────────────────────────────

Portfolio: 30 positions
Total Value: ₹2,487,350
Cash Remaining: ₹12,650
```

---

## Troubleshooting

### "No historical data loaded"
- Check your universe selection in the sidebar
- Verify internet connection
- Reduce lookback period in sidebar
- Ensure yfinance can access NSE data (may be rate-limited)
- For India/US Indexes, ensure the index constituent fetch succeeded (check status message)

### "No holdings generated"
- Some strategies may not generate signals for current market conditions
- Try a different analysis date
- Increase number of positions in sidebar
- Check that your universe has sufficient symbols (recommend 50+ for best results)

### "Conviction signals unavailable"
- Check that `current_df` has indicator columns
- Verify regime.py is functioning correctly
- Ensure data fetch completed successfully

### Slow execution or timeouts
- v7.0.5 executes in 20-40 seconds for small universes (<50 symbols)
- Larger universes (NIFTY 500, S&P 500) may take 1-3 minutes
- Reduce universe size if needed (e.g., use NIFTY 50 instead of NIFTY 500)
- Check internet connection stability
- Increase Streamlit server timeout if deploying remotely

### Rate limiting from yfinance
- Circuit breaker implemented in `circuit_breaker.py`
- Add fewer symbols or use smaller universe to distribute requests
- Consider using premium data source for production

---

## License

Proprietary Software License — See LICENSE file for details.

Copyright (c) 2024-2026 @thebullishvalue. All Rights Reserved.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and notable changes.

---

## Version History

| Version | Date | Architecture | Execution Time | Key Feature |
|---------|------|--------------|----------------|-------------|
| 7.0.5 | 2026-04-05 | 2 phases | 20-40 sec | Production hardening, dead code removal |
| 7.0.4 | 2026-04-02 | 2 phases | 20-40 sec | Style-aware dispersion (SIP/Swing) |
| 7.0.3 | 2026-04-02 | 2 phases | 20-40 sec | Aggressive conviction dispersion (+75%/-50%) |
| 7.0.2 | 2026-04-02 | 2 phases | 20-40 sec | Strong conviction dispersion (+40%/-30%) |
| 7.0.1 | 2026-04-02 | 2 phases | 20-40 sec | Conviction dispersion weighting |
| 7.0.0 | 2026-04-02 | 2 phases | 20-40 sec | Conviction-based curation |
| 6.0.0 | Previous | 4 phases | 2-5 min | Walk-forward evaluation |

---

## Disclaimer

This software is for educational and research purposes only. Not financial advice. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.

---

## Contact

**@thebullishvalue** — Portfolio Intelligence Systems
