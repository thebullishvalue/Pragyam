# PRAGYAM (प्रज्ञम) — Portfolio Intelligence

**Version:** 8.0.0
**Author:** @thebullishvalue
**License:** Proprietary (See LICENSE file)

Conviction-based portfolio curation for Indian equity markets using 95 quantitative strategies, with **self-tuning Bayesian calibration** of conviction weights per (universe, index, regime).

**Latest:** v8.0.0 — Intelligence Mode: per-scope passport calibration via Optuna TPE; Model Passport sidebar card; Phase 1.5 auto-calibration; explicit fallback diagnostics.

---

## Overview

PRAGYAM is a **two-and-a-half-phase** portfolio system:

1. **All 95 strategies run** — Every strategy generates candidate holdings
2. **Conviction scoring** — Each symbol scored 0–100 from 4 technical signals (RSI, Oscillator, Z-Score, MA alignment)
3. **Top N selection** — Highest conviction scores selected (default top 30)
4. **Conviction-based weighting** — `weight = (conviction / total) × 100`, bounded 1%–10%

The four signal weights — historically hard-coded `0.30 / 0.30 / 0.20 / 0.20` — are now **learned per (universe, index, regime)** by maximising the cross-sectional Information Ratio of conviction against forward 10-day returns. Different market regimes reward different signals; the system discovers the right mix automatically.

**Execution time:** ~20–40 s on the reuse path · ~30–50 s on first calibration of a new (universe, index, regime) scope.

---

## Features

| Feature | Description |
|---------|-------------|
| **Conviction Scoring** | 4 signals: RSI, Oscillator, Z-Score, MA alignment. Each in [-2, +2] |
| **Per-Scope Calibration** | Weights learned by Optuna TPE per `(universe, selected_index, regime)` |
| **Model Passport** | Sidebar card showing live profile state · trained-on label · regime · train/val IR · timestamp |
| **All Strategies** | 95 quantitative strategies contribute candidates |
| **Position Bounds** | 1% minimum, 10% maximum per position |
| **Regime Detection** | 7-factor market regime analysis (drives passport key + display) |
| **Profile Import/Export** | Share calibrated passports as JSON between machines |
| **Live Data** | Real-time NSE / US / global data via yfinance |
| **Graceful Fallback** | Insufficient data → defaults, with explicit reason surfaced in the UI |

---

## Installation

```bash
git clone <repository-url>
cd Pragyam-main

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.18.0
- yfinance >= 0.2.28
- scipy >= 1.11.0
- scikit-learn >= 1.3.0
- **optuna >= 3.5.0** (new in v8.0.0 — Bayesian calibration)
- colorama >= 0.4.6

---

## Usage

### 1. Launch

```bash
streamlit run app.py
```

### 2. Configure (Sidebar)

1. **Analysis Date** — Date for portfolio curation
2. **Portfolio Style** — Swing Trading or SIP Investment (drives dispersion concentration)
3. **Analysis Universe** — ETF Universe · India Indexes · US Indexes · Commodities · Currency · etc.
4. **Portfolio Parameters** — Capital, Number of Positions
5. **Run Analysis**
6. **Model Passport** (below the Run button) — toggle Intelligence Mode · view active passport · import/export/reset

### 3. Review Results

Four tabs on the result page:

| Tab | Contents |
|-----|----------|
| **Portfolio** | Holdings, conviction signals, position guide |
| **Position Guide** | Per-position signal breakdown with entry conditions |
| **Regime** | 7-factor regime composite + history |
| **Intelligence** | Calibration status, train/val IR, manual recalibrate / reset, fallback diagnostics |
| **System** | Execution metrics, configuration, version info |

---

## Conviction Scoring

### Signal Components (each ∈ [-2, +2])

| Signal | Calculation |
|--------|-------------|
| **RSI**       | > 60: +2, > 52: +1, < 48: −1, < 40: −2 |
| **Oscillator**| > EMA9 & > 0: +2, > EMA9: +1, < EMA9 & < 0: −2, else: −1 |
| **Z-Score**   | < −2: +2, < −1: +1, > +2: −2, > +1: −1 |
| **MA Align**  | Count of 5 bullish MA conditions, scaled to [-2, +2] |

### Composite

```python
raw = (rsi_signal    * w_rsi +
       osc_signal    * w_osc +
       zscore_signal * w_z +
       ma_signal     * w_ma)

# raw ∈ [-2, +2]  →  conviction_score ∈ [0, 100]
conviction_score = (raw + 2) / 4 * 100
```

### Weights — Two Modes

| Mode | `w_rsi` | `w_osc` | `w_z` | `w_ma` |
|------|---------|---------|-------|--------|
| **Standard** (Intelligence off, or no passport) | 0.30 | 0.30 | 0.20 | 0.20 |
| **Intelligence** (passport exists) | Learned | Learned | Learned | Learned |

In Intelligence mode the four weights sit on the 4-simplex (sum to 1, each ≥ 0) and are read from `IntelligencePassport(universe, selected_index, regime).get_weights()`.

### Conviction Dispersion (style-aware)

Style is a user preference, not something the calibrator learns:

| Style | Boost (Above Median) | Penalty (Below Median) |
|-------|---------------------|------------------------|
| **SIP Investment** | ×2.25 (+125%) | ×0.50 (−50%) |
| **Swing Trading**  | ×3.25 (+225%) | ×0.25 (−75%) |

### Portfolio Weighting

```python
# For the top N positions by adjusted conviction:
weight_i = (adjusted_conviction_i / Σ adjusted_conviction) * 100

# Apply bounds:
1% ≤ weight_i ≤ 10%
```

---

## Intelligence Mode — How Calibration Works

### Objective

Maximise the **Information Ratio (IR)** of the cross-sectional Spearman IC between the weighted conviction score and forward 10-day returns, evaluated on a chronological train/val split (70 / 30).

```
For each date d in train:
  IC_d = SpearmanCorr( conviction_score(weights, d), forward_return_10d(d) )
IR_train = mean(IC) / std(IC)
```

A passport is **saved only when val IR is measurable** on the held-out split.

### Search Space — 4-Simplex

The four weights must satisfy `w_rsi + w_osc + w_z + w_ma = 1` with each ≥ 0. The search reparameterises this as a softmax over four unconstrained scalars in `[-3, +3]`, giving Optuna's TPE sampler a smooth, full-support landscape with no boundary degeneracies.

### Passport Keying

```
.passports/passport_<universe>__<index>__<regime>.json
```

Examples:
- `.passports/passport_india_indexes__nifty_50__bear.json`
- `.passports/passport_etf_universe__all__weak_bull.json`
- `.passports/passport_commodities__all__chop.json`

Switching universe, index, or regime routes to a different passport file. Calibration learned on `NIFTY 500 · BEAR` is **never** silently applied to `Commodities · CHOP`.

### Passport Schema

```json
{
  "universe": "India Indexes",
  "selected_index": "NIFTY 500",
  "regime": "BEAR",
  "weights": {
    "w_rsi": 0.683,
    "w_osc": 0.035,
    "w_z":   0.261,
    "w_ma":  0.021
  },
  "train_ir": 0.243,
  "val_ir":   0.360,
  "n_train_dates": 127,
  "n_val_dates":   55,
  "n_trials": 100,
  "horizon": 10,
  "timestamp": "2026-05-11T15:42:00.000000",
  "engine_version": "v4-pragyam-conviction",
  "is_calibrated": true
}
```

The `engine_version` field invalidates stale passports automatically when the calibration schema changes.

### Model Passport Card (Sidebar)

Located directly below the **Run Analysis** button. Shows live state of the passport for the currently-selected `(universe, selected_index, regime)`:

| Field | Source |
|-------|--------|
| **Profile** | `Default` · `Calibrated` · `Default · Off` (toggle off) |
| **Trained on** | `selected_index` (falls back to `universe`) |
| **Regime** | Currently-detected market regime |
| **Train IR** | Optimizer's best train IR |
| **Val IR**   | Hold-out validation IR |
| **Updated** | ISO timestamp of last calibration |

Controls:
- **↑ Import Profile** — upload a passport JSON (or a `{"weights": {...}}` shape)
- **↓ Export Profile** — download the active passport as JSON
- **↺ Reset to Defaults** — delete the passport file for the current scope only

---

## Architecture

```
PRAGYAM v8.0.0 — 2.5 Phases

┌──────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA FETCHING                                       │
│ → Fetch historical data for all symbols (yfinance)           │
│ → Detect market regime (7-factor composite)                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ PHASE 1.5: INTELLIGENCE (on first encounter of a scope)      │
│ → Look up passport for (universe, index, regime)             │
│ → If exists  → reuse calibrated weights, skip to Phase 2     │
│ → If missing → harvest signals + forward returns             │
│              → Optuna TPE × 100 trials                       │
│              → save passport on measurable val IR            │
│              → st.rerun() so the sidebar repaints fresh      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ PHASE 2: CONVICTION-BASED CURATION                           │
│ → Run ALL 95 strategies                                      │
│ → Aggregate candidate holdings (~200–400 symbols)            │
│ → Score conviction using active weights (passport or default)│
│ → Select top N by conviction                                 │
│ → Apply style-aware dispersion (SIP / Swing)                 │
│ → Apply formula: weight = (adj_conviction / total) × 100     │
│ → Apply position bounds (1%–10%)                             │
└──────────────────────────────────────────────────────────────┘
```

### Module Structure

```
Pragyam-main/
├── app.py                    # Main UI (Streamlit) — ~1700 lines
├── intelligence.py           # Bayesian calibration + passport persistence — ~430 lines (NEW)
├── universe.py               # Universe definitions & selection — ~450 lines
├── portfolio.py              # Conviction-based weighting — ~165 lines
├── regime.py                 # Market regime + conviction scoring — ~690 lines
├── strategies.py             # 95 BaseStrategy implementations
├── backdata.py               # Data fetching (yfinance)
├── charts.py                 # Plotly visualisations — ~250 lines
├── circuit_breaker.py        # yfinance rate limiting — ~315 lines
├── logger_config.py          # Console output system — ~280 lines
├── metrics.py                # Execution metrics — ~270 lines
├── ui/
│   ├── theme.py              # CSS injection + chart theming
│   ├── theme.css             # Obsidian-style design system
│   └── components.py         # Reusable UI primitives
├── .passports/               # Calibrated passports (one JSON per scope) — NEW
│   └── passport_<universe>__<index>__<regime>.json
├── requirements.txt
└── pyproject.toml
```

---

## Performance

| Metric | v8.0.0 (reuse) | v8.0.0 (first cal) | v7.x | v6.0.0 |
|--------|----------------|---------------------|------|--------|
| Execution Time | 20–40 s | 30–50 s | 20–40 s | 2–5 min |
| Phases | 2.5 | 2.5 | 2 | 4 |
| Strategies | All 95 | All 95 | All 80+ | 4 selected |
| Calibration | per-scope, persisted | per-scope, persisted | none | none |

Calibration is a one-time cost per `(universe, index, regime)` tuple. The passport persists on disk; subsequent runs in the same scope reuse it instantly.

---

## Key Changes from v7.x

### New
- ✅ **Intelligence Mode** — per-scope conviction-weight calibration
- ✅ **Model Passport** sidebar card (mirrors Sanket's design)
- ✅ **Phase 1.5** — auto-calibration on first encounter of a scope
- ✅ **`intelligence.py`** module with Optuna TPE + passport persistence
- ✅ **Import / Export / Reset** controls for sharing passports across machines
- ✅ **Explicit fallback diagnostics** — Intelligence tab states *why* a calibration was skipped (insufficient history, sparse harvest, validation IR not measurable) instead of silently defaulting

### Changed
- 🔧 `regime.compute_conviction_signals(...)` — new `universe` + `selected_index` parameters
- 🔧 `portfolio.compute_conviction_based_weights(...)` — same new parameters threaded through
- 🔧 Conviction weights are now **scope-aware**: Standard mode = defaults, Intelligence mode = passport-driven
- 🔧 Dispersion remains style-aware (SIP / Swing); it is **not** something the calibrator learns

### Removed
- ❌ Hard-coded `0.30 / 0.30 / 0.20 / 0.20` as the only available weighting (still the fallback default)

---

## Example Run Output

```
Phase 1.5 (first time on India Indexes · NIFTY 500 · BEAR):
─────────────────────────────────────────────────
Building signal-return panel · 10-day horizon
Optuna TPE · 127 dates · 5,080 (date, symbol) rows
Trial 47/100 · best IR +0.341
…
Train IR +0.243 · Val IR +0.360 · Passport saved
─────────────────────────────────────────────────

Subsequent runs on the same scope:
─────────────────────────────────────────────────
Intelligence ready · India Indexes · NIFTY 500 · BEAR
Val IR +0.360 · calibrated 2026-05-11 15:42
─────────────────────────────────────────────────

Execution Summary
─────────────────────────────────────────────────
Run ID:             20260511_154200
Strategies Run:     95
Candidate Symbols:  287
Positions Selected: 30
Avg Conviction:     62.3/100
Top Conviction:     78/100
Status:             SUCCESS
─────────────────────────────────────────────────
```

---

## Troubleshooting

### Intelligence tab shows "FELL BACK TO DEFAULTS · SKIPPED"

The Intelligence tab now states the specific reason. Common causes:

| Reason | Resolution |
|--------|-----------|
| `Need >15 days of history (have N)` | Increase lookback or pick a later analysis date |
| `Harvest produced no usable rows` | Universe has too-sparse indicator coverage on the date range |
| `Validation IR not measurable` | Cross-section too small or signals too constant; try a broader universe |

### Sidebar Model Passport shows "Default" after Run Analysis

This was a v8.0.0 pre-release bug where the sidebar painted before Phase 1.5 ran. Fixed: after a successful first-run calibration, Phase 1.5 triggers an `st.rerun()` so the sidebar repaints with the freshly-saved passport. If you still see this, hit **Run Analysis** once more — the passport is on disk and will be picked up.

### Passport doesn't persist across `streamlit cloud` deploys

Streamlit Cloud rebuilds the container on each deploy and the `.passports/` directory is ephemeral. Use the **↓ Export Profile** button to download, then **↑ Import Profile** after the rebuild.

### "Engine version mismatch — passport ignored"

The `engine_version` field in the JSON is checked at load time. If you have old `passport_<regime>.json` (regime-only key, v7.x prototype) or `v3.0-sanket-fidelity` files, they are auto-rejected. Safe to delete `.passports/*.json` and recalibrate.

### Slow execution

- First calibration adds ~5–10 s; subsequent runs in the same scope are unchanged
- For NIFTY 500 / S&P 500 the calibration trial loop is still <10 s thanks to vectorised IC computation
- Disable Intelligence Mode (sidebar toggle) to skip Phase 1.5 entirely if you're benchmarking

### Rate limiting from yfinance

Circuit breaker in `circuit_breaker.py` handles this. Reduce universe size or use a premium data source for production.

---

## License

Proprietary Software License — See LICENSE file for details.

Copyright (c) 2024–2026 @thebullishvalue. All Rights Reserved.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.

---

## Version History

| Version | Date | Architecture | Headline |
|---------|------|--------------|----------|
| **8.0.0** | 2026-05-11 | 2.5 phases | **Intelligence Mode** — per-scope Bayesian calibration via Optuna TPE; Model Passport sidebar; Phase 1.5 |
| 7.2.0 | 2026-04-13 | 2 phases | "Terminal Glass" design system overhaul |
| 7.0.5 | 2026-04-05 | 2 phases | Production hardening, dead code removal |
| 7.0.4 | 2026-04-02 | 2 phases | Style-aware dispersion (SIP / Swing) |
| 7.0.0 | 2026-04-02 | 2 phases | Conviction-based curation |
| 6.0.0 | Previous | 4 phases | Walk-forward evaluation |

---

## Disclaimer

This software is for educational and research purposes only. Not financial advice. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.

---

## Contact

**@thebullishvalue** — Portfolio Intelligence Systems
