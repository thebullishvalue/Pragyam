# PRAGYAM (प्रज्ञम) — Portfolio Intelligence

**Version:** 10.0.1
**Author:** @thebullishvalue
**License:** Proprietary (See LICENSE file)

Conviction-based portfolio curation for Indian equity markets using 95 quantitative strategies, with **self-tuning Bayesian calibration** of the conviction weights (per universe, index, regime family). The market-regime detector uses fixed factor weights.

**Latest:** v10.0.1 — calibration is now actually reachable: Phase 1.5's 100-day
harvest window could yield at most 5 non-overlapping paired validation dates
while the significance gate requires 8, so every calibration burned 100 Optuna
trials and then failed the gate by construction. A dedicated ~18-month
estimation panel (`_CALIBRATION_LOOKBACK_FILES`) now backs both the automatic
and manual calibration paths, with a fail-fast reachability check up front.
Also: the sidebar regime/passport cards now always repaint after a run
completes; the F&O universe fetch reads the real NSE derivatives market-lots
file instead of silently falling back to NIFTY 500; two Analytics bugs are
fixed (the Relative Performance chart's benchmark line/legend disagreed with
the Benchmark Comparison card, and CAGR/Alpha/Calmar/Info-Ratio were not
annualized for 20–251-day windows); plot containers no longer clip at the
bottom edge; and the result page's section spacing is now a single consistent
rhythm instead of the previous mix of dividers and gaps. See `CHANGELOG.md`
for the full list. Built on v10.0.0 (full correctness/rigor audit — see
`AUDIT_DIRECTIVES.md`), v9.3.0 (fixed-weight regime detector), v9.2.0
(Analytics tab), v9.1.0 (Broker Sync), and v9.0.0 (Value Area / VAP signal +
8th acceptance factor).

---

## Overview

PRAGYAM is a **multi-phase** portfolio system:

1. **All 95 strategies run** — each strategy's own top-quartile picks cast an endorsement vote (see Strategy Endorsement below)
2. **Conviction scoring** — Each symbol scored 0–100 from 6 technical signals (RSI, Oscillator, Z-Score, MA alignment, Value Area / VAP from the volume profile, and Strategy Endorsement)
3. **Top N selection** — Highest conviction scores selected (default top 30) among symbols with at least one usable signal, ties broken by value-area position (a discount to accepted value is preferred)
4. **Conviction-based weighting** — continuous power-law dispersion (`adjusted = conviction ** gamma`) then `weight = (adjusted / total) × 100`, bounded 1%–10% (relaxed automatically when the position count makes the nominal bounds infeasible)

Five of the six signal weights are **learned per (universe, index, regime family)** by maximising the cross-sectional Information Ratio of conviction against forward 10-day returns on dates that actually occurred in that regime family; the sixth (Strategy Endorsement) has no historical values to calibrate against and stays fixed. A passport is only deployed when it beats the even `1/6 ×6` fallback by a **statistically significant** margin (paired significance test, not a raw IR comparison) — otherwise Pragyam explicitly falls back to defaults and says why. The eight regime-detection factor weights are **fixed** (not calibrated). Different market regimes reward different signals; Intelligence mode discovers the right conviction mix automatically, when the evidence actually supports it.

**Execution time:** ~20–40 s on the reuse path · first calibration of a new (universe, index, regime family) scope now fetches a dedicated ~18-month estimation panel (v10.0.1), so expect a noticeably longer first run (extra data fetch + a larger harvest) — every subsequent run on that scope is back to the reuse-path timing.

---

## Features

| Feature | Description |
|---------|-------------|
| **Conviction Scoring** | 6 signals: RSI, Oscillator, Z-Score, MA alignment, Value Area (VAP, volume profile), Strategy Endorsement (95-strategy vote rank). Each in [-2, +2] |
| **Per-Scope Calibration** | Five of six conviction weights learned by Optuna TPE per `(universe, selected_index, regime FAMILY)`, on dates that actually occurred in that family — a passport is deployed only when it beats the default baseline by a statistically significant margin. Strategy Endorsement stays fixed (no historical values to calibrate against). The regime detector uses fixed factor weights (not calibrated) |
| **Model Passport** | Sidebar card showing live profile state · trained-on label · regime · train/val IR · timestamp |
| **All Strategies** | 95 quantitative strategies contribute candidates; each strategy's own top-quartile picks cast an endorsement vote that feeds the Strategy Endorsement signal |
| **Position Bounds** | 1% minimum, 10% maximum per position (auto-relaxed when the position count makes the nominal bounds mathematically infeasible) |
| **Regime Detection** | 8-factor market regime analysis with fixed weights (drives passport key + display) |
| **Profile Import/Export** | Share calibrated passports as JSON between machines, with numeric + simplex validation on import |
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

Seven tabs on the result page:

| Tab | Contents |
|-----|----------|
| **Portfolio** | Holdings, conviction signals, position guide |
| **Position Guide** | Per-position signal breakdown with entry conditions |
| **Analytics** | Tracks the curated book vs a universe-matched benchmark: timeframe selector, normalized performance chart, and risk-adjusted / risk / benchmark metrics (Sharpe, Sortino, Calmar, Alpha, Beta, VaR, capture ratios) |
| **Regime** | 8-factor regime composite (learned weights) + history |
| **Intelligence** | Calibration status, train/val IR, active weights (5 conviction signals), manual recalibrate / reset, fallback diagnostics |
| **Broker Sync** | Writes the live curated units into broker order-template JSONs (e.g. Kite ETF.json), producing import-ready order files |
| **System** | Execution metrics, configuration, version info |

---

## Conviction Scoring

### Signal Components (each ∈ [-2, +2])

| Signal | Calculation |
|--------|-------------|
| **RSI**        | > 60: +2, > 52: +1, < 48: −1, < 40: −2 |
| **Oscillator** | > EMA9 & > 0: +2, > EMA9: +1, < EMA9 & < 0: −2, else: −1 |
| **Z-Score**    | < −2: +2, < −1: +1, > +2: −2, > +1: −1 |
| **MA Align**   | Count of 5 bullish MA conditions, scaled to [-2, +2] |
| **Value Area (VAP)** | Volume-profile premium/discount to value (POC), volatility-normalised. Discount: > 2 → +2, > 1 → +1; premium: < −2 → −2, < −1 → −1 |
| **Strategy Endorsement** | Cross-sectional percentile rank of how many of the 95 strategies picked this symbol among their own top-quartile holdings, mapped linearly onto [-2, +2] |

The Value Area signal is derived from a rolling **volume profile** (`backdata.compute_volume_profile`): it bins traded volume across price to find the point-of-control (POC) and value area, then scores how far the current close sits at a discount (mean-reversion long, positive) or premium (negative) to accepted value.

The Strategy Endorsement signal makes the 95-strategy layer materially influence which symbols are selected: each strategy's own top quartile (by its internal weighting) counts as an "endorsement," and a symbol's rank among all endorsement counts becomes its signal. It cannot be calibrated historically (it only exists for the live day a run executes on — recomputing it for every historical date would mean re-running all 95 strategies per day) so its weight is always fixed at the default.

A symbol with **zero** usable signals (absent from the live indicator snapshot, or every one of the six signals unavailable) is excluded from top-N selection entirely — it is not treated as "neutral," since the model has no opinion on it, not a neutral one.

### Composite

```python
raw = (rsi_signal    * w_rsi +
       osc_signal    * w_osc +
       zscore_signal * w_z +
       ma_signal     * w_ma +
       vap_signal    * w_vap +
       strat_signal  * w_strat)

# raw ∈ [-2, +2]  →  conviction_score ∈ [0, 100]
conviction_score = (raw + 2) / 4 * 100
```

If a symbol is missing one or more signals (e.g. a universe with zero-volume tickers, where the Oscillator/Z-Score/Value Area signals are structurally unavailable), the active weights are **renormalized over the signals that are actually present**, so the composite still spans the full [-2, +2] range instead of being compressed toward neutral by structurally-dead weight slots.

### Weights — Two Modes

| Mode | `w_rsi` | `w_osc` | `w_z` | `w_ma` | `w_vap` | `w_strat` |
|------|---------|---------|-------|--------|---------|-----------|
| **Standard** (Intelligence off, or no passport) | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |
| **Intelligence** (passport exists) | Learned | Learned | Learned | Learned | Learned | Fixed (1/6) |

In Intelligence mode, five of the six weights sit on a 5-simplex scaled to fill `(1 − w_strat)` and are read from `IntelligencePassport(universe, selected_index, regime).get_weights()`; `w_strat` is never searched.

### Conviction Dispersion (style-aware, continuous)

Style is a user preference, not something the calibrator learns. Position weights are concentrated via a **continuous power-law transform** — `adjusted = conviction_score ** gamma` — rather than a discontinuous median-step boost/penalty (the earlier formula created up to a ~13x weight ratio between two candidates one conviction point apart, purely from which side of the median they fell on):

| Style | `gamma` | Effect |
|-------|---------|--------|
| **SIP Investment** | 2.5 | Moderate concentration |
| **Swing Trading**  | 4.5 | Aggressive concentration |

Because the transform is continuous in the score itself, tied scores get identical weight and adjacent scores get a smoothly adjacent weight — no cliff at any boundary.

### Portfolio Weighting

```python
# Top N selected by conviction among symbols with >=1 usable signal; ties
# broken by value-area position (prefer names at a discount to accepted
# value). Then for those N:
adjusted_conviction_i = conviction_score_i ** gamma
weight_i = (adjusted_conviction_i / Σ adjusted_conviction) * 100

# Apply bounds (auto-relaxed if num_positions makes the nominal bounds
# mathematically infeasible — e.g. 5 positions can't all satisfy a 10% cap):
1% ≤ weight_i ≤ 10%
```

---

## Intelligence Mode — How Calibration Works

### Objective

Maximise the **Information Ratio (IR)** of the cross-sectional Spearman IC between the weighted conviction score and forward 10-day returns, evaluated on non-overlapping dates within an embargoed, chronological train/val split (50 / 50), **restricted to dates that occurred in the target regime family** (Bull / Chop / Bear Mix).

```
For each non-overlapping date d in train:
  IC_d = SpearmanCorr( conviction_score(weights, d), forward_return_10d(d) )
IR_train = mean(IC) / std(IC)
```

Non-overlapping date sampling avoids the serial-correlation bias that comes from adjacent dates sharing most of their forward-return window (Lo 2002); the 50/50 split (rather than a 70/30) leaves enough validation dates for the significance test below to be viable at realistic panel sizes.

A passport is **saved only when the learned weights beat the default (1/6 ×6) baseline by a statistically significant margin** — a paired one-sample t-test on per-date IC differences between the learned and default weights (not a naive comparison of two independently-estimated IRs, which was found to accept ~40% of pure-noise calibrations at small validation-date counts). The deployed weights are then **shrunk toward the default** in proportion to how strongly the validation split corroborates them, so a calibration that only marginally clears the bar isn't deployed at full conviction.

### Search Space — 5-Simplex (of Six Signals)

The five *calibrated* weights (RSI/OSC/Z/MA/VAP — Strategy Endorsement is excluded, see above) must satisfy `w_rsi + w_osc + w_z + w_ma + w_vap = (1 − w_strat)` with each ≥ 0. The search reparameterises this as a softmax over five unconstrained scalars in `[-3, +3]` scaled to fill the remaining simplex mass, giving Optuna's TPE sampler a smooth, full-support landscape with no boundary degeneracies.

### Regime Factor Weights

The **8 regime-detection factor weights** (momentum, trend, breadth, velocity, extremes, volatility, correlation, acceptance) are **fixed** (`regime.FACTOR_WEIGHTS`) — the regime detector is not calibrated. Only the conviction blend learns weights. The `acceptance` factor reads the cross-sectional volume-profile distribution (how much of the universe trades at a discount vs premium to value) and contributes at its fixed weight.

### Regime-Family Conditioning

A passport is keyed by regime, but the harvest feeding calibration is **filtered to dates that actually occurred in that regime's family** — Bull Market Mix, Chop/Consolidate Mix, or Bear Market Mix (the same three families `regime.REGIME_MIX_MAP` uses for strategy-mix selection). Calibration is conditioned on the coarse family rather than all 7 raw regimes individually, because conditioning on 7 buckets starves each one's sample count at Pragyam's realistic lookback windows. Without this conditioning, a passport keyed "BEAR" would previously be estimated on the *entire* trailing window regardless of what regime was actually in effect on each date — the regime label distinguished only when a calibration happened to run, not the market state the weights were learned under.

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
    "w_rsi":   0.412,
    "w_osc":   0.240,
    "w_z":     0.119,
    "w_ma":    0.118,
    "w_vap":   0.111,
    "w_strat": 0.167
  },
  "train_ir": 0.243,
  "val_ir":   0.360,
  "n_train_dates": 95,
  "n_val_dates":   95,
  "n_trials": 100,
  "horizon": 10,
  "timestamp": "2026-05-11T15:42:00.000000",
  "engine_version": "v7-pragyam-conviction-strat",
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
PRAGYAM v10.0.1 — Phases

┌──────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA FETCHING                                       │
│ → Fetch historical data for all symbols (yfinance),          │
│   100-day display/curation panel (_REGIME_LOOKBACK_FILES)    │
│ → Detect market regime (8-factor composite, fixed weights)   │
│ → Compute regime-family history series (for Phase 1.5)       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ PHASE 1.5: INTELLIGENCE (on first encounter of a scope)      │
│ → Look up passport for (universe, index, regime)             │
│ → If exists  → reuse calibrated weights, skip to Phase 2     │
│ → If missing → fetch a DEDICATED ~18-month estimation panel  │
│                (_CALIBRATION_LOOKBACK_FILES, 375 days) —     │
│                the 100-day display panel cannot supply       │
│                enough in-family dates for the gate below      │
│              → harvest 5 signals + forward returns over it,  │
│                tagged with regime FAMILY per date             │
│              → filter harvest to the target regime family    │
│              → fail fast if the validation split can't reach │
│                the paired-date floor (min_calibration_dates) │
│              → Optuna TPE × 100 trials (5-of-6-simplex,      │
│                Strategy Endorsement excluded/fixed)           │
│              → save passport ONLY if it beats the default     │
│                baseline by a statistically significant        │
│                margin (paired significance test), with        │
│                shrinkage toward the default                   │
│              → st.rerun() so the sidebar repaints fresh      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ PHASE 2: CONVICTION-BASED CURATION                           │
│ → Discover + run ALL 95 strategies; each strategy's own top  │
│   quartile casts an endorsement vote (feeds Strategy          │
│   Endorsement)                                                │
│ → Aggregate candidate holdings (~200–400 symbols)            │
│ → Score conviction using active weights (passport or default)│
│ → Exclude symbols with zero usable signals from selection    │
│ → Select top N by conviction (value-area tie-break)          │
│ → Apply style-aware CONTINUOUS power-law dispersion           │
│   (conviction ** gamma; SIP / Swing)                          │
│ → Apply formula: weight = (adj_conviction / total) × 100     │
│ → Apply position bounds (1%–10%, auto-relaxed if infeasible) │
└──────────────────────────────────────────────────────────────┘
```

One progress bar spans all three phases with a strictly non-decreasing
percentage (0–20 / 20–35 / 35–100 respectively); every intermediate milestone
label is Title Case and states only what is true at that point (e.g.
"Discovering Strategies" before Phase 2's strategy count is known, then
"Running Strategies · N strategies" once it is).

### Module Structure

```
Pragyam-main/
├── app.py                    # Main UI (Streamlit)
├── intelligence.py           # Bayesian conviction-weight calibration + passport persistence
├── universe.py                # Universe definitions & selection
├── portfolio.py               # Conviction-based weighting + value-area tie-break
├── regime.py                  # Eight-factor market regime + six-signal conviction scoring
├── strategies.py               # 95 BaseStrategy implementations
├── backdata.py                 # Data fetching (yfinance) + volume-profile features
├── charts.py                   # Plotly visualisations
├── circuit_breaker.py          # yfinance rate limiting
├── logger_config.py            # Console output system
├── metrics.py                  # Execution metrics (per-Streamlit-session)
├── ui/
│   ├── theme.py               # CSS injection + chart theming
│   ├── theme.css              # Obsidian Quant design system (dark only)
│   └── components.py          # Reusable UI primitives
├── .passports/                # Calibrated conviction passports (one JSON per scope)
│   └── passport_<universe>__<index>__<regime>.json      # conviction weights
├── AUDIT_DIRECTIVES.md         # Full-system correctness/rigor audit + resolutions
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

> **v10.0.1 note:** the "first cal" figure above predates the calibration-reachability fix. First calibration of a scope now fetches a dedicated ~18-month estimation panel instead of reusing the 100-day display panel (which could never pass the significance gate at all — see Troubleshooting), so first-calibration wall time is higher than 30–50s. The reuse-path figure is unchanged.

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
Regime-family filter: Bear Market Mix · 95 dates (of 190 total)
Optuna TPE · 95 dates · 2,850 (date, symbol) rows
Trial 47/100 · best IR +0.341
…
Train IR +0.812 · Val IR +6.761 (beat default +2.008) · Passport saved

Subsequent runs on the same scope:
─────────────────────────────────────────────────
Intelligence ready · India Indexes · NIFTY 500 · BEAR
Val IR +6.761 · calibrated 2026-05-11 15:42
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
| `Need >= 20 dates ... in the 'X Mix' regime family (have N)` | That regime family hasn't occurred often enough in the lookback window — try a longer lookback, or accept that this scope can't be calibrated yet |
| `Validation window can yield at most N non-overlapping dates (need >= 8) ... needs >= 142` | Reachability pre-check (v10.0.1): this regime family doesn't have enough dates in the ~18-month estimation panel to ever pass the significance gate, so Optuna isn't even run. Wait for more history to accumulate in that regime, or accept the fallback |
| `Only N paired non-overlapping validation dates available (need >= 8)` | The validation split is too thin to statistically distinguish learned weights from chance at this panel size |
| `Validation IR ... is not positive` | The learned weights would anti-predict forward returns out-of-sample — defaults are safer |
| `... did not significantly beat the default baseline` | The calibration doesn't demonstrably improve on the 1/6×6 fallback for this scope; this is an honest "no edge found," not a bug |

These are all **intentional, conservative refusals** — Pragyam will not deploy a calibration it cannot statistically justify, even if that means falling back to defaults more often than earlier versions did.

### Sidebar Model Passport shows "Default" after Run Analysis

This was a v8.0.0 pre-release bug where the sidebar painted before Phase 1.5 ran. Fixed: after a successful first-run calibration, Phase 1.5 triggers an `st.rerun()` so the sidebar repaints with the freshly-saved passport. If you still see this, hit **Run Analysis** once more — the passport is on disk and will be picked up.

### Sidebar regime card shows "Run Analysis to detect the market regime..."

The sidebar only auto-computes the regime card without a spinner when the current date/universe exactly matches the last completed Run Analysis (guaranteed already cached). Browsing to a *different* date or universe shows this awaiting state instead of silently blocking the sidebar for 10-30s on a cache miss — hit **Run Analysis** and the card fills in automatically once Phase 1 completes.

### F&O Stocks universe loads NIFTY 500 instead of the F&O list

Fixed in v10.0.1 — NSE retired the JSON endpoint Pragyam previously used, so
every fetch fell through to the NIFTY 500 depth proxy. `get_fno_stock_list()`
now reads the NSE derivatives market-lots file (`fo_mktlots.csv`) directly. If
you still see the NIFTY 500 fallback message, all three sources (market-lots
file, legacy JSON API, NIFTY 500 archive) failed — likely a transient NSE
outage; retry, or check your network can reach `nsearchives.nseindia.com`.

### Passport doesn't persist across `streamlit cloud` deploys

Streamlit Cloud rebuilds the container on each deploy and the `.passports/` directory is ephemeral. Use the **↓ Export Profile** button to download, then **↑ Import Profile** after the rebuild.

### "Engine version mismatch — passport ignored"

The `engine_version` field in the JSON is checked at load time. If you have old `passport_<regime>.json` (regime-only key, v7.x prototype), `v3.0-sanket-fidelity`, or pre-v7 (`v4`/`v5`/`v6-pragyam-conviction*`) files, they are auto-rejected. Safe to delete `.passports/*.json` and recalibrate.

### Slow execution

- First calibration adds ~5–10 s; subsequent runs in the same scope are unchanged
- For NIFTY 500 / S&P 500 the calibration trial loop is still <10 s thanks to vectorised IC computation
- Disable Intelligence Mode (sidebar toggle) to skip Phase 1.5 entirely if you're benchmarking

### Rate limiting from yfinance

Circuit breaker in `circuit_breaker.py` handles this, with a couple of quick backoff retries on transient failures before a call counts against the breaker. Reduce universe size or use a premium data source for production.

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
| **10.0.1** | 2026-07-06 | Phases | **Calibration reachability fix** — a dedicated ~18-month estimation panel (`_CALIBRATION_LOOKBACK_FILES`) replaces the 100-day display panel for Phase 1.5 and the manual Calibrate button, with a fail-fast reachability check before Optuna runs (previously every calibration was statistically guaranteed to fail the significance gate); sidebar regime/passport cards now always repaint post-run; F&O universe fetch reads the real NSE derivatives market-lots file; two Analytics correctness fixes (benchmark chart/card agreement, true CAGR annualization for sub-year windows); plot-container bottom clipping fixed; result-page section rhythm unified to one spacing doctrine |
| **10.0.0** | 2026-07-02 | Phases | **Full correctness/rigor audit** (`AUDIT_DIRECTIVES.md`) — sixth conviction signal (Strategy Endorsement, from real strategy votes); regime-FAMILY-conditioned calibration; statistically-significant beats-default gate with shrinkage; continuous power-law dispersion (no more weight cliffs); corrected data-layer warm-up/RSI/oscillator/volume-profile handling; fixed benchmark resolution for non-NIFTY universes; frozen run context (sidebar browsing no longer contaminates a curated portfolio's display); dozens of smaller infra/UI truth-in-labeling fixes |
| **9.3.0** | 2026-07-01 | Phases | **Regime detector uses fixed weights** (no longer calibrated) — adopts the legacy hardcoded-weighting design, keeping the 8th acceptance factor; only the conviction blend is calibrated |
| **9.2.0** | 2026-07-01 | Phases | **Analytics** tab — tracks the curated book vs a universe-matched benchmark (risk-adjusted / risk / benchmark metrics + normalized chart), adapting the SWING analysis engine into `analytics.py` |
| **9.1.0** | 2026-07-01 | Phases | **Broker Sync** tab — writes curated units into broker order JSONs (curate → sync → execute), absorbing the standalone JSON utility; regime-badge alignment + uploader styling fixes |
| **9.0.0** | 2026-07-01 | Phases | **Value Area (VAP)** fifth conviction signal from the volume profile; eight-factor regime detector with **learned** weights (incl. Acceptance); value-area tie-break in selection |
| 8.0.0 | 2026-05-11 | 2.5 phases | **Intelligence Mode** — per-scope Bayesian calibration via Optuna TPE; Model Passport sidebar; Phase 1.5 |
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
