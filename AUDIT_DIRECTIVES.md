# PRAGYAM v9.3.0 — Full-System Audit & Resolution Directives

**Audit date:** 2026-07-02 · **Scope:** every module (engines, logical stack, infrastructure, UI/UX, docs)
**Audience:** the implementing AI. Each finding carries: severity, evidence (file:line), the *why* (grounded in quantitative-finance / engineering literature), and a precise, self-contained implementation directive.

Severity scale: **P0** = invalidates the system's core claim or corrupts output · **P1** = materially wrong results or broken feature · **P2** = degrades quality/performance/trust · **P3** = polish, hygiene, documentation.

---

## PART A — QUANTITATIVE METHODOLOGY (the logical stack)

### A1 · P0 — Calibrated passports with **negative validation IR** are deployed to production scoring

**Evidence:** `intelligence.calibrate` (intelligence.py:441) saves a passport whenever `np.isfinite(val_ir) and n_val_ic >= 3` — sign is never checked. On-disk proof (`.passports/`):

| Passport | Train IR | Val IR |
|---|---|---|
| etf_index · BULL | +0.309 | **−0.050** |
| etf_index · CHOP | +0.234 | **−0.111** |
| test · CHOP | +0.386 | **−0.180** |
| testu · CHOP | +0.309 | **−0.384** |

Four of seven live passports have negative out-of-sample IR, i.e. the "learned" weights **anti-predict** forward returns on held-out data, yet Intelligence mode applies them in preference to the 0.20×5 default.

**Why (literature):** This is the canonical backtest-overfitting failure. Selecting the best of 100 trials on the training split guarantees an optimistic train IR (Bailey, Borwein, López de Prado & Zhu, *"The Probability of Backtest Overfitting"*, J. Computational Finance 2017; Bailey & López de Prado, *"The Deflated Sharpe Ratio"*, 2014). DeMiguel, Garlappi & Uppal (RFS 2009) show naive 1/N frequently beats estimated-parameter allocations out-of-sample precisely because estimation error dominates — which is exactly what these passports demonstrate empirically. Deploying a model that validates *worse than the fallback* is strictly value-destroying.

**Directive:**
1. In `intelligence.calibrate`, after computing `val_ir` for the optimized weights, also compute `default_val_ir, _ = _ir_for_weights(DEFAULT_WEIGHTS, val_df)`.
2. Save the passport **only if** `val_ir > 0` **and** `val_ir >= default_val_ir` (the learned mix must beat the fallback where it counts). Otherwise return a structured failure `{"status": "rejected", "reason": f"val IR {val_ir:+.3f} did not beat defaults {default_val_ir:+.3f}"}` instead of `None`, and surface that reason through the existing `last_intel_outcome` plumbing (app.py:2050–2058) and the Intelligence tab fallback card (app.py:1611).
3. Add **shrinkage toward 1/N**: instead of saving the raw optimum, save `w_final = λ·w_opt + (1−λ)·0.20` with `λ = clip(val_ir / (val_ir + train_ir), 0, 1)` or a fixed conservative λ (e.g. 0.5). Cite: shrinkage estimators dominate raw sample optima under estimation error (Ledoit & Wolf 2004 logic applied to signal weights; DeMiguel et al. 2009).
4. Migration: on load (`IntelligencePassport.exists`), treat an existing passport with `val_ir <= 0` as **not calibrated** (return defaults) so the four bad passports on disk are neutralized without deleting user files. Bump `PASSPORT_VERSION` only if you change the schema.
5. Delete `.passports/passport_test__all__chop.json` and `passport_testu__all__chop.json` (test artifacts in the repo).

---

### A2 · P0 — Regime-conditioned calibration is illusory: the harvest is **regime-agnostic**

**Evidence:** `calibrate(universe, selected_index, regime_name, harvest, …)` (intelligence.py:373) keys the passport by regime, but `build_harvest` (intelligence.py:207) builds the panel from the **entire ~190-day trailing window regardless of regime**. The regime label is merely the regime detected on the analysis date (app.py:1940–1977). Two consequences:
- A passport labeled `BEAR` is trained mostly on non-BEAR days; the "different regimes reward different signals" thesis (README §Intelligence) is never actually implemented — the conditioning variable never enters the estimation.
- For a fixed (universe, index) and the same analysis window, calibrating under two different regime labels would produce **byte-identical weights** (same harvest, same TPE seed 42), stored under two different keys — the keying differentiates *when you happened to run*, not market state.

**Why (literature):** Regime-switching models condition parameter estimates on the regime of each observation (Hamilton 1989; Ang & Bekaert, *"International Asset Allocation with Regime Shifts"*, RFS 2002). Estimating a single unconditional parameter set and labeling it with the terminal regime is a category error — it inherits none of the regime-conditioning benefits and adds fragmentation cost (7 regimes × universes × indexes of sparse, mostly-overlapping passports).

**Directive (choose implementation 1; fall back to 2 only if per-date regimes are too expensive):**
1. **Condition the harvest on the regime.** In Phase 1.5 (app.py:1967+) and the Intelligence-tab manual calibrate (app.py:1635+), compute the rolling regime series once — `get_regime_history_series(all_hist, window_size=10, step=1)` is *already computed and cached* at app.py:2163 — and pass a per-date regime label into `build_harvest`. In `build_harvest`, tag each emitted row with `regime_t` (regime at date *t*, from the rolling series aligned by index; note `date` in the harvest is the integer window index `i`, so alignment is `regime_series[i + (window_size−1)]`-style — derive carefully from the actual offsets). Then in `calibrate`, filter `harvest = harvest[harvest["regime_t"].map(REGIME_MIX_MAP-style grouping) == group(regime_name)]` **before** the date-count gate. Because per-regime samples will be small, group regimes into the three existing families via `REGIME_MIX_MAP` (Bull / Chop / Bear mixes) rather than 7 raw regimes, and key passports by family. Keep `MIN_TOTAL_DATES = 20` as the gate — if the filtered panel is thinner than that, report `skipped: insufficient {family} history` (this will happen often, which is honest).
2. **Or drop the regime key entirely** (passport per (universe, index) only) and update all copy — weaker, but no longer misrepresents what is learned.

Update README/CHANGELOG copy accordingly — currently they promise per-regime learning that does not exist.

---

### A3 · P1 — IR computed on **overlapping 10-day forward returns** → autocorrelated ICs → inflated IR and unstable weight selection

**Evidence:** `build_harvest` emits one row per day with `fwd_ret = price[t+10]/price[t] − 1` for consecutive `t` (intelligence.py:221–246). Adjacent dates share 9 of 10 return days, so the per-date ICs entering `_ir_for_weights` (intelligence.py:113) are strongly serially correlated. `sd = ics.std(ddof=1)` then underestimates the true dispersion, inflating `IR = mu/sd`, and the 70/30 chronological split leaks: the last ~10 train dates' forward windows extend into the validation period.

**Why (literature):** Lo, *"The Statistics of Sharpe Ratios"* (FAJ 2002) — serial correlation biases IR/SR estimates upward; Newey & West (1987) HAC correction; López de Prado, *Advances in Financial Machine Learning* (2018), ch.7 — **purging and embargoing**: training observations whose label windows overlap the validation window must be removed.

**Directive:**
1. In `calibrate`, **purge** the boundary: drop the last `horizon` train dates (`train_dates = set(dates[:n_train − horizon])`) so no train label overlaps validation (embargo of `horizon` days).
2. In `_ir_for_weights`, replace the naive std with a **Newey–West (HAC) standard error of the mean IC with lag = horizon − 1**, i.e. `IR = mean(IC) / se_NW(IC) · √n` normalized consistently, or more simply compute ICs on **non-overlapping dates only** (`dates[::horizon]`) for both train and val. Non-overlapping sampling is the simpler, more robust option; take it, and lower nothing else — if the non-overlapping panel falls below `MIN_TOTAL_DATES/horizon` usable ICs, report the skip reason.
3. Keep the vectorized structure; both changes are a few lines.

---

### A4 · P0 — The 95-strategy layer contributes **zero information** to the final portfolio (weights discarded, candidate pool ≈ whole universe)

**Evidence:** In Phase 2 (app.py:2105–2116) each strategy's output is reduced to `aggregated_holdings[symbol] = {"price": price, "weight": 1.0}` — the strategy's computed weight is thrown away, first-seen price kept. Every strategy's `_allocate_portfolio` (strategies.py:152–198) returns **all input rows** (weights clipped to [1%,10%], never filtered); only CL1/CL2 drop rows via a data-quality gate (strategies.py:419). Therefore the union over 95 strategies is (effectively) *every symbol with valid data* — identical to skipping the strategy layer and scoring the raw universe by conviction. The "95 quantitative strategies" headline (README, landing page, tagline) describes compute that cannot change the output. It also costs runtime and carries ~7,500 lines of unmaintained code, much of it decorative (e.g. `ResonanceEcho`, `DivergenceMirage` compute cross-sectional correlations between daily and weekly values of the same indicator — statistically meaningless as "resonance"; strategies.py:3126–3275).

**Why:** An ensemble adds value only if member outputs are aggregated with information intact (majority vote, weight averaging, stacking — Breiman 1996; Grinold & Kahn, *Active Portfolio Management*, on combining signals). A union of near-complete sets is the degenerate ensemble: selectivity zero, transfer coefficient of the strategy layer zero.

**Directive (implement 1; it is the smallest change that makes the layer real):**
1. **Turn strategy output into an endorsement-count feature.** In the aggregation loop, count votes: `aggregated_holdings[symbol]["votes"] += 1` won't discriminate while strategies return everything, so make endorsement meaningful: for each strategy, take only its **top quartile by `weightage_pct`** (e.g. `port.nlargest(max(5, len(port)//4), 'weightage_pct')`) as that strategy's "conviction picks" before aggregating votes. Pass the vote count through `compute_conviction_based_weights` → `compute_conviction_signals` as a sixth signal `strat_signal` mapped to [−2,+2] by cross-sectional rank of votes (rank-percentile ×4 −2), and add `w_strat` to `DEFAULT_WEIGHTS` (even 1/6 split) and to the calibrator's simplex (`_softmax6`). Bump `PASSPORT_VERSION` (auto-invalidates old passports, mechanism already exists).
2. **And/or prune the library**: delete the ~60 decorative strategies (everything from `ResonanceEcho` through `TranscendentAlpha`, strategies.py:3126–4404, plus similar) keeping the ~30 with defensible logic (PR/CL/MOM families, DualMomentum, CrossSectionalAlpha, VolatilityRegimeTrader, etc.). Update all "95" copy. If the user insists on keeping the count, at minimum stop *claiming* the strategies drive selection.
3. Log per-strategy failures instead of `except Exception: pass` (app.py:2115) — at `console.warning` level with strategy name, and count them in `metrics.add_warning`.

---

### A5 · P1 — Median step-function dispersion creates a **13× weight cliff** between adjacent ranks

**Evidence:** portfolio.py:126–139 — every position strictly above the median conviction gets `×3.25` (Swing) while every one at/below gets `×0.25`: two names one conviction point apart (e.g. 62 vs 61) receive a 13:1 raw weight ratio; the bounds pass then pins the entire bottom half at the 1% floor. The result is a de-facto binary book (top half ≈ equal-heavy, bottom half = 1% stubs) while presenting itself as "conviction-proportional".

**Why (literature):** Grinold & Kahn: portfolio weights should be a **monotone, continuous** function of the signal to preserve the transfer coefficient; discontinuous transforms inject turnover and noise at the boundary (a one-tick conviction change reshuffles ~5% of the book). Rank-based power weighting or softmax-temperature weighting achieves the same concentration smoothly.

**Directive:** Replace the boost/penalty step in `compute_conviction_based_weights` with a continuous concentration transform, preserving the two style levels:
```python
# rank in (0,1], 1 = best; gamma controls concentration
r = conv_df['conviction_score'].rank(pct=True)
gamma = 3.0 if investment_style == "Swing Trading" else 2.0   # tune so top-decile tilt matches old intent
conv_df['adjusted_conviction'] = conv_df['conviction_score'] * (r ** gamma)
```
Keep `dispersion_params` as an escape hatch (interpret as `gamma` override). Update the README/CHANGELOG dispersion tables and the System-tab method card. Verify: resulting weights should decline smoothly from max to min with no cliff, and the same 1–10% bounds applied after.

---

### A6 · P1 — Position-bound feasibility is never checked: with **N < 10 positions the 10% cap is mathematically infeasible**

**Evidence:** The sidebar slider allows `num_positions` from 5 (app.py:2320–2327) with `max_pos_pct = 0.10` fixed (app.py:2328–2329). Σweights = 1 with each ≤ 0.10 requires N ≥ 10. The clip-renormalize loop (portfolio.py:160–164) ends on a **normalize**, so the final weights silently violate the advertised cap (5 positions → 20% each, displayed as if bounded 1–10%). Symmetric edge: N = 100 with min 1% forces exactly-equal weights.

**Directive:** In `compute_conviction_based_weights`, before the bounds loop, check feasibility: `n = len(conv_df)`; if `n * max_pos_pct < 1.0`, set `max_pos_pct = 1.0 / n` is wrong (that forces equal) — instead **relax the cap** to `max(max_pos_pct, 1.0/n + 1e-9)`… the correct minimal relaxation is `max_pos_pct_eff = max(max_pos_pct, 1.0 / n)`; if `n * min_pos_pct > 1.0`, relax `min_pos_pct_eff = min(min_pos_pct, 1.0 / n)`. Return the effective bounds in the DataFrame attrs or a tuple so the UI (System tab config table, app.py:966–967) can display the *effective* bounds with a caption when relaxed. Also fix the loop to end on a **clip** followed by an exact water-filling redistribution, or simply assert post-loop `weights.max() <= max_eff + 1e-6`.

---

### A7 · P1 — Indicator warm-up measured in **calendar days** where **trading days** are required → first ~⅓ of every panel has NaN MA200 (biases regime trend factor, harvest, and regime-history chart)

**Evidence:** `generate_historical_data` skips snapshots before `start_date + timedelta(days=MAX_INDICATOR_PERIOD)` — 200 **calendar** days (backdata.py:489) — but `ma200 latest` needs 200 **trading** days ≈ 290 calendar days. With the app's fetch window (`end − ((100+200)·1.5+30)d` ≈ 480 days, app.py:206–209), snapshots span the last ~280 calendar days, and the first ~90 calendar days (~60 trading days ≈ 30% of the panel) carry NaN `ma200 latest`/`ma90 weekly` etc. Effects: (a) `MarketRegimeDetector._trend` computes `(df["price"] > df["ma200 latest"]).mean()` where NaN compares False → early regime-history readings systematically skew bearish; (b) harvest rows lose `ma_signal` (0.0) for a third of dates → calibration downweights MA structurally, not empirically.

**Directive:** In backdata.py, change the skip rule to count **rows of the downloaded index**: replace the calendar check with `date_range = all_data.index.normalize().unique()` and start snapshots at `date_range[MAX_INDICATOR_PERIOD:]` (i.e., skip the first 200 *bars*), still bounded by `end_date`. Keep `_load_historical_data`'s fetch multiplier (×1.5 + 30) — it already over-fetches enough calendar days to supply 200 bars + lookback. In `MarketRegimeDetector._trend`/`_momentum` etc., compute participation ratios on **non-NaN rows only** (`sub = df.dropna(subset=["ma200 latest"]); (sub["price"] > sub["ma200 latest"]).mean()`), guarding against empty.

---

### A8 · P1 — `calculate_rsi` back-fills **warm-up NaNs with RSI = 100**

**Evidence:** backdata.py:94–96 — `rsi = rsi.fillna(100.0)` is intended for the `avg_loss == 0` (all-gains) case, but it also converts the first `period` warm-up NaNs (from `min_periods=period`) into hard 100s. Any consumer reading an early row sees a maximally-overbought signal that is actually "no data". Weekly RSI (14 weekly bars ≈ 98 calendar days) intersects the live snapshot window, so this leaks into real snapshots, the regime momentum factor, and PR-family strategies that blend `rsi weekly`.

**Directive:** Fill only the legitimate case:
```python
rs = avg_gain / avg_loss.replace(0, pd.NA)
rsi = 100.0 - (100.0 / (1.0 + rs))
all_gain = (avg_loss == 0) & avg_gain.notna() & (avg_gain > 0)
rsi = rsi.where(~all_gain, 100.0)          # only true all-gain windows become 100
# leave warm-up NaNs as NaN — downstream already treats NaN as "no signal"
```

---

### A9 · P1 — `LiquidityOscillator` warm-up contamination and zero-range garbage via `fillna(1.0)`

**Evidence:** backdata.py:61 — `safe_vol_ma = vol_ma.replace(0, NA).fillna(1.0)`: during the first `length−1` bars `vol_ma` is NaN → treated as 1.0 → `spread · volume / 1.0` produces values ~10⁶× scale, which then sit inside the next 20-bar rolling means (`vwap_spread`, `price_impact`), contaminating bars ~20–40 of every symbol's oscillator. Similarly backdata.py:70 — a zero high-low *source range* becomes 1.0 → `oscillator = 200·(x−lo)/1 − 100` explodes for flat-price periods instead of being undefined.

**Directive:** Replace both `fillna(1.0)` with NaN-propagation: `safe_vol_ma = vol_ma.where(vol_ma > 0)` (NaN elsewhere) and `safe_range_value = range_value.where(range_value > 0)`. NaNs then flow through the rolling ops and are dropped downstream exactly like other warm-up NaNs (the codebase already relies on this convention — see compute_volume_profile docstring, backdata.py:133–135). Verify: oscillator values in bars 20–45 change materially for any symbol; snapshots earlier than full warm-up now show NaN osc instead of noise.

---

### A10 · P1 — `compute_volume_profile` crashes on NaN high/low with positive volume; and is an O(n·window) pure-Python hot loop

**Evidence:** backdata.py:168–179 — the binning loop guards `v` (volume) but not `wl[j]`/`wh[j]`: `int(np.floor((nan − lo)/step))` raises `ValueError: cannot convert float NaN to integer`. `generate_historical_data`'s ticker loop catches only `(pd.errors.DataError, KeyError, IndexError)` (backdata.py:478), so one bad bar in one symbol aborts the entire data phase. yfinance does ship rows with close/volume present but high/low NaN. Rows are pre-filtered only on `close, volume` (backdata.py:471). Separately, the double loop (`for end … for j in range(window)`) is ~13M Python iterations for a 500-symbol universe — a large share of the 20–40s runtime.

**Directive:**
1. Correctness: inside the inner loop, `if not (np.isfinite(v) and v > 0 and np.isfinite(wl[j]) and np.isfinite(wh[j])): continue`. Also compute `lo/hi` with `np.nanmin/np.nanmax` over **finite** rows only (already nan-aware — fine).
2. Robustness: broaden the ticker-loop catch to include `ValueError`, and log the symbol via `console.warning` instead of silent `continue`.
3. Performance: vectorize the per-window histogram. Precompute per-bar bin ranges once per symbol *only when the window's [lo,hi] is stable* is not possible (range changes per window) — instead, use the standard incremental approach: for each `end`, the window shifts by one bar; recompute only when `lo/hi` change, else update `hist` by subtracting the departing bar's slice and adding the arriving bar's. Fall back to a numba/`np.add.at` formulation if simpler: per window, `np.clip(((bars_lo − lo)/step).astype(int), 0, bins−1)` etc. and `np.add.at(hist, slice_indices, …)`. Target ≥10× speedup; validate by comparing `poc/vap/va_pos` on a fixture symbol before/after (must be identical within float tolerance).

---

### A11 · P1 — Benchmark resolution is broken for non-India universes: **US/Commodity/Crypto portfolios are benchmarked against NIFTY 50**

**Evidence:** `analytics.resolve_benchmark` (analytics.py:75–88): `_INDEX_BENCHMARKS` contains only NIFTY names, so `selected_index = "S&P 500"` misses; the fallback loop tests `key in universe.lower()` with keys `"us_sp500" / "us_nasdaq100" / "us_dow"` — but actual universe strings are `"US Indexes"`, `"Commodities"`, `"Crypto"` (universe.py:76–84), so **no key ever matches**. `_UNIVERSE_BENCHMARKS` is dead code; every non-NIFTY book silently gets `^NSEI`. All Analytics-tab alpha/beta/capture/tracking numbers for those universes are meaningless.

**Directive:** Rewrite `resolve_benchmark`:
```python
_INDEX_BENCHMARKS.update({
    "S&P 500":    ("^GSPC", "S&P 500"),
    "NASDAQ 100": ("^NDX",  "NASDAQ 100"),
    "DOW JONES":  ("^DJI",  "Dow Jones"),
})
_UNIVERSE_BENCHMARKS = {
    "US Indexes":  ("^GSPC",   "S&P 500"),
    "Commodities": ("DBC",     "Bloomberg Cmdty proxy (DBC)"),
    "Crypto":      ("BTC-USD", "Bitcoin"),
    "Currency":    ("UUP",     "USD Index proxy (UUP)"),
    "ETF Index":   ("^NSEI",   "NIFTY 50"),
    "India Indexes": ("^NSEI", "NIFTY 50"),
}
```
Match `universe` by exact key (not substring). Keep NIFTY 50 as final default. Also make `RISK_FREE_RATE` benchmark-aware (6.5% for Indian books, ~4.5% for US) or at minimum label the Sharpe card's "Rf = 6.5%" subtext dynamically (app.py:1433).

---

### A12 · P1 — Rendering context drifts from run context: tabs recompute conviction/benchmark with **live sidebar state**, not the state the portfolio was curated under

**Evidence:** `_intel_context()` (app.py:165–180) reads `selected_universe / selected_index / regime_result_dict` **live**. After a run, changing the sidebar date or universe (a) recomputes the sidebar regime (app.py:2272–2286) and (b) mutates `selected_universe/index` — the Portfolio/Position-Guide tabs then re-score the *old* portfolio with the *new* scope's passport (app.py:322–327, 501–506), and Analytics resolves the *new* universe's benchmark for the *old* book (app.py:1298–1299). Displayed conviction/weights can no longer reconcile with the curated `weightage_pct`.

**Directive:** At the end of a successful `_run_analysis`, snapshot the run context: `st.session_state.run_context = {"universe": _u, "selected_index": _idx, "regime_name": regime_name, "mode": _mode, "anchor_date": selected_date_display}`. Change `_intel_context()` to return `st.session_state.run_context` when a portfolio exists, falling back to live sidebar state only pre-run. Analytics should read `anchor_date` from `run_context` too (instead of live `selected_date`, app.py:1315). This makes results immutable views of the run, as any research terminal should be.

---

### A13 · P2 — Volatility factor uses **absolute BBW thresholds** calibrated for equities → mis-reads FX (always "SQUEEZE") and crypto (chronically "ELEVATED/PANIC")

**Evidence:** regime.py:377–394 — cutoffs 0.08/0.12/0.15 on mean Bollinger-width. Typical FX BBW ≈ 0.01–0.03; typical crypto ≈ 0.2–0.6. For those universes the factor is a constant, not a signal.

**Why:** Bollinger's band-width regime work and the volatility-regime literature use *relative* band-width (percentile vs its own history), precisely because raw width is scale- and asset-class-dependent.

**Directive:** Compute the BBW series over the full window (already available — `window` holds up to 10 snapshots; better, thread the longer `all_hist` in), then classify by **percentile of current BBW within the trailing distribution**: `SQUEEZE if pct < 0.2 and falling; PANIC if pct > 0.9 and rising; ELEVATED if pct > 0.75; else NORMAL`. Minimum change: pass the full historical BBW mean-series into `_volatility` (the detector already receives only 10 snapshots — extend `detect()` to accept an optional longer context or compute percentiles across the 10-snapshot window as a stopgap with wider bands).

---

### A14 · P2 — Currency (and partly Commodities) universes are structurally non-functional: volume-dependent signals all die

**Evidence:** FX tickers report zero volume on Yahoo; `LiquidityOscillator.calculate` correctly returns empty (backdata.py:54–55), and `compute_volume_profile` yields NaN (no volume). Consequence: osc/zscore/vap signals are 0 for every symbol; conviction collapses to RSI+MA only under renormalized-but-not-renormalized weights (dead weights just multiply 0), so scores cluster ~50±small and top-N selection is near-arbitrary — yet the UI presents it with the same confidence.

**Directive:**
1. In `compute_conviction_signals` / `_signals_from_row`, track which signals were *available* per row and **renormalize the weights over available signals** (`w_eff = w_i / Σ_available w_j`) so a two-signal asset still spans the full [−2,+2] range instead of being compressed toward 0.
2. In the UI, when >50% of the universe lacks osc/vap coverage, render a warning card ("volume-dependent signals unavailable for this universe; conviction uses RSI/MA only").
3. Optionally gate `Currency` behind that warning at selection time (universe.py:591).

---

### A15 · P2 — `MIN_XSECT = 5` is too thin for stable Spearman ICs

**Evidence:** intelligence.py:47. A 5-name cross-section gives Spearman IC a standard error of ~0.5 — pure noise entering the IR numerator/denominator.

**Why:** Grinold & Kahn's IC framework assumes breadth; standard practice requires ≥20 names per cross-section for IC to carry signal (IC standard error ≈ 1/√(n−1)).

**Directive:** Raise `MIN_XSECT` to 10 (hard floor) and add a breadth-aware note in the skip reason when dates are dropped for thin cross-sections. For the 30-name ETF universe this still passes; for tiny custom lists calibration will now honestly refuse.

---

### A16 · P2 — Passport import path: no simplex validation, wrong error text, and defaults-exports round-trip into "calibrated"

**Evidence:** app.py:2444–2465 — imported weights are only checked for key presence; a file with `w_rsi: 37` imports fine and corrupts scoring. Error message omits `w_vap` (app.py:2452). `IntelligencePassport.save` hard-codes `is_calibrated: True` (intelligence.py:335–349), so exporting a *defaults* profile (`is_calibrated: False`, app.py:2473–2480) and re-importing it marks the scope as Calibrated with IR 0.0.

**Directive:** In the import handler: coerce floats, require each `0 ≤ w ≤ 1` and `abs(sum − 1) < 0.02` (then renormalize exactly); reject otherwise with a precise message listing all five keys. Respect the payload's `is_calibrated`: if false/absent-with-zero-IRs, refuse import with "this file contains default weights — nothing to import". Add an optional `is_calibrated` parameter to `save()` defaulting True.

---

### A17 · P2 — Docstring/comment sign errors in the volume-profile feature (train/serve/docs must agree)

**Evidence:** `compute_volume_profile` docstring says "**Negative** = price trades at a *discount*" (backdata.py:117–121) but the code computes `vap = −(price − poc)/va_half` → discount is **positive** (backdata.py:208–214), which is what regime/intelligence assume. portfolio.py:110 comment says "va_pos < 0 = below the value area" — actually below the value-area **midpoint**.

**Directive:** Fix both docstrings/comments to match the code (discount → vap > 0; va_pos measures position vs VA midpoint, clipped [−1,1]). No code change.

---

### A18 · P2 — `analytics.compute_metrics` mislabels un-annualized short-window return as "CAGR", and alpha mixes annualization conventions

**Evidence:** analytics.py:189–199 — for `20 ≤ n_days < 252`, `ann_factor = min(252/n,1) = 1`, so "CAGR" = raw period return; for `n < 20` it linearly extrapolates (`total · 252/n`), which can print absurd three-digit "CAGR" after a 4-day window. `alpha` (analytics.py:285–288) compares this hybrid `p_cagr` against a `b_cagr` computed under the same hybrid but over possibly different day counts.

**Directive:** Keep behaviour-preservation out of it — this is now the only consumer. Use one convention: if `n_days ≥ 60`, annualize geometrically (`(1+r)^(252/n) − 1`) for both portfolio and benchmark over the **aligned** window; if `n_days < 60`, hide the CAGR/Alpha cards entirely and show only period return (annualizing sub-quarter windows is statistically indefensible — cf. CFA Institute GIPS guidance against annualizing periods < 1 year). Update the metric-card layer (app.py:1424–1446) to drop those two cards under the threshold.

---

### A19 · P3 — Conviction default of 50 for unscored symbols competes with genuinely-scored names

**Evidence:** regime.py:626–627/710–711 — symbols missing from `current_df` (or with all-NaN indicators) receive conviction 50 and can outrank real 40-conviction names into the top-N.

**Directive:** In `compute_conviction_based_weights`, exclude candidates whose row produced *no* signals (all five raw signals absent — expose a `signals_available` count from `compute_conviction_signals`) before top-N selection; they are unpriceable by the model. Keep the 50-fill for display-only paths.

---

## PART B — INFRASTRUCTURE, DATA & CORRECTNESS

### B1 · P1 — `get_index_stock_list` (India) is **uncached** and fires warm-up + API HTTP requests on every sidebar rerun

**Evidence:** universe.py:358 has no `@st.cache_data` (unlike F&O and US paths); the sidebar bottom info block calls `resolve_universe` on **every rerun** (app.py:2511), and `_run_analysis` + `_load_historical_data` resolve again. For India Indexes each widget interaction can trigger 2–4 NSE requests with 10–15s timeouts — the dominant cause of sidebar sluggishness.

**Directive:** Wrap the India path in the same cached-inner/uncached-outer pattern already used for US indexes (universe.py:434–445): cached inner raises on failure (so failures aren't memoized), `ttl=3600`. Also cache-resolve once per rerun: in `main()`, resolve the universe **once** into a local and pass it to both the info block and `run_params` instead of re-calling.

### B2 · P2 — `verify=False` on NSE archive requests

**Evidence:** universe.py:276, 407–408. Disables TLS validation (MITM exposure for the machine placing real orders downstream via Broker Sync) and spams `InsecureRequestWarning`.

**Directive:** Default to `verify=True`; only on `requests.exceptions.SSLError` retry once with `verify=False` wrapped in `warnings.catch_warnings()` + a `console.warning("NSE archives TLS verification failed — fell back to unverified")`. Never unverified-by-default.

### B3 · P2 — Wikipedia "NIFTY 200" fallback returns the first 200 **alphabetical** rows of the NIFTY 500 table

**Evidence:** universe.py:332–339 — the NIFTY 500 Wikipedia constituent table is not ordered by market cap, so "top 200 of NIFTY 500" is a fiction; the produced universe is systematically wrong (A–M tilt).

**Directive:** Remove the NIFTY 200 fallback (return the honest failure message), or fall back to NIFTY 100 (50+Next 50 union) labeled as such. Never silently fabricate an index.

### B4 · P2 — Three divergent "default ETF universe" definitions

**Evidence:** `universe.ETF_UNIVERSE` (CHEMICAL.NS, GROWWPOWER.NS…), `backdata.get_default_universe` fallback (SENSEXIETF.NS, BANKIETF.NS…), and `symbols.txt` are three different 30-name lists.

**Directive:** Make `universe.ETF_UNIVERSE` the single source. `backdata.get_default_universe` should import it and, on ImportError, read `symbols.txt` (keep that file in sync with universe.py). Delete the hard-coded fallback list in backdata.py.

### B5 · P2 — Global `metrics` singleton is shared across Streamlit sessions

**Evidence:** metrics.py:250 module-level instance; Streamlit serves all sessions from one process, so two concurrent users interleave phases/errors/run IDs (also `_run_analysis` resets it, app.py:1879).

**Directive:** Store the tracker in `st.session_state`: `get_metrics()` becomes a thin wrapper that lazily creates `st.session_state["_metrics"] = ExecutionMetrics()` when running under Streamlit (guard `streamlit.runtime.exists()`), falling back to the module global for CLI use (backdata standalone).

### B6 · P3 — `RetryWithBackoff` imported but never used; circuit breaker protects a single un-retried call

**Evidence:** backdata.py:32 imports it; nothing applies it. A single yfinance transient therefore raises immediately (breaker counts it, but the user's run dies).

**Directive:** Apply `@RetryWithBackoff(max_retries=2, initial_delay=2.0, exceptions=(Exception,))` **inside** the circuit-protected `download_data` (retry inside, breaker outside — so the breaker sees only exhausted failures). Or remove the import if retry is unwanted; don't keep dead machinery.

### B7 · P3 — Blocking `time.sleep(1.5)` in the run path; silent strategy-failure swallowing

**Evidence:** app.py:2189–2191; app.py:2115–2116.

**Directive:** Delete the sleep (the toast + rendered results are feedback enough; the progress card can simply be emptied). Replace `except Exception: pass` per A4.3.

### B8 · P3 — Broker Sync writes `quantity = 0` into templates for matched zero-unit holdings

**Evidence:** app.py:1082–1086 builds `qty_map` including units = 0; `_sync_broker_json` writes any match (app.py:1040–1049). A template row with an existing manual quantity gets zeroed while the method card claims non-matching instruments are "untouched" — a matched-but-zero holding is a third, undocumented case.

**Directive:** Product decision, but implement the safe default: only write `quantity` when `units > 0`; report matched-but-zero symbols in the results table as `SKIPPED (0 units)`. Mention in the method card.

### B9 · P3 — Repo hygiene

**Evidence:** `core/Pragyam-main.zip` (a zip of the repo inside the repo), `__pycache__/` trees, test passports, `.devcontainer` post-attach disabling XSRF/CORS.

**Directive:** Delete `core/`, all `__pycache__`, the two `test*` passports; add a `.gitignore` (`__pycache__/`, `.passports/*.json` optionally, `data/`). Keep devcontainer flags (dev-only) but comment why.

---

## PART C — UI / UX

### C1 · P1 — The dark/light theme toggle is **dead**: it runs inside a sandboxed, zero-height iframe

**Evidence:** `render_theme_toggle` uses `components.html(…, height=0)` (ui/components.py:298–358). `components.html` renders into an isolated iframe: (a) height 0 makes the button invisible/unclickable; (b) even if visible, the script sets `data-theme` on the **iframe's** `documentElement`, not the app's, so the `[data-theme="light"]` rules in theme.css (theme.css:92, 168, 188, 3090+) can never activate. Light mode is unreachable; ~hundreds of CSS lines are dead.

**Directive:** Either (preferred, simplest) remove the toggle + the `[data-theme="light"]` CSS block and commit to the Obsidian dark identity; or implement it properly: the script must target `window.parent.document.documentElement` (Streamlit component iframes are same-origin, so this works) and the container needs real height / `position:fixed` styling injected into the **parent** via the main CSS, with the iframe kept 40px tall for the button. If keeping light mode, audit the light-token coverage (most components hard-code dark rgba values inline — e.g. the HTML tables in app.py:344–428 — so light mode would be broken anyway; this argues for removal).

### C2 · P1 — Stale v9.0 copy contradicts the v9.3 fixed-weight regime detector across the UI

**Evidence:**
- Regime tab method card: "In Intelligence mode the eight factor weights are **learned** per (universe, index); in Standard mode a fixed default mix is used" (app.py:866–869) — false since v9.3.
- Regime tab pill: `f'8-factor composite · {_rmode_lbl} weights'` renders "Intelligence weights" (app.py:853–860) — the composite never varies by mode.
- Landing REGIME card: "regime-calibrated composite scoring" (app.py:1530) and lists a "Neutral" regime that doesn't exist (app.py:1532).
- Progress copy: "Optimize weights" on the landing prompt (app.py:1565) is fine for conviction but ambiguous.
- README passport schema example shows 4 weights (missing `w_vap`) at README:202–218.

**Directive:** Sweep all of the above to state: regime = fixed 8-factor weights, always; only the 5-signal conviction blend is calibrated. Fix the regime hierarchy list (Strong Bull · Bull · Weak Bull · Chop · Weak Bear · Bear · Crisis). Update the README schema example to five weights.

### C3 · P2 — Dead/broken component layer

**Evidence:** `render_export_button_row`, `render_collapsible_section(_close)`, `render_info_box`, `render_warning_box`, `render_chart_skeleton` are imported (app.py:56–64) but never called. `render_collapsible_section` opens 3 divs and closes 2, and the wrap-content-between-two-`st.markdown`-calls pattern cannot work in modern Streamlit (each markdown block is an isolated fragment). `render_export_button_row` embeds raw SVG in a `st.download_button` label, which renders as literal text.

**Directive:** Delete the five unused components and their imports. If a collapsible is ever needed, use `st.expander`.

### C4 · P2 — Sidebar regime auto-detection blocks the UI with a network fetch on every date/universe change

**Evidence:** app.py:2272–2286 — changing the date immediately triggers `_detect_regime_cached` → `_load_historical_data` → a full yfinance multi-symbol download inside a sidebar spinner. On cache miss this is 10–30 s of frozen sidebar just for browsing dates.

**Directive:** Make the sidebar regime card **lazy**: on change, render the card in a "stale" visual state (dimmed, "regime not computed for this date — will compute on Run") with a small "Refresh regime" button that triggers the fetch explicitly. Keep auto-compute only when the panel is already in cache (probe with a sentinel: wrap the cached call in a try with a `st.cache_data`-hit check is not natively available — simplest is: auto-compute only if `symbols_key`+date matches the last run's key, else lazy).

### C5 · P3 — Minor UI defects

1. System tab requests icon `"settings"` which doesn't exist in `ICONS` → silently falls back to "chart" (app.py:961; ui/components.py:57–60). Add a gear icon or use `"cpu"`.
2. `_render_results` footer and `_render_footer` duplicate the same footer HTML (app.py:1855–1864 vs 2204–2215) — extract one helper.
3. Chinese characters in a charts.py comment ("警示", charts.py:32) — replace with English.
4. Import-profile expander error message lists 4 keys (see A16).
5. `initial_sidebar_state="collapsed"` while the landing card instructs "Configure via the Sidebar" (app.py:125, 1563) — start expanded on first load.
6. Portfolio-tab caption says weights are "regime-calibrated" even in Standard mode (app.py:447–450) — make the caption mode-aware.
7. `_render_landing_page` STRATEGIES card claims "Universe: Nifty 500 + F&O symbols" regardless of actual selection (app.py:1546) — make it generic or dynamic.

---

## PART D — DOCUMENTATION / TRUTH-IN-LABELING

### D1 · P2 — README/CHANGELOG claims that no longer (or never) held

1. "Aggregate candidate holdings (~200–400 symbols)" — reflects data-valid universe size, not strategy selectivity (see A4). Reword once A4 lands.
2. README §Conviction table: "Discount: > 2 → +2" wording matches code, but the *backdata* docstring contradicts (A17).
3. README Troubleshooting "for NIFTY 500 / S&P 500 the calibration trial loop is still <10 s" — verify post-A3 (non-overlapping sampling reduces panel size; still true, but re-time).
4. CHANGELOG v9.2 claims a "timeframe selector (1M…MAX / YTD)" in Analytics — the shipped tab is anchor-dated with **no** selector; `analytics.TIMEFRAMES` is dead code. Either restore the selector (offer both: anchored to run date *and* fixed windows) or delete `TIMEFRAMES` and fix the changelog/README table.
5. Version-history table in README says v9.0.0 introduced "eight-factor regime detector with **learned** weights" then v9.3.0 reverted — keep, it's history — but the *Features* table row "Regime Detection … (drives passport key + display)" should note fixed weights (it does). OK.

### D2 · P3 — `_slug` regime keys: document that `regime="UNKNOWN"` calibrations land in `passport_<u>__<i>__unknown.json`

A failed regime detection still allows manual calibration under the UNKNOWN scope (app.py:1580–1651). Either block calibration when regime is UNKNOWN (recommended: the scope is meaningless) or document the behavior.

---

## SUGGESTED EXECUTION ORDER (dependency-aware)

1. **A1 + A3** (calibration integrity: validation gate, shrinkage, purge/non-overlap) — pure `intelligence.py`, self-contained, highest value.
2. **A7 + A8 + A9 + A10.1–2** (data-layer correctness: trading-day warmup, RSI fill, oscillator warmup, VP crash) — changes every downstream number; do before re-calibrating anything.
3. **A2** (regime-conditioned harvest) — builds on 1–2.
4. **A4** (strategy layer: endorsement votes or prune) — coordinate `PASSPORT_VERSION` bump with A1/A2 so passports invalidate once.
5. **A5 + A6** (portfolio construction: continuous dispersion, bounds feasibility).
6. **A11 + A12 + A18** (analytics correctness: benchmarks, frozen run context, CAGR labeling).
7. **B1–B8** (infra: caching, TLS, universes, singleton, retries, broker-sync zeroes).
8. **C1–C5, D1–D2** (UI truth + dead code + copy sweep).
9. Delete stale passports, regenerate on the fixed engine, re-verify the Intelligence tab shows honest skip/reject reasons.

**Regression checklist after each stage:** run a full ETF-Index analysis on a fixed historical date; assert (a) portfolio is non-empty with bounds satisfied, (b) conviction column equals Position-Guide display, (c) calibration either saves with val_ir > 0 ≥ defaults or reports a reasoned skip, (d) Analytics benchmark name matches universe, (e) no NaN/100-RSI rows in the first 30 panel dates.
