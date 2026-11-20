# Pragyam · Release Notes

The narrative behind the major releases, newest first. The [changelog](../CHANGELOG.md) is the
complete record.

## What changed in v13

v13 removes the two styles that read the tape — the **Conviction-Value Grid (CVG)**
and **Managed Momentum (MMOM)** — and everything that existed only to feed them.
The styles are back to Equal Weight, ERC and HRP.

- **Gone from the product:** `cvgrid.py`, `pragati.py`, `samanvaya.py` and
  `intraday.py` (the conviction and value tapes, the 3 × 3 state map, the macro
  hedge and the intraday ladder); the macro-driver download; the eight-year tape
  history the panel fetch carried for them; and Managed Momentum's long close
  history (`backdata.fetch_close_history`). A run now downloads only the estimation
  window it uses.
- **Gone from the screen:** the State / Push / Conv / Value tape columns, the
  Conviction-Value Map, the watchlist, the Momentum Overlay strip, the grid and
  overlay rows in the run log and the System tab, and both styles' comparison books.
- **Gone from `research/`:** the scripts whose subject was CVG or MMOM, or that
  cannot run without them (`audit_cvg.py`, `audit_mmom.py`, `cvg_*.py`,
  `mmom_ship.py`, `style_search_holdout.py`, and the style-search candidates built
  on the grid's weights). The shared harness (`style_blends.py`, `style_search.py`,
  `style_search_pit.py`) now builds EW, ERC, HRP and HRP+EW only, so `audit_hrp.py`
  and the ERC and HRP figures below still reproduce.
- **Unchanged:** Equal Weight, ERC and HRP build exactly the books they built in
  v12.2.1, through the same pipeline. The CHANGELOG keeps the record of both
  removed styles and every figure measured on them.

## What changed in v12.2

*(The parts that still apply after v13; the CVG and Managed Momentum notes are in
the CHANGELOG.)*

v12.2 is an audit release. HRP — with CVG and Managed Momentum, since removed — was
audited by a committee: three readers per style, a chair who merged their findings
and pre-registered the opportunities, a skeptic who tried to refute every bug on
real data, and a tester who ran the opportunities through the product code
(`research/audit_hrp.py`). Every reported bug reproduced and is fixed; the style
changes the audit measured were decided on that evidence, and the release diff was
then reviewed adversarially itself. Every figure in this README is re-measured on
the v12.2 code, with the research snapshots regenerated under it.

**Data every style reads.**
- yfinance leaves some Indian demergers and mis-dated splits unadjusted (BAJAJFINSV
  −64% and −93% in 2008, ADANIENT 2015-06-03, TMPV 2025-10-14, TRENT 2026-01-01). An
  Indian listing's ≥ 30% fall (or a doubling) on a ≥ 30% overnight gap is back-adjusted in
  the estimation panel (`backdata.corporate_action_gaps`); a move that
  reverses a recent spike is a bad print and is unpriced instead. The research return
  panel is repaired the same way (`style_blends.repair`). Every style had been scored
  on those fake losses: Nifty 50 Equal Weight's full span rises from 19.69 to 20.10 %/yr.
- A run during market hours no longer reads today's still-forming bar;
  exchange-holiday prints are dropped before the indicators.
- Whole-share rounding left up to ~14% of a ₹5L, 50-name book in cash. The leftover
  is now spent on the holdings furthest below target, never past the cap.

**HRP and ERC.** Dead quotes and closes ≤ 0 are unpriced before returns, gaps are no
longer padded into zero returns, a frozen or near-riskless column is left out (named in
the run log), and the coverage rule is 95%: at 80%, one late listing cut every name's
window by up to a fifth, and NIFTY SMLCAP 250 got no HRP or ERC book at all. ERC's
Ledoit-Wolf shrinkage gains the ρ term it was missing. **HRP now averages its fits on
three staggered windows** (ending 0, 21 and 42 sessions back): most of its turnover
was its leaf order re-drawn by estimation noise. Against the single fit: Nifty 50
+0.46 / +0.19 / +0.12 %/yr by era, Dow 30 +0.12 / +0.17 / +0.38, point-in-time Dow
+0.37, ETF book −0.25, none significant — and turnover down 40% (Nifty 1.23 → 0.72x/yr).
HRP is described as what it measures: the deepest volatility and drawdown cut of the
styles, at about 1.7x ERC's turnover.

**Not shipped.** A data-derived HRP dendrogram orientation (it would end the book's
dependence on the universe file's order) failed its bar inside the staggered average
(ETF −1.68 %/yr, t −2.1). Every backtest still rebalances on the first trading day (see
[Known limits](GUIDE.md#known-limits)).

## What changed in v12 and v12.1

v12 added the Conviction-Value Grid and v12.1 added Managed Momentum on top of it.
Both were removed in v13; see the CHANGELOG for what they were and how they measured.

## What changed in v11, and why

v11 removed the conviction scoring engine, the 95-strategy library and the
per-regime weight calibration — roughly 9,900 lines. Not a stylistic clean-up;
each was removed after measurement.

| Removed | Measured finding |
|---|---|
| Conviction blend | No cross-sectional predictive power on this universe: IC ~0.00–0.04, sign unstable across horizons. A top-quintile-by-conviction book *underperformed* a bottom-quintile one. |
| 95-strategy library | Mean pairwise return correlation **0.972** — an effective **1.03 independent strategies out of 92**. Ninety-five engines producing one opinion. |
| Per-regime calibration | Could not clear its own significance gate on realistic panels. |

The strategy redundancy is **structural, not authorial**. Long-only baskets of
assets that are themselves 52% correlated cannot decorrelate: 60 random
long-only portfolios of the same ETFs measure 0.962 correlated even when their
weights share nothing. No rewrite of the strategies would have fixed it.

### The reasoning behind the replacement

Grinold's Fundamental Law bounds excess return from *forecasting* at
`IR = IC × √BR × TC`. On 30 ETFs at ρ = 0.517 — only ~1.9 effective independent
bets, with measured transfer coefficient 0.88–0.92 — that ceiling is about
**1%/yr**, however good the signal.

Covariance-based allocation is not bound by it, because it forecasts nothing.
Covariance is estimable from a few hundred observations in a way expected
returns are not (López de Prado 2016, 2019).

### What it actually delivers — stated plainly

Measured on the shipped module across two **genuinely disjoint** periods:

```
                   CAGR    vs 1/N     Vol   Sharpe    MaxDD
A 2023-12..2024-12
       Equal      27.30%   +0.00%  13.81%    1.83    -6.89%
       HRP        26.33%   -0.96%  11.28%    2.14    -4.50%
B 2025-01..2026-07
       Equal      12.31%   +0.00%  12.97%    0.96    -7.07%
       HRP        11.08%   -1.23%  10.38%    1.07    -6.15%
```

**HRP does not beat equal weight on return — it loses ~1%/yr in both periods.**
What it consistently delivers is a ~20% cut in volatility and drawdown, and a
higher Sharpe. It is a volatility-reduction overlay. **Do not size it expecting
excess return.**

If you are maximising absolute return without leverage, Equal Weight is the
correct choice and ships as a first-class style.

> Note on method: an earlier nested-window test (2024+ fully contained in 2023+)
> reported HRP *beating* equal weight. That does not survive a disjoint split.
> Every figure above uses non-overlapping periods.
