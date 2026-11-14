# PRAGYAM (प्रज्ञम) — Portfolio Intelligence

**Version:** 13.0.11
**Author:** @thebullishvalue
**License:** Proprietary (See LICENSE file)

Portfolio curation over a chosen universe — the ETF book by default, or an index,
commodity, crypto or custom list, in three styles: Equal Weight, Equal Risk
Contribution and Hierarchical Risk Parity. None of them forecasts a return; they
are built to **spread risk**, not to predict returns.

---

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
[Known limits](#known-limits)).

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

---

## Architecture

```
app.py          Streamlit UI + 2-phase pipeline
nco.py          Equal Weight / ERC / HRP curation
regime.py       8-factor regime detection (context only)
backdata.py     yfinance fetch + indicator panel
analytics.py    portfolio-vs-benchmark metrics
charts.py       Plotly builders
universe.py     universe resolution
research/       offline evidence harness (not imported by the app)
```

**Pipeline:** Phase 1 data + regime → Phase 2 curation. About 2s once the panel
is cached.

---

## Usage

```bash
streamlit run app.py
```

**Sidebar:** Analysis Date · Portfolio Style · Universe · Capital · Positions.

**Styles**

| Style | Family | Behaviour |
|---|---|---|
| **Equal Weight** *(default)* | baseline | Identical `1/N` per holding. The default because nothing beat it reproducibly; see below. Lowest turnover of any style. |
| **Equal Risk Contribution** | preservation | Solves so every holding contributes the same share of portfolio variance. The preferred risk-reduction style on return: beats HRP on the any-date hit rate in 6 of 6 cells across two stock universes, at about 0.6x its turnover. In-repo (every name held, net, 2007-26): −0.49 %/yr against Equal Weight on Nifty 50 and −1.27 on Dow 30, at volatility 20.1 / 15.3 against 22.2 / 16.5. |
| **Risk Parity (HRP)** | preservation | Clusters by correlation distance, then splits capital by recursive bisection on cluster variance. Inverts no matrix. Averaged over three staggered estimation windows (v12.2). The deepest volatility and drawdown cut of the styles (volatility 18.8 / 14.2, max drawdown −49.3% / −36.4% on Nifty 50 / Dow 30), at −0.54 / −2.47 %/yr against Equal Weight and about 1.7× ERC's turnover. Measured holding every name: at 30 of 50 positions it is the 30 lowest-variance names (19.11 %/yr against 19.56). |

Every style travels the identical pipeline — same eligibility filter, same
clustering diagnostics, same risk decomposition — so any difference on screen is
the allocator and nothing else.

### Why Equal Weight is the default

A 36-candidate allocator search across three universes (this ETF book, Nifty 50
and Dow Jones 30; 14 years and 176 rebalances on the stock universes) found **no
method with a reproducible return improvement over `1/N`**:

```
                    ETF book    NIFTY 50    DOW 30     turnover/yr
  Equal Weight        —           —           —          0.12x
  ERC               +0.26%      -0.51%      -1.48%       0.26x
  Risk Parity       -1.33%      -1.08%      -2.94%       1.31x
```

What *does* reproduce is the ordering **among the risk-reduction styles**: ERC
beats HRP on the any-date hit rate in every cell tested, on both stock
universes, at a fifth of the turnover in that search (about 0.6x on the in-repo
harness since v12.2's staggered HRP, `research/style_search.py`, where HRP is the
deeper volatility and drawdown cut). That is a real improvement to the risk
leg — which the rest of this README has always said is the leg that reproduces.

**Position-count contract.** Every shipped style returns exactly the number of
positions you select. Max Diversification was evaluated, measured well on
lump-sum risk metrics, and **withdrawn anyway**: it is a corner-solution
optimiser that drives most weights to exactly zero, so it returned 10 holdings
when 15 were requested. A style that silently re-decides how many positions you
hold is not a weighting method. `nco_positions_short` and `nco_short_cause` now
record any shortfall and distinguish "the eligible universe ran out" (a data
condition) from "the allocator zeroed names" (a defect).
`nco_positions_unfunded` counts, for any style, the rows whose weight buys less
than one share at your capital (an expensive name such as MARUTI on a 50-name
book at ₹10L); a notice names them.

> **Correction (v11.1).** An earlier build of this analysis reported ERC and an
> ERC+momentum tilt beating Equal Weight in 98–100% of five-year SIP streams.
> That was an artifact of a defect in the ERC solver, which renormalised inside
> its descent loop and so converged to something between inverse-volatility and
> equal-risk. Against a corrected solver the result inverted to **0 of 115**.
> The momentum style was withdrawn before release; the fixed solver ships. See
> `research/README.md`.

**Result tabs**

| Tab | Contents |
|---|---|
| **Portfolio** | Holdings with weight, risk share, volatility, independence · risk-profile heatmap · **risk-contribution chart** · cluster correlation matrix |
| **Analytics** | Head-to-head table (book / EW Shadow / benchmark), a **Style Comparison** of the book each of the other two styles builds from the same run, and benchmark-relationship statistics, over an indexed performance chart |
| **Regime** | 8-factor composite + history. Context only — nothing is conditioned on it |
| **Broker Sync** | Writes curated units into broker order-template JSONs |
| **System** | Configuration, methodology, execution metrics |

---

## Reading the Portfolio tab

**Risk Share vs Weight** is the number the method exists to control. Weight is
share of capital; Risk Share is share of portfolio *variance*. Equal capital
does not mean equal risk, and the `Risk − Wt` column is that gap. It is the one
column where lower is better, so the colour follows the outcome rather than the
sign: green is a holding carrying *less* variance than its capital share, red is
one carrying more.

**Risk Concentration** (header card) is the heaviest holding's risk share
against its equal share. 1.0× is perfect balance; above ~3× means one holding
dominates the book's variance.

**Risk Contribution** puts capital share and variance share on one absolute
scale, against a dashed line at the equal share. What it *should* look like
depends on the style, and the chart says so: ERC solves for flat risk bars on
that line and reports its solver dispersion (target 0.000) separately from the
realised figure, because top-N selection and the position cap move the held book
away from the solution. HRP's bars are uneven **by design** — it balances between
the halves of a correlation-ordered list, not across holdings. For Equal Weight the gap between the two series *is* the
argument for risk-based weighting.

**Risk Structure** is the correlation matrix reordered by cluster. Crisp blocks
along the diagonal mean the clustering found real structure. A uniformly warm
matrix means the universe is effectively a single bet — which no allocator can
fix. No style allocates *from* this Ward tree, so it is labelled a diagnostic
for every style: HRP bisects its own single-linkage ordering, split in halves
by inverse cluster variance, whose first split cuts a Ward cluster every month.

## Reading the Analytics tab

The **Head to Head** table puts all three books side by side, so comparison is a
horizontal read. Green marks the best value per row. Max Drawdown, VaR and CVaR
are negative, so higher (least negative) wins.

**EW Shadow** there is the same holdings split 1/N — the like-for-like test of
the weights, with the names held fixed. **Benchmark** is the market. For ERC and
HRP, expect the book to lead on volatility and drawdown while trailing on return:
that is the trade, not a fault.

**Style Comparison** answers the third question: would a different style have
done better from this date? At run time the other two styles each build a book
from the run's own inputs — date, universe, prices, requested positions, capital
and cap — and Analytics prices them off the same download and calendar as the
book (their lines are in the chart legend, hidden until clicked). Two columns keep
the table from naming a winner it cannot support:

- **Overlap** is the capital two books hold in common. At 85% overlap two books
  cannot end far apart, whatever their styles are called.
- **t** sets the daily gap between the two books against how far they drift
  apart day to day. Under 2 it reads *within noise*; under 20 trading days it is
  not read at all.

Below the universe's size, **Equal Weight the style** keeps the first names in
listing order, so it can hold different names from the EW Shadow; the note under
the table says so when it happens. One window from one date ranks nothing, so
each style's long-run record is quoted beneath it, against Equal Weight.

**Relationship to Benchmark** holds the statistics that only exist for a pairing
— beta, alpha, correlation, tracking error, up/down capture — which is why they
cannot sit in the table above.

---

## Known limits

- **Breadth is the binding constraint.** 30 ETFs at ρ = 0.517 are ~1.9
  independent bets. Uncorrelated exposures (debt, gold, international) would
  raise the ceiling by more than any allocator change.
- **~3 years of usable history.** Most of these ETFs are too young for more, so
  every t-statistic in testing sits below 2. Treat all figures as directional.
- **Long-only.** Dollar-neutral construction measured a 28.6× breadth gain and a
  ~6× ceiling increase — but needs borrow that Indian thematic ETFs are unlikely
  to have.
- **The stock panels are today's constituents.** Survivorship flatters every
  style. Only the Dow has been re-read on point-in-time membership
  (`research/style_search_pit.py`).
- **The 36-candidate figures may have read a dead quote.** yfinance carries
  NESTLEIND.NS flat from the start of its history to Jan 2010; a zero-variance name
  takes nearly all of an inverse-variance split, and raw HRP put 100% on it in 2009.
  Since v12.2 the app unprices dead quotes before estimating (`nco.build_returns_matrix`),
  and the ERC and HRP `long_run` lines quote the in-repo harness, which repairs them.
  The 36-candidate search behind [Why Equal Weight is the default](#why-equal-weight-is-the-default)
  is not in this repository, so whether its Nifty 50 figures read the defect cannot
  be checked.
- **Every backtest rebalances on the first trading day of the month.** Margins
  move with the rebalance day (a spread of about ±0.35 %/yr was measured across
  rebalance days on the v12.1 panels), so read every per-cell margin in this README
  as one draw of that spread.
- **HRP's book depends on the order of the universe file.** Single linkage orients
  each merge by input position, so the same data in another order gives another
  book (up to 0.04-0.05 of raw weight); the order is fixed for a given universe
  file, so a run is reproducible. A data-derived orientation failed its bar.
- **Cited, not present.** `research/README.md` and the 36-candidate search are
  cited above but are not in this repository.

The harnesses in `research/` reproduce the ERC and HRP figures (`style_blends.py`,
`etf_blends.py`, `style_search.py`, `style_search_pit.py`) and the v12.2 HRP audit
(`audit_hrp.py`). The research snapshots (`research/cvg_reweight_*.pkl`, a name kept
from before v13; regenerated by `style_blends.snapshots` when absent) and the
style-search pickles (`python research/style_search.py build`) were rebuilt under the
v12.2 code; figures from before v12.2 in the research files' RESULT blocks were
measured on the old code, and the CVG and Managed Momentum figures in them are kept
as the record. No harness is imported by the app.
