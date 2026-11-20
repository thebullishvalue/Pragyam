# Pragyam · Guide

Why the default style is the default, how to read each result tab, and where the system stops.
The [README](../README.md) is the overview; this is the detail.

## Why Equal Weight Is the Default

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

## Reading the Portfolio Tab

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

## Reading the Analytics Tab

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

## Known Limits

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
