# PRAGYAM (प्रज्ञम) — Portfolio Intelligence

**Version:** 11.0.0
**Author:** @thebullishvalue
**License:** Proprietary (See LICENSE file)

Covariance-based portfolio curation over a fixed ETF universe. The book is built
to **spread risk**, not to predict returns.

---

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
nco.py          Equal Weight / ERC / HRP / Conviction-Value Grid curation
pragati.py      pragati.pine's conviction tape and histogram
cvgrid.py       the Conviction-Value Grid: nine states, graded map
samanvaya.py    Samanvaya's value tape (macro-hedged relative value)
regime.py       8-factor regime detection (context only)
backdata.py     yfinance fetch + indicator panel + macro drivers
analytics.py    portfolio-vs-benchmark metrics
charts.py       Plotly builders
universe.py     universe resolution
research/       offline evidence harness (not imported by the app)
```

**Pipeline:** Phase 1 data + regime → Phase 2 covariance curation. About 2s once
the panel is cached.

---

## Usage

```bash
streamlit run app.py
```

**Sidebar:** Analysis Date · Portfolio Style · Universe · Capital · Positions.

**Styles**

| Style | Family | Behaviour |
|---|---|---|
| **Equal Weight** *(default)* | baseline | Identical `1/N` per holding. The default because nothing beat it — see below. Lowest turnover of any style. |
| **Equal Risk Contribution** | preservation | Solves so every holding contributes the same share of portfolio variance. The preferred risk-reduction style: beats HRP on the any-date hit rate in 6 of 6 cells across two stock universes while trading ~5× less. |
| **Risk Parity (HRP)** | preservation | Clusters by correlation distance, then splits capital by recursive bisection on cluster variance. Inverts no matrix. Same job as ERC at five times the turnover; kept for continuity. |
| **Conviction-Value Grid (CVG)** | accumulation | Places every name in the 3 × 3 of the Pragati indicator's two tapes — conviction × value, each on D · W — and sizes it by that state, graded within each cell by the tapes' drawn intensity; the pane's histogram decides when a name changes row. Reads no covariance. Measured: within 0.5%/yr of Equal Weight, not above it (see [The Conviction-Value Grid](#the-conviction-value-grid)). |

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
universes, at a fifth of the turnover. That is a real improvement to the risk
leg — which the rest of this README has always said is the leg that reproduces.

**Position-count contract.** Every shipped style returns exactly the number of
positions you select. Max Diversification was evaluated, measured well on
lump-sum risk metrics, and **withdrawn anyway**: it is a corner-solution
optimiser that drives most weights to exactly zero, so it returned 10 holdings
when 15 were requested. A style that silently re-decides how many positions you
hold is not a weighting method. `nco_positions_short` and `nco_short_cause` now
record any shortfall and distinguish "the eligible universe ran out" (a data
condition) from "the allocator zeroed names" (a defect).

> **Correction (v11.1).** An earlier build of this analysis reported ERC and an
> ERC+momentum tilt beating Equal Weight in 98–100% of five-year SIP streams.
> That was an artifact of a defect in the ERC solver, which renormalised inside
> its descent loop and so converged to something between inverse-volatility and
> equal-risk. Against a corrected solver the result inverted to **0 of 115**.
> The momentum style was withdrawn before release; the fixed solver ships. See
> `research/README.md`.

### The Conviction-Value Grid

A pure reading of the **Pragati** indicator (`pragati.pine`; प्रगति, "progress" —
formerly Dhṛti), run inside Pragyam's pipeline: the same
eligibility, top-N selection, 10% cap, integer lots and risk diagnostics as
every other style — but the weights come from the indicator and nothing else:
its two tapes and its histogram. No covariance, no solver, no signal (▲▼ ◆ and
divergences are not used).

**The two tapes**, each read on the daily chart and the weekly frame above it,
the weekly rung rebuilt from the week as it forms (it lands on the settled
weekly value to 1e-13 and never sees the rest of the week):

- **Conviction** (`pragati.py`) — who controls, and how firmly:
  `100 · tanh(mean z)` of participation-weighted agreement `Σc·w / Σ|c|·w`,
  `c = ΔC / TR`. The Pine's engine and defaults, one adaptation: the weekly
  rung normalises over 52 weeks, not 200, which this panel cannot supply.
- **Value** (`samanvaya.py`) — rich or cheap against what the drivers explain:
  Samanvaya's blend of a hedged return spread and seven market-strength views.
  The hedge is fitted on at most three drivers, chosen by stepwise partial
  correlation past a Šidák-corrected floor, and applied only as far as its own
  out-of-sample skill has earned. Validated against the Pine's quoted skill:
  TLT 0.90 here (0.97 in the Pine), INFY 0.00 (0.04).

**The macro basket.** yfinance has the US yields, INR crosses, dollar index and
commodities directly; other 10-year yields are proxied by government-bond ETFs,
converted to yield moves by duration; non-US 2-year yields have no proxy and
drop out, as the Pine's pooling allows. The basket is **expanded** with Brent,
copper and the name's home equity index (Nifty / S&P 500), so value means rich
or cheap *after* what the market explains. Pre-registered test: kept because it
raised median out-of-sample hedge skill in every universe (ETF 0.01 → 0.45,
Nifty 50 0.005 → 0.23, Dow 30 0.01 → 0.20).

**The 3 × 3.** Each tape is cut at its own shading knee — conviction at its
inner zone (±30), value at θ (±42.9) — into nine states, each with its weight in
units (chosen from what the state means, not fitted):

|                          | value cheap      | value fair     | value rich          |
|--------------------------|------------------|----------------|---------------------|
| **conviction up** (≥ +30)| Turned · 3       | Building · 3   | Paid · 1.5          |
| **conviction faint**     | Basing · 1.5     | Idle · 1       | Stalling · 0.75     |
| **conviction down** (≤ −30)| Dislocated · 1 | Fading · 0.5   | Distribution · 0.25 |

**The histogram runs the rows.** Value moves a name between columns freely —
price is where it is. Control is different: the tape says where control is
going, and the pane's histogram — the Pine's primary read — says whether the
push is behind it. A name moves to its tape's row only while the histogram
*confirms* a push that way, on all three of its channels: the column is on the
side of the move (hue), it is an impulse, building or decelerating rather than
turning (lightness), and the regime is not quiet (saturation). Otherwise the
name keeps its row, and the table marks it **held**. A name whose histogram is
not yet calibrated follows its tape.

**The map is graded**, as the Pine draws it. Each cell's units are its centre;
inside the cell a name's weight moves toward one neighbouring cell per axis by
how intensely its tape is shaded — the Pine's own ramps: faint 0.12 → 0.45
inside the knee, a visible step, bright 0.65 → 1.0 to solid (±60 conviction,
±70 value). A conviction tape just past +30 sits a third of the way toward the
faint row; a solid one earns the full cell. A held row keeps between half and
all of its cell, by how intensely the push holding it is drawn. Weight is
continuous inside a cell and steps where the Pine's shading steps.

Every name is held — the state sets how much, never whether — and the book
fills heaviest first when N is below the universe.

**Measured** (pre-registered, monthly rebalances through the shipped pipeline,
every name held). The two parts are isolated: **gate** is the book minus the
same graded map with rows simply following the tape; **grading** is the book
minus the same engine on flat cells:

```
             CVG − EW             gate                  grading               turnover (CVG / flat / EW)
ETF book     −0.17%/yr (t −0.43)  −0.12%/yr (t −0.93)   +0.16%/yr (t +0.99)   0.91x / 1.18x / 0.47x
Nifty 50     −0.32%/yr (t −0.62)  +0.31%/yr (t +2.17)   −0.07%/yr (t −0.29)   1.47x / 2.30x / 0.33x
Dow 30       −0.41%/yr (t −0.78)  +0.28%/yr (t +1.47)   +0.28%/yr (t +0.99)   1.49x / 2.41x / 0.26x
```

It does not beat Equal Weight, but it is within half a percent everywhere and
none of the gaps is significant. The histogram earns its place on single stocks
— +0.3%/yr over the same map without it, and a third fewer row changes (Nifty
8.0 vs 12.0 per name-year) — and is neutral on the sector ETFs. Grading is
return-neutral and cuts turnover by a quarter to two-fifths. Several designs
were tried on these same panels, so read every t as directional. Two more
cautions. The two tapes are **+0.6 correlated** (the value tape's
breadth leg is momentum), so the corners that need them to disagree stay thin
(Turned ~0.6% of name-days, Distribution ~0.1%). And the next-month spread of
core over floor is positive on ETFs (+1.9%, t 1.5) but negative on single
stocks (Nifty −0.7%, Dow −0.9%, t ≈ −1.3): on stocks the tapes' direction still
partly reverses. `research/conviction_value_grid.py` reproduces every figure above.

**Result tabs**

| Tab | Contents |
|---|---|
| **Portfolio** | Holdings with weight, risk share, volatility, independence · risk-profile heatmap · **risk-contribution chart** · cluster correlation matrix · on a Conviction-Value Grid run: State / Push / Conv / Value tape columns, the **conviction-value map** of all nine states, the state census and the watchlist (Basing and Dislocated names, with how far each is from turning) |
| **Analytics** | Head-to-head table (book / EW Shadow / benchmark), a **Style Comparison** of the book each other style builds from the same run, and benchmark-relationship statistics, over an indexed performance chart |
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
away from the solution. HRP's bars are uneven **by design** — it balances across
clusters, not holdings. For Equal Weight the gap between the two series *is* the
argument for risk-based weighting.

**Risk Structure** is the correlation matrix reordered by cluster. Crisp blocks
along the diagonal mean the clustering found real structure. A uniformly warm
matrix means the universe is effectively a single bet — which no allocator can
fix. Only HRP allocates *from* this tree; for the other styles the matrix is
shown as a diagnostic and labelled as such.

## Reading the Analytics tab

The **Head to Head** table puts all three books side by side, so comparison is a
horizontal read. Green marks the best value per row. Max Drawdown, VaR and CVaR
are negative, so higher (least negative) wins.

**EW Shadow** there is the same holdings split 1/N — the like-for-like test of
the weights, with the names held fixed. **Benchmark** is the market. For ERC and
HRP, expect the book to lead on volatility and drawdown while trailing on return:
that is the trade, not a fault. The grid does not target risk; its weighting ran
within half a percent a year of 1/N over the long run, so expect a small gap
either way.

**Style Comparison** answers the third question: would a different style have
done better from this date? At run time the other three styles each build a book
from the run's own inputs — date, universe, prices, requested positions, capital
and cap — and Analytics prices them off the same download and calendar as the
book (their lines are in the chart legend, hidden until clicked). Two columns
keep the table from naming a winner it cannot support:

- **Overlap** is the capital two books hold in common. At 85% overlap two books
  cannot end far apart, whatever their styles are called.
- **t** sets the daily gap between the two books against how far they drift
  apart day to day. Under 2 it reads *within noise*; under 20 trading days it is
  not read at all.

Below the universe's size, **Equal Weight the style** keeps the first names in
listing order, so it can hold different names from the EW Shadow; the note under
the table says so when it happens. One window from one date ranks nothing, so
each style's long-run record against Equal Weight is quoted beneath it.

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

See `research/README.md` for the harness that reproduces every claim here.
