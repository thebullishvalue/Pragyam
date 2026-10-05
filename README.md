# PRAGYAM (प्रज्ञम) — Portfolio Intelligence

**Version:** 12.1.0
**Author:** @thebullishvalue
**License:** Proprietary (See LICENSE file)

Portfolio curation over a chosen universe — the ETF book by default, or an index,
commodity, crypto or custom list. Three of the five
styles forecast nothing and are built to **spread risk**, not to predict returns;
the other two — the Conviction-Value Grid and Managed Momentum — size by the tape:
the Pragati indicator's grid, and the grid plus a 12-1 momentum tilt.

---

## What changed in v12.1

v12.1 adds a fifth style, **Managed Momentum (MMOM)**: the grid's weights plus a
12-1 momentum overlay that stands down while the equal-weighted market's 24-month
return is negative and shrinks while its own volatility runs above its median. No
name falls below a quarter of its grid weight, so every weight stays positive and
the book always fills the position count you ask for. It came out of a
style search with its finalists and bar fixed before the holdout (five families, 43 configurations; chosen on 2007-19,
run once on 2020+) and is the one style that led the best of the eight earlier
styles and blends in every era on Nifty 50 and Dow 30 — by +0.25 to +1.66 %/yr as
shipped, **none of it significant**. The largest per-era t over the best earlier
style is 1.13; over the full span Nifty leads CVG by +1.78 %/yr (t 1.75) and
Equal Weight by +2.57 (t 2.6), nominal t's that survive neither the
43-configuration correction nor the survivorship caveat. On a point-in-time Dow it
trails CVG by 0.19 %/yr. Every figure is an every-name book; a top-N book (Nifty
50 at the default 30 positions, where momentum also picks which names are held)
was never measured for it. Read it as the grid with a tilt, not a reliable premium
([details](#managed-momentum)). The overlay reads daily closes from 2006
(`backdata.fetch_close_history`); when they cannot be fetched it stands down and
the book is the grid's. Everything else from v12 stands.

## What changed in v12

v12 adds a fourth style, the **Conviction-Value Grid (CVG)** — the Pragati indicator's two tapes
(conviction × value) placing every name in a 3 × 3 of states, each sized by measured units
(details below). Its conviction tape reads **Ladder down** (pragati.pine v9.3; the default since v9.1): the intraday
frames inside each day that yfinance carries, falling back to D · W on days older than that
history. Measured throughout — the units were re-weighted only where the change held in every
era; a trend tilt and a neutralised map were tested and rejected. Everything else from v11
stands.

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
nco.py          Equal Weight / ERC / HRP / Conviction-Value Grid / Managed Momentum curation
pragati.py      pragati.pine's conviction tape and histogram
cvgrid.py       the Conviction-Value Grid: nine states, graded map
samanvaya.py    Samanvaya's value tape (macro-hedged relative value)
regime.py       8-factor regime detection (context only)
backdata.py     yfinance fetch + indicator panel + macro drivers + long close history
analytics.py    portfolio-vs-benchmark metrics
charts.py       Plotly builders
universe.py     universe resolution
research/       offline evidence harness (not imported by the app)
```

**Pipeline:** Phase 1 data + regime → Phase 2 curation. About 2s once the panel
is cached. Phase 2 also loads the universe's daily closes since 2006
(`backdata.fetch_close_history`: one yfinance batch, then the snapshot panel's own
second pass for symbols the batch missed; a symbol listed twice keeps its last
column; dead quotes unpriced) for Managed Momentum's overlay — on every run,
because every style's comparison book is built — about 5s for Nifty 50. The
closes are cached for the hour per universe: fetched to today and sliced to the
run date, so another analysis date reuses them, and `nco` cuts them to the book's
own date again, so no later close can reach the overlay. A failed fetch does not
end the run and is not cached: the overlay **stands down** — strength 0, the book
is the grid's, `nco_mmom_stood_down` says why — because the estimation panel
(about 19 months, ~400 sessions) cannot read its 24-month bear gate; the run log, the notices and
the System tab say so. Names in the book's universe with no close in the history
rank 0 and are counted (`nco_mmom_coverage`); a notice names the shortfall.

---

## Usage

```bash
streamlit run app.py
```

**Sidebar:** Analysis Date · Portfolio Style · Universe · Capital · Positions.

**Styles**

| Style | Family | Behaviour |
|---|---|---|
| **Equal Weight** *(default)* | baseline | Identical `1/N` per holding. The default because nothing beat it reproducibly — Managed Momentum led it in every era tested, not significantly; see below. Lowest turnover of any style. |
| **Equal Risk Contribution** | preservation | Solves so every holding contributes the same share of portfolio variance. The preferred risk-reduction style on return: beats HRP on the any-date hit rate in 6 of 6 cells across two stock universes, at about a third of its turnover. In-repo (every name held, net, 2007-26): −0.39 %/yr against Equal Weight on Nifty 50 and −1.27 on Dow 30, at volatility 20.1 / 15.3 against 22.3 / 16.5. |
| **Risk Parity (HRP)** | preservation | Clusters by correlation distance, then splits capital by recursive bisection on cluster variance. Inverts no matrix. The deepest volatility and drawdown cut of the styles (volatility 18.6 / 14.3, max drawdown −50.6% / −36.3% on Nifty 50 / Dow 30), at a larger return cost than ERC (−0.56 / −2.70 %/yr against Equal Weight) and about 3× its turnover. Measured holding every name: at 30 of 50 positions it is the 30 lowest-variance names (18.48 %/yr against 19.13). |
| **Conviction-Value Grid (CVG)** | accumulation | Places every name in the 3 × 3 of the Pragati indicator's two tapes — conviction on Ladder down (the intraday frames inside each day; D · W before intraday history), value on D · W — and sizes it by that state, graded within each cell by the tapes' drawn intensity; the pane's histogram decides when a name changes row. Reads no covariance. Measured: the units beat the seed in every era on Nifty 50 and Dow 30 (v8, then Dislocated 3 → 4 in v12) and are level with or above Equal Weight after 2018 (see [The Conviction-Value Grid](#the-conviction-value-grid)). |
| **Managed Momentum (MMOM)** | accumulation | The grid's weights plus `λ · rank(12-1 momentum) / N` (λ = 1), switched off while the equal-weighted market's 24-month return is negative and scaled down while the overlay's own volatility runs above its median; no name below a quarter of its grid weight, so the book always fills the position count. Reads no covariance. Measured as shipped, every name held: ahead of the best of the eight earlier styles and blends in all six era cells (Nifty 50 +1.02 / +1.66 / +1.58 %/yr, Dow 30 +0.25 / +0.39 / +0.25), none significant (largest per-era t 1.13; full-span Nifty vs CVG +1.78 %/yr at t 1.75, nominal); −0.19 %/yr against CVG on a point-in-time Dow. Top-N books never measured. About 1.3x the grid's turnover (see [Managed Momentum](#managed-momentum)). |

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
universes, at a fifth of the turnover in that search (about a third on the
in-repo harness, `research/style_search.py`, where HRP is the deeper volatility
and drawdown cut). That is a real improvement to the risk
leg — which the rest of this README has always said is the leg that reproduces.

The v12.1 style search did not overturn this. Managed Momentum led Equal Weight
and every other earlier style in all six era cells, but no margin over the best of
them is significant (largest per-era t 1.13). Over the full span its Nifty 50 lead
over Equal Weight reads +2.57 %/yr at a nominal t of 2.6 — but it was one of 43
configurations tried, the panels are today's constituents, and on a point-in-time
Dow it trails the grid (−0.19 %/yr, t −0.28). Equal Weight stays the default.

**Position-count contract.** Every shipped style returns exactly the number of
positions you select. Max Diversification was evaluated, measured well on
lump-sum risk metrics, and **withdrawn anyway**: it is a corner-solution
optimiser that drives most weights to exactly zero, so it returned 10 holdings
when 15 were requested. A style that silently re-decides how many positions you
hold is not a weighting method. `nco_positions_short` and `nco_short_cause` now
record any shortfall and distinguish "the eligible universe ran out" (a data
condition) from "the allocator zeroed names" (a defect). Managed Momentum's floor
exists for this contract: the overlay as tested clipped at zero, so even a book
meant to hold every name held as few as 37 Nifty names; the shipped one keeps
every weight positive, at no less than a quarter of the grid's, so the book fills
whatever count you ask for. It does not keep a name in a smaller book: below the
universe's size the floored names are the first cut (Nifty 50 at 30 positions:
5-9 names floored a month over the past year, none held).
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

### The Conviction-Value Grid

A pure reading of the **Pragati** indicator (`pragati.pine`; प्रगति, "progress" —
formerly Dhṛti), run inside Pragyam's pipeline: the same
eligibility, top-N selection, 10% cap, integer lots and risk diagnostics as
every other style — but the weights come from the indicator and nothing else:
its two tapes and its histogram. No covariance, no solver, no signal (▲▼ ◆ and
divergences are not used).

**The two tapes:**

- **Conviction** (`pragati.py`) — who controls, and how firmly:
  `100 · tanh(mean z)` of participation-weighted agreement `Σc·w / Σ|c|·w`,
  `c = ΔC / TR`, over its ladder. **Ladder down** (v12; pragati.pine's
  default since v9.1): the daily chart plus every lower frame yfinance carries — 1m (7
  days), 3m, 5m / 15m / 30m (60 days), 1h (≈ 2 years), 4h — each running the
  engine on its own history and averaged inside the day, joining where it has
  calibrated (`intraday.py`, fetched once per universe). Days older than the
  intraday history read **Ladder up**, D · W — the weekly rung rebuilt from the
  week as it forms, normalised over 52 weeks — and the snapshot's
  `conv ladder down` is 0. Measured head to head in Sanket (2024-26): the grid
  tied on stocks; in this book's own old-vs-new comparison the change read
  slightly negative on Nifty (−0.35 %/yr since Nov 2024, t −0.8) — see the
  CHANGELOG. A product decision, watched.
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
units (measured — v8, and Dislocated 3 → 4 in v12; see below):

|                          | value cheap      | value fair     | value rich          |
|--------------------------|------------------|----------------|---------------------|
| **conviction up** (≥ +30)| Turned · 3       | Building · 1.5 | Paid · 0.75         |
| **conviction faint**     | Basing · 1.5     | Idle · 1       | Stalling · 0.75     |
| **conviction down** (≤ −30)| Dislocated · 4 | Fading · 1.5   | Distribution · 0.25 |

The seed units were Building 3, Paid 1.5, Dislocated 1 and Fading 0.5. The
Pragati v5 / v7 audit (Sanket, `studies/pine_audit.md`: 380 instruments, six
asset classes, 20 years) found capitulation — sellers in control at a cheap or
fair price — followed by gains in both eras on every class but crypto, and
adding at UP·fair earning nothing, so those four cells were moved. Re-measured in
this allocator (`research/cvg_reweight.py`; monthly, every name held, net of
10bp India / 3bp US costs; decided on data before 2018, confirmed once after):

```
               v8 − seed units            v8 − Equal Weight          turnover (v8 / seed)
               <2018          ≥2018       <2018          ≥2018       <2018        ≥2018
Nifty 50       +0.42 (t 0.6)  +0.98 (1.5) +0.83 (t 2.1)  +0.47 (1.3) 1.28x/1.47x  1.27x/1.49x
Dow 30         +0.55 (t 1.0)  +0.89 (1.0) −0.23 (t −0.4) +0.90 (2.4) 1.17x/1.47x  1.24x/1.49x
```

(%/yr.) The move is positive in both eras on both stock panels at lower
turnover; no single t clears 2, so it is shipped as consistent rather than
proven. The ETF book (1–27 funds, from 2012) is too thin to split and is not
tested. The figures below are the seed units' and are kept as the record.

Sanket's v9 audit (`studies/pragati_v9_audit.md`) re-measured the grid with look-ahead-free
scoring — the v8 audit had demeaned returns by each era's own mean, which leans toward
reversion — across three eras (2006–13, 2014–19, 2020–26) on daily and weekly bars. The
capitulation cell (Dislocated) was still followed by gains in every era (+0.06 to +0.08σ over
10–20 bars outside crypto) and Distribution by losses; the other cells are near zero. The
allocator test above is on portfolio returns and was never affected. The units stand.

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

Every weight stays positive — the state sets how much — so the book fills its
count: when the positions requested cover the universe every name is held; below
that the book fills heaviest first and the lowest weights are cut.

**Measured, seed units** (pre-registered, monthly rebalances through the shipped pipeline,
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

### Managed Momentum

The Conviction-Value Grid with a momentum overlay on top. Same pipeline as every
style — eligibility, top-N, 10% cap, integer lots, risk diagnostics — and, like the
grid, no covariance.

**Why momentum, and why managed.** Names that rose most over the past year,
skipping the latest month, have tended to keep outperforming over the next 3 to
12 months (Jegadeesh & Titman 1993). The pairing with the grid is the
literature's: value and momentum are negatively correlated in every market
studied (Asness, Moskowitz & Pedersen 2013), and the grid's heaviest cells are its
cheap ones. Momentum's known failure is the crash — heavy losses in the rebound
after a bear market, a state marked by a negative two-year market return (Daniel
& Moskowitz 2016) — and its risk is predictable from its own recent volatility,
so scaling it by that volatility cuts the crashes (Barroso & Santa-Clara 2015).
The windows are the papers' (12-1, 24 months, six months); the scale's target —
the overlay's own median volatility, capped so it only shrinks — is this style's
choice, and λ = 1 is one of the three strengths the search tried (0.5, 1, 2),
picked after the holdout over the λ = 2 that discovery ranked first (see the
cautions below).

**The formula.** For each of the N names the book is allocated over — every priced
name in the universe, not the position count:

```
weight_i = max( cvg_i + λ · gate · scale · rank_i / N ,  ¼ · cvg_i ),   λ = 1
```

- `cvg_i` — the grid's normalised weight, exactly as on a CVG run.
- `rank_i` — the name's centred cross-sectional rank, −1 to +1, of its 12-1 return
  (the close 21 sessions ago over the close 252 sessions ago, − 1). Fewer than 10
  names with a 12-1 return: no overlay.
- `gate` — 0 while the equal-weighted market's 24-month (504-session) return is
  negative, else 1. Under a year of history it cannot be read and stays open.
- `scale` — `min(1, median / current)` of the unit overlay's 126-day realised
  volatility, the median taken over every month start so far (at least 7 before it
  acts). It only ever shrinks the overlay.
- the floor — no name below a quarter of its grid weight (`nco.MMOM_FLOOR`, the
  grid's own Distribution-to-Idle ratio, 0.25 : 1).

At strength 1 the strongest 12-1 name gains one equal share (1/N) over its grid
weight before renormalising, and the weakest gives up as much, down to the floor.
The book then takes the usual last step: top-N by weight, renormalised, the 10%
cap, integer units. Below the universe's size that step lets momentum pick which
names are held as well as how much — names at the floor are the first cut — and no
such book was measured (below). The overlay reads the long close history (above);
when that cannot be fetched it stands down (strength 0) and the book is the
grid's, because the estimation panel cannot read the 24-month gate.

**Measured as shipped** (`research/mmom_ship.py`: `nco.compute_nco_portfolio(method="MMOM")`
itself, on the style search's panels — monthly, every name held, net of 10bp India /
3bp US costs; E1 2007-13, E2 2014-19, E3 2020+). Margin is MMOM's net CAGR minus the
best of the eight earlier styles and blends in that cell (H = HRP, C = CVG, EW =
Equal Weight), with its paired t:

```
                         Nifty 50                   Dow 30               ETF (27)
                   E1      E2      E3         E1      E2      E3        Mar 2025 →
best of eight    20.21H  19.68C  22.59C     14.39EW 18.50C  15.06C       17.57EW
MMOM             21.23   21.34   24.17      14.63   18.88   15.31        19.35
  margin         +1.02   +1.66   +1.58      +0.25   +0.39   +0.25        +1.79
  (t)            (0.75)  (0.84)  (1.13)     (0.28)  (0.41)  (0.15)       (0.53)
tested form      +1.17   +1.77   +1.70      +0.22   +0.41   +0.39        +1.96

Full span, net (Feb 2007 → Sep 2026; ETF from Mar 2025)
              CAGR    vol   ret/vol   maxDD   turnover/yr
Nifty  MMOM   22.26  22.40   1.02    −57.2     1.89
       CVG    20.48  22.65   0.94    −58.5     1.48
       EW     19.69  22.25   0.93    −58.4     0.37
Dow    MMOM   16.15  16.91   0.98    −39.1     1.83
       CVG    15.78  16.73   0.97    −39.3     1.41
       EW     15.52  16.48   0.96    −39.9     0.27
ETF    MMOM   19.35  12.68   1.47     −7.5     1.71
       CVG    17.09  13.05   1.28     −7.2     1.39
       EW     17.57  13.70   1.25     −8.0     0.20

Point-in-time Dow, E3 (that day's members, 29-30 names)
       MMOM 11.28 · CVG 11.47 (best of eight) · EW 11.00
       → −0.19 vs CVG (t −0.28), +0.28 vs EW
```

(CAGR and margins %/yr; vol and maxDD %; turnover x/yr.) The shipped weights
match the tested form plus the floor to 4.2e-17 over 584 month-books. The floor
costs about 0.1 %/yr (shipped − tested: Nifty −0.15 / −0.10 / −0.12, Dow +0.02 /
−0.02 / −0.14, ETF −0.18) and, in these every-name books, keeps every priced name
held (39-50 Nifty, 28-30 Dow, 27 ETF; the tested form held as few as 37, 24 and
25). Every figure above is an every-name book (`num_positions` = the universe);
**top-N books were never measured for Managed Momentum** (see the cautions).

**Cautions — read before sizing it.**

- **Not significant.** The largest per-era paired t over the best earlier style is
  1.13 (Nifty E3); Dow E3 is 0.15. Over the full span Nifty leads CVG by
  +1.78 %/yr (t 1.75) and Equal Weight by +2.57 (t 2.6; E1 alone, t 2.1 against
  Equal Weight); the Dow's are +0.37 against CVG (t 0.56) and +0.63 against Equal
  Weight (t 0.97). Those t's are nominal: the search ran 43 configurations and the
  shipped form is a post-holdout variant of one of them, so none survives a
  family-wise correction — and none survives the survivorship caveat below either.
- **Measured on every-name books only.** Every figure here holds every priced
  name. A top-N book — the app's default whenever the universe is larger than the
  position count, e.g. Nifty 50 at 30 — was never measured for this style; there
  momentum also decides which names are held, not only how much, and names at the
  floor are the first cut.
- **Three decisions made after the holdout.** λ = 1 over the λ = 2 that discovery
  ranked first, chosen on the point-in-time Dow (λ = 2 trailed CVG by 0.63 %/yr
  there) and the position-count result (λ = 2 zeroed up to 6 Nifty names); the
  ¼ floor (about −0.1 %/yr, above); and the month-to-date volatility reading,
  which this calendar cannot test — on a first-of-month rebalance that piece is
  empty, so a book built mid-month in the app can differ from the books measured
  above. The holdout's own verdict recommended a point-in-time Nifty test before
  any product decision, and it was not run (`research/style_search_holdout.py`,
  RESULT).
- **The 2020+ edge sits in a few names, led by late index entrants.** The panels
  are today's constituents. The tested form's summed E3 active contribution over
  CVG (`python research/style_search_holdout.py --attribution`; gross, every name
  held): Nifty 50 +10.3 pts — BSE +5.1, TRENT +3.7, BEL +2.4, ADANIENT +1.9 lead,
  and the top five names carry 143% of the total; Dow 30 +2.0 pts, the net of
  larger bets — NVDA +5.8, NKE +2.0, WMT +1.3, AMZN +1.0, CAT +0.8, CRM +0.8
  against DOW −2.4 and DIS −2.1; its top five carry 533% of the total. Momentum held the late
  entrants (BSE, TRENT, BEL, ADANIENT; NVDA, AMZN, CRM) through runs that came
  before they joined the index, which a book confined to the index's members
  could not have done.
- **On a point-in-time Dow it does not lead.** Rebuilt on each day's actual
  members, E3: −0.19 %/yr against CVG (t −0.28), a tie at best. There the ranks are
  over each day's members, while the bear gate and the volatility scale read the
  full panel, non-members included, as the research reference did. No
  point-in-time Nifty panel exists here (it needs NSE's constituent history), so
  the Nifty margins are untested on that count and rest on the same kind of names.
- **The gate rarely shuts** — 10 Nifty months in two episodes (2008-11 →
  2009-05, 2020-04 → 06), 13 Dow months in one (2008-11 → 2009-11), never on the
  ETF window — so E1's margin rests on few events. The research panels' closes start
  in Oct 2006, so until late 2008 the gate read under its 24 months; the app reads
  from 2006-01-01.
- **It trades more:** about 1.3x the grid's turnover and 5-7x Equal Weight's on the
  stock panels (Nifty 1.89x/yr vs 1.48x and 0.37x; Dow 1.83x vs 1.41x and 0.27x).
- **The ETF book is 19 months.** The +1.79 %/yr there sits above the ~1%/yr ceiling
  the v11 notes put on forecasting this book; at t 0.53 it reads as noise, not a
  broken bound.

The volatility scale averaged 0.96 (Nifty), 0.90 (Dow) and 0.86 (ETF), at minimum
0.61, 0.39 and 0.57.

**In the app.** The run log's Close history step reports who reads the closes, the
span and whether the cache served them; the Allocate step logs the overlay's
strength, gate (with the 24-month market return it read), volatility scale, names
ranked, and names at the floor — counted over the universe before top-N, with how
many of them the book actually holds (`nco_mmom_floored` / `nco_mmom_floored_held`).
Notices warn when the overlay stood down because the close history could not be
fetched (`nco_mmom_stood_down`), when it read under 24 months of history, and when
some of the universe has no close in the history (`nco_mmom_coverage` under 1;
those names rank 0). For every style, a notice names the rows whose weight buys
less than one share at your capital (`nco_positions_unfunded`).

**Result tabs**

| Tab | Contents |
|---|---|
| **Portfolio** | Holdings with weight, risk share, volatility, independence · risk-profile heatmap · **risk-contribution chart** · cluster correlation matrix · on a Conviction-Value Grid run: State / Push / Conv / Value tape columns, the **conviction-value map** of all nine states, the state census and the watchlist (Basing and Dislocated names, with how far each is from turning) · on a Managed Momentum run: all of the grid's columns and sections plus a **12-1 %** column, the **Momentum Overlay** strip (strength, bear gate, volatility scale, names ranked, names at the floor and how many of them the book holds, history read) and, while the overlay tilts, a Momentum row in the heatmap |
| **Analytics** | Head-to-head table (book / EW Shadow / benchmark), a **Style Comparison** of the book each of the other four styles builds from the same run, and benchmark-relationship statistics, over an indexed performance chart |
| **Regime** | 8-factor composite + history. Context only — nothing is conditioned on it |
| **Broker Sync** | Writes curated units into broker order-template JSONs |
| **System** | Configuration, methodology, execution metrics · on a Managed Momentum run, the overlay as applied (strength, gate, scale, ranks, floor, history) |

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
that is the trade, not a fault. The grid does not target risk; its weighting ran
within half a percent a year of 1/N over the long run, so expect a small gap
either way. Managed Momentum is the grid plus a momentum tilt, and its measured
edge over the grid is not significant: read its gap to the shadow as the grid's
gap plus noise.

**Style Comparison** answers the third question: would a different style have
done better from this date? At run time the other four styles each build a book
from the run's own inputs — date, universe, prices, requested positions, capital
and cap, and for Managed Momentum the same close history — and Analytics prices
them off the same download and calendar as the book (their lines are in the chart
legend, hidden until clicked). A comparison book built on less history than its
style asks for (Managed Momentum stood down to the grid for want of the close
history, or read a short one) is named under the table. Two columns keep the table from naming a winner it
cannot support:

- **Overlap** is the capital two books hold in common. At 85% overlap two books
  cannot end far apart, whatever their styles are called.
- **t** sets the daily gap between the two books against how far they drift
  apart day to day. Under 2 it reads *within noise*; under 20 trading days it is
  not read at all.

Below the universe's size, **Equal Weight the style** keeps the first names in
listing order, so it can hold different names from the EW Shadow; the note under
the table says so when it happens. One window from one date ranks nothing, so
each style's long-run record is quoted beneath it — against Equal Weight, and for
Managed Momentum against the best of the eight earlier styles and blends.

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
  style, momentum more than the grid: on the Dow in 2020+, Managed Momentum falls
  from 15.31 to 11.28 %/yr on point-in-time members, CVG from 15.06 to 11.47 (see
  [Managed Momentum](#managed-momentum)). Only the Dow has been re-read on
  point-in-time membership.
- **The Nifty 50 ERC and HRP long-run figures may have read a dead quote.** yfinance
  carries NESTLEIND.NS flat from the start of its history to Jan 2010 (986 repeats
  from Jan 2006 in the app's fetch; 786 sessions inside the research panels, which
  start Oct 2006); a
  zero-variance name takes nearly all of an inverse-variance split, and raw HRP put
  100% on it in 2009. `research/style_blends.py` found and repaired this. The
  36-candidate harness behind [Why Equal Weight is the default](#why-equal-weight-is-the-default)
  (and behind the ERC and HRP `long_run` lines) is not in this repository, so
  whether its Nifty 50 figures read the defect cannot be checked. On repaired data,
  `style_blends.py` measures ERC −0.75 and HRP −1.03 %/yr against Equal Weight on
  Nifty 50, and −1.33 and −2.70 on Dow 30 (paired monthly gap, Nov 2006 – Sep
  2026; a different harness and span, so not like for like). Not corrected in this
  release.
- **Cited, not present.** `research/README.md`, `research/conviction_value_grid.py`
  and the 36-candidate search are cited above but are not in this repository.

The harnesses in `research/` reproduce the CVG unit figures (`cvg_reweight.py`,
`cvg_v9.py`) and the v12.1 figures (`style_blends.py`, `etf_blends.py`,
`style_search.py`, `style_search_holdout.py` — its `--attribution` mode prints the
per-name E3 contributions — `style_search_pit.py`, `mmom_ship.py`), with one
exception: the full-span paired t's quoted for Managed Momentum (against CVG and
Equal Weight, and Nifty E1 against Equal Weight) were computed in review from
the monthly net returns `mmom_ship.py` builds (`panel(u, None)["runs"]` against
`["base"]`), which it does not print. No harness is imported by the app.
