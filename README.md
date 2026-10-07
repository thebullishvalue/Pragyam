# PRAGYAM (प्रज्ञम) — Portfolio Intelligence

**Version:** 12.2.0
**Author:** @thebullishvalue
**License:** Proprietary (See LICENSE file)

Portfolio curation over a chosen universe — the ETF book by default, or an index,
commodity, crypto or custom list. Three of the five
styles forecast nothing and are built to **spread risk**, not to predict returns;
the other two — the Conviction-Value Grid and Managed Momentum — size by the tape:
the Pragati indicator's grid, and the grid plus a 12-1 momentum tilt.

---

## What changed in v12.2

v12.2 is an audit release. CVG, HRP and Managed Momentum were each audited by a
committee — three readers per style, a chair who merged their findings and
pre-registered the opportunities, a skeptic who tried to refute every bug on real
data, and a tester who ran the opportunities through the product code
(`research/audit_cvg.py`, `audit_hrp.py`, `audit_mmom.py`). Every reported bug
reproduced and is fixed; the style changes the audit measured were decided on that
evidence, and the release diff was then reviewed adversarially itself. Every figure
in this README is re-measured on the v12.2 code, with the research snapshots
regenerated under it.

**Data every style reads.**
- yfinance leaves some Indian demergers and mis-dated splits unadjusted (BAJAJFINSV
  −64% and −93% in 2008, ADANIENT 2015-06-03, TMPV 2025-10-14, TRENT 2026-01-01). An
  Indian listing's ≥ 30% fall (or a doubling) on a ≥ 30% overnight gap is back-adjusted in the close
  history and the estimation panel (`backdata.corporate_action_gaps`); a move that
  reverses a recent spike is a bad print and is unpriced instead. The research return
  panel is repaired the same way (`style_blends.repair`). Every style had been scored
  on those fake losses: Nifty 50 Equal Weight's full span rises from 19.69 to 20.10 %/yr.
- A run during market hours no longer reads today's still-forming bar (it moved the
  Nifty book 2-5% by the hour); exchange-holiday prints are dropped before the tapes.
- Whole-share rounding left up to ~14% of a ₹5L, 50-name book in cash (a −1.6 %/yr
  drag on CVG since 2020). The leftover is now spent on the holdings furthest below
  target, never past the cap.

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

**CVG.** **The conviction tape reads D · W again.** Ladder down (v12.0-12.1) averaged the
daily rung with every intraday frame and restored no variance, so the tape's scale
shrank as rungs were added (cross-sectional sd ~30 on D · W, ~14 with all seven), its
±30 knee meant something different on every bar, and no backtest ever scored the
seven-rung tape. Every measured CVG figure is a D · W tape; no intraday data is fetched
now. **The app's tapes read 8 years of bars** (the snapshots keep the estimation
window): on ~19 months, 1 name in 8 sat in a different state than on full-history
tapes. Also: a notice when the cap forces 1/N; two wrong bond proxies dropped from the
macro pool; European drivers lagged for NSE names (they closed hours after the NSE).
Together, on the regenerated panels: Nifty 50 −0.05 / −0.07 / +0.25 %/yr, Dow 30 +0.00
/ +0.01 / +0.05, ETF +0.06, point-in-time Dow +0.09.

**Managed Momentum.** The gate and the volatility scale carry closes across gaps; the
windows follow the panel's own calendar (Crypto's gate had read shut at −22% on a +26%
market); the overlay stands down below 24 months of history whatever history it read;
a close ≤ 0 is not a price; a name listed twice keeps one listing. **The bear gate is
read at the first session of the month** and held through it, as every backtest read it
(read daily, a mid-month book could swing between momentum and the pure grid: 36.5%
one-day turnover on 2026-09-28). **The volatility scale is two-sided**,
`min(1.5, median / current)`: it now also grows the overlay, up to 1.5×, while its
volatility runs below its median (Barroso & Santa-Clara; Moreira & Muir). Pre-registered
and re-passed on the final data: ahead of the one-sided scale in all six cells (Nifty
+0.28 / +0.14 / +0.22, Dow +0.05 / +0.18 / +0.01 %/yr), level on the point-in-time Dow.
As shipped: ahead of the best earlier style in all six era cells — Nifty 50 +0.33 /
+1.72 / +1.59 %/yr, Dow 30 +0.77 / +0.57 / +0.29 — none significant (largest t 1.07);
−0.38 %/yr against CVG on a point-in-time Dow. A book cut below the universe is mostly
momentum's pick; that is now measured and stated ([details](#managed-momentum)).

**Not shipped.** A data-derived HRP dendrogram orientation (it would end the book's
dependence on the universe file's order) failed its bar inside the staggered average
(ETF −1.68 %/yr, t −2.1). Overlapping momentum formations (Jegadeesh-Titman K = 3)
failed theirs. Every backtest still rebalances on the first trading day (see
[Known limits](#known-limits)).

## What changed in v12.1

*(As published in v12.1; v12.2 changed and re-measured Managed Momentum — see above.)*

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
(details below). Its conviction tape read **Ladder down** (pragati.pine v9.3; the default since v9.1): the intraday
frames inside each day that yfinance carries, falling back to D · W on days older than that
history — reverted to D · W in v12.2. Measured throughout — the units were re-weighted only where the change held in every
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
| **Equal Risk Contribution** | preservation | Solves so every holding contributes the same share of portfolio variance. The preferred risk-reduction style on return: beats HRP on the any-date hit rate in 6 of 6 cells across two stock universes, at about 0.6x its turnover. In-repo (every name held, net, 2007-26): −0.49 %/yr against Equal Weight on Nifty 50 and −1.27 on Dow 30, at volatility 20.1 / 15.3 against 22.2 / 16.5. |
| **Risk Parity (HRP)** | preservation | Clusters by correlation distance, then splits capital by recursive bisection on cluster variance. Inverts no matrix. Averaged over three staggered estimation windows (v12.2). The deepest volatility and drawdown cut of the styles (volatility 18.8 / 14.2, max drawdown −49.3% / −36.4% on Nifty 50 / Dow 30), at −0.54 / −2.47 %/yr against Equal Weight and about 1.7× ERC's turnover. Measured holding every name: at 30 of 50 positions it is the 30 lowest-variance names (19.11 %/yr against 19.56). |
| **Conviction-Value Grid (CVG)** | accumulation | Places every name in the 3 × 3 of the Pragati indicator's two tapes — conviction and value, both on D · W (the daily chart and the weekly rung) — and sizes it by that state, graded within each cell by the tapes' drawn intensity; the pane's histogram decides when a name changes row. Reads no covariance. Measured: the units beat the seed in every era on Nifty 50 and Dow 30 (v8, then Dislocated 3 → 4 in v12) and are level with or above Equal Weight after 2018 (see [The Conviction-Value Grid](#the-conviction-value-grid)). |
| **Managed Momentum (MMOM)** | accumulation | The grid's weights plus `λ · rank(12-1 momentum) / N` (λ = 1), switched off while the equal-weighted market's 24-month return (read at the month's first session) is negative, and scaled by the overlay's own volatility against its median — down when above, up to 1.5× when below; no name below a quarter of its grid weight, so the book always fills the position count. Reads no covariance. Measured as shipped (v12.2), every name held: ahead of the best of the eight earlier styles and blends in all six era cells (Nifty 50 +0.33 / +1.72 / +1.59 %/yr, Dow 30 +0.77 / +0.57 / +0.29), none significant (largest per-era t 1.07; full-span Nifty vs CVG +1.37 %/yr at t 1.17, nominal); −0.38 %/yr against CVG on a point-in-time Dow. Below the universe's size momentum mostly picks the names: measured on v12.1, ahead of the grid on Nifty 50 at 30 positions, 1.8-4.3 %/yr behind it on the Dow since 2020 and 1.9-5.0 on a point-in-time Dow. About 1.3x the grid's turnover (see [Managed Momentum](#managed-momentum)). |

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

The v12.1 style search did not overturn this. Managed Momentum led Equal Weight
and every other earlier style in all six era cells, but no margin over the best of
them is significant (largest per-era t 1.07 as re-measured in v12.2). Over the full
span its Nifty 50 lead over Equal Weight reads +2.27 %/yr at a nominal t of 2.11 —
but it was one of 43 configurations tried, the panels are today's constituents, and
on a point-in-time Dow it trails the grid (−0.38 %/yr, t −0.45). Equal Weight stays
the default.

**Position-count contract.** Every shipped style returns exactly the number of
positions you select. Max Diversification was evaluated, measured well on
lump-sum risk metrics, and **withdrawn anyway**: it is a corner-solution
optimiser that drives most weights to exactly zero, so it returned 10 holdings
when 15 were requested. A style that silently re-decides how many positions you
hold is not a weighting method. `nco_positions_short` and `nco_short_cause` now
record any shortfall and distinguish "the eligible universe ran out" (a data
condition) from "the allocator zeroed names" (a defect). Managed Momentum's floor
exists for this contract: the overlay as tested clipped at zero, so even a book
meant to hold every name held as few as 36 Nifty names; the shipped one keeps
every weight positive, at no less than a quarter of the grid's, so the book fills
whatever count you ask for. It does not keep a name in a smaller book: below the
universe's size floored names usually fall outside it (Nifty 50 at 30 positions:
5-9 names floored a month over the past year, none held), though a floored
Dislocated name can outweigh an unfloored Idle one (the Dow at 25 positions held
1-2 floored names in 18 of 223 months).
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
  `c = ΔC / TR`, over its ladder: **D · W** — the daily chart and the weekly rung
  rebuilt from the week as it forms, normalised over 52 weeks (`pragati.LADDER =
  "up"`). v12.0 and v12.1 read **Ladder down** (pragati.pine's default since v9.1):
  the daily chart plus every lower frame yfinance carries, averaged inside the day.
  v12.2 reverted it: the mean of k rungs restores no variance, so the tape's scale
  fell as rungs were added (cross-sectional sd ~30 on D · W, ~14 with all seven),
  its ±30 knee meant something different on every bar, and no backtest ever scored
  the seven-rung tape. Every CVG figure in this README is a D · W tape. Ladder down
  stays behind the switch (`intraday.py`), off.
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
the overlay's own median volatility, capped at 1.5× (v12.2; until then it could only
shrink) — is this style's choice, and λ = 1 is one of the three strengths the search tried (0.5, 1, 2),
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
  negative, else 1, read as of the first session of the run's month and held through
  it. Under 24 months of history the overlay stands down altogether.
- `scale` — `min(1.5, median / current)` of the unit overlay's 126-day realised
  volatility, the median taken over every month start so far (at least 7 before it
  acts). It shrinks the overlay while that volatility runs above its median and grows
  it, up to 1.5×, while it runs below.
- the floor — no name below a quarter of its grid weight (`nco.MMOM_FLOOR`, the
  grid's own Distribution-to-Idle ratio, 0.25 : 1).

At strength 1 the strongest 12-1 name gains one equal share (1/N) over its grid
weight before renormalising, and the weakest gives up as much, down to the floor.
The book then takes the usual last step: top-N by weight, renormalised, the 10%
cap, integer units. Below the universe's size that step lets momentum pick which
names are held as well as how much (measured below). The overlay reads the long
close history (above); when that cannot be fetched, or holds under the gate's 24
months, it stands down (strength 0) and the book is the grid's. The windows count
rows of the panel's own calendar: a 7-day panel (Crypto, or a Custom List mixing
calendars) reads 365 / 30 / 730 / 183 rows.

**Measured as shipped** (v12.2; `research/mmom_ship.py --app-history`:
`nco.compute_nco_portfolio(method="MMOM")` itself, its overlay reading the app's own
close history from 2006 sliced to each date, on the style search's panels with the
corporate-action repair — monthly, every name held, net of 10bp India / 3bp US
costs; E1 2007-13, E2 2014-19, E3 2020+). Margin is MMOM's net CAGR minus the best of
the eight earlier styles and blends in that cell (H = HRP, C = CVG, EW = Equal
Weight), with its paired t:

```
                         Nifty 50                   Dow 30               ETF (27)
                   E1      E2      E3         E1      E2      E3        Mar 2025 →
best of eight    20.39H  19.82C  23.22C     14.39EW 18.50C  15.11C       17.57EW
MMOM             20.73   21.54   24.81      15.16   19.07   15.40        19.57
  margin         +0.33   +1.72   +1.59      +0.77   +0.57   +0.29        +2.01
  (t)            (0.54)  (0.82)  (1.07)     (0.70)  (0.52)  (0.18)       (0.63)
one-sided scale  +0.06   +1.58   +1.37      +0.72   +0.38   +0.28        +1.84
v12.1 published  +1.02   +1.66   +1.58      +0.25   +0.39   +0.25        +1.79

Full span, net (Feb 2007 → Sep 2026; ETF from Mar 2025)
              CAGR    vol   ret/vol   maxDD   turnover/yr
Nifty  MMOM   22.36  22.11   1.03    −57.1     1.89
       CVG    20.99  22.55   0.96    −56.6     1.46
       EW     20.10  22.16   0.94    −56.5     0.37
Dow    MMOM   16.42  16.88   0.99    −37.8     1.82
       CVG    15.80  16.74   0.97    −39.3     1.40
       EW     15.52  16.48   0.96    −39.9     0.27
ETF    MMOM   19.57  12.85   1.46     −8.1     1.62
       CVG    17.16  13.24   1.27     −7.8     1.25
       EW     17.57  13.70   1.25     −8.0     0.20

Point-in-time Dow, E3 (that day's members, 29-30 names; the overlay reads only them)
       MMOM 11.18 · CVG 11.56 (best of eight) · EW 11.00
       → −0.38 vs CVG (t −0.45), +0.17 vs EW
```

(CAGR and margins %/yr; vol and maxDD %; turnover x/yr.) "One-sided scale" is the
same book with the v12.1 scale, `min(1, median / current)`. What moved since v12.1:
the repaired return panel lifts every Nifty style (Equal Weight +0.40 %/yr full span)
and takes away an E1 edge the overlay drew from reading those fake losses as
momentum; the overlay stands down below 24 months of history (off until 2008-01 on
the app's history, which starts 2006-01-01); HRP's staggered windows raised the Nifty
E1 bar to 20.39; and the two-sided scale added +0.14 to +0.28 on Nifty. **The Nifty E1
margin is thin** (+0.33, t 0.54; +0.06 with the one-sided scale). The tested form of
v12.1 reads the research panel without the floor or the history minimum, so it no
longer matches the shipped weights exactly. In these every-name books every priced
name is held (39-50 Nifty, 28-30 Dow, 27 ETF).

**Cut books** (measured on v12.1, before the repairs, by the audit's skeptic). In a
book smaller than the universe the names held are mostly the 12-1 leaders: 0.86-0.94
of them are a pure 12-1 top-N's. Nifty 50 at 30 positions: 22.48 / 22.19 / 24.56
%/yr against CVG's 19.99 / 21.35 / 23.87, at lower turnover (2.63 vs 4.41x/yr). Dow
2020+: 1.75 (25 positions) to 4.30 %/yr (10) behind the grid; point-in-time Dow
2020+: 1.88 to 4.98 behind it. The cut-book lead appears only on Nifty, today's
constituents.

**Cautions — read before sizing it.**

- **Not significant.** The largest per-era paired t over the best earlier style is
  1.07 (Nifty E3); Dow E3 is 0.18. Over the full span Nifty leads CVG by
  +1.37 %/yr (t 1.17) and Equal Weight by +2.27 (t 2.11); the Dow's are +0.62
  against CVG (t 0.91) and +0.90 against Equal Weight (t 1.33). Those t's are
  nominal: the search ran 43 configurations and the shipped form is a post-holdout
  variant of one of them, so none survives a family-wise correction — and none
  survives the survivorship caveat below either.
- **A cut book is a momentum book.** The app's default whenever the universe is
  larger than the position count (Nifty 50 at 30) holds mostly the 12-1 leaders.
  That measured well on Nifty and badly on the Dow, point-in-time included (above):
  do not read the full-book figures as a description of a cut book.
- **Decisions made after the holdout.** λ = 1 over the λ = 2 that discovery
  ranked first, chosen on the point-in-time Dow (λ = 2 trailed CVG by 0.63 %/yr
  there) and the position-count result (λ = 2 zeroed up to 6 Nifty names); the
  ¼ floor (about −0.1 %/yr); the two-sided volatility scale (v12.2, pre-registered
  in the audit and passed: Nifty +0.14 to +0.28, Dow +0.01 to +0.18, none
  significant); and the month-to-date volatility reading,
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
  members, E3: −0.38 %/yr against CVG (t −0.45), +0.17 against Equal Weight. The
  ranks, the bear gate and the volatility scale all read that day's members only,
  as a live user's fetch would (v12.1 let the gate and scale read non-members too,
  which read −0.19). No point-in-time Nifty panel exists here (it needs NSE's
  constituent history), so the Nifty margins are untested on that count and rest on
  the same kind of names.
- **The gate rarely shuts, and reads a lax market** — 9 Nifty months in two
  episodes (2008-11 → 2009-04, 2020-04 → 06), 13 Dow months in one (2008-11 →
  2009-11), never on the ETF window — so E1's margin rests on few events. Its market
  is the equal-weighted return of today's constituents, a survivor-biased market whose
  24-month return ran 29-33 points above the Nifty indices (13-19 above the Dow and
  RSP) since 2008: it stayed open on 16-28 month starts where an index gate was shut.
  An index gate measured worse (Nifty 21.93 vs 22.26, Dow 16.02 vs 16.15, v12.1)
  and is not used.
- **It trades more:** about 1.3x the grid's turnover and 5-7x Equal Weight's on the
  stock panels (Nifty 1.89x/yr vs 1.46x and 0.37x; Dow 1.82x vs 1.40x and 0.27x).
- **The ETF book is 19 months.** The +2.01 %/yr there sits above the ~1%/yr ceiling
  the v11 notes put on forecasting this book; at t 0.63 it reads as noise, not a
  broken bound.

The volatility scale averaged 1.11 (Nifty), 1.01 (Dow) and 0.89 (ETF), at minimum
0.58, 0.39 and 0.54; the floor bound on 0-15 Nifty names a month (mean 7.0), 0-9 Dow
(3.7), 0-5 ETF (2.2), counted over the universe.

**In the app.** The run log's Close history step reports who reads the closes, the
span and whether the cache served them; the Allocate step logs the overlay's
strength, gate (with the 24-month market return it read), volatility scale, names
ranked, and names at the floor — counted over the universe before top-N, with how
many of them the book actually holds (`nco_mmom_floored` / `nco_mmom_floored_held`).
Notices warn when the overlay stood down because the close history could not be
fetched or held under 24 months (`nco_mmom_stood_down`), and when
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
  from 15.40 to 11.18 %/yr on point-in-time members, CVG from 15.11 to 11.56 (see
  [Managed Momentum](#managed-momentum)). Only the Dow has been re-read on
  point-in-time membership.
- **The 36-candidate figures may have read a dead quote.** yfinance carries
  NESTLEIND.NS flat from the start of its history to Jan 2010; a zero-variance name
  takes nearly all of an inverse-variance split, and raw HRP put 100% on it in 2009.
  Since v12.2 the app unprices dead quotes before estimating (`nco.build_returns_matrix`),
  and the ERC and HRP `long_run` lines quote the in-repo harness, which repairs them.
  The 36-candidate search behind [Why Equal Weight is the default](#why-equal-weight-is-the-default)
  is not in this repository, so whether its Nifty 50 figures read the defect cannot
  be checked.
- **Every backtest rebalances on the first trading day of the month.** Across
  rebalance days CVG's Nifty 50 margin over Equal Weight averaged +0.40 %/yr with a
  spread of ±0.35 on the v12.1 panels, and the first trading day (+0.79 there) was a
  favourable draw. Read every per-cell margin in this README as one draw of that
  spread.
- **HRP's book depends on the order of the universe file.** Single linkage orients
  each merge by input position, so the same data in another order gives another
  book (up to 0.04-0.05 of raw weight); the order is fixed for a given universe
  file, so a run is reproducible. A data-derived orientation failed its bar.
- **Cited, not present.** `research/README.md`, `research/conviction_value_grid.py`
  and the 36-candidate search are cited above but are not in this repository.

The harnesses in `research/` reproduce the CVG unit figures (`cvg_reweight.py`,
`cvg_v9.py`), the v12.1 figures (`style_blends.py`, `etf_blends.py`,
`style_search.py`, `style_search_holdout.py` — its `--attribution` mode prints the
per-name E3 contributions — `style_search_pit.py`) and the v12.2 audit
(`audit_cvg.py`, `audit_hrp.py`, `audit_mmom.py`, and `mmom_ship.py --app-history`
for Managed Momentum as shipped), with one exception: the full-span paired t's
quoted for Managed Momentum (against CVG and Equal Weight) were computed in review
from the monthly net returns `mmom_ship.py` builds (`panel(u, None, app=True)["runs"]`
against `["base"]`), which it does not print. The research snapshots
(`research/cvg_reweight_*.pkl`, regenerated by `style_blends.snapshots` when absent)
and the style-search pickles (`python research/style_search.py build`) were rebuilt
under the v12.2 code; figures from before v12.2 in the research files' RESULT blocks
were measured on the old code. No harness is imported by the app.
