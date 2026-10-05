"""
research/audit_hrp.py — HRP opportunity test: staggered-window HRP (O1) and an anchored leaf order (O2).

PRE-REGISTRATION (copied from the coordinator's brief without changes, before any variant was run;
no variants beyond O1 A/B/C and O2 A/B)
───────────────────────────────────────────────────────────────────────────────────────────────────
id      O1
title   Staggered-window HRP: average the shipped HRP over 252-row windows ending 0, 21, 42 (and 63)
        sessions back, inside the ~400-session panel

mechanism
    Most of HRP's month-to-month weight change is noise in the single-linkage leaf order, not
    covariance drift. The target moves 0.097/month on Nifty against 0.026 for inverse variance on
    the same covariance; leaf-order change accounts for 1.05 of the 1.15x/yr target change; the top
    split is unchanged in only 8-13% of months. Averaging HRP fits over a few staggered windows
    smooths this discontinuous order-to-split map: a stateless form of trading toward an averaged
    aim portfolio. It needs no saved state, fits inside the app's ~400-session estimation panel, and
    degrades exactly to today's HRP on the 100-day fallback panel. Independent block-bootstrap
    bagging gives the same turnover halving, which confirms the mechanism.

    In-sample harness results, disclosed because the auditors measured them before this
    pre-registration:
    - sess3x21: CAGR higher and return/vol higher in all six cells. Nifty E1 passes by a hair (CAGR
      +0.11, ret/vol +0.0003) and Nifty E3 by CAGR +0.04, both below the universe-reorder noise
      floor (era spread 0.25-0.29).
    - sess4x21: passes only on the CAGR rule (Nifty E1 +0.04 at vol +0.23).
    - Turnover: Nifty 1.23 -> 0.73/0.65, Dow 0.91 -> 0.52/0.47.
    - Point-in-time Dow E3 9.53 -> 9.94/9.97.
    - ETF -0.05.
    This run is therefore a confirmation through the in-nco implementation (implcheck.py matched the
    harness exactly on 9 Dow dates), not a discovery.

literature
    Garleanu & Pedersen (2013, JF 68(6)), trading toward an aim portfolio that averages targets.
    Olivares-Nadal & DeMiguel (2018, OR 66(4)), turnover control as robustness to covariance error.
    DeMiguel, Garlappi & Uppal (2009, RFS 22(5)), estimation noise shows up as turnover. Breiman
    (1996), bagging. Lopez de Prado (2016, JPM 42(4)), HRP.

variants
    A (HRP-S3): K=3 windows, step 21 sessions. For j in {0,1,2}: end = len(panel) - 21*j over the
        last 400 sessions only. Skip j>0 if end < 253, or if the window fails the shipped cov_ok
        rule. R_j = build_returns_matrix(history[:end], today's est_names, 252) with the shipped
        coverage, return and NaN rules. w_j = shipped hrp_weights(np.cov(R_j),
        nan_to_num(corrcoef(R_j))) over R_j's columns, reindexed to today's est_names. Final w =
        mean over windows with skipna, then fillna(0), renormalise, then the unchanged top-N / 10%
        cap / units path. solver_cov and risk diagnostics stay on today's covariance; record attrs
        nco_hrp_windows. Harness: hrp_lp/robust.smooth_sessions(3, 21, panel=400).
    B (HRP-S4): identical to A with K=4 (j in {0,1,2,3}), step 21. Harness: smooth_sessions(4, 21,
        panel=400).
    C (HRP-S3-canon): identical to A, but every hrp_weights call uses the canonical dendrogram
        orientation of bug B5: at each merge the child with more leaves goes first; on a size tie,
        the child with the lower inverse-variance cluster variance; then the child containing the
        lexicographically smallest symbol. Halves bisection unchanged (hrp/canon.py canon_order with
        the symbol tie-break). If several variants pass, prefer C (it also removes the order
        dependence), then A, then B.

bar
    Run ss.run(fn, d) with every name held, monthly, net of 10bp India / 3bp US, and the harness
    top-N(all) + 10% cap, on nifty_50 and dow_30 with holdout=True. Compare against the shipped HRP
    rebuilt at the same code state: ss.baselines(d)['HRP'] if no input fix has landed; otherwise the
    HRP rebuilt with B1-B3 applied, for both baseline and variant.
    PASS if either:
    (i) CAGR/vol is higher than shipped HRP in all six cells (E1/E2/E3 x Nifty, Dow) and vol rises
        by no more than 0.5 pt in any cell; or
    (ii) CAGR is higher in all six cells and vol rises by at most 0.5 pt in every cell.
    Report, not ruled on:
    - full-span turnover per universe;
    - ETF27;
    - Nifty top-30 (ss.run(fn, d, n=30));
    - point-in-time Dow E3 (style_search_pit.py ctx.priced monkeypatch);
    - each cell's margin next to the universe-reorder noise floor (era CAGR spread 0.25-0.29 across
      7 random orders, hrp/logs/perm_bt.log), flagging any pass that rests on a margin below it.
    Parameters are fixed and may not be changed.

needs_regeneration  false

───────────────────────────────────────────────────────────────────────────────────────────────────
id      O2
title   Quarter-anchored leaf order: take HRP's quasi-diagonal order from the window ending at the
        start of the calendar quarter, and bisect monthly on today's covariance

mechanism
    This targets the same leaf-order noise as O1 with a different trade-off. The tree and order are
    held fixed through a calendar period, while the bisection alphas are recomputed on today's
    covariance, so the risk sizing has no averaging lag. The rule is stateless: the anchor window is
    recomputed from the panel (quarter anchor up to about 63 sessions back plus 253 rows, half-year
    anchor up to about 126 plus 253, both within the ~398-400-session panel).
    In-sample evidence is the auditors' stateful form (refresh every 3 months from the backtest
    start, plus on any change in the name set; hrp_inputs/sticky2.py):
    - turnover: Nifty 1.23 -> 0.82, Dow 0.91 -> 0.60, ETF 0.92 -> 0.77;
    - mean monthly return differences by era: Nifty +0.33/+0.26/-0.05, Dow +0.02/-0.01/+0.30;
    - vol: Nifty 18.66 vs 18.77, Dow 14.18 vs 14.16.
    Nifty E3 and Dow E2 are at risk under the bar, and the fixed calendar phase is untested. A
    weaker candidate than O1, kept as the lag-free alternative.

literature
    Pfitzinger & Katzke (2019), constrained HRP: stability of hierarchical allocations. Carlsson &
    Memoli (2010, JMLR), instability of single linkage. Novy-Marx & Velikov (2016, RFS 29(1)),
    turnover-reducing rebalancing rules.

variants
    A (HRP-Q): anchor = the first panel session on or after the first day of the run date's
        calendar quarter (Jan 1, Apr 1, Jul 1, Oct 1). R_a = build_returns_matrix(history up to and
        including the anchor, today's est_names, 252) under the shipped rules. If R_a's column set
        equals today's est_names and passes cov_ok, order = the shipped quasi-diagonal order (single
        linkage on correlation_distance(nan_to_num(corrcoef(R_a))), shipped orientation, columns in
        est_names order). Otherwise order = today's order. Then the shipped halves bisection with
        cluster_var on TODAY's covariance, then the unchanged top-N / cap path. On quarter-start
        rebalance dates the anchor is the run date, so the result equals shipped HRP.
    B (HRP-H): identical to A with half-year anchors (Jan 1 and Jul 1). If both pass, prefer A; if
        O1 also passes, O1 takes precedence.

bar
    Same as O1. Run ss.run(fn, d) with every name held, monthly, net of 10bp India / 3bp US, top-N(all)
    + 10% cap, on nifty_50 and dow_30 with holdout=True, against the shipped HRP at the same code
    state.
    PASS if either:
    (i) CAGR/vol is higher in all six cells (E1/E2/E3 x Nifty, Dow) with vol up by no more than 0.5
        pt in any cell; or
    (ii) CAGR is higher in all six cells with vol up by at most 0.5 pt in every cell.
    Report, not ruled on: turnover per universe, ETF27, Nifty top-30, point-in-time Dow E3, and
    per-cell margins against the 0.25-0.29 era reorder noise floor.
    Parameters are fixed.

needs_regeneration  false

IMPLEMENTATION NOTES (how the registered text is read; written before any variant was run)
───────────────────────────────────────────────────────────────────────────────────────────────────
· Code state: no input fix (B1-B3) has landed in nco.py, so the ruled baseline is
  ss.baselines(d)["HRP"] — the stored raw HRP weights, built by nco.compute_nco_portfolio on a
  253-snapshot window (research/style_search.build).
· The in-nco path. Every book here is produced by nco.compute_nco_portfolio(history, prices, ...,
  method="HRP", max_pos_pct=1.0) on light (date, symbol/price) snapshots taken from the same
  stale-repaired research snapshots (sb.snapshots), exactly as sb.raw does. A variant's weight
  vector over today's est_names is computed by the reference functions below (hrp_staggered,
  hrp_anchored) and enters the pipeline in place of the hrp_weights(cov, corr) call — nco.hrp_weights
  is swapped for that one call, after checking the covariance it is handed is today's — so the
  top-N / >1e-9 drop / renormalisation / cap / units / risk diagnostics (on today's covariance) are
  the shipped code. Memoised but output-identical copies of nco.build_returns_matrix and
  nco.cluster_assets serve repeated calls on the same inputs within a month (CPU only).
· The panel: "the ~400-session panel" is the last PANEL = 400 snapshots ending at the rebalance
  date (fewer in the first 16-17 months of the backtest, where the snapshots begin Oct 2006).
· Zero-change controls (reproduces_current), run through the same path:
    Z0  the shipped book on the 253-snapshot research window — must equal the stored raw HRP;
    Z1  the shipped book on the 400-session panel (what app.py runs);
    Z1i hrp_staggered(K=1) through the swapped pipeline — must equal Z1 exactly;
    and bisect_halves(cov, shipped_order(corr)) must equal nco.hrp_weights(cov, corr) every month
    (O2's machinery), with HRP-Q / HRP-H equal to Z1 on their anchor months.
  If Z1 differs from Z0 (the 400-session panel pads a price from before the 253-row window across
  unpriced closes: pandas pct_change's default fill), each variant is read twice: against the
  ruled baseline ss.baselines(d)["HRP"] (the bar as written) and against Z1 (same panel, isolates
  the variant's own effect). Both verdicts are reported; the bar is ruled on the first.
· O2 anchors: quarter start = the first day of the run date's calendar quarter; half-year = Jan 1 /
  Jul 1. The anchor window is read literally ("history up to and including the anchor", the
  shipped 252 lookback and cov_ok); when the panel does not reach the anchor's 253 rows (only the
  first months of the backtest) the shipped rules decide, and a column set that differs falls back
  to today's order.
· ETF27 uses the snapshots without ETF_YOUNG, as style_search.build. The point-in-time Dow filters
  every snapshot of the 400-session panel to the members of the rebalance day (as
  style_search_pit.build does for its 253-row window) and holds only them (the ctx.priced patch).
· Noise floor for the margin flags: the largest era CAGR spread across the 7 random universe
  orders in hrp/logs/perm_bt.log — Nifty 0.25 (E3), Dow 0.29 (E3).
· Trials: every weight configuration run is counted, controls included.

Run:  AUDIT_HRP_CACHE=<dir> python research/audit_hrp.py      (cache optional; ~20 min, 1 core)

RESULT (2026-10-05) — O1-A (HRP-S3) PASSES both rules and is the registered pick.
───────────────────────────────────────────────────────────────────────────────────────────────────
Verdicts: O1-A passes (i) and (ii). O1-B (HRP-S4) passes (ii) only. O2-B (HRP-H) passes (i) and
(ii), but O1 takes precedence. O1-C (canon) and O2-A (quarter) fail. The precedence order is C,
then A, then B, and O1 before O2; C failed, so A is the pick. The passes are thin. Most of the
return margins sit inside the universe-reorder noise floor, and no era's paired t reaches 2. The
effect that is not noise is turnover: it falls by 40-46% in every universe.

code state   nco.py has not changed since 73756ff, and no B1-B3 input fix has landed. The ruled
             baseline is therefore ss.baselines(d)["HRP"].
trials       8 weight configurations: 3 zero-change controls (Z0, Z1, Z1i) and 5 variants. Each ran
             on Nifty 50, Dow 30, ETF27, PIT Dow and Nifty top-30, which makes 40 backtests. An
             earlier run of the same file and configurations was interrupted after it finished
             (scratch hrpopp/run1.log). This re-run reproduced it digit for digit and is not counted
             again.

zero-change (reproduces_current: yes)
    Z0   compute_nco_portfolio(method="HRP") on the 253-snapshot window, compared with the stored
         raw HRP: max |Δw| = 0.0 in every month (Nifty 236, Dow 236, ETF27 19, PIT Dow 93). Its
         backtest matches ss.baselines HRP to the last digit: Nifty 20.21/17.40/20.10, Dow
         12.48/15.40/11.06, ETF27 15.80, PIT Dow E3 9.53.
    Z1i  hrp_staggered(K=1) through the swapped pipeline, compared with Z1 (the shipped book on the
         400-session panel): max |Δw| = 5.6e-17. bisect_halves(shipped_order) against
         nco.hrp_weights: 0.0 in every month. Each variant through the pipeline against its direct
         reference: ≤ 8.3e-17. canon_order against scratch hrpsk/canon.py on 30 random problems:
         0.0; permutation invariance holds to 5.6e-17.
    Z1   equals Z0 on Dow, ETF27 and PIT Dow. On Nifty, 9 months differ (2009-04..06 and
         2010-02..07, max |Δw| 0.019). On the 400-session panel, pct_change pads a price from
         before the 252-row window across stale-repaired (unpriced) closes. In 2009-04..06 that
         pads BAJAJ-AUTO's 45 unpriced closes: the panel keeps 252 rows, where the 253-row window
         drops to 207. In 2010-02..07, NESTLEIND's padded stale span lifts it over the 80% coverage
         rule, so it enters the estimation set. Returns on the common rows are identical. Z1's Nifty
         E1 is 20.26 (+0.05); E2 and E3 are identical. Every variant runs on the 400-session panel
         and carries the same effect; the "vs Z1" reading isolates the variant itself. The same
         effect explains why O1-A's Nifty E1 here (20.39) is above the auditors' harness figure
         (20.32), because the harness cuts the 253-row window before pct_change. Every other O1-A
         cell matches the harness (Nifty 17.72/20.14, Dow 12.59/15.57/11.38, turnover 0.73/0.52).

ruled cells — net CAGR %: variant (gap vs shipped HRP, paired t of the monthly gap), Δvol, Δret/vol
  shipped HRP   Nifty 20.21 / 17.40 / 20.10   vol 23.69 / 13.04 / 17.54   r/v 0.901 / 1.302 / 1.140
                Dow   12.48 / 15.40 / 11.06   vol 15.84 / 11.15 / 14.87   r/v 0.825 / 1.347 / 0.783
                     E1                              E2                              E3
  O1-A S3     N 20.39 (+0.18 t+0.6) +0.09 +.004  17.72 (+0.32 t+1.0) −0.01 +.022  20.14 (+0.04 t+0.0) −0.18 +.012
              D 12.59 (+0.11 t+0.4) −0.01 +.007  15.57 (+0.17 t+0.9) +0.02 +.011  11.38 (+0.32 t+1.3) +0.05 +.017
              → rule (i) PASS, rule (ii) PASS
  O1-B S4     N 20.27 (+0.06 t+0.3) +0.20 −.003  18.01 (+0.60 t+1.7) −0.03 +.043  20.16 (+0.06 t+0.0) −0.23 +.016
              D 12.55 (+0.07 t+0.3) +0.00 +.004  15.59 (+0.19 t+0.9) +0.04 +.011  11.45 (+0.39 t+1.3) +0.10 +.019
              → (i) fails (Nifty E1 r/v −.003); (ii) PASS
  O1-C canon  N 19.83 (−0.37 t−0.4) +0.56 −.028  17.97 (+0.57 t+1.7) +0.15 +.024  20.30 (+0.20 t+0.2) −0.37 +.030
              D 12.84 (+0.36 t+1.0) +0.19 +.012  15.19 (−0.21 t−0.6) +0.15 −.033  11.53 (+0.46 t+1.4) +0.23 +.018
              → FAIL (Nifty E1 −0.37 at vol +0.56 > 0.5; Dow E2 −0.21)
  O2-A Q      N 20.32 (+0.11 t+0.2) −0.17 +.009  17.49 (+0.09 t+0.2) −0.01 +.007  20.23 (+0.13 t+0.2) −0.23 +.019
              D 12.36 (−0.12 t−0.5) −0.07 −.004  15.61 (+0.21 t+0.9) +0.04 +.012  11.14 (+0.08 t+0.2) −0.09 +.008
              → FAIL (Dow E1 −0.12)
  O2-B H      N 20.42 (+0.21 t+0.4) +0.07 +.006  17.45 (+0.05 t+0.1) −0.03 +.005  20.28 (+0.18 t+0.2) −0.39 +.031
              D 12.90 (+0.42 t+1.6) −0.05 +.026  15.94 (+0.54 t+1.8) −0.02 +.045  11.60 (+0.54 t+1.1) −0.01 +.033
              → rule (i) PASS, rule (ii) PASS (O1 takes precedence)
  Against Z1 (same 400-session panel) only Nifty E1 changes: S3 +0.13 (r/v +.002), S4 +0.01
  (r/v −.006), canon −0.42, Q +0.06, H +0.16. Every verdict is the same.

reported, not ruled
  full span      Nifty CAGR 19.31 → S3 19.48 (t+0.9), S4 19.54, canon 19.42, Q 19.42, H 19.46; vol 18.77 →
                 18.75 / 18.78 / 18.93 / 18.62 / 18.67.
                 Dow 12.87 → 13.07 (t+1.5), 13.09, 13.10, 12.92, 13.37 (t+2.4); vol 14.16 → 14.18 / 14.21 /
                 14.35 / 14.11 / 14.13.
  turnover /yr   Nifty 1.23 → S3 0.73, S4 0.66, canon 0.72, Q 0.84, H 0.75
                 Dow   0.91 → 0.52, 0.47, 0.54, 0.65, 0.56
                 ETF27 0.92 → 0.66, 0.61, 0.70, 0.77, 0.74
                 PIT   1.10 → 0.59, 0.52, 0.58, 0.75, 0.67
                 Nifty top-30  1.66 → 0.97, 0.85, 0.99, 1.10, 0.99
  ETF27          (Mar 2025 →, 19 months) 15.80 → S3 15.76 (−0.05 t−0.05), S4 15.74 (−0.06), canon 14.32
                 (−1.48 t−1.75), Q 15.46 (−0.34), H 16.03 (+0.23); vol +0.10/+0.06/+0.40/+0.05/+0.19.
  PIT Dow E3     9.53 → S3 9.94 (+0.41 t+1.35), S4 9.97 (+0.44 t+1.44), canon 10.03 (+0.50 t+1.40),
                 Q 9.54 (+0.01), H 9.65 (+0.12); vol −0.03/−0.02/+0.13/−0.05/−0.11.
  Nifty top-30   shipped 20.62/17.28/18.31. S3 +0.37/+0.51/+0.70 (vol −0.03/−0.15/−0.36); S4
                 +0.10/+0.65/+0.41; canon −0.13/+0.77/+0.76 (E1 vol +0.38); Q +0.31/−0.26/+0.42;
                 H +0.61/−0.36/−0.12.
  noise floor    The registered floors are 0.25 (Nifty) and 0.29 (Dow): the largest era CAGR spread
                 across 7 random universe orders (scratch hrp/logs/perm_bt.log). Cells below the
                 floor:
                 O1-A, 4 of 6: Nifty E1 +0.18, E3 +0.04; Dow E1 +0.11, E2 +0.17. Only Nifty E2 +0.32
                 and Dow E3 +0.32 clear it.
                 O1-B, 4 of 6: Nifty E1 +0.06, E3 +0.06; Dow E1 +0.07, E2 +0.19.
                 O2-B, 3 of 6: all of Nifty (+0.21/+0.05/+0.18). Dow clears in all three
                 (+0.42/+0.54/+0.54).
                 Floors set era by era (perm_bt.log spreads: Nifty 0.07/0.21/0.25, Dow 0.11/0.10/0.29)
                 leave O1-A below only in Nifty E3 (+0.04) and Dow E1 (+0.107 vs 0.11), and O2-B below
                 in Nifty E2 and E3.
                 Both O1-A passes therefore rest partly on margins below the floor. The narrowest is
                 rule (i)'s Nifty E1 ret/vol margin: +0.004, or +0.002 against Z1.

reading
    O1-A passes, as the pre-registration's disclosure predicted. Those in-sample harness numbers
    were measured before registration, so this run confirms them through the in-nco path rather
    than discovering anything. The return gain is small and not significant: full span +0.18 on
    Nifty (t 0.9) and +0.20 on Dow (t 1.5), with vol unchanged within ±0.2 in every cell. The
    turnover cut is not in doubt: 40-46% lower in every universe, PIT and top-30 included. At
    10bp/3bp the cut saves only about 0.05 pt/yr on Nifty and 0.01 on Dow, so most of the CAGR gap
    is gross. The averaged target holds slightly better weights, not just cheaper ones.
    Out-of-sample-style checks point the same way: PIT Dow E3 +0.41, and Nifty top-30 positive in
    every era with lower vol. ETF27 is −0.05 over 19 months, which is noise.
    O2-B is the strongest variant on Dow (+0.42/+0.54/+0.54, full-span t 2.4), but its Nifty
    margins are all below the floor, its top-30 is negative in E2 and E3, ETF27 is +0.23 and PIT
    +0.12. Its quarter-anchored sibling O2-A fails (Dow E1 −0.12). The gain does not survive a
    change of anchor period, which fits calendar-phase luck better than a mechanism. The
    registered precedence puts O1 first anyway.
    O1-C fails. Inside the staggered average, the canonical orientation (B5's order-dependence fix)
    costs Nifty E1 0.37 at +0.56 vol, Dow E2 0.21 and ETF27 1.48. It should not be bundled with S3.
    Shipping O1-A as registered: in compute_nco_portfolio's HRP branch, average the shipped
    hrp_weights over windows ending 0/21/42 sessions back inside the last 400 sessions, restricted
    to today's est_names (mean with skipna, fillna(0), renormalise). Record attrs nco_hrp_windows.
    solver_cov and the risk diagnostics stay on today's covariance. All three windows need ≥ 295
    sessions, which the app's ~400-session panel holds. A shorter panel uses fewer windows, and
    when only j=0 fits (the 100-day fallback) the result is exactly today's HRP.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import nco                                        # noqa: E402
import style_blends as sb                         # noqa: E402
import style_search as ss                         # noqa: E402
import style_search_pit as P                      # noqa: E402

PANEL, LOOK, STEP = 400, 252, 21
CACHE = os.environ.get("AUDIT_HRP_CACHE")
FLOOR = {"nifty_50": 0.25, "dow_30": 0.29}
STOCK = ("nifty_50", "dow_30")
ERAS3 = [e for e in ss.ERAS]
ORIG_HRP = nco.hrp_weights
ORIG_BRM = nco.build_returns_matrix
ORIG_CLU = nco.cluster_assets
VOL_TOL = 0.5


# ── the shipped pieces, copied so the variants can recombine them (identity checked every month) ──
def cov_ok(R: pd.DataFrame) -> bool:
    """The shipped estimability rule in nco.compute_nco_portfolio."""
    n = R.shape[1]
    return (not R.empty and n >= 2 and len(R) >= nco.MIN_OBS
            and len(R) >= nco.MIN_OBS_PER_ASSET * n)          # T >= n, before and after MIN_OBS_PER_ASSET 4 -> 1


def sample(R: pd.DataFrame):
    X = R.to_numpy(dtype=float)
    return np.cov(X, rowvar=False), np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)


def _linkage(corr: np.ndarray) -> np.ndarray:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    return linkage(squareform(nco.correlation_distance(corr), checks=False), method="single")


def shipped_order(corr: np.ndarray) -> list:
    """nco.hrp_weights' quasi-diagonal leaf order (single linkage, shipped orientation)."""
    Z = _linkage(corr).astype(int)
    srt = pd.Series([Z[-1, 0], Z[-1, 1]])
    num = Z[-1, 3]
    while srt.max() >= num:
        srt.index = range(0, srt.shape[0] * 2, 2)
        df0 = srt[srt >= num]
        i, j = df0.index, df0.values - num
        srt[i] = Z[j, 0]
        srt = pd.concat([srt, pd.Series(Z[j, 1], index=i + 1)]).sort_index()
    return [int(x) for x in srt.tolist()]


def cluster_var(cov: np.ndarray, idx: list) -> float:
    sub = cov[np.ix_(idx, idx)]
    w = nco.inverse_variance(sub)
    return float(w @ sub @ w)


def bisect_halves(cov: np.ndarray, order: list) -> np.ndarray:
    """nco.hrp_weights' recursive bisection: halves of the ordered list, alpha by cluster variance."""
    n = cov.shape[0]
    w = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[a:b] for c in clusters
                    for a, b in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters) - 1, 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = cluster_var(cov, c0), cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1) if (v0 + v1) > 1e-18 else 0.5
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    total = w.sum()
    return w / total if total > 1e-12 else np.full(n, 1.0 / n)


def canon_order(cov: np.ndarray, corr: np.ndarray, names: list) -> list:
    """B5's canonical orientation: at each merge the child with more leaves first; on a size tie the
    lower inverse-variance cluster variance; then the child holding the lexicographically smallest
    symbol. Same single-linkage tree as shipped; only the child order differs."""
    n = cov.shape[0]
    Z = _linkage(corr)
    leaves = {i: [i] for i in range(n)}
    for k, row in enumerate(Z):
        la, lb = leaves.pop(int(row[0])), leaves.pop(int(row[1]))
        key = lambda l: (-len(l), cluster_var(cov, l), min(names[i] for i in l))   # noqa: E731
        leaves[n + k] = la + lb if key(la) <= key(lb) else lb + la
    return leaves[2 * n - 2]


def hrp_canon(cov: np.ndarray, corr: np.ndarray, names: list) -> np.ndarray:
    if cov.shape[0] == 1:
        return np.ones(1)
    try:
        order = canon_order(cov, corr, list(names))
    except Exception:
        return nco.inverse_variance(cov)
    return bisect_halves(cov, order)


# ── the variants (reference forms; `builder` defaults to nco.build_returns_matrix) ────────────────
def hrp_staggered(history, prices: dict, k: int, step: int = STEP, canon: bool = False, builder=None):
    """O1: the mean of HRP fits on 252-row windows ending 0, step, … (k-1)·step sessions back, inside
    the last PANEL sessions. Returns (weights over today's est_names, info) or (None, info)."""
    builder = builder or nco.build_returns_matrix
    hist = list(history)[-PANEL:]
    R0 = builder(hist, list(prices), LOOK)
    if not cov_ok(R0):
        return None, {"windows": 0}
    est = list(R0.columns)
    parts, used = [], []
    for j in range(k):
        end = len(hist) - step * j
        if j > 0 and end < LOOK + 1:
            continue
        R = R0 if j == 0 else builder(hist[:end], est, LOOK)
        if not cov_ok(R):
            continue
        cov, corr = sample(R)
        w = hrp_canon(cov, corr, list(R.columns)) if canon else ORIG_HRP(cov, corr)
        parts.append(pd.Series(w, index=R.columns).reindex(est))
        used.append(j)
    W = pd.concat(parts, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    return W / W.sum(), {"windows": len(used), "used": tuple(used)}


def hrp_anchored(history, prices: dict, months: int, builder=None):
    """O2: the shipped leaf order of the window ending at the period's first session (months=3: the
    calendar quarter, 6: the half-year), bisected on today's covariance."""
    builder = builder or nco.build_returns_matrix
    hist = list(history)
    R0 = builder(hist, list(prices), LOOK)
    if not cov_ok(R0):
        return None, {"src": "none"}
    est = list(R0.columns)
    cov, corr = sample(R0)
    run = pd.Timestamp(hist[-1][0])
    start = pd.Timestamp(run.year, ((run.month - 1) // months) * months + 1, 1)
    ia = next(i for i, (t, _) in enumerate(hist) if pd.Timestamp(t) >= start)
    order, src = None, "today (anchor = run date)"
    if ia < len(hist) - 1:
        Ra = builder(hist[:ia + 1], est, LOOK)
        if set(Ra.columns) == set(est) and cov_ok(Ra):
            Ra = Ra[est]
            order, src = shipped_order(sample(Ra)[1]), "anchor"
        else:
            src = "today (anchor window: " + ("names differ)" if set(Ra.columns) != set(est) else "cov_ok fails)")
    try:
        order = order if order is not None else shipped_order(corr)
    except Exception:
        return pd.Series(ORIG_HRP(cov, corr), index=est), {"src": "shipped fallback"}
    return pd.Series(bisect_halves(cov, order), index=est), {"src": src, "anchor": pd.Timestamp(hist[ia][0])}


VARIANTS = {
    "O1-A HRP-S3":       lambda h, p, b: hrp_staggered(h, p, 3, builder=b),
    "O1-B HRP-S4":       lambda h, p, b: hrp_staggered(h, p, 4, builder=b),
    "O1-C HRP-S3-canon": lambda h, p, b: hrp_staggered(h, p, 3, canon=True, builder=b),
    "O2-A HRP-Q":        lambda h, p, b: hrp_anchored(h, p, 3, builder=b),
    "O2-B HRP-H":        lambda h, p, b: hrp_anchored(h, p, 6, builder=b),
}
CONTROLS = ("Z0 shipped/253", "Z1 shipped/400", "Z1i S1/400")


# ── the pipeline ──────────────────────────────────────────────────────────────────────────────────
class Memo:
    """Output-identical memoised build_returns_matrix / cluster_assets for one rebalance date."""

    def __init__(self):
        self.r, self.c = {}, {}

    def brm(self, history, symbols=None, lookback=252):
        h = list(history)
        key = (len(h), pd.Timestamp(h[0][0]) if h else None, pd.Timestamp(h[-1][0]) if h else None,
               tuple(symbols or ()), lookback)
        if key not in self.r:
            self.r[key] = ORIG_BRM(h, symbols, lookback)
        return self.r[key]

    def clu(self, corr, max_clusters=nco.MAX_CLUSTERS):
        key = (corr.shape, corr.tobytes(), max_clusters)
        if key not in self.c:
            self.c[key] = ORIG_CLU(corr, max_clusters)
        return self.c[key]


def prices_of(snap: pd.DataFrame) -> dict:
    return {s: float(p) for s, p in zip(snap["symbol"], pd.to_numeric(snap["price"], errors="coerce"))
            if np.isfinite(p) and p > 0}


def pipeline(history, prices: dict, memo: Memo, w: pd.Series | None = None) -> pd.Series:
    """nco.compute_nco_portfolio's HRP book over every name, uncapped (as sb.raw); with `w`, that
    vector replaces the hrp_weights(cov, corr) call on today's covariance."""
    nco.build_returns_matrix, nco.cluster_assets = memo.brm, memo.clu
    if w is not None:
        cov0 = sample(memo.brm(list(history), list(prices), LOOK))[0]

        def swapped(cov, corr):
            assert cov.shape == cov0.shape and np.array_equal(cov, cov0), "not today's covariance"
            return w.to_numpy(dtype=float)
        nco.hrp_weights = swapped
    try:
        book = nco.compute_nco_portfolio(history, prices, sb.CAPITAL, len(prices), method="HRP",
                                         max_pos_pct=1.0)
    finally:
        nco.build_returns_matrix, nco.cluster_assets, nco.hrp_weights = ORIG_BRM, ORIG_CLU, ORIG_HRP
    if book.empty:
        return pd.Series(dtype=float)
    x = pd.to_numeric(book["weightage_pct"], errors="coerce")
    return pd.Series((x / x.sum()).to_numpy(), index=book["symbol"].astype(str).to_numpy())


def _gap(a: pd.Series, b: pd.Series) -> float:
    idx = a.index.union(b.index)
    return float((a.reindex(idx, fill_value=0.0) - b.reindex(idx, fill_value=0.0)).abs().max()) if len(idx) else 0.0


def light(u: str) -> list:
    if u == "dow_pit":
        snaps = sb.unstale(pickle.load(open(os.path.join(HERE, "cvg_reweight_dow_pit.pkl"), "rb")))
    else:
        snaps = sb.snapshots("etf_book" if u == "etf_27" else u)
    if u == "etf_27":
        snaps = [(t, s[~s["symbol"].isin(ss.ETF_YOUNG)].reset_index(drop=True)) for t, s in snaps]
    return [(pd.Timestamp(t), s[["symbol", "price"]].reset_index(drop=True)) for t, s in snaps]


def weights(u: str, d: dict) -> dict:
    """Every configuration's raw weights for every rebalance month, the exactness gaps and the info."""
    L = light(u)
    assert [t for t, _ in L] == list(d["px"].index), "snapshots and panel calendars differ"
    pos = {t: i for i, (t, _) in enumerate(L)}
    members = P.member if u == "dow_pit" else None
    W = {k: {} for k in (*CONTROLS, *VARIANTS)}
    info, gaps = [], []
    t0 = time.time()
    for m, a in enumerate(d["months"][:-1]):
        i = pos[a]
        h4 = L[max(0, i - PANEL + 1): i + 1]
        if members is not None:
            h4 = [(t, s[s["symbol"].map(lambda x: members(x, a))].reset_index(drop=True)) for t, s in h4]
        h2 = h4[-(LOOK + 1):]
        prices = prices_of(h4[-1][1])
        memo = Memo()
        W["Z0 shipped/253"][a] = pipeline(h2, prices, memo)
        W["Z1 shipped/400"][a] = pipeline(h4, prices, memo)
        w1, _ = hrp_staggered(h4, prices, 1, builder=memo.brm)
        W["Z1i S1/400"][a] = pipeline(h4, prices, memo, w1)
        R0 = memo.brm(h4, list(prices), LOOK)
        cov0, corr0 = sample(R0)
        row = dict(date=a, n_est=R0.shape[1], obs=len(R0),
                   g_z0_stored=_gap(W["Z0 shipped/253"][a], d["raw"]["HRP"][a]),
                   g_z1i_z1=_gap(W["Z1i S1/400"][a], W["Z1 shipped/400"][a]),
                   g_z1_z0=_gap(W["Z1 shipped/400"][a], W["Z0 shipped/253"][a]),
                   g_bisect=float(np.abs(bisect_halves(cov0, shipped_order(corr0)) - ORIG_HRP(cov0, corr0)).max()))
        for k, f in VARIANTS.items():
            w, inf = f(h4, prices, memo.brm)
            W[k][a] = pipeline(h4, prices, memo, w)
            ww = w[w > 1e-9]
            row[f"g_direct::{k}"] = _gap(W[k][a], ww / ww.sum())
            row[f"info::{k}"] = inf.get("windows", inf.get("src"))
            if k.startswith("O2"):
                row[f"g_ship::{k}"] = _gap(W[k][a], W["Z1 shipped/400"][a])
        info.append(row)
        if m % 40 == 0:
            print(f"   {u} {a:%Y-%m} ({m + 1}/{len(d['months']) - 1}) {time.time() - t0:.0f}s", flush=True)
    return dict(W=W, info=pd.DataFrame(info).set_index("date"))


# ── measurement ───────────────────────────────────────────────────────────────────────────────────
def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan


def cells(r: pd.DataFrame, ref: pd.DataFrame, eras) -> dict:
    out = {}
    for e, a, b in eras:
        x, y = ss._era(r, a, b), ss._era(ref, a, b)
        m, mb = ss.metrics(x), ss.metrics(y)
        g = x["ret"] - y["ret"].reindex(x.index)
        out[e] = dict(cagr=m["cagr"], vol=m["vol"], rv=m["ret_vol"], dd=m["maxdd"], to=m["turnover"],
                      d_cagr=m["cagr"] - mb["cagr"], d_vol=m["vol"] - mb["vol"], d_rv=m["ret_vol"] - mb["ret_vol"],
                      gap=g.mean() * 1200, t=_t(g), months=m["months"])
    return out


def runs(u: str, d: dict, W: dict, n=None, pit: bool = False) -> dict:
    init = ss.Ctx.__init__
    if pit:
        def pit_init(self, data, a):
            init(self, data, a)
            self.priced = pd.Index([s for s in self.priced if P.member(s, a)])
        ss.Ctx.__init__ = pit_init
    try:
        out = {"HRP (ss.baselines)": ss.baselines(d, n)["HRP"]}
        for k in W:
            out[k] = ss.run(lambda c, k=k: W[k][c.date], d, n)
    finally:
        ss.Ctx.__init__ = init
    return out


def _eras_for(u: str):
    if u == "etf_27":
        return [("window", None, None)]
    if u == "dow_pit":
        return [("E3", "2020-01-01", None)]
    return list(ss.ERAS) + [("FULL", None, None)]


def table(u: str, R: dict, ref_key: str) -> pd.DataFrame:
    rows = []
    for k, r in R.items():
        for e, c in cells(r, R[ref_key], _eras_for(u)).items():
            rows.append(dict(config=k, era=e, **c))
    return pd.DataFrame(rows)


def bar(tabs: dict) -> pd.DataFrame:
    """The registered bar per configuration, over the six stock cells."""
    out = []
    configs = [k for k in tabs["nifty_50"]["config"].unique() if k != "HRP (ss.baselines)"]
    for k in configs:
        rec = dict(config=k)
        rv_ok, cg_ok, vol_ok, below = True, True, True, []
        for u in STOCK:
            t = tabs[u]
            for e in ("E1", "E2", "E3"):
                c = t[(t.config == k) & (t.era == e)].iloc[0]
                rv_ok &= bool(c.d_rv > 0)
                cg_ok &= bool(c.d_cagr > 0)
                vol_ok &= bool(c.d_vol <= VOL_TOL)
                if c.d_cagr < FLOOR[u]:
                    below.append(f"{u.split('_')[0]} {e} {c.d_cagr:+.2f}")
        rec.update(rule_i=rv_ok and vol_ok, rule_ii=cg_ok and vol_ok, PASS=(rv_ok or cg_ok) and vol_ok,
                   rv_up=rv_ok, cagr_up=cg_ok, vol_within=vol_ok, below_floor="; ".join(below))
        out.append(rec)
    return pd.DataFrame(out).set_index("config")


def _cached(name: str, build):
    if CACHE:
        p = os.path.join(CACHE, f"audit_hrp_{name}.pkl")
        if os.path.exists(p):
            return pickle.load(open(p, "rb"))
        os.makedirs(CACHE, exist_ok=True)
        obj = build()
        pickle.dump(obj, open(p, "wb"))
        return obj
    return build()


def load(u: str) -> dict:
    return pickle.load(open(P.PKL, "rb")) if u == "dow_pit" else ss.load(u, holdout=True)


def show(u: str, T: pd.DataFrame, label: str) -> None:
    cols = ["cagr", "vol", "rv", "d_cagr", "d_vol", "d_rv", "gap", "t", "to"]
    piv = T.set_index(["config", "era"])[cols]
    print(f"\n== {u} · {label}", flush=True)
    print(piv.round(3).to_string(), flush=True)


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    res = {}
    for u in ("nifty_50", "dow_30", "etf_27", "dow_pit"):
        t0 = time.time()
        d = load(u)
        M = _cached(f"weights_{u}", lambda u=u, d=d: weights(u, d))
        I = M["info"]
        print(f"\n== {u}: {len(I)} rebalances · weights {time.time() - t0:.0f}s", flush=True)
        print("   exactness (max |Δw| over months): "
              f"Z0 vs stored raw HRP {I.g_z0_stored.max():.1e} · Z1i vs Z1 {I.g_z1i_z1.max():.1e} · "
              f"bisect(shipped_order) vs hrp_weights {I.g_bisect.max():.1e} · "
              + " · ".join(f"{k.split()[0]} pipeline vs direct {I[f'g_direct::{k}'].max():.1e}" for k in VARIANTS),
              flush=True)
        dz = I.g_z1_z0
        print(f"   Z1 (400-session panel) vs Z0 (253 window): {int((dz > 1e-12).sum())} months differ, max |Δw| "
              f"{dz.max():.3f}: " + ", ".join(f"{a:%Y-%m}" for a in dz.index[dz > 1e-12]), flush=True)
        for k in VARIANTS:
            vc = I[f"info::{k}"].value_counts().to_dict()
            extra = ""
            if k.startswith("O2"):
                anch = I[f"g_ship::{k}"]
                extra = f" · equal to Z1 (≤1e-12) in {int((anch <= 1e-12).sum())} months"
            print(f"   {k}: {vc}{extra}", flush=True)
        R = runs(u, d, M["W"], pit=(u == "dow_pit"))
        res[u] = dict(R=R, I=I, T=table(u, R, "HRP (ss.baselines)"), Tz=table(u, R, "Z1 shipped/400"))
        if u == "nifty_50":
            R30 = runs(u, d, M["W"], n=30)
            res[u]["T30"] = table(u, R30, "HRP (ss.baselines)")
            res[u]["T30z"] = table(u, R30, "Z1 shipped/400")
        show(u, res[u]["T"], "vs ss.baselines HRP (ruled)")
        show(u, res[u]["Tz"], "vs Z1, the shipped book on the same 400-session panel")
        if "T30" in res[u]:
            show(u, res[u]["T30"], "top-30 · vs ss.baselines(d, 30) HRP")
        print(f"   [{time.time() - t0:.0f}s]", flush=True)
    tabs = {u: res[u]["T"] for u in STOCK}
    tabz = {u: res[u]["Tz"] for u in STOCK}
    print("\n══ THE BAR (vs ss.baselines HRP, ruled) ═════════════════════════════════════════", flush=True)
    print(bar(tabs).to_string(), flush=True)
    print("\n══ THE BAR read against Z1 (same 400-session panel; reported) ═══════════════════", flush=True)
    print(bar(tabz).to_string(), flush=True)
    if CACHE:
        pickle.dump({u: {k: v for k, v in r.items() if k != "R"} | {"R": r["R"]} for u, r in res.items()},
                    open(os.path.join(CACHE, "audit_hrp_results.pkl"), "wb"))


if __name__ == "__main__":
    main()
