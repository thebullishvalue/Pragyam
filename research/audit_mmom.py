"""
research/audit_mmom.py — MMOM opportunity test: a two-sided volatility scale (MM-O1) and overlapping-formation
momentum ranks (MM-O2), each measured through the product code against the shipped v12.1 Managed Momentum.

PRE-REGISTRATION (copied from the coordinator's brief without changes, before any variant was run; no variants
beyond MM-O1 BSC_UP2 / BSC_UP15 and MM-O2 JT3 / JT3R)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
id      MM-O1
title   Two-sided volatility scale: let the overlay grow above λ when its own volatility is below its long-run
        median

mechanism
    The shipped scale is min(1, median/now), so it only de-risks. Barroso & Santa-Clara and Moreira & Muir
    target volatility in both directions. In calm regimes the momentum overlay's risk-adjusted payoff is
    higher, so up-scaling raises the overlay's share exactly then.

    The hypothesis comes from existing measurements:
    - A constant λ=2 (L2) beat shipped in every Nifty era and in Dow E1/E2, but lost the point-in-time Dow
      (10.88 vs 11.28).
    - Up-scaling only in calm months (auditor 3, full span only, no era split) gave Nifty +0.28 (t 2.48), Dow
      +0.08, PIT E3 +0.01, ETF +0.14, country -0.03.

    The question is whether the gain holds in every era cell without the PIT loss. Prior: modest; the Dow
    margin is small. Not in the rejected research: the 'no scale' and λ variants were measured, but the
    two-sided scale was not.

literature
    Barroso & Santa-Clara (2015, JFE) 'Momentum has its moments'; Moreira & Muir (2017, JF)
    'Volatility-managed portfolios'; Daniel & Moskowitz (2016) for the unchanged gate.

variants
    BSC_UP2: strength = λ · gate · min(2.0, median/now). median and now are exactly the shipped mmom_scale
        quantities: the expanding median of the unit overlay's 126-row realised vol at month starts up to and
        including the run date, and that vol today. When fewer than MMOM_MIN_VOL_MONTHS=7 readings exist, or
        now is non-finite or <= 0, the multiplier is 1. Unchanged: λ=1, gate (504 rows, EW market), 12-1
        ranks, floor 0.25, top-N/cap/units.
    BSC_UP15: identical to BSC_UP2 with the cap 1.5: strength = λ · gate · min(1.5, median/now).

bar
    Ruling, on the shipped v12.1 code (no MM-B fixes applied, so the baseline is the published SHIP): monthly
    month-start books, every name held, net of 10bp India / 3bp US.

    The variant must beat SHIP on net CAGR, compared unrounded, in each of E1 / E2 / E3 on both:
    - Nifty 50 (SHIP 21.23 / 21.34 / 24.17)
    - Dow 30 (SHIP 14.63 / 18.88 / 15.31)

    It must also be >= SHIP on point-in-time Dow E3 in BOTH:
    - the legacy run (research/mmom_ship.py (d), SHIP 11.28)
    - the members-only run (price_history restricted to the day's members as in scratch
      mmlogic2/pit_members.py, SHIP 11.05)

    The ETF 27 result is reported but not ruled on.

    Implement via the product path: research/mmom_ship.weights with nco.mmom_overlay/mmom_scale patched in
    memory only, no repo edits. Scratch mmlp/lib.py (make(..., strength_fn=...), as in mmlp/scale2.py) may be
    used if SHIP reproduces with gap 0 on every panel.

    Also report, without ruling: t vs SHIP, vol, turnover, mean strength, mean nco_mmom_floored, and the
    Nifty N=30 and Dow N=20 cut books.

    No parameter changes. If both variants pass, the tester reports both and the coordinator picks the
    gentler one (1.5).

needs_regeneration  false

──────────────────────────────────────────────────────────────────────────────────────────────────────────────
id      MM-O2
title   Overlapping-formation momentum ranks (Jegadeesh-Titman K=3): size the overlay by the average of the last
        three monthly 12-1 rankings

mechanism
    Jegadeesh & Titman show the 12-1 winner portfolio keeps earning for several months after formation, and
    form overlapping K-month portfolios. Averaging the ranks formed at t, t-21 and t-42 rows:
    - averages out the noise of a single formation date;
    - cuts the overlay's turnover (costs are included);
    - keeps the 12-1 formation window itself unchanged.

    That last point is what separates it from the rejected 6-1 / 12-7 formation-length variants: here the
    holding-period overlap changes, not the formation. The re-ranked form restores the [-1, 1] dispersion, so
    the test is not confounded by a smaller effective λ (L0.5 lost everywhere).

    Prior: low; there is no existing measurement in any configuration. It is included as a cheap,
    mechanism-backed test that touches every era.

literature
    Jegadeesh & Titman (1993, JF): overlapping J/K portfolios; Novy-Marx & Velikov (2016, RFS) 'A taxonomy of
    anomalies and their trading costs': signal smoothing to cut turnover costs.

variants
    JT3: u_i = mean over k in {0, 1, 2} of mmom_ranks(12-1 momentum as of row i-21k, names priced at the run
        date). Momentum is taken on closes carried 5 bars with MMOM_LOOK=252 / SKIP=21 unchanged. A name not
        scoreable at row i-21k contributes 0 for that k. mmom_ranks' 10-name minimum applies per k. Weight =
        max(cvg + strength · u / N, 0.25 · cvg). Strength (gate, scale) is exactly as shipped, and the scale's
        unit overlay stays the shipped single-formation overlay.
    JT3R: as JT3, then u = mmom_ranks(JT3 average, over names with at least one scored formation, names priced
        at the run date): the averaged score re-ranked to a centred [-1, 1] rank. Everything else identical.

bar
    The same ruling bar as MM-O1, on the shipped v12.1 code:
    - Beat SHIP on every-name net CAGR in each of E1 / E2 / E3 on both Nifty 50 (21.23 / 21.34 / 24.17) and
      Dow 30 (14.63 / 18.88 / 15.31).
    - Be >= SHIP on point-in-time Dow E3 in both the legacy run (11.28) and the members-only run (11.05).
    - ETF 27 reported only.

    Rows i-21k are rows of ctx.px / the close panel, not month starts, so the app and the research compute the
    same thing.

    Report turnover alongside: the mechanism claims it falls, and a pass with turnover not lower should be
    flagged.

    No parameter changes.

needs_regeneration  false

METHOD (how the pre-registration is implemented; nothing here changes a variant)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
Product path. Every book is mmom_ship.shipped(hist, price_history), i.e. nco.compute_nco_portfolio(hist, prices,
1e10, len(prices), method="MMOM", max_pos_pct=1.0, price_history=px.loc[:a]), over the same month loop as
mmom_ship.weights (hist = the 253 stale-repaired snapshots ending on the rebalance date; ETF minus
ss.ETF_YOUNG; the point-in-time Dow filtered to that day's members). The loop is mmom_ship.weights' own, written
out here so that all configurations are built from one hist per month (mmom_ship.weights also recomputes the
b_momentum identity reference every call, which costs most of its time and is not needed for a variant). For each
configuration nco.mmom_overlay is replaced IN MEMORY by `overlay(cfg)` below (restored afterwards; no repo file is
edited):
    SHIP      no patch at all: the shipped nco.mmom_overlay / nco.mmom_scale.
    ZERO      the patched overlay at its identity setting (cap 1, K 1, no re-rank). Must equal SHIP exactly.
    BSC_UP2   cap 2.0     strength = λ · gate · min(cap, median/now), median/now from the shipped mmom_scale
    BSC_UP15  cap 1.5     (its (now, median, months) are returned by nco.mmom_scale and re-used unchanged).
    JT3       K 3         u = mean_k mmom_ranks(mmom_momentum(carried[: i-21k+1], names), names), k = 0,1,2.
    JT3R      K 3 rerank  u = mmom_ranks(JT3 average over names scored in at least one formation, names).
The shipped nco.mmom_scale result is memoised per (last date, rows, columns) inside a month so the configurations
after ZERO do not recompute it; it is a pure function of the panel, so this changes no number. "Scored formation"
(JT3R) = the name had a finite 12-1 momentum at that k AND that k's ranking was live (>= MMOM_MIN_RANKED scored
names), so a median name with rank exactly 0 still counts as scored.
Point-in-time Dow: legacy = mmom_ship (d) (price_history = every column of the panel); members-only = the
price_history columns restricted to the day's member snapshot names, as scratch mmlogic2/pit_members.py does.
ss.run (top-N all + 10% cap, monthly, 10bp / 3bp per unit one-way turnover) holds the raw weights; the
point-in-time runs with ctx.priced restricted to members (mmom_ship.point_in_time). Cut books: the same raw
weights through ss.run(·, d, 30) for Nifty and ss.run(·, d, 20) for Dow, as scratch mmlogic2/topn.py did.
Paired t = mean / (sd / sqrt n) of the monthly net-return gap variant − SHIP within the era.
Shipped code pinned: the ruling is on v12.1 as committed. With AUDIT_MMOM_CODE=DIR the product modules (nco,
cvgrid, pragati, samanvaya, intraday) are imported from DIR (a `git show REF:<file>` copy) ahead of the repo, so
edits in the working tree do not enter; the run prints, per module, whether the file it used is the blob at
AUDIT_MMOM_REF (default HEAD). The v12.1 ruling: REF = c28f5a5.
  mkdir D; for f in nco cvgrid pragati samanvaya intraday; do git show c28f5a5:$f.py > D/$f.py; done

Run:  AUDIT_MMOM_CODE=D AUDIT_MMOM_REF=c28f5a5 python research/audit_mmom.py [--cache DIR] [panels]
      (~30 min, one process; weights cached per panel)

RESULT (2026-10-05) — MM-O1: BSC_UP2 AND BSC_UP15 BOTH PASS THE BAR, thinly; by the registered rule the pick is
BSC_UP15. MM-O2: JT3 AND JT3R BOTH FAIL (they lose Nifty E1 and Dow E1 / E2).
Six configurations were run (SHIP, ZERO, BSC_UP2, BSC_UP15, JT3, JT3R); four of them are variants under test.
The run used c28f5a5's product modules, then HEAD (AUDIT_MMOM_CODE = a `git show c28f5a5:` copy; every module
printed "= HEAD"), because nco.py and app.py were being edited in the working tree while it ran. HEAD has since
moved: 59d3755 / 2a862e8 commit the HRP and CVG audit fixes, among them the whole-share top-up for every style and
the CVG driver pool, and none of that is measured here.

Reproduction: ZERO (the patched overlay at cap 1, K 1) equals unpatched SHIP to max |Δw| 0.0 raw and after the
cut, and max |Δret| 0.0, in all 677 month-books (236 Nifty, 236 Dow, 19 ETF, 93 + 93 point-in-time Dow). SHIP
reproduces every published cell (Nifty 21.23 / 21.34 / 24.17, Dow 14.63 / 18.88 / 15.31, ETF 19.35, PIT legacy
11.28, PIT members 11.05). Both SHIP and ZERO also equal mmom_ship.weights' saved books (scratch mmlogic2
ship_*.pkl, pit_full.pkl, pit_members.pkl) to max |Δw| 0.0. An independent numpy rebuild of the JT3 / JT3R ranks
and the BSC multipliers matches the patched overlay with zero difference on 12 checked months.

Net CAGR % (Δ vs SHIP, paired t of the monthly gap), every name held, net of 10bp / 3bp:

                      Nifty 50                                    Dow 30
              E1             E2             E3             E1             E2             E3
  SHIP      21.230         21.338         24.167         14.633         18.884         15.314
  BSC_UP2   21.560 +.330   21.500 +.161   24.516 +.349   14.699 +.066   19.046 +.162   15.324 +.009
            (1.36)         (1.10)         (1.92)         (0.49)         (1.27)         (0.49)
  BSC_UP15  21.519 +.289   21.500 +.161   24.500 +.334   14.699 +.066   19.047 +.163   15.324 +.009
            (1.44)         (1.10)         (1.89)         (0.52)         (1.29)         (0.49)
  JT3       20.999 −.231   21.398 +.060   24.170 +.003   14.477 −.156   18.666 −.218   15.845 +.530
            (−0.26)        (0.07)         (0.06)         (−0.34)        (−0.63)        (1.19)
  JT3R      21.135 −.094   21.399 +.061   24.191 +.024   14.384 −.249   18.710 −.174   15.849 +.535
            (−0.06)        (0.06)         (0.11)         (−0.59)        (−0.48)        (1.14)

                 PIT Dow E3 legacy      PIT Dow E3 members     ETF 27 (not ruled)
  SHIP           11.281                 11.050                 19.354
  BSC_UP2        11.292 +.011 (0.71)    11.057 +.007 (0.32)    19.493 +.139 (1.20)
  BSC_UP15       11.292 +.011 (0.71)    11.057 +.007 (0.32)    19.493 +.139 (1.20)
  JT3            11.657 +.376 (1.19)    11.409 +.359 (1.09)    19.564 +.210 (0.35)
  JT3R           11.612 +.331 (1.01)    11.351 +.302 (0.88)    19.407 +.053 (0.09)

  Full span (Feb 2007 → Sep 2026), CAGR Δ vs SHIP (t) · vol · turnover/yr (SHIP: Nifty 22.26, vol 22.40,
  TO 1.89; Dow 16.15, vol 16.91, TO 1.83):
    Nifty  BSC_UP2 +.285 (2.48) 22.46 1.91 · BSC_UP15 +.265 (2.57) 22.45 1.91 · JT3 −.063 (−0.14) 22.47 1.61
           · JT3R −.007 (0.03) 22.45 1.63
    Dow    BSC_UP2 +.075 (1.15) 16.97 1.85 · BSC_UP15 +.075 (1.21) 16.96 1.85 · JT3 +.062 (0.35) 17.02 1.56
           · JT3R +.044 (0.27) 17.02 1.59
  Mean strength (SHIP → BSC_UP15): Nifty .88/.99/.91 → 1.07/1.08/1.04 by era, Dow .84/.95/.78 → .98/1.11/.78,
  PIT E3 .74 → .75 (legacy) and .77 → .78 (members), ETF .86 → .90. Mean names at the floor (SHIP → BSC_UP15):
  Nifty 4.4/6.0/5.8 → 6.3/7.2/7.4, Dow 3.0/3.9/2.2 → 4.0/5.0/2.2. JT3 / JT3R leave strength unchanged by
  construction.
  Cut books (same raw weights, not ruled), Δ vs SHIP by era. Nifty N=30 (SHIP 22.48 / 22.19 / 24.56): BSC_UP15
  −.115 / +.213 / +.323, BSC_UP2 −.128 / +.213 / +.336, JT3 −.599 / +.316 / +.162, JT3R −.649 / +.332 / +.313.
  Dow N=20 (SHIP 14.83 / 18.70 / 15.12): BSC_UP15 +.030 / +.344 / +.058, BSC_UP2 +.031 / +.340 / +.058, JT3
  −.749 / −.337 / +.349, JT3R −.660 / −.206 / +.240.

Reading it:
  · MM-O1 passes every ruled cell, but the cells that carry the bar's risk are close to identity. The two-sided
    scale changes the book only in months when the overlay's 126-day vol sits below its expanding median, where
    the shipped scale is 1 and the variant goes above it. That is 142 of 236 Nifty rebalances (E1 40/83, E2 58/72,
    E3 44/81), 99 of 236 Dow (E1 43, E2 48, E3 only 8/81), and 11 of 81 point-in-time Dow E3 months. So the Dow
    E3 margin (+.009, t 0.49) and the PIT margins (+.011, +.007) say "no harm where it barely acts", not "a gain
    there". The pass rests on Nifty (every era up, full span t 2.5) and Dow E1 / E2 (+.07 / +.16). It is the same
    finding as the earlier calm-months probe, now split by era (full-span Nifty +.285, t 2.48, the auditor's
    number exactly).
  · BSC_UP2 vs BSC_UP15: the 1.5 cap binds in 30 Nifty months and 10 Dow months, and never on the PIT or ETF
    windows. The two differ only in Nifty E1 (+.330 vs +.289), Nifty E3 (+.349 vs +.334) and the third decimal
    elsewhere. Per the registration the coordinator takes the gentler BSC_UP15.
  · Costs of MM-O1, reported, not ruled: turnover +0.02/yr (Nifty 1.91 vs 1.89, Dow 1.85 vs 1.83). Vol +0.05.
    Up to two more names at the floor on average. In the cut Nifty N=30 book E1 turns negative (−.115, t −0.42);
    top-N books were not ruled on.
  · MM-O2 does what the mechanism says to turnover: −15% (Nifty 1.61 / 1.63 vs 1.89, Dow 1.56 / 1.59 vs 1.83).
    Return falls in the early eras: Nifty E1 and Dow E1 / E2 lose. Its gains sit in Dow E3 and the point-in-time
    Dow (+.30 to +.54, t ≤ 1.2), the cells where the shipped single-formation rank has been weakest. Neither form
    is close to the bar; re-ranking (JT3R) restores dispersion (mean |u| ~.51 vs ~.47 on the checked months) and helps Nifty E1 a little
    but loses more Dow E1.
  · Caveats for the pick: none of the margins is significant per cell (largest per-era t 1.92, Nifty E3, UP2).
    Everything is on today's constituents except the point-in-time Dow, where MM-O1 barely acts. The app's scale
    also reads the month to date between rebalances (mmom_ship WHY THIS FILE, MTD), so mid-month app books can
    up-scale on days this month-start backtest cannot see. The ruling is on c28f5a5, the shipped v12.1 as the bar
    asks. Re-check the pick on the code after 59d3755 before it ships (SHIP itself moves there).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
# AUDIT_MMOM_CODE: a directory holding the product modules to measure (e.g. `git show c28f5a5:nco.py` etc.), put
# ahead of the repo so the ruling is on the shipped code even while the working tree is being edited. The
# product modules are imported from it before any research module, so every later `import nco` gets that copy.
CODE = os.environ.get("AUDIT_MMOM_CODE")
REF = os.environ.get("AUDIT_MMOM_REF", "HEAD")    # the commit the provenance check compares each module with
if CODE:
    sys.path.insert(0, os.path.abspath(CODE))
PRODUCT = ("nco", "cvgrid", "pragati", "samanvaya", "intraday")
for _mod in PRODUCT:
    __import__(_mod)

import nco                                        # noqa: E402
import style_blends as sb                         # noqa: E402
import style_search as ss                         # noqa: E402
import style_search_pit as P                      # noqa: E402
import mmom_ship as M                             # noqa: E402

PANELS = ("nifty_50", "dow_30", "etf_27", "pit_legacy", "pit_members")
CONFIGS = {                                       # name -> (scale cap, formations K, re-rank)
    "SHIP": None,
    "ZERO": (1.0, 1, False),
    "BSC_UP2": (2.0, 1, False),
    "BSC_UP15": (1.5, 1, False),
    "JT3": (1.0, 3, False),
    "JT3R": (1.0, 3, True),
}
VARIANTS = ("BSC_UP2", "BSC_UP15", "JT3", "JT3R")
SHIP_PUBLISHED = {"nifty_50": (21.23, 21.34, 24.17), "dow_30": (14.63, 18.88, 15.31), "etf_27": (19.35,),
                  "pit_legacy": (11.28,), "pit_members": (11.05,)}
CUT_N = {"nifty_50": 30, "dow_30": 20}
STEP = 21                                         # rows between overlapping formations (MM-O2)
ORIG_OVERLAY, ORIG_SCALE = nco.mmom_overlay, nco.mmom_scale
_MEMO: dict = {}


# ── the patched overlay ──────────────────────────────────────────────────────────────────────────────────────
def _scale_memo(prices: pd.DataFrame):
    """The shipped nco.mmom_scale, memoised within a month (a pure function of the panel)."""
    key = (prices.index[-1], len(prices), tuple(prices.columns)) if len(prices) else None
    if key is None:
        return ORIG_SCALE(prices)
    if key not in _MEMO:
        _MEMO[key] = ORIG_SCALE(prices)
    return _MEMO[key]


def _formation_ranks(carried: pd.DataFrame, names: list, K: int):
    """[(rank_k, scored_k)] for k = 0..K-1: mmom_ranks of the 12-1 momentum as of row i - 21k."""
    i = len(carried) - 1
    out = []
    for k in range(K):
        j = i - STEP * k
        mom = nco.mmom_momentum(carried.iloc[: j + 1] if j >= 0 else carried.iloc[:0], names)
        rank = nco.mmom_ranks(mom, names)
        live = int(mom.notna().sum()) >= nco.MMOM_MIN_RANKED
        out.append((rank, (mom.notna() & live).reindex(names).fillna(False)))
    return out


def overlay(cap: float, K: int, rerank: bool):
    """A drop-in for nco.mmom_overlay. At (1, 1, False) it is the shipped overlay line for line."""
    def mmom_overlay(prices: pd.DataFrame, names):
        names = list(names)
        if prices is None:
            prices = pd.DataFrame()
        if len(prices) and not isinstance(prices.index, pd.DatetimeIndex):
            try:
                prices = prices.set_axis(pd.DatetimeIndex(pd.to_datetime(prices.index)), axis=0)
            except (TypeError, ValueError):
                prices = pd.DataFrame()
        carried = prices.ffill(limit=5)
        mom = nco.mmom_momentum(carried, names)                    # k = 0: the shipped 12-1
        if K == 1:
            rank = nco.mmom_ranks(mom, names)
        else:
            parts = _formation_ranks(carried, names, K)
            rank = sum(r for r, _ in parts) / float(K)            # unscored at k -> 0 for that k
            if rerank:
                scored = np.logical_or.reduce([s.to_numpy(dtype=bool) for _, s in parts])
                rank = nco.mmom_ranks(rank.where(pd.Series(scored, index=names)), names)
        gate, mkt = nco.mmom_gate(prices)
        if gate > 0:
            scale, vol, vol_med, vol_months = _scale_memo(prices)
            if vol_months >= nco.MMOM_MIN_VOL_MONTHS and np.isfinite(vol) and vol > 0:
                scale = float(min(cap, vol_med / vol))
        else:
            scale, vol, vol_med, vol_months = 1.0, float("nan"), float("nan"), 0
        return rank, mom, {
            "gate": gate, "market_24m": mkt, "scale": scale, "strength": nco.MMOM_LAMBDA * gate * scale,
            "overlay_vol": vol, "overlay_vol_median": vol_med, "vol_months": vol_months,
            "ranked": int(mom.notna().sum()), "history_days": int(len(prices)),
            "history_start": prices.index[0] if len(prices) else None,
        }
    return mmom_overlay


@contextmanager
def patched(cfg):
    if cfg is None:
        yield
        return
    nco.mmom_overlay = overlay(*cfg)
    try:
        yield
    finally:
        nco.mmom_overlay = ORIG_OVERLAY


# ── the month loop (mmom_ship.weights, every configuration per month) ────────────────────────────────────────
def _data(u: str):
    if u.startswith("pit"):
        d = pickle.load(open(P.PKL, "rb"))
        snaps = sb.unstale(pickle.load(open(os.path.join(HERE, "cvg_reweight_dow_pit.pkl"), "rb")))
        return d, snaps
    d = ss.load(u, holdout=True)
    snaps = sb.snapshots("etf_book" if u == "etf_27" else u)
    if u == "etf_27":
        snaps = [(t, s[~s["symbol"].isin(ss.ETF_YOUNG)].reset_index(drop=True)) for t, s in snaps]
    return d, snaps


def build(u: str) -> dict:
    d, snaps = _data(u)
    cal = list(d["px"].index)
    assert [pd.Timestamp(t) for t, _ in snaps] == cal, "snapshots and panel calendars differ"
    pos = {t: i for i, t in enumerate(cal)}
    members = P.member if u.startswith("pit") else None
    W = {k: {} for k in CONFIGS}
    info = {k: {} for k in CONFIGS}
    gap, rawgap = {}, {}
    t0 = time.time()
    for n, a in enumerate(d["months"][:-1]):
        hist = snaps[max(0, pos[a] - 252): pos[a] + 1]
        if members is not None:
            hist = [(t, s[s["symbol"].map(lambda x: members(x, a))].reset_index(drop=True)) for t, s in hist]
        px = d["px"].loc[:a]
        if u == "pit_members":                      # what a user on that day fetches: that day's universe
            keep = set(hist[-1][1]["symbol"].astype(str))
            px = px.reindex(columns=[c for c in px.columns if c in keep])
        _MEMO.clear()
        for k, cfg in CONFIGS.items():
            with patched(cfg):
                w, inf = M.shipped(hist, px)
            W[k][a], info[k][a] = w, inf
        priced = pd.Index(sorted(W["SHIP"][a].index))
        gap[a] = M._gap(M._cut(W["ZERO"][a], priced), M._cut(W["SHIP"][a], priced))
        same = list(W["ZERO"][a].index) == list(W["SHIP"][a].index)
        rawgap[a] = (float((W["ZERO"][a] - W["SHIP"][a]).abs().max()) if same else float("inf"))
        if n % 40 == 0:
            print(f"   {u} {a:%Y-%m} ({n + 1}/{len(d['months']) - 1}) {time.time() - t0:.0f}s", flush=True)
    return dict(W=W, info={k: pd.DataFrame(v).T for k, v in info.items()}, gap=pd.Series(gap),
                rawgap=pd.Series(rawgap), secs=time.time() - t0)


def runs(u: str, B: dict) -> dict:
    d = pickle.load(open(P.PKL, "rb")) if u.startswith("pit") else ss.load(u, holdout=True)
    cm = M.point_in_time() if u.startswith("pit") else _null()
    with cm:
        R = {k: ss.run(lambda c, k=k: B["W"][k][c.date], d) for k in CONFIGS}
        C = ({k: ss.run(lambda c, k=k: B["W"][k][c.date], d, CUT_N[u]) for k in ("SHIP",) + VARIANTS}
             if u in CUT_N else {})
        base = {k: v for k, v in ss.baselines(d).items() if k in ("EW", "CVG", "HRP")}
    return dict(R=R, C=C, base=base, d=d)


@contextmanager
def _null():
    yield


def provenance() -> list:
    """(module, file, same blob as at REF) for every module the books are built from."""
    import subprocess
    mods = [sys.modules[m] for m in PRODUCT] + [sb, ss, P, M, M.B, sys.modules.get("style_search_holdout")]
    out = []
    for m in mods:
        if m is None:
            continue
        f = os.path.abspath(m.__file__)
        base = os.path.abspath(CODE) if CODE and f.startswith(os.path.abspath(CODE) + os.sep) else ROOT
        rel = os.path.relpath(f, base)
        try:
            git = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True,  # noqa: E731
                                            check=True).stdout.strip()
            same = git("hash-object", f"--path={rel}", f) == git("rev-parse", f"{REF}:{rel}")
        except Exception:                                                  # noqa: BLE001
            same = None
        out.append((m.__name__, f, same))
    return out


# ── reporting ────────────────────────────────────────────────────────────────────────────────────────────────
def _t(x: pd.Series) -> float:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else np.nan


def eras(u: str):
    if u.startswith("pit"):
        return [("E3", "2020-01-01", None)]
    if u == "etf_27":
        return [("window", None, None)]
    return list(ss.ERAS)


def cells(u: str, R: dict, B: dict) -> pd.DataFrame:
    rows = []
    for k, r in R.items():
        for e, a, b in eras(u):
            x, y = ss._era(r, a, b), ss._era(R["SHIP"], a, b)
            m, my = ss.metrics(x), ss.metrics(y)
            inf = B["info"].get(k) if B is not None else None
            if inf is not None:
                ix = inf.index[(inf.index >= pd.Timestamp(a)) if a else np.ones(len(inf), bool)]
                ix = ix[(ix < pd.Timestamp(b)) if b else np.ones(len(ix), bool)]
                st = inf.loc[ix, "strength"].astype(float).mean()
                fl = inf.loc[ix, "floored"].astype(float).mean()
            else:
                st = fl = np.nan
            rows.append(dict(cfg=k, era=e, cagr=m["cagr"], d_ship=m["cagr"] - my["cagr"],
                             t=_t(x["ret"] - y["ret"]) if k != "SHIP" else np.nan, vol=m["vol"],
                             ret_vol=m["ret_vol"], to=m["turnover"], strength=st, floored=fl,
                             names=f"{int(x['names'].min())}-{int(x['names'].max())}"))
    return pd.DataFrame(rows)


def full_span(R: dict) -> pd.DataFrame:
    rows = {}
    for k, r in R.items():
        m = ss.metrics(r)
        rows[k] = dict(cagr=m["cagr"], d_ship=m["cagr"] - ss.metrics(R["SHIP"])["cagr"],
                       t=_t(r["ret"] - R["SHIP"]["ret"]) if k != "SHIP" else np.nan, vol=m["vol"],
                       ret_vol=m["ret_vol"], maxdd=m["maxdd"], to=m["turnover"])
    return pd.DataFrame(rows).T


def report(res: dict) -> None:
    pd.set_option("display.width", 250)
    print("\n══ REPRODUCTION: ZERO (patched, identity setting) vs SHIP (unpatched product code) ═════════════════")
    ok_all = True
    for u, x in res.items():
        B, c = x["B"], x["cells"]
        ship = c[c.cfg == "SHIP"].set_index("era")["cagr"]
        zero = c[c.cfg == "ZERO"].set_index("era")["cagr"]
        rz = x["R"]["ZERO"]["ret"] - x["R"]["SHIP"]["ret"]
        pub = SHIP_PUBLISHED[u]
        okp = all(abs(round(v, 2) - p) < 1e-9 for v, p in zip(ship.to_numpy(), pub))
        ok = B["rawgap"].max() == 0.0 and B["gap"].max() == 0.0 and float(rz.abs().max()) == 0.0
        ok_all &= ok and okp
        print(f"  {u:12s} months {len(B['gap'])} · max |Δw| raw {B['rawgap'].max():.1e} · after cut {B['gap'].max():.1e}"
              f" · max |Δret| {float(rz.abs().max()):.1e} · SHIP {' / '.join(f'{v:.2f}' for v in ship)}"
              f" (published {' / '.join(f'{p:.2f}' for p in pub)}: {'match' if okp else 'MISMATCH'})"
              f" · ZERO {' / '.join(f'{v:.2f}' for v in zero)} · {B['secs']:.0f}s")
    print(f"  reproduces_current: {ok_all}")
    for u, x in res.items():
        print(f"\n══ {u} · net CAGR %, Δ vs SHIP, paired t of the monthly gap, vol, ret/vol, turnover/yr, "
              f"mean strength, mean floored ═══")
        c = x["cells"].copy()
        print("   " + c.round(3).to_string(index=False).replace("\n", "\n   "))
        if not u.startswith("pit"):
            print("   full span")
            print("   " + full_span(x["R"]).round(3).to_string().replace("\n", "\n   "))
        if x.get("C"):
            print(f"   cut book N={CUT_N[u]} (same raw weights, ss.run(·, d, {CUT_N[u]}))")
            cc = cells(u, x["C"], None)
            print("   " + cc[["cfg", "era", "cagr", "d_ship", "t", "vol", "to", "names"]].round(3)
                  .to_string(index=False).replace("\n", "\n   "))
            print("   " + full_span(x["C"])[["cagr", "d_ship", "t", "to"]].round(3).to_string()
                  .replace("\n", "\n   "))
        b = x["base"]
        print("   baselines: " + " · ".join(
            f"{k} " + "/".join(f"{ss.metrics(ss._era(v, a, bb))['cagr']:.2f}" for e, a, bb in eras(u))
            for k, v in b.items()))
    if not all(u in res for u in PANELS):
        print(f"\n  the bar needs every panel ({', '.join(PANELS)}); ran {', '.join(res)}: no ruling")
        return
    print("\n══ THE BAR (unrounded): beat SHIP in Nifty E1/E2/E3 and Dow E1/E2/E3; >= SHIP on PIT Dow E3 legacy and "
          "members ═══")
    for k in VARIANTS:
        verdict, parts = True, []
        for u in ("nifty_50", "dow_30"):
            c = res[u]["cells"]
            for e in ("E1", "E2", "E3"):
                dv = float(c[(c.cfg == k) & (c.era == e)]["d_ship"].iloc[0])
                verdict &= dv > 0
                parts.append(f"{u.split('_')[0]} {e} {dv:+.3f}")
        for u in ("pit_legacy", "pit_members"):
            c = res[u]["cells"]
            dv = float(c[(c.cfg == k) & (c.era == "E3")]["d_ship"].iloc[0])
            verdict &= dv >= 0
            parts.append(f"{u} {dv:+.3f}")
        ce = res["etf_27"]["cells"]
        etf = float(ce[ce.cfg == k]["d_ship"].iloc[0])
        to = {u: (ss.metrics(res[u]["R"][k])["turnover"], ss.metrics(res[u]["R"]["SHIP"])["turnover"])
              for u in ("nifty_50", "dow_30")}
        flag = (" · TURNOVER NOT LOWER" if k.startswith("JT") and verdict and any(a >= b for a, b in to.values())
                else "")
        print(f"  {k:9s} {'PASS' if verdict else 'FAIL'} · " + " · ".join(parts) + f" · (ETF {etf:+.3f}, not ruled)"
              + " · turnover/yr " + " ".join(f"{u.split('_')[0]} {a:.2f} vs {b:.2f}" for u, (a, b) in to.items())
              + flag)
    print(f"\n  configurations run: {len(CONFIGS)} ({', '.join(CONFIGS)}); variants under test: {len(VARIANTS)}; "
          f"the cut books re-hold the same raw weights")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache", default=os.environ.get("AUDIT_MMOM_CACHE",
                                                      os.path.join(tempfile.gettempdir(), "audit_mmom")),
                    help="directory for the per-panel weight caches (outside the repo)")
    ap.add_argument("panels", nargs="*", default=list(PANELS))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    print(f"code: {os.path.abspath(CODE) if CODE else ROOT} (AUDIT_MMOM_CODE {'set' if CODE else 'not set'})")
    for name, f, same in provenance():
        print(f"   {name:22s} {f}  {f'= {REF}' if same else f'DIFFERS FROM {REF}' if same is False else '(no git)'}")
    res = {}
    for u in args.panels:
        f = os.path.join(args.cache, f"audit_mmom_{u}.pkl")
        if os.path.exists(f):
            B = pickle.load(open(f, "rb"))
            print(f"   {u}: weights from cache {f}", flush=True)
        else:
            print(f"\n== building {u}", flush=True)
            B = build(u)
            pickle.dump(B, open(f, "wb"))
        X = runs(u, B)
        res[u] = dict(B=B, R=X["R"], C=X["C"], base=X["base"], cells=cells(u, X["R"], B))
    report(res)


if __name__ == "__main__":
    main()
