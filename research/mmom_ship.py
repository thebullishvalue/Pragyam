"""
research/mmom_ship.py — Managed Momentum (nco "MMOM") re-measured AS SHIPPED, through the product code.

WHY THIS FILE
─────────────
The style search (research/style_search.py protocol; research/candidates/b_momentum.py; one holdout
run in research/style_search_holdout.py; research/style_search_pit.py) found one configuration,
managed_mom λ=1, that beat the best of the eight earlier styles in all six stock cells. The shipped
style is that candidate after three decisions, all made after the holdout was seen:

    λ = 1   over λ = 2, which discovery ranked first: chosen on the point-in-time Dow (λ=2 trailed
            CVG by 0.63 %/yr there) and the position count (λ=2 zeroed up to 6 Nifty names).
    FLOOR   no name below MMOM_FLOOR = ¼ of its CVG weight. The tested form clipped at 0, and so could
            zero names; the app promises the position count it was asked for. The floor keeps every
            weight positive; it does not keep a name in a book smaller than the universe, where the
            floored names are the first cut.
    MTD     the overlay's volatility includes the month to date. On a rebalance date (the first
            trading day of a month) that piece is empty, so this backtest cannot see it. In the app,
            between month starts, it can.

The holdout's own verdict recommended a point-in-time Nifty test before any product decision; it
was not run (no NSE constituent history here).

So on this calendar the shipped weights should be exactly max(managed_mom(λ=1), ¼ · CVG). This
harness checks that, then re-measures the shipped CODE: nco.compute_nco_portfolio(method="MMOM"),
not a re-implementation, on the same panels, book, costs and eras as every other style.

METHOD
──────
(a) For every rebalance month of ss.load(u, holdout=True), u in nifty_50, dow_30 and etf_27:
        hist   = the sb.PANEL (400; 253 before v12.2) snapshots ending on the rebalance date, from
                 style_blends.snapshots(key) (stale-close and corporate-action repairs applied; key
                 "etf_book" for etf_27, minus ss.ETF_YOUNG)
        prices = the last snapshot's priced names (every name is held: num_positions = len(prices))
        book   = nco.compute_nco_portfolio(hist, prices, 1e10, len(prices), method="MMOM",
                                           max_pos_pct=1.0, price_history=d["px"].loc[:a])
    The raw weights are weightage_pct, normalised. They go through ss.run unchanged: the 10% cap,
    monthly hold, 10bp / 3bp per unit of one-way turnover. Costs, turnover and eras therefore match
    every other style exactly. price_history is the panel's own close history (from Oct 2006),
    which plays the part of backdata.fetch_close_history in the app.
    EVERY-NAME BOOKS ONLY: num_positions = the universe, so every figure below holds every priced
    name. Top-N books — the app's default whenever the universe exceeds the position count, e.g.
    Nifty 50 at 30 — were never measured for this style; there momentum also decides which names
    are held, not only how much.
(b) Exactness. For each month, the shipped weights are compared after sb.cut(·, None) with
        IDENTITY   max(b_momentum.managed_mom(ctx, λ=1), MMOM_FLOOR · CVG), in-repo;
        REFERENCE  mmom_s(ctx), the scratch reference of the shipped form (--ref PATH; skipped
                   if not given).
(c) Per era (E1 < 2014, E2 2014-19, E3 ≥ 2020; the ETF book over its whole window), the shipped
    style is set against the eight baselines (ss.baselines) and the tested managed_mom(λ=1). Reported:
    net CAGR, margin over the best baseline in each cell, the paired t of that margin, cells won of
    six; and full-span CAGR / vol / ret-vol / maxDD / turnover against CVG and EW.
(d) Point-in-time Dow, E3: style_search_pit.PKL with ctx.priced restricted to that day's members,
    as style_search_pit.main does. The shipped weights are built from snapshots filtered to those
    members, with prices restricted to them, as style_search_pit.build does for the baselines.
    price_history is cut to that day's members (v12.2, MM-B8), as a live user's fetch would be: the
    ranks, the bear gate's market and the volatility scale all read members only. (Until v12.1 the
    gate and scale read the full panel, non-members included, as the research reference did.)
    --app-history replaces d["px"] with backdata.fetch_close_history from MMOM_HISTORY_START,
    cached once to research/mmom_close_{u}.pkl and sliced and dead-quote masked per date (MM-B9).

Run:  python research/mmom_ship.py [--ref /path/to/mmom_ref_keep.py]     (~8 min, one process)

RESULT (2026-10-03) — the shipped code is the style that was measured, plus the floor, and it keeps
the tested style's record less the floor's cost. It is not significant and it does not survive a
point-in-time Dow. Net CAGR %, margin over the best of the eight in each cell (paired t):

                                  Nifty 50                       Dow 30                     ETF (27)
                          E1      E2      E3            E1      E2      E3                 Mar 25 →
  best of the eight    20.21H  19.68C  22.59C        14.39EW 18.50C  15.06C                17.57EW
  CVG                  19.16   19.68   22.59         14.17   18.50   15.06                 17.09
  EW                   17.93   18.83   22.31         14.39   18.32   14.23                 17.57
  MMOM shipped         21.23   21.34   24.17         14.63   18.88   15.31                 19.35
    margin             +1.02   +1.66   +1.58         +0.25   +0.39   +0.25   6/6           +1.79
    (t)                (0.75)  (0.84)  (1.13)        (0.28)  (0.41)  (0.15)                (0.53)
  managed_mom λ=1      21.38   21.44   24.28         14.61   18.90   15.45                 19.53
    margin (tested)    +1.17   +1.77   +1.70         +0.22   +0.41   +0.39   6/6           +1.96

  Full span (Feb 2007 → Sep 2026; ETF Mar 2025 →), net:
                   CAGR    vol  ret/vol  maxDD  turnover/yr
    Nifty  MMOM   22.26  22.40   1.02   −57.2     1.89
           CVG    20.48  22.65   0.94   −58.5     1.48
           EW     19.69  22.25   0.93   −58.4     0.37
    Dow    MMOM   16.15  16.91   0.98   −39.1     1.83
           CVG    15.78  16.73   0.97   −39.3     1.41
           EW     15.52  16.48   0.96   −39.9     0.27
    ETF    MMOM   19.35  12.68   1.47    −7.5     1.71
           CVG    17.09  13.05   1.28    −7.2     1.39
           EW     17.57  13.70   1.25    −8.0     0.20

  Point-in-time Dow, E3 (members of the day, 29-30 names): MMOM 11.28 · CVG 11.47 (best of the
  eight) · EW 11.00 · managed_mom λ=1 11.41 → MMOM −0.19 vs CVG (t −0.28), +0.28 vs EW.

  · Exactness: over 584 month-books (236 Nifty, 236 Dow, 19 ETF, 93 point-in-time Dow) the shipped
    weights match max(managed_mom λ=1, ¼·CVG) and the scratch reference mmom_s to max |Δw| 4.2e-17.
    The MTD change cannot show on this calendar (see WHY THIS FILE); the app's mid-month books can
    differ from these.
  · The floor's price, shipped − tested: Nifty −0.15 / −0.10 / −0.12, Dow +0.02 / −0.02 / −0.14,
    ETF −0.18, point-in-time Dow E3 −0.13 %/yr, and ~0.1 less turnover. In return, in these
    every-name books, every priced name is held (39-50 Nifty, 28-30 Dow, 27 ETF); the tested form
    held as few as 37, 24 and 25. The floor binds on 0-10 Nifty names a month (mean 5.4), 0-7 Dow
    (3.0), 0-4 ETF (1.9), counted over the universe — in a top-N book those are the first cut.
  · The overlay: the bear gate shut 10 Nifty months (2008-11 → 2009-05, 2020-04 → 06) and 13 Dow
    months (2008-11 → 2009-11), never on the ETF window. The volatility scale averaged 0.96 Nifty /
    0.90 Dow / 0.86 ETF (min 0.61 / 0.39 / 0.57), 0.74 on the point-in-time Dow in E3. The panels'
    closes start in Oct 2006, so until late 2008 (22 Nifty, 21 Dow rebalances) the gate read less
    than its 24 months, as the reference did; the app fetches closes from nco.MMOM_HISTORY_START.
  · Not significant: the largest per-era paired t over the best of the eight is 1.13 (Nifty E3).
    Over the full span, from the monthly net returns this file builds (panel(u, None)["runs"][SHIP]
    against ["base"]; not printed): Nifty vs CVG +1.78 %/yr t 1.75, vs EW +2.57 t 2.64 (E1 vs EW
    t 2.12); Dow vs CVG +0.37 t 0.56, vs EW +0.63 t 0.97 — nominal. The shipped form is a
    post-holdout variant of one of 43 configurations, so none of it survives a family-wise
    correction (any family-wise p is 1.0), nor the survivorship caveat: the E3 margins sit in a few
    names, led by late index entrants (BSE, TRENT, BEL, ADANIENT; NVDA, AMZN, CRM — python
    research/style_search_holdout.py --attribution), and on the Dow's members of the day MMOM
    trails CVG.
  · Verdict: the product code is the tested style plus the floor, and the floor costs about 0.1 %/yr.
    The honest description is a momentum tilt on the grid that led all eight styles in every cell
    on today's constituents, by margins indistinguishable from noise, and tied CVG on the Dow's
    members of the day. It is not evidence that the style reliably beats CVG or Equal Weight.

RE-MEASURED (2026-10-05) on the v12.2 code and data — the audit's fixes: the overlay stands down
below 505 rows whatever its history (MM-B5), carried gate and scale returns (MM-B1), closes <= 0
unpriced (MM-B7), calendar windows (MM-B2; no change on these panels), the HRP / ERC input fixes,
and the research panels repaired for yfinance's unadjusted corporate actions (style_blends.repair;
BAJAJFINSV 2008 x2, ADANIENT 2015, TMPV 2025, TRENT 2026). Pickles rebuilt.

  --app-history (the overlay reads backdata.fetch_close_history from 2006, as the app does):
                                  Nifty 50                       Dow 30                     ETF (27)
                          E1      E2      E3            E1      E2      E3                 Mar 25 →
  best of the eight    20.08H+C 19.89C  22.97C        14.39EW 18.50C  15.06C                17.57EW
  MMOM shipped         20.51   21.46   24.33         15.11   18.88   15.31                 19.37
    margin             +0.43   +1.57   +1.37         +0.72   +0.39   +0.24   6/6           +1.80
    (t)                (0.57)  (0.80)  (0.98)        (0.70)  (0.41)  (0.14)                (0.54)
  Full span: Nifty 22.10 vs CVG 20.95 (+1.15, t 1.03) and EW 20.10 (+2.01, t 1.98); Dow 16.32 vs
  15.78 (+0.53, t 0.81) and 15.52 (+0.80, t 1.23). Point-in-time Dow E3, the overlay reading that
  day's members only (MM-B8): MMOM 11.06 · CVG 11.47 · EW 11.00 → −0.42 vs CVG (t −0.49).
  Gate shut 9 Nifty months (2008-11 → 2009-04, 2020-04 → 06), 13 Dow; overlay off before 2008-01
  (history under 505 rows). Scale mean 0.96 / 0.90 / 0.86 (min 0.58 / 0.39 / 0.54).

  default (the research panel's own closes from Oct 2006): Nifty 20.74 / 21.45 / 24.33, Dow
  14.58 / 18.88 / 15.31, ETF 19.35, point-in-time 11.05; margins +0.66 / +1.56 / +1.36, +0.19 /
  +0.39 / +0.25 — 6/6.

  · The Nifty E1 cell is won because the HRP fixes lowered HRP's E1 (20.21 → 19.99 on the repaired
    panel; best is now HRP+CVG 20.08). Against v12.1's HRP on the repaired panel (20.81) the cell is
    lost by 0.07 (research path) to 0.30 (app path) — the audit's skeptic, scratch skep/combo.py.
  · Exactness no longer holds by construction: the shipped form stands down before 505 rows and,
    with --app-history, reads a different history from managed_mom (max |Δw| 3.4e-02).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

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

PANELS = ("nifty_50", "dow_30", "etf_27")
SHIP, TESTED = "MMOM (shipped)", "managed_mom λ=1 (tested)"
ABBR = {"EW": "EW", "ERC": "ERC", "HRP": "H", "CVG": "C", "HRP+CVG": "H+C", "HRP+EW": "H+EW",
        "CVG+EW": "C+EW", "HRP+CVG+EW": "H+C+EW"}
ATTRS = ("gate", "market_24m", "scale", "strength", "floored", "history_days", "vol_months", "ranked",
         "source", "history_short")


def _module(path: str):
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _module(os.path.join(HERE, "candidates", "b_momentum.py"))


# ── (a) the shipped book, through the product code ────────────────────────────────────────────────
def shipped(hist: list, px_hist: pd.DataFrame) -> tuple:
    snap = hist[-1][1]
    prices = {s: float(p) for s, p in zip(snap["symbol"], pd.to_numeric(snap["price"], errors="coerce"))
              if np.isfinite(p) and p > 0}
    book = nco.compute_nco_portfolio(hist, prices, sb.CAPITAL, len(prices), method="MMOM",
                                     max_pos_pct=1.0, price_history=px_hist)
    if book.empty:
        raise ValueError(f"{hist[-1][0]}: the shipped MMOM book is empty")
    w = pd.to_numeric(book["weightage_pct"], errors="coerce")
    raw = pd.Series((w / w.sum()).to_numpy(), index=book["symbol"].astype(str).to_numpy())
    info = {k: book.attrs.get(f"nco_mmom_{k}") for k in ATTRS}
    info.update(priced=len(prices), held=len(book), names=frozenset(prices))
    return raw, info


def _cut(w: pd.Series, priced: pd.Index) -> pd.Series:
    """What ss.run does to a style's raw weights before it holds them."""
    w = pd.Series(w, dtype=float)
    w = w[w.index.isin(priced)].clip(lower=0.0).fillna(0.0)
    return sb.cut(w.sort_values(ascending=False, kind="stable"), None)


def _gap(a: pd.Series, b: pd.Series) -> float:
    idx = a.index.union(b.index)
    return float((a.reindex(idx, fill_value=0.0) - b.reindex(idx, fill_value=0.0)).abs().max())


CLOSE_PKL = os.path.join(HERE, "mmom_close_{}.pkl")


def app_history(u: str, syms: list):
    """price_history the app's way (MM-B9): backdata.fetch_close_history from MMOM_HISTORY_START,
    unmasked and cached once, then sliced to each run date and dead-quote masked after the slice."""
    import backdata
    path = CLOSE_PKL.format(u)
    if os.path.exists(path):
        close = pickle.load(open(path, "rb"))
    else:
        close = backdata.fetch_close_history(syms, datetime.fromisoformat(nco.MMOM_HISTORY_START), sb.END,
                                             mask_dead_quotes=False)
        want = {str(s).replace(".NS", "") for s in syms}
        if close is None or close.empty or len(want - set(close.columns)) > 0.05 * len(want):
            raise RuntimeError(f"{u}: the close-history fetch came back empty or short "
                               f"({0 if close is None else close.shape[1]} of {len(want)}); not cached")
        pickle.dump(close, open(path, "wb"))
    cache = {}
    return lambda a: cache.setdefault(a, backdata.mask_dead_quotes(close.loc[:a])[0])


def weights(d: dict, snaps: list, ref=None, members=None, history=None) -> dict:
    """Shipped and tested raw weights for every rebalance month of `d`, and the exactness gaps.
    `history(a)` is the overlay's price_history on run date a (default: the panel's own closes)."""
    cal = list(d["px"].index)
    assert [pd.Timestamp(t) for t, _ in snaps] == cal, "snapshots and panel calendars differ"
    pos = {t: i for i, t in enumerate(cal)}
    ship, tested, info, g_id, g_ref = {}, {}, {}, {}, {}
    for a in d["months"][:-1]:
        hist = snaps[max(0, pos[a] - (sb.PANEL - 1)): pos[a] + 1]
        if members is not None:
            hist = [(t, s[s["symbol"].map(lambda x: members(x, a))].reset_index(drop=True)) for t, s in hist]
        ph = history(a) if history is not None else d["px"].loc[:a]
        if members is not None:            # a live user fetches only that day's universe (MM-B8)
            ph = ph[[c for c in ph.columns if members(c, a)]]
        w, inf = shipped(hist, ph)
        ctx = ss.Ctx(d, a)
        assert inf["names"] == frozenset(ctx.priced), f"{a:%Y-%m-%d}: priced names differ"
        t = B.managed_mom(ctx, lam=1.0)
        c = ctx.raw["CVG"].reindex(ctx.priced).fillna(0.0).clip(lower=0.0)
        c = c / c.sum()
        ident = np.maximum(t, nco.MMOM_FLOOR * c)
        ws = _cut(w, ctx.priced)
        g_id[a] = _gap(ws, _cut(ident, ctx.priced))
        if ref is not None:
            g_ref[a] = _gap(ws, _cut(ref.mmom_s(ctx), ctx.priced))
        ship[a], tested[a], info[a] = w, t, inf
    return dict(ship=ship, tested=tested, info=pd.DataFrame(info).T, g_id=pd.Series(g_id),
                g_ref=pd.Series(g_ref, dtype=float))


# ── (c) per era, against the eight ────────────────────────────────────────────────────────────────
def _t(x: pd.Series) -> float:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else np.nan


def cell(r: pd.DataFrame, base: dict, a, b) -> dict:
    x = ss._era(r, a, b)
    best = max(ss.BASE, key=lambda k: ss.metrics(ss._era(base[k], a, b))["cagr"])
    bx = ss._era(base[best], a, b)
    m, bm = ss.metrics(x), ss.metrics(bx)
    ew, cvg = ss._era(base["EW"], a, b), ss._era(base["CVG"], a, b)
    return dict(cagr=m["cagr"], best=best, best_cagr=bm["cagr"], margin=m["cagr"] - bm["cagr"],
                t=_t(x["ret"] - bx["ret"]), vs_ew=m["cagr"] - ss.metrics(ew)["cagr"],
                t_ew=_t(x["ret"] - ew["ret"]), vs_cvg=m["cagr"] - ss.metrics(cvg)["cagr"],
                months=m["months"])


def _ranges(months: list) -> str:
    """Consecutive rebalance months as 'YYYY-MM → YYYY-MM' runs."""
    if not months:
        return "never"
    out, start, prev = [], months[0], months[0]
    for m in months[1:]:
        if (m.year - prev.year) * 12 + m.month - prev.month != 1:
            out.append((start, prev))
            start = m
        prev = m
    out.append((start, prev))
    return ", ".join(f"{a:%Y-%m}" + (f" → {b:%Y-%m}" if b != a else "") for a, b in out)


def overlay_line(info: pd.DataFrame) -> str:
    off = [a for a, g in info["gate"].items() if float(g) == 0.0]
    on = info[info["gate"].astype(float) > 0]
    sc = on["scale"].astype(float)
    fl = info["floored"].astype(int)
    short = info[info["history_short"].astype(bool)]
    return (f"gate off {len(off)} months ({_ranges(off)}) · scale mean {sc.mean():.2f} min {sc.min():.2f} "
            f"(<1 in {int((sc < 1).sum())} of {len(sc)} gate-on months) · floored {fl.min()}-{fl.max()} names "
            f"(mean {fl.mean():.1f}) · held {int(info['held'].min())}-{int(info['held'].max())} of "
            f"{int(info['priced'].min())}-{int(info['priced'].max())} priced"
            f"{'' if (info['held'] == info['priced']).all() else ' · SHORT BOOKS'} · history < 24 months in "
            f"{len(short)} months{f' (to {short.index[-1]:%Y-%m})' if len(short) else ''} · source "
            f"{', '.join(sorted(set(info['source'])))}")


def panel(u: str, ref, app: bool = False) -> dict:
    t0 = time.time()
    d = ss.load(u, holdout=True)
    key = "etf_book" if u == "etf_27" else u
    snaps = sb.snapshots(key)
    if u == "etf_27":
        snaps = [(t, s[~s["symbol"].isin(ss.ETF_YOUNG)].reset_index(drop=True)) for t, s in snaps]
    hist = None
    if app:
        syms = [s for s in sb.symbols(key) if s.replace(".NS", "") not in ss.ETF_YOUNG]
        hist = app_history(u, syms)
    W = weights(d, snaps, ref, history=hist)
    base = ss.baselines(d)
    runs = {SHIP: ss.run(lambda c: W["ship"][c.date], d), TESTED: ss.run(lambda c: W["tested"][c.date], d)}
    eras = list(ss.ERAS) if u != "etf_27" else [("window", None, None)]
    cells = {k: {e: cell(r, base, a, b) for e, a, b in eras} for k, r in runs.items()}
    full = {k: ss.metrics(v) for k, v in {**base, **runs}.items()}
    months = d["months"]
    print(f"\n== {d['name']} · {len(months) - 1} rebalances {months[0]:%Y-%m} → {months[-2]:%Y-%m} · every name · "
          f"net of {d['cost']:g}bp · {time.time() - t0:.0f}s", flush=True)
    print(f"   exactness  shipped vs max(managed_mom λ=1, ¼·CVG): max |Δw| {W['g_id'].max():.1e}"
          + (f" · vs reference mmom_s: max |Δw| {W['g_ref'].max():.1e}" if len(W["g_ref"]) else
             " · reference not given (--ref)"), flush=True)
    print(f"   overlay    {overlay_line(W['info'])}", flush=True)
    tab = pd.DataFrame({k: {e: ss.metrics(ss._era(r, a, b))["cagr"] for e, a, b in eras}
                        for k, r in {**base, **runs}.items()}).T
    for k in runs:
        tab.loc[f"  margin {k.split()[0]}"] = [cells[k][e]["margin"] for e, _, _ in eras]
        tab.loc[f"  t {k.split()[0]}"] = [cells[k][e]["t"] for e, _, _ in eras]
    print("   net CAGR % by era (margin = over the best of the eight in that era)")
    print("   " + tab.round(2).to_string().replace("\n", "\n   "), flush=True)
    print("   full span")
    f = pd.DataFrame(full).T[["months", "cagr", "vol", "ret_vol", "maxdd", "turnover"]]
    f["names"] = pd.Series({k: f"{int(v['names'].min())}-{int(v['names'].max())}"
                            for k, v in {**base, **runs}.items()})
    print("   " + f.loc[["EW", "CVG", TESTED, SHIP]].round(2).to_string().replace("\n", "\n   "), flush=True)
    print("   the floor's price (shipped − tested, %/yr): "
          + " · ".join(f"{e} {cells[SHIP][e]['cagr'] - cells[TESTED][e]['cagr']:+.2f}" for e, _, _ in eras)
          + f" · full span {full[SHIP]['cagr'] - full[TESTED]['cagr']:+.2f}", flush=True)
    return dict(d=d, W=W, base=base, runs=runs, cells=cells, full=full, eras=eras)


# ── (d) the point-in-time Dow ─────────────────────────────────────────────────────────────────────
@contextmanager
def point_in_time():
    init = ss.Ctx.__init__

    def pit_init(self, data, a):                       # the allocation universe = that day's members
        init(self, data, a)
        self.priced = pd.Index([s for s in self.priced if P.member(s, a)])
    ss.Ctx.__init__ = pit_init
    try:
        yield
    finally:
        ss.Ctx.__init__ = init


def pit(ref, app: bool = False) -> dict:
    t0 = time.time()
    d = pickle.load(open(P.PKL, "rb"))
    snaps = sb.repair(pickle.load(open(os.path.join(HERE, "cvg_reweight_dow_pit.pkl"), "rb")))
    hist = None
    if app:
        from universe import DOW_JONES_TICKERS
        hist = app_history("dow_pit", list(DOW_JONES_TICKERS) + list(P.ADDED))
    with point_in_time():
        W = weights(d, snaps, ref, members=P.member, history=hist)
        base = ss.baselines(d)
        runs = {SHIP: ss.run(lambda c: W["ship"][c.date], d), TESTED: ss.run(lambda c: W["tested"][c.date], d)}
    e3 = {k: ss._era(v, "2020-01-01", None) for k, v in {**base, **runs}.items()}
    t = pd.DataFrame({k: ss.metrics(v) for k, v in e3.items()}).T
    best = t.loc[list(ss.BASE), "cagr"].idxmax()
    t["vs_best"] = t["cagr"] - t.loc[best, "cagr"]
    t["t_best"] = [_t(v["ret"] - e3[best]["ret"]) if k != best else np.nan for k, v in e3.items()]
    t["names"] = pd.Series({k: f"{int(v['names'].min())}-{int(v['names'].max())}" for k, v in e3.items()})
    print(f"\n== Dow 30, point-in-time members · E3 2020-01 → {d['months'][-2]:%Y-%m} · net of costs · "
          f"best of the eight: {best} · {time.time() - t0:.0f}s", flush=True)
    print(f"   exactness  shipped vs max(managed_mom λ=1, ¼·CVG): max |Δw| {W['g_id'].max():.1e}"
          + (f" · vs reference mmom_s: max |Δw| {W['g_ref'].max():.1e}" if len(W["g_ref"]) else ""), flush=True)
    info_e3 = W["info"][W["info"].index >= pd.Timestamp("2020-01-01")]
    print(f"   overlay    {overlay_line(info_e3)}", flush=True)
    print("   " + t[["months", "cagr", "vol", "ret_vol", "maxdd", "turnover", "vs_best", "t_best", "names"]]
          .round(2).to_string().replace("\n", "\n   "), flush=True)
    return dict(W=W, table=t, best=best)


def summary(res: dict, p: dict) -> None:
    stock = [u for u in ("nifty_50", "dow_30") if u in res]
    cols = [(u, e) for u in stock for e, _, _ in res[u]["eras"]]
    print("\n══ THE BAR · six stock cells (E1, E2, E3 × Nifty 50, Dow 30); the ETF window reported ═══════", flush=True)

    def row(label, vals):
        print(f"  {label:22s}" + "".join(f"{v:>9s}" for v in vals), flush=True)
    row("", [f"{u.split('_')[0]} {e}" for u, e in cols] + (["ETF"] if "etf_27" in res else []))
    etf = res.get("etf_27")
    c = lambda k, u, e: res[u]["cells"][k][e]                                  # noqa: E731
    row("best of the eight", [f"{c(SHIP, u, e)['best_cagr']:.2f}{ABBR[c(SHIP, u, e)['best']]}" for u, e in cols]
        + ([f"{c(SHIP, 'etf_27', 'window')['best_cagr']:.2f}{ABBR[c(SHIP, 'etf_27', 'window')['best']]}"]
           if etf else []))
    for k in (SHIP, TESTED):
        row(k, [f"{c(k, u, e)['cagr']:.2f}" for u, e in cols] + ([f"{c(k, 'etf_27', 'window')['cagr']:.2f}"] if etf else []))
        row("  margin", [f"{c(k, u, e)['margin']:+.2f}" for u, e in cols]
            + ([f"{c(k, 'etf_27', 'window')['margin']:+.2f}"] if etf else []))
        row("  (t)", [f"({c(k, u, e)['t']:+.2f})" for u, e in cols]
            + ([f"({c(k, 'etf_27', 'window')['t']:+.2f})"] if etf else []))
        won = sum(c(k, u, e)["margin"] > 0 for u, e in cols)
        print(f"  {'':22s}cells won {won}/{len(cols)}"
              + (f" · ETF vs EW {c(k, 'etf_27', 'window')['vs_ew']:+.2f} (t {c(k, 'etf_27', 'window')['t_ew']:+.2f})"
                 if etf else ""), flush=True)
    if p is not None:
        t = p["table"]
        print(f"  point-in-time Dow E3: {SHIP} {t.loc[SHIP, 'cagr']:.2f} · CVG {t.loc['CVG', 'cagr']:.2f} · "
              f"EW {t.loc['EW', 'cagr']:.2f} · {TESTED} {t.loc[TESTED, 'cagr']:.2f} · best {p['best']} → "
              f"MMOM vs CVG {t.loc[SHIP, 'cagr'] - t.loc['CVG', 'cagr']:+.2f}, vs best {t.loc[SHIP, 'vs_best']:+.2f} "
              f"(t {t.loc[SHIP, 't_best']:+.2f}), vs EW {t.loc[SHIP, 'cagr'] - t.loc['EW', 'cagr']:+.2f}, "
              f"vs tested {t.loc[SHIP, 'cagr'] - t.loc[TESTED, 'cagr']:+.2f}", flush=True)
    gaps = [res[u]["W"]["g_id"] for u in res] + ([p["W"]["g_id"]] if p else [])
    refs = [res[u]["W"]["g_ref"] for u in res] + ([p["W"]["g_ref"]] if p else [])
    gi, gr = pd.concat(gaps), pd.concat(refs)
    print(f"  exactness over {len(gi)} month-books: max |Δw| vs identity {gi.max():.1e} "
          f"({int((gi > 1e-12).sum())} > 1e-12)"
          + (f" · vs reference {gr.max():.1e} ({int((gr > 1e-12).sum())} > 1e-12)" if len(gr) else ""), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ref", help="path to the scratch reference of the shipped form (defines mmom_s(ctx))")
    ap.add_argument("--app-history", action="store_true",
                    help="read the overlay's history the app's way (fetch_close_history from 2006)")
    ap.add_argument("panels", nargs="*", default=list(PANELS) + ["pit"])
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    ref = _module(args.ref) if args.ref else None
    res = {u: panel(u, ref, args.app_history) for u in PANELS if u in args.panels}
    p = pit(ref, args.app_history) if "pit" in args.panels else None
    summary(res, p)


if __name__ == "__main__":
    main()
