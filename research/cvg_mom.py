"""
research/cvg_mom.py — can the Conviction-Value Grid be styled for MOMENTUM, and would it do better?

PRE-REGISTRATION (fixed before any number below was seen)
─────────────────────────────────────────────────────────
The shipped units are a reversion map: cheap names average ~4.9× the units of rich ones, and the
top cell is Dislocated (cheap, sellers still in control). The momentum alternatives restyle the
same 3 × 3 map (rows DOWN / FAINT / UP by the conviction tape, columns cheap / fair / rich by the
value tape) — the engine, the states and the allocator are untouched; only units change:

    C4    current            DOWN 4 / 1.5 / .25 · FAINT 1.5 / 1 / .75 · UP 3 / 1.5 / .75
    M1    trend              DOWN .25 · FAINT 1 · UP 3, whatever the value column
    M2    strength           DOWN .25 / .25 / .5 · FAINT .5 / 1 / 1.5 · UP 1.5 / 3 / 4
    M3    trend pullback     DOWN .5 / .25 / .25 · FAINT 1 / 1 / .75 · UP 4 / 3 / 1.5
    M4    confirmed trend    M1, UP × 4/3 when conviction momentum confirms the row, × 2/3 not
    R12   classic momentum   by 12-1 month return tercile across the universe: 3 / 1 / .25
                             (not a grid map: the reference for whether momentum paid here at all)
    EW    equal weight

Method as research/cvg_v9.py: backdata snapshots → nco.compute_nco_portfolio(method="CVG"),
monthly, the app's 10% cap, net of 10bp (India) / 3bp (US) one-way costs; the every-name book and
the top-30 book (Nifty); Nifty 50 and Dow 30; eras E1 < 2014, E2 2014-2019, E3 ≥ 2020.

DECISION RULE: a momentum map is 'better' only if it beats C4 in ALL THREE eras on BOTH universes
(every-name book). Otherwise the reversion map stays.

RESULT (2026-09-28) — REJECTED. Every momentum map lost to C4 in every era on both universes, net
of costs, at higher turnover (%/yr vs C4, E1 / E2 / E3):
    Nifty 50 all     M1 −1.20 −1.82 −1.47 · M2 −1.50 −2.35 −1.73 · M3 −0.83 −1.25 −1.15 · M4 −1.69 −0.97 −1.64
    Nifty 50 top-30  M1 −1.90 −3.79 −3.60 · M2 −2.42 −4.42 −3.94 · M3 −1.54 −3.25 −2.63 · M4 −2.31 −2.77 −3.94
    Dow 30 all       M1 −1.09 −0.82 −0.94 · M2 −0.99 −0.92 −1.24 · M3 −1.11 −0.61 −0.82 · M4 −1.28 −0.93 −0.67
The reference R12 (12-1 month momentum, not a grid map) beat C4 on Nifty in 2014-19 and 2020+
(+1.38, +0.46) but lost −2.82 before 2014 and −1.04 on Dow after 2020: not consistent either.
Sanket's lab (380 instruments, look-ahead-free) agrees: the momentum maps read negative on
time-series scoring in E1 and E3 at every horizon; cross-sectionally they beat C4 only after 2020
at 40-60 bars. The one class where they win clearly is crypto.
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cvgrid                                     # noqa: E402
import nco                                        # noqa: E402

CAPITAL, CAP = 1e10, 0.10
BASE = dict(cvgrid.STATE_UNITS)
assert BASE["DISLOCATED"] == 4.0, "C4 is the shipped control"
ORDER = ("DISLOCATED", "FADING", "DISTRIBUTION", "BASING", "IDLE", "STALLING", "TURNED", "BUILDING", "PAID")
_m = lambda v: dict(zip(ORDER, v))                  # noqa: E731
VARIANTS = {
    "C4": ({}, None),
    "M1": (_m([.25, .25, .25, 1, 1, 1, 3, 3, 3]), None),
    "M2": (_m([.25, .25, .5, .5, 1, 1.5, 1.5, 3, 4]), None),
    "M3": (_m([.5, .25, .25, 1, 1, .75, 4, 3, 1.5]), None),
    "M4": (_m([.25, .25, .25, 1, 1, 1, 3, 3, 3]), "confirm"),
    "R12": ({}, "r12"),
}
COST = {"Nifty 50": 10.0, "Dow 30": 3.0}
ERAS = (("E1", None, "2014-01-01"), ("E2", "2014-01-01", "2020-01-01"), ("E3", "2020-01-01", None))
UP = {"TURNED", "BUILDING", "PAID"}

_orig_units = nco.cvg_units
_CTX = {"mode": None, "hist": None}


def _units(readings):
    mode = _CTX["mode"]
    if mode == "r12":
        hist = _CTX["hist"]
        px = pd.DataFrame({pd.Timestamp(d): pd.to_numeric(s.drop_duplicates("symbol", keep="last")
                                                          .set_index("symbol")["price"], errors="coerce")
                           for d, s in hist}).T.sort_index()
        px = px.reindex(columns=readings.index)
        if len(px) < 253:
            return pd.Series(1.0, index=readings.index)
        r = np.log(px.iloc[-22] / px.iloc[-253])
        q = r.rank(pct=True)
        return pd.Series(np.where(q > 2 / 3, 3.0, np.where(q > 1 / 3, 1.0, 0.25)), index=readings.index).where(r.notna(), 1.0)
    u = _orig_units(readings)
    if mode == "confirm":
        cd = pd.to_numeric(readings.get("conviction_daily"), errors="coerce")
        ct = pd.to_numeric(readings.get("conviction"), errors="coerce")
        ld = pd.to_numeric(readings.get("ladder_down"), errors="coerce").fillna(0.0)
        mom = np.where(ld == 1, ct - cd, cd - ct)            # the faster view minus the slower
        up = readings["state"].isin(UP).to_numpy()
        f = np.where(up, np.where(np.nan_to_num(mom) > 0, 4 / 3, 2 / 3), 1.0)
        return u * f
    return u


nco.cvg_units = _units


def set_variant(key):
    over, mode = VARIANTS[key]
    cvgrid.STATE_UNITS.clear(); cvgrid.STATE_UNITS.update(BASE); cvgrid.STATE_UNITS.update(over)
    _CTX["mode"] = mode


def weights(hist, method, n):
    snap = hist[-1][1]
    _CTX["hist"] = hist[-253:]
    prices = {s: float(p) for s, p in zip(snap["symbol"], pd.to_numeric(snap["price"], errors="coerce"))
              if np.isfinite(p) and p > 0}
    book = nco.compute_nco_portfolio(hist[-253:], prices, CAPITAL, n, method=method, max_pos_pct=CAP)
    if book.empty:
        return pd.Series(dtype=float)
    v = pd.to_numeric(book["value"], errors="coerce")
    return pd.Series((v / v.sum()).to_numpy(), index=book["symbol"].astype(str).to_numpy())


def backtest(snaps, n_book):
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in snaps])
    px = {pd.Timestamp(d): pd.to_numeric(df.drop_duplicates("symbol", keep="last").set_index("symbol")["price"],
                                         errors="coerce") for d, df in snaps}
    months = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first().to_list()
    pos = {d: i for i, d in enumerate(dates)}
    rows, prev = [], {}
    for a, b in zip(months[:-1], months[1:]):
        hist = snaps[: pos[a] + 1]
        n = len(hist[-1][1]) if n_book is None else n_book
        rec = {"date": a}
        for key in list(VARIANTS) + ["EW"]:
            if key == "EW":
                set_variant("C4"); w = weights(hist, "EQUAL", n)
            else:
                set_variant(key); w = weights(hist, "CVG", n)
            r = (px[b].reindex(w.index) / px[a].reindex(w.index) - 1.0).fillna(0.0)
            rec[key] = float((w * r).sum())
            if key in prev:
                w0, r0 = prev[key]; d = w0 * (1 + r0); d = d / d.sum()
                idx = w.index.union(d.index)
                rec[f"to::{key}"] = float(0.5 * (w.reindex(idx, fill_value=0) - d.reindex(idx, fill_value=0)).abs().sum())
            prev[key] = (w, r)
        rows.append(rec)
    set_variant("C4")
    return pd.DataFrame(rows).set_index("date")


def summarise(bt, cost_bps):
    bt = bt.copy()
    keys = list(VARIANTS) + ["EW"]
    for k in keys:
        bt[k] = bt[k] - bt.get(f"to::{k}", pd.Series(0.0, index=bt.index)).fillna(0.0) * cost_bps / 1e4
    out = []
    for era, a, b in ERAS:
        m = np.ones(len(bt), bool)
        if a: m &= bt.index >= pd.Timestamp(a)
        if b: m &= bt.index < pd.Timestamp(b)
        x = bt[m]
        for k in keys:
            d = x[k] - x["C4"]
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
            out.append(dict(era=era, variant=k, months=len(d), vs_C4_pa=d.mean() * 12 * 100, t=t,
                            cagr=((1 + x[k]).prod() ** (12 / len(x)) - 1) * 100,
                            vol=x[k].std() * np.sqrt(12) * 100,
                            turnover_pa=x.get(f"to::{k}", pd.Series(np.nan)).mean() * 12))
    return pd.DataFrame(out)


if __name__ == "__main__":
    res = {}
    for name, f in (("Nifty 50", "nifty_50"), ("Dow 30", "dow_30")):
        snaps = pickle.load(open(os.path.join(HERE, f"cvg_reweight_{f}.pkl"), "rb"))
        for book, n in (("all", None), ("top30", 30)):
            if name == "Dow 30" and book == "top30":
                continue
            bt = backtest(snaps, n)
            res[(name, book)] = bt
            s = summarise(bt, COST[name])
            print(f"\n== {name} · {book} book (net of costs, %/yr)", flush=True)
            print(s.pivot_table(index="variant", columns="era", values=["vs_C4_pa", "t"]).round(2).to_string(), flush=True)
            print(s.pivot_table(index="variant", columns="era", values=["cagr", "vol", "turnover_pa"]).round(2).to_string(), flush=True)
    pickle.dump(res, open(os.path.join(HERE, "cvg_mom_results.pkl"), "wb"))
