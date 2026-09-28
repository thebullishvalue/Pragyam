"""
research/cvg_flip.py — the state map plus the histogram flipping green, as an allocator tilt.

PRE-REGISTRATION (fixed before any number below was seen)
─────────────────────────────────────────────────────────
Signal: the pane's histogram ('conv hist') crosses from ≤ 0 to > 0 — the column turns green. On a
monthly book it enters as a tilt: a name whose histogram flipped green within the last 10 sessions
holds 1.5× its C4 units, when the flip happened in the named states (the state on the flip day):

    C4       shipped units, no tilt                                   (the control)
    F-any    flip in any state                                        (does the flip alone help?)
    F-cheap  flip in a cheap state: Dislocated, Basing, Turned        (the map's cheap column)
    F-cap    flip in Dislocated (capitulation)                        (the map's top cell)
    EW       equal weight

Method as research/cvg_mom.py (monthly, 10% cap, 10bp / 3bp one-way, every-name and top-30
books, Nifty 50 and Dow 30, eras E1 < 2014 / E2 2014-2019 / E3 ≥ 2020).

DECISION RULE: a flip tilt is 'better' only if it beats C4 in ALL THREE eras on BOTH universes
(every-name book) AND beats F-any (the state map must add something to the flip).

RESULT (2026-09-28) — REJECTED. %/yr vs C4, net of costs, E1 / E2 / E3:
    Nifty 50 all     F-any −0.38 −0.15 −0.10 · F-cheap +0.04 −0.07 −0.01 · F-cap −0.01 +0.01 −0.04
    Nifty 50 top-30  F-any +0.01 −0.81 −0.60 · F-cheap −0.08 −0.10 +0.01 · F-cap −0.03 −0.01 −0.04
    Dow 30 all       F-any +0.02 −0.32 −0.02 · F-cheap +0.11 +0.09 −0.04 · F-cap +0.04 +0.05 −0.06
No variant beats C4 in all three eras on both universes. The state-conditioned tilts sit within
±0.1 %/yr of C4 (they touch few names); the flip in any state costs return and adds turnover.
Sanket's lab agrees: the green flip alone reads 0.000 in every era; conditioned on the grid it
is positive only in washout (+0.02 to +0.04σ, every era, both scorers) — about half the shipped
▲'s size — and capitulation-conditioned flips fail 2014-19.
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
VARIANTS = {"C4": ({}, None), "F-any": ({}, "any"),
            "F-cheap": ({}, {"DISLOCATED", "BASING", "TURNED"}), "F-cap": ({}, {"DISLOCATED"})}
LOOK, TILT = 10, 1.5
COST = {"Nifty 50": 10.0, "Dow 30": 3.0}
ERAS = (("E1", None, "2014-01-01"), ("E2", "2014-01-01", "2020-01-01"), ("E3", "2020-01-01", None))
_orig_units = nco.cvg_units
_CTX = {"mode": None, "hist": None}


def _flipped(symbols, states):
    """Names whose histogram turned green within the last LOOK sessions, in `states` then."""
    hist = _CTX["hist"][-(LOOK + 1):]
    H, S = {}, {}
    for d, s in hist:
        s = s.drop_duplicates("symbol", keep="last").set_index("symbol")
        H[d] = pd.to_numeric(s.get("conv hist"), errors="coerce")
        S[d] = s.get("cvg state")
    H = pd.DataFrame(H).T.reindex(columns=symbols)
    S = pd.DataFrame(S).T.reindex(columns=symbols)
    flip = (H > 0) & (H.shift(1) <= 0)
    if states != "any":
        flip &= S.isin(states)
    return flip.iloc[1:].any(axis=0)


def _units(readings):
    u = _orig_units(readings)
    mode = _CTX["mode"]
    if mode is None or _CTX["hist"] is None:
        return u
    f = _flipped(list(readings.index), mode).reindex(readings.index).fillna(False).to_numpy(bool)
    return u * np.where(f, TILT, 1.0)


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
    pickle.dump(res, open(os.path.join(HERE, "cvg_flip_results.pkl"), "wb"))
