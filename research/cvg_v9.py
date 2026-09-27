"""
research/cvg_v9.py — does the CVG hold as it states, and does a long-horizon trend axis help?

PRE-REGISTRATION (fixed before any number below was seen)
─────────────────────────────────────────────────────────
Sanket's v9 audit (studies/pragati_v9_audit.md, look-ahead-free scoring, three eras) found the
grid's edge lives in three cells: Dislocated (capitulation) +, Fading (washout) weakly +,
Distribution −. Basing, Building, Stalling and Paid — units 1.5 / 1.5 / 0.75 / 0.75 — read ≈ 0:
their units state a view the data does not support. A 200-day trend split inside the cells was
too collinear with the tapes to say anything; across names the 200-day trend was positive in
every era. So two hypotheses, each tested through the shipped pipeline:

    V8    current units                                   (the control)
    M     measured units: Basing 1, Building 1, Stalling 1, Paid 1 (others unchanged)
    T     V8 × trend tilt: ×1.25 above the 200-day mean, ×0.8 below
    MT    M × the same tilt
    EW    equal weight

Method: backdata snapshots → nco.compute_nco_portfolio(method="CVG"/"EQUAL"), monthly, the app's
10% cap, net of 10bp (India) / 3bp (US) one-way costs, two books: every name held, and the app's
default top-30. Universes: Nifty 50 and Dow 30 (today's constituents — survivorship applies to
every variant alike). Eras: E1 < 2014, E2 2014-2019, E3 ≥ 2020.

DECISION RULE: a variant replaces V8 only if it beats V8 in ALL THREE eras on BOTH universes
(every-name book). Among variants that pass, the simplest (fewest changes) wins.

ITERATION 2 (registered after iteration 1 failed — M, T and MT all rejected; the rule unchanged):
    P     V8, but a Dislocated name whose value is still CHEAPENING (chart value below the value
          tape: the fast end widening) holds 2.25 units — half-way to Fading, the Pine's 5 × 5
          rule — while one whose value is reverting keeps its 3. Sanket's v9 audit: capitulation
          with value reverting beat widening in every era.
    C4    V8 with Dislocated at 4 units (sensitivity: is the one robust cell under-weighted?)
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
MEASURED = {"BASING": 1.0, "BUILDING": 1.0, "STALLING": 1.0, "PAID": 1.0}
VARIANTS = {"V8": ({}, False), "M": (MEASURED, False), "T": ({}, True), "MT": (MEASURED, True),
            "P": ({}, "phase"), "C4": ({"DISLOCATED": 4.0}, False)}
if os.environ.get("CVG_ITER") == "2":
    VARIANTS = {k: VARIANTS[k] for k in ("V8", "P", "C4")}
TILT_UP, TILT_DN = 1.25, 0.8
COST = {"Nifty 50": 10.0, "Dow 30": 3.0}
ERAS = (("E1", None, "2014-01-01"), ("E2", "2014-01-01", "2020-01-01"), ("E3", "2020-01-01", None))

_orig_units = nco.cvg_units
_TILT = {"on": False, "snap": None}


def _tilted_units(readings):
    u = _orig_units(readings)
    if _TILT["on"] == "phase":
        vd = pd.to_numeric(readings.get("value_daily"), errors="coerce")
        vt = pd.to_numeric(readings.get("value_tape"), errors="coerce")
        widening = (readings["state"] == "DISLOCATED") & (vd < vt)
        return u * np.where(widening.fillna(False).to_numpy(), 0.75, 1.0)
    if not _TILT["on"] or _TILT["snap"] is None:
        return u
    s = _TILT["snap"].drop_duplicates("symbol", keep="last").set_index("symbol")
    px = pd.to_numeric(s.get("price"), errors="coerce").reindex(u.index)
    ma = pd.to_numeric(s.get("ma200 latest"), errors="coerce").reindex(u.index)
    up = (px > ma)
    tilt = np.where(ma.notna() & px.notna(), np.where(up, TILT_UP, TILT_DN), 1.0)
    return u * tilt


nco.cvg_units = _tilted_units


def set_variant(key):
    over, tilt = VARIANTS[key]
    cvgrid.STATE_UNITS.clear(); cvgrid.STATE_UNITS.update(BASE); cvgrid.STATE_UNITS.update(over)
    _TILT["on"] = tilt


def weights(hist, method, n):
    snap = hist[-1][1]
    _TILT["snap"] = snap
    prices = {s: float(p) for s, p in zip(snap["symbol"], pd.to_numeric(snap["price"], errors="coerce"))
              if np.isfinite(p) and p > 0}
    book = nco.compute_nco_portfolio(hist[-253:], prices, CAPITAL, n, method=method, max_pos_pct=CAP)
    if book.empty:
        return pd.Series(dtype=float)
    v = pd.to_numeric(book["value"], errors="coerce")
    return pd.Series((v / v.sum()).to_numpy(), index=book["symbol"].astype(str).to_numpy())


def backtest(snaps, n_book):
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in snaps])
    px = {pd.Timestamp(d): pd.to_numeric(df.set_index("symbol")["price"], errors="coerce") for d, df in snaps}
    months = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first().to_list()
    pos = {d: i for i, d in enumerate(dates)}
    rows, prev = [], {}
    for a, b in zip(months[:-1], months[1:]):
        hist = snaps[: pos[a] + 1]
        n = len(hist[-1][1]) if n_book is None else n_book
        rec = {"date": a}
        for key in list(VARIANTS) + ["EW"]:
            if key == "EW":
                set_variant("V8"); w = weights(hist, "EQUAL", n)
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
    set_variant("V8")
    return pd.DataFrame(rows).set_index("date")


def summarise(bt, cost_bps):
    bt = bt.copy()
    for k in list(VARIANTS) + ["EW"]:
        bt[k] = bt[k] - bt.get(f"to::{k}", pd.Series(0.0, index=bt.index)).fillna(0.0) * cost_bps / 1e4
    out = []
    for era, a, b in ERAS:
        m = np.ones(len(bt), bool)
        if a: m &= bt.index >= pd.Timestamp(a)
        if b: m &= bt.index < pd.Timestamp(b)
        x = bt[m]
        for k in list(VARIANTS) + ["EW"]:
            for ref in ("V8", "EW"):
                if k == ref:
                    continue
                d = x[k] - x[ref]
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
                out.append(dict(era=era, variant=k, vs=ref, months=len(d), excess_pa=d.mean() * 12 * 100, t=t,
                                cagr=((1 + x[k]).prod() ** (12 / len(x)) - 1) * 100,
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
            print(s.pivot_table(index=["variant", "vs"], columns="era", values=["excess_pa", "t"]).round(2).to_string(), flush=True)
            print(s[s.vs == "EW"].pivot_table(index="variant", columns="era", values=["cagr", "turnover_pa"]).round(2).to_string(), flush=True)
    pickle.dump(res, open(os.path.join(HERE, "cvg_v9_results.pkl"), "wb"))
