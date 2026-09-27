"""
research/cvg_reweight.py — does Sanket's capitulation finding transfer to Pragyam's 3 × 3 book?

PRE-REGISTRATION (fixed before any number below was seen)
─────────────────────────────────────────────────────────
Sanket's from-scratch audit of pragati.pine (thebullishvalue/sanket, studies/pine_audit.md)
found that names with sellers FIRMLY in control at a cheap or below-fair price were followed
by gains over 10–20 bars in both eras on every asset class but crypto. Pragyam's grid differs:
3 × 3 rows (UP / FAINT / DOWN at ±30), monthly rebalances, a graded map, every name held. So
the finding is a hypothesis here, not a result.

Variants — each changes STATE_UNITS only, in memory; everything else is the shipped pipeline:
    V0  current                       DISLOCATED 1 · FADING 0.5 · BUILDING 3 · PAID 1.5
    V1  DISLOCATED 1 → 3              the capitulation analogue (sellers in control, cheap)
    V2  V1 + FADING 0.5 → 1.5         + the washout analogue (sellers in control, fair)
    V3  V2 + BUILDING 3 → 1.5         + the downgrade of buyers in control at a fair price
    V4  V3 + PAID 1.5 → 0.75          + buyers in control, rich
    EW  equal weight                  the bar every Pragyam style is held to

Method: backdata.generate_historical_data → nco.compute_nco_portfolio(method="CVG"/"EQUAL"),
every name held (num_positions = the universe), the app's 10% position cap, first trading day
of each month, held to the next. Three universes, as Pragyam's CVG evidence uses: the ETF book,
Nifty 50 and Dow 30 (today's constituents — survivorship bias applies to every variant alike).
Eras: DISCOVERY before 2018-01-01, HOLDOUT from it.

DECISION RULE: pick the variant with the best average paired monthly improvement over V0 on
DISCOVERY. Change Pragyam's units only if that variant ALSO beats V0 on the HOLDOUT on average
AND in at least two of the three universes in EACH era. Otherwise document and leave the units.

Run:  python research/cvg_reweight.py            (caches snapshots to research/*.pkl)
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cvgrid                                     # noqa: E402
from backdata import generate_historical_data     # noqa: E402
from nco import compute_nco_portfolio             # noqa: E402

SPLIT = pd.Timestamp("2018-01-01")
START, END = datetime(2006, 1, 1), datetime(2026, 9, 25)
CAPITAL = 1e10            # large enough that integer units do not distort the weights
CAP = 0.10                # the app's default position cap

BASE = dict(cvgrid.STATE_UNITS)
VARIANTS = {
    "V0 current": {},
    "V1 DISLOCATED 3": {"DISLOCATED": 3.0},
    "V2 + FADING 1.5": {"DISLOCATED": 3.0, "FADING": 1.5},
    "V3 + BUILDING 1.5": {"DISLOCATED": 3.0, "FADING": 1.5, "BUILDING": 1.5},
    "V4 + PAID 0.75": {"DISLOCATED": 3.0, "FADING": 1.5, "BUILDING": 1.5, "PAID": 0.75},
}


def universes() -> dict:
    from universe import DOW_JONES_TICKERS, ETF_UNIVERSE, get_index_stock_list
    nifty, _ = get_index_stock_list("NIFTY 50")
    return {"ETF book": list(ETF_UNIVERSE), "Nifty 50": list(nifty or []),
            "Dow 30": list(DOW_JONES_TICKERS)}


def snapshots(name: str, symbols: list) -> list:
    p = os.path.join(HERE, f"cvg_reweight_{name.replace(' ', '_').lower()}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    snaps = generate_historical_data(symbols, START, END)
    pickle.dump(snaps, open(p, "wb"))
    return snaps


def set_units(over: dict) -> None:
    cvgrid.STATE_UNITS.clear()
    cvgrid.STATE_UNITS.update(BASE)
    cvgrid.STATE_UNITS.update(over)


def weights(hist: list, method: str, n: int) -> pd.Series:
    snap = hist[-1][1]
    prices = {s: float(p) for s, p in zip(snap["symbol"], pd.to_numeric(snap["price"], errors="coerce"))
              if np.isfinite(p) and p > 0}
    book = compute_nco_portfolio(hist[-1:], prices, CAPITAL, n, method=method, max_pos_pct=CAP)
    if book.empty:
        return pd.Series(dtype=float)
    v = pd.to_numeric(book["value"], errors="coerce")
    return pd.Series((v / v.sum()).to_numpy(), index=book["symbol"].astype(str).to_numpy())


def backtest(snaps: list) -> pd.DataFrame:
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in snaps])
    px = {pd.Timestamp(d): pd.to_numeric(df.set_index("symbol")["price"], errors="coerce") for d, df in snaps}
    months = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first().to_list()
    pos = {d: i for i, d in enumerate(dates)}
    rows, prev = [], {}
    for a, b in zip(months[:-1], months[1:]):
        hist = snaps[: pos[a] + 1]
        n = len(hist[-1][1])
        p0, p1 = px[a], px[b]
        rec = {"date": a}
        for key, over in list(VARIANTS.items()) + [("EW", None)]:
            if over is None:
                set_units({}); w = weights(hist, "EQUAL", n)
            else:
                set_units(over); w = weights(hist, "CVG", n)
            r = (p1.reindex(w.index) / p0.reindex(w.index) - 1.0).fillna(0.0)   # a name gone = cash
            rec[key] = float((w * r).sum())
            # one-way turnover into this book from last month's book, drifted to today's prices
            old = prev.get(key)
            if old is not None:
                w_old, r_old = old
                drift = w_old * (1.0 + r_old)
                drift = drift / drift.sum() if drift.sum() > 0 else drift
                idx = w.index.union(drift.index)
                rec[f"to::{key}"] = float(0.5 * (w.reindex(idx, fill_value=0) - drift.reindex(idx, fill_value=0)).abs().sum())
            prev[key] = (w, r)
        rows.append(rec)
    set_units({})
    return pd.DataFrame(rows).set_index("date")


COST_BPS = {"ETF book": 10.0, "Nifty 50": 10.0, "Dow 30": 3.0}   # per unit of one-way turnover


def summarise(bt: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    bt = bt.copy()
    for k in list(VARIANTS) + ["EW"]:          # net of costs: one-way turnover × cost, charged monthly
        to = bt.get(f"to::{k}", pd.Series(0.0, index=bt.index)).fillna(0.0)
        bt[k] = bt[k] - to * cost_bps / 1e4
    out = []
    for era, m in (("discovery", bt.index < SPLIT), ("holdout", bt.index >= SPLIT)):
        x = bt[m]
        for k in VARIANTS:
            for ref in ("V0 current", "EW"):
                if k == ref:
                    continue
                d = x[k] - x[ref]
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
                tk = x.get(f"to::{k}", pd.Series(np.nan, index=x.index)).mean() * 12
                out.append(dict(era=era, variant=k, vs=ref, months=len(d),
                                excess_pa=d.mean() * 12 * 100, t=t, turnover_pa=tk))
    return pd.DataFrame(out)


if __name__ == "__main__":
    res = {}
    for name, syms in universes().items():
        print(f"== {name}: {len(syms)} symbols", flush=True)
        bt = backtest(snapshots(name, syms))
        res[name] = bt
        print(summarise(bt, COST_BPS[name]).round(3).to_string(index=False), flush=True)
    pickle.dump(res, open(os.path.join(HERE, "cvg_reweight_results.pkl"), "wb"))
