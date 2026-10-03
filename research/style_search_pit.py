"""
research/style_search_pit.py — the Dow 30 holdout re-read on point-in-time membership (a sensitivity).

REGISTERED AFTER THE HOLDOUT WAS SEEN — a check on it, not a new test. The style search's holdout
(research/style_search_holdout.py) found managed_mom clearing all six cells. Its E3 margin over CVG
on the Dow came from NVDA, AMZN and CRM, held through runs that preceded their joining the Dow:
the panel is TODAY's constituents. This harness rebuilds the Dow as it stood each day of E3:

    members   S&P Dow Jones Indices' changes since 2020-01-01:
              2020-08-31  out XOM, PFE, RTX        in CRM, AMGN, HON
              2024-02-26  out WBA                  in AMZN
              2024-11-08  out INTC, DOW            in NVDA, SHW
              AXP — a member throughout, absent from this repo's Dow list — is added.
              WBA (a member until 2024-02, taken private 2025) has no yfinance history: the
              universe is one name short for those 50 months, for every style alike.
    method    every name priced from 2006 (backdata snapshots, the same pipeline); each month the
              shipped styles' raw weights are rebuilt over the names that were members that day
              only, and every style and finalist may hold only those names. Signals may read a
              name's prices from before it joined (that history was public).
    scored    E3 (2020-01 → 2026-09) only; the rebalances from 2019 seed turnover.

Run:  python research/style_search_pit.py

RESULT (2026-10-03) — E3 net CAGR on the point-in-time Dow (29-30 members; WBA missing):
  CVG 11.47 (best existing) · EW 11.00 · CVG+EW 11.24 · ERC 10.69 · HRP 9.53
  A capit_rev_cvg λ=2 11.61 (+0.14, t 0.29) · λ=1 11.37 (−0.10)
  B managed_mom   λ=1 11.41 (−0.06, t −0.15) · λ=2 10.85 (−0.63; holds 20-29 names)
  C kelly_egr λ=2  7.13 (−4.34) · D ivol_tilt 11.35 (−0.12) · E V_REGIME 10.58 (−0.90)
On today's constituents managed_mom led CVG by +0.39 / +0.23 here; on the members of the day it
does not. Every style earns less (CVG 15.06 → 11.47): the late entrants flattered them all, and
momentum and the high-volatility Kelly book most.
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import style_blends as sb                         # noqa: E402
import style_search as ss                         # noqa: E402
import style_search_holdout as H                  # noqa: E402

ADDED = ("AXP", "INTC", "PFE", "XOM", "RTX")
JOINED = {"CRM": "2020-08-31", "AMGN": "2020-08-31", "HON": "2020-08-31", "AMZN": "2024-02-26",
          "NVDA": "2024-11-08", "SHW": "2024-11-08"}
LEFT = {"XOM": "2020-08-31", "PFE": "2020-08-31", "RTX": "2020-08-31", "INTC": "2024-11-08",
        "DOW": "2024-11-08"}
START_RUN = pd.Timestamp("2019-01-01")
PKL = os.path.join(HERE, "search_dow_pit_full.pkl")


def member(s: str, a: pd.Timestamp) -> bool:
    if s in JOINED and a < pd.Timestamp(JOINED[s]):
        return False
    if s in LEFT and a >= pd.Timestamp(LEFT[s]):
        return False
    return True


def build() -> dict:
    from backdata import generate_historical_data
    from universe import DOW_JONES_TICKERS
    snap_pkl = os.path.join(HERE, "cvg_reweight_dow_pit.pkl")
    if os.path.exists(snap_pkl):
        snaps = pickle.load(open(snap_pkl, "rb"))
    else:
        snaps = generate_historical_data(list(DOW_JONES_TICKERS) + list(ADDED), datetime(2006, 1, 1),
                                         datetime(2026, 10, 2))
        pickle.dump(snaps, open(snap_pkl, "wb"))
    snaps = sb.unstale(snaps)
    by_date = {pd.Timestamp(d): s.drop_duplicates("symbol", keep="last").set_index("symbol") for d, s in snaps}
    px = pd.DataFrame({d: pd.to_numeric(s["price"], errors="coerce") for d, s in by_date.items()}).T.sort_index()
    num = [c for c in snaps[-1][1].columns if c not in ("date", "symbol", "price")
           and pd.to_numeric(snaps[-1][1][c], errors="coerce").notna().any()]
    panels = {c: pd.DataFrame({d: pd.to_numeric(s[c], errors="coerce") for d, s in by_date.items()}).T.sort_index()
              for c in num}
    cal = px.index
    months = [m for m in pd.Series(cal, index=cal).groupby([cal.year, cal.month]).first() if m >= START_RUN]
    pos = {d: i for i, d in enumerate(cal)}
    raw, snap = {k: {} for k in sb.METHOD}, {}
    for a in months:
        hist = [(d, s[s["symbol"].map(lambda x: member(x, a))].reset_index(drop=True))
                for d, s in snaps[max(0, pos[a] - 252): pos[a] + 1]]
        for k, m in sb.METHOD.items():
            raw[k][a] = sb.raw(hist, m)
        snap[a] = by_date[a]
    d = dict(u="dow_pit", name="Dow 30 (point-in-time)", cost=3.0, px=px, panels=panels, months=months,
             raw=raw, snap=snap)
    pickle.dump(d, open(PKL, "wb"))
    return d


def main() -> None:
    pd.set_option("display.width", 250)
    d = pickle.load(open(PKL, "rb")) if os.path.exists(PKL) else build()
    init = ss.Ctx.__init__

    def pit_init(self, data, a):                       # the allocation universe = that day's members
        init(self, data, a)
        self.priced = pd.Index([s for s in self.priced if member(s, a)])
    ss.Ctx.__init__ = pit_init
    base = ss.baselines(d)
    cands = {H._label(p, n, q): ss.run(H.build_fn(p, n, q, f, d), d) for p, n, q, f in H.FINALISTS}
    ss.Ctx.__init__ = init
    e3 = {k: ss._era(v, "2020-01-01", None) for k, v in {**base, **cands}.items()}
    t = pd.DataFrame({k: ss.metrics(v) for k, v in e3.items()}).T
    best = t.loc[list(ss.BASE), "cagr"].idxmax()
    t["vs_best"] = t["cagr"] - t.loc[best, "cagr"]
    t["t_best"] = [((v["ret"] - e3[best]["ret"]).mean() / ((v["ret"] - e3[best]["ret"]).std(ddof=1)
                    / np.sqrt(len(v)))) if k != best else np.nan for k, v in e3.items()]
    names = {k: f"{int(v['names'].min())}-{int(v['names'].max())}" for k, v in e3.items()}
    t["names"] = pd.Series(names)
    print(f"Dow 30, point-in-time members · E3 2020-01 → 2026-09 · net of costs · best existing: {best}")
    print(t[["months", "cagr", "vol", "ret_vol", "maxdd", "turnover", "vs_best", "t_best", "names"]].round(2).to_string())


if __name__ == "__main__":
    main()
