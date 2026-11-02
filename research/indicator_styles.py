"""
research/indicator_styles.py — do Momentum Masters and Volatility Surfer beat the shipped styles?

THE QUESTION
────────────
Two strategies from the strategy library (strategies.py in this repo, a line-for-line port of
VolatilitySurfer and MomentumMasters — checked equal to the source classes to 5.6e-17 over 300
random snapshots) score each day's indicator snapshot and weight every name by its score. Are
they better than what Pragyam ships?

PRE-REGISTRATION (written before either strategy was run on these panels)
───────────────────────────────────────────────────────────────────────────
    CANDIDATES  MMAST (Momentum Masters) and VSURF (Volatility Surfer), exactly as the source
                defines them: its scoring, its data cleaning and its [1%, 10%] weight band. No
                parameter is tuned here; two configurations, two trials.
    PATH        the product code: nco.compute_nco_portfolio(method="MMAST" | "VSURF") on the
                rebalance day's snapshot (style_blends.raw), then the harness's book —
                research/style_search.run: top-N by weight, the 10% cap, monthly from the first
                trading day, net of 10bp (India) / 3bp (US) per unit of one-way turnover.
    PANELS      the v12.2 research panels (style_search.load(u, holdout=True)): Nifty 50 and Dow
                30, Feb 2007 → Sep 2026, today's constituents; the 27-fund ETF window (Mar 2025
                →); the point-in-time Dow, 2020+ (style_search_pit; each day's members only).
                These strategies were not built on these panels, so there is no discovery split
                to protect: the whole span is out of sample for this repo (their author may
                have tuned them on similar data — not known).
    EXISTING    the five shipped styles: Equal Weight, ERC, HRP, CVG and Managed Momentum
                (MMOM as shipped: mmom_ship.weights with the app's close history, --app-history).
    THE BAR     BEATS THE EXISTING STYLES = net CAGR above the best of the five in EVERY one of
                the six stock cells — E1 2007-13, E2 2014-19, E3 2020+ × Nifty 50, Dow 30 —
                every name held (the books every shipped style's evidence is measured on).
    REPORTED    (not ruled) each candidate against each style per cell, the paired monthly t of
                the margin over the best, full-span CAGR / volatility / ret-vol / max drawdown /
                turnover, the ETF window, the point-in-time Dow E3, and cut books (Nifty 50 at
                30 positions, the app's default; Dow 30 at 15).
    THEN        both styles are added to the app whatever the verdict, with this evidence stated
                in their registry entries (as for ERC and HRP, which do not beat Equal Weight).

Run:  python research/indicator_styles.py        (~10 min; reads the cached panels and snapshots)
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

import mmom_ship as M                             # noqa: E402
import style_blends as sb                         # noqa: E402
import style_search as ss                         # noqa: E402
import style_search_pit as P                      # noqa: E402

CANDS = {"MMAST": "Momentum Masters", "VSURF": "Volatility Surfer"}
SHIPPED = ("EW", "ERC", "HRP", "CVG", "MMOM")
CUT = {"nifty_50": 30, "dow_30": 15}


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan


def snaps_of(u: str) -> list:
    if u == "dow_pit":
        return sb.repair(pickle.load(open(os.path.join(HERE, "cvg_reweight_dow_pit.pkl"), "rb")))
    s = sb.snapshots("etf_book" if u == "etf_27" else u)
    if u == "etf_27":
        s = [(t, x[~x["symbol"].isin(ss.ETF_YOUNG)].reset_index(drop=True)) for t, x in s]
    return s


def cand_weights(d: dict, snaps: list, method: str, members=None) -> dict:
    """The candidate's raw weights on every rebalance day, through nco.compute_nco_portfolio.
    The strategies read only the day's snapshot, so that is all `hist` needs to carry."""
    pos = {pd.Timestamp(t): i for i, (t, _) in enumerate(snaps)}
    out = {}
    for a in d["months"][:-1]:
        snap = snaps[pos[a]][1]
        if members is not None:
            snap = snap[snap["symbol"].map(lambda x: members(x, a))].reset_index(drop=True)
        out[a] = sb.raw([(a, snap)], method)
    return out


def mmom_weights(u: str, d: dict, snaps: list, members=None) -> dict:
    """Managed Momentum as shipped (mmom_ship.weights, the app's close history)."""
    if u == "dow_pit":
        from universe import DOW_JONES_TICKERS
        hist = M.app_history("dow_pit", list(DOW_JONES_TICKERS) + list(P.ADDED))
    else:
        key = "etf_book" if u == "etf_27" else u
        hist = M.app_history(u, [s for s in sb.symbols(key) if s.replace(".NS", "") not in ss.ETF_YOUNG])
    return M.weights(d, snaps, None, members=members, history=hist)["ship"]


def books(u: str) -> dict:
    """Monthly net returns of the shipped styles, the blends and the candidates, every name held
    (and the cut book where defined)."""
    t0 = time.time()
    pit = u == "dow_pit"
    d = pickle.load(open(P.PKL, "rb")) if pit else ss.load(u, holdout=True)
    snaps = snaps_of(u)
    members = P.member if pit else None
    W = {m: cand_weights(d, snaps, m, members) for m in CANDS}
    W["MMOM"] = mmom_weights(u, d, snaps, members)
    init = ss.Ctx.__init__
    if pit:
        def pit_init(self, data, a):
            init(self, data, a)
            self.priced = pd.Index([s for s in self.priced if P.member(s, a)])
        ss.Ctx.__init__ = pit_init
    try:
        out = {}
        for n in ((None, CUT[u]) if u in CUT else (None,)):
            r = ss.baselines(d, n)
            for k, w in W.items():
                r[k] = ss.run(lambda c, w=w: w[c.date], d, n)
            out[n] = r
    finally:
        ss.Ctx.__init__ = init
    print(f"   [{u} built in {time.time() - t0:.0f}s]", flush=True)
    return out


def cells(r: dict, eras) -> pd.DataFrame:
    rows = []
    for e, a, b in eras:
        cag = {k: ss.metrics(ss._era(v, a, b))["cagr"] for k, v in r.items()}
        best = max(SHIPPED, key=lambda k: cag[k])
        for c in CANDS:
            gap = ss._era(r[c], a, b)["ret"] - ss._era(r[best], a, b)["ret"]
            rows.append(dict(cand=c, era=e, cagr=cag[c], best=best, best_cagr=cag[best],
                             margin=cag[c] - cag[best], t=_t(gap),
                             **{f"vs_{k}": cag[c] - cag[k] for k in SHIPPED}))
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    verdict = {}
    for u in ("nifty_50", "dow_30", "etf_27", "dow_pit"):
        R = books(u)
        eras = (list(ss.ERAS) if u in ("nifty_50", "dow_30")
                else [("E3", "2020-01-01", None)] if u == "dow_pit" else [("window", None, None)])
        for n, r in R.items():
            label = f"{u} · {'every name' if n is None else f'top-{n}'}"
            tab = pd.DataFrame({k: {e: ss.metrics(ss._era(v, a, b))["cagr"] for e, a, b in eras}
                                for k, v in r.items()}).T
            print(f"\n== {label} · net CAGR %/yr by era", flush=True)
            print(tab.loc[list(SHIPPED) + list(CANDS)].round(2).to_string(), flush=True)
            C = cells(r, eras)
            print("   margin over the best shipped style (paired t):", flush=True)
            for c in CANDS:
                x = C[C["cand"] == c]
                print(f"   {c:6s} " + " · ".join(f"{e} {m:+.2f} vs {b} (t {t:+.2f})" for e, m, b, t in
                                                   zip(x["era"], x["margin"], x["best"], x["t"])), flush=True)
            full = pd.DataFrame({k: ss.metrics(v) for k, v in r.items()}).T
            print("   full span:", flush=True)
            print("   " + full.loc[list(SHIPPED) + list(CANDS), ["cagr", "vol", "ret_vol", "maxdd", "turnover"]]
                  .round(2).to_string().replace("\n", "\n   "), flush=True)
            if n is None and u in ("nifty_50", "dow_30"):
                for c in CANDS:
                    verdict.setdefault(c, []).extend((C[C["cand"] == c]["margin"] > 0).tolist())
    print("\n══ THE BAR · beat the best of EW / ERC / HRP / CVG / MMOM in all six stock cells, every name held", flush=True)
    for c, wins in verdict.items():
        print(f"   {CANDS[c]} ({c}): {sum(wins)}/{len(wins)} cells → "
              f"{'BEATS THE EXISTING STYLES' if all(wins) and len(wins) == 6 else 'does not beat them'}", flush=True)


if __name__ == "__main__":
    main()
