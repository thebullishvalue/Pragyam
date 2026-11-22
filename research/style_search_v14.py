"""
research/style_search_v14.py — the v14 style search: ten new styles against the three shipped ones.

THE BRIEF (owner, 2026-11-22)
    Develop ten portfolio styles: five from pragyam.pine (thebullishvalue/Indicator) and VAAYDO
    (research/candidates/f_pragyam.py), five from mathematics first (research/candidates/
    g_mathematics.py). Test them all; a style enters the system only if it beats the existing styles.

PROTOCOL (fixed and committed before any candidate was run; the same harness as style_search.py)
────────────────────────────────────────────────────────────────────────────────────────────────
    EXISTING    the shipped styles: Equal Weight, ERC, HRP (nco.METHOD_ORDER), raw weights exactly as
                nco.compute_nco_portfolio builds them, through style_blends / style_search unchanged.
    THE BOOK    style_search.run: monthly, first trading day, every name held (n = None), the 10% cap,
                net of 10bp (India) / 3bp (US) per unit of one-way turnover on the drifted book.
    DISCOVERY   E1 (< 2014) and E2 (2014-2019) on Nifty 50 and Dow 30. The discovery panels stop at the
                first trading day of 2020, so no candidate can read the holdout. All 22 configurations
                run here; nothing is added after the first result is seen.
    FINALISTS   one per candidate function (ten): the configuration with the largest smallest margin over
                the best existing style across the four discovery cells; ties go to the declared default.
                A function's finalist goes to the holdout whether or not it beat anything in discovery.
    HOLDOUT     E3 (≥ 2020) on Nifty 50 and Dow 30, run ONCE on the ten finalists after discovery is
                recorded; with it the point-in-time Dow (style_search_pit: each day's actual members) for
                E3, the 27-fund ETF window, and a top-30 book on Nifty 50 / Dow 30 (the app's default).

THE BAR — a style is ACCEPTED only if all three hold:
    1. CELLS        net CAGR above the best of EW / ERC / HRP in every one of the six cells (E1, E2, E3 ×
                    Nifty 50, Dow 30). The best differs by cell.
    2. SURVIVORSHIP net CAGR on the point-in-time Dow, E3, at or above the best of EW / ERC / HRP there.
                    The stock panels are today's constituents; this is the one test free of that bias.
    3. SIGNIFICANCE the full-span (2007-02 → 2026-09) monthly net return over the best existing style of
                    each universe (the one with the highest full-span CAGR there), averaged across Nifty 50
                    and Dow 30 month by month: one-sided t ≥ 2.576 — 5% family-wise over the ten finalists
                    (Bonferroni).
    A style that passes 1 and 2 but not 3 is reported as UNPROVEN and is not shipped; the owner decides.
    The ETF window (19 months) and the top-30 books are reported, not ruled on.

RESULT — DISCOVERY (run 2026-11-22 16:05; all 22 declared configurations, none added; committed before
the holdout was run). Net CAGR margin over the best existing style in each cell, %/yr. Best existing:
Nifty E1 HRP 20.39, Nifty E2 EW 18.99, Dow E1 EW 14.39, Dow E2 EW 18.32. (The rebuilt panels reproduce the
recorded baselines: Nifty 50 full span EW 20.10%, ERC −0.49, volatility 22.2 / 20.1 / 18.8.)
                                  Nifty E1  Nifty E2  Dow E1  Dow E2   smallest   cells
    F forecast_risk_parity p=0.5     -2.21     -0.42   -0.49   -0.77      -2.21     0/4  ← finalist
    F forecast_risk_parity p=1       -2.76     -0.78   -0.93   -1.50      -2.76     0/4
    F forecast_risk_parity p=2       -3.90     -1.26   -1.90   -2.81      -3.90     0/4
    F forecast_erc window=126        -1.67     -0.82   -0.94   -1.24      -1.67     0/4
    F forecast_erc window=252        -1.35     -0.65   -0.98   -1.25      -1.35     0/4  ← finalist
    F exposure_dial floor=0.25       -2.32     +0.28   -0.41   -0.07      -2.32     1/4
    F exposure_dial floor=0.5        -2.11     +0.20   -0.25   -0.06      -2.11     1/4
    F exposure_dial floor=0.75       -2.11     +0.09   -0.17   -0.05      -2.11     1/4  ← finalist
    F trend_tape floor=0.25          -1.86     +0.30   -0.34   -0.02      -1.86     1/4
    F trend_tape floor=0.5           -1.63     +0.24   -0.15   -0.01      -1.63     1/4
    F trend_tape floor=0.75          -1.63     +0.13   -0.05   -0.00      -1.63     1/4  ← finalist
    F breadth_switch thr=0.4         -2.30     -0.29   -0.33   -0.24      -2.30     0/4
    F breadth_switch thr=0.5         -1.97     -0.38   -0.07   -0.29      -1.97     0/4  ← finalist
    F breadth_switch thr=0.6         -2.03     -0.80   +0.31   -0.76      -2.03     1/4
    G rmt_erc window=252             -0.56     -0.79   -0.91   -1.45      -1.45     0/4
    G rmt_erc window=504             -0.95     -0.38   -0.89   -1.37      -1.37     0/4  ← finalist
    G max_enb                        -0.14     -0.30   -1.05   -1.15      -1.15     0/4  ← finalist
    G network_periphery lam=0.25     -0.13     -0.06   +0.13   +0.65      -0.13     2/4  ← finalist
    G network_periphery lam=0.5      +1.37     -0.28   +0.14   +1.48      -0.28     3/4
    G exp_gradient eta=0.05          -1.78     +0.20   -0.01   +0.14      -1.78     2/4  ← finalist
    G exp_gradient eta=0.3           -2.11     +1.46   -0.04   +0.92      -2.11     2/4
    G spectral_parity                +0.29     +0.58   +0.42   -0.54      -0.54     3/4  ← finalist
NO configuration beats the best existing style in all four discovery cells, so none can meet bar 1.
Family F loses Nifty E1 (the 2008 crash era, HRP's cell) in every configuration, by 1.35 to 3.90 %/yr:
the risk-forecast books are not defensive enough to match HRP there, and the dial and the trend tape
lean toward recent strength, which that era punished. Closest: network_periphery (3/4 at λ = 0.5,
2/4 at 0.25) and spectral_parity (3/4). As declared, all ten finalists still go to the holdout once.

Run:   python research/style_search_v14.py discovery        (writes research/v14_discovery.pkl)
       python research/style_search_v14.py holdout          (once; writes research/v14_holdout.pkl)
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "candidates"))
sys.path.insert(0, os.path.dirname(HERE))

import style_search as ss                         # noqa: E402
import f_pragyam as F                             # noqa: E402
import g_mathematics as G                         # noqa: E402

EXISTING = ("EW", "ERC", "HRP")
DISC_ERAS = (("E1", None, "2014-01-01"), ("E2", "2014-01-01", "2020-01-01"))
ALL_ERAS = DISC_ERAS + (("E3", "2020-01-01", None),)
STOCKS = ("nifty_50", "dow_30")
T_BAR = 2.576
DISC_PKL = os.path.join(HERE, "v14_discovery.pkl")
HOLD_PKL = os.path.join(HERE, "v14_holdout.pkl")
FAMILIES = (("F", F), ("G", G))
_DATA: dict = {}


def configs():
    """(family, function name, params) for every declared configuration, in declaration order."""
    for fam, mod in FAMILIES:
        for name, (_fn, grid) in mod.CANDIDATES.items():
            keys = list(grid)
            for vals in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
                yield fam, name, dict(zip(keys, vals))


def label(name: str, params: dict) -> str:
    return name + ("(" + ", ".join(f"{k}={v:g}" for k, v in params.items()) + ")" if params else "")


def fn_of(fam: str, name: str, params: dict):
    mod = dict(FAMILIES)[fam]
    f = mod.CANDIDATES[name][0]
    return lambda ctx: f(ctx, **params)


def cagr(r: pd.DataFrame) -> float:
    return ss.metrics(r)["cagr"] if len(r) else float("nan")


def _job(args):
    """One configuration on one universe (a worker reads the data loaded before the fork)."""
    fam, name, params, u, n = args
    t0 = time.time()
    r = ss.run(fn_of(fam, name, params), _DATA[u], n)
    return (fam, name, tuple(sorted(params.items())), u, n), r, time.time() - t0


def _existing(d: dict, n=None) -> dict:
    return {k: ss.run(lambda c, k=k: c.raw[k], d, n) for k in EXISTING}


def era_table(runs: dict, base: dict, eras) -> pd.DataFrame:
    rows = []
    for lab, r in runs.items():
        for era, a, b in eras:
            x = ss._era(r, a, b)
            if not len(x):
                continue
            best = max(EXISTING, key=lambda k: cagr(ss._era(base[k], a, b)))
            bm = ss._era(base[best], a, b)
            m = ss.metrics(x)
            dd = x["ret"] - bm["ret"].reindex(x.index)
            rows.append(dict(style=lab, era=era, cagr=m["cagr"], vol=m["vol"], maxdd=m["maxdd"],
                             turnover=m["turnover"], best=best, best_cagr=cagr(bm),
                             margin=m["cagr"] - cagr(bm),
                             t=float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if dd.std() > 0 else np.nan))
    return pd.DataFrame(rows)


def discovery() -> None:
    for u in STOCKS:
        _DATA[u] = ss.load(u, holdout=False)
    base = {u: _existing(_DATA[u]) for u in STOCKS}
    jobs = [(fam, name, params, u, None) for fam, name, params in configs() for u in STOCKS]
    print(f"discovery · {len(jobs) // 2} configurations × {len(STOCKS)} universes · "
          f"{datetime.now():%Y-%m-%d %H:%M}", flush=True)
    runs = {}
    with Pool(4) as pool:
        for key, r, secs in pool.imap_unordered(_job, jobs):
            runs[key] = r
            print(f"  {key[3]:9s} {label(key[1], dict(key[2])):36s} {secs:6.0f}s", flush=True)
    tabs = {}
    for u in STOCKS:
        lab_runs = {(fam, name, params): runs[(fam, name, params, u, None)]
                    for (fam, name, params, uu, n) in runs if uu == u}
        tabs[u] = era_table({k: v for k, v in lab_runs.items()}, base[u], DISC_ERAS)
    pickle.dump(dict(runs=runs, base=base, tabs=tabs, at=datetime.now().isoformat(timespec="seconds")),
                open(DISC_PKL, "wb"))
    report_discovery()


def margins_frame(disc: dict) -> pd.DataFrame:
    rows = []
    for fam, name, params in configs():
        key = (fam, name, tuple(sorted(params.items())))
        row = dict(family=fam, function=name, config=label(name, params), key=key,
                   default=params == dict(FAMILIES)[fam].DEFAULTS[name])
        for u in STOCKS:
            t = disc["tabs"][u]
            t = t[t["style"].apply(lambda s: s == key)]
            for _, x in t.iterrows():
                row[f"{u}:{x['era']}"] = x["margin"]
                row[f"{u}:{x['era']}:cagr"] = x["cagr"]
        cells = [c for c in row if c.count(":") == 1]
        row["min_margin"] = min(row[c] for c in cells)
        row["cells_beaten"] = sum(row[c] > 0 for c in cells)
        rows.append(row)
    return pd.DataFrame(rows)


def finalists(disc: dict) -> pd.DataFrame:
    m = margins_frame(disc)
    m = m.sort_values(["function", "min_margin", "default"], ascending=[True, False, False], kind="stable")
    return m.groupby("function", sort=False).head(1)


def report_discovery() -> None:
    disc = pickle.load(open(DISC_PKL, "rb"))
    pd.set_option("display.width", 250)
    print(f"\n══ DISCOVERY (run {disc['at']}) · net CAGR margin over the best existing style per cell, %/yr", flush=True)
    for u in STOCKS:
        b = disc["base"][u]
        line = " · ".join(f"{era} best {max(EXISTING, key=lambda k: cagr(ss._era(b[k], a, c)))} "
                          f"{max(cagr(ss._era(b[k], a, c)) for k in EXISTING):.2f}" for era, a, c in DISC_ERAS)
        print(f"   {u}: {line}")
    m = margins_frame(disc)
    cols = ["family", "config", "nifty_50:E1", "nifty_50:E2", "dow_30:E1", "dow_30:E2", "min_margin", "cells_beaten"]
    print(m[cols].round(2).to_string(index=False))
    f = finalists(disc)
    print("\nFINALISTS (largest smallest margin per function):")
    print(f[["family", "config", "min_margin", "cells_beaten"]].round(2).to_string(index=False))


def holdout() -> None:
    if os.path.exists(HOLD_PKL):
        raise SystemExit("the holdout has been run once already — see research/v14_holdout.pkl")
    disc = pickle.load(open(DISC_PKL, "rb"))
    fin = finalists(disc)
    import style_search_pit as P
    for u in STOCKS + ("etf_27",):
        _DATA[u] = ss.load(u, holdout=True)
    _DATA["dow_pit"] = pickle.load(open(P.PKL, "rb")) if os.path.exists(P.PKL) else P.build()
    picks = [(r["family"], r["function"], dict(r["key"][2])) for _, r in fin.iterrows()]
    jobs = [(fam, name, params, u, None) for fam, name, params in picks for u in STOCKS + ("etf_27",)]
    jobs += [(fam, name, params, u, 30) for fam, name, params in picks for u in STOCKS]
    print(f"holdout · {len(picks)} finalists · {datetime.now():%Y-%m-%d %H:%M}", flush=True)
    runs = {}
    with Pool(4) as pool:
        for key, r, secs in pool.imap_unordered(_job, jobs):
            runs[key] = r
            print(f"  {key[3]:9s} n={str(key[4]):4s} {label(key[1], dict(key[2])):36s} {secs:6.0f}s", flush=True)
    # The point-in-time Dow: the allocation universe is that day's members (style_search_pit.main).
    init = ss.Ctx.__init__

    def pit_init(self, data, a):
        init(self, data, a)
        self.priced = pd.Index([s for s in self.priced if P.member(s, a)])
    ss.Ctx.__init__ = pit_init
    try:
        base_pit = {k: ss.run(lambda c, k=k: c.raw[k], _DATA["dow_pit"]) for k in EXISTING}
        for fam, name, params in picks:
            runs[(fam, name, tuple(sorted(params.items())), "dow_pit", None)] = ss.run(
                fn_of(fam, name, params), _DATA["dow_pit"])
    finally:
        ss.Ctx.__init__ = init
    base = {u: _existing(_DATA[u]) for u in STOCKS + ("etf_27",)}
    base30 = {u: _existing(_DATA[u], 30) for u in STOCKS}
    pickle.dump(dict(runs=runs, base=base, base30=base30, base_pit=base_pit, picks=picks,
                     at=datetime.now().isoformat(timespec="seconds")), open(HOLD_PKL, "wb"))
    report_holdout()


def report_holdout() -> None:
    H = pickle.load(open(HOLD_PKL, "rb"))
    runs, base = H["runs"], H["base"]
    pd.set_option("display.width", 250)
    rows = []
    for fam, name, params in H["picks"]:
        k = (fam, name, tuple(sorted(params.items())))
        row = dict(family=fam, style=label(name, params))
        ok_cells = True
        for u in STOCKS:
            t = era_table({"x": runs[k + (u, None)]}, base[u], ALL_ERAS)
            for _, x in t.iterrows():
                row[f"{u[:4]}:{x['era']}"] = x["margin"]
                ok_cells &= x["margin"] > 0
        e3 = lambda r: ss._era(r, "2020-01-01", None)      # noqa: E731
        pit_best = max(cagr(e3(v)) for v in H["base_pit"].values())
        row["pit_E3"] = cagr(e3(runs[k + ("dow_pit", None)])) - pit_best
        ex = []
        for u in STOCKS:
            bk = max(EXISTING, key=lambda q: cagr(base[u][q]))
            ex.append((runs[k + (u, None)]["ret"] - base[u][bk]["ret"]).rename(u))
        X = pd.concat(ex, axis=1).dropna().mean(axis=1)
        row["t_full"] = float(X.mean() / (X.std(ddof=1) / np.sqrt(len(X))))
        row["excess_pa"] = float(X.mean() * 12 * 100)
        etf = runs[k + ("etf_27", None)]
        row["etf_vs_best"] = cagr(etf) - max(cagr(v) for v in base["etf_27"].values())
        for u in STOCKS:
            r30 = runs[k + (u, 30)]
            row[f"top30_{u[:4]}"] = cagr(r30) - max(cagr(v) for v in H["base30"][u].values())
        row["CELLS"] = ok_cells
        row["SURV"] = row["pit_E3"] >= 0
        row["SIG"] = row["t_full"] >= T_BAR
        row["VERDICT"] = ("ACCEPTED" if ok_cells and row["SURV"] and row["SIG"] else
                          "UNPROVEN" if ok_cells and row["SURV"] else "REJECTED")
        rows.append(row)
    out = pd.DataFrame(rows)
    print(f"\n══ HOLDOUT (run {H['at']}) · margins over the best existing style, %/yr net", flush=True)
    for u in STOCKS:
        line = " · ".join(f"{era} {max(EXISTING, key=lambda q: cagr(ss._era(base[u][q], a, b)))} "
                          f"{max(cagr(ss._era(base[u][q], a, b)) for q in EXISTING):.2f}" for era, a, b in ALL_ERAS)
        print(f"   {u}: {line}")
    print(out.round(2).to_string(index=False))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"discovery": discovery, "holdout": holdout,
     "report": lambda: (report_discovery(), os.path.exists(HOLD_PKL) and report_holdout())}[cmd]()
