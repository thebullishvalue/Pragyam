"""
research/style_search_holdout.py — the style search's one holdout run, on the finalists only.

Reads each family's finalists (research/candidates/*.py, as each agent declared them), runs them
unchanged through research/style_search.py on the FULL panels — Nifty 50 and Dow 30, Feb 2007 →
Sep 2026, and the 27-fund ETF window — and applies the bar fixed in style_search.py:

    BEATS ALL   net CAGR above the best of the eight existing styles in all six stock cells
                (E1, E2, E3 × Nifty 50, Dow 30). The ETF window is reported, not ruled on.
    HAIRCUT     each finalist's E3 gap to the best existing style, as a one-sided t, Bonferroni-
                adjusted for every configuration the search ran (TRIALS below).

Run once:  python research/style_search_holdout.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style_search as ss                         # noqa: E402

# family file, candidate name, fixed parameters, factory? — filled in from the agents' reports
FINALISTS: list = []
TRIALS = 0                                         # configurations run across the whole search


def _load(path: str):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_fn(path: str, name: str, params: dict, factory: bool, data: dict):
    fn = _load(os.path.join(HERE, "candidates", path)).CANDIDATES[name][0]
    if factory:
        return fn(data, **params)
    return lambda ctx: fn(ctx, **params)


def main() -> None:
    pd.set_option("display.width", 250)
    rows = []
    for u in ("nifty_50", "dow_30", "etf_27"):
        d = ss.load(u, holdout=True)
        base = ss.baselines(d)
        cands = {}
        for path, name, params, factory in FINALISTS:
            label = f"{path.split('_')[0].upper()}:{name}"
            cands[label] = ss.run(build_fn(path, name, params, factory, d), d)
        out = ss.report(cands, d, base, show_base=True)
        out["universe"] = d["name"]
        rows.append(out)
        full = {k: ss.metrics(v) for k, v in {**base, **cands}.items()}
        print(pd.DataFrame(full).T[["months", "cagr", "vol", "ret_vol", "maxdd", "turnover"]].round(2).to_string(), flush=True)
    res = pd.concat(rows)
    print("\n══ THE BAR · six stock cells (E1, E2, E3 × Nifty 50, Dow 30) ═══════════════════════", flush=True)
    labels = [f"{p.split('_')[0].upper()}:{n}" for p, n, _, _ in FINALISTS]
    for lab in labels:
        x = res[(res["style"] == lab) & res["universe"].isin(["Nifty 50", "Dow 30"]) & res["era"].isin(["E1", "E2", "E3"])]
        wins = int((x["vs_best"] > 0).sum())
        e3 = x[x["era"] == "E3"]
        p = [st.t.sf(t, df=m - 1) if np.isfinite(t) else np.nan for t, m in zip(e3["t_best"], e3["months"])]
        p_adj = [min(1.0, q * max(TRIALS, 1)) for q in p]
        etf = res[(res["style"] == lab) & (res["universe"] == "ETF book (27)")]
        print(f"{lab:32s} cells won {wins}/6 {'· BEATS ALL' if wins == 6 else ''}"
              f" | E3 vs best: " + ", ".join(f"{u} {v:+.2f} (t {t:+.2f}, p {q:.3f}, adj {qa:.2f})" for u, v, t, q, qa in
                                           zip(e3["universe"], e3["vs_best"], e3["t_best"], p, p_adj))
              + (f" | ETF vs best {etf['vs_best'].iloc[0]:+.2f}" if len(etf) else ""), flush=True)
    res.to_pickle(os.path.join(HERE, "style_search_holdout.pkl"))


if __name__ == "__main__":
    main()
