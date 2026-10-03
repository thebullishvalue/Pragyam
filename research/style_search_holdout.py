"""
research/style_search_holdout.py — the style search's one holdout run, on the finalists only.

Reads each family's finalists (research/candidates/*.py, as each agent declared them), runs them
unchanged through research/style_search.py on the FULL panels — Nifty 50 and Dow 30, Feb 2007 →
Sep 2026, and the 27-fund ETF window — and applies the bar fixed in style_search.py:

    BEATS ALL   net CAGR above the best of the eight existing styles in all six stock cells
                (E1, E2, E3 × Nifty 50, Dow 30). The ETF window is reported, not ruled on.
    HAIRCUT     each finalist's E3 gap to the best existing style, as a one-sided t, Bonferroni-
                adjusted for every configuration the search ran (TRIALS below).

Run once:     python research/style_search_holdout.py
Attribution:  python research/style_search_holdout.py --attribution     (~2 min; reads the same full
              panels and re-runs only managed_mom λ=1 and λ=2 against CVG over E3 — no new candidate,
              no new holdout read; it reproduces the per-name figures quoted below)

RESULT (2026-10-03) — one style clears the bar on the panels as built; it does not survive a
point-in-time Dow. Net CAGR %, margin over the best existing style in each cell:

                                  Nifty 50                       Dow 30                     ETF (27)
                          E1      E2      E3            E1      E2      E3                 Mar 25 →
  best existing        20.21H  19.68C  22.59C        14.39EW 18.50C  15.06C                17.57EW
  B managed_mom λ=2    +1.82   +2.68   +2.77         +0.46   +0.63   +0.23   6/6          +2.51
  B managed_mom λ=1    +1.17   +1.77   +1.70         +0.22   +0.41   +0.39   6/6          +1.96
  A capit_rev_cvg λ=2  +0.72   +0.62   −0.24         +0.52   +0.76   +0.10   5/6          −0.91
  A capit_rev_cvg λ=1  +0.29   +0.49   −0.19         +0.36   +0.53   −0.07   4/6          −0.85
  C kelly_egr λ=2      −1.44   +6.44   +5.03         −1.74   +4.59   −1.25   3/6          +5.96
  D ivol_tilt λ=0.5    −1.63   −0.31   −0.98         −0.76   −1.02   −1.33   0/6          −2.18
  E V_REGIME W=63      −0.41   −0.45   −0.60         −0.85   −0.66   −1.26   0/6          −0.81

  · managed_mom — CVG plus a 12-1 momentum overlay switched off when the equal-weighted market's
    24-month return is negative (Daniel & Moskowitz 2016) — beats all eight in all six cells at
    both λ, and EW on the ETF window. Full span: Nifty 23.26 / 22.39% vs CVG 20.48, Dow 16.29 /
    16.20% vs 15.78, at CVG's volatility and ~1.4x its turnover.
  · Not significant: E3 t over the best existing style is +1.21 / +1.15 (Nifty) and +0.07 / +0.25
    (Dow); Bonferroni over 43 configurations leaves every adjusted p at 1.0.
  · The E3 edge sits in a few names, led by late index entrants (--attribution: summed monthly
    active contribution vs CVG, gross of costs, every name held, rebalances 2020-01 → 2026-09).
    λ=1: Nifty +10.3 pts — top BSE +5.1, TRENT +3.7, BEL +2.4, ADANIENT +1.9, JSWSTEEL +1.5, M&M
    +1.4; bottom BAJAJFINSV −2.2, SBILIFE −1.9, BAJFINANCE −1.5, SBIN −1.5; the top five carry
    +14.7 pts, 143% of the total. Dow +2.0 pts — top NVDA +5.8, NKE +2.0, WMT +1.3, AMZN +1.0, CAT
    +0.8, CRM +0.8; bottom DOW −2.4, DIS −2.1, HON −1.3, KO −1.0; the top five carry +10.9 pts, 533%
    of the total, so the Dow's net edge is a small residual of larger bets. λ=2: Nifty +16.8, Dow
    +1.1 pts. The panels are today's constituents, so momentum held the late entrants (BSE, TRENT,
    BEL, ADANIENT; NVDA, AMZN, CRM) through runs that preceded their joining the index — an
    investor in the index could not. (Before --attribution was committed, this note quoted the
    Dow's late entrants only and left out NKE and WMT, its second and third names.)
  · research/style_search_pit.py rebuilds the Dow on point-in-time membership for E3: managed_mom
    then trails CVG (λ=1 −0.06 %/yr, t −0.15; λ=2 −0.63) — a tie at best. No point-in-time Nifty
    panel exists here (needs NSE's constituent history), so the Nifty E3 margin is unverified and
    rests on the same kind of names.
  · The bear gate fired twice in 20 years (Nifty 2008-11 → 2009-05 and 2020-04 → 06; Dow 2008-11 →
    2009-11): the E1 margin is one avoided momentum crash. At λ=2 the additive overlay zeroes up to
    6 Nifty names (10 on the point-in-time Dow) — a shipped version would need a multiplicative
    tilt to keep the position-count contract.
  · Verdict: no style found here reliably beats all eight. managed_mom is the one worth a
    point-in-time Nifty test before any product decision; CVG stays the best existing style.

AFTERWARDS (v12.1.0) — the point-in-time Nifty test recommended above was NOT run: no NSE constituent
history exists here. managed_mom shipped as nco "MMOM" anyway, with three decisions made after this
holdout was seen — λ = 1 over the discovery-ranked λ = 2 (on the point-in-time Dow and the position
count), the ¼-of-CVG floor, and the month-to-date volatility reading. research/mmom_ship.py
re-measures the shipped code; every figure there is an every-name book, and top-N books were never
measured for this style.
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

# family file, candidate name, fixed parameters, factory? — from the agents' reports (2026-10-03).
# Families A and B each had two configurations clear all four discovery cells; C, D and E had none
# and name their best by the fallback rule (they cannot pass: the bar includes E1 and E2).
FINALISTS: list = [
    ("a_reversal.py", "A3_capit_rev_cvg", {"lam": 2.0}, False),
    ("a_reversal.py", "A3_capit_rev_cvg", {"lam": 1.0}, False),
    ("b_momentum.py", "managed_mom", {"lam": 2.0}, False),
    ("b_momentum.py", "managed_mom", {"lam": 1.0}, False),
    ("c_weighting.py", "kelly_egr", {"lam": 2.0}, False),
    ("d_anomaly.py", "ivol_tilt", {"lam": 0.5}, False),
    ("e_ensemble.py", "V_REGIME", {"W": 63}, False),
]
TRIALS = 43            # A 10 (incl. the harness demo) · B 9 · C 9 · D 6 · E 9 configurations


def _load(path: str):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _label(path: str, name: str, params: dict) -> str:
    return f"{path.split('_')[0].upper()}:{name}(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")"


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
            cands[_label(path, name, params)] = ss.run(build_fn(path, name, params, factory, d), d)
        out = ss.report(cands, d, base, show_base=True)
        out["universe"] = d["name"]
        rows.append(out)
        full = {k: ss.metrics(v) for k, v in {**base, **cands}.items()}
        print(pd.DataFrame(full).T[["months", "cagr", "vol", "ret_vol", "maxdd", "turnover"]].round(2).to_string(), flush=True)
    res = pd.concat(rows)
    print("\n══ THE BAR · six stock cells (E1, E2, E3 × Nifty 50, Dow 30) ═══════════════════════", flush=True)
    labels = [_label(p, n, q) for p, n, q, _ in FINALISTS]
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


# ── --attribution · which names the E3 edge over CVG came from ──────────────────────────────────────
ATTR_START = pd.Timestamp("2020-01-01")         # E3
ATTR_PANELS = ("nifty_50", "dow_30")
ATTR_LAMS = (1.0, 2.0)


def _held(w, priced: pd.Index) -> pd.Series:
    """What ss.run holds for raw weights `w`: priced names only, ≥ 0, sorted stably, sb.cut(·, None)."""
    w = pd.Series(w, dtype=float)
    w = w[w.index.isin(priced)].clip(lower=0.0).fillna(0.0)
    return ss.sb.cut(w.sort_values(ascending=False, kind="stable"), None)


def attribution_panel(d: dict, lam: float, start: pd.Timestamp = ATTR_START) -> pd.Series:
    """Summed monthly active contribution of the TESTED managed_mom(λ) over CVG, per name, from `start`.

    For every rebalance month a ≥ start with next month b, on the full panel (ss.load(u, holdout=True)):
        w   = the held managed_mom(ctx, lam) book    (b_momentum.py, as the holdout ran it)
        wc  = the held CVG book                       (ctx.raw["CVG"] over ctx.priced)
        r   = px[b] / px[a] − 1, NaN → 0             (a name gone is cash, as in ss.run)
        active_i += (w_i − wc_i) · r_i                over the union of the two books' names
    Gross, no costs; every name held (n = None), as in the holdout. Returns fractions, per name.
    """
    fn = build_fn("b_momentum.py", "managed_mom", {"lam": lam}, False, d)
    px, months = d["px"], d["months"]
    act = pd.Series(dtype=float)
    for a, b in zip(months[:-1], months[1:]):
        if a < start:
            continue
        ctx = ss.Ctx(d, a)
        w = _held(fn(ctx), ctx.priced)
        wc = _held(ctx.raw["CVG"].reindex(ctx.priced).fillna(0.0), ctx.priced)
        idx = w.index.union(wc.index)
        r = (px.loc[b].reindex(idx) / px.loc[a].reindex(idx) - 1.0).fillna(0.0)
        act = act.add((w.reindex(idx, fill_value=0.0) - wc.reindex(idx, fill_value=0.0)) * r, fill_value=0.0)
    return act


def attribution(panels=ATTR_PANELS, lams=ATTR_LAMS) -> dict:
    """Print, per panel and λ, the E3 active total over CVG in percentage points, the top 6 and bottom 4
    names, and the share of the total the top 5 carry. The source of the 'late index entrants' figures
    quoted in the RESULT above, the README and the CHANGELOG."""
    out = {}
    print(f"\n══ ATTRIBUTION · tested managed_mom vs CVG · E3 (rebalances from {ATTR_START:%Y-%m}) · summed monthly "
          "active contribution, gross of costs, every name held ═══", flush=True)
    for u in panels:
        d = ss.load(u, holdout=True)
        e3 = [a for a in d["months"][:-1] if a >= ATTR_START]
        for lam in lams:
            act = attribution_panel(d, lam) * 100.0                     # percentage points
            s = act.sort_values(ascending=False, kind="stable")
            tot = float(s.sum())
            top5 = float(s.head(5).sum())
            share = f"{top5 / tot:.0%}" if tot else "n/a"
            fmt = lambda x: ", ".join(f"{k} {v:+.1f}" for k, v in x.items())  # noqa: E731
            print(f"{d['name']:9s} λ={lam:g} · {len(e3)} months {e3[0]:%Y-%m} → {e3[-1]:%Y-%m} · total {tot:+.1f} pts"
                  f" · top 5 = {top5:+.1f} pts ({share} of the total)", flush=True)
            print(f"    top 6     {fmt(s.head(6))}", flush=True)
            print(f"    bottom 4  {fmt(s.tail(4).iloc[::-1])}", flush=True)
            out[(u, lam)] = s
    return out


if __name__ == "__main__":
    if "--attribution" in sys.argv[1:]:
        attribution()
    else:
        main()
