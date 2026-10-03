"""
research/style_blends.py — every shipped style head to head, and four blends of HRP, CVG and 1/N.

PRE-REGISTRATION (fixed before any number below was seen)
─────────────────────────────────────────────────────────
The four shipped styles, each exactly as nco.compute_nco_portfolio builds it:

    EW          Equal Weight                 1 / N
    ERC         Equal Risk Contribution      Ledoit-Wolf covariance, equal variance shares
    HRP         Risk Parity (HRP)            recursive bisection on cluster variance
    CVG         Conviction-Value Grid        the shipped units (Dislocated 4), graded, gated

and four blends. A blend's weight formula is the plain average of its members' raw weight
vectors over the union of their allocation universes (a name a member cannot hold — HRP has no
covariance estimate for it — takes 0 from that member); the result then goes through the
pipeline's last step as every style does: top-N by weight, renormalised, the 10% cap.

    HRP+CVG     ½ HRP + ½ CVG
    HRP+EW      ½ HRP + ½ EW
    CVG+EW      ½ CVG + ½ EW
    HRP+CVG+EW  ⅓ each

Method as research/cvg_mom.py: backdata snapshots → nco.compute_nco_portfolio (each style once
per rebalance, over every name, uncapped; every book below is cut from that by the pipeline's own
top-N + nco._apply_cap), 252-day covariance window, first trading day of each month held to the
next, a name gone = cash, net of 10bp (India) / 3bp (US) per unit of one-way turnover.
Universes: the ETF book, Nifty 50, Dow 30 (today's constituents — survivorship applies to every
style alike). Books: every name held; top-30 (the app's default; Nifty 50 only — it is every name
on the 30-name panels); top-15 on all three. A book is scored from the first month every style
can build it with at least 10 names, so the young ETF book is not read on one or two funds.
Eras: E1 < 2014, E2 2014-2019, E3 ≥ 2020, and the full span.

Reported per book and style: CAGR, volatility, return / volatility (rf 0), max drawdown (monthly
NAV), turnover, the paired monthly gap to EW (%/yr, t, share of months ahead) and the any-date hit
rate — the share of rolling 36-month windows in which the style's CAGR beats EW's. For a blend,
also the gap to the mean of its members (what blending added beyond averaging the books' returns).

DECISION RULE: a blend is BETTER than a style on return only if it beats that style, net of costs,
in ALL THREE eras on BOTH stock universes (every-name book); BETTER on risk-adjusted return under
the same rule on return / volatility. The ETF book (1-30 funds from 2012) and the top-N books are
reported, not ruled on. This is a measurement; no product change follows from it alone.

Run:  python research/style_blends.py [etf_book|nifty_50|dow_30]     (all three by default;
      snapshots cached to research/cvg_reweight_<universe>.pkl, shared with the other harnesses)
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
import nco                                        # noqa: E402

CAPITAL, CAP, MIN_NAMES, ROLL = 1e10, 0.10, 10, 36
START, END = datetime(2006, 1, 1), datetime(2026, 10, 2)
assert cvgrid.STATE_UNITS["DISLOCATED"] == 4.0, "the shipped CVG units are the control"

METHOD = {"EW": "EQUAL", "ERC": "ERC", "HRP": "HRP", "CVG": "CVG"}
BLENDS = {"HRP+CVG": ("HRP", "CVG"), "HRP+EW": ("HRP", "EW"), "CVG+EW": ("CVG", "EW"),
          "HRP+CVG+EW": ("HRP", "CVG", "EW")}
KEYS = list(METHOD) + list(BLENDS)
UNIVERSES = {"etf_book": ("ETF book", 10.0), "nifty_50": ("Nifty 50", 10.0), "dow_30": ("Dow 30", 3.0)}
SIZES = (("all", None), ("top30", 30), ("top15", 15))
ERAS = (("E1", None, "2014-01-01"), ("E2", "2014-01-01", "2020-01-01"), ("E3", "2020-01-01", None),
        ("FULL", None, None))


def symbols(key: str) -> list:
    from universe import DOW_JONES_TICKERS, ETF_UNIVERSE, get_index_stock_list
    if key == "etf_book":
        return list(ETF_UNIVERSE)
    if key == "nifty_50":
        return list(get_index_stock_list("NIFTY 50")[0] or [])
    return list(DOW_JONES_TICKERS)


def snapshots(key: str) -> list:
    p = os.path.join(HERE, f"cvg_reweight_{key}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    from backdata import generate_historical_data
    snaps = generate_historical_data(symbols(key), START, END)
    pickle.dump(snaps, open(p, "wb"))
    return snaps


def raw(hist, method: str) -> pd.Series:
    """A style's weights over its whole allocation universe, uncapped, in the order its book fills."""
    snap = hist[-1][1]
    prices = {s: float(p) for s, p in zip(snap["symbol"], pd.to_numeric(snap["price"], errors="coerce"))
              if np.isfinite(p) and p > 0}
    book = nco.compute_nco_portfolio(hist, prices, CAPITAL, len(prices), method=method, max_pos_pct=1.0)
    if book.empty:
        return pd.Series(dtype=float)
    w = pd.to_numeric(book["weightage_pct"], errors="coerce")
    return pd.Series((w / w.sum()).to_numpy(), index=book["symbol"].astype(str).to_numpy())


def blend(parts: list) -> pd.Series:
    idx = list(dict.fromkeys(s for p in parts for s in p.index))
    w = sum(p.reindex(idx, fill_value=0.0) for p in parts) / len(parts)
    return w.sort_values(ascending=False, kind="stable")


def cut(w: pd.Series, n) -> pd.Series:
    """The pipeline's last step, as in nco.compute_nco_portfolio: top-N by weight, renormalised, capped."""
    w = w[w > 1e-9]
    if n is not None:
        w = w.head(n)
    if w.empty:
        return w
    w = w / w.sum()
    return pd.Series(nco._apply_cap(w.to_numpy(dtype=float), CAP), index=w.index)


def backtest(snaps: list, sizes) -> dict:
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in snaps])
    px = {pd.Timestamp(d): pd.to_numeric(df.drop_duplicates("symbol", keep="last").set_index("symbol")["price"],
                                         errors="coerce") for d, df in snaps}
    months = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first().to_list()
    pos = {d: i for i, d in enumerate(dates)}
    rows, prev, started = {sz: [] for sz, _ in sizes}, {}, False
    for a, b in zip(months[:-1], months[1:]):
        hist = snaps[max(0, pos[a] - 252): pos[a] + 1]
        w_raw = {k: raw(hist, m) for k, m in METHOD.items()}
        if not started:
            if min(len(v) for v in w_raw.values()) < MIN_NAMES:
                continue
            started = True
        w_raw.update({k: blend([w_raw[m] for m in mem]) for k, mem in BLENDS.items()})
        ret = px[b] / px[a] - 1.0
        for sz, n in sizes:
            rec = {"date": a}
            for k in KEYS:
                w = cut(w_raw[k], n)
                r = ret.reindex(w.index).fillna(0.0)
                rec[k] = float((w * r).sum())
                if (sz, k) in prev:
                    w0, r0 = prev[(sz, k)]
                    d = w0 * (1.0 + r0)
                    d = d / d.sum()
                    idx = w.index.union(d.index)
                    rec[f"to::{k}"] = float(0.5 * (w.reindex(idx, fill_value=0) - d.reindex(idx, fill_value=0)).abs().sum())
                prev[(sz, k)] = (w, r)
                rec[f"n::{k}"] = len(w)
            rows[sz].append(rec)
    return {sz: pd.DataFrame(v).set_index("date") for sz, v in rows.items()}


def net(bt: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    out = pd.DataFrame(index=bt.index)
    for k in KEYS:
        out[k] = bt[k] - bt.get(f"to::{k}", pd.Series(0.0, index=bt.index)).fillna(0.0) * cost_bps / 1e4
        out[f"to::{k}"] = bt.get(f"to::{k}")
    return out


def _t(d: pd.Series) -> float:
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else np.nan


def _roll_hit(x: pd.DataFrame, k: str) -> tuple:
    if len(x) < ROLL + 1:
        return np.nan, np.nan
    g = np.log1p(x[[k, "EW"]]).rolling(ROLL).sum().dropna() * 12 / ROLL
    d = np.expm1(g[k]) - np.expm1(g["EW"])
    return float((d > 0).mean()), float(d.median() * 100)


def summarise(nt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for era, a, b in ERAS:
        m = np.ones(len(nt), bool)
        if a:
            m &= nt.index >= pd.Timestamp(a)
        if b:
            m &= nt.index < pd.Timestamp(b)
        x = nt[m]
        if len(x) < 12:
            continue
        for k in KEYS:
            r = x[k]
            nav = np.concatenate([[1.0], (1.0 + r).cumprod().to_numpy()])
            d = r - x["EW"]
            hit, roll_med = _roll_hit(x, k)
            rec = dict(era=era, style=k, months=len(r),
                       cagr=(nav[-1] ** (12 / len(r)) - 1) * 100,
                       vol=r.std(ddof=1) * np.sqrt(12) * 100,
                       ret_vol=r.mean() * 12 / (r.std(ddof=1) * np.sqrt(12)),
                       maxdd=(nav / np.maximum.accumulate(nav) - 1).min() * 100,
                       turnover=x[f"to::{k}"].mean() * 12,
                       vs_ew=d.mean() * 12 * 100, t_ew=_t(d), ahead=(d > 0).mean() * 100,
                       roll36_hit=hit * 100 if np.isfinite(hit) else np.nan, roll36_med=roll_med)
            if k in BLENDS:
                dm = r - x[list(BLENDS[k])].mean(axis=1)
                rec.update(vs_members=dm.mean() * 12 * 100, t_members=_t(dm))
            out.append(rec)
    return pd.DataFrame(out)


def report(name: str, sz: str, s: pd.DataFrame) -> None:
    order = {k: i for i, k in enumerate(KEYS)}
    print(f"\n══ {name} · {sz} book · net of costs ═══════════════════════════════════════════", flush=True)
    by_era = s[s.era != "FULL"].pivot_table(index="style", columns="era", values=["cagr", "vs_ew", "t_ew"])
    by_era = by_era.sort_index(key=lambda i: i.map(order))
    print("per era — CAGR %, vs EW %/yr, t vs EW", flush=True)
    print(by_era.round(2).to_string(), flush=True)
    full = s[s.era == "FULL"].set_index("style").reindex(KEYS)
    cols = ["months", "cagr", "vol", "ret_vol", "maxdd", "turnover", "vs_ew", "t_ew", "ahead",
            "roll36_hit", "roll36_med", "vs_members", "t_members"]
    print("full span", flush=True)
    print(full[[c for c in cols if c in full]].round(2).to_string(), flush=True)


def verdict(summ: dict) -> None:
    """The pre-registered rule: every-name book, both stock universes, all three eras."""
    print("\n══ DECISION RULE · every-name book · Nifty 50 and Dow 30 · E1 / E2 / E3 ═══════════", flush=True)
    for metric, label in (("cagr", "return"), ("ret_vol", "return / volatility")):
        print(f"\n{label}: eras won (of 6) by each blend against each style", flush=True)
        rows = []
        for bl in BLENDS:
            rec = {"blend": bl}
            for ref in KEYS:
                if ref == bl:
                    continue
                wins, cells = 0, 0
                for u in ("Nifty 50", "Dow 30"):
                    s = summ.get((u, "all"))
                    if s is None:
                        continue
                    for era in ("E1", "E2", "E3"):
                        x = s[s.era == era].set_index("style")[metric]
                        if bl in x and ref in x:
                            cells += 1
                            wins += int(x[bl] > x[ref])
                rec[ref] = f"{wins}/{cells}" + (" ✓" if cells == 6 and wins == 6 else "")
            rows.append(rec)
        print(pd.DataFrame(rows).set_index("blend").to_string(), flush=True)


if __name__ == "__main__":
    keys = [a for a in sys.argv[1:] if a in UNIVERSES] or list(UNIVERSES)
    res_path = os.path.join(HERE, "style_blends_results.pkl")
    res = pickle.load(open(res_path, "rb")) if os.path.exists(res_path) else {}
    for key in keys:
        name, cost = UNIVERSES[key]
        snaps = snapshots(key)
        n_uni = len(snaps[-1][1])
        sizes = [(sz, n) for sz, n in SIZES if n is None or n < n_uni]
        print(f"\n== {name}: {n_uni} symbols, {len(snaps)} snapshots "
              f"{pd.Timestamp(snaps[0][0]):%Y-%m-%d} → {pd.Timestamp(snaps[-1][0]):%Y-%m-%d}", flush=True)
        for sz, bt in backtest(snaps, sizes).items():
            res[(name, sz)] = net(bt, cost).assign(**{f"n::{k}": bt[f"n::{k}"] for k in KEYS})
    pickle.dump(res, open(res_path, "wb"))
    summ = {k: summarise(v) for k, v in res.items()}
    for (name, sz), s in summ.items():
        report(name, sz, s)
    verdict(summ)
