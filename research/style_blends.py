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

ITERATION 2 (registered after iteration 1's numbers were seen — a sensitivity, not a new rule):
the HRP blends trailed the mean of their members by 0.25 %/yr on Nifty 50. Averaging RAW weights
lets HRP's uncapped low-volatility names keep more than the 10%-capped HRP book gives them. The
other reading of "an average of two styles" — an equal share of each finished, capped book (·b) —
is measured alongside on the every-name book; its gross return is exactly its members' mean, so
its gap to them is the turnover the netting saves.

DATA REPAIR (found in iteration 1, applied to every universe and style alike): yfinance carries
NESTLEIND.NS as a flat line from Oct 2006 to Jan 2010 (786 unchanged closes) and BAJAJ-AUTO.NS
flat for 45 sessions around its 2008 demerger relisting. A zero-variance name takes nearly all of
an inverse-variance split — raw HRP put 100% on NESTLEIND in 2009, and the HRP and ERC books held
it at the 10% cap — so iteration 1's Nifty 50 books before 2010 partly read the defect. A close that repeats the
previous one inside a run of ≥ STALE_RUN sessions is now unpriced: no style holds the name on
those days, and the covariance styles admit it once its window has real returns. Dow 30 and the
ETF book have no such run; their books are unchanged.

RESULT (2026-10-03) — no blend beats Equal Weight or CVG on return; HRP+CVG is a better risk
style than ERC. Every-name book, Nov 2006 – Sep 2026 (236 months), net of costs, repaired data:

                       Nifty 50                                     Dow 30
              CAGR   vol  ret/vol  maxDD  turn  vs EW (t)     CAGR   vol  ret/vol  maxDD  turn  vs EW (t)
  EW         19.69  22.3   0.93  -58.4  0.37      —          15.52  16.5   0.96  -39.9  0.27      —
  ERC        19.28  20.3   0.98  -54.6  0.42  -0.75 (-1.2)   14.23  15.3   0.95  -37.7  0.31  -1.33 (-2.6)
  HRP        19.31  18.8   1.04  -50.3  1.23  -1.03 (-0.8)   12.87  14.2   0.93  -35.5  0.91  -2.70 (-2.8)
  CVG        20.48  22.7   0.94  -58.5  1.48  +0.76 (+2.4)   15.78  16.7   0.97  -39.3  1.41  +0.28 (+1.0)
  HRP+CVG    19.98  20.6   0.99  -54.5  1.08  -0.11 (-0.2)   14.39  15.3   0.96  -36.8  0.95  -1.18 (-2.4)
  HRP+EW     19.55  20.4   0.98  -54.5  0.71  -0.51 (-0.8)   14.25  15.2   0.96  -37.2  0.54  -1.32 (-2.7)
  CVG+EW     20.10  22.4   0.93  -58.4  0.87  +0.39 (+2.4)   15.65  16.6   0.96  -39.6  0.79  +0.14 (+1.0)
  HRP+CVG+EW 19.90  21.1   0.97  -55.8  0.80  -0.07 (-0.2)   14.77  15.7   0.96  -37.9  0.69  -0.78 (-2.4)

  (vs EW is the paired monthly gap, %/yr arithmetic, and its t; a lower-volatility book gives up
  less CAGR than that gap, so read CAGR for the compounded result.)

  CAGR by era, E1 / E2 / E3     Nifty 50                  Dow 30
  EW                            17.93 / 18.83 / 22.31     14.39 / 18.32 / 14.23
  ERC                           18.57 / 18.27 / 20.93     13.37 / 16.77 / 12.88
  HRP                           20.21 / 17.40 / 20.10     12.48 / 15.40 / 11.06
  CVG                           19.16 / 19.68 / 22.59     14.17 / 18.50 / 15.06
  HRP+CVG                       19.80 / 18.59 / 21.40     13.48 / 16.92 / 13.10
  HRP+EW                        19.15 / 18.15 / 21.23     13.58 / 16.83 / 12.68
  CVG+EW                        18.55 / 19.26 / 22.45     14.28 / 18.41 / 14.65
  HRP+CVG+EW                    19.20 / 18.69 / 21.71     13.79 / 17.39 / 13.49

Against the rule (six cells: two universes × three eras):
  · return — no blend beats EW (CVG+EW 5/6: it trails by 0.11 on Dow E1; the HRP blends 1/6) or
    CVG (≤ 1/6). HRP+CVG and HRP+CVG+EW beat ERC 6/6; HRP+CVG+EW beats HRP+EW 6/6.
  · return / volatility — only HRP+CVG passes, over ERC (6/6): ERC's volatility (20.6 vs 20.3
    Nifty, 15.3 vs 15.3 Dow) at +0.70 / +0.16 %/yr more CAGR, at 2.5-3x its turnover (netted).
  · a blend is its members' midpoint: within +0.01 to +0.03 %/yr of their mean on both stock
    panels (the turnover the netting saves); raw-weight and finished-book averages agree to ±0.03.
  · CVG+EW is CVG at half the active bet: half its edge (+0.39 / +0.14), the same t, 0.6x its
    turnover. HRP+EW is an ERC at 1.7x the trading. Return / volatility spans only 0.93-1.04
    on Nifty and 0.93-0.97 on Dow across all eight — the styles trade return for risk at ~par.
Iteration 1 (stale data): Nifty HRP read 18.75% (E1 18.60%) and EW 19.45%; the flat names took
the 10% cap in the HRP and ERC books and ~1/N in the rest, so every Nifty style read low.
Not ruled on: the ETF book is 27 months (Jul 2024 on, ≥ 10 funds) — the HRP family +0.4 to +0.8
%/yr over EW, CVG −0.45, no t above 0.7. The top-N books measure selection more than weighting
(HRP / ERC keep the lowest-volatility names, 1/N keeps listing order): Nifty top-30 CAGR CVG 21.73
· CVG+EW 21.44 · EW 20.29 · HRP+CVG 20.03 · HRP+CVG+EW 20.00 · HRP+EW 18.86 · HRP 18.80 · ERC 17.64.

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

CAPITAL, CAP, MIN_NAMES, ROLL, STALE_RUN = 1e10, 0.10, 10, 36, 10
START, END = datetime(2006, 1, 1), datetime(2026, 10, 2)
assert cvgrid.STATE_UNITS["DISLOCATED"] == 4.0, "the shipped CVG units are the control"

METHOD = {"EW": "EQUAL", "ERC": "ERC", "HRP": "HRP", "CVG": "CVG"}
BLENDS = {"HRP+CVG": ("HRP", "CVG"), "HRP+EW": ("HRP", "EW"), "CVG+EW": ("CVG", "EW"),
          "HRP+CVG+EW": ("HRP", "CVG", "EW")}
BOOK_BLENDS = {f"{k}·b": v for k, v in BLENDS.items()}       # iteration 2, every-name book only
MEMBERS = {**BLENDS, **BOOK_BLENDS}
KEYS = list(METHOD) + list(BLENDS)
ALL_KEYS = KEYS + list(BOOK_BLENDS)
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
        return unstale(pickle.load(open(p, "rb")))
    from backdata import generate_historical_data
    snaps = generate_historical_data(symbols(key), START, END)
    pickle.dump(snaps, open(p, "wb"))
    return unstale(snaps)


def unstale(snaps: list) -> list:
    """Unprice every close that repeats the one before it inside a run of >= STALE_RUN sessions."""
    px = pd.DataFrame({pd.Timestamp(d): pd.to_numeric(s.drop_duplicates("symbol", keep="last")
                                                      .set_index("symbol")["price"], errors="coerce")
                       for d, s in snaps}).T.sort_index()
    same = px.diff().eq(0)
    run = same.apply(lambda c: c.groupby((~c).cumsum()).transform("sum"))
    bad = same & (run >= STALE_RUN)
    if not bad.to_numpy().any():
        return snaps
    out = []
    for d, s in snaps:
        row = bad.loc[pd.Timestamp(d)]
        drop = set(row.index[row.to_numpy()])
        if drop:
            s = s.copy()
            s.loc[s["symbol"].isin(drop), "price"] = np.nan
        out.append((d, s))
    print(f"   unpriced {int(bad.to_numpy().sum())} stale closes: "
          + ", ".join(f"{c} {int(n)}" for c, n in bad.sum().items() if n), flush=True)
    return out


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
            books = {k: cut(w_raw[k], n) for k in KEYS}
            if n is None:
                books.update({k: cut(blend([books[m] for m in mem]), n) for k, mem in BOOK_BLENDS.items()})
            for k, w in books.items():
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
    for k in (k for k in ALL_KEYS if k in bt):
        out[k] = bt[k] - bt.get(f"to::{k}", pd.Series(0.0, index=bt.index)).fillna(0.0) * cost_bps / 1e4
        out[f"to::{k}"] = bt.get(f"to::{k}")
    return out


def _t(d: pd.Series) -> float:
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else np.nan


def _roll_hit(x: pd.DataFrame, k: str) -> tuple:
    if len(x) < ROLL + 1 or k == "EW":
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
        for k in (k for k in ALL_KEYS if k in nt):
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
            if k in MEMBERS:
                dm = r - x[list(MEMBERS[k])].mean(axis=1)
                rec.update(vs_members=dm.mean() * 12 * 100, t_members=_t(dm))
            out.append(rec)
    return pd.DataFrame(out)


def report(name: str, sz: str, s: pd.DataFrame) -> None:
    keys = [k for k in ALL_KEYS if k in set(s["style"])]
    order = {k: i for i, k in enumerate(keys)}
    print(f"\n══ {name} · {sz} book · net of costs ═══════════════════════════════════════════", flush=True)
    by_era = s[s.era != "FULL"].pivot_table(index="style", columns="era", values=["cagr", "vs_ew", "t_ew"])
    by_era = by_era.sort_index(key=lambda i: i.map(order))
    print("per era — CAGR %, vs EW %/yr, t vs EW", flush=True)
    print(by_era.round(2).to_string(), flush=True)
    full = s[s.era == "FULL"].set_index("style").reindex(keys)
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
            for col, ref in [(k, k) for k in KEYS] + [("its ·b", f"{bl}·b")]:
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
                rec[col] = f"{wins}/{cells}" + (" ✓" if cells == 6 and wins == 6 else "")
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
            res[(name, sz)] = net(bt, cost).assign(**{f"n::{k}": bt[f"n::{k}"] for k in ALL_KEYS if k in bt})
    pickle.dump(res, open(res_path, "wb"))
    summ = {k: summarise(v) for k, v in res.items()}
    for (name, sz), s in summ.items():
        report(name, sz, s)
    verdict(summ)
