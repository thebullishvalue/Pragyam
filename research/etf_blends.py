"""
research/etf_blends.py — the ETF book on the window where every fund in the test is priced throughout.

PRE-REGISTRATION (fixed before any number below was seen)
─────────────────────────────────────────────────────────
research/style_blends.py could read the ETF book only from Jul 2024, and with the book still
filling (10 → 30 funds). yfinance carries this universe's funds only from their current tickers:
the ICICI renames of Dec 2023 and Feb 2025 cut the history (none of the predecessor symbols
resolve), and NIFTYIETF (from 22 Sep 2026), CHEMICAL (31 Jul 2026) and GROWWPOWER (27 Jan 2026)
are too young to test. So:

    Universe    the 27 funds priced since 3 Feb 2025 — ETF_UNIVERSE less NIFTYIETF, CHEMICAL and
                GROWWPOWER — every one of them priced on every day of the test.
    Window A    the first month-start with all 27 priced → the end of the panel (2 Oct 2026).
    Window B    the months of A in which EVERY style holds all 27: HRP and ERC admit a fund only
                once it carries 80% of its 252-day covariance window, so the four Feb-2025 funds
                join their books about ten months in. B is the strict reading; A1 = A before B.
    Styles      as research/style_blends.py: EW, ERC, HRP, CVG, and the blends HRP+CVG, HRP+EW,
                CVG+EW, HRP+CVG+EW (members' raw weights averaged, then top-N and the 10% cap).
    Books       every fund held (27); top-15 reported alongside.
    Marking     rebalanced on the first trading day of each month as before, but valued DAILY
                between rebalances (weights drift with prices), so volatility, drawdown and the
                t on the gap to EW rest on ~400 daily returns rather than ~19 monthly ones.
                Net of 10bp per unit of one-way turnover, charged on the rebalance.

Reported per window and style: CAGR, volatility, return / volatility (rf 0), max drawdown,
turnover, the daily gap to EW (%/yr and t), tracking error to EW, months ahead of EW, and for a
blend the gap to the mean of its members.

READING RULE: one universe, ~19 months — nothing here can be significant or ruled on; the
three-era rule in style_blends.py stays the test. A style "held up" on this book only if it is
ahead of EW in both A1 and B (two disjoint periods). Descriptive only; no product change.

RESULT (2026-10-03) — nothing beat Equal Weight on return, and no style held up: every style
trailed EW through the 2025 rally (A1) and led it through the flat 2026 (B). The gap's sign follows
the market — the risk styles' lower beta, not an edge. The HRP blends had the best return /
volatility and the shallowest drawdowns, by margins this sample cannot separate. Every fund held
(27), daily-marked, net of costs:

               Window A · Mar 2025 – Oct 2026 · 396 days    A1 · Mar – Nov 2025    B · Dec 2025 – Oct 2026
              CAGR   vol  ret/vol  maxDD  turn  vs EW (t)    CAGR  vs EW (t)        CAGR  vs EW (t)
  EW         17.71  13.8   1.25  -12.3  0.20      —         40.64      —            0.86      —
  ERC        15.40  13.1   1.16  -13.2  0.50  -2.08 (-1.1)  32.65  -6.01 (-2.2)    2.26  +1.34 (+0.6)
  HRP        15.92  12.6   1.23  -11.9  0.94  -1.69 (-0.7)  33.26  -5.58 (-1.7)    2.72  +1.69 (+0.5)
  CVG        17.22  13.7   1.23  -12.3  1.44  -0.44 (-0.5)  37.69  -2.15 (-2.2)    1.94  +1.04 (+0.8)
  HRP+CVG    16.82  12.8   1.28  -12.0  1.01  -0.90 (-0.6)  36.40  -3.23 (-1.2)    2.13  +1.13 (+0.6)
  HRP+EW     16.98  12.9   1.28  -12.0  0.59  -0.76 (-0.5)  37.73  -2.25 (-0.9)    1.51  +0.54 (+0.3)
  CVG+EW     17.47  13.7   1.24  -12.3  0.77  -0.22 (-0.5)  39.16  -1.07 (-2.2)    1.41  +0.52 (+0.8)
  HRP+CVG+EW 16.98  13.1   1.27  -12.1  0.73  -0.73 (-0.6)  37.61  -2.31 (-1.2)    1.61  +0.65 (+0.5)

  · HRP and ERC held 17 → 23 funds in A1, not 23 as registered: the ten funds listed after Apr
    2024 lacked 80% of a covariance window at the start. They held all 27 from Dec 2025, where B
    begins. A blend containing HRP held all 27 throughout, the late funds taking only its other
    member's share — hence its gap to its members' mean (+0.6 %/yr in A1, −0.2 in B).
  · ERC drew down further than EW (−13.2 vs −12.3) while giving up 2.1 %/yr.
  · Top-15 books are selection: 1/N keeps listing order, which here includes the silver and gold
    funds (EW top-15 22.3% over A). CVG's top-15 turned over 4.2x/yr and made 13.7%; HRP+CVG's
    18.6% (ret/vol 1.42) led the books that are not 1/N.

Run:  python research/etf_blends.py     (reads research/cvg_reweight_etf_book.pkl, as built by
      research/style_blends.py)
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import style_blends as sb                         # noqa: E402

YOUNG = ("NIFTYIETF", "CHEMICAL", "GROWWPOWER")
COST_BPS = 10.0
SIZES = (("all", None), ("top15", 15))
KEYS = sb.KEYS


def panel() -> tuple:
    snaps = [(d, s[~s["symbol"].isin(YOUNG)].reset_index(drop=True)) for d, s in sb.snapshots("etf_book")]
    px = pd.DataFrame({pd.Timestamp(d): pd.to_numeric(s.drop_duplicates("symbol", keep="last")
                                                      .set_index("symbol")["price"], errors="coerce")
                       for d, s in snaps}).T.sort_index()
    return snaps, px


def run(snaps: list, px: pd.DataFrame) -> dict:
    cal = px.index
    first = px.apply(lambda c: c.first_valid_index())
    assert px.loc[first.max():].notna().all().all(), "a fund in the test has a gap after it lists"
    months = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).first()
    months = [m for m in months if px.loc[m].notna().all()]          # all 27 priced on the day
    bounds = months + ([cal[-1]] if cal[-1] > months[-1] else [])
    pos = {d: i for i, d in enumerate(cal)}
    daily = {sz: {k: [] for k in KEYS} for sz, _ in SIZES}
    meta = {sz: [] for sz, _ in SIZES}
    prev = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        hist = snaps[max(0, pos[a] - (sb.PANEL - 1)): pos[a] + 1]
        w_raw = {k: sb.raw(hist, m) for k, m in sb.METHOD.items()}
        w_raw.update({k: sb.blend([w_raw[m] for m in mem]) for k, mem in sb.BLENDS.items()})
        seg = px.loc[a:b]
        rel = seg / seg.iloc[0]                                        # growth of 1 since a
        for sz, n in SIZES:
            rec = {"date": a}
            for k in KEYS:
                w = sb.cut(w_raw[k], n)
                val = (rel[w.index] * w).sum(axis=1)                   # book value, drifting
                r = val.pct_change().iloc[1:]
                to = 0.0
                if (sz, k) in prev:
                    d = prev[(sz, k)]
                    idx = w.index.union(d.index)
                    to = float(0.5 * (w.reindex(idx, fill_value=0) - d.reindex(idx, fill_value=0)).abs().sum())
                if len(r):
                    r.iloc[0] -= to * COST_BPS / 1e4
                daily[sz][k].append(r)
                end = w * rel[w.index].iloc[-1]
                prev[(sz, k)] = end / end.sum()
                rec[f"to::{k}"] = to
                rec[f"n::{k}"] = len(w)
            meta[sz].append(rec)
    return {sz: (pd.DataFrame({k: pd.concat(v) for k, v in daily[sz].items()}),
                 pd.DataFrame(meta[sz]).set_index("date")) for sz, _ in SIZES}


def stats(r: pd.DataFrame, meta: pd.DataFrame, a, b) -> pd.DataFrame:
    x = r.loc[(r.index > a) & (r.index <= b)]
    m = meta.loc[(meta.index >= a) & (meta.index < b)]
    yrs = len(x) / 252
    pid = np.searchsorted(m.index.to_numpy(), x.index.to_numpy(), side="left") - 1   # its rebalance
    mo = (1 + x).groupby(pid).prod() - 1
    out = []
    for k in KEYS:
        nav = np.concatenate([[1.0], (1 + x[k]).cumprod().to_numpy()])
        d = x[k] - x["EW"]
        rec = dict(style=k, days=len(x), months=len(m),
                   cagr=(nav[-1] ** (1 / yrs) - 1) * 100,
                   vol=x[k].std(ddof=1) * np.sqrt(252) * 100,
                   ret_vol=x[k].mean() * 252 / (x[k].std(ddof=1) * np.sqrt(252)),
                   maxdd=(nav / np.maximum.accumulate(nav) - 1).min() * 100,
                   turnover=m[f"to::{k}"].sum() / yrs,
                   vs_ew=d.mean() * 252 * 100,
                   t_ew=d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else np.nan,
                   te_ew=d.std(ddof=1) * np.sqrt(252) * 100,
                   months_ahead=f"{int((mo[k] > mo['EW']).sum())}/{len(mo)}" if k != "EW" else "",
                   names=f"{int(m[f'n::{k}'].min())}-{int(m[f'n::{k}'].max())}")
        if k in sb.BLENDS:
            dm = x[k] - x[list(sb.BLENDS[k])].mean(axis=1)
            rec["vs_members"] = dm.mean() * 252 * 100
        out.append(rec)
    return pd.DataFrame(out).set_index("style")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    snaps, px = panel()
    print(f"universe: {px.shape[1]} funds · all priced from {px.apply(lambda c: c.first_valid_index()).max():%Y-%m-%d}"
          f" · panel to {px.index[-1]:%Y-%m-%d}", flush=True)
    res = run(snaps, px)
    _, meta_all = res["all"]
    full = (meta_all[[f"n::{k}" for k in KEYS]] == px.shape[1]).all(axis=1)
    a0, b_start, end = meta_all.index[0], full[full].index[0], px.index[-1]
    assert full.loc[b_start:].all(), "a style dropped a fund after holding all of them"
    windows = (("A  (all 27 priced)", a0, end), ("A1 (before B)", a0, b_start), ("B  (every style holds all 27)", b_start, end))
    for sz, (r, meta) in res.items():
        for label, a, b in windows:
            s = stats(r, meta, a, b)
            print(f"\n══ ETF book · {sz} · window {label} · {a:%Y-%m-%d} → {b:%Y-%m-%d} · net of costs ══", flush=True)
            print(s.round(2).to_string(), flush=True)
    res_path = os.path.join(HERE, "etf_blends_results.pkl")
    pd.to_pickle(res, res_path)
