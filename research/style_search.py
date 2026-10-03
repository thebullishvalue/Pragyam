"""
research/style_search.py — the shared harness for the style search: is there a style that beats all eight?

THE QUESTION
────────────
research/style_blends.py and research/etf_blends.py measured the four shipped styles (EW, ERC, HRP,
CVG) and four blends of HRP / CVG / EW. None beat Equal Weight and CVG on return in every era. This
harness lets candidate styles — proposed from the literature and from what this repo has already
measured — run through the identical book, fast, and keeps the test honest.

PROTOCOL (fixed before any candidate was run)
─────────────────────────────────────────────
    DISCOVERY   E1 < 2014 and E2 2014-2019, Nifty 50 and Dow 30, every name held. Candidates are
                built, read and chosen here only. The discovery files stop at the first trading day
                of 2020 (the last period's end price), so a candidate cannot see the holdout.
    HOLDOUT     E3 ≥ 2020 on Nifty 50 and Dow 30, and the 27-fund ETF window (Mar 2025 →), run
                ONCE on the finalists after discovery closes.
    THE BAR     a candidate BEATS ALL only if its net CAGR exceeds the best of the eight existing
                styles in EVERY cell — E1, E2 and E3 on both stock universes (six cells; the best
                differs by cell). The ETF window is reported, not ruled on (19 months).
    THE COUNT   every candidate and grid point tried is counted; a holdout pass is read against
                that count (a family-wise haircut on its holdout t), not on its own.

THE BOOK (as style_blends.py)
─────────────────────────────
A candidate is a function `fn(ctx) -> pd.Series` of non-negative raw weights over the names priced
on the rebalance date (`ctx.priced`). The harness renormalises them, applies the pipeline's last
step (sb.cut: top-N by weight, the 10% cap), holds the book from the first trading day of the month
to the next — a name gone = cash — and charges 10bp (India) / 3bp (US) per unit of one-way turnover
on the drifted book. Long-only and fully invested by construction; every priced name is eligible.

`ctx` carries, as of the rebalance date and never after it:
    ctx.date      the rebalance date
    ctx.priced    the symbols with a price that day (the allocation universe)
    ctx.px        daily closes, date × symbol, up to and including ctx.date (stale closes unpriced)
    ctx.snap      that day's snapshot, symbol-indexed: RSI, oscillators, MAs, volume profile,
                  Pragati's tapes and the CVG state (see backdata.COLUMN_ORDER)
    ctx.raw       the four shipped styles' raw weights that day: {"EW","ERC","HRP","CVG"} → Series
    ctx.panel(c)  any numeric snapshot column as a daily date × symbol panel, up to ctx.date

Build:   python research/style_search.py build          (all three universes; ~5 min each)
Use:     import style_search as ss
         d = ss.load("nifty_50")                         (discovery; holdout=True is the finalists' run)
         r = ss.run(my_fn, d)                            (monthly net returns, turnover, names)
         ss.report({"mine": r}, d)                       (per era, against all eight)
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import style_blends as sb                         # noqa: E402

DISC_END = pd.Timestamp("2020-01-01")
UNIV = {"nifty_50": ("Nifty 50", 10.0), "dow_30": ("Dow 30", 3.0), "etf_27": ("ETF book (27)", 10.0)}
ETF_YOUNG = ("NIFTYIETF", "CHEMICAL", "GROWWPOWER")
ERAS = (("E1", None, "2014-01-01"), ("E2", "2014-01-01", "2020-01-01"), ("E3", "2020-01-01", None))
BASE = ("EW", "ERC", "HRP", "CVG", "HRP+CVG", "HRP+EW", "CVG+EW", "HRP+CVG+EW")


class Ctx:
    def __init__(self, data: dict, a: pd.Timestamp):
        self._d, self.date = data, a
        self.px = data["px"].loc[:a]
        self.snap = data["snap"][a]
        self.raw = {k: v[a] for k, v in data["raw"].items()}
        p = self.px.iloc[-1]
        self.priced = p.index[p.notna() & (p > 0)]

    def panel(self, col: str) -> pd.DataFrame:
        return self._d["panels"][col].loc[:self.date]


def _path(u: str, holdout: bool) -> str:
    return os.path.join(HERE, f"search_{u}_{'full' if holdout else 'disc'}.pkl")


def build(u: str) -> None:
    key = "etf_book" if u == "etf_27" else u
    snaps = sb.snapshots(key)
    if u == "etf_27":
        snaps = [(d, s[~s["symbol"].isin(ETF_YOUNG)].reset_index(drop=True)) for d, s in snaps]
    by_date = {pd.Timestamp(d): s.drop_duplicates("symbol", keep="last").set_index("symbol") for d, s in snaps}
    px = pd.DataFrame({d: pd.to_numeric(s["price"], errors="coerce") for d, s in by_date.items()}).T.sort_index()
    num = [c for c in snaps[-1][1].columns if c not in ("date", "symbol", "price")
           and pd.api.types.is_numeric_dtype(pd.to_numeric(snaps[-1][1][c], errors="coerce"))
           and pd.to_numeric(snaps[-1][1][c], errors="coerce").notna().any()]
    panels = {c: pd.DataFrame({d: pd.to_numeric(s[c], errors="coerce") for d, s in by_date.items()}).T.sort_index()
              for c in num}
    cal = px.index
    months = list(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).first())
    pos = {d: i for i, d in enumerate(cal)}
    raw, snap, used = {k: {} for k in sb.METHOD}, {}, []
    started = False
    for a in months:
        if u == "etf_27" and not px.loc[a].notna().all():
            continue
        hist = snaps[max(0, pos[a] - 252): pos[a] + 1]
        w = {k: sb.raw(hist, m) for k, m in sb.METHOD.items()}
        if not started:
            if min(len(v) for v in w.values()) < sb.MIN_NAMES:
                continue
            started = True
        for k in w:
            raw[k][a] = w[k]
        snap[a] = by_date[a]
        used.append(a)
    full = dict(u=u, name=UNIV[u][0], cost=UNIV[u][1], px=px, panels=panels, months=used, raw=raw, snap=snap)
    pickle.dump(full, open(_path(u, True), "wb"))
    if u != "etf_27":
        m_disc = [a for a in used if a < DISC_END]
        end = next(a for a in used if a >= DISC_END)          # the last discovery period's end price
        disc = dict(full, px=px.loc[:end], panels={c: p.loc[:m_disc[-1]] for c, p in panels.items()},
                    months=m_disc + [end], raw={k: {a: v[a] for a in m_disc} for k, v in raw.items()},
                    snap={a: snap[a] for a in m_disc})
        pickle.dump(disc, open(_path(u, False), "wb"))
    print(f"built {u}: {len(used)} rebalances {used[0]:%Y-%m-%d} → {used[-1]:%Y-%m-%d}", flush=True)


def load(u: str, holdout: bool = False) -> dict:
    return pickle.load(open(_path(u, holdout), "rb"))


def run(fn, data: dict, n=None) -> pd.DataFrame:
    """Monthly net returns of the book `fn` builds, as style_blends.backtest does for the shipped styles."""
    px, months, cost = data["px"], data["months"], data["cost"]
    rows, prev = [], None
    for a, b in zip(months[:-1], months[1:]):
        ctx = Ctx(data, a)
        w = pd.Series(fn(ctx), dtype=float)
        w = w[w.index.isin(ctx.priced)].clip(lower=0.0).fillna(0.0)
        if w.sum() <= 0:
            raise ValueError(f"{a:%Y-%m-%d}: the candidate returned no weight")
        w = sb.cut(w.sort_values(ascending=False, kind="stable"), n)
        r = (px.loc[b].reindex(w.index) / px.loc[a].reindex(w.index) - 1.0).fillna(0.0)
        to = np.nan
        if prev is not None:
            d = prev[0] * (1.0 + prev[1])
            d = d / d.sum()
            idx = w.index.union(d.index)
            to = float(0.5 * (w.reindex(idx, fill_value=0) - d.reindex(idx, fill_value=0)).abs().sum())
        gross = float((w * r).sum())
        rows.append(dict(date=a, gross=gross, ret=gross - (0.0 if np.isnan(to) else to) * cost / 1e4,
                         to=to, names=len(w), top=float(w.max())))
        prev = (w, r)
    return pd.DataFrame(rows).set_index("date")


def baselines(data: dict, n=None) -> dict:
    """The eight existing styles, rebuilt from the stored raw weights through `run`."""
    fns = {k: (lambda c, k=k: c.raw[k]) for k in sb.METHOD}
    fns.update({k: (lambda c, mem=mem: sb.blend([c.raw[m] for m in mem])) for k, mem in sb.BLENDS.items()})
    return {k: run(f, data, n) for k, f in fns.items()}


def _era(df: pd.DataFrame, a, b) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    if a:
        m &= df.index >= pd.Timestamp(a)
    if b:
        m &= df.index < pd.Timestamp(b)
    return df[m]


def metrics(r: pd.DataFrame) -> dict:
    x = r["ret"]
    nav = np.concatenate([[1.0], (1 + x).cumprod().to_numpy()])
    return dict(months=len(x), cagr=(nav[-1] ** (12 / len(x)) - 1) * 100,
                vol=x.std(ddof=1) * np.sqrt(12) * 100,
                ret_vol=x.mean() * 12 / (x.std(ddof=1) * np.sqrt(12)),
                maxdd=(nav / np.maximum.accumulate(nav) - 1).min() * 100,
                turnover=r["to"].mean() * 12)


def report(cands: dict, data: dict, base: dict | None = None, n=None, show_base: bool = True) -> pd.DataFrame:
    """Per-era metrics of each candidate, and its CAGR margin over the best existing style per era."""
    base = base if base is not None else baselines(data, n)
    span = data["months"][0], data["months"][-1]
    eras = [e for e in ERAS if len(_era(next(iter(base.values())), e[1], e[2])) >= 12] or [("ALL", None, None)]
    rows = []
    for name, r in {**(base if show_base else {}), **cands}.items():
        for era, a, b in eras:
            x = _era(r, a, b)
            best = max(BASE, key=lambda k: metrics(_era(base[k], a, b))["cagr"])
            bm = metrics(_era(base[best], a, b))
            m = metrics(x)
            d = x["ret"] - _era(base[best], a, b)["ret"]
            dew = x["ret"] - _era(base["EW"], a, b)["ret"]
            rows.append(dict(style=name, era=era, **m, best=best, vs_best=m["cagr"] - bm["cagr"],
                             t_best=d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else np.nan,
                             vs_ew=m["cagr"] - metrics(_era(base["EW"], a, b))["cagr"],
                             t_ew=dew.mean() / (dew.std(ddof=1) / np.sqrt(len(dew))) if dew.std() > 0 else np.nan))
    out = pd.DataFrame(rows)
    print(f"\n== {data['name']} · {span[0]:%Y-%m} → {span[1]:%Y-%m} · net of costs · "
          f"{'every name' if n is None else f'top-{n}'}", flush=True)
    piv = out.pivot_table(index="style", columns="era", values=["cagr", "vs_best"], sort=False)
    print(piv.round(2).to_string(), flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        for u in (sys.argv[2:] or list(UNIV)):
            build(u)
