"""
research/pragyam_panels.py — the Pragyam risk engine (research/pragyam_engine.py), run per symbol over
the v14 style-search universes, stored as daily date × symbol panels.

What it reads and how it is cleaned, in this order:
    1. Daily OHLC and adjusted close from yfinance (auto_adjust=False), 2005-01-01 → 2026-10-02 (the
       research panels' END), so the engine's 252-bar warm-up is done before the first 2007 rebalance.
    2. OHLC scaled by adjusted / raw close — as Indicator/research/lib.load prepares every series the
       engine was validated on.
    3. An Indian listing's unadjusted corporate action back-adjusted and its bad prints dropped
       (backdata.corporate_action_gaps / back_adjust, the same repair the app's panel gets).
    4. Indicator/research/lib.load's own cleaning: close > 0, high ≥ low, a missing or misprinted open
       (|log(open / close)| ≥ 0.3) set to the close, one-day moves of 60% or more dropped as bad prints.
    5. Dead quotes (a close repeated in a run of ≥ 10) and exchange-holiday prints (dates on which at
       least half the names that report volume printed a flat, zero-volume bar) dropped, as the app's
       panel does (backdata).
    6. pragyam_engine.run(...) at its defaults (H = 10), on each symbol's own trading days.
The panels are then aligned to the style-search calendar by the harness, which carries a value over a
gap of at most 5 sessions, as it does prices.

Every series the engine draws is causal: the value on a date uses that date's bar and earlier ones only.

Run:   python research/pragyam_panels.py [nifty_50|dow_35|etf_book]     (all three by default)
Out:   research/pragyam_<universe>.pkl  {"panels": {name: date × symbol}, "meta": {...}}
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import pragyam_engine as eng                      # noqa: E402

START, END = "2005-01-01", "2026-10-03"           # END exclusive: the research panels end 2026-10-02
KEEP = ("sig", "T", "E", "dialR", "dialT", "reg", "prk", "slope")
DEAD_RUN = 10
UNIVERSES = ("nifty_50", "dow_35", "etf_book")


def tickers(key: str) -> list:
    from universe import DOW_JONES_TICKERS, ETF_UNIVERSE, get_index_stock_list
    if key == "nifty_50":
        return list(get_index_stock_list("NIFTY 50")[0] or [])
    if key == "etf_book":
        return list(ETF_UNIVERSE)
    from style_search_pit import ADDED                # the point-in-time Dow's extra members
    return list(dict.fromkeys(list(DOW_JONES_TICKERS) + list(ADDED)))


def fetch(tk: list) -> dict:
    import yfinance as yf
    raw = yf.download(tk, start=START, end=END, auto_adjust=False, progress=False, threads=True)
    out = {}
    for t in tk:
        try:
            d = pd.DataFrame({k.lower().replace(" ", "_"): pd.to_numeric(raw[k][t], errors="coerce")
                              for k in ("Open", "High", "Low", "Close", "Adj Close", "Volume")})
        except KeyError:
            continue
        d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated(keep="last")].dropna(subset=["close"]).sort_index()
        if len(d):
            out[t] = d
    return out


def clean(t: str, d: pd.DataFrame, holidays: pd.DatetimeIndex) -> pd.DataFrame:
    import backdata as bd
    d = d[(d.close > 0) & (d.high >= d.low)].copy()
    k = (d.adj_close / d.close).where(d.adj_close > 0, 1.0)
    for c in ("open", "high", "low", "close"):
        d[c] = d[c] * k
    if t.upper().endswith((".NS", ".BO")):
        ev, bad = bd.corporate_action_gaps(d[["close"]].set_axis([t], axis=1), d[["open"]].set_axis([t], axis=1))
        for _c, t0, t1 in bad:
            d = d[~((d.index >= t0) & (d.index < t1))]
        if ev:
            d = bd.back_adjust(d, ev, columns=lambda _s: ["open", "high", "low", "close"])
    d["open"] = d.open.where((d.open > 0) & (np.log(d.open / d.close).abs() < 0.3), d.close)
    d = d[d.close.pct_change().abs().fillna(0) < 0.6]
    same = d.close.diff().eq(0)
    run = same.groupby((~same).cumsum()).transform("sum")
    d = d[~(same & (run >= DEAD_RUN))]
    if len(holidays):
        flat = d.index.isin(holidays) & (d.volume.fillna(0) <= 0) & (d.high == d.low)
        d = d[~flat]
    return d


def holiday_dates(frames: dict) -> pd.DatetimeIndex:
    vol = pd.DataFrame({t: d.volume for t, d in frames.items()})
    flat = pd.DataFrame({t: (d.volume.fillna(0) <= 0) & (d.high == d.low) for t, d in frames.items()})
    reports = (vol > 0).any()
    if int(reports.sum()) < 5:
        return pd.DatetimeIndex([])
    share = flat.loc[:, reports].sum(axis=1) / vol.loc[:, reports].notna().sum(axis=1).clip(lower=1)
    return pd.DatetimeIndex(share.index[share >= 0.5])


def build(key: str) -> None:
    t0 = time.time()
    tk = tickers(key)
    frames = fetch(tk)
    hol = holiday_dates(frames)
    cols = {k: {} for k in KEEP}
    bars = {}
    for t, d in frames.items():
        d = clean(t, d, hol)
        if len(d) < eng.WARM + 20:
            continue
        r = eng.run(d[["open", "high", "low", "close"]])
        name = t.replace(".NS", "")
        for k in KEEP:
            cols[k][name] = r[k]
        bars[name] = len(d)
    panels = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    meta = dict(universe=key, requested=len(tk), built=len(bars), missing=sorted(set(tk) - set(frames)),
                holidays=len(hol), bars=bars, start=START, end=END, at=datetime.now().isoformat(timespec="seconds"))
    pickle.dump({"panels": panels, "meta": meta}, open(os.path.join(HERE, f"pragyam_{key}.pkl"), "wb"))
    print(f"built pragyam_{key}: {len(bars)} of {len(tk)} symbols · {panels['sig'].index[0]:%Y-%m-%d} → "
          f"{panels['sig'].index[-1]:%Y-%m-%d} · {len(hol)} holiday dates · {time.time() - t0:.0f}s"
          + (f" · missing {meta['missing']}" if meta["missing"] else ""), flush=True)


def load(key: str) -> dict:
    return pickle.load(open(os.path.join(HERE, f"pragyam_{key}.pkl"), "rb"))["panels"]


if __name__ == "__main__":
    for k in (sys.argv[1:] or UNIVERSES):
        build(k)
