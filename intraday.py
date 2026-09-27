"""
PRAGYAM — intraday bars for the conviction ladder, Ladder DOWN (pragati.pine v9.1's default)
══════════════════════════════════════════════════════════════════════════════════════════

On a daily chart the Pine's Ladder down reads every standard frame below the chart —
1m · 3m · 5m · 15m · 30m · 1h · 4h — each running the full conviction engine on its own
history, and takes each one's participation-weighted average inside the day. This module
supplies those frames from what yfinance carries:

    1m   last  7 days       3m   built from 1m
    5m   last 60 days       15m  last 60 days       30m  last 60 days
    1h   last 730 days      4h   built from 1h (four hourly bars per chunk of the session)

A rung joins where its own history has calibrated — exactly as the Pine's rungs join when
their intrabars reach back far enough — so the finest rungs speak only for the last days
and the 1h / 4h rungs for the last ~2 years. Older bars have NO lower frame at all; there
pragati.compute falls back to Ladder up and marks the tape ↺ (the Pine's FALLBACK rule,
extended from "no frames on this chart" to "no intrabars on this bar").

Fetches are batched per universe (``prefetch``) and cached for the session; a symbol not
prefetched is fetched on demand. Anything that is not genuinely intraday (a test double,
an empty reply) is dropped rather than guessed at. ``ENABLED = False`` turns the whole
module off — the engine then reads Ladder up everywhere.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

ENABLED = True
# frame label -> (yfinance interval, period)
FETCH = {"1m": ("1m", "7d"), "5m": ("5m", "60d"), "15m": ("15m", "60d"),
         "30m": ("30m", "60d"), "1h": ("60m", "730d")}
# frame label -> (source frame, bars per chunk inside a session)
DERIVED = {"3m": ("1m", 3), "4h": ("1h", 4)}
DAILY_FRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h")     # the Pine's daily ladder down
WEEKLY_FRAMES = ("1h", "4h")                                     # + the daily rung (pragati)

_CACHE: dict = {}          # (symbol, frame) -> DataFrame (lower-case OHLCV, tz-aware index)
_LOCK = threading.Lock()


def _normalise(d: pd.DataFrame | None) -> pd.DataFrame | None:
    if d is None or len(d) == 0:
        return None
    d = d.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)
    if not {"open", "high", "low", "close"}.issubset(d.columns):
        return None
    if "volume" not in d.columns:
        d["volume"] = np.nan
    d = d[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    d = d[d["close"].notna() & (d["close"] > 0)]
    if len(d) < 50:
        return None
    ix = pd.DatetimeIndex(d.index)
    step = pd.Series(ix).diff().median()
    if pd.isna(step) or step >= pd.Timedelta(hours=12):      # not intraday — refuse it
        return None
    if ix.tz is None:
        ix = ix.tz_localize("UTC")
    d.index = ix
    return d[~d.index.duplicated(keep="last")].sort_index()


def _download(symbols: list, interval: str, period: str) -> dict:
    import yfinance as yf
    out = {}
    for k in range(3):
        try:
            raw = yf.download(symbols, period=period, interval=interval, progress=False,
                              auto_adjust=False, group_by="ticker", threads=True)
            break
        except Exception:
            raw = None
            time.sleep(2 * 2 ** k)
    if raw is None or len(raw) == 0:
        return out
    if len(symbols) == 1 and not isinstance(raw.columns, pd.MultiIndex):
        out[symbols[0]] = _normalise(raw)
        return out
    for s in symbols:
        try:
            out[s] = _normalise(raw[s].dropna(how="all"))
        except Exception:
            out[s] = None
    return out


def _derive(src: pd.DataFrame | None, n: int) -> pd.DataFrame | None:
    """n consecutive bars of the source frame inside each local session day, as one bar."""
    if src is None or len(src) == 0:
        return None
    day = pd.Series(src.index.tz_convert(src.index.tz).date, index=src.index)
    k = day.groupby(day.values).cumcount().to_numpy() // n
    key = pd.MultiIndex.from_arrays([day.values, k])
    g = src.groupby(key)
    o = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum(min_count=1)})
    first_ts = pd.Series(src.index, index=src.index).groupby(key).first()
    o.index = pd.DatetimeIndex(first_ts.reindex(o.index).to_numpy())
    if o.index.tz is None:
        o.index = o.index.tz_localize(src.index.tz)
    return o.sort_index()


def sources(wanted=DAILY_FRAMES) -> tuple:
    """The fetched frames a set of ladder frames is built from (3m <- 1m, 4h <- 1h)."""
    return tuple(dict.fromkeys(DERIVED[f][0] if f in DERIVED else f for f in wanted))


def prefetch(symbols, frames=tuple(FETCH), on_frame=None, report=DAILY_FRAMES) -> dict:
    """Batch-fetch intraday frames for a universe into the session cache.

    ``on_frame(i, n, frame)`` is called before each frame (for a progress bar). Returns the
    coverage — {frame: symbols that have it} over ``report``, derived frames included."""
    if not ENABLED:
        return {}
    syms = [s for s in dict.fromkeys(symbols) if s]
    frames = tuple(frames)
    for i, f in enumerate(frames):
        if on_frame is not None:
            on_frame(i, len(frames), f)
        need = [s for s in syms if (s, f) not in _CACHE]
        for j in range(0, len(need), 50):
            chunk = need[j:j + 50]
            got = _download(chunk, *FETCH[f])
            with _LOCK:
                for s in chunk:
                    _CACHE[(s, f)] = got.get(s)
    with _LOCK:
        for s in syms:
            for f, (src, n) in DERIVED.items():
                if (s, f) not in _CACHE and (s, src) in _CACHE:
                    _CACHE[(s, f)] = _derive(_CACHE[(s, src)], n)
        return {f: sum(1 for s in syms if _CACHE.get((s, f)) is not None) for f in report}


def frames(symbol: str, wanted=DAILY_FRAMES) -> dict:
    """{frame: bars} for one symbol — only the frames that exist. Fetches what is missing."""
    if not ENABLED or not symbol:
        return {}
    base = sources(wanted)
    missing = [f for f in base if (symbol, f) not in _CACHE]
    if missing:
        prefetch([symbol], frames=tuple(missing))
    with _LOCK:
        for f, (src, n) in DERIVED.items():
            if f in wanted and (symbol, f) not in _CACHE and (symbol, src) in _CACHE:
                _CACHE[(symbol, f)] = _derive(_CACHE[(symbol, src)], n)
        return {f: _CACHE[(symbol, f)] for f in wanted
                if _CACHE.get((symbol, f)) is not None and len(_CACHE[(symbol, f)])}


def clear() -> None:
    with _LOCK:
        _CACHE.clear()


def session_days(ix: pd.DatetimeIndex, chart_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The daily chart bar each intraday bar belongs to.

    The local trading date — except an evening session (a bar opening 17:00 or later on an
    American exchange clock: CME / ICE futures open 18:00 ET the evening before), which
    belongs to the NEXT chart day. Bars with no chart day to belong to map to NaT.
    """
    days = pd.DatetimeIndex(chart_days).normalize().unique().sort_values()
    loc = pd.DatetimeIndex(ix)
    tzname = str(loc.tz) if loc.tz is not None else ""
    naive = loc.tz_localize(None) if loc.tz is not None else loc
    d = naive.normalize()
    evening = tzname.startswith("America/") & (naive.hour >= 17)
    pos = days.searchsorted(d, side="left")
    pos = np.where(evening, days.searchsorted(d, side="right"), pos)
    ok = pos < len(days)
    out = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[ok] = days.to_numpy()[pos[ok]]
    # a regular-session bar whose date is not itself a chart day (a holiday print) is dropped
    same = (~evening) & ok
    bad = same & (days.to_numpy()[np.minimum(pos, len(days) - 1)] != d.to_numpy())
    out[bad] = np.datetime64("NaT")
    return pd.DatetimeIndex(out)


__all__ = ["DAILY_FRAMES", "DERIVED", "ENABLED", "FETCH", "WEEKLY_FRAMES", "clear", "frames",
           "prefetch", "session_days", "sources"]
