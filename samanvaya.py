"""
PRAGYAM — Samanvaya value engine, ported from pragati.pine section 4c
══════════════════════════════════════════════════════════════════════════════

Pragati's second tape: WHERE PRICE STANDS. Rich (+) or cheap (−) against what the
macro drivers explain, read on the daily chart and on the weekly frame above it.
This module produces the value tape; pragati.py produces the conviction tape;
the Conviction-Value Grid (cvgrid.py) reads the two to place every name in its
state.

The engine, as the Pine runs it on a daily chart with its defaults
──────────────────────────────────────────────────────────────────
RV LEG     The name's daily log return, regressed on up to 3 of the macro
           factors, chosen by stepwise partial correlation over a 250-day
           window read 12 days in arrears (so the fit applied to any day ended
           12 days earlier), admitted past a Fisher / Šidák significance floor,
           solved by ridge Gram-Schmidt. The hedge is then applied only in
           proportion to its own out-of-sample skill over the last 104 days —
           "the hedge weighs itself". The running sum of the hedged residual is
           the spread; its z over five timescales (8 … 55), each high-passed by
           a 400-bar EMA, averaged and rescaled by 0.887, is the RV z.
BREADTH    Seven views of the name's own price through the Market Strength
           Factor (momentum, structure, flow) at five timescales.
BLEND      0.5 / 0.5 in z-space, variance restored with the measured leg
           correlation, soft-bounded once: the chart's value reading.
TAPE       Ladder up — on a daily chart, W · D. The weekly rung is the whole
           composite on the weekly frame: the RV ensemble on the spread sampled
           at weekly closes, finished with today's spread as the forming week,
           and breadth at the last closed week. The tape is the ladder mean of
           z, NOT variance-restored — frames that disagree pull it toward zero.

The macro basket — what yfinance can carry
──────────────────────────────────────────
The Pine requests 21 TradingView series. yfinance carries the US yields, the
INR crosses, the dollar index and the commodity futures directly. Every other
10-year yield is proxied by that market's government-bond ETF, converted from
a price move to a yield move with its approximate modified duration, so it can
pool with real yields in the Pine's own units (percentage points):

        Δy ≈ −100 · Δln(P) / D

Non-US 2-year yields have no usable proxy and drop out of their pools — the
Pine's own rule: "a symbol your data plan does not carry drops out of its
factor instead of poisoning it". So the global curve factor is the US curve.

EXPANDED (the default here; `BASKETS`): Brent joins WTI in the energy pool
(India prices crude off Brent), copper enters as an industrial-growth factor,
and the name's HOME EQUITY INDEX enters as a factor — Nifty for NSE names, the
S&P 500 otherwise. That last one changes what value means: rich or cheap AFTER
what the market and the macro explain, which is the comparison a book across
many names needs. The Pine's own machinery polices the extra candidates: at
most three are ever used, the significance floor is Šidák-corrected for the
number offered, and the hedge is applied only as far as it has earned. Volatility
indices are deliberately absent — the Pine warns against hedging a target
against its own volatility.

DRIVER TIMING, as the Pine: a driver whose daily bar closes more than a third
of a day after the name's is read at its PREVIOUS close — the value known when
the name closed. US drivers against an NSE name are 10½ hours late and lag a
day; European and Japanese ones do not.

What is not carried: the ▲▼ ◆ signal machinery and everything that exists only
for it (the basket-warm gate, the arm/confirm state). Only the readings.

Author: @thebullishvalue
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Study-fixed constants (pragati.pine section 4c, each swept across 24 targets) ─
CORR_LEN = 250          # estimation sample, anchor bars
HOLD_OUT = 12           # selection hold-out, anchor bars
HYST_MARG = 0.10        # selection hysteresis
RIDGE_LAM = 0.05        # ridge shrinkage (scaled by 200/n below 200 obs)
MIN_RHO = 0.0           # economic floor; Fisher does the work
MAX_VARS = 3            # maximum model order
SKILL_WIN = 104         # anchor bars over which hedge skill is measured
SKILL_MIN = 20          # skill needs this many periods before it is trusted
MODEL_MIN = 10          # the model is "built" at this many observations
GAIN = 0.5              # display scaling; monotone
DRIFT_LEN = 400         # high-pass EMA on each ensemble member
Z_WINDOWS = (8, 13, 21, 34, 55)
ENS_SCALE = 0.887       # sd of the mean of five members at measured rho 0.734
B_WINDOWS = (10, 14, 20, 28, 40)
ROC_FRAC = 0.70
REGIME_SENS = 8.0
BASE_WEIGHT = 0.0
CLIP_Z = 3.0
VAR_CORR_LEN = 200      # window for the leg correlation
LEG_MIX = 0.5           # weight on the RV leg — Samanvaya's measured best
THETA = 1.5             # entry threshold θ, in |z|
DRIVER_LATE = 1.0 / 3.0
SELF_RHO = 0.99         # self-containment screen
SELF_WIN = 200
WEEKLY_BREADTH_MIN = 2 * B_WINDOWS[-1] + 20   # weekly bars before breadth is trusted


def softbound(x):
    """The Pine's f_softBound: x / (1 + |x|)."""
    return x / (1.0 + np.abs(x))


# θ on the tape's ±100 scale: the knee where value becomes RICH or CHEAP.
THETA_OSC = float(100.0 * softbound(THETA * GAIN))          # 42.857…

# ── Drivers: yfinance ticker, unit, modified duration (proxies), close (UTC h) ──
# `yield` is in percentage points; `price` becomes a log; `bond` is a bond-ETF
# PRICE converted to a yield move with its approximate modified duration.
DRIVERS: Dict[str, Tuple[str, str, Optional[float], float]] = {
    "US10Y":  ("^TNX",          "yield", None, 20.5),
    "US30Y":  ("^TYX",          "yield", None, 20.5),
    "US02Y":  ("ZT=F",          "bond",  1.9,  21.0),   # 2-year T-note futures
    "IN10Y":  ("SETF10GILT.NS", "bond",  6.8,  10.0),   # 10-year gilt ETF
    "JP10Y":  ("1482.T",        "bond",  9.5,  6.0),    # Japan govt bond ETF
    "CN10Y":  ("CBON",          "bond",  5.0,  20.5),   # China bond ETF (USD-listed)
    "EU10Y":  ("EXX6.DE",       "bond",  6.0,  15.5),   # German govt bond ETF
    "GB10Y":  ("IGLT.L",        "bond",  11.0, 15.5),   # UK gilt ETF
    "DXY":    ("DX-Y.NYB",      "price", None, 21.0),
    "USOIL":  ("CL=F",          "price", None, 21.0),
    "BRENT":  ("BZ=F",          "price", None, 21.0),
    "GOLD":   ("GC=F",          "price", None, 21.0),
    "SILVER": ("SI=F",          "price", None, 21.0),
    "COPPER": ("HG=F",          "price", None, 21.0),
    "USDINR": ("USDINR=X",      "price", None, 22.0),
    "EURINR": ("EURINR=X",      "price", None, 22.0),
    "GBPINR": ("GBPINR=X",      "price", None, 22.0),
    "JPYINR": ("JPYINR=X",      "price", None, 22.0),
    "NIFTY":  ("^NSEI",         "price", None, 10.0),
    "SPX":    ("^GSPC",         "price", None, 20.5),
}
DRIVER_TICKERS: List[str] = sorted({v[0] for v in DRIVERS.values()})

# Factor definitions: (name, kind, spec). kind "pool" averages its constituents'
# returns na-safely; "sub" is the difference of two; "pool_sub" pools differences;
# "home" is the name's home equity index.
_SAMANVAYA = [
    ("Global 10Y",   "pool",     ["US10Y", "IN10Y", "JP10Y", "CN10Y", "EU10Y", "GB10Y"]),
    ("Global curve", "pool_sub", [("US10Y", "US02Y")]),
    ("US rates",     "pool",     ["US02Y", "US10Y", "US30Y"]),
    ("US 30s2s",     "sub",      ("US30Y", "US02Y")),
    ("Dollar",       "pool",     ["DXY"]),
    ("Energy",       "pool",     ["USOIL"]),
    ("Precious",     "pool",     ["GOLD", "SILVER"]),
    ("INR basket",   "pool",     ["USDINR", "EURINR", "GBPINR", "JPYINR"]),
]
_EXPANDED = [
    f if f[0] != "Energy" else ("Energy", "pool", ["USOIL", "BRENT"]) for f in _SAMANVAYA
] + [
    ("Industrial metals", "pool", ["COPPER"]),
    ("Home market",       "home", None),
]
BASKETS = {"samanvaya": _SAMANVAYA, "expanded": _EXPANDED}
DEFAULT_BASKET = "expanded"

VALUE_COLUMNS = ("value tape", "value daily", "value hedge", "value drivers")


def _close_utc(symbol: str) -> float:
    """Approximate daily close of the name's own market, in UTC hours."""
    s = str(symbol).upper()
    if s.endswith(".NS") or s.endswith(".BO") or s in ("^NSEI", "^NSEBANK", "^BSESN", "^INDIAVIX"):
        return 10.0
    if s.endswith("=X"):
        return 22.0
    if s.endswith("-USD"):
        return 24.0
    if s.endswith("=F"):
        return 21.0
    return 20.5


def _home_driver(symbol: str) -> str:
    return "NIFTY" if _close_utc(symbol) == 10.0 else "SPX"


def _sidak_z(k: int) -> float:
    """Two-sided critical z at 5% family-wise over the k candidates offered."""
    from scipy.stats import norm
    a = 1.0 - (1.0 - 0.05) ** (1.0 / max(k, 1))
    return float(norm.ppf(1.0 - a / 2.0))


# ══════════════════════════════════════════════════════════════════════════════
#  1-3 · DATA, NATURAL UNITS, FACTORS
# ══════════════════════════════════════════════════════════════════════════════
def driver_returns(closes: Optional[pd.DataFrame], index: pd.DatetimeIndex,
                   symbol: str) -> Dict[str, pd.Series]:
    """Each driver's return on the name's own calendar, in natural units.

    Aligned by date with the Pine's DRIVER TIMING rule, then differenced once.
    Yields move in percentage points, prices in log returns, bond-ETF proxies
    in yield points via their duration.
    """
    out: Dict[str, pd.Series] = {}
    if closes is None or closes.empty:
        return out
    tgt_close = _close_utc(symbol)
    for key, (tk, kind, dur, close_utc) in DRIVERS.items():
        if tk not in closes.columns:
            continue
        s = pd.to_numeric(closes[tk], errors="coerce").dropna()
        if s.empty:
            continue
        if kind != "yield":
            s = s[s > 0]
        # Late by more than a third of a day: read at the PREVIOUS close, on the
        # driver's own calendar — close[1] of the bar matched to this date.
        if max(0.0, close_utc - tgt_close) / 24.0 > DRIVER_LATE:
            s = s.shift(1).dropna()
        s = s.reindex(s.index.union(index)).ffill().reindex(index)
        lvl = s if kind == "yield" else np.log(s)
        v = lvl.diff()
        if kind == "bond":
            v = -100.0 * v / float(dur)
        out[key] = v
    return out


def _self_screen(y: pd.Series, v: pd.Series, symbol: str, driver_ticker: str) -> pd.Series:
    """The Pine's SELF-CONTAINMENT screen on one constituent, as a latch.

    True from the first bar the constituent is identified as the target itself —
    by name, or by |ρ| > 0.99 of returns over 200 bars — and forever after.
    """
    if str(symbol).upper() == str(driver_ticker).upper():
        return pd.Series(True, index=v.index)
    c = y.rolling(SELF_WIN, min_periods=SELF_WIN).corr(v)
    return (c.abs() > SELF_RHO).cummax().fillna(False).astype(bool)


def build_factors(y: pd.Series, drivers: Dict[str, pd.Series], symbol: str,
                  basket: str = DEFAULT_BASKET) -> Tuple[pd.DataFrame, List[str]]:
    """Factor returns for one name: the basket's pools, after the self-screen."""
    spec = BASKETS[basket]
    kept: Dict[str, pd.Series] = {}
    for key, v in drivers.items():
        tk = DRIVERS[key][0]
        kept[key] = v.where(~_self_screen(y, v, symbol, tk))
    home = _home_driver(symbol)
    cols, names = {}, []
    for name, kind, parts in spec:
        if kind == "pool":
            ser = [kept[p] for p in parts if p in kept]
            f = pd.concat(ser, axis=1).mean(axis=1) if ser else pd.Series(np.nan, index=y.index)
        elif kind == "sub":
            a, b = parts
            f = (kept[a] - kept[b]) if a in kept and b in kept else pd.Series(np.nan, index=y.index)
        elif kind == "pool_sub":
            ser = [kept[a] - kept[b] for a, b in parts if a in kept and b in kept]
            f = pd.concat(ser, axis=1).mean(axis=1) if ser else pd.Series(np.nan, index=y.index)
        else:   # home
            f = kept.get(home, pd.Series(np.nan, index=y.index))
        cols[name] = f
        names.append(name if kind != "home" else f"Home market ({'Nifty' if home == 'NIFTY' else 'S&P 500'})")
    return pd.DataFrame(cols, index=y.index), names


# ══════════════════════════════════════════════════════════════════════════════
#  4-10 · ANCHOR SAMPLE, SELECTION, REGRESSION, HEDGE SKILL
# ══════════════════════════════════════════════════════════════════════════════
def _partial(rxy, rxz, ryz):
    d = np.sqrt(np.maximum(1e-12, (1.0 - rxz * rxz) * (1.0 - ryz * ryz)))
    return np.clip((rxy - rxz * ryz) / d, -1.0, 1.0)


def _hold(prev, raw, scores, ex_a, ex_b):
    if prev is not None and prev != raw and prev != ex_a and prev != ex_b:
        return raw if abs(scores[raw]) > abs(scores[prev]) * (1.0 + HYST_MARG) else prev
    return raw


def _argmax(scores, ex_a, ex_b):
    a = np.abs(scores).astype(float)
    if ex_a >= 0:
        a[ex_a] = -1.0
    if ex_b >= 0:
        a[ex_b] = -1.0
    return int(np.argmax(a))


def fit_residual(y: pd.Series, F: pd.DataFrame
                 ) -> Tuple[pd.Series, pd.Series, pd.Series, List[Tuple[int, ...]]]:
    """Sections 4-10 on a daily anchor: the out-of-sample residual path.

    Returns (resid, hedge weight, n obs in the fit, chosen driver indices per
    bar). Missing factor data is per-column, never per-row: a missing return is
    a zero, and a factor that never exists stays constant and is never chosen.
    """
    T, K = len(y), F.shape[1]
    Y = y.to_numpy(dtype=float)
    Fv = F.to_numpy(dtype=float)
    valid = np.isfinite(Y)
    O = np.column_stack([np.where(valid, Y, 0.0), np.nan_to_num(Fv, nan=0.0)])
    O[~valid] = 0.0
    M = K + 1

    # Observation sequence and cumulative moments over it. At bar t the buffer
    # holds the observations pushed before t, minus the 12 still in the FIFO.
    seq = np.flatnonzero(valid)
    Os = O[seq]
    CS = np.vstack([np.zeros((1, M)), np.cumsum(Os, axis=0)])
    CP = np.concatenate([np.zeros((1, M, M)),
                         np.cumsum(Os[:, :, None] * Os[:, None, :], axis=0)])
    pushed = np.concatenate([[0], np.cumsum(valid)[:-1]])
    hi = np.clip(pushed - HOLD_OUT, 0, len(seq))
    lo = np.clip(hi - CORR_LEN, 0, None)
    n = (hi - lo).astype(float)
    S = CS[hi] - CS[lo]
    SP = CP[hi] - CP[lo]
    diag = np.einsum("tii->ti", SP)
    num = n[:, None, None] * SP - S[:, :, None] * S[:, None, :]
    dvar = n[:, None] * diag - S * S
    den = np.sqrt(np.maximum(dvar, 0.0)[:, :, None] * np.maximum(dvar, 0.0)[:, None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where((n[:, None, None] > 2) & (den > 1e-12),
                        np.clip(num / np.where(den > 0, den, 1.0), -1.0, 1.0), 0.0)
        var = np.where(n[:, None] > 1, (diag - S * S / np.where(n > 0, n, 1.0)[:, None])
                       / np.where(n > 0, n, 1.0)[:, None], np.nan)
        sd = np.sqrt(np.maximum(var, 0.0))
        mean = np.where(n[:, None] > 0, S / np.where(n > 0, n, 1.0)[:, None], np.nan)

    z_crit = _sidak_z(K)
    y_hat = np.full(T, np.nan)
    chosen: List[Tuple[int, ...]] = []
    h1 = h2 = h3 = None
    for t in range(T):
        C = corr[t]
        nF = n[t]
        s1 = C[0, 1:]
        i1 = _hold(h1, _argmax(s1, -1, -1), s1, -1, -1)
        h1 = i1
        ry1 = s1[i1]
        rk1 = C[1:, i1 + 1]
        s2 = _partial(s1, ry1, rk1)
        s2[i1] = 0.0
        i2 = _hold(h2, _argmax(s2, i1, -1), s2, i1, -1)
        h2 = i2
        r12 = C[i1 + 1, i2 + 1]
        ry2 = s1[i2]
        ry21 = _partial(ry2, ry1, r12)
        rk2 = C[1:, i2 + 1]
        rk21 = _partial(rk2, rk1, r12)
        ryk1 = _partial(s1, ry1, rk1)
        s3 = _partial(ryk1, ry21, rk21)
        s3[i1] = 0.0
        s3[i2] = 0.0
        i3 = _hold(h3, _argmax(s3, i1, i2), s3, i1, i2)
        h3 = i3
        q2, q3 = abs(s2[i2]), abs(s3[i3])
        f2 = max(MIN_RHO, float(np.tanh(z_crit / np.sqrt(nF - 4.0))) if nF - 4.0 > 0 else 1.0)
        f3 = max(MIN_RHO, float(np.tanh(z_crit / np.sqrt(nF - 5.0))) if nF - 5.0 > 0 else 1.0)
        k_eff = 1
        if MAX_VARS >= 2 and q2 >= f2:
            k_eff = 2
            if MAX_VARS >= 3 and q3 >= f3:
                k_eff = 3
        chosen.append((i1, i2, i3)[:k_eff])
        if nF <= 0:
            continue

        # Section 8 · Gram-Schmidt in correlation space, with ridge.
        lam = RIDGE_LAM * max(1.0, 200.0 / max(nF, 1.0))
        r13 = C[i1 + 1, i3 + 1]
        r23 = C[i2 + 1, i3 + 1]
        ry3 = s1[i3]
        vu2 = max(0.0, 1.0 - r12 * r12)
        a32 = (r23 - r13 * r12) / vu2 if vu2 > 1e-10 else 0.0
        vu3 = max(0.0, 1.0 - r13 * r13 - a32 * a32 * vu2)
        c1 = ry1
        c2 = ry2 - r12 * ry1
        c3 = ry3 - r13 * ry1 - a32 * c2
        g1 = c1 / (1.0 + lam)
        g2 = c2 / (vu2 + lam) if k_eff >= 2 else 0.0
        g3 = c3 / (vu3 + lam) if k_eff >= 3 else 0.0
        w3 = g3
        w2 = g2 - a32 * g3
        w1 = g1 - r12 * g2 - (r13 - r12 * a32) * g3
        sdy = sd[t, 0]
        hs, ms = [], []
        for w_k, i_k in ((w1, i1), (w2, i2), (w3, i3)):
            s_k = sd[t, i_k + 1]
            hs.append(w_k * sdy / s_k if np.isfinite(s_k) and s_k > 1e-12 and np.isfinite(sdy) else 0.0)
            ms.append(mean[t, i_k + 1])
        icept = mean[t, 0] - sum(h * m for h, m in zip(hs, ms))
        # Section 9 · on a daily anchor the chart bar IS the period: no rescale.
        fx = [Fv[t, i] if np.isfinite(Fv[t, i]) else 0.0 for i in (i1, i2, i3)]
        y_hat[t] = icept + sum(h * f for h, f in zip(hs, fx))

    resid_full = Y - y_hat

    # Section 10 · THE HEDGE WEIGHS ITSELF: skill over the last 104 periods,
    # each residual measured with the coefficients in force at the time, read
    # through the PREVIOUS bar.
    ok = np.isfinite(resid_full) & np.isfinite(Y)
    e2 = pd.Series(np.where(ok, resid_full ** 2, np.nan), index=y.index).dropna()
    y2 = pd.Series(np.where(ok, Y ** 2, np.nan), index=y.index).dropna()
    se = e2.rolling(SKILL_WIN, min_periods=SKILL_MIN).sum()
    sy = y2.rolling(SKILL_WIN, min_periods=SKILL_MIN).sum()
    skill = (1.0 - se / sy).where(sy > 1e-18)
    skill = skill.reindex(y.index).ffill().shift(1)
    hedge = skill.clip(lower=0.0, upper=1.0).fillna(0.0)

    resid = pd.Series(np.where(np.isfinite(resid_full), Y - hedge.to_numpy() * (Y - resid_full), Y),
                      index=y.index)
    return resid.where(np.isfinite(Y)), hedge, pd.Series(n, index=y.index), chosen


# ══════════════════════════════════════════════════════════════════════════════
#  11-12 · THE SPREAD AND THE RV ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def rv_z(spread: pd.Series) -> pd.Series:
    """Five timescale members, each high-passed, averaged, rescaled by 0.887."""
    acc = 0.0
    for zl in Z_WINDOWS:
        sd = spread.rolling(zl).std(ddof=0)
        m = spread.rolling(zl).mean()
        z = ((spread - m) / sd).where(sd > 1e-14, 0.0).fillna(0.0)
        acc = acc + (z - _ema(z, DRIFT_LEN))
    return acc / len(Z_WINDOWS) / ENS_SCALE


# ══════════════════════════════════════════════════════════════════════════════
#  LEG 2 · BREADTH — Tattva's Swayam price-action half, seven MSF members
# ══════════════════════════════════════════════════════════════════════════════
def _sig(x, s):
    return 2.0 / (1.0 + np.exp(-x / s)) - 1.0


def _zc(src: pd.Series, win: int) -> pd.Series:
    m = src.rolling(win).mean().shift(1).fillna(0.0)
    sd = src.rolling(win).std(ddof=1).shift(1)
    z = ((src - m) / sd).where(sd.notna() & (sd != 0))
    return z.fillna(0.0).clip(-CLIP_Z, CLIP_Z)


def _msf(df: pd.DataFrame, win: int, roc: int, mask: int, tr: pd.Series) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"] if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    c5, c10 = c.shift(5), c.shift(10)
    roc_raw = (c - c.shift(roc)) / c.shift(roc)
    momentum = _sig(_zc(roc_raw, win), 1.5)
    if mask == 1:
        return _sig(momentum, 1.0)
    vol_ma = v.rolling(win).mean()
    vol_ratio = (v / vol_ma).where(vol_ma.notna() & (vol_ma != 0), 1.0)
    trend_slow = c.rolling(win).mean()
    structure = flow = 0.0
    if mask in (0, 2):
        vw_dir = (((h + l) / 2.0 - o) * vol_ratio).rolling(win).mean()
        vw_imp = ((c - c5) * vol_ratio).rolling(win).mean()
        micro = _sig(_zc(vw_dir - vw_imp, win), 1.5)
        atrv = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        ctz = (_zc(c.rolling(5).mean() - trend_slow, win) + _zc(c - 2.0 * c5 + c10, win)
               + _zc((c - c5) / atrv, win) + _zc(c - trend_slow, win)) / 2.0
        structure = (micro + _sig(ctz, 1.5)) / np.sqrt(2.0)
    if mask in (0, 3):
        mf = (h + l + c) / 3.0 * v
        up, dn = c > c.shift(1), c < c.shift(1)
        pos = mf.where(up, 0.0).rolling(win).mean()
        neg = mf.where(dn, 0.0).rolling(win).mean()
        tot = pos + neg
        accum = 2.0 * ((pos / tot).where(tot.notna() & (tot != 0), 0.5) - 0.5)
        pct = c.pct_change()
        reg = pd.Series(np.select([pct > 0.0033, pct < -0.0033], [1.0, -1.0], 0.0), index=c.index)
        cnt = reg.cumsum()
        regime = _sig(_zc(cnt - cnt.rolling(win).mean(), win), 1.5)
        flow = (accum + regime) / np.sqrt(2.0)
    msum = {0: momentum + structure + flow, 2: structure, 3: flow}[mask]
    return _sig(msum, 1.0)


def _unified_msf(msf: pd.Series) -> pd.Series:
    cl = np.abs(msf) ** REGIME_SENS
    w_ad = cl / (cl + 0.001)
    w_fin = 0.5 * BASE_WEIGHT + 0.5 * w_ad
    mmr = 0.5 * (1.0 - BASE_WEIGHT)
    return (w_fin / (w_fin + mmr) * msf).clip(-1.0, 1.0)


def breadth_z(df: pd.DataFrame) -> pd.Series:
    """The breadth leg: seven MSF members, mean × 10 — already unit variance."""
    c = df["close"]
    pc = c.shift(1)
    tr = (pd.concat([df["high"], pc], axis=1).max(axis=1)
          - pd.concat([df["low"], pc], axis=1).min(axis=1)).where(pc.notna(), df["high"] - df["low"])
    rb = [int(max(5.0, np.floor(ROC_FRAC * b + 0.5))) for b in B_WINDOWS]
    mem = [_unified_msf(_msf(df, b, r, 0, tr)) for b, r in zip(B_WINDOWS, rb)]
    mem.append(_unified_msf(_msf(df, B_WINDOWS[0], rb[0], 1, tr)))        # fastest · momentum
    flow = _unified_msf(_msf(df, B_WINDOWS[-1], rb[-1], 3, tr))           # slowest · flow
    v = df["volume"] if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    has_vol = ((v.notna() & (v != 0)).cumsum() / np.arange(1, len(df) + 1)) >= 0.5
    base = sum(mem)
    return ((base + flow.where(has_vol, 0.0)) * 10.0 / np.where(has_vol, 7.0, 6.0)).rename("breadth_z")


def _blend(rz: pd.Series, bz: pd.Series, rho: pd.Series) -> pd.Series:
    """Blend in z-space with the measured leg ρ, variance restored — once."""
    w_rv, w_br = LEG_MIX, 1.0 - LEG_MIX
    rc = rho.fillna(0.71).clip(-0.90, 0.99)
    mv = w_rv * w_rv + w_br * w_br + 2.0 * w_rv * w_br * rc
    both = (w_rv * rz + w_br * bz) * np.where(mv > 1e-9, 1.0 / np.sqrt(mv), 1.0)
    out = both.where(rz.notna() & bz.notna())
    if w_rv > 0:
        out = out.fillna(rz.where(bz.isna()))
    if w_br > 0:
        out = out.fillna(bz.where(rz.isna()))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MTF LADDER · the weekly rung, reconstructed on the daily bars
# ══════════════════════════════════════════════════════════════════════════════
def weekly_rv(spread: pd.Series, built: pd.Series) -> pd.Series:
    """The RV ensemble on the weekly frame, finished with today's spread.

    Settled samples are the spread at each closed week's last bar (where the
    model was built); each member's drift EMA advances once per closed week.
    The forming week's reading uses the current spread as its close — so on a
    week's final bar it lands exactly on the value that week then settles on.
    """
    key = pd.Series(spread.index.to_period("W-FRI"), index=spread.index)
    last = key != key.shift(-1)
    settle_mask = last & built
    # A week settles on the first bar of the NEXT week, so the final week in the
    # data is still forming.
    settle_mask.iloc[-1] = False
    smp = spread[settle_mask].to_numpy(dtype=float)
    smp_week = key[settle_mask].to_numpy()
    # How many settled samples each daily bar can see: every settled week before
    # its own.
    avail = np.searchsorted(smp_week, key.to_numpy(), side="left")
    cs1 = np.concatenate([[0.0], np.cumsum(smp)])
    cs2 = np.concatenate([[0.0], np.cumsum(smp * smp)])
    al = 2.0 / (DRIFT_LEN + 1.0)
    cur = spread.to_numpy(dtype=float)
    acc = np.zeros(len(cur))
    okk = np.ones(len(cur), dtype=bool)
    for zl in Z_WINDOWS:
        w = zl - 1
        # settled z of each sample j, window = samples j-zl+1 … j
        j = np.arange(len(smp))
        jj = j + 1
        lo = jj - zl
        valid_j = lo >= 0
        a1 = np.where(valid_j, cs1[jj] - cs1[np.clip(lo, 0, None)], np.nan)
        a2 = np.where(valid_j, cs2[jj] - cs2[np.clip(lo, 0, None)], np.nan)
        mu = a1 / zl
        sdv = np.sqrt(np.maximum(a2 / zl - mu * mu, 0.0))
        zs = np.where(sdv > 1e-14, (smp - mu) / np.where(sdv > 0, sdv, 1.0), 0.0)
        zs = np.where(valid_j, zs, np.nan)
        ema = pd.Series(zs).ewm(alpha=al, adjust=False, ignore_na=True).mean().to_numpy()
        ema = np.where(np.isfinite(pd.Series(zs).ffill().to_numpy()), ema, np.nan)
        # live z at each daily bar: last w settled samples + today's spread
        p = avail
        ok = p >= w
        s1 = np.where(ok, cs1[p] - cs1[np.clip(p - w, 0, None)], np.nan) + cur
        s2 = np.where(ok, cs2[p] - cs2[np.clip(p - w, 0, None)], np.nan) + cur * cur
        mu_l = s1 / zl
        sd_l = np.sqrt(np.maximum(s2 / zl - mu_l * mu_l, 0.0))
        zl_live = np.where(sd_l > 1e-14, (cur - mu_l) / np.where(sd_l > 0, sd_l, 1.0), 0.0)
        e_prev = np.where(p > 0, ema[np.clip(p - 1, 0, None)] if len(ema) else np.nan, np.nan)
        ek = np.where(np.isfinite(e_prev), al * zl_live + (1.0 - al) * e_prev, zl_live)
        good = ok & np.isfinite(zl_live)
        okk &= good
        acc = acc + np.where(good, zl_live - ek, 0.0)
    out = np.where(okk & built.to_numpy() & np.isfinite(cur), acc / len(Z_WINDOWS) / ENS_SCALE, np.nan)
    return pd.Series(out, index=spread.index)


def weekly_breadth(df: pd.DataFrame) -> pd.Series:
    """Breadth on the weekly frame, as of the last CLOSED week, per daily bar."""
    key = df.index.to_period("W-FRI")
    g = df.groupby(key)
    wk = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                       "low": g["low"].min(), "close": g["close"].last(),
                       "volume": g["volume"].sum(min_count=1) if "volume" in df.columns else np.nan})
    bz = breadth_z(wk)
    bz = bz.where(np.arange(len(bz)) > WEEKLY_BREADTH_MIN)
    prev = bz.shift(1).reindex(key)
    prev.index = df.index
    return prev


# ══════════════════════════════════════════════════════════════════════════════
#  THE VALUE READINGS FOR ONE NAME
# ══════════════════════════════════════════════════════════════════════════════
def compute_value(df: pd.DataFrame, driver_closes: Optional[pd.DataFrame], symbol: str,
                  basket: Optional[str] = None) -> pd.DataFrame:
    """The value tape and its parts for one name's daily OHLCV.

    Columns (VALUE_COLUMNS): the MTF value tape (±100, + rich), the chart's own
    value reading, the hedge weight applied, and the drivers in use. The tape is
    NaN until the model is built AND the weekly rung can calibrate — the Pine's
    tape stays grey until then, and a guessed value is not a neutral one.
    """
    out = pd.DataFrame(index=df.index, columns=list(VALUE_COLUMNS), dtype=object)
    if df is None or df.empty or not {"open", "high", "low", "close"}.issubset(df.columns):
        return out
    df = df.sort_index()
    df = df[df["close"].notna() & (df["close"] > 0)]
    if len(df) < max(Z_WINDOWS) + HOLD_OUT + MODEL_MIN + 2:
        return out.reindex(df.index)

    y = np.log(df["close"]).diff()
    F, names = build_factors(y, driver_returns(driver_closes, df.index, symbol), symbol,
                             basket or DEFAULT_BASKET)
    resid, hedge, n_obs, chosen = fit_residual(y, F)

    built = n_obs >= MODEL_MIN
    ready = built & resid.notna()
    spread = resid.where(ready, 0.0).fillna(0.0).cumsum()

    rz = rv_z(spread)
    bz = breadth_z(df)
    rho = rz.rolling(VAR_CORR_LEN, min_periods=VAR_CORR_LEN).corr(bz)
    unified = _blend(rz, bz, rho)
    value_daily = (100.0 * softbound(unified * GAIN)).clip(-100.0, 100.0).where(built)

    w_rv = weekly_rv(spread, built)
    w_bz = weekly_breadth(df)
    w_rho = w_rv.rolling(VAR_CORR_LEN, min_periods=VAR_CORR_LEN).corr(w_bz)
    w_u = _blend(w_rv, w_bz, w_rho)
    tape_z = (unified + w_u) / 2.0
    tape = (100.0 * softbound(tape_z * GAIN)).clip(-100.0, 100.0).where(built & w_u.notna())

    drv = [" · ".join(names[i] for i in c) if b else "" for c, b in zip(chosen, built)]
    res = pd.DataFrame({
        "value tape": tape.astype(float),
        "value daily": value_daily.astype(float),
        "value hedge": hedge.where(built).astype(float),
        "value drivers": pd.Series(drv, index=df.index).where(built),
    }, index=df.index)
    return res


__all__ = [
    "BASKETS",
    "DEFAULT_BASKET",
    "DRIVERS",
    "DRIVER_TICKERS",
    "THETA",
    "THETA_OSC",
    "VALUE_COLUMNS",
    "build_factors",
    "breadth_z",
    "compute_value",
    "driver_returns",
    "fit_residual",
    "rv_z",
    "softbound",
    "weekly_breadth",
    "weekly_rv",
]
