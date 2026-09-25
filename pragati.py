"""
PRAGYAM — Pragati, the conviction tape, ported from pragati.pine
══════════════════════════════════════════════════════════════════════════════

Pragati · प्रगति — progress: how much of each bar's travel became net
displacement, and how consistently, with how much participation behind it. "Is this move paid for — and is it paid for on
every timeframe?"

This module ports the indicator's CONVICTION side — the pane engine, its
multi-timeframe tape and its histogram. Its VALUE tape is Samanvaya's engine,
carried whole in samanvaya.py. The Conviction-Value Grid style reads both
(cvgrid.py). Nothing here reads a signal — no ▲▼, no ◆, no divergence.

The conviction tape (the Pine's own header, WHAT IT MEASURES)
─────────────────────────────────────────────────────────────
    conviction      c = (C - C[1]) / TR                 bounded -1 … +1
    participation   w = min(V / EMA(V), cap)            relative TR if no volume
    agreement       raw = 100 * Σ(c·w) / Σ(|c|·w)       over the lookback
    scaling         100 * tanh(raw / 3σ), σ over the normalization window
    tape            100 * tanh(mean z over the ladder D · W), EMA(3)

The weekly rung is RECONSTRUCTED the way the Pine rebuilds every higher frame:
the parent's settled state as of its last closed week, completed with the week
now forming. It lands exactly on the settled value at each week's close
(`reconstruction_error`) and sees nothing of the rest of the week.

The histogram
─────────────
The pane's primary read: the trace minus its signal EMA(9), taken on its three
channels exactly as section 9 draws it — hue (which way conviction is being
pushed), lightness (impulse, building, decelerating or turning, each with its
own magnitude gradient) and saturation (quiet regime). `push_reading` returns
the drawn intensity and the confirmation the grid's rows are gated on.

ADAPTED — the weekly rung's normalization window is 52 weeks, not the Pine's
200 rung-bars: at 200 a weekly rung needs four years of history, and this
panel carries about two and a half. The tape is read as a level once the chart
rung is calibrated, without the Pine's extra hold (which exists for its
signals). And the quiet-regime test ranks rawSd over up to 800 bars as the Pine
does, but over whatever history exists once there are 200 — the app's panel
cannot supply 800.

NAMING. The line runs Siddhi → Nishchaya → Dhṛti → Pragati. Pragati names what
the engine measures at its root — c = ΔC / TR, how much of a bar's travel became
progress — read across timeframes and against value. Published before as
Dhṛti; the pane is bit-identical to Nishchaya v3.

Author: @thebullishvalue
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── The Pine's defaults, section 1 · Conviction engine ────────────────────────
# None of these is fitted. The Pine's own finding across 900 randomised
# parameter sets: fitted edge and unseen edge correlate at about zero; lookback
# >= 20 and inner zone <= 40 were the only preferences that held.
LOOKBACK = 20            # inpLen   — bars of effort the reading accounts for
SMOOTHING = 3            # inpSm    — final EMA on the trace and the tape
NORMALIZATION = 200      # inpNorm  — sample behind the scaling σ
PART_BASELINE = 20       # inpVN    — EMA length of the participation baseline
PART_CAP = 3.0           # inpCap   — ceiling on the participation weight

# Weekly rung normalization — the one ADAPTED parameter (see module docstring).
NORMALIZATION_WEEKLY = 52

# Zones, section 6 — the conviction tape's shading knees.
INNER_ZONE = 30.0
OUTER_ZONE = 60.0

# The histogram, sections 5-6 and 9 — its reading exactly as the Pine draws it.
SIGNAL_LEN = 9           # inpSig   — the signal EMA; histogram = trace − signal
IMPULSE_K = 0.5          # inpK     — the knee of the brightest tier, k·σ(hist)
SLOPE_DEADBAND = 0.1     # a trace move inside 0.1σ of its bar-to-bar change is flat
QUIET_LEN = 800          # min(4 · inpNorm, 4999) — rawSd's own history, in bars
QUIET_MIN = 200          # ADAPTED: fewer than 800 bars → use what exists, ≥ 200
QUIET_PCT = 20.0         # quiet when rawSd sits in the bottom fifth of it
QUIET_KEEP = 0.45        # a quiet column is mixed 55% toward neutral
# Lightness per tier: the column's transparency at the bottom and top of its
# magnitude gradient (section 9, cImpulse … cTurning). Ink = 1 − T/100.
TIER_T = {"impulse": (18.0, 0.0), "building": (60.0, 38.0),
          "decelerating": (70.0, 50.0), "turning": (88.0, 72.0)}

CONVICTION_COLUMNS = ("conv tape", "conv daily", "conv weekly")
PUSH_COLUMNS = ("conv hist", "conv push", "conv push tier", "conv push gate")
COLUMNS = CONVICTION_COLUMNS + PUSH_COLUMNS


def _tanh(x):
    """The Pine's f_tanh: clamped to ±10 before exponentiating."""
    return np.tanh(np.clip(x, -10.0, 10.0))


def _ema(x: pd.Series, n: int) -> pd.Series:
    """Pine's ta.ema: alpha = 2/(n+1), seeded with the first value."""
    return x.ewm(span=n, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Pine's ta.tr(true): high - low on the first bar, gap-aware after it."""
    pc = close.shift(1)
    tr = pd.concat([high, pc], axis=1).max(axis=1) - pd.concat([low, pc], axis=1).min(axis=1)
    return tr.where(pc.notna(), high - low)


def _participation(tr: pd.Series, volume: pd.Series, cap: float = PART_CAP
                   ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Section 1 · participation, AUTO mode: volume where it exists, else TR.

    Returns (w, volLast, volAvg, trAvg). A hollow bar — holiday, half session,
    no print — carries the last good volume into the baseline rather than
    feeding it NaN; the Pine measured that without the carry volume weighting
    was live on only 73% of bars.
    """
    vol_ok = volume.notna() & (volume > 0)
    vol_last = volume.where(vol_ok).ffill()
    vol_avg = _ema(vol_last, PART_BASELINE)
    tr_avg = _ema(tr, PART_BASELINE)
    w_vol = (volume / vol_avg).where(vol_ok & vol_avg.notna() & (vol_avg > 1e-12))
    w_rng = (tr / tr_avg).where(tr_avg.notna() & (tr_avg > 1e-12), 1.0)
    w = w_vol.fillna(w_rng).fillna(1.0).clip(lower=0.0, upper=cap)
    return w, vol_last, vol_avg, tr_avg


def chart_rung(df: pd.DataFrame) -> pd.DataFrame:
    """Sections 1-2: the chart's own conviction.

    Columns: z (the tanh argument), trace (the smoothed pane trace), ready.
    `ready` is the Pine's sdOK: the σ window holds no warm-up zeros.
    """
    high, low, close = df["high"], df["low"], df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    tr = _true_range(high, low, close)
    disp = close - close.shift(1).fillna(close)
    conv = (disp / tr).where(tr > 1e-12, 0.0)

    w, _, _, _ = _participation(tr, volume)

    num = (conv * w).rolling(LOOKBACK).mean()
    den_a = (conv.abs() * w).rolling(LOOKBACK).mean()          # agreement: Σ|c|·w
    # Ratio of sums, and pinned to 0 while the window fills — a dead bar
    # contributes nothing to either side rather than an undefined ratio.
    raw = (100.0 * num / den_a).where(den_a > 1e-12, 0.0).fillna(0.0)

    raw_sd = raw.rolling(NORMALIZATION).std(ddof=0)            # ta.stdev is biased
    z = (raw / (3.0 * raw_sd)).where(raw_sd.notna() & (raw_sd >= 1e-9), 0.0)
    trace = _ema(100.0 * _tanh(z), SMOOTHING) if SMOOTHING > 1 else 100.0 * _tanh(z)

    bar = np.arange(len(df))
    ready = pd.Series(bar >= LOOKBACK + PART_BASELINE + NORMALIZATION, index=df.index)
    return pd.DataFrame({"z": z, "trace": trace, "ready": ready, "raw_sd": raw_sd})


def _percentrank_available(x: pd.Series, max_len: int, min_len: int) -> pd.Series:
    """Pine's ta.percentrank over up to `max_len` previous values, ADAPTED to
    use whatever history exists once there are at least `min_len` of them."""
    a = x.to_numpy(dtype=float)
    out = np.full(len(a), np.nan)
    for t in range(len(a)):
        if not np.isfinite(a[t]):
            continue
        prev = a[max(0, t - max_len):t]
        prev = prev[np.isfinite(prev)]
        if len(prev) >= min_len:
            out[t] = 100.0 * np.count_nonzero(prev <= a[t]) / len(prev)
    return pd.Series(out, index=x.index)


def _gradient_ink(mag: pd.Series, lo: pd.Series, hi: pd.Series, tier: str) -> pd.Series:
    """color.from_gradient on the column's transparency, read back as ink."""
    t_lo, t_hi = TIER_T[tier]
    span = (hi - lo).where((hi - lo) > 1e-12)
    f = ((mag - lo) / span).clip(0.0, 1.0).fillna(0.0)
    return 1.0 - (t_lo + (t_hi - t_lo) * f) / 100.0


def push_reading(ch: pd.DataFrame) -> pd.DataFrame:
    """The histogram, read on its three channels exactly as section 9 draws it.

        hue         which way conviction is being PUSHED — the trace's side of
                    its own signal line (not who controls: that is the tape)
        lightness   the tier — IMPULSE (growing, past k·σ), BUILDING (growing),
                    DECELERATING (shrinking, trace still with it) or TURNING
                    (shrinking, trace flat or against) — and, within the tier,
                    the column's magnitude
        saturation  whether the reading is real: in a QUIET regime (rawSd in the
                    bottom fifth of its own history) the column keeps 45% of
                    its colour

    `push` is the column as the eye receives it: direction × ink × saturation,
    in [−1, +1]. `gate` is what the state engine reads: +1 a confirmed push UP,
    −1 a confirmed push DOWN, 0 a push that cannot confirm (TURNING, or QUIET).
    All three are NaN until the histogram's own σ window is clean — the Pine
    draws those columns grey.
    """
    osc = ch["trace"]
    sig = _ema(osc, SIGNAL_LEN)
    hist = osc - sig

    # Readiness: the Pine's `ready` — σ(hist) over a window free of warm-up.
    n_osc = ch["ready"].astype(int).cumsum()
    ready = n_osc > NORMALIZATION + SMOOTHING + SIGNAL_LEN

    hist_sd = hist.rolling(NORMALIZATION).std(ddof=0)
    thr = IMPULSE_K * hist_sd.fillna(0.0)
    d_osc = osc - osc.shift(1).fillna(osc)
    slope_db = SLOPE_DEADBAND * d_osc.rolling(NORMALIZATION).std(ddof=0).fillna(0.0)

    h_prev = hist.shift(1).fillna(0.0)
    mag = hist.abs()
    above = hist >= 0.0
    expand = (above & (hist > h_prev)) | (~above & (hist < h_prev))
    with_tr = (above & (d_osc > slope_db)) | (~above & (d_osc < -slope_db))
    imp = mag >= thr
    h_hi = pd.concat([2.0 * hist_sd.fillna(0.0), 1.5 * thr,
                      pd.Series(1e-9, index=hist.index)], axis=1).max(axis=1)
    zero = pd.Series(0.0, index=hist.index)

    tier = pd.Series(np.select([expand & imp, expand, with_tr],
                               ["impulse", "building", "decelerating"], "turning"),
                     index=hist.index)
    ink = pd.Series(np.select(
        [tier == "impulse", tier == "building", tier == "decelerating"],
        [_gradient_ink(mag, thr, h_hi, "impulse"),
         _gradient_ink(mag, zero, thr.clip(lower=1e-9), "building"),
         _gradient_ink(mag, zero, h_hi, "decelerating")],
        _gradient_ink(mag, zero, h_hi, "turning")), index=hist.index)

    raw_pr = _percentrank_available(ch["raw_sd"], QUIET_LEN, QUIET_MIN)
    quiet = ready & raw_pr.notna() & (raw_pr < QUIET_PCT)
    push = (np.where(above, 1.0, -1.0) * ink * np.where(quiet, QUIET_KEEP, 1.0))
    label = (np.where(above, "up · ", "down · ") + tier
             + np.where(quiet, " · quiet", ""))
    live = (tier != "turning") & ~quiet
    gate = np.where(live, np.where(above, 1.0, -1.0), 0.0)
    return pd.DataFrame({
        "conv hist": hist.where(ready),
        "conv push": pd.Series(push, index=hist.index).where(ready),
        "conv push tier": pd.Series(label, index=hist.index).where(ready),
        "conv push gate": pd.Series(gate, index=hist.index).where(ready),
    }, index=hist.index)


def _weekly_bars(df: pd.DataFrame, key: pd.PeriodIndex) -> pd.DataFrame:
    """Settled weekly OHLCV, one row per calendar week present in `df`."""
    g = df.groupby(key)
    return pd.DataFrame({
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(min_count=1) if "volume" in df.columns else np.nan,
    })


def _parent_settled(wk: pd.DataFrame, norm: int) -> pd.DataFrame:
    """Section 4 · f_parentSettled on the weekly bars.

    Every running sum is over ONE BAR FEWER than its window, so the forming
    week can complete it. `z_settled` is the week's own finished z, kept for
    the reconstruction check.
    """
    tr = _true_range(wk["high"], wk["low"], wk["close"])
    conv = ((wk["close"] - wk["close"].shift(1).fillna(wk["close"])) / tr).where(tr > 1e-12, 0.0)
    w, vol_last, vol_avg, tr_avg = _participation(tr, wk["volume"])

    num = (conv * w).rolling(LOOKBACK).mean()
    den_a = (conv.abs() * w).rolling(LOOKBACK).mean()
    raw = (100.0 * num / den_a).where(den_a > 1e-12, 0.0).fillna(0.0)
    sd = raw.rolling(norm).std(ddof=0)
    z_settled = (raw / (3.0 * sd)).where(sd.notna() & (sd >= 1e-9), 0.0)

    s_num = (conv * w).rolling(LOOKBACK - 1).sum()
    s_da = (conv.abs() * w).rolling(LOOKBACK - 1).sum()
    s_r = raw.rolling(norm - 1).sum()
    s_r2 = (raw * raw).rolling(norm - 1).sum()
    # pN: bars since the window and the participation baseline filled.
    n_bars = pd.Series(np.maximum(np.arange(len(wk)) - (LOOKBACK + PART_BASELINE) + 1, 0),
                       index=wk.index)
    rdy = (n_bars > norm) & s_num.notna() & s_r.notna() & vol_avg.notna() & tr_avg.notna()
    return pd.DataFrame({
        "c_prev": wk["close"], "v_last": vol_last, "v_avg": vol_avg, "tr_avg": tr_avg,
        "s_num": s_num, "s_da": s_da, "s_r": s_r, "s_r2": s_r2,
        "z_settled": z_settled, "rdy": rdy,
    })


def weekly_rung(df: pd.DataFrame, norm: int = NORMALIZATION_WEEKLY
                ) -> tuple[pd.Series, pd.Series]:
    """Section 4 · the weekly parent, RECONSTRUCTED on the daily bars.

    Each daily bar reads the parent's state as of its last CLOSED week and
    completes it with the week now forming — the high, low and volume of this
    week's days so far and today's close. Nothing from the rest of the week is
    visible: this is the Pine's lookahead_on-on-the-previous-bar construction.

    Returns (z, recon_err): the rung's live z per daily bar (NaN until the
    parent can calibrate), and on each week's final daily bar the gap between
    the live reading and the value the week settled on, in trace points.
    """
    key = df.index.to_period("W-FRI")
    wk = _weekly_bars(df, key)
    st = _parent_settled(wk, norm)
    # The parent a daily bar sees is the PREVIOUS week's settled state.
    p = st.shift(1).reindex(key)
    p.index = df.index

    # f_agg — the forming week, from the daily bars since it opened.
    a_h = df["high"].groupby(key).cummax()
    a_l = df["low"].groupby(key).cummin()
    vol = df["volume"] if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    a_v = vol.fillna(0.0).groupby(key).cumsum()
    a_c = df["close"]

    # f_develop — the parent's arithmetic, finished with the forming bar.
    cp = p["c_prev"]
    tr_d = np.maximum(a_h, cp) - np.minimum(a_l, cp)
    c_d = ((a_c - cp) / tr_d).where(tr_d > 1e-12, 0.0)
    v_ok = a_v > 0
    alpha = 2.0 / (PART_BASELINE + 1)
    vl = a_v.where(v_ok, p["v_last"])
    v_a = (alpha * vl + (1.0 - alpha) * p["v_avg"]).where(vl.notna(), p["v_avg"])
    t_a = alpha * tr_d + (1.0 - alpha) * p["tr_avg"]
    w_v = (a_v / v_a).where(v_ok & v_a.notna() & (v_a > 1e-12))
    w_r = (tr_d / t_a).where(t_a.notna() & (t_a > 1e-12), 1.0)
    w_d = w_v.fillna(w_r).fillna(1.0).clip(lower=0.0, upper=PART_CAP)
    nm = (p["s_num"] + c_d * w_d) / LOOKBACK
    d_a = (p["s_da"] + c_d.abs() * w_d) / LOOKBACK
    rw = (100.0 * nm / d_a).where(d_a > 1e-12, 0.0)
    mu = (p["s_r"] + rw) / norm
    var = (p["s_r2"] + rw * rw) / norm - mu * mu
    sd = np.sqrt(var.clip(lower=0.0))
    z = (rw / (3.0 * sd)).where(sd >= 1e-9, 0.0)
    z = z.where(p["rdy"].astype("boolean").fillna(False).astype(bool))

    # RECONSTRUCTION CHECK — on a week's last daily bar the forming week IS the
    # settled week, so the live reading must land on the settled one.
    wk_of = pd.Series(key, index=df.index)
    last_of_week = wk_of != wk_of.shift(-1)
    settled_here = st["z_settled"].reindex(key)
    settled_here.index = df.index
    err = (100.0 * (_tanh(z) - _tanh(settled_here)).abs()).where(last_of_week & z.notna())
    return z, err


def compute_conviction(df: pd.DataFrame) -> pd.DataFrame:
    """The conviction tape and its two rungs for one name's daily OHLCV.

    Section 5 · the tape: every rung's z — the chart's included — averaged in
    linear space and compressed ONCE, with no second normalization, so
    agreement across frames is rarer than any one frame's reading. A rung
    whose history starts later joins when it can calibrate and moves the
    level, never the scale.
    """
    cols = list(CONVICTION_COLUMNS + PUSH_COLUMNS)
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return out
    df = df.sort_index()
    df = df[df["close"].notna() & df["high"].notna() & df["low"].notna()]
    if len(df) < LOOKBACK + PART_BASELINE + NORMALIZATION + 1:
        return out.reindex(df.index)

    ch = chart_rung(df)
    zw, _ = weekly_rung(df)
    ready = ch["ready"]
    z_chart = ch["z"].where(ready)
    z_lad = ((z_chart + zw.fillna(0.0)) / (ready.astype(float) + zw.notna().astype(float))).where(ready)
    tape = _ema(100.0 * _tanh(z_lad), SMOOTHING) if SMOOTHING > 1 else 100.0 * _tanh(z_lad)
    res = pd.DataFrame({
        "conv tape": tape.where(ready),
        "conv daily": ch["trace"].where(ready),
        "conv weekly": (100.0 * _tanh(zw)).where(zw.notna()),
    }, index=df.index)
    return pd.concat([res, push_reading(ch)], axis=1)[cols]


def reconstruction_error(df: pd.DataFrame) -> pd.Series:
    """The weekly rung's reconstruction error per settled week, in trace points.

    The Pine's Data Window row of the same name: how closely the reconstructed
    parent landed on the value the parent itself settled on. Should be ~0.
    """
    _, err = weekly_rung(df.sort_index())
    return err.dropna()

__all__ = [
    "COLUMNS",
    "CONVICTION_COLUMNS",
    "PUSH_COLUMNS",
    "LOOKBACK",
    "SMOOTHING",
    "NORMALIZATION",
    "NORMALIZATION_WEEKLY",
    "PART_BASELINE",
    "PART_CAP",
    "INNER_ZONE",
    "OUTER_ZONE",
    "chart_rung",
    "weekly_rung",
    "push_reading",
    "compute_conviction",
    "reconstruction_error",
]
