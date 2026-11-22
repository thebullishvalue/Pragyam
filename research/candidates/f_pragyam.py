"""
research/candidates/f_pragyam.py — FAMILY F: PRAGYAM'S RISK ENGINE AND VAAYDO'S BREADTH (v14 style search)

PRE-REGISTRATION (written 2026-11-22, committed before any configuration below was run on any data)
═══════════════════════════════════════════════════════════════════════════════════════════════════

WHERE THESE COME FROM
    pragyam.pine (thebullishvalue/Indicator, v2.6) and its reference implementation, vendored verbatim as
    research/pragyam_engine.py and run per symbol by research/pragyam_panels.py. Its own research, on 151
    instruments 2001-2026: no single-chart state forecasts DIRECTION in all three eras; RISK is forecast
    (next-10-bar variance, out of sample log R² 0.71; QLIKE 0.354 against 0.509 for a 20-bar standard
    deviation). So every candidate here is a RISK allocator or a reading of the engine's tapes, never a
    return forecast. The engine's outputs used, each as of the rebalance close:
        σ̂      sig    per-bar risk forecast: HAR blend of Garman-Klass variance (5 / 22 / 252-bar EMAs)
        T      T      trend in σ: the mean of the 63-, 126- and 252-bar moves, each over σ̂·√j
        dialR  dialR  min(1, own average σ̂ to date / σ̂ now): the exposure dial before its trend throttle
    VAAYDO (thebullishvalue/Vaaydo, v7.4.2) carries the same engine and, for the market, NSE breadth
    (advance / decline, 2006 on). Its own edge — option positioning (put / call lean) — exists in its data
    only from 2024-07-08: 27 months, too short for this protocol's eras, so it is not tested here (stated,
    not hidden). Its NSE breadth and Nifty valuation series are India-only and could not run on the Dow, so
    F5 reads breadth from each universe's own names instead.

CANDIDATES (5 functions, 14 configurations; the declared default marked *)
    F1 forecast_risk_parity(p)   wᵢ ∝ σ̂ᵢ^(−p). Inverse-risk weighting on Pragyam's forecast in place of a
                                 sample standard deviation (the 36-candidate search's inverse-vol used the
                                 sample). p ∈ {0.5, 1*, 2}.
    F2 forecast_erc(window)      Equal risk contribution (nco.erc_weights) on Σ = D R D: D = diag(σ̂), R the
                                 correlation of the Ledoit-Wolf covariance (nco.ledoit_wolf) of the last
                                 `window` daily returns. The diagonal is the forecast; only the correlation
                                 is estimated from history. window ∈ {126, 252*}.
    F3 exposure_dial(floor)      wᵢ ∝ dialRᵢ · g(Tᵢ), g(T) = floor + (1 − floor)(½ + ½ tanh T): the engine's
                                 exposure dial, cross-sectionally (its 0.10 band is not applied: the book is
                                 rebalanced monthly). floor ∈ {0.25, 0.5*, 0.75}; 0.5 is the engine's own.
    F4 trend_tape(floor)         wᵢ ∝ g(Tᵢ): the trend tape's throttle alone, a risk-adjusted three-horizon
                                 trend tilt. floor ∈ {0.25, 0.5*, 0.75}.
    F5 breadth_switch(thr)       Vaaydo's breadth reading on Pragyam's tape: b = the share of the priced names
                                 whose trend tape is up (Tᵢ > 0). b ≥ thr → Equal Weight; b < thr → F1 (p = 1).
                                 thr ∈ {0.4, 0.5*, 0.6}.

FIXED DESIGN (declared, not tuned)
    * Readings are the engine's values on the rebalance date, carried over a gap of at most 5 sessions as
      the harness carries prices; a priced name with no reading (fewer than 252 bars of its own) takes the
      cross-sectional median of that reading — neutral, never guessed high or low.
    * F2's correlation set: the priced names with ≥ 95% of the window (nco.MIN_COVERAGE), rows with any gap
      dropped; a priced name outside it is held at 1/N and the rest share the remaining mass. Fewer than
      10 names in the set: the whole book is 1/N that month.
    * Weights are raw; the harness renormalises and applies the 10% cap and (for top-N) the cut.

PRE-STATED EXPECTATION
    F1, F2 tilt to low risk: the known tension says they win Nifty E1 (the crash era) and lose Dow E1 /
    Nifty E2. F3 and F4 tilt to names in uptrends, a momentum-like bet: expect the reverse, and
    survivorship flattery on today's constituents (the point-in-time Dow is the check). F5 switches
    between the two ends on breadth; the 2007-26 sample has few breadth troughs, so its edge, if any,
    rests on few events.

RESULT (2026-11-22; tables in research/style_search_v14.py) — ALL FIVE REJECTED.
    Discovery: no configuration beat the best existing style in more than one of the four cells. Every
    configuration lost Nifty E1 (the 2008 crash era, HRP's cell) by 1.35 to 3.90 %/yr: the forecast-risk
    books (F1, F2) are less defensive than HRP's bisection there, and F3 / F4 lean toward recent strength,
    which that era punished. Outside Nifty E1, F3 and F4 sit within ±0.7 %/yr of the best existing style
    in every cell (their monthly returns 0.999-1.000 correlated with 1/N's); F1, F2 and F5 trail it. Largest full-span t
    0.20 (exposure_dial). Pragyam's risk engine forecasts risk well on one chart; across 30-50 names whose
    risks move together, a better per-name forecast barely changes the book.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # research/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root

import nco                                                   # noqa: E402
import pragyam_panels as pp                                  # noqa: E402

ENGINE_KEY = {"nifty_50": "nifty_50", "dow_30": "dow_35", "dow_pit": "dow_35", "etf_27": "etf_book"}
MIN_SET = 10
_ALIGNED: dict = {}


def _aligned(ctx, name: str) -> pd.DataFrame:
    """The engine panel `name` on the universe's own calendar, carried over gaps of ≤ 5 sessions."""
    d = ctx._d
    key = (d["u"], name, len(d["px"].index))
    if key not in _ALIGNED:
        P = pp.load(ENGINE_KEY[d["u"]])[name]
        cal = d["px"].index
        _ALIGNED[key] = P.reindex(P.index.union(cal)).ffill(limit=5).reindex(cal)
    return _ALIGNED[key]


def reading(ctx, name: str) -> pd.Series:
    """The engine's `name` for every priced name on the rebalance date; no reading → the median."""
    v = _aligned(ctx, name).loc[ctx.date].reindex(ctx.priced).astype(float)
    v = v.where(np.isfinite(v))
    return v.fillna(v.median()) if v.notna().any() else pd.Series(np.nan, index=ctx.priced)


def throttle(T: pd.Series, floor: float) -> pd.Series:
    return floor + (1.0 - floor) * (0.5 + 0.5 * np.tanh(T))


def returns_set(ctx, window: int):
    """(names with ≥ 95% of the window, their gap-free daily returns, names held at 1/N outside it)."""
    px = ctx.px.reindex(columns=ctx.priced).tail(window + 1)
    r = px.pct_change(fill_method=None).iloc[1:]
    keep = [c for c in r.columns if r[c].notna().sum() >= nco.MIN_COVERAGE * len(r)]
    R = r[keep].dropna(how="any")
    out = [c for c in ctx.priced if c not in keep]
    return keep, R, out


def assemble(ctx, keep: list, w_keep: np.ndarray, out: list) -> pd.Series:
    """Names outside the estimation set at 1/N each; the set shares what is left, by w_keep."""
    n = len(ctx.priced)
    w = pd.Series(0.0, index=ctx.priced)
    w[out] = 1.0 / n
    wk = np.clip(np.nan_to_num(np.asarray(w_keep, dtype=float)), 0.0, None)
    w[keep] = (1.0 - len(out) / n) * (wk / wk.sum() if wk.sum() > 0 else np.full(len(keep), 1.0 / len(keep)))
    return w


def equal(ctx) -> pd.Series:
    return pd.Series(1.0 / len(ctx.priced), index=ctx.priced)


# ── F1 ─────────────────────────────────────────────────────────────────────────────────────────────
def forecast_risk_parity(ctx, p: float = 1.0) -> pd.Series:
    s = reading(ctx, "sig")
    if not s.notna().any():
        return equal(ctx)
    return s.clip(lower=1e-6) ** (-p)


# ── F2 ─────────────────────────────────────────────────────────────────────────────────────────────
def forecast_erc(ctx, window: int = 252) -> pd.Series:
    keep, R, out = returns_set(ctx, window)
    if len(keep) < MIN_SET or len(R) < max(nco.MIN_OBS, len(keep)):
        return equal(ctx)
    S = nco.ledoit_wolf(R.to_numpy(dtype=float))
    sd = np.sqrt(np.clip(np.diag(S), 1e-20, None))
    C = S / np.outer(sd, sd)
    sig = reading(ctx, "sig").reindex(keep).to_numpy(dtype=float)
    cov = C * np.outer(sig, sig)
    return assemble(ctx, keep, nco.erc_weights(cov), out)


# ── F3 ─────────────────────────────────────────────────────────────────────────────────────────────
def exposure_dial(ctx, floor: float = 0.5) -> pd.Series:
    dial, T = reading(ctx, "dialR"), reading(ctx, "T")
    if not (dial.notna().any() and T.notna().any()):
        return equal(ctx)
    return dial * throttle(T, floor)


# ── F4 ─────────────────────────────────────────────────────────────────────────────────────────────
def trend_tape(ctx, floor: float = 0.5) -> pd.Series:
    T = reading(ctx, "T")
    if not T.notna().any():
        return equal(ctx)
    return throttle(T, floor)


# ── F5 ─────────────────────────────────────────────────────────────────────────────────────────────
def breadth(ctx) -> float:
    T = _aligned(ctx, "T").loc[ctx.date].reindex(ctx.priced)
    T = T[np.isfinite(T)]
    return float((T > 0).mean()) if len(T) else float("nan")


def breadth_switch(ctx, thr: float = 0.5) -> pd.Series:
    b = breadth(ctx)
    if not np.isfinite(b) or b >= thr:
        return equal(ctx)
    return forecast_risk_parity(ctx, 1.0)


CANDIDATES = {
    "forecast_risk_parity": (forecast_risk_parity, {"p": [0.5, 1.0, 2.0]}),
    "forecast_erc": (forecast_erc, {"window": [126, 252]}),
    "exposure_dial": (exposure_dial, {"floor": [0.25, 0.5, 0.75]}),
    "trend_tape": (trend_tape, {"floor": [0.25, 0.5, 0.75]}),
    "breadth_switch": (breadth_switch, {"thr": [0.4, 0.5, 0.6]}),
}
DEFAULTS = {"forecast_risk_parity": {"p": 1.0}, "forecast_erc": {"window": 252},
            "exposure_dial": {"floor": 0.5}, "trend_tape": {"floor": 0.5}, "breadth_switch": {"thr": 0.5}}
