"""
PRAGYAM — two indicator-scored strategies: Volatility Surfer and Momentum Masters
══════════════════════════════════════════════════════════════════════════════

Ported from the strategy library `strategies.py` (VolatilitySurfer, MomentumMasters), which
scores a day's indicator snapshot and weights every name by its score. The scoring, the data
cleaning and the weight band are carried over as written; only the plumbing changed, so the
two run through Pragyam's shared pipeline like every other style (nco.compute_nco_portfolio:
the top N by weight, the 10% cap, whole shares).

    MOMENTUM MASTERS   trend following. A soft trend multiplier from price against its 90- and
                       200-day means (1.5 golden cross · 1.0 above both · 0.5 above the 200 ·
                       0.1 below), times 0.4 RSI strength + 0.3 oscillator momentum + 0.3
                       z-score velocity.
    VOLATILITY SURFER  breakout and expansion. Proximity to (or past) the upper 2σ Bollinger
                       band of the 20-day mean, a bonus for a tight band (a squeeze), times
                       oscillator / z-score confirmation and a soft trend multiplier (price
                       above its 90-week mean 1.0, above its 20-day mean 0.5, else 0.1).

Both add 1e-6 to every score ("fully inclusive"), weight by score, and pass the weights through
the source's band: clipped to [1%, 10%] and renormalised once. Every name therefore keeps at
least about 1%; the book fills the heaviest first.

Inputs are snapshot columns backdata already computes (rsi, osc and its 9 / 21 EMAs, zscore,
ma20 / ma90 / ma200, dev20). Nothing here reads a covariance or a price history.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# The columns each strategy reads (the source's `required_columns`, less symbol and price).
MOMENTUM_MASTERS_COLUMNS = ("rsi latest", "rsi weekly", "osc latest", "osc weekly",
                            "9ema osc latest", "21ema osc latest", "zscore latest",
                            "ma90 latest", "ma200 latest")
VOLATILITY_SURFER_COLUMNS = ("osc latest", "zscore latest", "ma90 weekly", "ma20 latest",
                             "dev20 latest")

# The source's weight band (BaseStrategy._allocate_portfolio): every weight clipped to
# [BAND_MIN, BAND_MAX], then renormalised.
BAND_MIN, BAND_MAX = 0.01, 0.10


def clean_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """The source's BaseStrategy._clean_data: RSI gaps read 50, an implausible moving average
    (missing, <= 0, or more than 10x away from price) reads the universe's median price, and
    every remaining gap or infinity reads 0."""
    out = df.copy()
    for col in ("rsi latest", "rsi weekly"):
        if col in out.columns:
            out[col] = out[col].fillna(50.0)
    fallback_ma = out["price"].median()
    for col in ("ma90 latest", "ma200 latest", "ma90 weekly", "ma200 weekly", "ma20 latest",
                "ma20 weekly"):
        if col in out.columns:
            invalid = ((out[col].isna()) | (out[col] <= 0) | (out[col] / out["price"] > 10)
                       | (out[col] / out["price"] < 0.1))
            out[col] = np.where(invalid, fallback_ma, out[col])
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def momentum_masters_score(df: pd.DataFrame) -> pd.Series:
    """MomentumMasters' composite score, as in the source."""
    conditions = [
        (df["price"] > df["ma90 latest"]) & (df["ma90 latest"] > df["ma200 latest"]),   # golden cross
        (df["price"] > df["ma90 latest"]) & (df["price"] > df["ma200 latest"]),         # strong uptrend
        (df["price"] > df["ma200 latest"]),                                             # general uptrend
    ]
    trend = np.select(conditions, [1.5, 1.0, 0.5], default=0.1)
    rsi_score = np.clip((df["rsi latest"] - 50) / 30, 0, 2.0)
    osc_score = ((df["osc latest"] > 20).astype(int) * 0.5
                 + (df["osc weekly"] > 0).astype(int) * 0.5
                 + (df["9ema osc latest"] > df["21ema osc latest"]).astype(int) * 1.0)
    velocity = np.clip(df["zscore latest"], 0, 3.0)
    return (rsi_score * 0.4 + osc_score * 0.3 + velocity * 0.3) * trend + 1e-6


def volatility_surfer_score(df: pd.DataFrame) -> pd.Series:
    """VolatilitySurfer's composite score, as in the source."""
    trend = np.select([df["price"] > df["ma90 weekly"], df["price"] > df["ma20 latest"]],
                      [1.0, 0.5], default=0.1)
    upper_band = df["ma20 latest"] + 2 * df["dev20 latest"]
    proximity = (df["price"] - upper_band) / (df["ma20 latest"] + 1e-6)
    breakout = np.clip(1 + (proximity * 10), 0, 3.0)
    band_width = (2 * 2 * df["dev20 latest"]) / (df["ma20 latest"] + 1e-6)
    squeeze = np.select([band_width < 0.05, band_width < 0.10], [1.5, 1.2], default=1.0)
    confirmation = (np.clip(df["osc latest"] / 50, 0, 2.0) * 0.5
                    + np.clip(df["zscore latest"], 0, 3.0) * 0.5)
    return breakout * squeeze * confirmation * trend + 1e-6


SCORERS = {"MMAST": (momentum_masters_score, MOMENTUM_MASTERS_COLUMNS),
           "VSURF": (volatility_surfer_score, VOLATILITY_SURFER_COLUMNS)}


def band_weights(score: pd.Series) -> pd.Series:
    """Score-proportional weights through the source's band: clip to [1%, 10%], renormalise.

    The source loops this up to ten times but stops as soon as the weights sum to 1, which they
    do after the first renormalisation — so it is one clip and one renormalisation, and a weight
    can end slightly outside the band. The pipeline's cap then holds every holding at 10%.
    """
    total = float(score.sum())
    w = score / total if total > 0 else pd.Series(1.0 / len(score), index=score.index)
    for _ in range(10):
        w = w.clip(lower=BAND_MIN, upper=BAND_MAX)
        w = w / w.sum()
        if abs(w.sum() - 1.0) < 1e-6:
            break
    return w


def strategy_scores(history: Sequence[Tuple[object, pd.DataFrame]], symbols: List[str],
                    method: str) -> pd.DataFrame:
    """Each symbol's inputs and composite score as of the LAST snapshot in `history`.

    A symbol without a snapshot row has no price to clean against and is scored on zeros, as the
    source scores a name whose indicators are missing.
    """
    scorer, cols = SCORERS[method]
    frame = pd.DataFrame(index=list(symbols), columns=["price", *cols], dtype=float)
    if history:
        snap = history[-1][1]
        if snap is not None and not snap.empty and "symbol" in snap.columns:
            snap = snap.drop_duplicates("symbol", keep="last").set_index("symbol")
            for col in ["price", *cols]:
                if col in snap.columns:
                    frame[col] = pd.to_numeric(snap[col], errors="coerce").reindex(frame.index)
    clean = clean_snapshot(frame)
    clean["score"] = np.asarray(scorer(clean), dtype=float)
    return clean


def strategy_weights(history: Sequence[Tuple[object, pd.DataFrame]], symbols: List[str],
                     method: str) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """(weights, fill order, the scored inputs): the band weights, ordered by composite score
    (highest first; ties keep the universe's order), so a book smaller than the universe holds
    the highest-scoring names."""
    scored = strategy_scores(history, symbols, method)
    order = list(scored["score"].sort_values(ascending=False, kind="stable").index)
    w = band_weights(scored["score"].reindex(order))
    return w.to_numpy(dtype=float), order, scored.reindex(order)


__all__ = [
    "BAND_MIN",
    "BAND_MAX",
    "MOMENTUM_MASTERS_COLUMNS",
    "VOLATILITY_SURFER_COLUMNS",
    "SCORERS",
    "clean_snapshot",
    "momentum_masters_score",
    "volatility_surfer_score",
    "band_weights",
    "strategy_scores",
    "strategy_weights",
]
