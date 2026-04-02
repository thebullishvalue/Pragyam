"""
PRAGYAM — Market Regime Intelligence
══════════════════════════════════════════════════════════════════════════════

Institutional-grade market regime detection with multi-factor composite scoring.

7-Factor Composite Scoring Architecture:
  1. Momentum    (30%) — RSI trajectory + oscillator direction
  2. Trend       (25%) — Price/MA alignment + pct stocks above 200DMA
  3. Breadth     (15%) — Cross-sectional RSI/oscillator distribution
  4. Velocity    (15%) — dRSI/dt (first derivative) + d²RSI/dt² (acceleration)
  5. Extremes    (10%) — Z-score distribution: oversold / overbought crowding
  6. Volatility   (5%) — Bollinger Band Width regime (squeeze → panic)
  7. Correlation  (0%) — Herding proxy (diagnostic only, not scored)

Regime Hierarchy (composite score, descending):
  STRONG_BULL (≥1.50) → BULL (≥1.00) → WEAK_BULL (≥0.50)
  → CHOP (≥0.10) → WEAK_BEAR (≥−0.10) → BEAR (≥−0.50) → CRISIS (<−0.50)

Author: Hemrek Capital
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — REGIME COLOURS / ICONS / LABELS
# ══════════════════════════════════════════════════════════════════════════════

REGIME_COLORS: Dict[str, str] = {
    "STRONG_BULL": "#10b981",  # Emerald
    "BULL":        "#34d399",  # Light emerald
    "WEAK_BULL":   "#a3e635",  # Lime
    "CHOP":        "#f59e0b",  # Amber
    "WEAK_BEAR":   "#fb923c",  # Orange
    "BEAR":        "#ef4444",  # Red
    "CRISIS":      "#dc2626",  # Deep red
    "UNKNOWN":     "#6b7280",  # Gray
}

REGIME_ICONS: Dict[str, str] = {
    "STRONG_BULL": "🚀", "BULL": "📈", "WEAK_BULL": "↗️",
    "CHOP": "↔️", "WEAK_BEAR": "↘️", "BEAR": "📉",
    "CRISIS": "⚡", "UNKNOWN": "❓",
}

REGIME_DESCRIPTIONS: Dict[str, str] = {
    "STRONG_BULL": "Dominant uptrend with broad participation and accelerating momentum. Full momentum allocation.",
    "BULL":        "Healthy uptrend with positive breadth. Momentum and trend-following strategies favored.",
    "WEAK_BULL":   "Uptrend showing divergence or waning momentum. Selective momentum with defensive overlay.",
    "CHOP":        "Directionless market with no clear bias. Mean-reversion and relative-value strategies preferred.",
    "WEAK_BEAR":   "Downtrend developing with deteriorating breadth. Defensive positioning, reduce exposure.",
    "BEAR":        "Established downtrend with weak breadth and negative momentum. Primarily defensive.",
    "CRISIS":      "Severe market stress with panic volatility and capitulation breadth. Maximum capital preservation.",
    "UNKNOWN":     "Insufficient data to classify market regime reliably.",
}

REGIME_MIX_MAP: Dict[str, str] = {
    "STRONG_BULL": "Bull Market Mix", "BULL": "Bull Market Mix",
    "WEAK_BULL":   "Chop/Consolidate Mix", "CHOP": "Chop/Consolidate Mix",
    "WEAK_BEAR":   "Chop/Consolidate Mix", "BEAR": "Bear Market Mix",
    "CRISIS":      "Bear Market Mix", "UNKNOWN": "Chop/Consolidate Mix",
}

# Score normalisation helpers for factor bars in the UI
# Score range per factor: [-2, +2]
FACTOR_SCORE_RANGE = (-2.0, 2.0)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FactorScores:
    """Breakdown of all 7 regime detection factors."""
    momentum:    Dict[str, Any] = field(default_factory=dict)
    trend:       Dict[str, Any] = field(default_factory=dict)
    breadth:     Dict[str, Any] = field(default_factory=dict)
    volatility:  Dict[str, Any] = field(default_factory=dict)
    extremes:    Dict[str, Any] = field(default_factory=dict)
    correlation: Dict[str, Any] = field(default_factory=dict)
    velocity:    Dict[str, Any] = field(default_factory=dict)

    _WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "momentum": 0.30, "trend": 0.25, "breadth": 0.15,
        "volatility": 0.05, "extremes": 0.10, "correlation": 0.00, "velocity": 0.15,
    }, repr=False)

    def composite_score(self) -> float:
        weights = {"momentum": 0.30, "trend": 0.25, "breadth": 0.15,
                   "volatility": 0.05, "extremes": 0.10, "correlation": 0.00, "velocity": 0.15}
        total = 0.0
        for name, w in weights.items():
            factor_dict = getattr(self, name)
            total += factor_dict.get("score", 0.0) * w
        return round(total, 3)

    def to_display_list(self) -> List[Tuple[str, float, str, float]]:
        """Return [(factor_name, score, label, weight)] for UI rendering."""
        weights = {"Momentum": 0.30, "Trend": 0.25, "Breadth": 0.15,
                   "Velocity": 0.15, "Extremes": 0.10, "Volatility": 0.05}
        mapping = {
            "Momentum": (self.momentum, "strength"),
            "Trend":    (self.trend, "quality"),
            "Breadth":  (self.breadth, "quality"),
            "Velocity": (self.velocity, "acceleration"),
            "Extremes": (self.extremes, "type"),
            "Volatility": (self.volatility, "regime"),
        }
        result = []
        for fname, (fdict, label_key) in mapping.items():
            result.append((
                fname,
                float(fdict.get("score", 0.0)),
                str(fdict.get(label_key, "—")),
                weights.get(fname, 0.0),
            ))
        return result


@dataclass
class RegimeResult:
    """Complete result of regime detection for a single date."""
    date: datetime
    regime: str
    mix_name: str
    confidence: float
    composite_score: float
    factors: FactorScores
    explanation: str

    # ── computed on creation ─────────────────────────────────────────────────
    color: str = field(init=False)
    icon: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.color = REGIME_COLORS.get(self.regime, REGIME_COLORS["UNKNOWN"])
        self.icon  = REGIME_ICONS.get(self.regime, "❓")
        self.description = REGIME_DESCRIPTIONS.get(self.regime, "")

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable dict for st.session_state / st.cache_data."""
        return {
            "date": self.date.isoformat(),
            "regime": self.regime,
            "mix_name": self.mix_name,
            "confidence": self.confidence,
            "composite_score": self.composite_score,
            "color": self.color,
            "icon": self.icon,
            "description": self.description,
            "explanation": self.explanation,
            "factors": {
                "momentum":    self.factors.momentum,
                "trend":       self.factors.trend,
                "breadth":     self.factors.breadth,
                "volatility":  self.factors.volatility,
                "extremes":    self.factors.extremes,
                "correlation": self.factors.correlation,
                "velocity":    self.factors.velocity,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeResult":
        """Reconstruct from to_dict() output."""
        factors = FactorScores(
            momentum=d["factors"]["momentum"],
            trend=d["factors"]["trend"],
            breadth=d["factors"]["breadth"],
            volatility=d["factors"]["volatility"],
            extremes=d["factors"]["extremes"],
            correlation=d["factors"]["correlation"],
            velocity=d["factors"]["velocity"],
        )
        return cls(
            date=datetime.fromisoformat(d["date"]),
            regime=d["regime"],
            mix_name=d["mix_name"],
            confidence=d["confidence"],
            composite_score=d["composite_score"],
            factors=factors,
            explanation=d["explanation"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class MarketRegimeDetector:
    """
    Institutional-grade market regime detector.

    Uses a 7-factor composite score. Each factor contributes a signal score
    in the range [−2, +2]; the composite is the weighted sum of all factors.

    Reference architecture:
    - Momentum / Trend:  Jegadeesh & Titman (1993), Fama & French (1988)
    - Breadth:           Zweig Breadth Thrust (1986)
    - Velocity:          Hamilton (1989) regime-switching; first/second derivatives
    - Volatility:        Bollinger (1983) Band Width
    """

    _THRESHOLDS: List[Tuple[str, float, float]] = [
        # (regime, min_score, base_confidence)
        ("STRONG_BULL", 1.50, 0.85),
        ("BULL",        1.00, 0.75),
        ("WEAK_BULL",   0.50, 0.65),
        ("CHOP",        0.10, 0.60),
        ("WEAK_BEAR",  -0.10, 0.65),
        ("BEAR",       -0.50, 0.75),
        ("CRISIS",     -9.99, 0.85),  # catch-all floor
    ]

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(
        self,
        historical_data: List[Tuple[datetime, pd.DataFrame]],
        analysis_date: Optional[datetime] = None,
    ) -> RegimeResult:
        """
        Detect market regime from a list of (date, DataFrame) historical snapshots.

        Args:
            historical_data: Chronologically ordered list of (date, indicator_df) tuples.
                             Minimum 5 entries; 10 recommended for meaningful classification.
            analysis_date:   Override the result timestamp (defaults to last entry's date).

        Returns:
            RegimeResult with complete factor breakdown and composite classification.
        """
        if len(historical_data) < 5:
            date = analysis_date or datetime.now()
            return RegimeResult(
                date=date, regime="UNKNOWN", mix_name="Chop/Consolidate Mix",
                confidence=0.30, composite_score=0.0, factors=FactorScores(),
                explanation="Insufficient data (< 5 periods) for regime classification.",
            )

        window = historical_data[-min(10, len(historical_data)):]
        last_date, latest_df = window[-1]
        result_date = analysis_date or last_date

        factors = FactorScores(
            momentum=self._momentum(window),
            trend=self._trend(window),
            breadth=self._breadth(latest_df),
            volatility=self._volatility(window),
            extremes=self._extremes(latest_df),
            correlation=self._correlation(latest_df),
            velocity=self._velocity(window),
        )

        score = factors.composite_score()
        regime, confidence = self._classify(score, factors)

        # ── Crisis override: panic vol + capitulation breadth ───────────────
        if (
            factors.volatility.get("regime") == "PANIC"
            and score < -0.5
            and factors.breadth.get("quality") == "CAPITULATION"
        ):
            regime, confidence = "CRISIS", 0.92

        explanation = self._explain(regime, confidence, factors, score)

        return RegimeResult(
            date=result_date,
            regime=regime,
            mix_name=REGIME_MIX_MAP.get(regime, "Chop/Consolidate Mix"),
            confidence=confidence,
            composite_score=score,
            factors=factors,
            explanation=explanation,
        )

    # ── Factor computation ───────────────────────────────────────────────────

    def _momentum(self, window: list) -> Dict[str, Any]:
        rsi_vals = [df["rsi latest"].mean() for _, df in window]
        osc_vals = [df["osc latest"].mean() for _, df in window]
        cur_rsi = float(rsi_vals[-1])
        rsi_trend = float(np.polyfit(range(len(rsi_vals)), rsi_vals, 1)[0])
        cur_osc = float(osc_vals[-1])
        osc_trend = float(np.polyfit(range(len(osc_vals)), osc_vals, 1)[0])

        if cur_rsi > 65 and rsi_trend > 0.5:    strength, score = "STRONG_BULLISH", 2.0
        elif cur_rsi > 55 and rsi_trend >= 0:    strength, score = "BULLISH", 1.0
        elif cur_rsi < 35 and rsi_trend < -0.5:  strength, score = "STRONG_BEARISH", -2.0
        elif cur_rsi < 45 and rsi_trend <= 0:    strength, score = "BEARISH", -1.0
        else:                                    strength, score = "NEUTRAL", 0.0

        return {
            "strength": strength, "score": score,
            "current_rsi": round(cur_rsi, 1), "rsi_trend": round(rsi_trend, 3),
            "current_osc": round(cur_osc, 1), "osc_trend": round(osc_trend, 3),
        }

    def _trend(self, window: list) -> Dict[str, Any]:
        above200 = [(df["price"] > df["ma200 latest"]).mean() for _, df in window]
        align90 = [(df["ma90 latest"] > df["ma200 latest"]).mean() for _, df in window]
        cur200   = float(above200[-1])
        cur_align = float(align90[-1])
        consistency = float(np.polyfit(range(len(above200)), above200, 1)[0])

        if cur200 > 0.75 and cur_align > 0.70 and consistency >= 0: quality, score = "STRONG_UPTREND", 2.0
        elif cur200 > 0.60 and cur_align > 0.55:                    quality, score = "UPTREND", 1.0
        elif cur200 < 0.30 and cur_align < 0.30 and consistency < 0:quality, score = "STRONG_DOWNTREND", -2.0
        elif cur200 < 0.45 and cur_align < 0.45:                    quality, score = "DOWNTREND", -1.0
        else:                                                        quality, score = "TRENDLESS", 0.0

        return {
            "quality": quality, "score": score,
            "above_200dma": round(cur200, 3),
            "ma_alignment": round(cur_align, 3),
            "trend_consistency": round(consistency, 4),
        }

    def _breadth(self, df: pd.DataFrame) -> Dict[str, Any]:
        rsi_bull = float((df["rsi latest"] > 50).mean())
        osc_pos  = float((df["osc latest"] > 0).mean())
        rsi_weak = float((df["rsi latest"] < 40).mean())
        osc_os   = float((df["osc latest"] < -50).mean())
        divergence = abs(rsi_bull - osc_pos)

        if rsi_bull > 0.70 and osc_pos > 0.60 and divergence < 0.15: quality, score = "STRONG_BROAD", 2.0
        elif rsi_bull > 0.55 and osc_pos > 0.45:                      quality, score = "HEALTHY", 1.0
        elif rsi_weak > 0.60 and osc_os > 0.50:                       quality, score = "CAPITULATION", -2.0
        elif rsi_weak > 0.45 and osc_os > 0.35:                       quality, score = "WEAK", -1.0
        elif divergence > 0.25:                                        quality, score = "DIVERGENT", -0.5
        else:                                                          quality, score = "MIXED", 0.0

        return {
            "quality": quality, "score": score,
            "rsi_bullish_pct": round(rsi_bull, 3),
            "osc_positive_pct": round(osc_pos, 3),
            "divergence": round(divergence, 3),
        }

    def _volatility(self, window: list) -> Dict[str, Any]:
        bbw = [
            ((4.0 * df["dev20 latest"]) / (df["ma20 latest"] + 1e-6)).mean()
            for _, df in window
        ]
        cur_bbw = float(bbw[-1])
        trend = float(np.polyfit(range(len(bbw)), bbw, 1)[0])

        if cur_bbw < 0.08 and trend < 0:   regime, score = "SQUEEZE", 0.5
        elif cur_bbw > 0.15 and trend > 0:  regime, score = "PANIC", -1.0
        elif cur_bbw > 0.12:                regime, score = "ELEVATED", -0.5
        else:                               regime, score = "NORMAL", 0.0

        return {
            "regime": regime, "score": score,
            "current_bbw": round(cur_bbw, 4),
            "vol_trend": round(trend, 5),
        }

    def _extremes(self, df: pd.DataFrame) -> Dict[str, Any]:
        os_pct = float((df["zscore latest"] < -2.0).mean())
        ob_pct = float((df["zscore latest"] > 2.0).mean())

        if os_pct > 0.40:   ext, score = "DEEPLY_OVERSOLD", 1.5
        elif ob_pct > 0.40: ext, score = "DEEPLY_OVERBOUGHT", -1.5
        elif os_pct > 0.20: ext, score = "OVERSOLD", 0.75
        elif ob_pct > 0.20: ext, score = "OVERBOUGHT", -0.75
        else:               ext, score = "NORMAL", 0.0

        return {
            "type": ext, "score": score,
            "oversold_pct": round(os_pct, 3),
            "overbought_pct": round(ob_pct, 3),
        }

    def _correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        rsi_med = float(df["rsi latest"].median())
        osc_med = float(df["osc latest"].median())
        rsi_dir = abs(float((df["rsi latest"] > rsi_med).mean()) - 0.5) * 2.0
        osc_dir = abs(float((df["osc latest"] > osc_med).mean()) - 0.5) * 2.0
        agree = (
            float(((df["rsi latest"] < 40) & (df["osc latest"] < -30)).mean()) +
            float(((df["rsi latest"] > 60) & (df["osc latest"] > 30)).mean())
        )
        disp = (df["rsi latest"].std() / 50 + df["osc latest"].std() / 100) / 2
        raw = float(np.clip((rsi_dir + osc_dir) / 2 * (1.0 - disp) + agree * 0.3, 0, 1))

        if raw > 0.7:   regime, score = "HIGH_CORRELATION", -0.5
        elif raw < 0.4: regime, score = "LOW_CORRELATION", 0.5
        else:           regime, score = "NORMAL", 0.0

        return {
            "regime": regime, "score": score,
            "correlation_score": round(raw, 3),
        }

    def _velocity(self, window: list) -> Dict[str, Any]:
        if len(window) < 5:
            return {"acceleration": "UNKNOWN", "score": 0.0, "avg_velocity": 0.0, "acceleration_value": 0.0}

        rsis = np.array([w[1]["rsi latest"].mean() for w in window[-5:]])
        vel = np.diff(rsis)
        avg_vel = float(np.mean(vel))
        accel_vals = np.diff(vel)
        cur_accel = float(accel_vals[-1]) if len(accel_vals) else 0.0

        if avg_vel > 1.5 and cur_accel > 0:         label, score = "ACCELERATING_UP", 1.5
        elif avg_vel > 1.0 and cur_accel >= 0:       label, score = "RISING_FAST", 1.0
        elif avg_vel > 0.5:                          label, score = "RISING", 0.5
        elif avg_vel < -1.5 and cur_accel < 0:       label, score = "ACCELERATING_DOWN", -1.5
        elif avg_vel < -1.0 and cur_accel <= 0:      label, score = "FALLING_FAST", -1.0
        elif avg_vel < -0.5:                         label, score = "FALLING", -0.5
        elif abs(avg_vel) < 0.5 and cur_accel > 0.5: label, score = "COILING_UP", 0.3
        elif abs(avg_vel) < 0.5 and cur_accel < -0.5:label, score = "COILING_DOWN", -0.3
        else:                                        label, score = "STABLE", 0.0

        return {
            "acceleration": label, "score": score,
            "avg_velocity": round(avg_vel, 3),
            "acceleration_value": round(cur_accel, 3),
        }

    # ── Classification ───────────────────────────────────────────────────────

    def _classify(self, score: float, factors: FactorScores) -> Tuple[str, float]:
        breadth_div = factors.breadth.get("quality") == "DIVERGENT"
        for regime, threshold, base_conf in self._THRESHOLDS:
            if score >= threshold:
                conf = base_conf * 0.75 if breadth_div else base_conf
                return regime, round(conf, 2)
        return "CRISIS", 0.85

    # ── Explanation ──────────────────────────────────────────────────────────

    def _explain(self, regime: str, confidence: float, factors: FactorScores, score: float) -> str:
        icon = REGIME_ICONS.get(regime, "")
        lines = [
            f"**Detected Regime:** {icon} {regime.replace('_', ' ')}",
            f"**Composite Score:** {score:+.2f} | **Confidence:** {confidence:.0%}",
            "",
            f"**Market Assessment:** {REGIME_DESCRIPTIONS.get(regime, '')}",
            "",
            "**Factor Breakdown:**",
        ]

        display = factors.to_display_list()
        for fname, fscore, flabel, fweight in display:
            direction = "▲" if fscore > 0.2 else ("▼" if fscore < -0.2 else "—")
            lines.append(f"• **{fname}** ({fweight:.0%} weight): {flabel} {direction} {fscore:+.1f}")

        if factors.breadth.get("quality") == "DIVERGENT":
            lines += ["", "⚠️ **Alert:** Breadth divergence detected — narrow leadership may not persist."]

        if factors.volatility.get("regime") == "SQUEEZE":
            lines += ["", "📌 **Note:** Volatility squeeze detected — potential directional breakout imminent."]

        if factors.extremes.get("type") in ("DEEPLY_OVERSOLD", "OVERSOLD"):
            lines += ["", "🔍 **Opportunity:** Statistical oversold conditions present — mean-reversion potential elevated."]

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# REGIME HISTORY SERIES
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_history_series(
    historical_data: List[Tuple[datetime, pd.DataFrame]],
    window_size: int = 10,
    step: int = 1,
) -> List[RegimeResult]:
    """
    Compute a rolling time series of regime readings over a historical window.

    Slides a look-back window of `window_size` days forward by `step` days,
    yielding one RegimeResult per position.  Enables regime transition charts.

    Args:
        historical_data: Chronologically ordered (date, DataFrame) tuples.
        window_size:     Days per detection window (default 10).
        step:            Slide step between successive windows (default 1).

    Returns:
        List of RegimeResult objects, one per window position.
    """
    detector = MarketRegimeDetector()
    results: List[RegimeResult] = []

    if len(historical_data) < window_size:
        return results

    for i in range(window_size, len(historical_data) + 1, step):
        window = historical_data[max(0, i - window_size): i]
        analysis_date, _ = historical_data[i - 1]
        try:
            result = detector.detect(window, analysis_date=analysis_date)
            results.append(result)
        except Exception:
            continue

    return results


def compute_conviction_signals(
    portfolio: "pd.DataFrame",
    current_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute signal-based conviction scores for each portfolio holding.

    Reads the live indicator snapshot (current_df) and produces four binary
    signals plus a composite conviction score (0–100) per position.

    Signal definitions
    ──────────────────
    RSI:       +2 overbought-bull (>60), +1 mild bull, −1 mild bear, −2 bearish (<40)
    OSC:       +2 osc above 9EMA and positive, +1 above 9EMA, −1/−2 below
    Z-Score:   +2 deeply oversold (<−2σ), −2 deeply overbought (>+2σ)
    MA Align:  0–5 count of: price>MA20, price>MA90, price>MA200, MA20>MA90, MA90>MA200
    Conviction: normalised weighted composite → 0–100

    Returns the original portfolio DataFrame with appended signal columns.
    """
    if portfolio.empty or current_df.empty:
        return portfolio.copy()

    result = portfolio.copy()

    # Build a symbol-indexed lookup — strip .NS suffix if present
    lookup_df = current_df.copy()
    if "symbol" in lookup_df.columns:
        lookup_df = lookup_df.set_index("symbol")

    rows = []
    for _, port_row in result.iterrows():
        sym = port_row["symbol"]
        sig: Dict[str, Any] = {
            "symbol": sym,
            "rsi_signal": 0, "osc_signal": 0,
            "zscore_signal": 0, "ma_signal": 0,
            "rsi_value": None, "osc_value": None,
            "zscore_value": None, "ma_count": None,
            "conviction_score": 50,
        }

        if sym not in lookup_df.index:
            rows.append(sig)
            continue

        d = lookup_df.loc[sym]

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi = d.get("rsi latest")
        if rsi is not None and not pd.isna(rsi):
            rsi = float(rsi)
            sig["rsi_value"] = round(rsi, 1)
            if rsi > 60:        sig["rsi_signal"] = 2
            elif rsi > 52:      sig["rsi_signal"] = 1
            elif rsi < 40:      sig["rsi_signal"] = -2
            elif rsi < 48:      sig["rsi_signal"] = -1

        # ── Oscillator ───────────────────────────────────────────────────────
        osc  = d.get("osc latest")
        ema9 = d.get("9ema osc latest")
        if osc is not None and ema9 is not None and not pd.isna(osc) and not pd.isna(ema9):
            osc, ema9 = float(osc), float(ema9)
            sig["osc_value"] = round(osc, 1)
            if osc > ema9 and osc > 0:    sig["osc_signal"] = 2
            elif osc > ema9:              sig["osc_signal"] = 1
            elif osc < ema9 and osc < 0:  sig["osc_signal"] = -2
            else:                         sig["osc_signal"] = -1

        # ── Z-Score ──────────────────────────────────────────────────────────
        zscore = d.get("zscore latest")
        if zscore is not None and not pd.isna(zscore):
            zscore = float(zscore)
            sig["zscore_value"] = round(zscore, 2)
            if zscore < -2.0:     sig["zscore_signal"] = 2
            elif zscore < -1.0:   sig["zscore_signal"] = 1
            elif zscore > 2.0:    sig["zscore_signal"] = -2
            elif zscore > 1.0:    sig["zscore_signal"] = -1

        # ── MA Alignment ─────────────────────────────────────────────────────
        price = d.get("price")
        ma20  = d.get("ma20 latest")
        ma90  = d.get("ma90 latest")
        ma200 = d.get("ma200 latest")
        vals  = [price, ma20, ma90, ma200]
        if all(v is not None and not pd.isna(v) and float(v) > 0 for v in vals):
            price, ma20, ma90, ma200 = [float(v) for v in vals]
            count = sum([price > ma20, price > ma90, price > ma200, ma20 > ma90, ma90 > ma200])
            sig["ma_count"] = count
            # Map 0-5 to -2.0 to +2.0
            sig["ma_signal"] = round((count - 2.5) * (4.0 / 5.0), 1)

        # ── Composite Conviction (0–100) ──────────────────────────────────────
        # Weighted sum: RSI 30%, OSC 30%, Z-Score 20%, MA 20%
        raw = (
            sig["rsi_signal"]    * 0.30 +
            sig["osc_signal"]    * 0.30 +
            sig["zscore_signal"] * 0.20 +
            sig["ma_signal"]     * 0.20
        )
        # raw range: [-2, +2] → normalise to [0, 100]
        sig["conviction_score"] = int(round(max(0.0, min(100.0, (raw + 2.0) / 4.0 * 100.0))))

        rows.append(sig)

    conv_df = pd.DataFrame(rows)
    # Merge back by symbol; drop duplicate columns that already exist
    merge_cols = [c for c in conv_df.columns if c not in result.columns or c == "symbol"]
    return result.merge(conv_df[merge_cols], on="symbol", how="left")


__all__ = [
    "MarketRegimeDetector",
    "RegimeResult",
    "FactorScores",
    "REGIME_COLORS",
    "REGIME_ICONS",
    "REGIME_DESCRIPTIONS",
    "REGIME_MIX_MAP",
    "get_regime_history_series",
    "compute_conviction_signals",
]
