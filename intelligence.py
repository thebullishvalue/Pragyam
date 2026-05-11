"""
PRAGYAM — Intelligence (Per-regime conviction-weight calibration)
══════════════════════════════════════════════════════════════════════════════

Pragyam composes four conviction signals — RSI, Oscillator, Z-Score, MA-alignment
— into a 0–100 score using fixed weights (0.30 / 0.30 / 0.20 / 0.20). Different
market regimes reward those signals differently: bull regimes favour momentum
(RSI / MA), choppy regimes favour mean-reversion (Z-score). This module learns
a per-regime override of those four weights from historical forward-return data.

What it optimizes
─────────────────
A four-element weight vector on the simplex (w_rsi + w_osc + w_z + w_ma = 1,
each ≥ 0). The objective is the average cross-sectional Spearman IC of the
weighted conviction score against forward simple returns at horizon H,
divided by its std across dates (Information Ratio).

What it does NOT do
───────────────────
No long/short asymmetry. No tier multipliers. No factor rename to F1–F6.
Pragyam is long-only and uses four named signals; the calibration mirrors that
exactly. The whole search space is three free parameters (the simplex has
dimension 3) parameterized via softmax over four scalars.

Persistence
───────────
One JSON file per regime under .passports/passport_<regime>.json. Loading is
defensive: if the file is missing, malformed, or the calibration failed
validation (val IR not measurable), defaults are used.

Author: @thebullishvalue
Version: 4.0.0 (Pragyam-conviction-fidelity)
"""

import os
import re
import json
import math
import numpy as np
import pandas as pd
import optuna
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable

# ── Constants ──────────────────────────────────────────────────────────────────
PASSPORT_DIR = ".passports"
PASSPORT_VERSION = "v4-pragyam-conviction"
DEFAULT_HORIZON = 10           # forward-return horizon in trading days (~2 weeks)
MIN_XSECT = 5                  # min symbols per date to compute a usable IC
MIN_TOTAL_DATES = 20           # min total dates with valid IC for calibration

# Pragyam's canonical conviction weights (per the README) — used as fallback.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_rsi": 0.30,
    "w_osc": 0.30,
    "w_z":   0.20,
    "w_ma":  0.20,
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Simplex parameterization ───────────────────────────────────────────────────
def _softmax4(a: float, b: float, c: float, d: float) -> Tuple[float, float, float, float]:
    """Map four unconstrained scalars to the 4-simplex via softmax."""
    x = np.array([a, b, c, d], dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    e = e / e.sum()
    return float(e[0]), float(e[1]), float(e[2]), float(e[3])


# ── Spearman IC (vectorized per-date) ──────────────────────────────────────────
def _ic_per_date(scores: np.ndarray, returns: np.ndarray, dates: np.ndarray, min_xsect: int) -> np.ndarray:
    """Spearman rank correlation of `scores` vs `returns` within each date group."""
    out: List[float] = []
    for d in np.unique(dates):
        mask = dates == d
        if int(mask.sum()) < min_xsect:
            continue
        s = scores[mask]
        r = returns[mask]
        if np.all(s == s[0]) or np.all(r == r[0]):
            continue
        s_rank = pd.Series(s).rank().to_numpy()
        r_rank = pd.Series(r).rank().to_numpy()
        sm = s_rank.mean()
        rm = r_rank.mean()
        num = float(((s_rank - sm) * (r_rank - rm)).sum())
        den = math.sqrt(float(((s_rank - sm) ** 2).sum()) * float(((r_rank - rm) ** 2).sum()))
        if den == 0.0:
            continue
        out.append(num / den)
    return np.array(out, dtype=np.float64)


def _ir_for_weights(weights: Dict[str, float], harvest: pd.DataFrame, ret_col: str = "fwd_ret",
                    min_xsect: int = MIN_XSECT) -> Tuple[float, int]:
    """Information Ratio (mean IC / std IC) and the count of valid date-ICs."""
    if harvest.empty:
        return float("nan"), 0
    raw = (
        weights["w_rsi"] * harvest["rsi_signal"].to_numpy(dtype=np.float64) +
        weights["w_osc"] * harvest["osc_signal"].to_numpy(dtype=np.float64) +
        weights["w_z"]   * harvest["zscore_signal"].to_numpy(dtype=np.float64) +
        weights["w_ma"]  * harvest["ma_signal"].to_numpy(dtype=np.float64)
    )
    rets = harvest[ret_col].to_numpy(dtype=np.float64)
    dates = harvest["date"].to_numpy()
    ics = _ic_per_date(raw, rets, dates, min_xsect)
    n = len(ics)
    if n < 3:
        return float("nan"), n
    mu = float(ics.mean())
    sd = float(ics.std(ddof=1))
    return mu / max(sd, 1e-6), n


# ── Per-row signal recomputation (mirrors regime.compute_conviction_signals) ───
def _signals_from_row(r) -> Optional[Dict[str, float]]:
    """Compute (rsi_signal, osc_signal, zscore_signal, ma_signal) from a row of indicators.

    Mirrors the integer-step thresholds used in regime.compute_conviction_signals
    so calibration and inference use the same signal definitions.
    """
    rsi   = r.get("rsi latest")
    osc   = r.get("osc latest")
    ema9  = r.get("9ema osc latest")
    z     = r.get("zscore latest")
    price = r.get("price")
    ma20  = r.get("ma20 latest")
    ma90  = r.get("ma90 latest")
    ma200 = r.get("ma200 latest")

    rsi_signal = osc_signal = z_signal = ma_signal = 0.0
    have_any = False

    if rsi is not None and not pd.isna(rsi):
        rsi = float(rsi); have_any = True
        if rsi > 60:   rsi_signal = 2.0
        elif rsi > 52: rsi_signal = 1.0
        elif rsi < 40: rsi_signal = -2.0
        elif rsi < 48: rsi_signal = -1.0

    if osc is not None and ema9 is not None and not pd.isna(osc) and not pd.isna(ema9):
        osc = float(osc); ema9 = float(ema9); have_any = True
        if osc > ema9 and osc > 0:    osc_signal = 2.0
        elif osc > ema9:              osc_signal = 1.0
        elif osc < ema9 and osc < 0:  osc_signal = -2.0
        else:                          osc_signal = -1.0

    if z is not None and not pd.isna(z):
        z = float(z); have_any = True
        if z < -2.0:   z_signal = 2.0
        elif z < -1.0: z_signal = 1.0
        elif z > 2.0:  z_signal = -2.0
        elif z > 1.0:  z_signal = -1.0

    if all(v is not None and not pd.isna(v) and float(v) > 0 for v in (price, ma20, ma90, ma200)):
        price, ma20, ma90, ma200 = float(price), float(ma20), float(ma90), float(ma200)
        count = sum([price > ma20, price > ma90, price > ma200, ma20 > ma90, ma90 > ma200])
        ma_signal = round((count - 2.5) * (4.0 / 5.0), 2)
        have_any = True

    if not have_any:
        return None
    return {
        "rsi_signal":    rsi_signal,
        "osc_signal":    osc_signal,
        "zscore_signal": z_signal,
        "ma_signal":     ma_signal,
    }


def build_harvest(history_window: List[Tuple[datetime, pd.DataFrame]],
                  horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Build the (date, symbol) panel needed for calibration.

    For each historical day t whose forward day t+horizon also has prices, emit
    one row per symbol with the four signals at t and the realized simple
    return between price[t] and price[t+horizon]. Rows with missing price or
    missing signals are dropped.
    """
    if not history_window or len(history_window) <= horizon:
        return pd.DataFrame()

    rows: List[Dict] = []
    n = len(history_window)
    for i in range(n - horizon):
        day_t = history_window[i][1]
        day_fwd = history_window[i + horizon][1]
        if day_t is None or day_fwd is None or day_t.empty or day_fwd.empty:
            continue
        try:
            fwd_prices = day_fwd.set_index("symbol")["price"]
        except Exception:
            continue
        for _, r in day_t.iterrows():
            sym = r.get("symbol")
            p_t = r.get("price")
            if sym is None or p_t is None or pd.isna(p_t) or float(p_t) <= 0:
                continue
            p_fwd = fwd_prices.get(sym)
            if p_fwd is None or pd.isna(p_fwd) or float(p_fwd) <= 0:
                continue
            sigs = _signals_from_row(r)
            if sigs is None:
                continue
            rows.append({
                "date": i,
                "symbol": sym,
                **sigs,
                "fwd_ret": float(p_fwd) / float(p_t) - 1.0,
            })
    return pd.DataFrame(rows)


# ── Passport persistence ───────────────────────────────────────────────────────
def _slug(s: Optional[str]) -> str:
    """Filesystem-safe slug. Empty / None becomes 'all'."""
    if not s:
        return "all"
    out = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return out or "all"


def passport_filename(universe: str, selected_index: Optional[str], regime: str) -> str:
    """Composite filename: passport_<universe>__<index>__<regime>.json."""
    return f"passport_{_slug(universe)}__{_slug(selected_index)}__{_slug(regime)}.json"


class IntelligencePassport:
    """Calibrated conviction weights keyed by (universe, selected_index, regime).

    Different universes — and different sub-indexes within a universe — score
    the four signals differently because their cross-sectional dynamics differ.
    Persisting per (universe, index, regime) prevents one universe's calibration
    from being applied to another.
    """

    def __init__(self, universe: str, selected_index: Optional[str], regime_name: str):
        self.universe = universe or "default"
        self.selected_index = selected_index
        self.regime_name = regime_name or "UNKNOWN"
        self.path = os.path.join(PASSPORT_DIR, passport_filename(self.universe, self.selected_index, self.regime_name))
        os.makedirs(PASSPORT_DIR, exist_ok=True)
        self.data: Dict = self._load()

    @property
    def label(self) -> str:
        """Human-readable name, e.g. 'NIFTY 50 · BEAR' or 'ETF Universe · WEAK_BULL'."""
        idx = f" · {self.selected_index}" if self.selected_index else ""
        return f"{self.universe}{idx} · {self.regime_name}"

    def _load(self) -> Dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def exists(self) -> bool:
        return bool(self.data and self.data.get("is_calibrated") and self.data.get("engine_version") == PASSPORT_VERSION)

    def get_weights(self) -> Dict[str, float]:
        if self.exists():
            w = self.data.get("weights", {})
            if all(k in w for k in DEFAULT_WEIGHTS):
                return {k: float(w[k]) for k in DEFAULT_WEIGHTS}
        return DEFAULT_WEIGHTS.copy()

    def metrics(self) -> Dict[str, float]:
        return {
            "train_ir":      float(self.data.get("train_ir", float("nan"))) if self.data else float("nan"),
            "val_ir":        float(self.data.get("val_ir",   float("nan"))) if self.data else float("nan"),
            "n_train_dates": int(self.data.get("n_train_dates", 0)) if self.data else 0,
            "n_val_dates":   int(self.data.get("n_val_dates",   0)) if self.data else 0,
            "n_trials":      int(self.data.get("n_trials",      0)) if self.data else 0,
            "horizon":       int(self.data.get("horizon", DEFAULT_HORIZON)) if self.data else DEFAULT_HORIZON,
        }

    @property
    def last_calibrated(self) -> str:
        ts = self.data.get("timestamp") if self.data else None
        if not ts:
            return "Never"
        try:
            return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ts

    @property
    def version(self) -> str:
        return self.data.get("engine_version", "—") if self.data else "—"

    def save(self,
             weights: Dict[str, float],
             train_ir: float, val_ir: float,
             n_train_dates: int, n_val_dates: int,
             n_trials: int, horizon: int) -> None:
        self.data = {
            "universe": self.universe,
            "selected_index": self.selected_index,
            "regime": self.regime_name,
            "weights": {k: weights[k] for k in DEFAULT_WEIGHTS},
            "train_ir": train_ir,
            "val_ir":   val_ir,
            "n_train_dates": n_train_dates,
            "n_val_dates":   n_val_dates,
            "n_trials": n_trials,
            "horizon":  horizon,
            "timestamp": datetime.now().isoformat(),
            "engine_version": PASSPORT_VERSION,
            "is_calibrated": True,
        }
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def delete(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)
        self.data = {}


# ── Public API ────────────────────────────────────────────────────────────────
def get_active_weights(universe: str, selected_index: Optional[str], regime_name: str,
                       mode: str = "Standard") -> Dict[str, float]:
    """Return the four conviction weights to use for a given (universe, index, regime, mode).

    Standard mode → defaults. Intelligence mode → calibrated weights from the
    passport for this exact (universe, index, regime) tuple, falling back to
    defaults if no usable passport exists.
    """
    if mode != "Intelligence":
        return DEFAULT_WEIGHTS.copy()
    return IntelligencePassport(universe, selected_index, regime_name).get_weights()


def calibrate(universe: str,
              selected_index: Optional[str],
              regime_name: str,
              harvest: pd.DataFrame,
              n_trials: int = 100,
              train_frac: float = 0.7,
              horizon: int = DEFAULT_HORIZON,
              progress_callback: Optional[Callable[[int, int, float], None]] = None
              ) -> Optional[Dict]:
    """Optimize the four conviction weights against forward IC and save the passport.

    Returns a dict {weights, train_ir, val_ir, n_train_dates, n_val_dates} on
    success, or None when the harvest is too sparse to produce a usable IC.
    """
    if harvest is None or harvest.empty:
        return None

    dates = sorted(harvest["date"].unique())
    if len(dates) < MIN_TOTAL_DATES:
        return None

    n_train = max(1, int(len(dates) * train_frac))
    train_dates = set(dates[:n_train])
    val_dates   = set(dates[n_train:])
    if not val_dates:
        return None

    train_df = harvest[harvest["date"].isin(train_dates)]
    val_df   = harvest[harvest["date"].isin(val_dates)]

    # Sanity: defaults must score on train (otherwise the harvest is unusable).
    default_train_ir, _ = _ir_for_weights(DEFAULT_WEIGHTS, train_df)
    if not np.isfinite(default_train_ir):
        return None

    def objective(trial: "optuna.trial.Trial") -> float:
        a = trial.suggest_float("a_rsi", -3.0, 3.0)
        b = trial.suggest_float("a_osc", -3.0, 3.0)
        c = trial.suggest_float("a_z",   -3.0, 3.0)
        d = trial.suggest_float("a_ma",  -3.0, 3.0)
        w_rsi, w_osc, w_z, w_ma = _softmax4(a, b, c, d)
        weights = {"w_rsi": w_rsi, "w_osc": w_osc, "w_z": w_z, "w_ma": w_ma}
        ir, _ = _ir_for_weights(weights, train_df)
        if not np.isfinite(ir):
            ir = -1.0
        if progress_callback is not None:
            try:
                progress_callback(trial.number + 1, n_trials, ir)
            except Exception:
                pass
        return ir

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    p = study.best_params
    w_rsi, w_osc, w_z, w_ma = _softmax4(p["a_rsi"], p["a_osc"], p["a_z"], p["a_ma"])
    weights = {"w_rsi": w_rsi, "w_osc": w_osc, "w_z": w_z, "w_ma": w_ma}

    train_ir, _ = _ir_for_weights(weights, train_df)
    val_ir, n_val_ic = _ir_for_weights(weights, val_df)

    if not np.isfinite(val_ir) or n_val_ic < 3:
        return None

    passport = IntelligencePassport(universe, selected_index, regime_name)
    passport.save(
        weights=weights,
        train_ir=train_ir,
        val_ir=val_ir,
        n_train_dates=len(train_dates),
        n_val_dates=len(val_dates),
        n_trials=n_trials,
        horizon=horizon,
    )
    return {
        "weights": weights,
        "train_ir": train_ir,
        "val_ir": val_ir,
        "n_train_dates": len(train_dates),
        "n_val_dates": len(val_dates),
    }


__all__ = [
    "DEFAULT_WEIGHTS",
    "DEFAULT_HORIZON",
    "PASSPORT_VERSION",
    "IntelligencePassport",
    "passport_filename",
    "get_active_weights",
    "build_harvest",
    "calibrate",
]
