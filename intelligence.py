"""
PRAGYAM — Intelligence (Per-regime conviction-weight calibration)
══════════════════════════════════════════════════════════════════════════════

Pragyam composes FIVE conviction signals — RSI, Oscillator, Z-Score, MA-alignment,
and VAP (value-area position, from the volume profile) — into a 0–100 score.
Different market regimes reward those signals differently: bull regimes favour
momentum (RSI / MA), choppy regimes favour mean-reversion (Z-score / VAP). This
module learns a per-regime override of those five weights from historical
forward-return data.

Only the CONVICTION blend is calibrated here. The market-regime detector uses
fixed factor weights (regime.FACTOR_WEIGHTS) and is NOT calibrated.

What it optimizes
─────────────────
Conviction: a five-element weight vector on the simplex (sums to 1, each ≥ 0)
maximizing the cross-sectional Spearman IC of the weighted conviction score
vs forward returns at horizon H, divided by its std (Information Ratio).

Persistence
───────────
One JSON per (universe, index, regime) under .passports/passport_<...>.json.
Loading is defensive: missing, malformed, wrong-version, or validation-failed
→ defaults are used.

Author: @thebullishvalue
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
# v7 adds the sixth conviction signal, Strategy Endorsement (w_strat): the
# 95-strategy layer's output is now converted into a real cross-sectional vote
# signal instead of being discarded after aggregation (see
# AUDIT_DIRECTIVES.md A4 — previously every strategy's weights were thrown
# away and the union of "candidates" across 95 strategies was ~= the whole
# data-valid universe, so the layer contributed zero information to
# selection). Bumping the version auto-invalidates v6 passports (calibrated
# on five signals) so they are transparently re-calibrated on the 6-signal
# blend.
#
# v6 tightened calibration integrity: passports are only saved when validation
# IR beats the default-weights baseline on the same held-out split (previously
# any *measurable* val IR was accepted, including negative ones that
# anti-predict forward returns), applies shrinkage toward the default
# proportional to out-of-sample confidence, and purges/embargoes the
# train/val boundary plus switches IC estimation to non-overlapping dates to
# remove the serial-correlation bias in the IR estimate.
PASSPORT_VERSION = "v7-pragyam-conviction-strat"
DEFAULT_HORIZON = 10           # forward-return horizon in trading days (~2 weeks)
MIN_XSECT = 10                 # min symbols per date to compute a usable IC (raised
                                # from 5 — Spearman IC standard error ~= 1/sqrt(n-1),
                                # 5 names is pure noise; see Grinold & Kahn breadth)
MIN_TOTAL_DATES = 20           # min total dates with valid IC for calibration

# Pragyam's conviction weights — used as the Standard-mode FALLBACK only.
#
# These are *defaults*, not the truth: the moment Intelligence mode calibrates,
# the per-(universe, index, regime) passport learns these six weights from
# forward returns (IR-maximising) and overrides every number here. So the exact
# split below matters only before a universe has ever been calibrated. Every
# signal — including the strategy-endorsement vote (w_strat) — is a full
# first-class peer and the calibrator is free to up- or down-weight any of
# them from data.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_rsi":   1.0 / 6.0,
    "w_osc":   1.0 / 6.0,
    "w_z":     1.0 / 6.0,
    "w_ma":    1.0 / 6.0,
    "w_vap":   1.0 / 6.0,
    "w_strat": 1.0 / 6.0,
}

# NOTE: only the CONVICTION blend is calibrated. The market-regime detector uses
# fixed factor weights (see regime.FACTOR_WEIGHTS) — it is not calibrated.

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Simplex parameterization ───────────────────────────────────────────────────
def _softmax5(a: float, b: float, c: float, d: float, e_: float
              ) -> Tuple[float, float, float, float, float]:
    """Map five unconstrained scalars to the 5-simplex via softmax.

    Calibration searches RSI / OSC / Z / MA / VAP only — the sixth conviction
    signal, Strategy-Endorsement (w_strat), is held at a FIXED weight (see
    calibrate()'s free_mass scaling) because it has no historical values to
    evaluate an IC against (see _signals_from_row's docstring). This softmax
    output is scaled by (1 - w_strat_fixed) so the six deployed weights still
    sum to 1 overall. The simplex keeps the five searched weights
    non-negative and summing to 1 among themselves, so calibration is a pure
    re-allocation of trust across the historically-evaluable signals — no
    signal can dominate via raw magnitude.
    """
    x = np.array([a, b, c, d, e_], dtype=np.float64)
    x = x - x.max()
    ex = np.exp(x)
    ex = ex / ex.sum()
    return float(ex[0]), float(ex[1]), float(ex[2]), float(ex[3]), float(ex[4])


# ── Spearman IC (vectorized per-date) ──────────────────────────────────────────
def _ic_per_date(scores: np.ndarray, returns: np.ndarray, dates: np.ndarray, min_xsect: int,
                  return_dates: bool = False):
    """Spearman rank correlation of `scores` vs `returns` within each date group.

    When ``return_dates`` is True, also returns the array of dates that
    produced a usable IC (same order/length as the returned IC array) — needed
    to pair two IC series on their common dates for a paired significance test.
    """
    out: List[float] = []
    kept_dates: List = []
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
        kept_dates.append(d)
    if return_dates:
        return np.array(out, dtype=np.float64), np.array(kept_dates)
    return np.array(out, dtype=np.float64)


def _non_overlapping_dates(harvest: pd.DataFrame, horizon: int) -> np.ndarray:
    """Dates spaced >= horizon apart, so no two rows' forward-return windows
    overlap (each covers [t, t+horizon)). ``date`` is the integer trading-day
    index assigned in build_harvest, so this is a simple stride over the
    sorted unique dates actually present in this (possibly filtered) harvest —
    NOT a stride over the raw window index, which would silently skip dates
    that build_harvest / regime filtering already dropped.

    Overlapping-window ICs are serially correlated (adjacent dates share
    horizon-1 of their horizon forward-return days), which understates the
    true standard deviation of the IC series and inflates the resulting
    Information Ratio (Lo 2002, "The Statistics of Sharpe Ratios"; purging in
    López de Prado, AFML 2018 ch.7). Sampling on a non-overlapping stride is
    the simplest robust fix.
    """
    uniq = np.unique(harvest["date"].to_numpy())
    if len(uniq) == 0:
        return uniq
    keep = [uniq[0]]
    for d in uniq[1:]:
        if d - keep[-1] >= horizon:
            keep.append(d)
    return np.array(keep)


def _weighted_score(weights: Dict[str, float], harvest: pd.DataFrame) -> np.ndarray:
    """The six-signal weighted composite for every row of `harvest`."""
    n = len(harvest)
    strat = (
        harvest["strat_signal"].to_numpy(dtype=np.float64)
        if "strat_signal" in harvest.columns else np.zeros(n, dtype=np.float64)
    )
    return (
        weights["w_rsi"] * harvest["rsi_signal"].to_numpy(dtype=np.float64) +
        weights["w_osc"] * harvest["osc_signal"].to_numpy(dtype=np.float64) +
        weights["w_z"]   * harvest["zscore_signal"].to_numpy(dtype=np.float64) +
        weights["w_ma"]  * harvest["ma_signal"].to_numpy(dtype=np.float64) +
        weights.get("w_vap", 0.0) * harvest["vap_signal"].to_numpy(dtype=np.float64) +
        weights.get("w_strat", 0.0) * strat
    )


def _ir_for_weights(weights: Dict[str, float], harvest: pd.DataFrame, ret_col: str = "fwd_ret",
                    min_xsect: int = MIN_XSECT, horizon: int = DEFAULT_HORIZON,
                    non_overlapping: bool = True) -> Tuple[float, int]:
    """Information Ratio (mean IC / std IC) and the count of valid date-ICs.

    By default ICs are computed on a non-overlapping date stride (see
    _non_overlapping_dates) to avoid the serial-correlation bias that comes
    from horizon-overlapping forward-return windows on consecutive dates.
    """
    if harvest.empty:
        return float("nan"), 0
    if non_overlapping:
        keep_dates = _non_overlapping_dates(harvest, horizon)
        harvest = harvest[harvest["date"].isin(keep_dates)]
        if harvest.empty:
            return float("nan"), 0
    raw = _weighted_score(weights, harvest)
    rets = harvest[ret_col].to_numpy(dtype=np.float64)
    dates = harvest["date"].to_numpy()
    ics = _ic_per_date(raw, rets, dates, min_xsect)
    n = len(ics)
    if n < 3:
        return float("nan"), n
    mu = float(ics.mean())
    sd = float(ics.std(ddof=1))
    return mu / max(sd, 1e-6), n


# Minimum number of PAIRED non-overlapping validation dates (dates where both
# the candidate and the default weights produce a usable IC) required before
# the beats-default comparison in calibrate() is trusted at all. Below this,
# an IR-vs-IR point-estimate comparison is dominated by sampling noise — see
# AUDIT_DIRECTIVES.md A1/A3 and the empirical false-positive-rate check that
# motivated this constant (a naive "val_ir > default_val_ir" gate accepted
# ~40% of PURE-NOISE calibrations at n=6 paired dates; the paired t-test
# below brought that to 0/30 at n=6, but 6 paired one-sided-95% comparisons
# is still a very thin reed — 8 (t-critical(df=7) ~= 1.90) is the practical
# floor at which Pragyam's actual production panel sizes (~100-190 trading
# days at DEFAULT_HORIZON=10, see backdata._REGIME_LOOKBACK_FILES /
# app._REGIME_LOOKBACK_FILES) can ever produce a passport at all. This is an
# explicit, documented trade of some statistical power for calibration being
# achievable in practice; MIN_PAIRED_T_STAT below is raised to compensate.
MIN_PAIRED_VAL_DATES = 8
# One-sided critical t-value backing the beats-default gate. 1.895 is the
# t-critical(df=7, one-sided, 95%) value, matching the practical floor of 8
# paired dates above rather than the asymptotic normal 1.645 used for larger
# samples — the gate should not get easier to pass just because the panel is
# smaller.
MIN_PAIRED_T_STAT = 1.895


def _paired_beats_default(opt_weights: Dict[str, float], default_weights: Dict[str, float],
                           val_df: pd.DataFrame, horizon: int, min_xsect: int = MIN_XSECT
                           ) -> Tuple[bool, float, int, float, float]:
    """Paired significance test: does `opt_weights` beat `default_weights` on
    the SAME validation dates, by more than sampling noise would explain?

    Comparing two independently-computed IR point estimates (mean/std of two
    different IC series) is unreliable at the small validation-date counts
    calibration operates under — two noise-only weight vectors can differ in
    IR by chance alone (empirically ~40% false-positive rate at n=6 dates,
    see MIN_PAIRED_VAL_DATES). The statistically correct comparison is a
    PAIRED one-sample t-test on the per-date IC differences
    ``d_i = IC_opt(date_i) - IC_default(date_i)`` against zero (same logic as
    comparing two forecasters on identical out-of-sample periods). This nets
    out date-level effects common to both weight vectors and is far more
    powerful than an unpaired comparison at the same sample size.

    Returns (passes, mean_diff, n_paired, t_stat, opt_val_ir) where `passes`
    requires n_paired >= MIN_PAIRED_VAL_DATES, mean_diff > 0, and a one-sided
    t-stat >= MIN_PAIRED_T_STAT (~95% one-sided confidence at the sample size
    the floor is calibrated for).
    """
    if val_df.empty:
        return False, 0.0, 0, 0.0, float("nan")
    keep_dates = _non_overlapping_dates(val_df, horizon)
    vdf = val_df[val_df["date"].isin(keep_dates)]
    if vdf.empty:
        return False, 0.0, 0, 0.0, float("nan")

    opt_raw = _weighted_score(opt_weights, vdf)
    def_raw = _weighted_score(default_weights, vdf)
    rets = vdf[ret_col].to_numpy(dtype=np.float64) if (ret_col := "fwd_ret") in vdf.columns else vdf["fwd_ret"].to_numpy(dtype=np.float64)
    dates = vdf["date"].to_numpy()

    ic_opt, dates_opt = _ic_per_date(opt_raw, rets, dates, min_xsect, return_dates=True)
    ic_def, dates_def = _ic_per_date(def_raw, rets, dates, min_xsect, return_dates=True)

    # Pair on dates common to both IC series (usually identical since both use
    # the same xsect/date gate, but be explicit rather than assume alignment).
    common = np.intersect1d(dates_opt, dates_def)
    if len(common) < MIN_PAIRED_VAL_DATES:
        opt_ir = float(ic_opt.mean() / max(ic_opt.std(ddof=1), 1e-6)) if len(ic_opt) >= 3 else float("nan")
        return False, 0.0, len(common), 0.0, opt_ir

    idx_opt = {d: i for i, d in enumerate(dates_opt)}
    idx_def = {d: i for i, d in enumerate(dates_def)}
    diffs = np.array([ic_opt[idx_opt[d]] - ic_def[idx_def[d]] for d in common], dtype=np.float64)

    n = len(diffs)
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1))
    se = sd_diff / math.sqrt(n) if sd_diff > 1e-9 else 0.0
    t_stat = (mean_diff / se) if se > 1e-9 else (float("inf") if mean_diff > 0 else 0.0)

    opt_ir = float(ic_opt.mean() / max(ic_opt.std(ddof=1), 1e-6)) if len(ic_opt) >= 3 else float("nan")
    passes = bool(mean_diff > 0.0 and t_stat >= MIN_PAIRED_T_STAT)
    return passes, mean_diff, n, t_stat, opt_ir


# ── Per-row signal recomputation (mirrors regime.compute_conviction_signals) ───
def _signals_from_row(r) -> Optional[Dict[str, float]]:
    """Compute (rsi_signal, osc_signal, zscore_signal, ma_signal, vap_signal) from
    a row of indicators.

    Mirrors the integer-step thresholds used in regime.compute_conviction_signals
    so calibration and inference use the same signal definitions.

    Deliberately does NOT compute strat_signal (strategy-endorsement vote rank):
    that signal only exists for the LIVE day a Run Analysis executes on, since
    it requires running all 95 strategies (see app.py Phase 2) — historical
    indicator snapshots in history_window never carry it, and re-running all
    95 strategies against every historical day to backfill it would multiply
    Phase 1.5's calibration cost by up to ~95x. w_strat is therefore excluded
    from the calibration search space entirely (see calibrate()) and always
    applied at its fixed default weight during live scoring
    (regime.compute_conviction_signals) — the calibrator learns how to weight
    the five signals it CAN evaluate historically, not whether to trust the
    strategy layer, which stays a constant.
    """
    rsi   = r.get("rsi latest")
    osc   = r.get("osc latest")
    ema9  = r.get("9ema osc latest")
    z     = r.get("zscore latest")
    price = r.get("price")
    ma20  = r.get("ma20 latest")
    ma90  = r.get("ma90 latest")
    ma200 = r.get("ma200 latest")
    vap   = r.get("vap latest")

    rsi_signal = osc_signal = z_signal = ma_signal = vap_signal = 0.0
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

    # ── Value-Area Position (volume-profile mean reversion) ──────────────────
    #  vap > 0  → price at a discount to accepted value (long bias)
    #  vap < 0  → price at a premium to accepted value (sell bias)
    #  Thresholds mirror the z-score bands so the signal lives in [-2, +2] like
    #  the other four (already volatility-normalised in compute_volume_profile).
    if vap is not None and not pd.isna(vap):
        vap = float(vap); have_any = True
        if vap > 2.0:    vap_signal = 2.0
        elif vap > 1.0:  vap_signal = 1.0
        elif vap < -2.0: vap_signal = -2.0
        elif vap < -1.0: vap_signal = -1.0

    if not have_any:
        return None
    return {
        "rsi_signal":    rsi_signal,
        "osc_signal":    osc_signal,
        "zscore_signal": z_signal,
        "ma_signal":     ma_signal,
        "vap_signal":    vap_signal,
    }


# Regime families a passport can be conditioned on. Grouping the 7 raw regimes
# (STRONG_BULL..CRISIS) into the 3 families already used for strategy-mix
# selection (regime.REGIME_MIX_MAP) keeps each family's sample count usable —
# conditioning on all 7 raw regimes individually would starve every bucket at
# Pragyam's realistic panel sizes. See regime_family_for() / build_harvest.
REGIME_FAMILIES = ("Bull Market Mix", "Chop/Consolidate Mix", "Bear Market Mix")


def regime_family_for(regime_name: str) -> str:
    """Map a raw regime label (BULL, WEAK_BEAR, ...) to its coarse family.

    Local import of regime.REGIME_MIX_MAP to avoid a module-level circular
    import (regime.py already locally imports intelligence.py inside
    compute_conviction_signals — this mirrors that pattern in reverse).
    """
    from regime import REGIME_MIX_MAP
    return REGIME_MIX_MAP.get((regime_name or "UNKNOWN").upper(), "Chop/Consolidate Mix")


def regime_labels_for_window(
    history_window: List[Tuple[datetime, "pd.DataFrame"]],
    window_size: int = 10,
) -> Dict[int, str]:
    """Per-date regime FAMILY for every index in `history_window`.

    Reuses regime.get_regime_history_series (already computed once per run and
    cached in st.session_state.regime_history_series by the caller — this
    function is cheap to call again since detection itself is fast, but
    callers that already have that series should prefer regime_labels_from_series
    to avoid recomputation).

    get_regime_history_series(step=1) returns one RegimeResult per index i in
    range(window_size, len(history_window)+1), where result[k] is the regime
    detected using the trailing `window_size`-day window ending at
    history_window[window_size - 1 + k]. That is: RESULT k CORRESPONDS TO
    history_window INDEX (window_size - 1 + k) — dates before window_size-1
    have no regime reading (insufficient trailing history) and are left
    unlabeled (mapped to "UNKNOWN"'s family, Chop/Consolidate Mix, the
    detector's own failure sentinel — see MarketRegimeDetector.detect).
    """
    from regime import get_regime_history_series
    series = get_regime_history_series(history_window, window_size=window_size, step=1)
    return regime_labels_from_series(series, len(history_window), window_size)


def regime_labels_from_series(series: list, n_dates: int, window_size: int = 10) -> Dict[int, str]:
    """Same mapping as regime_labels_for_window, but from an ALREADY-COMPUTED
    regime series (e.g. st.session_state.regime_history_series) — avoids
    recomputing regime detection when the caller already has it cached.
    """
    labels: Dict[int, str] = {}
    for k, result in enumerate(series):
        idx = window_size - 1 + k
        if idx < n_dates:
            labels[idx] = regime_family_for(getattr(result, "regime", "UNKNOWN"))
    return labels


def build_harvest(history_window: List[Tuple[datetime, pd.DataFrame]],
                  horizon: int = DEFAULT_HORIZON,
                  regime_labels: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    """Build the (date, symbol) panel needed for calibration.

    For each historical day t whose forward day t+horizon also has prices, emit
    one row per symbol with the five signals at t and the realized simple
    return between price[t] and price[t+horizon]. Rows with missing price or
    missing signals are dropped.

    When `regime_labels` is provided (see regime_labels_for_window /
    regime_labels_from_series), each row is also tagged with the regime
    FAMILY in effect at date t, so calibrate() can condition the harvest on
    regime rather than learning one unconditional weight set for the whole
    window and labeling it with whatever regime happened to be current when
    the calibration was triggered (see AUDIT_DIRECTIVES.md A2). Dates with no
    regime reading get "Chop/Consolidate Mix" (the detector's own UNKNOWN
    fallback family).
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
        regime_family = regime_labels.get(i, "Chop/Consolidate Mix") if regime_labels else None
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
            row = {
                "date": i,
                "symbol": sym,
                **sigs,
                "fwd_ret": float(p_fwd) / float(p_t) - 1.0,
            }
            if regime_family is not None:
                row["regime_family"] = regime_family
            rows.append(row)
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
    the five signals differently because their cross-sectional dynamics differ.
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
        if not (self.data and self.data.get("is_calibrated") and self.data.get("engine_version") == PASSPORT_VERSION):
            return False
        # Defensive: a passport that validated worse than the default fallback
        # must never be treated as usable, regardless of how it got on disk
        # (belt-and-suspenders alongside the save-time gate in calibrate()).
        # Imported passports (see app.py import handler) also pass through
        # this check on every read.
        val_ir = self.data.get("val_ir")
        try:
            return val_ir is not None and float(val_ir) > 0.0
        except (TypeError, ValueError):
            return False

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
             n_trials: int, horizon: int,
             is_calibrated: bool = True) -> None:
        """Persist a passport. `is_calibrated=False` writes a defaults-shape
        envelope (used by the sidebar export when no real passport exists) —
        such a file must be REJECTED on import (see the sidebar import
        handler), not silently accepted as if it were a real calibration."""
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
            "is_calibrated": is_calibrated,
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
    """Return the five conviction weights to use for a given (universe, index, regime, mode).

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
              train_frac: float = 0.5,
              horizon: int = DEFAULT_HORIZON,
              progress_callback: Optional[Callable[[int, int, float], None]] = None
              ) -> Dict:
    """Optimize the five conviction weights against forward IC and save the passport.

    Integrity rails (see AUDIT_DIRECTIVES.md A1/A3):
      - train_frac defaults to 0.5 (was 0.7): at Pragyam's realistic panel
        sizes (~100-190 trading days), a 70/30 split leaves too few
        non-overlapping validation dates for the paired significance test
        below to ever clear its bar (empirically 6 at 190 dates / horizon 10).
        A 50/50 split trades train-set size for a validation set that can
        actually support a statistical decision — there is no point learning
        precise weights that can never be validated.
      - Embargo: the last `horizon` train dates are dropped so no training row's
        forward-return window overlaps the validation window (purging, per
        López de Prado AFML ch.7).
      - IC estimation samples non-overlapping dates only (see
        _non_overlapping_dates) to avoid the serial-correlation bias that
        inflates a naive overlapping-window IR (Lo 2002).
      - A passport is saved ONLY when the optimized weights beat the default
        (0.20x5) weights BY A STATISTICALLY SIGNIFICANT MARGIN on the same
        held-out split — a paired one-sample t-test on per-date IC
        differences, not a naive point-estimate IR comparison (see
        _paired_beats_default; a naive comparison was empirically found to
        accept ~40% of PURE-NOISE calibrations at small validation-date
        counts). Optimizing on the training split alone guarantees an
        optimistic train IR (Bailey et al. 2017, "Probability of Backtest
        Overfitting"); requiring a significant out-of-sample edge over the
        naive fallback is the same logic DeMiguel/Garlappi/Uppal (2009) used
        to show 1/N often beats estimated-weight allocations under
        estimation error.
      - The saved weights are shrunk toward the 0.20x5 default in proportion to
        how much the validation split actually supports them
        (lambda = clip(val_ir / (val_ir + train_ir), 0, 1)), so a calibration
        that only marginally beats defaults doesn't get deployed at full
        conviction (a shrinkage estimator dominates a raw sample optimum under
        estimation error — Ledoit & Wolf shrinkage logic applied here to signal
        weights instead of a covariance matrix).

    Returns a dict always carrying `"success": bool` and `"reason": str`. On
    success it also carries {weights, train_ir, val_ir, n_train_dates,
    n_val_dates, default_val_ir, shrinkage}. On failure `"reason"` states
    precisely why (too little history, unmeasurable validation IR, or the
    calibration failed to beat the default-weights baseline) so callers can
    surface an honest diagnostic instead of a generic failure message.
    """
    if harvest is None or harvest.empty:
        return {"success": False, "reason": "Harvest produced no usable rows."}

    # Regime-conditioning: if the harvest carries a per-row regime_family tag
    # (see build_harvest(regime_labels=...)), restrict calibration to dates
    # that were actually in this regime's family. Without this, a passport
    # keyed "BEAR" was previously estimated on the ENTIRE trailing window
    # regardless of regime — the regime label distinguished only when you
    # happened to click Run, not the market state the weights were learned
    # under (see AUDIT_DIRECTIVES.md A2). Raw 7-way regimes are grouped into
    # the 3 coarse families already used for strategy-mix selection
    # (regime.REGIME_MIX_MAP) because conditioning on all 7 individually
    # starves every bucket at Pragyam's realistic panel sizes.
    if "regime_family" in harvest.columns:
        target_family = regime_family_for(regime_name)
        harvest = harvest[harvest["regime_family"] == target_family]
        if harvest.empty:
            return {"success": False,
                    "reason": f"No historical dates fell in the '{target_family}' regime family "
                              f"within this lookback window."}

    dates = sorted(harvest["date"].unique())
    if len(dates) < MIN_TOTAL_DATES:
        return {"success": False,
                "reason": f"Need >= {MIN_TOTAL_DATES} dates with usable signals in the "
                          f"'{regime_family_for(regime_name)}' regime family (have {len(dates)}). "
                          f"This regime hasn't occurred often enough in the lookback window to calibrate."}

    n_train_raw = max(1, int(len(dates) * train_frac))
    # Embargo: drop the trailing `horizon` train dates so no train row's
    # [t, t+horizon) forward-return window overlaps the validation dates that
    # start right after the split.
    n_train = max(1, n_train_raw - horizon)
    train_dates = set(dates[:n_train])
    val_dates   = set(dates[n_train_raw:])
    if not val_dates or not train_dates:
        return {"success": False, "reason": "Train/validation split left an empty partition after embargo."}

    train_df = harvest[harvest["date"].isin(train_dates)]
    val_df   = harvest[harvest["date"].isin(val_dates)]

    # Sanity: defaults must score on train (otherwise the harvest is unusable).
    default_train_ir, _ = _ir_for_weights(DEFAULT_WEIGHTS, train_df, horizon=horizon)
    if not np.isfinite(default_train_ir):
        return {"success": False, "reason": "Default weights produced no measurable train IR — harvest too sparse/degenerate."}

    # w_strat (strategy-endorsement) is EXCLUDED from the calibration search
    # space: it only exists for the live day a run executes on (see app.py
    # Phase 2 — it requires running all 95 strategies), so historical harvest
    # rows have no strat_signal to evaluate an IC against. Re-running all 95
    # strategies against every historical day to backfill it would multiply
    # Phase 1.5's calibration cost by up to ~95x. w_strat is therefore held
    # fixed at its default weight and the five signals that CAN be evaluated
    # historically are searched on a simplex scaled to fill the remaining
    # mass (1 - w_strat_fixed), so the six weights still sum to 1 overall.
    w_strat_fixed = DEFAULT_WEIGHTS["w_strat"]
    free_mass = max(1.0 - w_strat_fixed, 1e-9)

    def objective(trial: "optuna.trial.Trial") -> float:
        a = trial.suggest_float("a_rsi", -3.0, 3.0)
        b = trial.suggest_float("a_osc", -3.0, 3.0)
        c = trial.suggest_float("a_z",   -3.0, 3.0)
        d = trial.suggest_float("a_ma",  -3.0, 3.0)
        e = trial.suggest_float("a_vap", -3.0, 3.0)
        w_rsi, w_osc, w_z, w_ma, w_vap = _softmax5(a, b, c, d, e)
        weights = {
            "w_rsi": w_rsi * free_mass, "w_osc": w_osc * free_mass,
            "w_z": w_z * free_mass, "w_ma": w_ma * free_mass,
            "w_vap": w_vap * free_mass, "w_strat": w_strat_fixed,
        }
        ir, _ = _ir_for_weights(weights, train_df, horizon=horizon)
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
    w_rsi, w_osc, w_z, w_ma, w_vap = _softmax5(
        p["a_rsi"], p["a_osc"], p["a_z"], p["a_ma"], p["a_vap"]
    )
    opt_weights = {
        "w_rsi": w_rsi * free_mass, "w_osc": w_osc * free_mass,
        "w_z": w_z * free_mass, "w_ma": w_ma * free_mass,
        "w_vap": w_vap * free_mass, "w_strat": w_strat_fixed,
    }

    train_ir, _ = _ir_for_weights(opt_weights, train_df, horizon=horizon)

    # Gate: the learned weights must beat the fallback out-of-sample, and by
    # more than sampling noise would explain at these (typically small)
    # validation-date counts. A raw "val_ir(opt) > val_ir(default)" point
    # comparison was empirically found to accept ~40% of PURE-NOISE
    # calibrations at realistic panel sizes (two independently-estimated IRs
    # both carry huge sampling error at n<10 dates). The paired one-sample
    # t-test on per-date IC differences nets out common date-level effects
    # and requires the edge to clear a ~95% one-sided confidence bar before
    # a passport is ever written (see AUDIT_DIRECTIVES.md A1/A3 and
    # _paired_beats_default's docstring for the full rationale).
    passes, mean_diff, n_paired, t_stat, val_ir = _paired_beats_default(
        opt_weights, DEFAULT_WEIGHTS, val_df, horizon
    )
    default_val_ir, _ = _ir_for_weights(DEFAULT_WEIGHTS, val_df, horizon=horizon)

    if n_paired < MIN_PAIRED_VAL_DATES:
        return {"success": False,
                "reason": f"Only {n_paired} paired non-overlapping validation dates available "
                          f"(need >= {MIN_PAIRED_VAL_DATES}) — too few to reliably tell the learned "
                          f"weights apart from chance. Using default weights."}
    if not np.isfinite(val_ir) or val_ir <= 0.0:
        return {"success": False,
                "reason": f"Validation IR {val_ir:+.3f} is not positive — learned weights would anti-predict forward returns out-of-sample."}
    if not passes:
        return {"success": False,
                "reason": f"Learned weights did not significantly beat the default baseline on "
                          f"{n_paired} paired validation dates (mean IC edge {mean_diff:+.4f}, "
                          f"t={t_stat:+.2f}, need t>=1.645). Using default weights."}

    # Shrinkage toward the default mix, proportional to how much of the signal
    # is corroborated out-of-sample vs. in-sample. lambda -> 1 when val_ir is
    # strong relative to train_ir (little overfitting); lambda -> 0 when val_ir
    # barely clears the bar (mostly noise), pulling the deployed weights back
    # toward the even split. w_strat is a no-op under this formula (opt_weights
    # and DEFAULT_WEIGHTS agree on it exactly, being the same fixed value), so
    # it stays untouched by shrinkage as intended.
    denom = val_ir + max(train_ir, 0.0)
    shrink_lambda = float(np.clip(val_ir / denom, 0.0, 1.0)) if denom > 1e-9 else 0.0
    weights = {
        k: shrink_lambda * opt_weights[k] + (1.0 - shrink_lambda) * DEFAULT_WEIGHTS[k]
        for k in DEFAULT_WEIGHTS
    }
    # Renormalize (the convex combination of two simplex points is already on
    # the simplex, but guard float drift explicitly).
    total_w = sum(weights.values())
    if total_w > 1e-9:
        weights = {k: v / total_w for k, v in weights.items()}

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
        "success": True,
        "reason": f"Optimized {n_trials} trials; beat default on {n_paired} paired validation dates "
                  f"(mean IC edge {mean_diff:+.4f}, t={t_stat:+.2f}; val IR {val_ir:+.3f} vs "
                  f"default {default_val_ir:+.3f}; shrinkage lambda={shrink_lambda:.2f}).",
        "weights": weights,
        "train_ir": train_ir,
        "val_ir": val_ir,
        "n_train_dates": len(train_dates),
        "n_val_dates": len(val_dates),
        "default_val_ir": default_val_ir,
        "shrinkage": shrink_lambda,
        "t_stat": t_stat,
        "n_paired_val_dates": n_paired,
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
    "REGIME_FAMILIES",
    "regime_family_for",
    "regime_labels_for_window",
    "regime_labels_from_series",
]
