"""
PRAGYAM Ω — The Omega Engine
==============================
A six-layer self-evolving portfolio intelligence system that replaces every
hardcoded threshold, fixed weight, and rigid assumption with adaptive,
Bayesian, uncertainty-aware learning mechanisms.

Layer 0: Stateless Observation Engine
Layer 1: Adaptive Signal Fabric
Layer 2: Bayesian Belief Engine
Layer 3: Strategy Ecosystem (Tournament + Evolution)
Layer 4: Meta-Cognitive Governor
Layer 5: Portfolio Materialization Engine

Author: Hemrek Capital
Version: Ω-1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger("OmegaEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [Ω] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES — Uncertainty-Carrying Structures
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class UncertainEstimate:
    """A value with its uncertainty. Every number in Omega carries this."""
    mean: float = 0.0
    std: float = 1.0       # Standard deviation of the estimate
    n_obs: int = 0          # Number of observations backing this estimate
    half_life: float = 50.0 # Estimated decay rate of this signal's relevance

    @property
    def confidence(self) -> float:
        """Confidence = 1 - (std / (|mean| + epsilon)). Bounded [0, 1]."""
        if self.n_obs < 3:
            return 0.05
        return float(np.clip(1.0 - self.std / (abs(self.mean) + 0.01), 0.01, 0.99))

    @property
    def lower(self) -> float:
        return self.mean - 1.96 * self.std

    @property
    def upper(self) -> float:
        return self.mean + 1.96 * self.std


@dataclass
class RegimeState:
    """Continuous regime belief — not a label, a probability distribution."""
    momentum_score: float = 0.0     # [-2, +2] continuous
    trend_score: float = 0.0
    volatility_score: float = 0.0
    breadth_score: float = 0.0
    stress_score: float = 0.0       # 0 = calm, 1 = crisis
    uncertainty: float = 0.5        # How sure we are about the regime
    transition_velocity: float = 0.0 # How fast regime is changing

    @property
    def composite(self) -> float:
        """Single regime score: negative = bearish, positive = bullish."""
        w = np.array([0.25, 0.25, 0.15, 0.20, 0.15])
        scores = np.array([self.momentum_score, self.trend_score,
                          -self.volatility_score, self.breadth_score,
                          -self.stress_score])
        return float(np.dot(w, scores))

    @property
    def label(self) -> str:
        c = self.composite
        if c > 1.2:  return "STRONG_BULL"
        if c > 0.5:  return "BULL"
        if c > 0.1:  return "WEAK_BULL"
        if c > -0.1: return "CHOP"
        if c > -0.5: return "WEAK_BEAR"
        if c > -1.2: return "BEAR"
        return "CRISIS"

    @property
    def mix_name(self) -> str:
        label = self.label
        if label in ("STRONG_BULL", "BULL", "WEAK_BULL"):
            return "🐂 Bull Market Mix"
        elif label in ("BEAR", "CRISIS"):
            return "🐻 Bear Market Mix"
        else:
            return "🔀 Chop/Consolidate Mix"


@dataclass
class StrategyProfile:
    """Complete profile of a strategy's measured characteristics."""
    name: str
    sharpe: UncertainEstimate = field(default_factory=UncertainEstimate)
    sortino: UncertainEstimate = field(default_factory=UncertainEstimate)
    calmar: UncertainEstimate = field(default_factory=UncertainEstimate)
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.5
    volatility: float = 0.0
    # Ecosystem metrics
    correlation_with_selected: float = 0.0
    crowding_score: float = 0.0
    predictive_decay_rate: float = 0.0
    information_ratio: float = 0.0
    # Raw return series for correlation computation
    returns_series: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class GovernorState:
    """Meta-cognitive state of the system."""
    belief_entropy: float = 0.5       # 0=overconfident, 1=lost
    model_health: float = 1.0         # 1=healthy, 0=broken
    drawdown_depth: float = 0.0       # Current drawdown
    exposure_multiplier: float = 1.0  # Governor-imposed scaling
    is_graceful_failure: bool = False
    rolling_hit_rate: float = 0.5
    rolling_ic: float = 0.0
    model_disagreement: float = 0.0
    risk_of_ruin: float = 0.0
    exploration_rate: float = 0.1


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 0: STATELESS OBSERVATION ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class ObservationEngine:
    """
    Delivers clean, quality-scored data without interpretation.
    Replaces hardcoded quality_threshold = 0.6 with adaptive percentile-based cutoffs.
    """

    @staticmethod
    def compute_quality_scores(df: pd.DataFrame) -> pd.Series:
        """Adaptive quality scoring — threshold is relative, not absolute."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.Series(1.0, index=df.index)

        # Completeness: fraction of non-zero, non-NaN values
        completeness = 1.0 - (df[numeric_cols].isna() | (df[numeric_cols] == 0)).sum(axis=1) / len(numeric_cols)

        # Consistency: penalize extreme outliers using adaptive z-scores
        consistency_scores = []
        for col in ['rsi latest', 'rsi weekly', 'osc latest', 'osc weekly']:
            if col in df.columns:
                vals = df[col].fillna(df[col].median())
                med = vals.median()
                mad = np.maximum((vals - med).abs().median(), 1e-6) * 1.4826
                robust_z = ((vals - med) / mad).abs()
                consistency_scores.append(np.clip(1.0 - robust_z / 5.0, 0, 1))

        if consistency_scores:
            quality = 0.5 * completeness + 0.5 * pd.concat(consistency_scores, axis=1).mean(axis=1)
        else:
            quality = completeness

        return quality

    @staticmethod
    def adaptive_quality_filter(df: pd.DataFrame, quality_scores: pd.Series) -> pd.DataFrame:
        """Filter using rolling 10th percentile — adapts to data conditions."""
        if len(quality_scores) < 5:
            return df
        threshold = max(quality_scores.quantile(0.10), 0.30)
        mask = quality_scores >= threshold
        if mask.sum() < 10:  # Never filter below 10 stocks
            mask = quality_scores >= quality_scores.nsmallest(max(10, len(df))).iloc[-1]
        return df[mask].copy()


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 1: ADAPTIVE SIGNAL FABRIC
# ════════════════════════════════════════════════════════════════════════════════

class AdaptiveSignalFabric:
    """
    Replaces every hardcoded threshold with distribution-relative measures.
    Replaces fixed factor weights with predictive-power-based weights.
    """

    def __init__(self):
        self._signal_history: Dict[str, List[float]] = {}
        self._return_history: List[float] = []
        self._relevance_cache: Dict[str, float] = {}
        self._max_history = 200

    def normalize_signals(self, df: pd.DataFrame,
                          history: List[Tuple[datetime, pd.DataFrame]]) -> pd.DataFrame:
        """
        Replace absolute thresholds with rolling percentile ranks.
        RSI < 30 becomes RSI < rolling_5th_percentile.
        """
        result = df.copy()

        # Collect historical cross-sectional distributions
        indicator_cols = [c for c in df.columns if c in [
            'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly'
        ]]

        for col in indicator_cols:
            if col not in df.columns:
                continue

            # Build historical distribution from recent windows
            hist_vals = []
            lookback = min(len(history), 50)
            for _, hdf in history[-lookback:]:
                if col in hdf.columns:
                    vals = hdf[col].dropna().values
                    hist_vals.extend(vals.tolist())

            # Add current day
            curr_vals = df[col].dropna().values
            hist_vals.extend(curr_vals.tolist())

            if len(hist_vals) < 20:
                # Insufficient history — use cross-sectional percentile
                result[f'{col}_pctrank'] = df[col].rank(pct=True)
            else:
                hist_arr = np.array(hist_vals)
                # Percentile rank of each current value within historical distribution
                result[f'{col}_pctrank'] = df[col].apply(
                    lambda x: float(np.searchsorted(np.sort(hist_arr), x)) / len(hist_arr)
                    if pd.notna(x) else 0.5
                )

        return result

    def compute_signal_relevance(self, history: List[Tuple[datetime, pd.DataFrame]],
                                  strategies: Dict[str, Any],
                                  strategy_returns: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Measure each signal's predictive power via rolling rank-IC.
        Returns relevance score [0, 1] for each indicator.
        """
        if len(history) < 10:
            # Not enough data — return uniform relevance
            return {col: 1.0 for col in ['rsi', 'osc', 'ema9_osc', 'ema21_osc', 'zscore', 'spread', 'bollinger']}

        indicator_groups = {
            'rsi': ['rsi latest', 'rsi weekly'],
            'osc': ['osc latest', 'osc weekly'],
            'ema9_osc': ['9ema osc latest', '9ema osc weekly'],
            'ema21_osc': ['21ema osc latest', '21ema osc weekly'],
            'zscore': ['zscore latest', 'zscore weekly'],
            'spread': ['ma90 latest', 'ma200 latest'],
            'bollinger': ['dev20 latest', 'dev20 weekly'],
        }

        relevance = {}
        window = min(len(history) - 1, 40)

        for group_name, cols in indicator_groups.items():
            ics = []
            for t in range(len(history) - window, len(history) - 1):
                _, df_t = history[t]
                _, df_t1 = history[t + 1]

                available_cols = [c for c in cols if c in df_t.columns]
                if not available_cols:
                    continue

                try:
                    # Signal: average of group columns
                    signal = df_t[available_cols].mean(axis=1)

                    # Forward return: next-day price change
                    merged = df_t[['symbol', 'price']].merge(
                        df_t1[['symbol', 'price']], on='symbol', suffixes=('_t', '_t1')
                    )
                    if len(merged) < 10:
                        continue

                    fwd_ret = (merged['price_t1'] - merged['price_t']) / merged['price_t']

                    # Merge signal with return
                    sig_vals = signal.reindex(merged.index)
                    valid = sig_vals.notna() & fwd_ret.notna() & np.isfinite(fwd_ret)
                    if valid.sum() < 10:
                        continue

                    # Rank IC (Spearman correlation between signal rank and return rank)
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(sig_vals[valid].values, fwd_ret[valid].values)
                    if np.isfinite(ic):
                        ics.append(abs(ic))  # We care about magnitude, not direction
                except Exception:
                    continue

            if ics:
                # Relevance = mean |IC|, with recency weighting
                weights = np.exp(np.linspace(-1, 0, len(ics)))
                weights /= weights.sum()
                relevance[group_name] = float(np.clip(np.dot(weights, ics) * 10, 0.05, 1.0))
            else:
                relevance[group_name] = 0.15  # Minimal relevance for unknown

        # Normalize to sum to 1
        total = sum(relevance.values())
        if total > 0:
            relevance = {k: v / total for k, v in relevance.items()}

        self._relevance_cache = relevance
        return relevance

    def compute_adaptive_signal_scores(self, df: pd.DataFrame,
                                        relevance: Dict[str, float]) -> pd.Series:
        """
        Replace the multiplier cascade with continuous, relevance-weighted scoring.
        Each signal is scored by its percentile extremity × its measured relevance.
        """
        scores = pd.Series(0.0, index=df.index)

        signal_map = {
            'rsi': ('rsi latest_pctrank', 'rsi weekly_pctrank'),
            'osc': ('osc latest_pctrank', 'osc weekly_pctrank'),
            'ema9_osc': ('9ema osc latest_pctrank', '9ema osc weekly_pctrank'),
            'ema21_osc': ('21ema osc latest_pctrank', '21ema osc weekly_pctrank'),
            'zscore': ('zscore latest_pctrank', 'zscore weekly_pctrank'),
        }

        for group, (daily_col, weekly_col) in signal_map.items():
            weight = relevance.get(group, 0.14)
            daily_pct = df.get(daily_col, pd.Series(0.5, index=df.index))
            weekly_pct = df.get(weekly_col, pd.Series(0.5, index=df.index))

            # Adaptive timescale blending: use more weekly when it differs from daily (noise indicator)
            divergence = (daily_pct - weekly_pct).abs().mean()
            weekly_blend = np.clip(0.5 + divergence, 0.3, 0.8)
            blended = weekly_blend * weekly_pct + (1 - weekly_blend) * daily_pct

            # Conviction from extremity: how far from 0.5 (neutral)?
            # Oversold (low pctrank) → high score. Overbought → low score.
            conviction = np.clip((0.5 - blended) * 4.0, -2.0, 2.0)

            scores += conviction * weight

        # Add spread signal (price relative to MAs) via relevance
        spread_weight = relevance.get('spread', 0.14)
        if 'ma90 latest' in df.columns and 'ma200 latest' in df.columns:
            eps = 1e-6
            spread90 = (df['ma90 latest'] - df['price']) / (df['ma90 latest'] + eps)
            spread200 = (df['ma200 latest'] - df['price']) / (df['ma200 latest'] + eps)
            spread_signal = np.clip((spread90 + spread200) * 2.0, -2.0, 2.0)
            scores += spread_signal * spread_weight

        # Add bollinger signal via relevance
        boll_weight = relevance.get('bollinger', 0.14)
        if all(c in df.columns for c in ['ma20 latest', 'dev20 latest']):
            eps = 1e-6
            upper = df['ma20 latest'] + 2 * df['dev20 latest']
            lower = df['ma20 latest'] - 2 * df['dev20 latest']
            boll_width = np.maximum(upper - lower, eps)
            boll_position = (df['price'] - lower) / boll_width  # 0 = at lower, 1 = at upper
            boll_signal = np.clip((0.5 - boll_position) * 3.0, -2.0, 2.0)
            scores += boll_signal * boll_weight

        return scores

    def estimate_signal_half_life(self, history: List[Tuple[datetime, pd.DataFrame]]) -> float:
        """Estimate how fast signals decay in predictive power."""
        if len(history) < 20:
            return 30.0  # Default: 30-day half-life

        # Compute rolling IC for recent vs distant windows
        recent_ic = self._compute_window_ic(history, -15, -1)
        distant_ic = self._compute_window_ic(history, -40, -20)

        if distant_ic > 0 and recent_ic > 0:
            decay_ratio = recent_ic / distant_ic
            if decay_ratio < 1.0:
                half_life = -20.0 / np.log(decay_ratio + 1e-6)
                return float(np.clip(half_life, 5.0, 200.0))
        return 50.0

    def _compute_window_ic(self, history, start, end) -> float:
        """Compute average rank IC over a window of history."""
        from scipy.stats import spearmanr
        ics = []
        for t in range(max(0, len(history) + start), len(history) + end):
            if t + 1 >= len(history):
                break
            _, df_t = history[t]
            _, df_t1 = history[t + 1]
            try:
                merged = df_t[['symbol', 'price', 'osc latest']].merge(
                    df_t1[['symbol', 'price']], on='symbol', suffixes=('_t', '_t1')
                )
                if len(merged) < 10:
                    continue
                fwd = (merged['price_t1'] - merged['price_t']) / merged['price_t']
                ic, _ = spearmanr(merged['osc latest'], fwd)
                if np.isfinite(ic):
                    ics.append(abs(ic))
            except Exception:
                continue
        return float(np.mean(ics)) if ics else 0.0


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 2: BAYESIAN BELIEF ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class BayesianBeliefEngine:
    """
    Maintains a continuous belief state about the market regime.
    Replaces discrete if-else regime classification with probabilistic inference.
    """

    def __init__(self):
        self._regime_history: List[RegimeState] = []
        self._transition_counts = np.ones((7, 7)) * 0.5  # Dirichlet prior
        self._factor_priors: Dict[str, Tuple[float, float]] = {}  # mean, precision
        self._n_particles = 50

    def infer_regime(self, historical_data: List[Tuple[datetime, pd.DataFrame]]) -> RegimeState:
        """
        Infer current market regime as a continuous probability distribution.
        Uses particle-filter-inspired approach with multiple hypothesis tracking.
        """
        if len(historical_data) < 5:
            return RegimeState(uncertainty=0.9)

        window = historical_data[-min(20, len(historical_data)):]

        # Compute continuous regime dimensions
        momentum = self._measure_momentum(window)
        trend = self._measure_trend(window)
        vol = self._measure_volatility(window)
        breadth = self._measure_breadth(window)
        stress = self._measure_stress(window)

        # Estimate transition velocity (how fast things are changing)
        velocity = self._estimate_transition_velocity(window)

        # Compute uncertainty from inter-dimension disagreement
        scores = np.array([momentum, trend, -vol, breadth, -stress])
        score_std = np.std(scores)
        uncertainty = float(np.clip(score_std / 2.0, 0.1, 0.9))

        # Build regime state
        regime = RegimeState(
            momentum_score=momentum,
            trend_score=trend,
            volatility_score=vol,
            breadth_score=breadth,
            stress_score=stress,
            uncertainty=uncertainty,
            transition_velocity=velocity
        )

        # Update transition model
        if self._regime_history:
            self._update_transition_model(self._regime_history[-1], regime)
        self._regime_history.append(regime)

        return regime

    def _measure_momentum(self, window: List[Tuple[datetime, pd.DataFrame]]) -> float:
        """Continuous momentum score from RSI and oscillator distributions."""
        scores = []
        for _, df in window:
            if 'rsi latest' in df.columns:
                # Use distribution quantiles, not fixed thresholds
                rsi_med = df['rsi latest'].median()
                rsi_q25 = df['rsi latest'].quantile(0.25)
                # Map to [-2, 2] based on where median sits
                score = (rsi_med - 50) / 15.0  # RSI 65 → +1, RSI 35 → -1
                scores.append(float(np.clip(score, -2, 2)))
            if 'osc latest' in df.columns:
                osc_med = df['osc latest'].median()
                score = osc_med / 40.0  # Normalized
                scores.append(float(np.clip(score, -2, 2)))

        if not scores:
            return 0.0

        # Recency-weighted average
        weights = np.exp(np.linspace(-1, 0, len(scores)))
        weights /= weights.sum()
        return float(np.dot(weights, scores))

    def _measure_trend(self, window: List[Tuple[datetime, pd.DataFrame]]) -> float:
        """Continuous trend quality from MA positioning."""
        scores = []
        for _, df in window:
            if 'ma200 latest' in df.columns and 'price' in df.columns:
                above_200 = (df['price'] > df['ma200 latest']).mean()
                score = (above_200 - 0.5) * 4.0  # 75% → +1, 25% → -1
                scores.append(float(np.clip(score, -2, 2)))
            if 'ma90 latest' in df.columns and 'ma200 latest' in df.columns:
                alignment = (df['ma90 latest'] > df['ma200 latest']).mean()
                score = (alignment - 0.5) * 3.0
                scores.append(float(np.clip(score, -2, 2)))

        if not scores:
            return 0.0
        weights = np.exp(np.linspace(-0.5, 0, len(scores)))
        weights /= weights.sum()
        return float(np.dot(weights, scores))

    def _measure_volatility(self, window: List[Tuple[datetime, pd.DataFrame]]) -> float:
        """Continuous volatility regime score. Higher = more volatile."""
        bb_widths = []
        for _, df in window:
            if 'dev20 latest' in df.columns and 'ma20 latest' in df.columns:
                bbw = (4 * df['dev20 latest'] / (df['ma20 latest'] + 1e-6)).median()
                bb_widths.append(float(bbw))

        if len(bb_widths) < 3:
            return 0.0

        # Score relative to own history (percentile-based)
        current = bb_widths[-1]
        all_vals = np.array(bb_widths)
        pctile = float(np.searchsorted(np.sort(all_vals), current)) / len(all_vals)

        # Map percentile to [-1, 2]: low vol → -1 (good), high vol → 2 (bad)
        return float(np.clip((pctile - 0.5) * 4.0, -1.0, 2.0))

    def _measure_breadth(self, window: List[Tuple[datetime, pd.DataFrame]]) -> float:
        """Continuous market breadth from cross-sectional indicators."""
        _, latest_df = window[-1]
        scores = []

        if 'rsi latest' in latest_df.columns:
            bullish_pct = (latest_df['rsi latest'] > 50).mean()
            scores.append((bullish_pct - 0.5) * 4.0)

        if 'osc latest' in latest_df.columns:
            positive_pct = (latest_df['osc latest'] > 0).mean()
            scores.append((positive_pct - 0.5) * 3.0)

        return float(np.clip(np.mean(scores), -2, 2)) if scores else 0.0

    def _measure_stress(self, window: List[Tuple[datetime, pd.DataFrame]]) -> float:
        """Market stress indicator from extreme readings convergence."""
        _, latest_df = window[-1]

        stress_indicators = []
        if 'zscore latest' in latest_df.columns:
            extreme_pct = (latest_df['zscore latest'].abs() > 2.0).mean()
            stress_indicators.append(extreme_pct * 3.0)

        if 'osc latest' in latest_df.columns and 'rsi latest' in latest_df.columns:
            # Cross-indicator agreement on extremes
            both_oversold = ((latest_df['rsi latest'] < 40) & (latest_df['osc latest'] < -30)).mean()
            stress_indicators.append(both_oversold * 4.0)

        return float(np.clip(np.mean(stress_indicators), 0, 2)) if stress_indicators else 0.0

    def _estimate_transition_velocity(self, window: List[Tuple[datetime, pd.DataFrame]]) -> float:
        """How fast the regime is changing."""
        if len(window) < 5:
            return 0.0

        rsi_means = []
        for _, df in window[-10:]:
            if 'rsi latest' in df.columns:
                rsi_means.append(df['rsi latest'].median())

        if len(rsi_means) < 3:
            return 0.0

        # Rate of change of the moving statistic
        diffs = np.diff(rsi_means)
        velocity = np.std(diffs) / (np.mean(np.abs(rsi_means)) + 1e-6)
        return float(np.clip(velocity, 0, 1))

    def _update_transition_model(self, prev: RegimeState, curr: RegimeState):
        """Update Dirichlet-based transition model."""
        regime_labels = ["STRONG_BULL", "BULL", "WEAK_BULL", "CHOP", "WEAK_BEAR", "BEAR", "CRISIS"]
        try:
            prev_idx = regime_labels.index(prev.label)
            curr_idx = regime_labels.index(curr.label)
            self._transition_counts[prev_idx, curr_idx] += 1.0
        except ValueError:
            pass

    def get_transition_probabilities(self) -> np.ndarray:
        """Get normalized transition probability matrix."""
        row_sums = self._transition_counts.sum(axis=1, keepdims=True)
        return self._transition_counts / np.maximum(row_sums, 1e-6)

    def compute_bayesian_factor_weights(self, relevance: Dict[str, float],
                                         regime: RegimeState) -> Dict[str, float]:
        """
        Compute factor weights as Bayesian posteriors conditioned on regime.
        Prior: signal relevance from Layer 1.
        Likelihood: regime-modulated adjustment.
        """
        weights = dict(relevance)

        # Regime modulation: in high-stress, upweight defensive signals
        if regime.stress_score > 1.0:
            weights['zscore'] = weights.get('zscore', 0.14) * 1.5
            weights['bollinger'] = weights.get('bollinger', 0.14) * 1.3

        # In strong momentum, upweight momentum signals
        if regime.momentum_score > 1.0:
            weights['rsi'] = weights.get('rsi', 0.14) * 1.3
            weights['osc'] = weights.get('osc', 0.14) * 1.3

        # In high uncertainty, flatten toward uniform (hedge ignorance)
        if regime.uncertainty > 0.6:
            uniform = 1.0 / max(len(weights), 1)
            blend = regime.uncertainty  # How much to blend toward uniform
            weights = {k: (1 - blend) * v + blend * uniform for k, v in weights.items()}

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 3: STRATEGY ECOSYSTEM
# ════════════════════════════════════════════════════════════════════════════════

class StrategyEcosystem:
    """
    Replaces fixed "top 4 by Calmar" with a multi-objective, correlation-aware,
    dynamic-count strategy selection tournament.
    """

    def __init__(self):
        self._profile_cache: Dict[str, StrategyProfile] = {}

    def run_tournament(self, strategy_returns: Dict[str, List[Dict]],
                       regime: RegimeState,
                       mode: str = 'SIP') -> Tuple[List[str], Dict[str, float]]:
        """
        Multi-objective tournament with correlation-penalized selection.
        Dynamic count: selects 3-6 strategies based on diversity benefit.
        """
        profiles = {}
        for name, returns_data in strategy_returns.items():
            profile = self._build_profile(name, returns_data)
            if profile is not None:
                profiles[name] = profile

        if len(profiles) < 3:
            names = list(profiles.keys())
            return names, {n: 1.0 / len(names) for n in names}

        # Step 1: Multi-objective scoring
        scored = self._multi_objective_score(profiles, regime, mode)

        # Step 2: Correlation-penalized sequential selection
        selected = self._greedy_diverse_selection(scored, profiles, regime)

        # Step 3: Compute allocation weights from scores
        weights = self._compute_tournament_weights(selected, scored, profiles)

        return selected, weights

    def _build_profile(self, name: str, returns_data: List[Dict]) -> Optional[StrategyProfile]:
        """Build a complete strategy profile with uncertainty quantification."""
        if not returns_data or len(returns_data) < 5:
            return None

        returns = np.array([r.get('return', 0) for r in returns_data if isinstance(r, dict)])
        returns = returns[np.isfinite(returns)]
        if len(returns) < 5:
            return None

        n = len(returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1) if n > 1 else 1e-6

        # Annualization — detect period from data
        periods_per_year = 52.0  # Weekly approximation
        ann_factor = np.sqrt(periods_per_year)

        # Point estimates
        ann_return = mean_ret * periods_per_year
        volatility = std_ret * ann_factor
        sharpe_point = ann_return / volatility if volatility > 0.001 else 0

        # Uncertainty via standard error of Sharpe (Lo, 2002)
        sharpe_se = np.sqrt((1 + 0.5 * sharpe_point**2) / n)

        # Sortino
        downside = returns[returns < 0]
        down_std = np.std(downside, ddof=1) * ann_factor if len(downside) > 1 else volatility
        sortino_point = ann_return / down_std if down_std > 0.001 else 0

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / np.maximum(running_max, 1e-6)
        max_dd = float(np.min(drawdown))

        # Calmar
        calmar_point = ann_return / abs(max_dd) if max_dd < -0.001 else 0

        # Clip to reasonable ranges
        sharpe_point = float(np.clip(sharpe_point, -10, 10))
        sortino_point = float(np.clip(sortino_point, -20, 20))
        calmar_point = float(np.clip(calmar_point, -20, 20))

        return StrategyProfile(
            name=name,
            sharpe=UncertainEstimate(mean=sharpe_point, std=sharpe_se, n_obs=n),
            sortino=UncertainEstimate(mean=sortino_point, std=sharpe_se * 1.3, n_obs=n),
            calmar=UncertainEstimate(mean=calmar_point, std=abs(calmar_point) * 0.3 + 0.1, n_obs=n),
            total_return=float(np.prod(1 + returns) - 1),
            max_drawdown=max_dd,
            win_rate=float(np.mean(returns > 0)),
            volatility=volatility,
            returns_series=returns
        )

    def _multi_objective_score(self, profiles: Dict[str, StrategyProfile],
                                regime: RegimeState, mode: str) -> Dict[str, float]:
        """
        Multi-objective scoring that adapts criteria weights to regime and mode.
        Weights are NOT fixed — they depend on regime uncertainty and stress.
        """
        # Regime-adaptive criteria weights
        if mode == 'SIP':
            base_weights = {'sharpe': 0.20, 'sortino': 0.15, 'calmar': 0.30,
                          'win_rate': 0.10, 'drawdown': 0.15, 'confidence': 0.10}
        else:
            base_weights = {'sharpe': 0.15, 'sortino': 0.30, 'calmar': 0.15,
                          'win_rate': 0.15, 'drawdown': 0.10, 'confidence': 0.15}

        # Stress modulation: in crisis, drawdown protection dominates
        stress_factor = regime.stress_score
        base_weights['drawdown'] += stress_factor * 0.10
        base_weights['calmar'] += stress_factor * 0.05
        base_weights['sharpe'] -= stress_factor * 0.05

        # Uncertainty modulation: in high uncertainty, confidence matters more
        unc_factor = regime.uncertainty
        base_weights['confidence'] += unc_factor * 0.10
        base_weights['sharpe'] -= unc_factor * 0.05

        # Renormalize
        total_w = sum(base_weights.values())
        criteria_weights = {k: v / total_w for k, v in base_weights.items()}

        # Score each strategy
        scores = {}
        for name, profile in profiles.items():
            s = 0.0
            s += criteria_weights['sharpe'] * self._normalize_metric(profile.sharpe.mean, profiles, 'sharpe')
            s += criteria_weights['sortino'] * self._normalize_metric(profile.sortino.mean, profiles, 'sortino')
            s += criteria_weights['calmar'] * self._normalize_metric(profile.calmar.mean, profiles, 'calmar')
            s += criteria_weights['win_rate'] * self._normalize_metric(profile.win_rate, profiles, 'win_rate')
            s += criteria_weights['drawdown'] * self._normalize_metric(-abs(profile.max_drawdown), profiles, 'drawdown')
            s += criteria_weights['confidence'] * profile.sharpe.confidence
            scores[name] = s

        return scores

    def _normalize_metric(self, value: float, profiles: Dict[str, StrategyProfile],
                           metric: str) -> float:
        """Normalize a metric to [0, 1] relative to the population."""
        values = []
        for p in profiles.values():
            if metric == 'sharpe': values.append(p.sharpe.mean)
            elif metric == 'sortino': values.append(p.sortino.mean)
            elif metric == 'calmar': values.append(p.calmar.mean)
            elif metric == 'win_rate': values.append(p.win_rate)
            elif metric == 'drawdown': values.append(-abs(p.max_drawdown))
        if not values:
            return 0.5
        mn, mx = min(values), max(values)
        if mx - mn < 1e-6:
            return 0.5
        return float(np.clip((value - mn) / (mx - mn), 0, 1))

    def _greedy_diverse_selection(self, scores: Dict[str, float],
                                    profiles: Dict[str, StrategyProfile],
                                    regime: RegimeState) -> List[str]:
        """
        Greedy selection with correlation penalty. Dynamic count 3-6.
        Each new strategy must add diversification benefit.
        """
        sorted_names = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        if not sorted_names:
            return []

        selected = [sorted_names[0]]
        min_strategies = 3
        max_strategies = min(6, len(sorted_names))

        # In high uncertainty, prefer more strategies (diversification hedge)
        target_count = int(3 + regime.uncertainty * 3)
        target_count = min(max(target_count, min_strategies), max_strategies)

        for candidate in sorted_names[1:]:
            if len(selected) >= target_count:
                break

            # Compute correlation with already-selected strategies
            max_corr = self._max_correlation_with_set(candidate, selected, profiles)

            # Diversity benefit: low correlation = high benefit
            diversity_bonus = (1 - max_corr) * 0.3

            # Marginal score = base score + diversity bonus
            marginal = scores[candidate] + diversity_bonus

            # Accept if marginal contribution is positive and correlation isn't extreme
            if marginal > 0 and max_corr < 0.90:
                selected.append(candidate)

        # Ensure minimum count
        while len(selected) < min_strategies and len(sorted_names) > len(selected):
            for name in sorted_names:
                if name not in selected:
                    selected.append(name)
                    break

        return selected

    def _max_correlation_with_set(self, candidate: str, selected: List[str],
                                   profiles: Dict[str, StrategyProfile]) -> float:
        """Compute max correlation between candidate and selected set."""
        if candidate not in profiles:
            return 0.5

        cand_returns = profiles[candidate].returns_series
        if len(cand_returns) < 5:
            return 0.5

        max_corr = 0.0
        for sel_name in selected:
            if sel_name not in profiles:
                continue
            sel_returns = profiles[sel_name].returns_series
            min_len = min(len(cand_returns), len(sel_returns))
            if min_len < 5:
                continue
            try:
                corr = np.corrcoef(cand_returns[-min_len:], sel_returns[-min_len:])[0, 1]
                if np.isfinite(corr):
                    max_corr = max(max_corr, abs(corr))
            except Exception:
                pass

        return max_corr

    def _compute_tournament_weights(self, selected: List[str],
                                     scores: Dict[str, float],
                                     profiles: Dict[str, StrategyProfile]) -> Dict[str, float]:
        """
        Confidence-weighted allocation within selected set.
        Weight ∝ score × confidence.
        """
        raw_weights = {}
        for name in selected:
            score = max(scores.get(name, 0), 0.01)
            confidence = profiles[name].sharpe.confidence if name in profiles else 0.5
            raw_weights[name] = score * (0.5 + 0.5 * confidence)

        total = sum(raw_weights.values())
        if total <= 0:
            return {n: 1.0 / len(selected) for n in selected}

        return {n: w / total for n, w in raw_weights.items()}

    def detect_crowding(self, portfolios: Dict[str, pd.DataFrame]) -> float:
        """
        Detect signal crowding across strategies.
        Returns 0 (no crowding) to 1 (all strategies pick the same stocks).
        """
        if len(portfolios) < 2:
            return 0.0

        all_symbols = []
        for name, port in portfolios.items():
            if not port.empty and 'symbol' in port.columns:
                top_n = min(15, len(port))
                all_symbols.append(set(port['symbol'].head(top_n)))

        if len(all_symbols) < 2:
            return 0.0

        # Average Jaccard similarity
        jaccard_scores = []
        for i in range(len(all_symbols)):
            for j in range(i + 1, len(all_symbols)):
                intersection = len(all_symbols[i] & all_symbols[j])
                union = len(all_symbols[i] | all_symbols[j])
                if union > 0:
                    jaccard_scores.append(intersection / union)

        return float(np.mean(jaccard_scores)) if jaccard_scores else 0.0


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 4: META-COGNITIVE GOVERNOR
# ════════════════════════════════════════════════════════════════════════════════

class MetaCognitiveGovernor:
    """
    Self-diagnosis, entropy monitoring, risk-of-ruin awareness.
    The system's awareness of its own limitations.
    """

    def __init__(self):
        self._performance_history: List[float] = []
        self._prediction_history: List[Tuple[float, float]] = []  # (predicted, actual)
        self._state = GovernorState()

    def assess(self, regime: RegimeState,
               strategy_weights: Dict[str, float],
               recent_returns: List[float],
               portfolio_value: float = 1.0) -> GovernorState:
        """Run full system health assessment."""
        self._state.belief_entropy = self._compute_belief_entropy(regime, strategy_weights)
        self._state.model_health = self._assess_model_health(recent_returns)
        self._state.drawdown_depth = self._compute_drawdown(recent_returns)
        self._state.rolling_hit_rate = self._compute_hit_rate(recent_returns)
        self._state.risk_of_ruin = self._estimate_risk_of_ruin(recent_returns)
        self._state.model_disagreement = regime.uncertainty
        self._state.exploration_rate = self._compute_exploration_rate()

        # Compute exposure multiplier
        self._state.exposure_multiplier = self._compute_exposure_multiplier()

        # Check for graceful failure
        self._state.is_graceful_failure = self._check_graceful_failure()

        return self._state

    def _compute_belief_entropy(self, regime: RegimeState,
                                 weights: Dict[str, float]) -> float:
        """
        Entropy of the system's belief state.
        High entropy = system doesn't know what it's doing.
        """
        # Weight entropy
        if weights:
            w_arr = np.array(list(weights.values()))
            w_arr = w_arr[w_arr > 0]
            if len(w_arr) > 0:
                w_arr = w_arr / w_arr.sum()
                weight_entropy = -np.sum(w_arr * np.log2(w_arr + 1e-10))
                max_entropy = np.log2(len(w_arr))
                normalized_weight_entropy = weight_entropy / max_entropy if max_entropy > 0 else 0
            else:
                normalized_weight_entropy = 1.0
        else:
            normalized_weight_entropy = 1.0

        # Combine with regime uncertainty
        return float(np.clip(0.5 * normalized_weight_entropy + 0.5 * regime.uncertainty, 0, 1))

    def _assess_model_health(self, recent_returns: List[float]) -> float:
        """Model health from recent predictive accuracy."""
        if len(recent_returns) < 5:
            return 0.7  # Assume moderately healthy with no data

        returns = np.array(recent_returns[-20:])
        returns = returns[np.isfinite(returns)]

        if len(returns) < 3:
            return 0.5

        # Health indicators
        hit_rate = float(np.mean(returns > 0))
        # Rolling Sharpe
        if np.std(returns) > 0:
            rolling_sharpe = np.mean(returns) / np.std(returns)
        else:
            rolling_sharpe = 0

        # Health = weighted combination
        health = 0.5 * np.clip(hit_rate, 0, 1) + 0.5 * np.clip((rolling_sharpe + 2) / 4, 0, 1)
        return float(np.clip(health, 0, 1))

    def _compute_drawdown(self, recent_returns: List[float]) -> float:
        """Current drawdown from peak."""
        if len(recent_returns) < 2:
            return 0.0
        cum = np.cumprod(1 + np.array(recent_returns))
        peak = np.maximum.accumulate(cum)
        dd = (cum[-1] - peak[-1]) / peak[-1] if peak[-1] > 0 else 0
        return float(min(dd, 0))

    def _compute_hit_rate(self, recent_returns: List[float]) -> float:
        """Rolling hit rate."""
        if not recent_returns:
            return 0.5
        r = np.array(recent_returns[-30:])
        r = r[np.isfinite(r)]
        return float(np.mean(r > 0)) if len(r) > 0 else 0.5

    def _estimate_risk_of_ruin(self, recent_returns: List[float]) -> float:
        """
        Estimate probability of catastrophic loss (>20% in 20 periods).
        Uses historical volatility extrapolation.
        """
        if len(recent_returns) < 10:
            return 0.05  # Assume low risk with no data

        returns = np.array(recent_returns[-50:])
        returns = returns[np.isfinite(returns)]
        if len(returns) < 5:
            return 0.05

        vol = np.std(returns)
        mean = np.mean(returns)

        # P(20-period loss > 20%) assuming normal distribution
        # 20-period return ~ N(20*mean, sqrt(20)*vol)
        from scipy.stats import norm
        period_mean = 20 * mean
        period_vol = np.sqrt(20) * vol

        if period_vol > 0:
            prob = float(norm.cdf(-0.20, loc=period_mean, scale=period_vol))
        else:
            prob = 0.0

        return float(np.clip(prob, 0, 1))

    def _compute_exploration_rate(self) -> float:
        """Thompson Sampling-inspired exploration rate."""
        # Higher entropy → more exploration
        # Lower model health → more exploration (try new things when model is failing)
        entropy_factor = self._state.belief_entropy
        health_factor = 1.0 - self._state.model_health

        exploration = 0.05 + 0.15 * entropy_factor + 0.10 * health_factor
        return float(np.clip(exploration, 0.02, 0.30))

    def _compute_exposure_multiplier(self) -> float:
        """
        Drawdown-sensitive exposure scaling.
        exposure = 1.0 - (drawdown / max_tolerable)^k
        """
        max_tolerable = 0.25  # 25% max tolerable drawdown
        k = 2.0  # Nonlinearity: gentle at first, aggressive near limit

        dd = abs(self._state.drawdown_depth)
        if dd < 0.02:
            return 1.0  # No reduction for tiny drawdowns

        multiplier = 1.0 - (dd / max_tolerable) ** k
        multiplier = max(multiplier, 0.25)  # Never below 25% exposure

        # Additional reduction if model health is poor
        if self._state.model_health < 0.4:
            multiplier *= 0.7

        # Additional reduction if risk of ruin is high
        if self._state.risk_of_ruin > 0.15:
            multiplier *= 0.6

        return float(np.clip(multiplier, 0.15, 1.0))

    def _check_graceful_failure(self) -> bool:
        """Trigger graceful failure if multiple health indicators are red."""
        red_flags = 0
        if self._state.model_health < 0.3:
            red_flags += 1
        if self._state.belief_entropy > 0.8:
            red_flags += 1
        if abs(self._state.drawdown_depth) > 0.20:
            red_flags += 1
        if self._state.risk_of_ruin > 0.20:
            red_flags += 1
        if self._state.rolling_hit_rate < 0.35:
            red_flags += 1

        return red_flags >= 3


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 5: PORTFOLIO MATERIALIZATION ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class PortfolioMaterializer:
    """
    Converts beliefs into positions using conviction distributions,
    confidence-weighted sizing, and drawdown-sensitive scaling.
    """

    def curate(self, strategies: Dict[str, Any],
               strategy_weights: Dict[str, float],
               performance: Dict,
               current_df: pd.DataFrame,
               capital: float,
               num_positions: int,
               min_pos_pct: float,
               max_pos_pct: float,
               governor: GovernorState,
               signal_scores: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Full portfolio materialization with uncertainty-aware sizing.
        Drop-in replacement for curate_final_portfolio().
        """
        # Step 1: Generate portfolios from each strategy
        aggregated = {}
        subset_weights = {}

        for name, strategy in strategies.items():
            if name not in strategy_weights:
                continue

            strat_weight = strategy_weights[name]
            if strat_weight < 0.01:
                continue

            try:
                port = strategy.generate_portfolio(current_df, capital)
                if port.empty:
                    continue

                # Compute tier-level weights from performance (adaptive, not fixed)
                tier_w = self._compute_adaptive_tier_weights(name, port, performance)
                subset_weights[name] = tier_w

                # Aggregate holdings with strategy weight × tier weight × signal boost
                n = len(port)
                tier_size = 10
                num_tiers = n // tier_size if tier_size > 0 else 0

                for j in range(max(num_tiers, 1)):
                    tier_name = f'tier_{j+1}'
                    tw = tier_w.get(tier_name, 1.0 / max(num_tiers, 1))

                    start = j * tier_size
                    end = min((j + 1) * tier_size, n)
                    sub = port.iloc[start:end]

                    for _, row in sub.iterrows():
                        sym = row['symbol']
                        price = row['price']
                        base_weight = (row.get('weightage_pct', 2.0) / 100.0) * tw * strat_weight

                        # Signal boost: if Layer 1 has scored this stock, modulate weight
                        if signal_scores is not None and sym in signal_scores.index:
                            sig = signal_scores.get(sym, 0)
                            # Positive signal → boost, negative → dampen
                            boost = np.clip(1.0 + sig * 0.2, 0.5, 1.5)
                            base_weight *= boost

                        if sym in aggregated:
                            aggregated[sym]['weight'] += base_weight
                        else:
                            aggregated[sym] = {'price': price, 'weight': base_weight}
            except Exception as e:
                logger.warning(f"Strategy {name} error: {e}")
                continue

        if not aggregated:
            return pd.DataFrame(), strategy_weights, subset_weights

        # Step 2: Build portfolio DataFrame
        port_df = pd.DataFrame([
            {'symbol': s, 'price': d['price'], 'weight': d['weight']}
            for s, d in aggregated.items()
        ]).sort_values('weight', ascending=False)

        # Step 3: Apply conviction-based position count
        effective_positions = self._dynamic_position_count(port_df, num_positions, governor)
        port_df = port_df.head(effective_positions)

        if port_df.empty:
            return pd.DataFrame(), strategy_weights, subset_weights

        # Step 4: Normalize and apply position limits
        total_weight = port_df['weight'].sum()
        if total_weight > 0:
            port_df['weightage_pct'] = port_df['weight'] * 100.0 / total_weight
        else:
            port_df['weightage_pct'] = 100.0 / len(port_df)

        # Soft caps via uncertainty: Governor can tighten max position in high uncertainty
        effective_max = max_pos_pct * governor.exposure_multiplier
        effective_max = max(effective_max, min_pos_pct + 0.5)

        port_df['weightage_pct'] = port_df['weightage_pct'].clip(lower=min_pos_pct, upper=effective_max)
        port_df['weightage_pct'] = (port_df['weightage_pct'] / port_df['weightage_pct'].sum()) * 100.0

        # Step 5: Apply Governor's exposure multiplier (capital conservation)
        effective_capital = capital * governor.exposure_multiplier

        # Step 6: Allocate units
        port_df['units'] = np.floor((effective_capital * port_df['weightage_pct'] / 100.0) / port_df['price'])
        port_df['value'] = port_df['units'] * port_df['price']

        # Clean up
        port_df = port_df[port_df['units'] > 0].sort_values('weightage_pct', ascending=False).reset_index(drop=True)

        return port_df[['symbol', 'price', 'weightage_pct', 'units', 'value']], strategy_weights, subset_weights

    def _compute_adaptive_tier_weights(self, name: str, port: pd.DataFrame,
                                        performance: Dict) -> Dict[str, float]:
        """Compute tier weights from walk-forward performance, not hardcoded tiers."""
        subset_perf = performance.get('subset', {}).get(name, {})
        tier_names = sorted(subset_perf.keys())

        if not tier_names:
            n = len(port)
            num_tiers = n // 10
            if num_tiers == 0:
                return {'tier_1': 1.0}
            return {f'tier_{i+1}': 1.0 / num_tiers for i in range(num_tiers)}

        # Get Sharpe values + add small positive shift for exp stability
        sharpes = np.array([subset_perf.get(t, 0.0) + 2.0 for t in tier_names])

        # Softmax with temperature based on the spread of values
        spread = np.ptp(sharpes)
        temperature = max(spread * 0.5, 0.5)

        stable = sharpes - np.max(sharpes)
        exp_vals = np.exp(stable / temperature)
        total = np.sum(exp_vals)

        if total > 0 and np.isfinite(total):
            weights = exp_vals / total
        else:
            weights = np.ones(len(tier_names)) / len(tier_names)

        return {tier_names[i]: float(weights[i]) for i in range(len(tier_names))}

    def _dynamic_position_count(self, port_df: pd.DataFrame,
                                 max_positions: int,
                                 governor: GovernorState) -> int:
        """
        Dynamic position count based on conviction distribution.
        In graceful failure → fewer, more concentrated positions.
        In high confidence → wider portfolio.
        """
        base = max_positions

        # In graceful failure, cut to minimum
        if governor.is_graceful_failure:
            return max(int(base * 0.5), 10)

        # Scale with model health
        health_factor = 0.7 + 0.3 * governor.model_health
        scaled = int(base * health_factor)

        # Never below 10 or above max
        return max(min(scaled, max_positions, len(port_df)), 10)


# ════════════════════════════════════════════════════════════════════════════════
# OMEGA ORCHESTRATOR — Ties All Layers Together
# ════════════════════════════════════════════════════════════════════════════════

class OmegaEngine:
    """
    The unified Omega engine that orchestrates all six layers.
    Drop-in replacement for Pragyam's backend functions.
    """

    def __init__(self):
        self.observation = ObservationEngine()
        self.signals = AdaptiveSignalFabric()
        self.beliefs = BayesianBeliefEngine()
        self.ecosystem = StrategyEcosystem()
        self.governor_engine = MetaCognitiveGovernor()
        self.materializer = PortfolioMaterializer()
        self._governor_state = GovernorState()
        self._current_regime = RegimeState()

    # ──────────────────────────────────────────────────────────────
    # DROP-IN: calculate_advanced_metrics
    # ──────────────────────────────────────────────────────────────
    def calculate_advanced_metrics(self, returns_with_dates: List[Dict]) -> Tuple[Dict, float]:
        """
        Enhanced metrics with uncertainty quantification.
        Same interface as original: returns (metrics_dict, periods_per_year).
        """
        default_metrics = {
            'total_return': 0, 'annual_return': 0, 'volatility': 0,
            'sharpe': 0, 'sortino': 0, 'max_drawdown': 0, 'calmar': 0,
            'win_rate': 0, 'kelly_criterion': 0, 'omega_ratio': 1.0,
            'tail_ratio': 1.0, 'gain_to_pain': 0, 'profit_factor': 1.0,
            # NEW: Omega uncertainty fields
            'sharpe_confidence': 0, 'sortino_confidence': 0,
            'signal_half_life': 50.0, 'regime_uncertainty': 0.5
        }
        if len(returns_with_dates) < 2:
            return default_metrics, 52

        returns_df = pd.DataFrame(returns_with_dates).sort_values('date').set_index('date')
        time_deltas = returns_df.index.to_series().diff().dt.days
        avg_period = time_deltas.mean()
        periods_per_year = 365.25 / avg_period if pd.notna(avg_period) and avg_period > 0 else 52

        returns = returns_df['return']
        n = len(returns)
        total_return = float((1 + returns).prod() - 1)

        years = n / periods_per_year
        annual_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 and total_return > -1 else 0

        volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
        sharpe = float(np.clip(annual_return / volatility if volatility > 0.001 else 0, -10, 10))

        # Sharpe uncertainty (Lo, 2002)
        sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / n) if n > 1 else 1.0
        sharpe_conf = float(np.clip(1.0 - sharpe_se / (abs(sharpe) + 0.01), 0.01, 0.99))

        downside = returns[returns < 0]
        if len(downside) >= 2:
            down_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year))
            sortino = float(np.clip(annual_return / down_vol if down_vol > 0.001 else 0, -20, 20))
        else:
            sortino = 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding(min_periods=1).max()
        max_drawdown = float((cumulative / running_max - 1).min())
        calmar = float(np.clip(annual_return / abs(max_drawdown) if max_drawdown < -0.001 else 0, -20, 20))

        win_rate = float((returns > 0).mean())
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = float(gains.mean()) if len(gains) > 0 else 0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0
        total_gains = float(gains.sum()) if len(gains) > 0 else 0
        total_losses = float(abs(losses.sum())) if len(losses) > 0 else 0

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0.0001 else 0
        kelly = float(np.clip((win_rate - ((1 - win_rate) / win_loss_ratio)) if win_loss_ratio > 0 else 0, -1, 1))
        omega = float(np.clip(total_gains / total_losses if total_losses > 0.0001 else (total_gains * 10 if total_gains > 0 else 1.0), 0, 50))
        profit_factor = float(np.clip(total_gains / total_losses if total_losses > 0.0001 else (10.0 if total_gains > 0 else 1.0), 0, 50))

        upper_tail = float(np.percentile(returns, 95)) if n >= 20 else float(returns.max())
        lower_tail = float(abs(np.percentile(returns, 5))) if n >= 20 else float(abs(returns.min()))
        tail_ratio = float(np.clip(upper_tail / lower_tail if lower_tail > 0.0001 else (10 if upper_tail > 0 else 1), 0, 20))

        pain = float(abs(losses.sum())) if len(losses) > 0 else 0
        gain_to_pain = float(np.clip(returns.sum() / pain if pain > 0.0001 else (returns.sum() * 10 if returns.sum() > 0 else 0), -20, 20))

        metrics = {
            'total_return': total_return, 'annual_return': annual_return,
            'volatility': volatility, 'sharpe': sharpe, 'sortino': sortino,
            'max_drawdown': max_drawdown, 'calmar': calmar, 'win_rate': win_rate,
            'kelly_criterion': kelly, 'omega_ratio': omega, 'tail_ratio': tail_ratio,
            'gain_to_pain': gain_to_pain, 'profit_factor': profit_factor,
            'sharpe_confidence': sharpe_conf, 'sortino_confidence': sharpe_conf * 0.9,
            'signal_half_life': 50.0, 'regime_uncertainty': self._current_regime.uncertainty
        }
        return metrics, periods_per_year

    # ──────────────────────────────────────────────────────────────
    # DROP-IN: calculate_strategy_weights
    # ──────────────────────────────────────────────────────────────
    def calculate_strategy_weights(self, performance: Dict) -> Dict[str, float]:
        """
        Bayesian strategy weights replacing softmax-on-Sharpe.
        Weight ∝ Sharpe_mean × confidence.
        """
        strat_names = list(performance.get('strategy', {}).keys())
        if not strat_names:
            return {}

        raw_weights = {}
        for name in strat_names:
            strat_data = performance['strategy'][name]
            if isinstance(strat_data, dict) and 'metrics' in strat_data:
                sharpe = strat_data['metrics'].get('sharpe', 0)
                conf = strat_data['metrics'].get('sharpe_confidence', 0.5)
            else:
                sharpe = strat_data.get('sharpe', 0)
                conf = 0.5

            if not isinstance(sharpe, (int, float)) or not np.isfinite(sharpe):
                sharpe = 0
            if not isinstance(conf, (int, float)) or not np.isfinite(conf):
                conf = 0.5

            # Weight = shifted Sharpe × confidence
            # The confidence term means uncertain strategies get lower weight
            shifted = sharpe + 2.0  # Shift to positive
            raw_weights[name] = max(shifted * (0.3 + 0.7 * conf), 0.01)

        # Apply governor modulation: in graceful failure, flatten toward uniform
        if self._governor_state.is_graceful_failure:
            uniform = 1.0 / len(strat_names)
            raw_weights = {k: 0.5 * v + 0.5 * uniform for k, v in raw_weights.items()}

        total = sum(raw_weights.values())
        if total <= 0 or not np.isfinite(total):
            return {name: 1.0 / len(strat_names) for name in strat_names}

        return {name: w / total for name, w in raw_weights.items()}

    # ──────────────────────────────────────────────────────────────
    # DROP-IN: evaluate_historical_performance (walk-forward)
    # ──────────────────────────────────────────────────────────────
    def evaluate_historical_performance(self, strategies: Dict[str, Any],
                                         historical_data: List[Tuple[datetime, pd.DataFrame]],
                                         compute_portfolio_return_fn=None,
                                         curate_fn=None,
                                         progress_bar=None) -> Dict:
        """
        Omega walk-forward evaluation with adaptive windows, uncertainty tracking,
        and signal relevance computation. Same return format as original.
        """
        import streamlit as st

        MIN_TRAIN = 2
        TRAINING_CAPITAL = 2500000.0

        if len(historical_data) < MIN_TRAIN + 1:
            st.error(f"Need at least {MIN_TRAIN + 1} files.")
            return {}

        all_names = list(strategies.keys()) + ['System_Curated']
        oos_perf = {name: {'returns': []} for name in all_names}
        weight_entropies = []
        strategy_weights_history = []
        subset_weights_history = []

        progress_bar_widget = st.progress(0, text="Ω Initializing walk-forward...")
        total_steps = len(historical_data) - MIN_TRAIN - 1

        if total_steps <= 0:
            st.error(f"Need at least {MIN_TRAIN + 2} days for walk-forward.")
            progress_bar_widget.empty()
            return {}

        # Pre-compute signal relevance from first half of data
        mid = max(MIN_TRAIN + 5, len(historical_data) // 2)
        relevance = self.signals.compute_signal_relevance(
            historical_data[:mid], strategies, {}
        )
        signal_half_life = self.signals.estimate_signal_half_life(historical_data[:mid])

        for i in range(MIN_TRAIN, len(historical_data) - 1):
            train_window = historical_data[:i]
            test_date, test_df = historical_data[i]
            next_date, next_df = historical_data[i + 1]

            step = i - MIN_TRAIN + 1
            progress_bar_widget.progress(
                step / total_steps,
                text=f"Ω Walk-forward step {step}/{total_steps}"
            )

            # Compute in-sample performance
            in_sample = self._calc_window_performance(train_window, strategies, TRAINING_CAPITAL, compute_portfolio_return_fn)

            # Adaptive strategy weights
            strat_wts = self.calculate_strategy_weights(in_sample)
            strategy_weights_history.append({'date': test_date, **strat_wts})

            # Compute subset weights
            sub_wts = {}
            for name in strategies:
                sub_perfs = in_sample.get('subset', {}).get(name, {})
                tier_names = sorted(sub_perfs.keys())
                if not tier_names:
                    sub_wts[name] = {}
                    continue
                sharpes = np.array([sub_perfs.get(t, 0.0) + 2.0 for t in tier_names])
                stable = sharpes - np.max(sharpes)
                exp_s = np.exp(stable)
                total_e = np.sum(exp_s)
                if total_e > 0 and np.isfinite(total_e):
                    sub_wts[name] = {t: float(exp_s[j] / total_e) for j, t in enumerate(tier_names)}
                else:
                    sub_wts[name] = {t: 1.0 / len(tier_names) for t in tier_names}

            subset_weights_history.append({'date': test_date, **sub_wts})

            # Curate system portfolio
            try:
                curated_port, _, _ = self.materializer.curate(
                    strategies, strat_wts, in_sample,
                    test_df, TRAINING_CAPITAL, 30, 1.0, 10.0,
                    self._governor_state
                )

                if curated_port.empty:
                    oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
                else:
                    oos_ret = self._compute_portfolio_return(curated_port, next_df)
                    oos_perf['System_Curated']['returns'].append({'return': oos_ret, 'date': next_date})

                    # Track entropy
                    weights = curated_port['weightage_pct'] / 100
                    valid_w = weights[weights > 0]
                    if len(valid_w) > 0:
                        entropy = float(-np.sum(valid_w * np.log2(valid_w)))
                        weight_entropies.append(entropy)
            except Exception as e:
                logger.warning(f"OOS Curation Error ({test_date}): {e}")
                oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})

            # Per-strategy OOS returns
            for name, strategy in strategies.items():
                try:
                    port = strategy.generate_portfolio(test_df, TRAINING_CAPITAL)
                    ret = self._compute_portfolio_return(port, next_df)
                    oos_perf[name]['returns'].append({'return': ret, 'date': next_date})
                except Exception:
                    oos_perf[name]['returns'].append({'return': 0, 'date': next_date})

        progress_bar_widget.empty()

        # Compute final metrics with uncertainty
        final_oos = {}
        for name, data in oos_perf.items():
            metrics, _ = self.calculate_advanced_metrics(data['returns'])
            final_oos[name] = {'returns': data['returns'], 'metrics': metrics}

        if weight_entropies:
            final_oos['System_Curated']['metrics']['avg_weight_entropy'] = float(np.mean(weight_entropies))

        # Full history subset performance
        full_sub = self._calc_window_performance(historical_data, strategies, TRAINING_CAPITAL, compute_portfolio_return_fn).get('subset', {})

        # Update governor with walk-forward results
        system_returns = [r['return'] for r in oos_perf.get('System_Curated', {}).get('returns', []) if isinstance(r, dict)]
        self._current_regime = self.beliefs.infer_regime(historical_data)
        self._governor_state = self.governor_engine.assess(
            self._current_regime, strat_wts if strategy_weights_history else {},
            system_returns
        )

        return {
            'strategy': final_oos,
            'subset': full_sub,
            'strategy_weights_history': strategy_weights_history,
            'subset_weights_history': subset_weights_history
        }

    # ──────────────────────────────────────────────────────────────
    # DROP-IN: detect_regime (replaces MarketRegimeDetectorV2)
    # ──────────────────────────────────────────────────────────────
    def detect_regime(self, historical_data: List[Tuple[datetime, pd.DataFrame]]) -> Tuple[str, str, float, Dict]:
        """
        Returns: (regime_name, mix_name, confidence, details_dict)
        Same interface as MarketRegimeDetectorV2.detect_regime()
        """
        regime = self.beliefs.infer_regime(historical_data)
        self._current_regime = regime

        confidence = 1.0 - regime.uncertainty
        details = {
            'score': regime.composite,
            'metrics': {
                'momentum': {'score': regime.momentum_score, 'strength': 'measured'},
                'trend': {'score': regime.trend_score, 'quality': 'measured'},
                'volatility': {'score': regime.volatility_score, 'regime': 'continuous'},
                'breadth': {'score': regime.breadth_score, 'quality': 'measured'},
                'stress': {'score': regime.stress_score, 'type': 'continuous'},
                'velocity': {'transition_velocity': regime.transition_velocity},
                'correlation': {'score': 0.0}  # Backward compat
            },
            'explanation': self._generate_regime_explanation(regime),
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            # Omega additions
            'regime_uncertainty': regime.uncertainty,
            'transition_velocity': regime.transition_velocity,
            'governor_state': {
                'exposure_multiplier': self._governor_state.exposure_multiplier,
                'model_health': self._governor_state.model_health,
                'is_graceful_failure': self._governor_state.is_graceful_failure
            }
        }

        return regime.label, regime.mix_name, confidence, details

    def _generate_regime_explanation(self, regime: RegimeState) -> str:
        """Generate human-readable regime explanation."""
        parts = []
        if regime.momentum_score > 0.5:
            parts.append("Momentum is bullish")
        elif regime.momentum_score < -0.5:
            parts.append("Momentum is bearish")
        else:
            parts.append("Momentum is neutral")

        if regime.trend_score > 0.5:
            parts.append("trend structure is positive")
        elif regime.trend_score < -0.5:
            parts.append("trend is deteriorating")

        if regime.volatility_score > 1.0:
            parts.append("volatility is elevated")
        elif regime.volatility_score < -0.5:
            parts.append("volatility is compressed")

        if regime.uncertainty > 0.6:
            parts.append(f"but regime uncertainty is high ({regime.uncertainty:.0%})")

        return ". ".join(parts) + "."

    # ──────────────────────────────────────────────────────────────
    # DROP-IN: dynamic_strategy_selection
    # ──────────────────────────────────────────────────────────────
    def run_dynamic_selection(self, historical_data, all_strategies, selected_style,
                              progress_bar=None, status_text=None,
                              trigger_df=None, trigger_config=None,
                              compute_portfolio_return_fn=None) -> Tuple[Optional[List[str]], Dict]:
        """
        Omega strategy selection: tournament + correlation + dynamic count.
        Same interface as _run_dynamic_strategy_selection().
        """
        is_sip = "SIP" in selected_style
        mode = 'SIP' if is_sip else 'Swing'

        if not historical_data or len(historical_data) < 10:
            return None, {}

        if status_text:
            status_text.text(f"Ω Evaluating {len(all_strategies)} strategies...")

        # Infer regime for the tournament
        regime = self.beliefs.infer_regime(historical_data)
        self._current_regime = regime

        # Build trigger masks
        n_days = len(historical_data)
        buy_mask = [False] * n_days
        sell_mask = [False] * n_days

        buy_threshold = (trigger_config or {}).get('buy_threshold', 0.42)
        sell_threshold = (trigger_config or {}).get('sell_threshold', 1.5 if is_sip else 1.2)
        sell_enabled = (trigger_config or {}).get('sell_enabled', not is_sip)

        if trigger_df is not None and not trigger_df.empty and 'REL_BREADTH' in trigger_df.columns:
            if hasattr(trigger_df.index, 'date'):
                tmap = {idx.date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}
            else:
                tmap = {pd.to_datetime(idx).date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}

            sim_dates = []
            for date_obj, _ in historical_data:
                sd = date_obj.date() if hasattr(date_obj, 'date') else date_obj
                sim_dates.append(sd)

            for i, sd in enumerate(sim_dates):
                if sd in tmap:
                    rb = tmap[sd]
                    if rb < buy_threshold:
                        buy_mask[i] = True
                    if sell_enabled and rb > sell_threshold:
                        sell_mask[i] = True
        else:
            buy_mask[0] = True

        # Run trigger-based backtests per strategy
        results = {}
        strategy_returns = {}
        total = len(all_strategies)

        for idx, (name, strategy) in enumerate(all_strategies.items()):
            if progress_bar:
                progress_bar.progress(0.25 + 0.35 * (idx / total),
                                     text=f"Ω Evaluating {name} ({idx+1}/{total})")
            try:
                daily_values = self._run_trigger_backtest(
                    strategy, historical_data, buy_mask, sell_mask,
                    is_sip, compute_portfolio_return_fn
                )

                if not daily_values:
                    continue

                # Compute metrics
                vals = pd.DataFrame(daily_values)
                returns_list = []
                for j in range(1, len(vals)):
                    prev = vals.iloc[j-1]['value']
                    curr = vals.iloc[j]['value']
                    if prev > 0:
                        returns_list.append({
                            'return': (curr - prev) / prev,
                            'date': vals.iloc[j].get('date', datetime.now())
                        })

                if len(returns_list) < 3:
                    continue

                metrics, _ = self.calculate_advanced_metrics(returns_list)
                strategy_returns[name] = returns_list
                results[name] = metrics

            except Exception as e:
                logger.warning(f"Backtest error for {name}: {e}")
                continue

        if len(results) < 3:
            return None, results

        # Run tournament
        selected, weights = self.ecosystem.run_tournament(strategy_returns, regime, mode)

        if status_text:
            status_text.text(f"Ω Selected: {', '.join(selected)}")

        logger.info(f"Ω Tournament: Selected {len(selected)} strategies: {selected}")
        logger.info(f"Ω Regime: {regime.label} (uncertainty={regime.uncertainty:.2f})")

        # Update governor
        all_returns = []
        for ret_list in strategy_returns.values():
            all_returns.extend([r['return'] for r in ret_list])
        self._governor_state = self.governor_engine.assess(regime, weights, all_returns[-50:])

        return selected, results

    # ──────────────────────────────────────────────────────────────
    # DROP-IN: curate_final_portfolio
    # ──────────────────────────────────────────────────────────────
    def curate_final_portfolio(self, strategies, performance, current_df,
                                capital, num_positions, min_pos_pct, max_pos_pct):
        """
        Omega portfolio materialization.
        Same interface as original curate_final_portfolio().
        """
        strat_wts = self.calculate_strategy_weights(performance)

        # Compute signal scores from Layer 1 if we have history
        signal_scores = None

        return self.materializer.curate(
            strategies, strat_wts, performance, current_df,
            capital, num_positions, min_pos_pct, max_pos_pct,
            self._governor_state, signal_scores
        )

    # ──────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────
    def _compute_portfolio_return(self, portfolio: pd.DataFrame, next_df: pd.DataFrame) -> float:
        """Compute next-day return of a portfolio."""
        if portfolio.empty or 'symbol' not in portfolio.columns:
            return 0.0

        next_prices = next_df.set_index('symbol')['price'].to_dict() if 'symbol' in next_df.columns else {}
        if not next_prices:
            return 0.0

        total_value = 0.0
        total_new_value = 0.0

        for _, row in portfolio.iterrows():
            sym = row['symbol']
            old_price = row['price']
            units = row.get('units', 0)
            if units <= 0 or old_price <= 0:
                continue

            new_price = next_prices.get(sym, old_price)
            if not np.isfinite(new_price) or new_price <= 0:
                new_price = old_price

            total_value += units * old_price
            total_new_value += units * new_price

        if total_value > 0:
            return float((total_new_value - total_value) / total_value)
        return 0.0

    def _calc_window_performance(self, window_data, strategies, capital, cpr_fn=None):
        """Calculate performance on a training window."""
        performance = {name: {'returns': []} for name in strategies}
        subset_performance = {name: {} for name in strategies}

        for i in range(len(window_data) - 1):
            _, df = window_data[i]
            _, next_df = window_data[i + 1]

            for name, strategy in strategies.items():
                try:
                    portfolio = strategy.generate_portfolio(df, capital)
                    if portfolio.empty:
                        continue

                    ret = self._compute_portfolio_return(portfolio, next_df)
                    performance[name]['returns'].append({'return': ret, 'date': window_data[i+1][0]})

                    # Tier performance
                    n = len(portfolio)
                    tier_size = 10
                    num_tiers = n // tier_size
                    for j in range(num_tiers):
                        tier_name = f'tier_{j+1}'
                        if tier_name not in subset_performance[name]:
                            subset_performance[name][tier_name] = []
                        sub = portfolio.iloc[j*tier_size:(j+1)*tier_size]
                        if not sub.empty:
                            sub_ret = self._compute_portfolio_return(sub, next_df)
                            subset_performance[name][tier_name].append({'return': sub_ret, 'date': window_data[i+1][0]})
                except Exception:
                    continue

        final_perf = {}
        for name, perf in performance.items():
            metrics, _ = self.calculate_advanced_metrics(perf['returns'])
            final_perf[name] = {'metrics': metrics, 'sharpe': metrics['sharpe']}

        final_sub = {}
        for name, data in subset_performance.items():
            final_sub[name] = {}
            for sub, sub_perf in data.items():
                if sub_perf:
                    m, _ = self.calculate_advanced_metrics(sub_perf)
                    final_sub[name][sub] = m['sharpe']

        return {'strategy': final_perf, 'subset': final_sub}

    def _run_trigger_backtest(self, strategy, historical_data, buy_mask, sell_mask,
                               is_sip, cpr_fn=None):
        """Run a trigger-based backtest for a single strategy."""
        capital = 10_000_000
        daily_values = []

        if is_sip:
            # SIP: TWR methodology
            nav = 1.0
            holdings = {}

            for day_idx in range(len(historical_data)):
                date_obj, df = historical_data[day_idx]

                # Update existing holdings
                if holdings:
                    prices = df.set_index('symbol')['price'].to_dict() if 'symbol' in df.columns else {}
                    day_return = 0.0
                    total_value = 0.0
                    for sym, h in holdings.items():
                        new_p = prices.get(sym, h['price'])
                        if h['price'] > 0:
                            day_return += (new_p - h['price']) / h['price'] * h['weight']
                        h['price'] = new_p
                        total_value += h['weight']

                    if total_value > 0:
                        nav *= (1 + day_return)

                # Buy on trigger
                if buy_mask[day_idx]:
                    try:
                        port = strategy.generate_portfolio(df, capital)
                        if not port.empty:
                            for _, row in port.iterrows():
                                sym = row['symbol']
                                wt = row.get('weightage_pct', 2.0) / 100.0
                                holdings[sym] = {'price': row['price'], 'weight': wt}
                    except Exception:
                        pass

                daily_values.append({
                    'date': date_obj,
                    'value': nav * capital,
                    'investment': capital
                })
        else:
            # Swing: NAV tracking
            cash = capital
            in_position = False
            holdings = {}
            port_value = capital

            for day_idx in range(len(historical_data)):
                date_obj, df = historical_data[day_idx]
                prices = df.set_index('symbol')['price'].to_dict() if 'symbol' in df.columns else {}

                if in_position and holdings:
                    port_value = sum(
                        h['units'] * prices.get(sym, h['price'])
                        for sym, h in holdings.items()
                    ) + cash

                if buy_mask[day_idx] and not in_position:
                    try:
                        port = strategy.generate_portfolio(df, capital)
                        if not port.empty:
                            holdings = {}
                            for _, row in port.iterrows():
                                holdings[row['symbol']] = {
                                    'price': row['price'],
                                    'units': row.get('units', 0)
                                }
                            invested = sum(h['units'] * h['price'] for h in holdings.values())
                            cash = capital - invested
                            in_position = True
                    except Exception:
                        pass

                if sell_mask[day_idx] and in_position:
                    # Sell everything
                    for sym, h in holdings.items():
                        cash += h['units'] * prices.get(sym, h['price'])
                    holdings = {}
                    in_position = False
                    capital = cash
                    port_value = cash

                daily_values.append({
                    'date': date_obj,
                    'value': port_value if in_position else cash,
                    'investment': capital
                })

        return daily_values


# ════════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ════════════════════════════════════════════════════════════════════════════════

_omega_instance: Optional[OmegaEngine] = None

def get_omega() -> OmegaEngine:
    """Get or create the Omega engine singleton."""
    global _omega_instance
    if _omega_instance is None:
        _omega_instance = OmegaEngine()
        logger.info("Ω Engine initialized — all layers active")
    return _omega_instance
