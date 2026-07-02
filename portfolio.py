"""
PRAGYAM — Portfolio Construction Engine
══════════════════════════════════════════════════════════════════════════════

Conviction-based portfolio curation using regime.py signal scoring.

Formula: weight_i = (conviction_score_i / Σ all_conviction_scores) × 100

Conviction Dispersion Weighting: a CONTINUOUS rank-power transform
(adjusted = conviction * rank_pct ** gamma) concentrates capital in
high-conviction names without a discontinuous cliff at the median — see
_apply_dispersion's docstring for the rationale (a median-step boost/penalty
created an ~13x weight ratio between two names one conviction point apart).
  → SIP Mode:   gamma = 2.0 (moderate concentration)
  → Swing Mode: gamma = 3.0 (aggressive concentration)

Selection: top-N by conviction, ties broken by value-area position (prefer a
discount to accepted value). See regime.compute_conviction_signals for scoring.

Author: @thebullishvalue
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from regime import compute_conviction_signals


def compute_conviction_based_weights(
    aggregated_holdings: Dict[str, Dict],
    current_df: pd.DataFrame,
    sip_amount: float,
    num_positions: int,
    min_pos_pct: float,
    max_pos_pct: float,
    apply_dispersion: bool = True,
    investment_style: str = "SIP Investment",
    dispersion_params: Optional[tuple] = None,  # (boost_multiplier, penalty_multiplier)
    universe: str = "default",
    selected_index: Optional[str] = None,
    regime_name: str = "NEUTRAL",
    mode: str = "Standard",
) -> pd.DataFrame:
    """
    Compute portfolio weights based on conviction scores.

    Uses regime.py::compute_conviction_signals() to score all candidates,
    then applies the conviction-based weighting formula.

    Process:
    1. Build temporary portfolio DataFrame from aggregated_holdings
    2. Compute conviction scores using regime.py (RSI, OSC, Z-Score, MA, VAP, Strategy)
    3. Select top num_positions by conviction score (NO threshold filter),
       breaking ties by value-area position (prefer a discount to accepted value)
    4. Apply conviction dispersion weighting: adjusted = conviction_score ** gamma
       (continuous in the score — see the module docstring for why this replaced
       the old median-step boost/penalty)
    5. Apply formula: weight_i = (adjusted_conviction_i / Σ all_adjusted) × 100
    6. Apply bounds (min/max position limits), relaxed for feasibility when
       num_positions makes the nominal bounds mathematically unsatisfiable
    7. Calculate units and value

    Args:
        aggregated_holdings: Dict of symbol → {price, weight}
        current_df: Current indicator data for conviction computation
        sip_amount: Total capital to allocate
        num_positions: Maximum number of positions to hold
        min_pos_pct: Minimum position weight (e.g., 0.01 = 1%)
        max_pos_pct: Maximum position weight (e.g., 0.10 = 10%)
        apply_dispersion: Whether to apply conviction dispersion weighting (default: True)
        investment_style: "SIP Investment" or "Swing Trading"
            - SIP: gamma = 2.5 (moderate concentration)
            - Swing: gamma = 4.5 (aggressive concentration)
        dispersion_params: Optional gamma override (float). A legacy
            (boost_multiplier, penalty_multiplier) tuple is also accepted for
            backward compatibility — only its first element is used, as an
            approximate gamma — but new callers should pass a bare float.

    Returns:
        DataFrame with conviction-based portfolio holdings. `.attrs` carries
        `min_pos_pct_eff` / `max_pos_pct_eff`, the bounds actually applied
        after feasibility relaxation (may differ from the requested
        min_pos_pct / max_pos_pct when num_positions makes the nominal
        bounds infeasible).
    """
    if not aggregated_holdings or current_df.empty:
        return pd.DataFrame()

    # Step 1: Build temporary portfolio DataFrame
    temp_portfolio = pd.DataFrame([
        {'symbol': symbol, 'price': data['price'], 'weightage_pct': 1.0}
        for symbol, data in aggregated_holdings.items()
    ])

    if temp_portfolio.empty:
        return pd.DataFrame()

    # Step 2: Use regime.py compute_conviction_signals to get conviction scores
    # This is the SAME function used for UI display (Flow 2)
    portfolio_with_conviction = compute_conviction_signals(
        temp_portfolio, current_df,
        universe=universe, selected_index=selected_index,
        regime_name=regime_name, mode=mode,
    )

    # Step 2.5: Exclude symbols with NO usable signal at all. A symbol absent
    # from current_df (or with every one of the six signals unavailable)
    # receives a display-only conviction_score of 50 (the neutral fallback in
    # compute_conviction_signals), which can outrank a genuinely-scored
    # candidate with a real but low conviction (e.g. 40) into the top-N purely
    # because it was never actually evaluated — the model has no opinion on
    # it, not a neutral one. Only exclude from SELECTION; the display-only 50
    # fallback is left intact for any other consumer (see AUDIT_DIRECTIVES.md
    # A19).
    if "signals_available" in portfolio_with_conviction.columns:
        selectable = portfolio_with_conviction[portfolio_with_conviction["signals_available"] > 0].copy()
    else:
        selectable = portfolio_with_conviction

    # Step 3: Select top num_positions by conviction score (NO threshold filter)
    # All symbols are eligible regardless of conviction score.
    #
    # STRUCTURAL TIE-BREAK (volume-profile selection refinement): conviction is
    # coarse (it sums integer-step signals), so many candidates tie at the
    # selection boundary. When that happens, prefer names trading at a DISCOUNT
    # to accepted value (va_pos < 0 = price sits below the value area's own
    # MIDPOINT, i.e. the cheaper half of the value area = more mean-reversion
    # room) over names sitting in the richer half above the midpoint. This
    # sharpens *which* of the tied names make the cut, using measured
    # structure — it never overrides a genuinely higher conviction score.
    ranked = selectable.copy()
    if "va_pos" in ranked.columns:
        # va_pos in [-1, +1] = (price - value_area_midpoint) / value_area_half_width;
        # lower (discount) is preferred, so sort ascending as the secondary
        # key. NaN va_pos sinks to the bottom of a tie (no edge info).
        ranked["_vp_key"] = ranked["va_pos"].fillna(1.0)
        ranked = ranked.sort_values(
            ["conviction_score", "_vp_key"], ascending=[False, True]
        )
        conv_df = ranked.head(num_positions).drop(columns=["_vp_key"]).copy()
    else:
        conv_df = ranked.nlargest(num_positions, 'conviction_score').copy()

    if conv_df.empty:
        return pd.DataFrame()

    # Step 4: Apply conviction dispersion weighting (concentrate high conviction)
    #
    # CONTINUOUS power-law transform: adjusted = conviction_score ** gamma.
    # The previous median-step formula (boost above median / penalty below)
    # created a discontinuous cliff at the median: two candidates one
    # conviction point apart (e.g. 65 vs 64), straddling the median, received
    # up to a ~13x weight ratio purely from which side of the median line they
    # fell on — an artifact of the step function, not a real difference in
    # conviction. A rank-based continuous alternative was tried first but
    # rejected: it differentiates TIED scores purely by sort-order position,
    # which is worse (see AUDIT_DIRECTIVES.md A5 implementation notes). Raising
    # the raw score to a power is monotone and continuous in the score itself —
    # ties get identical weight, adjacent scores get a smoothly adjacent
    # weight, and gamma controls concentration exactly like the old
    # boost/penalty pair did, without the boundary discontinuity.
    if apply_dispersion and len(conv_df) > 1:
        # Style-aware concentration. Intelligence mode does not override this:
        # Pragyam's calibration learns the six conviction signal weights, not
        # the concentration policy, which is a user-facing risk preference.
        if dispersion_params is None:
            gamma = 4.5 if investment_style == "Swing Trading" else 2.5
        else:
            # Back-compat: a (boost_multiplier, penalty_multiplier) tuple from
            # the old API is interpreted as a gamma override via its first
            # element scaled to a comparable concentration range, so existing
            # callers passing dispersion_params don't hard-crash — but this
            # path is deprecated; pass a bare float gamma instead.
            gamma = float(dispersion_params[0]) if isinstance(dispersion_params, tuple) else float(dispersion_params)

        conv_df['adjusted_conviction'] = np.power(
            conv_df['conviction_score'].astype(float).clip(lower=1e-6), gamma
        )
        total_conviction = conv_df['adjusted_conviction'].sum()

    else:
        # Fallback to original linear weighting
        conv_df['adjusted_conviction'] = conv_df['conviction_score']
        total_conviction = conv_df['conviction_score'].sum()

    if total_conviction <= 0:
        return pd.DataFrame()

    # THE FORMULA: adjusted_conviction / total_adjusted_conviction * 100
    conv_df['weightage_pct'] = (conv_df['adjusted_conviction'].astype(float) / total_conviction) * 100.0

    # Step 5: Apply bounds (min/max position limits)
    #
    # FEASIBILITY: sum(weights) == 1 with every weight <= max_pos_pct requires
    # n * max_pos_pct >= 1 (need at least 1/max_pos_pct positions to fill 100%
    # within the cap); symmetrically, n * min_pos_pct <= 1 or the floor alone
    # overflows 100%. Without this check, e.g. 5 positions with a 10% cap is
    # mathematically infeasible — the clip-renormalize loop below silently
    # converges to 20% each (double the advertised cap) because it always ends
    # on a renormalize. Relax whichever bound is infeasible to the minimal
    # value that restores feasibility (n_positions is small and known here, so
    # this is an exact fix, not an approximation).
    n = len(conv_df)
    max_pos_pct_eff = max(max_pos_pct, 1.0 / n) if n > 0 else max_pos_pct
    min_pos_pct_eff = min(min_pos_pct, 1.0 / n) if n > 0 else min_pos_pct

    weights = conv_df['weightage_pct'].astype(float).to_numpy() / 100.0
    for _ in range(10):
        weights = np.clip(weights, min_pos_pct_eff, max_pos_pct_eff)
        weights = weights / weights.sum()
    conv_df['weightage_pct'] = weights * 100.0

    # Surface the effective bounds (may differ from the requested min/max_pos_pct
    # when relaxed for feasibility) so callers/UI can display what was actually
    # applied rather than the nominal 1%-10% regardless of position count.
    conv_df.attrs['min_pos_pct_eff'] = min_pos_pct_eff
    conv_df.attrs['max_pos_pct_eff'] = max_pos_pct_eff

    # Step 6: Calculate units and value
    conv_df['units'] = np.floor((sip_amount * conv_df['weightage_pct'].astype(float) / 100.0) / conv_df['price'].astype(float))
    conv_df['value'] = conv_df['units'] * conv_df['price'].astype(float)

    # Sort by conviction score (descending)
    return conv_df.sort_values('conviction_score', ascending=False).reset_index(drop=True)


__all__ = ["compute_conviction_based_weights"]
