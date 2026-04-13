"""
PRAGYAM — Portfolio Construction Engine
══════════════════════════════════════════════════════════════════════════════

Conviction-based portfolio curation using regime.py signal scoring.

Formula: weight_i = (conviction_score_i / Σ all_conviction_scores) × 100

Conviction Dispersion Weighting (v7.0.5):
  → SIP Mode: +75% boost above median, -50% penalty below
  → Swing Mode: +225% boost above median, -75% penalty below (2σ more aggressive)
  → Dynamic dispersion based on investment style and conviction volatility

Author: @thebullishvalue
Version: 7.0.5
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_conviction_based_weights(
    aggregated_holdings: Dict[str, Dict],
    current_df: pd.DataFrame,
    sip_amount: float,
    num_positions: int,
    min_pos_pct: float,
    max_pos_pct: float,
    apply_dispersion: bool = True,
    investment_style: str = "SIP Investment",
    dispersion_params: Optional[tuple] = None  # (boost_multiplier, penalty_multiplier)
) -> pd.DataFrame:
    """
    Compute portfolio weights based on conviction scores.

    Uses regime.py::compute_conviction_signals() to score all candidates,
    then applies the conviction-based weighting formula.

    Process:
    1. Build temporary portfolio DataFrame from aggregated_holdings
    2. Compute conviction scores using regime.py (RSI, OSC, Z-Score, MA)
    3. Select top num_positions by conviction score (NO threshold filter)
    4. Apply conviction dispersion weighting (boost above median, penalize below)
    5. Apply formula: weight_i = (adjusted_conviction_i / Σ all_adjusted) × 100
    6. Apply bounds (min/max position limits)
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
            - SIP: Conservative dispersion (+75%/-50%)
            - Swing: Aggressive dispersion (+225%/-75%) = 2σ more concentration
        dispersion_params: Optional tuple of (boost_multiplier, penalty_multiplier)
            - If None, auto-selects based on investment_style
            - SIP: (2.25, 0.50) = +125% boost, -50% penalty
            - Swing: (3.25, 0.25) = +225% boost, -75% penalty

    Returns:
        DataFrame with conviction-based portfolio holdings
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
    from regime import compute_conviction_signals
    portfolio_with_conviction = compute_conviction_signals(temp_portfolio, current_df)

    # Step 3: Select top num_positions by conviction score (NO threshold filter)
    # All symbols are eligible regardless of conviction score
    conv_df = portfolio_with_conviction.nlargest(num_positions, 'conviction_score').copy()

    if conv_df.empty:
        return pd.DataFrame()

    # Step 4: Apply conviction dispersion weighting (boost high conviction, penalize low)
    if apply_dispersion and len(conv_df) > 1:
        # Auto-select dispersion params based on investment style if not provided
        if dispersion_params is None:
            if investment_style == "Swing Trading":
                # Swing Trading: Aggressive dispersion (2σ more concentration)
                # +225% boost for above median, -75% penalty for below
                boost_mult, penalty_mult = 3.25, 0.25
            else:
                # SIP Investment: Moderate dispersion
                # +125% boost for above median, -50% penalty for below
                boost_mult, penalty_mult = 2.25, 0.50
        else:
            boost_mult, penalty_mult = dispersion_params
        
        median_conviction = conv_df['conviction_score'].median()

        # Apply boost/penalty based on median
        conv_df['adjusted_conviction'] = conv_df['conviction_score'].apply(
            lambda score: score * boost_mult if score > median_conviction else score * penalty_mult
        )

        # Use adjusted conviction for weighting
        total_conviction = conv_df['adjusted_conviction'].sum()

        # Log dispersion effect (for debugging)
        above_median = (conv_df['conviction_score'] > median_conviction).sum()
        below_median = (conv_df['conviction_score'] <= median_conviction).sum()

    else:
        # Fallback to original linear weighting
        conv_df['adjusted_conviction'] = conv_df['conviction_score']
        total_conviction = conv_df['conviction_score'].sum()

    if total_conviction <= 0:
        return pd.DataFrame()

    # THE FORMULA: adjusted_conviction / total_adjusted_conviction * 100
    conv_df['weightage_pct'] = (conv_df['adjusted_conviction'] / total_conviction) * 100

    # Step 5: Apply bounds (min/max position limits)
    weights = conv_df['weightage_pct'].values / 100
    for _ in range(10):
        weights = np.clip(weights, min_pos_pct, max_pos_pct)
        weights = weights / weights.sum()
    conv_df['weightage_pct'] = weights * 100

    # Step 6: Calculate units and value
    conv_df['units'] = np.floor((sip_amount * conv_df['weightage_pct'] / 100) / conv_df['price'])
    conv_df['value'] = conv_df['units'] * conv_df['price']

    # Sort by conviction score (descending)
    return conv_df.sort_values('conviction_score', ascending=False).reset_index(drop=True)


__all__ = ["compute_conviction_based_weights"]
