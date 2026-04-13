"""
PRAGYAM — Chart Components
══════════════════════════════════════════════════════════════════════════════

Obsidian Quant Terminal Design System — Institutional-grade financial visualization.

All charts use chart_layout() and style_axes() from ui/theme.py for consistent theming.
Aesthetics match Nishkarsh v1.2.0 chart patterns (line widths, fills, markers, trace colors).

Version: 1.4.0
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from ui.theme import chart_layout, style_axes


# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE — Terminal Glass
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    # Primary: Amber Gold (system anchor)
    "amber": "#D4A853",
    "amber_dim": "rgba(212, 168, 83, 0.6)",
    "amber_glow": "rgba(212, 168, 83, 0.25)",

    # Heatmap / Signal: Diverging scale (Rose → Slate → Emerald)
    # Bearish: Rose (sharp,警示)
    "rose": "#E8555A",
    "rose_dim": "rgba(232, 85, 90, 0.5)",
    "rose_glow": "rgba(232, 85, 90, 0.2)",
    # Neutral: Warm slate (not cold gray — maintains warmth)
    "slate": "#8B7E6A",
    "slate_dim": "rgba(139, 126, 106, 0.4)",
    # Bullish: Emerald (rich, deep)
    "emerald": "#2DD4A8",
    "emerald_dim": "rgba(45, 212, 168, 0.5)",
    "emerald_glow": "rgba(45, 212, 168, 0.2)",

    # Accent palette (used sparingly for UI elements)
    "cyan": "#06B6D4",
    "cyan_glow": "rgba(6, 182, 212, 0.2)",
    "violet": "#8B5CF6",
    "violet_glow": "rgba(139, 92, 246, 0.2)",
    "orange": "#F59E0B",
    "orange_glow": "rgba(245, 158, 11, 0.2)",
}


# ══════════════════════════════════════════════════════════════════════════════
# REGIME INTELLIGENCE CHARTS
# ══════════════════════════════════════════════════════════════════════════════


def create_regime_history_chart(regime_series: list) -> go.Figure:
    """Timeline chart of market regime transitions over a rolling window.

    Matches Nishkarsh aesthetic: subtle reference lines, dual-fill pattern,
    dynamic marker sizing, no marker borders, 1.5px line width.

    Trace colors match Nishkarsh exactly:
    - Main line: amber (#D4A853) at 1.5px
    - Positive fills: emerald rgba(52,211,153,0.06/0.08)
    - Negative fills: rose rgba(251,113,133,0.06/0.08)
    - Reference lines: 0.5px at 15% opacity

    Args:
        regime_series: List of RegimeResult objects from get_regime_history_series().

    Returns:
        Plotly Figure with dual-layer regime timeline.
    """
    if not regime_series:
        fig = go.Figure()
        fig.update_layout(**chart_layout(height=300))
        return fig

    dates = [r.date for r in regime_series]
    scores = [r.composite_score for r in regime_series]
    colors = [r.color for r in regime_series]
    regimes = [r.regime.replace("_", " ") for r in regime_series]
    confs = [r.confidence for r in regime_series]

    # Dynamic marker sizing based on confidence (matches Nishkarsh pattern)
    marker_sizes = [7 if c >= 0.7 else 5 if c >= 0.5 else 4 for c in confs]

    fig = go.Figure()

    # Upper band (invisible line for fill pattern - Nishkarsh style)
    upper = [s + c * 0.4 for s, c in zip(scores, confs)]
    lower = [s - c * 0.4 for s, c in zip(scores, confs)]

    # Positive confidence fill (above zero)
    upper_positive = [max(0, u) for u in upper]
    lower_positive = [max(0, l) for l in lower]
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=upper_positive + lower_positive[::-1],
            fill="toself",
            fillcolor="rgba(45, 212, 168, 0.07)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip",
            showlegend=False,
            name="",
        )
    )

    # Negative confidence fill (below zero)
    upper_negative = [min(0, u) for u in upper]
    lower_negative = [min(0, l) for l in lower]
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=upper_negative + lower_negative[::-1],
            fill="toself",
            fillcolor="rgba(232, 85, 90, 0.07)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip",
            showlegend=False,
            name="",
        )
    )

    # Composite score line — warm slate
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=scores,
            mode="lines+markers",
            name="Composite Score",
            line=dict(color=COLORS["slate"], width=2.5, shape='spline'),
            marker=dict(
                size=marker_sizes,
                color=colors,
                symbol='circle',
                line=dict(width=1, color='rgba(255,255,255,0.15)'),
            ),
            customdata=list(zip(regimes, [f"{c:.0%}" for c in confs])),
            hovertemplate="<b>%{customdata[0]}</b><br>Score: %{y:+.2f}<br>Confidence: %{customdata[1]}<br><span style='opacity:0.7;'>%{x|%Y-%m-%d}</span><extra></extra>",
            fill='tozeroy',
            fillcolor='rgba(139, 126, 106, 0.06)',
        )
    )

    # Reference lines — Terminal Glass aesthetic
    for y_val, color, label in [
        (1.0, "rgba(45, 212, 168, 0.25)", "Bull"),
        (0.1, "rgba(212, 168, 83, 0.25)", "Chop"),
        (-0.5, "rgba(232, 85, 90, 0.25)", "Bear"),
    ]:
        fig.add_hline(
            y=y_val,
            line_dash="dot",
            line_color=color,
            line_width=0.8,
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(color=color, size=10, family="IBM Plex Mono, monospace"),
            annotation_font_size=9,
            opacity=0.9,
        )

    # Apply Obsidian Quant theme
    fig.update_layout(**chart_layout(height=320, show_legend=False))
    style_axes(fig, y_title="Composite Score", y_range=[-2.5, 2.5])
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CONVICTION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════


def create_conviction_heatmap(portfolio_with_signals: pd.DataFrame) -> go.Figure:
    """Signal-strength heatmap for portfolio holdings.

    Each row is a position; columns are RSI, Oscillator, Z-Score, MA Alignment,
    and the composite Conviction score. Colours run red → amber → green.

    Trace colors match Nishkarsh exactly:
    - Rose (bear): #FB7185
    - Emerald (bull): #34D399
    - Slate (neutral): rgba(148,163,184,0.4)

    Args:
        portfolio_with_signals: DataFrame from compute_conviction_signals().

    Returns:
        Plotly Figure (heatmap).
    """
    required = ["symbol", "rsi_signal", "osc_signal", "zscore_signal", "ma_signal", "conviction_score"]
    if portfolio_with_signals.empty or not all(c in portfolio_with_signals.columns for c in required):
        fig = go.Figure()
        fig.update_layout(**chart_layout(height=200))
        return fig

    df = portfolio_with_signals.sort_values("conviction_score", ascending=False).head(40)

    signal_cols = ["rsi_signal", "osc_signal", "zscore_signal", "ma_signal"]
    col_labels = ["RSI", "Oscillator", "Z-Score", "MA Align"]

    # Conviction score column: scale [0,100] → [-2,2] for unified colorscale
    conviction_normalised = (df["conviction_score"] / 100.0) * 4.0 - 2.0

    z_matrix = np.column_stack(
        [df[col].fillna(0).values for col in signal_cols] + [conviction_normalised.values]
    )

    text_matrix = np.column_stack(
        [df[col].fillna(0).apply(lambda x: f"{x:+.0f}").values for col in signal_cols]
        + [df["conviction_score"].apply(lambda x: f"{int(x)}").values]
    )

    # Terminal Glass diverging colorscale: Rose → Warm Slate → Emerald
    # More saturated at extremes, muted in center for clarity
    fig = go.Figure(
        go.Heatmap(
            z=z_matrix.T,
            x=df["symbol"].values,
            y=col_labels + ["Conviction"],
            colorscale=[
                [0.0, "#E8555A"],       # Deep rose (strong bear)
                [0.15, "#E07060"],      # Rose-orange transition
                [0.35, "#B8956A"],      # Warm amber-slate
                [0.5, "#8B7E6A"],       # Warm slate (neutral)
                [0.65, "#6A9E78"],      # Sage green
                [0.85, "#3DC49A"],      # Bright emerald
                [1.0, "#2DD4A8"],       # Deep emerald (strong bull)
            ],
            zmid=0,
            zmin=-2,
            zmax=2,
            text=text_matrix.T,
            texttemplate="%{text}",
            textfont=dict(size=10, family="IBM Plex Mono, monospace", color="rgba(255,255,255,0.9)"),
            showscale=True,
            colorbar=dict(
                title="Signal",
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=["Strong Bear", "Bear", "Neutral", "Bull", "Strong Bull"],
                tickfont=dict(family="IBM Plex Mono, monospace", color="#94A3B8", size=10),
                bgcolor="rgba(10, 14, 23, 0.95)",
                bordercolor="rgba(212, 168, 83, 0.25)",
                borderwidth=1,
                thickness=16,
                len=0.75,
                y=0.5,
                yanchor="middle",
                x=1.02,
                xanchor="left",
                outlinewidth=0,
                tickangle=0,
            ),
            hovertemplate="<b>%{x}</b><br>%{y}: %{text}<br><span style='opacity:0.7;'>Signal Strength</span><extra></extra>",
            xgap=2,
            ygap=2,
        )
    )

    n_positions = len(df)
    fig_height = max(220, min(500, 60 + n_positions * 22))

    # Apply Obsidian Quant theme
    fig.update_layout(**chart_layout(height=fig_height, show_legend=False))
    style_axes(fig)

    # Additional heatmap-specific styling
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10, family="IBM Plex Mono, monospace", color="#64748B"), gridwidth=1, gridcolor="rgba(255,255,255,0.03)")
    fig.update_yaxes(tickfont=dict(size=11, family="IBM Plex Mono, monospace", color="#64748B"), gridwidth=1, gridcolor="rgba(255,255,255,0.03)")
    
    return fig


__all__ = [
    "COLORS",
    "create_regime_history_chart",
    "create_conviction_heatmap",
]
