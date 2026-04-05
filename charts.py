"""
PRAGYAM — Chart Components
══════════════════════════════════════════════════════════════════════════════

@thebullishvalue Design System — Institutional-grade financial visualization.

Version: 1.1.0
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any


# ══════════════════════════════════════════════════════════════════════════════
# @thebullishvalue DESIGN SYSTEM — COLOR PALETTE
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#FFC300",
    "primary_rgb": "255, 195, 0",
    "background": "#0F0F0F",
    "card": "#1A1A1A",
    "elevated": "#2A2A2A",
    "border": "#2A2A2A",
    "border_light": "#3A3A3A",
    "text": "#EAEAEA",
    "text_secondary": "#CCCCCC",
    "muted": "#888888",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#06b6d4",
    "neutral": "#888888",
    "bull": "#10b981",
    "bear": "#ef4444",
    "palette": ["#FFC300", "#10b981", "#06b6d4", "#f59e0b", "#a855f7", "#ec4899", "#84cc16", "#f97316"],
}


def get_chart_layout(
    title: str = "",
    height: int = 450,
    show_legend: bool = True,
    legend_position: str = "top",
) -> dict:
    """Standardized institutional-grade chart layout configuration."""
    legend_config = {
        "top": dict(orientation="h", y=1.02, x=0.5, xanchor="center", yanchor="bottom"),
        "bottom": dict(orientation="h", y=-0.15, x=0.5, xanchor="center", yanchor="top"),
        "right": dict(orientation="v", y=0.5, x=1.02, xanchor="left", yanchor="middle"),
    }

    config = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": COLORS["card"],
        "height": height,
        "margin": dict(l=60, r=30, t=70 if title else 40, b=60),
        "font": dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color=COLORS["text"], size=13),
        "showlegend": show_legend,
        "legend": legend_config.get(legend_position, legend_config["top"]),
        "hovermode": "x unified",
        "hoverlabel": dict(
            bgcolor=COLORS["elevated"],
            bordercolor=COLORS["border_light"],
            font_size=13,
            font_family="Inter, sans-serif",
        ),
    }

    if title:
        config["title"] = dict(
            text=title,
            font=dict(family="Inter, sans-serif", size=15, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        )
    else:
        config["title"] = dict(text="", font=dict(size=1))

    return config


# ══════════════════════════════════════════════════════════════════════════════
# REGIME INTELLIGENCE CHARTS
# ══════════════════════════════════════════════════════════════════════════════


def create_regime_history_chart(regime_series: list) -> go.Figure:
    """Timeline chart of market regime transitions over a rolling window.

    Args:
        regime_series: List of RegimeResult objects from get_regime_history_series().

    Returns:
        Plotly Figure with dual-layer regime timeline.
    """
    if not regime_series:
        fig = go.Figure()
        fig.update_layout(**get_chart_layout("No regime data available", height=300))
        return fig

    dates = [r.date for r in regime_series]
    scores = [r.composite_score for r in regime_series]
    colors = [r.color for r in regime_series]
    regimes = [r.regime.replace("_", " ") for r in regime_series]
    confs = [r.confidence for r in regime_series]

    fig = go.Figure()

    # Shaded confidence band
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=[s + c * 0.4 for s, c in zip(scores, confs)]
            + [s - c * 0.4 for s, c in zip(scores[::-1], confs[::-1])],
            fill="toself",
            fillcolor="rgba(255,195,0,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Confidence Band",
        )
    )

    # Composite score line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=scores,
            mode="lines+markers",
            name="Composite Score",
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(
                size=9,
                color=colors,
                line=dict(color="rgba(255,255,255,0.5)", width=1),
            ),
            customdata=list(zip(regimes, [f"{c:.0%}" for c in confs])),
            hovertemplate="<b>%{customdata[0]}</b><br>Score: %{y:+.2f}<br>Confidence: %{customdata[1]}<extra></extra>",
        )
    )

    # Reference lines
    for y_val, color, label in [
        (1.0, COLORS["success"], "Bull"),
        (0.1, COLORS["warning"], "Chop"),
        (-0.5, COLORS["danger"], "Bear"),
    ]:
        fig.add_hline(
            y=y_val,
            line_dash="dot",
            line_color=color,
            line_width=1,
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(color=color, size=10),
        )

    layout = get_chart_layout("Regime Score History", height=320, show_legend=False)
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["border"])
    fig.update_yaxes(
        title="Composite Score",
        showgrid=True,
        gridcolor=COLORS["border"],
        zeroline=True,
        zerolinecolor=COLORS["muted"],
        zerolinewidth=1,
        range=[-2.5, 2.5],
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CONVICTION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════


def create_conviction_heatmap(portfolio_with_signals: pd.DataFrame) -> go.Figure:
    """Signal-strength heatmap for portfolio holdings.

    Each row is a position; columns are RSI, Oscillator, Z-Score, MA Alignment,
    and the composite Conviction score. Colours run red → amber → green.

    Args:
        portfolio_with_signals: DataFrame from compute_conviction_signals().

    Returns:
        Plotly Figure (heatmap).
    """
    required = ["symbol", "rsi_signal", "osc_signal", "zscore_signal", "ma_signal", "conviction_score"]
    if portfolio_with_signals.empty or not all(c in portfolio_with_signals.columns for c in required):
        fig = go.Figure()
        fig.update_layout(**get_chart_layout("No signal data available", height=200))
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

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix.T,
            x=df["symbol"].values,
            y=col_labels + ["Conviction"],
            colorscale=[
                [0.0, COLORS["danger"]],
                [0.25, "#f97316"],
                [0.5, COLORS["muted"]],
                [0.75, "#a3e635"],
                [1.0, COLORS["success"]],
            ],
            zmid=0,
            zmin=-2,
            zmax=2,
            text=text_matrix.T,
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            showscale=True,
            colorbar=dict(
                title="Signal",
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=["Strong Bear", "Bear", "Neutral", "Bull", "Strong Bull"],
                tickfont=dict(color=COLORS["text_secondary"], size=10),
            ),
            hovertemplate="<b>%{x}</b><br>%{y}: %{text}<extra></extra>",
        )
    )

    n_positions = len(df)
    fig_height = max(220, min(500, 60 + n_positions * 22))
    layout = get_chart_layout("", height=fig_height, show_legend=False)
    fig.update_layout(**layout)
    fig.update_layout(
        margin=dict(l=90, r=60, t=20, b=60),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


__all__ = [
    "COLORS",
    "get_chart_layout",
    "create_regime_history_chart",
    "create_conviction_heatmap",
]
