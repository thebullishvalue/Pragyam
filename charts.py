"""
PRAGYAM - Unified Chart Components
══════════════════════════════════════════════════════════════════════════════

Hedge Fund Grade Visualization System following Hemrek Capital Design Standards.
Consistent with Nirnay and other Pragyam Product Family members.

Features:
- Dark theme optimized for financial data
- Consistent color scheme across all visualizations
- Institutional-grade chart layouts
- Performance-optimized rendering

Version: 1.0.0
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# HEMREK CAPITAL DESIGN SYSTEM - COLOR PALETTE
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    # Brand Colors
    'primary': '#FFC300',
    'primary_rgb': '255, 195, 0',
    
    # Background Hierarchy
    'background': '#0F0F0F',
    'card': '#1A1A1A',
    'elevated': '#2A2A2A',
    
    # Border System
    'border': '#2A2A2A',
    'border_light': '#3A3A3A',
    
    # Text Hierarchy
    'text': '#EAEAEA',
    'text_secondary': '#CCCCCC',
    'muted': '#888888',
    
    # Semantic Colors
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'info': '#06b6d4',
    'neutral': '#888888',
    
    # Trading Signals
    'bull': '#10b981',
    'bear': '#ef4444',
    
    # Chart Palette (for multi-series)
    'palette': ['#FFC300', '#10b981', '#06b6d4', '#f59e0b', '#a855f7', '#ec4899', '#84cc16', '#f97316']
}


def get_chart_layout(
    title: str = "",
    height: int = 400,
    show_legend: bool = True,
    legend_position: str = 'top'
) -> dict:
    """
    Get standardized chart layout configuration.
    
    Args:
        title: Chart title
        height: Chart height in pixels
        show_legend: Whether to show legend
        legend_position: 'top', 'bottom', 'right'
    
    Returns:
        Dictionary of layout configuration
    """
    legend_config = {
        'top': dict(orientation='h', y=1.02, x=0.5, xanchor='center', yanchor='bottom'),
        'bottom': dict(orientation='h', y=-0.15, x=0.5, xanchor='center', yanchor='top'),
        'right': dict(orientation='v', y=0.5, x=1.02, xanchor='left', yanchor='middle')
    }
    
    return {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': COLORS['card'],
        'height': height,
        'margin': dict(l=60, r=20, t=60 if title else 30, b=50),
        'font': dict(family='Inter, -apple-system, sans-serif', color=COLORS['text'], size=12),
        'title': dict(
            text=title,
            font=dict(size=16, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ) if title else None,
        'showlegend': show_legend,
        'legend': legend_config.get(legend_position, legend_config['top']),
        'hovermode': 'x unified',
        'hoverlabel': dict(
            bgcolor=COLORS['elevated'],
            font_size=12,
            font_family='Inter'
        )
    }


def get_axis_config(
    title: str = "",
    tickformat: str = None,
    showgrid: bool = True,
    zeroline: bool = False
) -> dict:
    """Get standardized axis configuration."""
    config = {
        'showgrid': showgrid,
        'gridcolor': COLORS['border'],
        'gridwidth': 1,
        'zeroline': zeroline,
        'zerolinecolor': COLORS['muted'],
        'zerolinewidth': 1,
        'linecolor': COLORS['border'],
        'tickfont': dict(color=COLORS['muted'], size=11),
        'titlefont': dict(color=COLORS['text'], size=12)
    }
    
    if title:
        config['title'] = title
    if tickformat:
        config['tickformat'] = tickformat
    
    return config


# ══════════════════════════════════════════════════════════════════════════════
# EQUITY & PERFORMANCE CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def create_equity_drawdown_chart(
    returns_df: pd.DataFrame,
    date_col: str = 'date',
    return_col: str = 'return'
) -> go.Figure:
    """
    Create institutional-grade equity curve with underwater analysis.
    
    Args:
        returns_df: DataFrame with date and return columns
        date_col: Name of date column
        return_col: Name of return column
    
    Returns:
        Plotly Figure object
    """
    df = returns_df.copy().sort_values(date_col)
    df['equity'] = (1 + df[return_col]).cumprod()
    df['peak'] = df['equity'].expanding().max()
    df['drawdown'] = (df['equity'] / df['peak']) - 1
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=None
    )
    
    # Equity Curve
    fig.add_trace(go.Scatter(
        x=df[date_col], 
        y=df['equity'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color=COLORS['primary'], width=2.5),
        fill='tozeroy',
        fillcolor=f'rgba({COLORS["primary_rgb"]}, 0.08)'
    ), row=1, col=1)
    
    # High Water Mark
    fig.add_trace(go.Scatter(
        x=df[date_col], 
        y=df['peak'],
        mode='lines',
        name='High Water Mark',
        line=dict(color=COLORS['muted'], width=1, dash='dot')
    ), row=1, col=1)
    
    # Drawdown Fill
    fig.add_trace(go.Scatter(
        x=df[date_col], 
        y=df['drawdown'],
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color=COLORS['danger'], width=1),
        fillcolor='rgba(239, 68, 68, 0.3)'
    ), row=2, col=1)
    
    # Zero line for drawdown
    fig.add_hline(y=0, line_color=COLORS['muted'], line_width=1, row=2, col=1)
    
    # Layout
    layout = get_chart_layout(height=500, legend_position='top')
    layout['annotations'] = [
        dict(
            text="<b>Growth of ₹1 Investment</b>",
            xref="paper", yref="paper",
            x=0, y=1.08,
            showarrow=False,
            font=dict(size=13, color=COLORS['text'])
        ),
        dict(
            text="<b>Underwater Curve</b>",
            xref="paper", yref="paper",
            x=0, y=0.28,
            showarrow=False,
            font=dict(size=13, color=COLORS['text'])
        )
    ]
    fig.update_layout(**layout)
    
    # Axis formatting
    fig.update_yaxes(title_text="Portfolio Value (₹)", row=1, col=1, **get_axis_config())
    fig.update_yaxes(title_text="Drawdown", tickformat='.1%', row=2, col=1, **get_axis_config())
    fig.update_xaxes(**get_axis_config(), row=1, col=1)
    fig.update_xaxes(title_text="Date", **get_axis_config(), row=2, col=1)
    
    return fig


def create_rolling_metrics_chart(
    returns_df: pd.DataFrame,
    window: int = 12,
    date_col: str = 'date',
    return_col: str = 'return',
    periods_per_year: int = 52
) -> go.Figure:
    """
    Create rolling Sharpe and Sortino ratio chart.
    
    Args:
        returns_df: DataFrame with date and return columns
        window: Rolling window size
        date_col: Name of date column
        return_col: Name of return column
        periods_per_year: Annualization factor
    
    Returns:
        Plotly Figure object
    """
    df = returns_df.copy().sort_values(date_col)
    
    # Calculate rolling metrics
    rolling_mean = df[return_col].rolling(window=window).mean()
    rolling_std = df[return_col].rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(periods_per_year)
    
    # Downside deviation for Sortino
    downside_returns = df[return_col].apply(lambda x: x if x < 0 else 0)
    rolling_downside = downside_returns.rolling(window=window).std()
    rolling_sortino = (rolling_mean / rolling_downside.replace(0, np.nan)) * np.sqrt(periods_per_year)
    
    fig = go.Figure()
    
    # Rolling Sharpe
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=rolling_sharpe,
        mode='lines',
        name=f'Rolling Sharpe ({window}-period)',
        line=dict(color=COLORS['primary'], width=2)
    ))
    
    # Rolling Sortino
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=rolling_sortino,
        mode='lines',
        name=f'Rolling Sortino ({window}-period)',
        line=dict(color=COLORS['success'], width=2)
    ))
    
    # Reference lines
    fig.add_hline(y=0, line_dash='dash', line_color=COLORS['muted'], line_width=1)
    fig.add_hline(y=1, line_dash='dot', line_color=COLORS['success'], line_width=1,
                  annotation_text="Sharpe = 1", annotation_position="right",
                  annotation_font_color=COLORS['muted'])
    fig.add_hline(y=2, line_dash='dot', line_color=COLORS['warning'], line_width=1,
                  annotation_text="Sharpe = 2", annotation_position="right",
                  annotation_font_color=COLORS['muted'])
    
    layout = get_chart_layout(
        title="<b>Rolling Risk-Adjusted Performance</b>",
        height=380
    )
    fig.update_layout(**layout)
    fig.update_xaxes(**get_axis_config(title="Date"))
    fig.update_yaxes(**get_axis_config(title="Ratio (Annualized)", zeroline=True))
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION & HEATMAP CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Strategy Correlation Matrix"
) -> go.Figure:
    """
    Create institutional-grade correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    # Custom colorscale: Blue (negative) -> Gray (zero) -> Red (positive)
    colorscale = [
        [0.0, '#3b82f6'],    # Blue (strong negative)
        [0.25, '#60a5fa'],   # Light blue
        [0.5, COLORS['muted']],  # Gray (zero)
        [0.75, '#f87171'],   # Light red
        [1.0, '#ef4444']     # Red (strong positive)
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=colorscale,
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=11, color='white'),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(
            title=dict(text='Correlation', font=dict(color=COLORS['muted'])),
            tickfont=dict(color=COLORS['muted']),
            thickness=15,
            len=0.8
        )
    ))
    
    layout = get_chart_layout(
        title=f"<b>{title}</b>",
        height=max(350, len(corr_matrix) * 35),
        show_legend=False
    )
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    
    return fig


def create_tier_sharpe_heatmap(
    subset_perf: Dict,
    strategy_names: List[str]
) -> go.Figure:
    """
    Create Sharpe ratio by tier heatmap.
    
    Args:
        subset_perf: Dictionary of strategy -> tier performance
        strategy_names: List of strategy names
    
    Returns:
        Plotly Figure object
    """
    # Build heatmap data
    max_tier = 0
    for strat in strategy_names:
        if strat in subset_perf and subset_perf[strat]:
            tier_nums = [int(tier.split('_')[1]) for tier in subset_perf[strat].keys()]
            if tier_nums:
                max_tier = max(max_tier, max(tier_nums))
    
    if max_tier == 0:
        return None
    
    heatmap_data = {}
    for strat in strategy_names:
        row = []
        for i in range(max_tier):
            val = subset_perf.get(strat, {}).get(f'tier_{i+1}', np.nan)
            row.append(val)
        heatmap_data[strat] = row
    
    df = pd.DataFrame(heatmap_data).T
    df.columns = [f'Tier {i+1}' for i in range(df.shape[1])]
    
    # Sort by average Sharpe
    df['avg'] = df.mean(axis=1)
    df = df.sort_values('avg', ascending=False)
    df = df.drop('avg', axis=1)
    
    # Create heatmap
    colorscale = [
        [0.0, '#ef4444'],    # Red (negative)
        [0.4, '#fbbf24'],    # Yellow
        [0.5, COLORS['muted']],  # Gray (zero)
        [0.6, '#86efac'],    # Light green
        [1.0, '#10b981']     # Green (positive)
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale=colorscale,
        zmid=0,
        text=np.round(df.values, 2),
        texttemplate='%{text:.2f}',
        textfont=dict(size=10, color='white'),
        hovertemplate='%{y}<br>%{x}: Sharpe = %{z:.3f}<extra></extra>',
        colorbar=dict(
            title=dict(text='Sharpe', font=dict(color=COLORS['muted'])),
            tickfont=dict(color=COLORS['muted']),
            thickness=15
        )
    ))
    
    layout = get_chart_layout(
        title="<b>Strategy Performance by Position Tier</b><br><sup>Sharpe ratio for each 10-stock tier</sup>",
        height=max(300, len(df) * 30),
        show_legend=False
    )
    fig.update_layout(**layout)
    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=10))
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SCATTER & FRONTIER CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def create_risk_return_scatter(
    strategy_data: List[Dict],
    show_cml: bool = True
) -> go.Figure:
    """
    Create risk-return efficient frontier scatter plot.
    
    Args:
        strategy_data: List of dicts with Strategy, CAGR, Volatility, Sharpe, Max DD
        show_cml: Whether to show Capital Market Line
    
    Returns:
        Plotly Figure object
    """
    df = pd.DataFrame(strategy_data)
    
    if df.empty:
        return go.Figure()
    
    # Ensure positive size values for bubble
    df['Size'] = np.abs(df['Max DD']) * 100 + 5
    
    fig = go.Figure()
    
    # Strategy bubbles
    fig.add_trace(go.Scatter(
        x=df['Volatility'] * 100,
        y=df['CAGR'] * 100,
        mode='markers+text',
        marker=dict(
            size=df['Size'],
            color=df['Sharpe'],
            colorscale='RdYlGn',
            cmin=-1,
            cmax=2,
            showscale=True,
            colorbar=dict(
                title=dict(text='Sharpe', font=dict(color=COLORS['muted'])),
                tickfont=dict(color=COLORS['muted']),
                thickness=15
            ),
            line=dict(width=1, color=COLORS['border'])
        ),
        text=df['Strategy'].apply(lambda x: x[:12] + '...' if len(x) > 12 else x),
        textposition='top center',
        textfont=dict(size=9, color=COLORS['muted']),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'CAGR: %{y:.1f}%<br>'
            'Volatility: %{x:.1f}%<br>'
            'Sharpe: %{marker.color:.2f}<br>'
            'Max DD: %{customdata[1]:.1%}'
            '<extra></extra>'
        ),
        customdata=df[['Strategy', 'Max DD']].values,
        name='Strategies'
    ))
    
    # Capital Market Line
    if show_cml and len(df) > 2:
        max_sharpe_idx = df['Sharpe'].idxmax()
        tangent_vol = df.loc[max_sharpe_idx, 'Volatility'] * 100
        tangent_ret = df.loc[max_sharpe_idx, 'CAGR'] * 100
        
        cml_x = [0, tangent_vol * 1.8]
        cml_y = [0, tangent_ret * 1.8]
        
        fig.add_trace(go.Scatter(
            x=cml_x,
            y=cml_y,
            mode='lines',
            name='Capital Market Line',
            line=dict(color=COLORS['muted'], dash='dash', width=1.5)
        ))
        
        # Mark tangent portfolio
        fig.add_trace(go.Scatter(
            x=[tangent_vol],
            y=[tangent_ret],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(
                size=15,
                color=COLORS['primary'],
                symbol='star',
                line=dict(width=2, color=COLORS['text'])
            )
        ))
    
    layout = get_chart_layout(
        title="<b>Risk-Return Efficient Frontier</b><br><sup>Bubble size = Max Drawdown magnitude</sup>",
        height=500
    )
    fig.update_layout(**layout)
    fig.update_xaxes(**get_axis_config(title="Annualized Volatility (%)"))
    fig.update_yaxes(**get_axis_config(title="CAGR (%)"))
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# RADAR & FACTOR CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def create_factor_radar(
    factor_data: List[Dict],
    max_strategies: int = 4
) -> go.Figure:
    """
    Create strategy factor fingerprint radar chart.
    
    Args:
        factor_data: List of dicts with Strategy and factor scores
        max_strategies: Maximum strategies to display
    
    Returns:
        Plotly Figure object
    """
    df = pd.DataFrame(factor_data)
    
    if df.empty:
        return go.Figure()
    
    # Get top strategies by efficiency
    if 'Efficiency' in df.columns:
        df = df.nlargest(min(max_strategies, len(df)), 'Efficiency')
    else:
        df = df.head(max_strategies)
    
    # Factor categories (excluding Strategy column)
    categories = [col for col in df.columns if col != 'Strategy']
    
    fig = go.Figure()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[cat] for cat in categories]
        values.append(values[0])  # Close the polygon
        
        color = COLORS['palette'][idx % len(COLORS['palette'])]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=row['Strategy'][:20],
            line_color=color,
            fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
            opacity=0.8
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                tickfont=dict(size=9, color=COLORS['muted']),
                gridcolor=COLORS['border']
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS['text']),
                gridcolor=COLORS['border']
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.15,
            x=0.5,
            xanchor='center',
            font=dict(size=10)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=COLORS['text']),
        title=dict(
            text="<b>Strategy Factor Fingerprints</b>",
            font=dict(size=14),
            x=0.5
        ),
        height=450,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT EVOLUTION CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def create_weight_evolution_chart(
    weight_history: List[Dict],
    title: str = "Strategy Weight Evolution"
) -> go.Figure:
    """
    Create stacked area chart for weight evolution.
    
    Args:
        weight_history: List of dicts with date and strategy weights
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    if not weight_history:
        return go.Figure()
    
    df = pd.DataFrame(weight_history)
    if 'date' not in df.columns:
        return go.Figure()
    
    # Get weight columns
    weight_cols = [col for col in df.columns if col != 'date']
    
    fig = go.Figure()
    
    for idx, col in enumerate(weight_cols):
        color = COLORS['palette'][idx % len(COLORS['palette'])]
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[col],
            mode='lines',
            name=col[:20],
            stackgroup='one',
            line=dict(width=0.5, color=color),
            fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}'
        ))
    
    layout = get_chart_layout(
        title=f"<b>{title}</b>",
        height=400
    )
    layout['yaxis'] = dict(**get_axis_config(title="Weight"), tickformat='.0%')
    fig.update_layout(**layout)
    fig.update_xaxes(**get_axis_config(title="Date"))
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CONVICTION & SIGNAL CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def create_signal_heatmap(
    signal_data: pd.DataFrame,
    value_col: str = 'signal',
    symbol_col: str = 'symbol'
) -> go.Figure:
    """
    Create signal strength heatmap grid.
    
    Args:
        signal_data: DataFrame with symbol and signal columns
        value_col: Column name for signal values
        symbol_col: Column name for symbols
    
    Returns:
        Plotly Figure object
    """
    if signal_data.empty:
        return go.Figure()
    
    # Sort by signal strength
    sorted_df = signal_data.sort_values(value_col, ascending=False)
    
    symbols = sorted_df[symbol_col].apply(lambda x: x.replace('.NS', '')).tolist()
    values = sorted_df[value_col].tolist()
    
    # Create grid
    n_cols = min(8, len(symbols))
    n_rows = (len(symbols) + n_cols - 1) // n_cols
    
    # Pad to fill grid
    while len(symbols) < n_rows * n_cols:
        symbols.append('')
        values.append(0)
    
    z = np.array(values).reshape(n_rows, n_cols)
    text = np.array(symbols).reshape(n_rows, n_cols)
    
    colorscale = [
        [0.0, COLORS['danger']],
        [0.5, COLORS['neutral']],
        [1.0, COLORS['success']]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=10, color='white'),
        colorscale=colorscale,
        zmid=0,
        showscale=True,
        colorbar=dict(
            title=dict(text='Signal', font=dict(color=COLORS['muted'])),
            tickfont=dict(color=COLORS['muted']),
            thickness=15
        ),
        hovertemplate='%{text}<br>Signal: %{z:.2f}<extra></extra>'
    ))
    
    layout = get_chart_layout(
        height=max(200, n_rows * 50),
        show_legend=False
    )
    fig.update_layout(**layout)
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, autorange='reversed')
    
    return fig


def create_bar_chart(
    data: List[Dict],
    x_col: str,
    y_col: str,
    title: str = "",
    color_by_value: bool = True,
    horizontal: bool = False
) -> go.Figure:
    """
    Create standardized bar chart.
    
    Args:
        data: List of dicts with x and y values
        x_col: X-axis column name
        y_col: Y-axis column name
        title: Chart title
        color_by_value: Color bars by positive/negative
        horizontal: Create horizontal bar chart
    
    Returns:
        Plotly Figure object
    """
    df = pd.DataFrame(data)
    
    if df.empty:
        return go.Figure()
    
    if color_by_value:
        colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in df[y_col]]
    else:
        colors = COLORS['primary']
    
    if horizontal:
        fig = go.Figure(go.Bar(
            x=df[y_col],
            y=df[x_col],
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.2f}" if isinstance(v, (int, float)) else str(v) for v in df[y_col]],
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        fig.add_vline(x=0, line_color=COLORS['muted'], line_width=1)
    else:
        fig = go.Figure(go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color=colors,
            text=[f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in df[y_col]],
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        fig.add_hline(y=0, line_color=COLORS['muted'], line_width=1)
    
    layout = get_chart_layout(title=f"<b>{title}</b>" if title else "", height=400)
    fig.update_layout(**layout)
    fig.update_xaxes(**get_axis_config())
    fig.update_yaxes(**get_axis_config())
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'COLORS',
    'get_chart_layout',
    'get_axis_config',
    'create_equity_drawdown_chart',
    'create_rolling_metrics_chart',
    'create_correlation_heatmap',
    'create_tier_sharpe_heatmap',
    'create_risk_return_scatter',
    'create_factor_radar',
    'create_weight_evolution_chart',
    'create_signal_heatmap',
    'create_bar_chart'
]
