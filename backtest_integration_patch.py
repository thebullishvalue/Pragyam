"""
Backtest Engine Integration Patch
==================================

This module patches the existing backtest_engine.py to use the advanced
strategy selection logic from advanced_strategy_selector.py.

Usage:
    # Option 1: Import the patched version
    from backtest_engine_v3 import get_dynamic_portfolio_styles
    
    # Option 2: Patch at runtime
    from backtest_integration_patch import patch_backtest_engine
    patch_backtest_engine()
    
    # Then use normally
    from backtest_engine import get_dynamic_portfolio_styles

Author: Hemrek Capital
Version: 3.0.0
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger("BacktestIntegration")


def patch_backtest_engine():
    """
    Monkey-patch the existing backtest_engine module to use advanced selection.
    Call this before using any backtest_engine functions.
    """
    try:
        import backtest_engine
        from advanced_strategy_selector import (
            EnhancedDynamicPortfolioStylesGenerator,
            get_advanced_dynamic_portfolio_styles
        )
        
        # Patch the generator class
        backtest_engine.DynamicPortfolioStylesGenerator = EnhancedDynamicPortfolioStylesGenerator
        
        # Patch the main function
        backtest_engine.get_dynamic_portfolio_styles = get_advanced_dynamic_portfolio_styles
        
        logger.info("Successfully patched backtest_engine with advanced selection")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to patch backtest_engine: {e}")
        return False


def create_enhanced_engine_wrapper():
    """
    Create an enhanced wrapper around the existing engine.
    Returns a drop-in replacement class.
    """
    from backtest_engine import UnifiedBacktestEngine
    from advanced_strategy_selector import (
        AdvancedStrategySelector,
        AdvancedMetricsCalculator,
        RegimeDetector,
        MultiCriteriaOptimizer,
        DiversificationAnalyzer
    )
    
    class EnhancedBacktestEngine(UnifiedBacktestEngine):
        """
        Enhanced backtest engine with advanced strategy selection.
        Inherits from UnifiedBacktestEngine and adds advanced capabilities.
        """
        
        def __init__(self, capital: float = 10_000_000, risk_free_rate: float = 0.0):
            super().__init__(capital, risk_free_rate)
            
            # Initialize advanced components
            self.advanced_selector = AdvancedStrategySelector(
                risk_free_rate=risk_free_rate,
                bootstrap_samples=500
            )
            self.metrics_calculator = AdvancedMetricsCalculator(
                risk_free_rate=risk_free_rate
            )
            self.regime_detector = RegimeDetector()
            self.mco = MultiCriteriaOptimizer()
            self.diversification_analyzer = DiversificationAnalyzer()
        
        def select_top_strategies_advanced(
            self,
            results: Dict[str, Dict],
            mode: str,
            n_strategies: int = 4,
            market_returns: Optional[pd.Series] = None
        ) -> Tuple[List[str], Dict]:
            """
            Advanced strategy selection using multi-criteria optimization.
            
            Returns:
                Tuple of (selected_strategy_names, selection_metadata)
            """
            selection_result = self.advanced_selector.select_strategies(
                results,
                market_returns=market_returns.values if market_returns is not None else None,
                mode=mode,
                n_strategies=n_strategies,
                regime_aware=True
            )
            
            metadata = {
                'scores': selection_result.selection_scores,
                'confidence_intervals': selection_result.confidence_intervals,
                'diversification_benefit': selection_result.diversification_benefit,
                'expected_portfolio_sharpe': selection_result.expected_portfolio_sharpe,
                'regime_allocations': {
                    k.value: v for k, v in selection_result.regime_allocations.items()
                },
                'meta_breakdown': selection_result.meta_score_breakdown
            }
            
            return selection_result.selected_strategies, metadata
        
        def get_comprehensive_metrics(
            self,
            results: Dict[str, Dict]
        ) -> pd.DataFrame:
            """
            Get comprehensive metrics for all strategies.
            """
            metrics_list = []
            
            for name, data in results.items():
                daily_data = data.get('daily_data', pd.DataFrame())
                if daily_data.empty:
                    continue
                
                metrics = self.metrics_calculator.calculate(daily_data, name)
                
                metrics_list.append({
                    'Strategy': name,
                    'Total Return': metrics.total_return,
                    'CAGR': metrics.cagr,
                    'Volatility': metrics.volatility,
                    'Sharpe': metrics.sharpe_ratio,
                    'Sortino': metrics.sortino_ratio,
                    'Calmar': metrics.calmar_ratio,
                    'Omega': metrics.omega_ratio,
                    'Max DD': metrics.max_drawdown,
                    'CVaR 95%': metrics.cvar_95,
                    'CVaR 99%': metrics.cvar_99,
                    'Win Rate': metrics.win_rate,
                    'Profit Factor': metrics.profit_factor,
                    'Hurst': metrics.hurst_exponent,
                    'Skewness': metrics.skewness,
                    'Kurtosis': metrics.kurtosis,
                    'Sharpe p-value': metrics.sharpe_pvalue,
                    'Sharpe CI Low': metrics.bootstrap_sharpe_ci_lower,
                    'Sharpe CI High': metrics.bootstrap_sharpe_ci_upper,
                    'Stability': metrics.sharpe_stability
                })
            
            return pd.DataFrame(metrics_list)
        
        def detect_current_regime(
            self,
            market_returns: pd.Series
        ) -> Tuple[str, Dict[str, float]]:
            """
            Detect current market regime.
            
            Returns:
                Tuple of (regime_name, regime_probabilities)
            """
            regime, probs = self.regime_detector.detect_regime(market_returns.values)
            return regime.value, probs
        
        def compute_correlation_matrix(
            self,
            results: Dict[str, Dict]
        ) -> pd.DataFrame:
            """
            Compute strategy correlation matrix.
            """
            strategy_returns = {}
            
            for name, data in results.items():
                daily_data = data.get('daily_data', pd.DataFrame())
                if daily_data.empty or 'value' not in daily_data.columns:
                    continue
                
                values = daily_data['value'].values
                returns = pd.Series(values).pct_change().dropna().values
                
                if len(returns) >= 63:
                    strategy_returns[name] = returns
            
            if len(strategy_returns) < 2:
                return pd.DataFrame()
            
            return self.diversification_analyzer.compute_correlation_matrix(strategy_returns)
        
        def get_risk_parity_weights(
            self,
            results: Dict[str, Dict],
            strategies: List[str]
        ) -> Dict[str, float]:
            """
            Compute risk parity weights for selected strategies.
            """
            strategy_returns = {}
            
            for name in strategies:
                if name not in results:
                    continue
                    
                daily_data = results[name].get('daily_data', pd.DataFrame())
                if daily_data.empty or 'value' not in daily_data.columns:
                    continue
                
                values = daily_data['value'].values
                returns = pd.Series(values).pct_change().dropna().values
                
                if len(returns) >= 63:
                    strategy_returns[name] = returns
            
            if len(strategy_returns) < 2:
                return {name: 1.0 / len(strategies) for name in strategies}
            
            return self.diversification_analyzer.compute_risk_parity_weights(strategy_returns)
    
    return EnhancedBacktestEngine


# ============================================================================
# QUICK COMPARISON FUNCTION
# ============================================================================

def compare_selection_methods(
    backtest_results: Dict[str, Dict],
    mode: str = 'sip',
    n_strategies: int = 4
) -> pd.DataFrame:
    """
    Compare old vs new selection methods.
    
    Returns DataFrame showing which strategies are selected by each method.
    """
    from advanced_strategy_selector import AdvancedStrategySelector
    
    # Old method: simple metric ranking
    if mode.lower() == 'sip':
        metric_key = 'calmar_ratio'
    else:
        metric_key = 'sortino_ratio'
    
    # Extract old-style metrics
    old_scores = {}
    for name, data in backtest_results.items():
        metrics = data.get('metrics', {})
        score = metrics.get(metric_key, 0)
        if score > -100:
            old_scores[name] = score
    
    old_sorted = sorted(old_scores.items(), key=lambda x: x[1], reverse=True)
    old_selected = [name for name, _ in old_sorted[:n_strategies]]
    
    # New method: advanced selection
    selector = AdvancedStrategySelector()
    selection = selector.select_strategies(
        backtest_results,
        mode=mode,
        n_strategies=n_strategies
    )
    new_selected = selection.selected_strategies
    
    # Build comparison DataFrame
    all_strategies = set(old_selected) | set(new_selected)
    
    comparison_data = []
    for name in sorted(all_strategies):
        comparison_data.append({
            'Strategy': name,
            f'Old ({metric_key})': old_scores.get(name, 0),
            'Old Rank': old_selected.index(name) + 1 if name in old_selected else None,
            'Old Selected': name in old_selected,
            'New MCO Score': selection.selection_scores.get(name, 0),
            'New Rank': new_selected.index(name) + 1 if name in new_selected else None,
            'New Selected': name in new_selected,
            'Change': 'Added' if name in new_selected and name not in old_selected else (
                'Removed' if name not in new_selected and name in old_selected else 'Same'
            )
        })
    
    return pd.DataFrame(comparison_data).sort_values('New MCO Score', ascending=False)


# ============================================================================
# STREAMLIT UI ENHANCEMENT
# ============================================================================

def create_advanced_selection_ui():
    """
    Create enhanced Streamlit UI components for advanced selection.
    Returns a dictionary of UI component functions.
    """
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        return {}
    
    def render_regime_indicator(regime: str, probabilities: Dict[str, float]):
        """Render market regime indicator."""
        st.subheader("🌡️ Market Regime Detection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            regime_colors = {
                'bull': '🟢',
                'bear': '🔴',
                'high_volatility': '🟠',
                'low_volatility': '🔵',
                'trending': '🟣',
                'mean_reverting': '⚪',
                'crisis': '⚫',
                'recovery': '🟡'
            }
            st.metric(
                "Current Regime",
                f"{regime_colors.get(regime, '❓')} {regime.replace('_', ' ').title()}"
            )
        
        with col2:
            # Bar chart of regime probabilities
            fig = go.Figure(go.Bar(
                x=list(probabilities.values()),
                y=list(probabilities.keys()),
                orientation='h',
                marker_color='gold'
            ))
            fig.update_layout(
                title="Regime Probabilities",
                height=200,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Probability",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_selection_breakdown(meta_breakdown: Dict[str, Dict[str, float]]):
        """Render detailed selection breakdown."""
        st.subheader("📊 Selection Criteria Breakdown")
        
        if not meta_breakdown:
            st.info("No breakdown data available")
            return
        
        # Create radar chart for each selected strategy
        categories = ['Sharpe', 'Sortino', 'Calmar', 'Omega', 'Win Rate', 'Stability']
        
        fig = go.Figure()
        
        for name, scores in meta_breakdown.items():
            values = [
                min(scores.get('sharpe_ratio', 0) / 3, 1),  # Normalize to 0-1
                min(scores.get('sortino_ratio', 0) / 5, 1),
                min(scores.get('calmar_ratio', 0) / 3, 1),
                min(scores.get('omega_ratio', 0) / 3, 1),
                scores.get('win_rate', 0),
                1 - min(scores.get('sharpe_stability', 0.5), 1)  # Lower is better
            ]
            values.append(values[0])  # Close the radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=name[:20]  # Truncate long names
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_heatmap(corr_matrix: pd.DataFrame):
        """Render strategy correlation heatmap."""
        st.subheader("🔗 Strategy Correlations")
        
        if corr_matrix.empty:
            st.info("Insufficient data for correlation analysis")
            return
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale='RdYlGn',
            zmin=-1, zmax=1
        )
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_confidence_intervals(
        strategies: List[str],
        intervals: Dict[str, Tuple[float, float]],
        point_estimates: Dict[str, float]
    ):
        """Render bootstrap confidence intervals."""
        st.subheader("📈 Sharpe Ratio Confidence Intervals")
        
        if not intervals:
            st.info("No confidence interval data available")
            return
        
        fig = go.Figure()
        
        for name in strategies:
            if name in intervals:
                ci_low, ci_high = intervals[name]
                point = point_estimates.get(name, (ci_low + ci_high) / 2)
                
                fig.add_trace(go.Scatter(
                    x=[ci_low, point, ci_high],
                    y=[name, name, name],
                    mode='lines+markers',
                    marker=dict(size=[10, 15, 10], color=['gray', 'gold', 'gray']),
                    line=dict(color='gray', width=2),
                    name=name
                ))
        
        fig.update_layout(
            title="95% Bootstrap Confidence Intervals for Sharpe Ratio",
            xaxis_title="Sharpe Ratio",
            yaxis_title="Strategy",
            height=300,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_portfolio_metrics(
        diversification_ratio: float,
        expected_sharpe: float,
        n_strategies: int
    ):
        """Render portfolio-level metrics."""
        st.subheader("💼 Portfolio Characteristics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Diversification Ratio",
                f"{diversification_ratio:.2f}",
                delta=f"+{(diversification_ratio - 1) * 100:.1f}%" if diversification_ratio > 1 else None,
                help="Ratio > 1 indicates diversification benefit"
            )
        
        with col2:
            st.metric(
                "Expected Portfolio Sharpe",
                f"{expected_sharpe:.2f}",
                help="Risk-parity weighted portfolio Sharpe ratio"
            )
        
        with col3:
            st.metric(
                "Strategies Selected",
                n_strategies,
                help="Number of strategies in the portfolio"
            )
    
    return {
        'render_regime_indicator': render_regime_indicator,
        'render_selection_breakdown': render_selection_breakdown,
        'render_correlation_heatmap': render_correlation_heatmap,
        'render_confidence_intervals': render_confidence_intervals,
        'render_portfolio_metrics': render_portfolio_metrics
    }


if __name__ == "__main__":
    print("Backtest Integration Patch Module")
    print("=" * 50)
    print("\nUsage:")
    print("  from backtest_integration_patch import patch_backtest_engine")
    print("  patch_backtest_engine()")
    print("\nOr create enhanced engine:")
    print("  from backtest_integration_patch import create_enhanced_engine_wrapper")
    print("  EnhancedEngine = create_enhanced_engine_wrapper()")
    print("  engine = EnhancedEngine(capital=10_000_000)")
