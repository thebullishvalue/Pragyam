"""
PRAGYAM Adaptive Strategy Selection Framework
=============================================

Market-driven strategy evaluation using REL_BREADTH triggers.
No fixed thresholds - let the market tell us what works.

Execution Modes:
- SIP: Accumulate when REL_BREADTH < trigger_low
- Swing: Buy when REL_BREADTH < trigger_low, Sell when REL_BREADTH >= trigger_high

The triggers themselves are derived from market distribution, not hardcoded.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

BREADTH_SHEET_URL = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/export?format=csv"

def load_breadth_data(lookback_rows: int = 400) -> pd.DataFrame:
    """
    Load REL_BREADTH data from Google Sheets.
    
    Args:
        lookback_rows: Number of recent rows to load
        
    Returns:
        DataFrame with DATE and REL_BREADTH columns
    """
    try:
        df = pd.read_csv(BREADTH_SHEET_URL)
        
        # Standardize column names
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Ensure required columns exist
        if 'DATE' not in df.columns:
            # Try common date column names
            date_cols = [c for c in df.columns if 'DATE' in c or 'TIME' in c]
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'DATE'})
        
        if 'REL_BREADTH' not in df.columns:
            breadth_cols = [c for c in df.columns if 'BREADTH' in c or 'REL' in c]
            if breadth_cols:
                df = df.rename(columns={breadth_cols[0]: 'REL_BREADTH'})
        
        # Parse dates
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE', 'REL_BREADTH'])
        df = df.sort_values('DATE', ascending=True)
        
        # Take most recent rows
        df = df.tail(lookback_rows).reset_index(drop=True)
        
        return df[['DATE', 'REL_BREADTH']]
        
    except Exception as e:
        print(f"Error loading breadth data: {e}")
        return pd.DataFrame(columns=['DATE', 'REL_BREADTH'])


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE TRIGGER COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_triggers(breadth_data: pd.DataFrame) -> Dict[str, float]:
    """
    Compute market-adaptive trigger levels from breadth distribution.
    
    Instead of fixed 0.42/0.50, we derive triggers from the actual
    distribution of REL_BREADTH values.
    
    Returns:
        Dictionary with adaptive trigger levels
    """
    if breadth_data.empty:
        # Fallback to standard triggers if no data
        return {
            'sip_trigger': 0.42,
            'swing_buy': 0.42,
            'swing_sell': 0.50,
            'method': 'fallback'
        }
    
    breadth = breadth_data['REL_BREADTH'].dropna()
    
    # Compute distribution statistics
    mean_breadth = breadth.mean()
    std_breadth = breadth.std()
    
    # Percentile-based triggers (market-derived)
    p20 = breadth.quantile(0.20)  # Bottom 20% = weakness
    p35 = breadth.quantile(0.35)  # Moderate weakness
    p50 = breadth.quantile(0.50)  # Median
    p65 = breadth.quantile(0.65)  # Recovery zone
    
    # SIP trigger: Buy during bottom quartile conditions
    # This adapts to the actual distribution of market weakness
    sip_trigger = p35  # Bottom 35% of historical breadth
    
    # Swing triggers: Buy in weakness, sell in recovery
    swing_buy = p35   # Same as SIP - buy in weakness
    swing_sell = p65  # Sell when breadth recovers to top 35%
    
    # Alternative: Z-score based triggers
    # sip_trigger = mean_breadth - 0.5 * std_breadth
    # swing_sell = mean_breadth + 0.5 * std_breadth
    
    return {
        'sip_trigger': sip_trigger,
        'swing_buy': swing_buy,
        'swing_sell': swing_sell,
        'mean_breadth': mean_breadth,
        'std_breadth': std_breadth,
        'p20': p20,
        'p35': p35,
        'p50': p50,
        'p65': p65,
        'method': 'adaptive_percentile'
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIP MODE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def execute_sip_mode(
    breadth_data: pd.DataFrame,
    strategy_returns: Dict[str, pd.DataFrame],
    triggers: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Execute SIP accumulation mode.
    
    For each date where REL_BREADTH < sip_trigger:
    - Initialize/accumulate portfolio for each strategy
    - Track cumulative performance
    
    Args:
        breadth_data: DataFrame with DATE and REL_BREADTH
        strategy_returns: Dict of strategy name -> DataFrame with date, symbol, return
        triggers: Adaptive trigger levels
        
    Returns:
        Dictionary of strategy metrics from SIP execution
    """
    sip_trigger = triggers['sip_trigger']
    
    # Find SIP entry dates (weakness conditions)
    sip_dates = breadth_data[breadth_data['REL_BREADTH'] < sip_trigger]['DATE'].tolist()
    
    if not sip_dates:
        return {}
    
    results = {}
    
    for strategy_name, returns_df in strategy_returns.items():
        if returns_df.empty:
            continue
            
        # Ensure date column
        if 'date' not in returns_df.columns:
            continue
            
        returns_df = returns_df.copy()
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        
        # Accumulate returns for each SIP entry
        cumulative_returns = []
        
        for entry_date in sip_dates:
            # Get returns from entry date forward
            future_returns = returns_df[returns_df['date'] >= entry_date]
            
            if not future_returns.empty:
                # Calculate cumulative return from this entry
                if 'return' in future_returns.columns:
                    equity = (1 + future_returns['return']).cumprod()
                    final_return = equity.iloc[-1] - 1
                    cumulative_returns.append({
                        'entry_date': entry_date,
                        'final_return': final_return,
                        'periods': len(future_returns)
                    })
        
        if cumulative_returns:
            df_entries = pd.DataFrame(cumulative_returns)
            
            # Compute SIP metrics
            avg_return = df_entries['final_return'].mean()
            win_rate = (df_entries['final_return'] > 0).mean()
            
            # Risk-adjusted metrics
            returns_array = df_entries['final_return'].values
            if len(returns_array) > 1 and returns_array.std() > 0:
                sharpe = returns_array.mean() / returns_array.std()
                
                # Downside deviation
                downside = returns_array[returns_array < 0]
                if len(downside) > 0:
                    sortino = returns_array.mean() / downside.std()
                else:
                    sortino = sharpe * 1.5  # No downside = premium
            else:
                sharpe = 0
                sortino = 0
            
            # Max drawdown from SIP entries
            equity_curve = (1 + df_entries['final_return']).cumprod()
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve / peak) - 1
            max_dd = drawdown.min()
            
            # Calmar
            cagr = avg_return  # Simplified for SIP
            calmar = cagr / abs(max_dd) if max_dd < 0 else cagr * 10
            
            results[strategy_name] = {
                'mode': 'SIP',
                'entries': len(sip_dates),
                'avg_return': avg_return,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_dd,
                'calmar': calmar,
                'total_return': (1 + avg_return) ** len(sip_dates) - 1
            }
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SWING MODE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def execute_swing_mode(
    breadth_data: pd.DataFrame,
    strategy_returns: Dict[str, pd.DataFrame],
    triggers: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Execute Swing trading mode.
    
    Buy when REL_BREADTH < swing_buy
    Sell when REL_BREADTH >= swing_sell
    
    Args:
        breadth_data: DataFrame with DATE and REL_BREADTH
        strategy_returns: Dict of strategy name -> DataFrame with date, symbol, return
        triggers: Adaptive trigger levels
        
    Returns:
        Dictionary of strategy metrics from Swing execution
    """
    swing_buy = triggers['swing_buy']
    swing_sell = triggers['swing_sell']
    
    # Identify buy/sell cycles
    cycles = []
    in_position = False
    entry_date = None
    
    for _, row in breadth_data.iterrows():
        date = row['DATE']
        breadth = row['REL_BREADTH']
        
        if not in_position and breadth < swing_buy:
            # Enter position
            in_position = True
            entry_date = date
        elif in_position and breadth >= swing_sell:
            # Exit position
            cycles.append({
                'entry': entry_date,
                'exit': date
            })
            in_position = False
            entry_date = None
    
    # Handle open position (still in trade)
    if in_position and entry_date is not None:
        cycles.append({
            'entry': entry_date,
            'exit': breadth_data['DATE'].iloc[-1],
            'open': True
        })
    
    if not cycles:
        return {}
    
    results = {}
    
    for strategy_name, returns_df in strategy_returns.items():
        if returns_df.empty:
            continue
            
        if 'date' not in returns_df.columns:
            continue
            
        returns_df = returns_df.copy()
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        
        # Calculate returns for each cycle
        cycle_returns = []
        
        for cycle in cycles:
            entry = cycle['entry']
            exit_date = cycle['exit']
            
            # Get returns within this cycle
            mask = (returns_df['date'] >= entry) & (returns_df['date'] <= exit_date)
            cycle_data = returns_df[mask]
            
            if not cycle_data.empty and 'return' in cycle_data.columns:
                equity = (1 + cycle_data['return']).cumprod()
                cycle_return = equity.iloc[-1] - 1
                
                cycle_returns.append({
                    'entry': entry,
                    'exit': exit_date,
                    'return': cycle_return,
                    'periods': len(cycle_data),
                    'open': cycle.get('open', False)
                })
        
        if cycle_returns:
            df_cycles = pd.DataFrame(cycle_returns)
            
            # Compute Swing metrics
            completed = df_cycles[~df_cycles['open']]
            
            if len(completed) > 0:
                avg_return = completed['return'].mean()
                win_rate = (completed['return'] > 0).mean()
                
                returns_array = completed['return'].values
                if len(returns_array) > 1 and returns_array.std() > 0:
                    sharpe = returns_array.mean() / returns_array.std()
                    
                    downside = returns_array[returns_array < 0]
                    if len(downside) > 0:
                        sortino = returns_array.mean() / downside.std()
                    else:
                        sortino = sharpe * 1.5
                else:
                    sharpe = 0
                    sortino = 0
                
                # Drawdown from swing equity curve
                cumulative = (1 + completed['return']).cumprod()
                peak = cumulative.expanding().max()
                drawdown = (cumulative / peak) - 1
                max_dd = drawdown.min()
                
                # Total compounded return
                total_return = cumulative.iloc[-1] - 1
                
                # Annualize (approximate)
                avg_periods = completed['periods'].mean()
                trades_per_year = 52 / avg_periods if avg_periods > 0 else 12
                cagr = ((1 + total_return) ** (1 / max(1, len(completed) / trades_per_year))) - 1
                
                calmar = cagr / abs(max_dd) if max_dd < 0 else cagr * 10
                
                results[strategy_name] = {
                    'mode': 'Swing',
                    'completed_trades': len(completed),
                    'open_trades': len(df_cycles) - len(completed),
                    'avg_return': avg_return,
                    'win_rate': win_rate,
                    'sharpe': sharpe,
                    'sortino': sortino,
                    'max_drawdown': max_dd,
                    'calmar': calmar,
                    'total_return': total_return,
                    'avg_holding_periods': avg_periods
                }
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE STRATEGY RANKING
# ══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_scores(
    metrics: Dict[str, Dict],
    mode: str = 'SIP'
) -> pd.DataFrame:
    """
    Compute adaptive selection scores without fixed thresholds.
    
    Uses rank-based scoring where weights are derived from
    the cross-sectional distribution of strategy performance.
    
    Args:
        metrics: Dictionary of strategy metrics
        mode: 'SIP' or 'Swing'
        
    Returns:
        DataFrame with adaptive scores
    """
    if not metrics:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df['strategy'] = df.index
    df = df.reset_index(drop=True)
    
    # Core metrics for scoring
    score_metrics = ['sharpe', 'sortino', 'calmar', 'win_rate']
    penalty_metrics = ['max_drawdown']  # Lower is better
    
    # Rank-based scoring (no fixed weights)
    # Each metric gets equal initial weight, then adjusted by dispersion
    
    for metric in score_metrics:
        if metric in df.columns:
            # Percentile rank (0 to 1)
            df[f'{metric}_rank'] = df[metric].rank(pct=True)
    
    for metric in penalty_metrics:
        if metric in df.columns:
            # Reverse rank for penalties (less negative = better)
            df[f'{metric}_rank'] = df[metric].rank(pct=True, ascending=False)
    
    # Compute dispersion-weighted composite score
    # Metrics with higher cross-sectional dispersion get more weight
    # (they better differentiate between strategies)
    
    rank_cols = [c for c in df.columns if c.endswith('_rank')]
    
    if rank_cols:
        # Calculate dispersion (std) of each rank metric
        dispersions = {col: df[col].std() for col in rank_cols}
        total_dispersion = sum(dispersions.values())
        
        if total_dispersion > 0:
            # Normalize dispersions to weights
            weights = {col: disp / total_dispersion for col, disp in dispersions.items()}
        else:
            # Equal weights if no dispersion
            weights = {col: 1.0 / len(rank_cols) for col in rank_cols}
        
        # Compute weighted score
        df['adaptive_score'] = sum(df[col] * weight for col, weight in weights.items())
        
        # Store weights for transparency
        df['weight_config'] = str(weights)
    else:
        df['adaptive_score'] = 0
    
    # Sort by adaptive score
    df = df.sort_values('adaptive_score', ascending=False)
    
    return df


def compute_adaptive_strategy_weights(
    scores_df: pd.DataFrame,
    concentration: float = 0.5
) -> Dict[str, float]:
    """
    Compute strategy allocation weights adaptively.
    
    Uses score distribution to determine weights,
    not fixed allocation rules.
    
    Args:
        scores_df: DataFrame with adaptive_score column
        concentration: 0 = equal weight, 1 = winner-take-all
        
    Returns:
        Dictionary of strategy -> weight
    """
    if scores_df.empty or 'adaptive_score' not in scores_df.columns:
        return {}
    
    scores = scores_df.set_index('strategy')['adaptive_score']
    
    # Normalize scores to 0-1
    score_min = scores.min()
    score_max = scores.max()
    
    if score_max > score_min:
        normalized = (scores - score_min) / (score_max - score_min)
    else:
        normalized = pd.Series(1.0, index=scores.index)
    
    # Apply concentration factor
    # concentration=0: equal weights
    # concentration=1: score-proportional weights
    
    equal_weight = 1.0 / len(normalized)
    score_weight = normalized / normalized.sum()
    
    blended = (1 - concentration) * equal_weight + concentration * score_weight
    
    # Normalize to sum to 1
    weights = blended / blended.sum()
    
    return weights.to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# REGIME SENSITIVITY ADAPTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_regime_sensitivity(
    breadth_data: pd.DataFrame,
    strategy_returns: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Compute how sensitive each strategy is to market regime changes.
    
    Strategies with higher regime sensitivity should get higher
    weight adjustment during regime transitions.
    
    Returns:
        Dictionary of strategy -> regime_sensitivity score
    """
    if breadth_data.empty:
        return {}
    
    # Define regimes based on breadth distribution
    breadth = breadth_data['REL_BREADTH']
    p33 = breadth.quantile(0.33)
    p67 = breadth.quantile(0.67)
    
    # Classify each date into regime
    def classify_regime(b):
        if b < p33:
            return 'bearish'
        elif b > p67:
            return 'bullish'
        else:
            return 'neutral'
    
    breadth_data = breadth_data.copy()
    breadth_data['regime'] = breadth_data['REL_BREADTH'].apply(classify_regime)
    
    sensitivity = {}
    
    for strategy_name, returns_df in strategy_returns.items():
        if returns_df.empty or 'date' not in returns_df.columns:
            continue
            
        returns_df = returns_df.copy()
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        
        # Merge with regime data
        merged = returns_df.merge(
            breadth_data[['DATE', 'regime']], 
            left_on='date', 
            right_on='DATE', 
            how='inner'
        )
        
        if merged.empty or 'return' not in merged.columns:
            continue
        
        # Calculate returns by regime
        regime_returns = merged.groupby('regime')['return'].agg(['mean', 'std'])
        
        if len(regime_returns) >= 2:
            # Sensitivity = how different are returns across regimes
            regime_means = regime_returns['mean'].values
            sensitivity_score = np.std(regime_means) / (np.mean(np.abs(regime_means)) + 1e-6)
            sensitivity[strategy_name] = sensitivity_score
        else:
            sensitivity[strategy_name] = 0.5  # Neutral if insufficient data
    
    return sensitivity


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ADAPTIVE SELECTION INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveStrategySelector:
    """
    Market-driven strategy selection without fixed thresholds.
    
    Usage:
        selector = AdaptiveStrategySelector()
        selector.load_market_data()
        results = selector.evaluate_strategies(strategy_returns, mode='SIP')
        weights = selector.get_adaptive_weights()
    """
    
    def __init__(self, lookback_rows: int = 400):
        self.lookback_rows = lookback_rows
        self.breadth_data = None
        self.triggers = None
        self.results = None
        self.scores = None
        self.weights = None
        self.regime_sensitivity = None
        
    def load_market_data(self) -> bool:
        """Load breadth data and compute adaptive triggers."""
        self.breadth_data = load_breadth_data(self.lookback_rows)
        
        if self.breadth_data.empty:
            return False
            
        self.triggers = compute_adaptive_triggers(self.breadth_data)
        return True
    
    def evaluate_strategies(
        self, 
        strategy_returns: Dict[str, pd.DataFrame],
        mode: str = 'SIP'
    ) -> Dict[str, Dict]:
        """
        Evaluate strategies using market-driven triggers.
        
        Args:
            strategy_returns: Dict of strategy name -> returns DataFrame
            mode: 'SIP' or 'Swing'
            
        Returns:
            Dictionary of strategy metrics
        """
        if self.breadth_data is None or self.triggers is None:
            self.load_market_data()
        
        if mode.upper() == 'SIP':
            self.results = execute_sip_mode(
                self.breadth_data, 
                strategy_returns, 
                self.triggers
            )
        else:
            self.results = execute_swing_mode(
                self.breadth_data, 
                strategy_returns, 
                self.triggers
            )
        
        # Compute adaptive scores
        self.scores = compute_adaptive_scores(self.results, mode)
        
        # Compute regime sensitivity
        self.regime_sensitivity = compute_regime_sensitivity(
            self.breadth_data, 
            strategy_returns
        )
        
        return self.results
    
    def get_adaptive_weights(self, concentration: float = 0.5) -> Dict[str, float]:
        """
        Get market-derived strategy weights.
        
        Args:
            concentration: 0 = equal weight, 1 = score-proportional
            
        Returns:
            Dictionary of strategy -> weight
        """
        if self.scores is None:
            return {}
            
        self.weights = compute_adaptive_strategy_weights(self.scores, concentration)
        return self.weights
    
    def get_trigger_info(self) -> Dict:
        """Get current adaptive trigger levels."""
        return self.triggers or {}
    
    def get_regime_info(self) -> Dict:
        """Get current regime sensitivity info."""
        return self.regime_sensitivity or {}
    
    def summary(self) -> str:
        """Get human-readable summary of selection."""
        if self.triggers is None:
            return "No data loaded"
        
        lines = [
            "═" * 50,
            "ADAPTIVE STRATEGY SELECTION SUMMARY",
            "═" * 50,
            f"Method: {self.triggers.get('method', 'unknown')}",
            f"SIP Trigger: {self.triggers.get('sip_trigger', 0):.4f}",
            f"Swing Buy: {self.triggers.get('swing_buy', 0):.4f}",
            f"Swing Sell: {self.triggers.get('swing_sell', 0):.4f}",
            "",
            "Distribution Stats:",
            f"  Mean Breadth: {self.triggers.get('mean_breadth', 0):.4f}",
            f"  Std Breadth: {self.triggers.get('std_breadth', 0):.4f}",
            f"  P20: {self.triggers.get('p20', 0):.4f}",
            f"  P35: {self.triggers.get('p35', 0):.4f}",
            f"  P50: {self.triggers.get('p50', 0):.4f}",
            f"  P65: {self.triggers.get('p65', 0):.4f}",
        ]
        
        if self.weights:
            lines.append("")
            lines.append("Adaptive Weights:")
            for strat, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
                lines.append(f"  {strat}: {weight:.2%}")
        
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'AdaptiveStrategySelector',
    'load_breadth_data',
    'compute_adaptive_triggers',
    'execute_sip_mode',
    'execute_swing_mode',
    'compute_adaptive_scores',
    'compute_adaptive_strategy_weights',
    'compute_regime_sensitivity'
]
