"""
PRAGYAM Adaptive Strategy Selection Framework
=============================================

Market-driven strategy evaluation using REL_BREADTH triggers from breadth data.

Execution Modes:
- SIP: Accumulate when REL_BREADTH < 0.42
- Swing: Buy when REL_BREADTH < 0.42, Sell when REL_BREADTH >= 0.50

Strategy SCORING uses adaptive rank-based weights (no fixed formulas).
But the TRIGGERS are intentional domain values from market research.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS - INTENTIONAL MARKET TRIGGERS (NOT ARBITRARY)
# ══════════════════════════════════════════════════════════════════════════════

# REL_BREADTH thresholds - derived from market research, intentionally set
SIP_TRIGGER = 0.42      # Buy/accumulate when breadth falls below this
SWING_BUY = 0.42        # Enter swing position below this
SWING_SELL = 0.50       # Exit swing position at or above this

# Data source
BREADTH_SHEET_ID = "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c"
BREADTH_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{BREADTH_SHEET_ID}/export?format=csv"

# Lookback window
DEFAULT_LOOKBACK_ROWS = 400


# ══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_breadth_data(lookback_rows: int = DEFAULT_LOOKBACK_ROWS) -> pd.DataFrame:
    """
    Load REL_BREADTH data from Google Sheets.
    
    Source: https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c
    
    Args:
        lookback_rows: Number of recent rows to load (default: 400)
        
    Returns:
        DataFrame with DATE and REL_BREADTH columns
    """
    try:
        print(f"Fetching breadth data from Google Sheet...")
        df = pd.read_csv(BREADTH_SHEET_URL)
        print(f"Loaded {len(df)} rows from sheet")
        
        # Debug: print column names
        print(f"Columns found: {df.columns.tolist()}")
        
        # Standardize column names (strip whitespace, uppercase)
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Find DATE column
        if 'DATE' not in df.columns:
            date_cols = [c for c in df.columns if 'DATE' in c.upper()]
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'DATE'})
                print(f"Renamed '{date_cols[0]}' to 'DATE'")
            else:
                # Try first column as date
                df = df.rename(columns={df.columns[0]: 'DATE'})
                print(f"Using first column as DATE")
        
        # Find REL_BREADTH column
        if 'REL_BREADTH' not in df.columns:
            breadth_cols = [c for c in df.columns if 'REL' in c.upper() and 'BREADTH' in c.upper()]
            if not breadth_cols:
                breadth_cols = [c for c in df.columns if 'BREADTH' in c.upper()]
            if not breadth_cols:
                breadth_cols = [c for c in df.columns if 'REL' in c.upper()]
            
            if breadth_cols:
                df = df.rename(columns={breadth_cols[0]: 'REL_BREADTH'})
                print(f"Renamed '{breadth_cols[0]}' to 'REL_BREADTH'")
            else:
                print(f"ERROR: Could not find REL_BREADTH column. Available: {df.columns.tolist()}")
                return pd.DataFrame(columns=['DATE', 'REL_BREADTH'])
        
        # Parse dates
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        
        # Convert REL_BREADTH to numeric
        df['REL_BREADTH'] = pd.to_numeric(df['REL_BREADTH'], errors='coerce')
        
        # Drop invalid rows
        df = df.dropna(subset=['DATE', 'REL_BREADTH'])
        
        # Sort by date ascending
        df = df.sort_values('DATE', ascending=True)
        
        # Take most recent N rows
        df = df.tail(lookback_rows).reset_index(drop=True)
        
        print(f"Returning {len(df)} rows of breadth data")
        print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
        print(f"REL_BREADTH range: {df['REL_BREADTH'].min():.4f} to {df['REL_BREADTH'].max():.4f}")
        
        return df[['DATE', 'REL_BREADTH']]
        
    except Exception as e:
        print(f"Error loading breadth data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['DATE', 'REL_BREADTH'])


def get_trigger_info() -> Dict[str, float]:
    """
    Get the trigger configuration.
    These are intentional values from market research.
    """
    return {
        'sip_trigger': SIP_TRIGGER,
        'swing_buy': SWING_BUY,
        'swing_sell': SWING_SELL,
        'description': 'Hardcoded triggers from market research'
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIP MODE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def identify_sip_dates(breadth_data: pd.DataFrame) -> List[datetime]:
    """
    Identify all dates where SIP accumulation should occur.
    
    Trigger: REL_BREADTH < 0.42
    
    Args:
        breadth_data: DataFrame with DATE and REL_BREADTH
        
    Returns:
        List of dates meeting SIP trigger condition
    """
    if breadth_data.empty:
        return []
    
    sip_mask = breadth_data['REL_BREADTH'] < SIP_TRIGGER
    sip_dates = breadth_data.loc[sip_mask, 'DATE'].tolist()
    
    print(f"SIP Mode: Found {len(sip_dates)} trigger dates (REL_BREADTH < {SIP_TRIGGER})")
    
    return sip_dates


def execute_sip_mode(
    breadth_data: pd.DataFrame,
    strategy_returns: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """
    Execute SIP accumulation mode.
    
    Process:
    1. Find all dates where REL_BREADTH < 0.42
    2. For each trigger date, simulate buying/accumulating
    3. Track cumulative performance to terminal date
    4. Compute selection metrics from terminal portfolio
    
    Args:
        breadth_data: DataFrame with DATE and REL_BREADTH
        strategy_returns: Dict of strategy name -> DataFrame with date, return columns
        
    Returns:
        Dictionary of strategy metrics from SIP execution
    """
    sip_dates = identify_sip_dates(breadth_data)
    
    if not sip_dates:
        print("No SIP trigger dates found")
        return {}
    
    terminal_date = breadth_data['DATE'].max()
    results = {}
    
    for strategy_name, returns_df in strategy_returns.items():
        if returns_df.empty:
            continue
            
        # Ensure date column exists
        date_col = None
        for col in ['date', 'DATE', 'Date']:
            if col in returns_df.columns:
                date_col = col
                break
        
        if date_col is None:
            continue
            
        returns_df = returns_df.copy()
        returns_df[date_col] = pd.to_datetime(returns_df[date_col])
        
        # Find return column
        return_col = None
        for col in ['return', 'RETURN', 'Return', 'returns']:
            if col in returns_df.columns:
                return_col = col
                break
        
        if return_col is None:
            continue
        
        # Accumulate returns for each SIP entry
        sip_entries = []
        
        for entry_date in sip_dates:
            # Get returns from entry date to terminal date
            mask = (returns_df[date_col] >= entry_date) & (returns_df[date_col] <= terminal_date)
            future_returns = returns_df.loc[mask, return_col]
            
            if len(future_returns) > 0:
                # Calculate cumulative return from this entry
                equity = (1 + future_returns).cumprod()
                final_return = equity.iloc[-1] - 1 if len(equity) > 0 else 0
                
                sip_entries.append({
                    'entry_date': entry_date,
                    'final_return': final_return,
                    'periods': len(future_returns)
                })
        
        if sip_entries:
            df_entries = pd.DataFrame(sip_entries)
            
            # Terminal portfolio metrics (cumulative master portfolio)
            returns_array = df_entries['final_return'].values
            
            # Average return across all SIP entries
            avg_return = returns_array.mean()
            
            # Win rate
            win_rate = (returns_array > 0).mean()
            
            # Risk metrics
            if len(returns_array) > 1 and returns_array.std() > 0:
                sharpe = avg_return / returns_array.std()
                
                downside = returns_array[returns_array < 0]
                if len(downside) > 0 and downside.std() > 0:
                    sortino = avg_return / downside.std()
                else:
                    sortino = sharpe * 1.5  # No downside = premium
            else:
                sharpe = 0
                sortino = 0
            
            # Drawdown from cumulative SIP equity
            cumulative_equity = (1 + df_entries['final_return']).cumprod()
            peak = cumulative_equity.expanding().max()
            drawdown = (cumulative_equity / peak) - 1
            max_dd = drawdown.min()
            
            # Calmar ratio
            calmar = avg_return / abs(max_dd) if max_dd < 0 else avg_return * 10
            
            # Total compounded return
            total_return = cumulative_equity.iloc[-1] - 1
            
            results[strategy_name] = {
                'mode': 'SIP',
                'trigger': f'REL_BREADTH < {SIP_TRIGGER}',
                'entries': len(sip_dates),
                'avg_return': avg_return,
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_dd,
                'calmar': calmar
            }
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SWING MODE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def identify_swing_cycles(breadth_data: pd.DataFrame) -> List[Dict]:
    """
    Identify all swing trade cycles.
    
    Buy: REL_BREADTH < 0.42
    Sell: REL_BREADTH >= 0.50
    
    Args:
        breadth_data: DataFrame with DATE and REL_BREADTH
        
    Returns:
        List of cycle dictionaries with entry/exit dates
    """
    if breadth_data.empty:
        return []
    
    cycles = []
    in_position = False
    entry_date = None
    entry_breadth = None
    
    for _, row in breadth_data.iterrows():
        date = row['DATE']
        breadth = row['REL_BREADTH']
        
        if not in_position and breadth < SWING_BUY:
            # Enter position
            in_position = True
            entry_date = date
            entry_breadth = breadth
            
        elif in_position and breadth >= SWING_SELL:
            # Exit position
            cycles.append({
                'entry': entry_date,
                'exit': date,
                'entry_breadth': entry_breadth,
                'exit_breadth': breadth,
                'status': 'closed'
            })
            in_position = False
            entry_date = None
    
    # Handle open position (still in trade at end of data)
    if in_position and entry_date is not None:
        cycles.append({
            'entry': entry_date,
            'exit': breadth_data['DATE'].iloc[-1],
            'entry_breadth': entry_breadth,
            'exit_breadth': breadth_data['REL_BREADTH'].iloc[-1],
            'status': 'open'
        })
    
    print(f"Swing Mode: Found {len(cycles)} trade cycles")
    print(f"  Closed: {len([c for c in cycles if c['status'] == 'closed'])}")
    print(f"  Open: {len([c for c in cycles if c['status'] == 'open'])}")
    
    return cycles


def execute_swing_mode(
    breadth_data: pd.DataFrame,
    strategy_returns: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """
    Execute Swing trading mode.
    
    Process:
    1. Identify buy/sell cycles using 0.42/0.50 triggers
    2. Calculate returns for each completed cycle
    3. Compute selection metrics from net performance
    
    Args:
        breadth_data: DataFrame with DATE and REL_BREADTH
        strategy_returns: Dict of strategy name -> DataFrame with date, return columns
        
    Returns:
        Dictionary of strategy metrics from Swing execution
    """
    cycles = identify_swing_cycles(breadth_data)
    
    if not cycles:
        print("No swing cycles found")
        return {}
    
    results = {}
    
    for strategy_name, returns_df in strategy_returns.items():
        if returns_df.empty:
            continue
        
        # Find columns
        date_col = None
        for col in ['date', 'DATE', 'Date']:
            if col in returns_df.columns:
                date_col = col
                break
        
        return_col = None
        for col in ['return', 'RETURN', 'Return', 'returns']:
            if col in returns_df.columns:
                return_col = col
                break
        
        if date_col is None or return_col is None:
            continue
            
        returns_df = returns_df.copy()
        returns_df[date_col] = pd.to_datetime(returns_df[date_col])
        
        # Calculate returns for each cycle
        cycle_results = []
        
        for cycle in cycles:
            entry = cycle['entry']
            exit_date = cycle['exit']
            
            # Get returns within this cycle
            mask = (returns_df[date_col] >= entry) & (returns_df[date_col] <= exit_date)
            cycle_returns = returns_df.loc[mask, return_col]
            
            if len(cycle_returns) > 0:
                equity = (1 + cycle_returns).cumprod()
                cycle_return = equity.iloc[-1] - 1
                
                cycle_results.append({
                    'entry': entry,
                    'exit': exit_date,
                    'return': cycle_return,
                    'periods': len(cycle_returns),
                    'status': cycle['status']
                })
        
        if cycle_results:
            df_cycles = pd.DataFrame(cycle_results)
            
            # Separate closed and open trades
            closed = df_cycles[df_cycles['status'] == 'closed']
            open_trades = df_cycles[df_cycles['status'] == 'open']
            
            # Compute metrics from closed trades
            if len(closed) > 0:
                returns_array = closed['return'].values
                
                avg_return = returns_array.mean()
                win_rate = (returns_array > 0).mean()
                
                if len(returns_array) > 1 and returns_array.std() > 0:
                    sharpe = avg_return / returns_array.std()
                    
                    downside = returns_array[returns_array < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        sortino = avg_return / downside.std()
                    else:
                        sortino = sharpe * 1.5
                else:
                    sharpe = 0
                    sortino = 0
                
                # Drawdown from swing equity curve
                cumulative = (1 + closed['return']).cumprod()
                peak = cumulative.expanding().max()
                drawdown = (cumulative / peak) - 1
                max_dd = drawdown.min()
                
                # Total compounded return
                total_return = cumulative.iloc[-1] - 1
                
                # Calmar
                calmar = avg_return / abs(max_dd) if max_dd < 0 else avg_return * 10
                
                # Average holding period
                avg_periods = closed['periods'].mean()
                
                results[strategy_name] = {
                    'mode': 'Swing',
                    'buy_trigger': f'REL_BREADTH < {SWING_BUY}',
                    'sell_trigger': f'REL_BREADTH >= {SWING_SELL}',
                    'completed_trades': len(closed),
                    'open_trades': len(open_trades),
                    'avg_return': avg_return,
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'sharpe': sharpe,
                    'sortino': sortino,
                    'max_drawdown': max_dd,
                    'calmar': calmar,
                    'avg_holding_periods': avg_periods
                }
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE STRATEGY RANKING (No Fixed Weight Formulas)
# ══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_scores(metrics: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compute adaptive selection scores using rank-based weighting.
    
    NO FIXED FORMULAS like "0.30×Sharpe + 0.25×Sortino"
    
    Instead:
    - Each metric gets percentile rank (0-1)
    - Weights derived from cross-sectional dispersion
    - Metrics that better differentiate strategies get higher weight
    
    Args:
        metrics: Dictionary of strategy metrics
        
    Returns:
        DataFrame with adaptive scores
    """
    if not metrics:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df['strategy'] = df.index
    df = df.reset_index(drop=True)
    
    # Metrics to use for scoring (higher is better)
    positive_metrics = ['sharpe', 'sortino', 'calmar', 'win_rate', 'avg_return']
    # Metrics where lower is better
    negative_metrics = ['max_drawdown']
    
    # Compute percentile ranks
    for metric in positive_metrics:
        if metric in df.columns:
            df[f'{metric}_rank'] = df[metric].rank(pct=True)
    
    for metric in negative_metrics:
        if metric in df.columns:
            # Reverse rank: less negative (closer to 0) is better
            df[f'{metric}_rank'] = df[metric].rank(pct=True, ascending=False)
    
    # Get all rank columns
    rank_cols = [c for c in df.columns if c.endswith('_rank')]
    
    if not rank_cols:
        df['adaptive_score'] = 0
        return df
    
    # Compute dispersion (std) of each rank - higher dispersion = more discriminating
    dispersions = {}
    for col in rank_cols:
        std = df[col].std()
        dispersions[col] = std if std > 0 else 0.01  # Avoid zero
    
    total_dispersion = sum(dispersions.values())
    
    # Normalize dispersions to weights
    weights = {col: disp / total_dispersion for col, disp in dispersions.items()}
    
    # Compute weighted score
    df['adaptive_score'] = sum(df[col] * weight for col, weight in weights.items())
    
    # Sort by score
    df = df.sort_values('adaptive_score', ascending=False)
    
    # Store weight info
    df['_weight_info'] = str({k.replace('_rank', ''): f"{v:.2%}" for k, v in weights.items()})
    
    print("Adaptive Scoring Weights (derived from dispersion):")
    for col, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {col.replace('_rank', '')}: {weight:.2%}")
    
    return df


def compute_strategy_weights(
    scores_df: pd.DataFrame,
    concentration: float = 0.5
) -> Dict[str, float]:
    """
    Compute strategy allocation weights from adaptive scores.
    
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
    
    # Blend between equal weight and score-proportional
    equal_weight = 1.0 / len(normalized)
    score_weight = normalized / normalized.sum()
    
    blended = (1 - concentration) * equal_weight + concentration * score_weight
    weights = blended / blended.sum()
    
    return weights.to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveStrategySelector:
    """
    Market-driven strategy selection.
    
    - Triggers: Hardcoded 0.42/0.50 from market research
    - Scoring: Adaptive rank-based (no fixed weight formulas)
    
    Usage:
        selector = AdaptiveStrategySelector()
        selector.load_market_data()
        results = selector.evaluate_strategies(strategy_returns, mode='SIP')
        weights = selector.get_strategy_weights()
    """
    
    def __init__(self, lookback_rows: int = DEFAULT_LOOKBACK_ROWS):
        self.lookback_rows = lookback_rows
        self.breadth_data = None
        self.results = None
        self.scores = None
        self.weights = None
        
    def load_market_data(self) -> bool:
        """Load breadth data from Google Sheet."""
        self.breadth_data = load_breadth_data(self.lookback_rows)
        return not self.breadth_data.empty
    
    def evaluate_strategies(
        self, 
        strategy_returns: Dict[str, pd.DataFrame],
        mode: str = 'SIP'
    ) -> Dict[str, Dict]:
        """
        Evaluate strategies using REL_BREADTH triggers.
        
        Args:
            strategy_returns: Dict of strategy name -> returns DataFrame
            mode: 'SIP' or 'Swing'
            
        Returns:
            Dictionary of strategy metrics
        """
        if self.breadth_data is None or self.breadth_data.empty:
            if not self.load_market_data():
                print("Failed to load market data")
                return {}
        
        print(f"\nEvaluating strategies in {mode} mode...")
        print(f"Triggers: SIP/Buy < {SIP_TRIGGER}, Sell >= {SWING_SELL}")
        
        if mode.upper() == 'SIP':
            self.results = execute_sip_mode(self.breadth_data, strategy_returns)
        else:
            self.results = execute_swing_mode(self.breadth_data, strategy_returns)
        
        # Compute adaptive scores
        self.scores = compute_adaptive_scores(self.results)
        
        return self.results
    
    def get_strategy_weights(self, concentration: float = 0.5) -> Dict[str, float]:
        """Get adaptive strategy weights."""
        if self.scores is None or self.scores.empty:
            return {}
        self.weights = compute_strategy_weights(self.scores, concentration)
        return self.weights
    
    def get_scores_dataframe(self) -> pd.DataFrame:
        """Get the full scores DataFrame."""
        return self.scores if self.scores is not None else pd.DataFrame()
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "═" * 60,
            "ADAPTIVE STRATEGY SELECTION SUMMARY",
            "═" * 60,
            "",
            "TRIGGERS (Hardcoded from market research):",
            f"  SIP Accumulate: REL_BREADTH < {SIP_TRIGGER}",
            f"  Swing Buy:      REL_BREADTH < {SWING_BUY}",
            f"  Swing Sell:     REL_BREADTH >= {SWING_SELL}",
            "",
            f"Data Source: Google Sheet {BREADTH_SHEET_ID[:8]}...",
            f"Lookback: {self.lookback_rows} rows",
        ]
        
        if self.breadth_data is not None and not self.breadth_data.empty:
            lines.extend([
                f"Date Range: {self.breadth_data['DATE'].min().date()} to {self.breadth_data['DATE'].max().date()}",
                f"Breadth Range: {self.breadth_data['REL_BREADTH'].min():.4f} to {self.breadth_data['REL_BREADTH'].max():.4f}",
            ])
        
        if self.weights:
            lines.append("")
            lines.append("ADAPTIVE WEIGHTS (from dispersion-based scoring):")
            for strat, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
                lines.append(f"  {strat}: {weight:.2%}")
        
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'AdaptiveStrategySelector',
    'load_breadth_data',
    'identify_sip_dates',
    'identify_swing_cycles',
    'execute_sip_mode',
    'execute_swing_mode',
    'compute_adaptive_scores',
    'compute_strategy_weights',
    'get_trigger_info',
    'SIP_TRIGGER',
    'SWING_BUY',
    'SWING_SELL'
]
