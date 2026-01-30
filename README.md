# PRAGYAM v2.0 - Dynamic Strategy Selection Integration

## Overview

This update transforms Pragyam from using hardcoded strategy dictionaries to **dynamically selecting strategies based on backtest performance**.

### Key Changes

| Mode | Selection Criteria | Rationale |
|------|-------------------|-----------|
| **SIP Investment** | Top 4 by **Calmar Ratio** | Best drawdown recovery for long-term wealth accumulation |
| **Swing Trading** | Top 4 by **Sortino Ratio** | Best risk-adjusted returns for short-term holds |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRAGYAM v2.0                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  backdata   │───▶│ backtest_engine  │───▶│  Dynamic      │  │
│  │  (data API) │    │ (unified engine) │    │  PORTFOLIO    │  │
│  └─────────────┘    └──────────────────┘    │  _STYLES      │  │
│         │                    │              └───────────────┘  │
│         │                    │                      │          │
│         ▼                    ▼                      ▼          │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ DataCache   │    │   95 Strategies  │    │   Portfolio   │  │
│  │  Manager    │    │  (strategies.py) │    │   Curation    │  │
│  └─────────────┘    └──────────────────┘    └───────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Files

### New Files

1. **`backtest_engine.py`** - Unified backtesting engine
   - `UnifiedBacktestEngine` - Core backtesting class
   - `DynamicPortfolioStylesGenerator` - Generates PORTFOLIO_STYLES
   - `DataCacheManager` - Shared data caching
   - `PerformanceMetrics` - Institutional-grade metrics calculator

2. **`app_updated.py`** - Updated main application
   - Integrates dynamic strategy selection
   - Toggle between dynamic/static modes
   - Strategy leaderboard display

### Modified Files

- `backdata.py` - No changes needed (already exports required functions)
- `strategies.py` - No changes needed (all 95 strategies loaded automatically)

## Integration Flow

```
1. User clicks "Run Analysis"
         │
         ▼
2. If Dynamic Mode enabled:
   ┌─────────────────────────────────┐
   │ Run Backtest Engine             │
   │   • Fetch 365 days of data      │
   │   • Load all 95 strategies      │
   │   • Run SIP backtest            │
   │   • Run Swing backtest          │
   │   • Calculate metrics           │
   │   • Select top 4 by mode        │
   └─────────────────────────────────┘
         │
         ▼
3. Generate Dynamic PORTFOLIO_STYLES
         │
         ▼
4. Continue with normal portfolio curation
```

## Usage

### Running the Application

```bash
# Navigate to project directory
cd Pragyam-main

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_updated.py
```

### Dynamic vs Static Mode

The sidebar includes a toggle to switch between:

- **Dynamic Mode**: Strategies selected based on live backtest results
- **Static Mode**: Uses predefined strategy selections (fallback)

### Strategy Selection Criteria

**SIP Investment Mode (Calmar Ratio)**:
- Calmar = Annual Return / Max Drawdown
- Prioritizes strategies that recover quickly from drawdowns
- Better for long-term systematic investing

**Swing Trading Mode (Sortino Ratio)**:
- Sortino = Excess Return / Downside Deviation
- Prioritizes strategies with good upside, limited downside
- Better for short-term momentum capture

## Performance Optimizations

1. **Shared Data Cache**
   - `DataCacheManager` singleton prevents duplicate API calls
   - 30-minute TTL for cached data
   - Automatic key generation based on symbols and dates

2. **Strategy Caching**
   - Strategies loaded once per session
   - Reused across multiple backtest runs

3. **Parallel Processing Ready**
   - Engine architecture supports ThreadPoolExecutor
   - Can be enabled for larger universes

## API Reference

### UnifiedBacktestEngine

```python
engine = UnifiedBacktestEngine(capital=10_000_000)

# Load data
engine.load_data(symbols, start_date, end_date)

# Load strategies
engine.load_strategies()

# Run backtest
results = engine.run_backtest(mode='sip')

# Select top strategies
top_strategies = engine.select_top_strategies(results, mode='sip', n_strategies=4)
```

### DynamicPortfolioStylesGenerator

```python
generator = DynamicPortfolioStylesGenerator(engine)

# Run comprehensive backtest
sip_results, swing_results = generator.run_comprehensive_backtest()

# Generate PORTFOLIO_STYLES
portfolio_styles = generator.generate_portfolio_styles(n_strategies=4)

# Get leaderboard
leaderboard = generator.get_strategy_leaderboard('sip')
```

### Quick Integration

```python
from backtest_engine import get_dynamic_portfolio_styles

# One-liner to get dynamic styles
portfolio_styles = get_dynamic_portfolio_styles(
    symbols=SYMBOLS_UNIVERSE,
    start_date=start_date,
    end_date=end_date,
    n_strategies=4
)
```

## Metrics Calculated

| Metric | Description |
|--------|-------------|
| Total Return | Overall return from start to end |
| CAGR | Compound Annual Growth Rate |
| Volatility | Annualized standard deviation |
| Sharpe Ratio | Risk-adjusted return (vs risk-free) |
| Sortino Ratio | Return vs downside deviation |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | Annual return / Max drawdown |
| Win Rate | Percentage of positive days |

## Error Handling

- Graceful fallback to static mode if backtest fails
- Automatic retry with cached data
- Detailed logging for debugging

## File Structure

```
Pragyam-main/
├── app_updated.py          # Updated main app with dynamic selection
├── backtest_engine.py      # New unified backtest engine
├── backdata.py             # Data fetching module (unchanged)
├── strategies.py           # 95 trading strategies (unchanged)
├── pragati.py              # Original app (preserved)
├── symbols.txt             # Stock universe
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Migration Notes

To migrate from the original `app.py`:

1. Copy `backtest_engine.py` to your project
2. Replace `app.py` with `app_updated.py`
3. Ensure `strategies.py` and `backdata.py` are in the same directory
4. Run: `streamlit run app_updated.py`

The original files (`app.py`, `pragati.py`) are preserved and can be used as fallback.

## Support

For issues or questions, check:
- System logs: `system.log`
- Enable debug logging in `backtest_engine.py`

---

*Hemrek Capital - Institutional Portfolio Intelligence*
