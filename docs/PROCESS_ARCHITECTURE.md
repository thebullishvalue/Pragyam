# Pragyam v3.0 — Process Architecture

## Design Philosophy

Pragyam answers three orthogonal questions in sequence:

1. **Strategy Selection** (Phase 2): "Which strategies work best with our timing system?"
2. **Curation Quality** (Phase 3): "How well can we pick stocks using those strategies?"
3. **Portfolio Action** (Phase 4): "What should we buy/sell today?"

Each phase has a distinct statistical methodology. Conflating them (e.g., using triggers in curation evaluation) produces nonsense metrics.

---

## Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA ACQUISITION                                      │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Yahoo Finance │  │ Google Sheets│  │ Regime Detection     │  │
│  │ Price Data    │  │ REL_BREADTH  │  │ (Momentum, Breadth,  │  │
│  │ 100+ days     │  │ Trigger Data │  │  Volatility, Trend)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                  │                      │              │
│         ▼                  ▼                      ▼              │
│  training_data_window    trigger_df         suggested_mix       │
│  (100 trading days)      (buy/sell masks)   (Bull/Bear/Neutral) │
└─────────────────────────────────────────────────────────────────┘
                    │                │
                    ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: STRATEGY SELECTION (Trigger-Based Backtest)           │
│  Window: 100 days  │  Input: ALL 95+ strategies + trigger_df    │
│                                                                  │
│  Purpose: Identify which strategies perform best when combined   │
│  with our REL_BREADTH timing system.                            │
│                                                                  │
│  SIP Mode (TWR Methodology):                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ nav = 1.0                                               │    │
│  │ For each day:                                           │    │
│  │   1. Compute value of EXISTING holdings                 │    │
│  │   2. nav *= (1 + day_return)     ← market movement     │    │
│  │   3. If buy trigger: add positions ← no nav impact     │    │
│  │   4. Record nav as daily_value                          │    │
│  │                                                         │    │
│  │ Time-Weighted Return isolates investment performance    │    │
│  │ from capital injection effects (mutual fund standard)   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Swing Mode (NAV Tracking):                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ capital = 10M                                           │    │
│  │ For each day:                                           │    │
│  │   1. If buy trigger & no position: buy with capital     │    │
│  │   2. If sell trigger & in position: sell, return cash   │    │
│  │   3. daily_value = portfolio_value + cash               │    │
│  │                                                         │    │
│  │ Single-tranche NAV: capital preserved across cycles     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Metrics: Sharpe, Sortino, Calmar from daily_values              │
│  Selection: Top 4 by Calmar (SIP) or Sortino (Swing)           │
│  Output: 4 strategy names                                       │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼  (4 selected strategies)
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: WALK-FORWARD CURATION (Pure — NO Triggers)           │
│  Window: 50 days  │  Input: 4 selected strategies               │
│                                                                  │
│  Purpose: Evaluate the stock-picking quality of our curation     │
│  engine using out-of-sample walk-forward methodology.            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ For each day t in [MIN_TRAIN..N-1]:                     │    │
│  │                                                         │    │
│  │   TRAIN: [0..t-1]                                       │    │
│  │   ├── Run strategies on historical window               │    │
│  │   ├── Compute in-sample performance per strategy        │    │
│  │   └── Derive Sharpe-based exponential weights           │    │
│  │                                                         │    │
│  │   CURATE: day t                                         │    │
│  │   ├── Each strategy generates portfolio on day t data   │    │
│  │   ├── Blend portfolios using learned weights            │    │
│  │   └── Produce System_Curated portfolio                  │    │
│  │                                                         │    │
│  │   TEST: day t+1 (out-of-sample)                         │    │
│  │   ├── Observe actual market returns                     │    │
│  │   └── Record OOS return for System_Curated              │    │
│  │                                                         │    │
│  │ ~44 walk-forward steps → statistically meaningful       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Output: System_Curated + per-strategy OOS performance           │
│  Metrics: CAGR, Sharpe, Sortino, Calmar, MaxDD                  │
│  Used for: Phase 4 strategy weighting                           │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼  (strategy weights from walk-forward)
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: FINAL PORTFOLIO CURATION                              │
│  Window: Current day  │  Input: 4 strategies + their weights    │
│                                                                  │
│  1. Apply walk-forward-derived strategy weights                 │
│  2. Each strategy generates portfolio on CURRENT data           │
│  3. Blend positions: weight × strategy allocation               │
│  4. Apply tier-level Sharpe optimization within strategies      │
│  5. Position sizing with min/max constraints                    │
│  6. Output: actionable portfolio with ₹ allocations             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Bug Fixes Applied

### Bug 1: SIP Capital Injection (Phase 2)

**Before**: Each SIP buy trigger called `generate_portfolio(df, 10M)`, injecting fresh
10M every time. After 5 triggers, portfolio held ~50M in units. Metrics saw 10M → 50M
trajectory and computed +435% "return" — pure cash injection, not performance.

**Result**: ALL 95 strategies: Sharpe ~35, Sortino = 20 (clip ceiling), Calmar = 20.
Strategy selection was essentially random (first 4 alphabetically).

**Fix**: Time-Weighted Return (TWR / NAV-index). The nav_index tracks only market
movement of existing holdings. New SIP installments add units but don't affect the nav.
This is the same methodology mutual funds use to report performance — SEBI-mandated
for Indian AMCs.

**After**: Strategies will show realistic 5-15% returns with Sharpe 1-5, properly
differentiating stock-picking quality across timing signals.

### Bug 2: Phase 3 Used Trigger-Based Walk-Forward

**Before**: Phase 3 called `evaluate_historical_performance_trigger_based()`, which
only curated portfolios on buy/sell trigger days. Out of 50 days, only ~16 had triggers.
The other 34 days recorded 0% return, diluting the equity curve.

**Result**: SIP Phase 3: CAGR 3.6%, Sharpe 0.75 | Swing Phase 3: CAGR 1.6%, Sharpe 0.32.
These are meaningless — they measure "how often do triggers fire?" not "how good is our
stock-picking?"

**Fix**: Phase 3 always uses `evaluate_historical_performance()` (pure walk-forward).
Every day is a rebalancing day. This measures curation quality independent of timing.

**After**: ~44 daily OOS return observations, each representing a full portfolio
curation cycle. Metrics reflect actual stock-picking ability.

### Bug 3: Lookback Window Design

Phase 2 and Phase 3 serve different purposes and need different windows:

| Phase | Window | Rationale |
|-------|--------|-----------|
| Phase 2 | 100 days | Evaluate 95 strategies across varied regimes. Need diverse market conditions to avoid selecting strategies that only work in one regime. |
| Phase 3 | 50 days | Evaluate curation quality for the 4 selected strategies. Recent data is more relevant for adaptive weight learning. 50 days ≈ 44 walk-forward steps, providing statistically robust metrics. |

---

## Separation of Concerns

| Aspect | Phase 2 | Phase 3 |
|--------|---------|---------|
| **Question** | Which strategies? | How well do we curate? |
| **Uses triggers?** | YES | NO |
| **# Strategies** | All 95+ | Selected 4 |
| **Evaluation** | Full-period backtest | Walk-forward OOS |
| **Return methodology** | TWR (SIP) / NAV (Swing) | Daily OOS returns |
| **Output** | 4 strategy names | Performance-based weights |
