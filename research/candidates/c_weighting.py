"""
research/candidates/c_weighting.py — FAMILY C: rebalancing premium & return-seeking weighting schemes.

DECLARED BEFORE ANY CANDIDATE WAS RUN (2026-10-03). Discovery only (Feb 2007 – Dec 2019, E1 < 2014,
E2 2014-2019), Nifty 50 and Dow 30, every name held, harness = research/style_search.py.

LITERATURE BASIS
────────────────
* Booth & Fama (1992, FAJ 48(3) "Diversification Returns and Asset Contributions"): an asset's
  contribution to a rebalanced portfolio's compound return exceeds its own compound return by its
  diversification return ½(σᵢ² − σᵢ,ₚ). Summed over the book this is the excess growth rate
        γ*(w) = ½ (Σ wᵢσᵢ² − w'Σw)          (Fernholz & Shay 1982, JF; Fernholz 2002, SPT)
  — the growth a constant-mix (monthly rebalanced) book earns over the weighted average of its
  members' own growth rates. It is larger the more weight sits on HIGH-variance, LOW-correlation
  names: the opposite of risk parity / low-volatility weighting.
* Willenbrock (2011, FAJ 67(4)): the diversification return is earned BY rebalancing (selling
  relative winners, buying relative losers); it explains most of a monthly rebalanced commodity
  index's excess return.
* Bouchey, Nemtchinov, Paulsen & Stein (2012, J. Wealth Mgmt 15(2) "Volatility Harvesting"): more
  growth is harvested as individual volatilities rise and correlations fall; the equal-weight edge
  over cap weight is attributed to this.
* Ding & Qi (2023, arXiv 2303.01657 "An Optimization Study of Diversification Return Portfolios"):
  derive the maximum-diversification-return portfolio under a given risk level (the efficient DR
  frontier) and show it coincides with the Markowitz frontier when expected returns are
  proportional to variances — i.e., under the prior used below.
* MacLean, Thorp & Ziemba (2011, "The Kelly Capital Growth Investment Criterion", World Scientific):
  growth-optimal (Kelly) sizing is dangerously sensitive to mean estimates; fractional Kelly
  (λ > 1 times the log-utility risk aversion) trades growth for a smoother path. Half Kelly is the
  customary default.
* Kirby & Ostdiek (2012, JFQA 47(2) "It's All in the Timing"): weights ∝ (1/σᵢ²)^η, ignoring
  correlations, with a tuning exponent η, beat 1/N net of costs at low turnover. Here the
  inverse-vol direction is ALREADY known to lose outside the Nifty 2007-13 crash era (ERC/HRP), so
  candidate C1 uses the mirror image (positive exponent on σ), the volatility-harvesting direction.
* DeMiguel, Garlappi, Nogales & Uppal (2009, Mgmt Sci "Constraining Portfolio Norms") and
  Jagannathan & Ma (2003, JF): weight bounds act as shrinkage; the 1/(2N) floor below is that kind
  of constraint, and it is what keeps every name held (no corner solutions).
* Tu & Zhou (2011, JFE "Markowitz meets Talmud"), Kan & Zhou (2007, JFQA), Jorion (1986, JFQA):
  combine / shrink sample rules toward 1/N. With ~252 daily observations and 30–50 names the
  Bayes-Stein intensity on means is ≈ 1, i.e., all expected returns collapse to a common prior —
  so the candidates here take the prior seriously instead of estimating means at all.

EVIDENCE AGAINST (stated up front)
──────────────────────────────────
* Low-risk anomaly: low-volatility stocks earn higher risk-adjusted (often also raw) returns, and
  the effect is strong in India (Joshipura & Joshipura 2016/2020: NSE top-500, 2000-2018, low-minus-
  high volatility alpha spread ≈ 25%/yr). A high-σ tilt bets against it. This repo's own
  measurement agrees in the Nifty 2007-13 crash era (HRP/ERC won there).
* Cuthbertson, Hayley, Motson & Nitzsche (2016, IJFE "What Does Rebalancing Really Achieve?"): the
  "diversification return" is partly earned by unrebalanced books too; rebalanced books are not
  guaranteed more terminal wealth, and empirical support is weak. Bouchey et al. note that in
  trending markets (e.g., US tech 1998-99) rebalancing lost to buy-and-hold.
* SURVIVORSHIP: the universes are TODAY's constituents. Names that were volatile, lowly
  correlated small/mid caps in 2007 and are in the index now are, by construction, the winners.
  A high-σ / low-ρ tilt loads exactly on them, so discovery (and holdout) results of this family
  are biased UP relative to 1/N. Any edge found here must be read with that in mind.

MECHANISM AND WHY IT MIGHT BEAT BOTH 1/N AND CVG
────────────────────────────────────────────────
No means are forecast. The prior is that every name has the SAME expected GEOMETRIC growth g
(the "stable market" of stochastic portfolio theory: no name grows to dominate), i.e. arithmetic
μᵢ = g + ½σᵢ². Under that prior the expected log-growth of a constant-mix book is
        g + γ*(w) − ((λ−1)/2)·w'Σw      (λ = 1 is full Kelly; λ > 1 fractional Kelly)
so the growth-optimal fully-invested book maximises the excess growth rate, with λ trading it
against portfolio variance. (The other common prior — equal ARITHMETIC means — gives minimum
variance, which is what ERC/HRP approximate; it won only the Nifty 2007-13 cell.) 1/N earns
γ*(1/N); these books earn more γ* by construction, and the harness's monthly rebalance is what
realises it. CVG earns its edge by buying capitulation (a contrarian, signal-driven state);
volatility harvesting is the same contrarian rebalancing trade applied systematically, sized
where the relative moves (and hence the rebalancing payoff) are largest — no signal needed.

CANDIDATES (3 functions, 9 configurations; literature default marked *)
──────────────────────────────────────────────────────────────────────
C1 vol_power(ctx, k)            wᵢ ∝ σᵢᵏ, σ = 252-day sample st.dev. of daily returns, correlations
                                ignored (Kirby-Ostdiek form with the sign reversed: k = −2η).
                                k ∈ {0.5, 1*, 2}. k = 1 is the mirror image of inverse-vol weighting;
                                k = 2 (wᵢ ∝ σᵢ²) is the mirror of KO volatility timing with η = 1.
C2 kelly_egr(ctx, lam)          fractional-Kelly book under the equal-geometric-growth prior:
                                maximise ½Σwᵢσᵢ² − (λ/2) w'Σw, Σ = Ledoit-Wolf (nco.ledoit_wolf,
                                constant-correlation target) on 252 days, σᵢ² = diag Σ;
                                Σw = 1, 1/(2N) ≤ wᵢ ≤ 10%. λ ∈ {1, 2*, 4}: full, half*, quarter Kelly
                                (λ = 1 is the maximum-excess-growth-rate book; λ → ∞ min variance).
C3 egr_riskbudget(ctx, kappa)   Ding-Qi efficient-DR point at a risk budget tied to 1/N: maximise
                                γ*(w) subject to σₚ(w) ≤ κ·σₚ(1/N) (ex-ante, same Σ), same floor and
                                cap. κ ∈ {0.8, 0.9, 1.0*}: 1/N's risk*, ≈ERC's risk (−10%), ≈HRP's
                                risk (−20%) — the vol cuts measured for ERC/HRP. If the floor makes
                                the budget infeasible, the minimum-variance book within the bounds
                                is held; if the unconstrained max-γ* book is already within budget,
                                it is held.

Fixed design (declared, not tuned): daily simple returns (no padding of stale closes) over the
last 252 trading days ≤ ctx.date; the covariance set is the priced names with ≥ 80% coverage of
that window (nco.MIN_COVERAGE), rows with any gap dropped (as nco.build_returns_matrix); a priced
name outside that set (recent listing) is held at 1/N and the rest share the remaining mass; in C1
a name with < 20 observations takes the cross-sectional median σ. N = number of priced names.
The harness renormalises and applies the 10% cap (C2/C3 already satisfy it).

PRE-STATED EXPECTATION: C1/C2(λ=1) most at risk in Nifty E1 (low-risk anomaly, crash era);
C2(λ=4) and C3(κ=0.8) most at risk in Dow E1/E2 and Nifty E2 (they drift toward low volatility).

RESULT (discovery only, run 2026-10-03 after the declaration above; 9 configurations, none added)
──────────────────────────────────────────────────────────────────────────────────────────────────
Net CAGR % per cell and margin vs the best existing style in that cell (bar: Nifty E1 20.21 HRP,
Nifty E2 19.68 CVG, Dow E1 14.39 EW, Dow E2 18.50 CVG). vol / maxDD / turnover per yr per cell.

                        Nifty E1        Nifty E2        Dow E1          Dow E2        min    cells
config                  CAGR   marg     CAGR   marg     CAGR   marg     CAGR   marg   margin beaten
vol_power k=0.5         17.91  -2.29    19.36  -0.32    14.81  +0.42    19.39  +0.90  -2.29  2/4
vol_power k=1           17.73  -2.48    19.95  +0.27    15.22  +0.83    20.66  +2.16  -2.48  3/4
vol_power k=2           17.21  -3.00    21.36  +1.68    16.07  +1.68    22.27  +3.77  -3.00  3/4
kelly_egr lam=1         17.51  -2.69    28.35  +8.68    17.08  +2.70    24.10  +5.61  -2.69  3/4
kelly_egr lam=2         18.77  -1.44    26.12  +6.44    12.65  -1.74    23.09  +4.59  -1.74  2/4  ← finalist
kelly_egr lam=4         22.22  +2.01    20.32  +0.65    12.30  -2.09    18.32  -0.18  -2.09  2/4
egr_riskbudget k=0.8    22.36  +2.15    17.93  -1.75    12.40  -1.99    14.95  -3.55  -3.55  1/4
egr_riskbudget k=0.9    23.69  +3.48    19.77  +0.10    11.55  -2.84    17.90  -0.60  -2.84  2/4
egr_riskbudget k=1.0    22.08  +1.88    21.48  +1.81    12.12  -2.27    20.41  +1.91  -2.27  3/4

vol % (NE1/NE2/DE1/DE2), maxDD % (same), turnover/yr (same):
vol_power k=0.5       30.8/15.3/20.1/13.2   -60.3/-16.4/-41.4/-12.8   0.43/0.35/0.31/0.24
vol_power k=1         32.3/15.9/21.0/13.8   -62.5/-17.4/-43.0/-13.9   0.47/0.38/0.36/0.28
vol_power k=2         35.3/17.4/22.9/14.8   -66.6/-20.0/-45.6/-16.2   0.57/0.48/0.46/0.37
kelly_egr lam=1       37.4/19.3/23.9/15.3   -67.8/-17.8/-46.6/-16.2   1.03/0.83/0.63/0.61
kelly_egr lam=2       32.4/17.3/19.3/12.9   -64.1/-14.7/-46.0/-12.6   0.96/0.89/0.78/0.70
kelly_egr lam=4       24.5/13.8/15.9/11.0   -48.8/-12.2/-36.7/ -8.7   0.98/0.86/0.59/0.68
egr_riskbudget k=0.8  24.2/13.2/15.7/10.8   -45.5/-12.5/-34.2/ -9.3   1.09/0.82/0.52/0.57
egr_riskbudget k=0.9  27.0/13.4/17.0/11.0   -49.6/-12.1/-39.4/ -8.8   1.05/0.98/0.64/0.66
egr_riskbudget k=1.0  30.8/14.4/18.8/12.0   -59.6/-12.4/-42.9/-11.0   1.04/1.08/0.73/0.71

NO configuration beats the best existing style in all 4 discovery cells. The family splits exactly
along the known tension: the high-σ / max-γ* end (C1, C2 λ=1) wins Dow E1/E2 and Nifty E2 but
loses Nifty E1 (the low-risk crash era); the low-risk end (C2 λ=4, C3) wins Nifty E1 but loses
Dow E1 (and mostly Dow E2). No point on the γ*-vs-risk frontier clears both Nifty E1 and Dow E1.
Dow E1 is fragile in λ: +2.70 at λ=1, −1.74 at λ=2.

FINALIST (rule 6, none beat all → largest smallest-margin): kelly_egr(ctx, lam=2.0) — half-Kelly
under the equal-geometric-growth prior; smallest margin −1.74 (Dow E1), also −1.44 Nifty E1;
+6.44 Nifty E2, +4.59 Dow E2. It fails 2 of 4 discovery cells, so it is not a "beats all"
candidate. Runner-up (not nominated): kelly_egr lam=4, −2.09.

Attribution (finalist and λ=1, active contribution vs 1/N): the book is ~34/47 Nifty and ~18-20/30
Dow names pinned at the 1/(2N) floor plus a concentrated sleeve; the E2 edge comes from names that
JOINED today's index after large runs — ADANIENT, EICHERMOT, INDIGO, BEL, TITAN (Nifty E2); NVDA,
AMZN, CRM (Dow E2); BAJFINANCE, SHRIRAMFIN (Nifty E1). This is the survivorship bias stated up
front: high idiosyncratic volatility + low correlation in 2007-2019 ≈ "small then, in the index
now". The E2 margins should be read as mostly selection, not harvesting.

Solver health: SLSQP fallbacks 0/310 rebalances (C2 finalist), 4/465 Nifty and 1/465 Dow for C3 (min-var or
clipped solution held those months).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # research/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root

import nco                                                # noqa: E402

LOOKBACK, CAP, MIN_OBS_C1 = 252, 0.10, 20
STATS = {"fallback": 0, "solves": 0}                      # solver health, reported with the results


# ── shared estimation ────────────────────────────────────────────────────────────────────────────
def _rets(ctx) -> pd.DataFrame:
    px = ctx.px[ctx.priced]
    return px.pct_change(fill_method=None).iloc[1:].tail(LOOKBACK)


def _cov_set(ctx):
    """(names in the covariance set, Ledoit-Wolf Σ over them, names held at 1/N outside it)."""
    r = _rets(ctx)
    cover = r.notna().sum()
    keep = [c for c in r.columns if cover[c] >= nco.MIN_COVERAGE * len(r)]
    R = r[keep].dropna(how="any")
    S = nco.ledoit_wolf(R.to_numpy(dtype=float))
    out = [c for c in ctx.priced if c not in keep]
    return keep, S, out


def _assemble(ctx, keep, w_keep, out) -> pd.Series:
    N = len(ctx.priced)
    w = pd.Series(1.0 / N, index=ctx.priced, dtype=float)
    w.loc[keep] = w_keep
    return w


def _solve(S, s2, lam, lo, hi, total, var_cap=None, x0=None):
    """max ½ s2·w − (λ/2) w'Sw  s.t. Σw = total, lo ≤ w ≤ hi, optionally w'Sw ≤ var_cap (SLSQP)."""
    n = len(s2)
    x0 = np.full(n, total / n) if x0 is None else x0
    cons = [{"type": "eq", "fun": lambda w: w.sum() - total, "jac": lambda w: np.ones(n)}]
    if var_cap is not None:
        cons.append({"type": "ineq", "fun": lambda w: var_cap - w @ S @ w, "jac": lambda w: -2.0 * S @ w})
    res = minimize(lambda w: -(0.5 * s2 @ w - 0.5 * lam * w @ S @ w), x0,
                   jac=lambda w: -(0.5 * s2 - lam * S @ w), bounds=[(lo, hi)] * n,
                   constraints=cons, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
    STATS["solves"] += 1
    w = np.clip(res.x, lo, hi)
    ok = res.success and abs(w.sum() - total) < 1e-6 and (var_cap is None or w @ S @ w <= var_cap * (1 + 1e-6))
    return w, ok


def _min_var(S, lo, hi, total):
    n = S.shape[0]
    res = minimize(lambda w: w @ S @ w, np.full(n, total / n), jac=lambda w: 2.0 * S @ w,
                   bounds=[(lo, hi)] * n, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14},
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - total, "jac": lambda w: np.ones(n)}])
    STATS["solves"] += 1
    return np.clip(res.x, lo, hi), res.success


def _setup(ctx):
    keep, S, out = _cov_set(ctx)
    S = S / np.mean(np.diag(S))                           # scale-free: the argmax is invariant to it
    N = len(ctx.priced)
    lo, total = 0.5 / N, len(keep) / N                    # floor 1/(2N); outside names hold 1/N each
    return keep, S, np.diag(S).copy(), out, lo, total


# ── C1 ───────────────────────────────────────────────────────────────────────────────────────────
def vol_power(ctx, k: float = 1.0) -> pd.Series:
    r = _rets(ctx)
    sd = r.std(ddof=1).where(r.notna().sum() >= MIN_OBS_C1)
    sd = sd.reindex(ctx.priced)
    sd = sd.fillna(sd.median())
    return sd ** k


# ── C2 ───────────────────────────────────────────────────────────────────────────────────────────
def kelly_egr(ctx, lam: float = 2.0) -> pd.Series:
    keep, S, s2, out, lo, total = _setup(ctx)
    w, ok = _solve(S, s2, lam, lo, CAP, total)
    if not ok:
        STATS["fallback"] += 1
    return _assemble(ctx, keep, w, out)


# ── C3 ───────────────────────────────────────────────────────────────────────────────────────────
def egr_riskbudget(ctx, kappa: float = 1.0) -> pd.Series:
    keep, S, s2, out, lo, total = _setup(ctx)
    n = len(keep)
    we = np.full(n, total / n)
    budget = kappa ** 2 * float(we @ S @ we)
    w1, ok1 = _solve(S, s2, 1.0, lo, CAP, total)          # unconstrained max-γ* within the bounds
    if ok1 and w1 @ S @ w1 <= budget:
        return _assemble(ctx, keep, w1, out)
    wmv, okmv = _min_var(S, lo, CAP, total)
    if wmv @ S @ wmv >= budget:                           # budget infeasible under the floor
        if not okmv:
            STATS["fallback"] += 1
        return _assemble(ctx, keep, wmv, out)
    w, ok = _solve(S, s2, 1.0, lo, CAP, total, var_cap=budget, x0=wmv)
    if not ok:
        STATS["fallback"] += 1
        w = wmv
    return _assemble(ctx, keep, w, out)


CANDIDATES = {
    "vol_power": (vol_power, {"k": [0.5, 1.0, 2.0]}),
    "kelly_egr": (kelly_egr, {"lam": [1.0, 2.0, 4.0]}),
    "egr_riskbudget": (egr_riskbudget, {"kappa": [0.8, 0.9, 1.0]}),
}
