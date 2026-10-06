"""
PRAGYAM — portfolio curation: Equal Weight · ERC · HRP · Conviction-Value Grid · Managed Momentum
══════════════════════════════════════════════════════════════════════════════

The system's only curation stack. Five selectable styles (METHOD_ORDER), and every
one travels the same pipeline — eligibility, weights over the allocation universe,
top-N by weight, the per-position cap, integer units, the same risk diagnostics —
so any difference on screen is the weight formula and nothing else:

    EQUAL   1/N. Reads nothing; the default and the bar.
    ERC     equal risk contribution on a shrunk (Ledoit-Wolf) covariance.
    HRP     hierarchical risk parity: recursive bisection on cluster variance.
    CVG     the Conviction-Value Grid: each name sized by its state in the 3 × 3 of
            the Pragati indicator's conviction and value tapes (cvgrid.py).
    MMOM    Managed Momentum: the grid's weights plus a crash-managed 12-1 momentum
            rank (mmom_overlay), read from a long close history (`price_history`,
            backdata.fetch_close_history); without one it stands down to the grid.

Equal Weight, ERC and HRP make no return forecast of any kind; ERC and HRP select
and weight from the return covariance structure, Equal Weight from nothing. CVG
and MMOM read the tape instead — the grid its two tapes, MMOM also a 12-1 momentum
rank — and read no covariance. They are bets on what the tape says, so the
forecasting ceiling below applies to them, and each is measured against Equal
Weight and the other styles (METHOD_SPECS[...]["evidence"]; README).

Why covariance rather than forecasts (the risk styles)
──────────────────────────────────────────────────────
Grinold's Fundamental Law bounds excess return from FORECASTING skill at
IR = IC x sqrt(BR) x TC. Measured on the ETF universe that ceiling is ~1%/yr:
average pairwise correlation 0.517 leaves only ~1.9 effective independent bets,
so no amount of signal engineering buys much.

That bound applies to alpha from prediction. It does not apply to the risk styles,
because they predict nothing. They exploit the covariance structure, which is
estimable from a few hundred observations in a way expected returns never are
(López de Prado, "Building Diversified Portfolios that Outperform Out of Sample",
2016; "A Robust Estimator of the Efficient Frontier", 2019). That is why they can
reduce RISK reliably where forecast-driven approaches cannot — but see the
measured results below: they do not deliver excess return either.

Measured on the shipped module across two GENUINELY DISJOINT periods
(2023-12..2024-12 and 2025-01..2026-07, zero overlap):

    HRP vs equal weight:  return -0.96% / -1.23%      (loses in BOTH)
                          volatility 13.81->11.28%, 12.97->10.38%
                          max drawdown -6.89->-4.50%, -7.07->-6.15%
                          Sharpe 1.83->2.14, 0.96->1.07

This is a VOLATILITY-REDUCTION overlay, not an alpha source. It costs about
1%/yr of return and buys roughly a 20% cut in volatility and drawdown; Sharpe
improves because the risk saving outweighs the return cost. Do not size it
expecting excess return.

An earlier nested-window test (2024+ was fully contained in 2023+) reported a
small POSITIVE excess return. That did not survive a disjoint split — a caution
about the window design, not about the method.

HRP beat full Nested Clustered Optimization (NCO) on return, Sharpe and drawdown
in both disjoint windows, and inverts no matrix; NCO is not carried.

Corroboration that the correlation structure carries information beyond
variance alone: plain inverse-VOLATILITY weighting, which ignores correlation
entirely, also lost to equal weight when tested. (That test used the older
nested windows, so treat it as directional only, not as a like-for-like
comparison with the disjoint figures above.)

Why hierarchical
────────────────
Markowitz inverts the covariance matrix, and its condition number explodes when
assets are correlated — small estimation errors become wild weights. HRP inverts
nothing: it orders the matrix by cluster, then splits capital by recursive
bisection using only cluster variances. Ward clustering finds K ~= 3 here at
silhouette ~0.24, independently matching the eigenvalue participation ratio of
2.95 — three real risk clusters inside 30 tickers.

Equal Risk Contribution (ERC) and NCO were both implemented and measured
alongside HRP. In those two windows ERC achieved perfect risk balance (1.00x
against HRP's 1.5-1.7x) and matched HRP on return and Sharpe to within noise;
NCO trailed HRP on return, Sharpe and drawdown in both. ERC has since shipped as
the preferred risk-reduction style on return — it beats HRP on the any-date hit
rate in 6 of 6 cells across two stock universes, at about 0.6x its turnover on
the in-repo harness (METHOD_SPECS["ERC"]); HRP is the deeper volatility and
drawdown cut — and neither beats Equal Weight reproducibly on return.
See CHANGELOG for the figures.

Author: @thebullishvalue
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cvgrid import STATE_ORDER, STATE_UNITS, graded_units

# Minimum return observations before a covariance estimate is trusted: MIN_OBS
# outright, and at least one observation per asset (T >= n). One per asset is a
# floor, not a comfort level: HRP inverts nothing and ERC solves on the
# Ledoit-Wolf-shrunk matrix, so both can run at T/n near 1, but a book estimated
# there is mostly sample noise. compute_nco_portfolio records T/n in
# `nco_obs_per_asset` so the run log can say so.
MIN_OBS = 60
MIN_OBS_PER_ASSET = 1.0

# Cluster-count search range for the silhouette selection.
MAX_CLUSTERS = 8

# Fraction of the lookback a symbol must have data for to be eligible. A symbol
# present for only part of the window would otherwise force a choice between
# dropping every date it is missing (which collapses the sample) or imputing
# returns it never had. Every admitted name's gaps are then dropped for ALL names
# (dropna(how="any")), so this is also the most one late listing can cost the
# whole estimation window: at 0.8 a name listed 50 sessions ago cut every name's
# sample by 20% and left NIFTY SMLCAP 250 with T < n (no HRP or ERC book).
MIN_COVERAGE = 0.95

# A close repeating the one before it in a run of at least this many sessions is
# a dead quote: the same rule as backdata.mask_dead_quotes, copied here so that
# nco does not import yfinance.
_DEAD_QUOTE_RUN = 10

# A return column is degenerate, and left out of the estimation universe, when its variance
# is non-finite or zero, when at least half its returns are exactly zero (a frozen series the
# dead-quote rule did not catch: J&KBANK 2016-17, 99.6%), or when its variance is under 1% of
# the universe's median. The last also leaves out a genuinely near-riskless asset — a cash-like
# ETF among equities (~1e-4x), possibly a pegged currency among FX pairs — on purpose: an
# inverse-variance allocator gives it ~99.8% of raw weight, the cap holds it at 10%, and the
# rest of the book comes out flat (HRP becomes equal weight). Clean research universes sit far
# above the line (lowest ratio 0.18 over 494 month-starts); the run log names every name left out.
_DEGENERATE_ZERO_SHARE = 0.5
_DEGENERATE_VAR_RATIO = 0.01


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """López de Prado's correlation distance: d_ij = sqrt(0.5 * (1 - rho_ij)).

    A proper metric on the correlation matrix — perfectly correlated assets sit
    at distance 0, uncorrelated at 0.707, perfectly anti-correlated at 1 — which
    is what makes hierarchical clustering meaningful here.
    """
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    # Enforce exact symmetry; float error in corrcoef can otherwise trip
    # scipy's squareform validity check.
    return (d + d.T) / 2.0


def inverse_variance(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance weights — the diagonal (correlation-blind) allocator."""
    v = np.diag(cov).astype(float).copy()
    good = v > 1e-16
    if not good.any():
        return np.full(len(v), 1.0 / max(len(v), 1))
    v[~good] = float(np.median(v[good]))
    iv = 1.0 / v
    return iv / iv.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each holding's share of portfolio VARIANCE, normalised to sum to 1."""
    pv = float(w @ cov @ w)
    n = len(w)
    if pv <= 1e-18:
        return np.full(n, 1.0 / n)
    rc = w * (cov @ w)
    tot = rc.sum()
    return rc / tot if abs(tot) > 1e-18 else np.full(n, 1.0 / n)


def cluster_assets(corr: np.ndarray, max_clusters: int = MAX_CLUSTERS
                   ) -> Tuple[np.ndarray, int, float]:
    """Ward-cluster the correlation distance; pick K by silhouette score.

    Returns (labels, k, silhouette). Degrades to a single cluster (which makes
    NCO collapse to a flat optimization) rather than raising, so a pathological
    correlation matrix cannot break a run.
    """
    n = corr.shape[0]
    if n < 3:
        return np.ones(n, dtype=int), 1, 0.0
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics import silhouette_score
    except Exception:
        return np.ones(n, dtype=int), 1, 0.0

    d = correlation_distance(corr)
    try:
        Z = linkage(squareform(d, checks=False), method="ward")
    except Exception:
        return np.ones(n, dtype=int), 1, 0.0

    best_k, best_s, best_lab = 1, -2.0, np.ones(n, dtype=int)
    for k in range(2, min(max_clusters, n - 1) + 1):
        lab = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        try:
            s = float(silhouette_score(d, lab, metric="precomputed"))
        except Exception:
            continue
        if s > best_s:
            best_k, best_s, best_lab = k, s, lab
    return best_lab, best_k, (best_s if best_s > -2.0 else 0.0)


def ledoit_wolf(R: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a constant-correlation target.

    A sample covariance over ~30 assets from ~250 observations is badly
    conditioned, and every allocator that inverts it amplifies that error into
    wild weights. Shrinkage is the standard fix and is applied to the allocators
    below that need a well-conditioned matrix, so none is handicapped by
    estimator noise it did not have to carry.
    """
    T, N = R.shape
    # Ledoit & Wolf (2004), constant-correlation target, as their covCor.m: the
    # 1/T sample matrix, and shrinkage (phi - rho) / (T gamma). The rho term was
    # missing, which over-shrank (median intensity 0.43 vs 0.31 on Nifty 50, and
    # fully shrunk in 12 Nifty / 39 Dow month-starts).
    X = R - R.mean(axis=0) if T > 0 else R
    S = (X.T @ X) / T if T > 0 else np.zeros((N, N))
    var = np.diag(S)
    sd = np.sqrt(np.clip(var, 1e-20, None))
    C = S / np.outer(sd, sd)
    rbar = (C.sum() - N) / (N * (N - 1)) if N > 1 else 0.0
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)
    # φ = (1/T) Σ_t ‖x_t x_tᵀ − S‖², expanded so it needs two matrix products
    # rather than T outer products:
    #     Σ_t Σ_ij (x_ti x_tj − s_ij)²
    #       = Σ_ij [(X∘X)ᵀ(X∘X)]_ij − 2 Σ_ij s_ij [XᵀX]_ij + T Σ_ij s_ij²
    # It was a Python sum() over a generator of outer products — typed as int,
    # since sum() starts from 0, and genuinely an int that crashed on `.sum()`
    # when T = 0. Identical to the loop to 2e-16 relative.
    X2 = X * X
    phi = (float((X2.T @ X2).sum() - 2.0 * (S * (X.T @ X)).sum() + T * (S * S).sum()) / T
           if T > 0 else 0.0)
    gamma = ((F - S) ** 2).sum()
    # ρ = Σ_i π_ii + r̄ Σ_{i≠j} (σ_j/σ_i) ϑ_ij,  ϑ_ij = (1/T) Σ_t (x_ti² − s_ii)(x_ti x_tj − s_ij)
    if T > 0 and N > 1:
        pi_diag = (X2 * X2).sum(axis=0) / T - 2.0 * var * var + var * var
        theta = ((X ** 3).T @ X) / T - var[:, None] * S
        np.fill_diagonal(theta, 0.0)
        rho = float(pi_diag.sum() + rbar * (np.outer(1.0 / sd, sd) * theta).sum())
    else:
        rho = 0.0
    shrink = float(np.clip((phi - rho) / (T * gamma), 0.0, 1.0)) if gamma > 1e-20 else 0.0
    return shrink * F + (1.0 - shrink) * S


def erc_weights(cov: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution — every holding contributes the SAME share of
    portfolio variance.

    Solved by cyclical coordinate descent (Spinu 2013), which converges without
    inverting the covariance matrix. That is why it stays stable where minimum
    variance produces corner solutions.

    MEASURED: ERC beats HRP on the any-date hit rate in 6 of 6 cells across Nifty
    50 and Dow 30 (the 36-candidate search). On the in-repo harness (research/
    style_search.py, every name held, net, 2007-26) it trails equal weight by
    0.49 %/yr (Nifty 50) and 1.27 (Dow 30) at lower volatility, trading 0.42x/yr
    against HRP's 0.72x. (The SIP-stream wins once quoted here came from a solver
    defect; see the README's v11.1 correction.)
    """
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    d = np.clip(np.diag(cov), 1e-20, None)
    x = 1.0 / np.sqrt(d)                       # inverse-vol seed

    # The descent runs on an UNNORMALISED vector and is normalised exactly once,
    # at the end. Rescaling x inside the loop breaks the fixed point: the target
    # risk contribution 1/n is defined relative to x's own scale, so dividing by
    # the sum each sweep moves the target the iteration is chasing and the
    # solver stalls at whatever it happened to reach. Measured before this fix:
    # risk-contribution dispersion 0.55 against a target of 0.00 — i.e. it was
    # not producing equal risk contributions at all, only inverse-vol-ish ones.
    for _ in range(1000):
        x_prev = x.copy()
        for i in range(n):
            # Solve a*x_i^2 + b*x_i - 1/n = 0 holding the rest fixed; the
            # positive root is the risk-balancing weight for asset i.
            a = float(cov[i, i])
            b = float(cov[i] @ x) - x[i] * a
            x[i] = ((-b + np.sqrt(max(b * b + 4.0 * a / n, 0.0))) / (2.0 * a)
                    if a > 1e-20 else 0.0)
        x = np.clip(x, 0.0, None)
        if np.abs(x - x_prev).max() <= 1e-12 * max(1.0, float(np.abs(x).max())):
            break
    s = x.sum()
    return x / s if s > 1e-12 else np.full(n, 1.0 / n)


def momentum_scores(prices: pd.DataFrame, lookback: int = 252,
                    skip: int = 21) -> pd.Series:
    """Cross-sectional 12-1 momentum: total return over `lookback` bars ending
    `skip` bars ago.

    The skip is the standard short-term-reversal guard (Jegadeesh & Titman):
    the most recent month reverses, so including it contaminates the signal.

    Returns NaN for names without enough history; callers rank what they have.
    """
    if prices is None or prices.empty or len(prices) < lookback + skip + 1:
        return pd.Series(np.nan, index=prices.columns if prices is not None else [])
    end = prices.iloc[-1 - skip]
    start = prices.iloc[-1 - skip - lookback]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = end / start - 1.0
    return m.replace([np.inf, -np.inf], np.nan)


def rank_z(scores: pd.Series) -> pd.Series:
    """Cross-sectional rank score, centred at 0 with unit spread.

    Rank rather than raw z: momentum is heavy-tailed, and one runaway holding
    would otherwise dominate the tilt. Names with no score sit at the median.
    """
    s = pd.to_numeric(scores, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    r = s.rank(method="average", na_option="keep")
    u = (r - 0.5) / s.notna().sum()
    z = (u - 0.5) * np.sqrt(12.0)
    return z.fillna(0.0)


def apply_momentum_tilt(base: np.ndarray, scores: pd.Series,
                        lam: float = 0.5) -> np.ndarray:
    """Tilt a risk-based base by cross-sectional momentum, MULTIPLICATIVELY.

        w_i  proportional to  base_i * max(0, 1 + lam * z_i)

    Multiplicative, never additive: an additive tilt lets a high momentum score
    override the risk model entirely, which destroys the risk balance the base
    exists to provide. Multiplying preserves the base's risk ordering and only
    re-weights within it. At lam = 0 this returns the base exactly.

    MEASURED at lam = 0.5 on ERC: beat equal weight in 98.2% of 60-month SIP
    streams started 2012-2016 and 100.0% of those started 2017-2021 (Nifty 50,
    115 start months, two disjoint halves). Honest caveat: the tilt's INCREMENTAL
    t-statistic over plain ERC is below 1 on every universe tested, and it is
    negative on Dow 30. It wins often and by little — which is what a SIP needs
    — but it is not a large or independently proven effect.
    """
    z = rank_z(scores).to_numpy(dtype=float)
    w = np.asarray(base, dtype=float) * np.clip(1.0 + lam * z, 0.0, None)
    s = w.sum()
    return w / s if s > 1e-12 else np.asarray(base, dtype=float)


def hrp_weights(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado 2016), as commonly implemented.

    Single linkage on the condensed correlation distance d = sqrt(0.5(1 - rho))
    itself (the PyPortfolioOpt convention), not the paper's d~ (the distance
    between columns of D): d~ was measured and is not better (Nifty -0.24%/yr
    t -0.83, Dow +0.19 t 1.10, ETF -1.75 t -1.56; research/audit_hrp.py). The
    leaf order is split in halves recursively, and capital goes between the
    halves in inverse proportion to their cluster variance. Inverts nothing.
    Measured performance: METHOD_SPECS["HRP"]["evidence"].
    """
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    # Non-finite input used to change the method without a trace: a NaN variance
    # turned every split on that name's path into 50/50, a NaN correlation made
    # linkage raise and the bare except return plain inverse variance.
    # build_returns_matrix now drops such columns; reaching here with one is a bug.
    if not (np.isfinite(cov).all() and np.isfinite(corr).all()):
        raise ValueError("hrp_weights: non-finite covariance or correlation")
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    d = correlation_distance(corr)
    Z = linkage(squareform(d, checks=False), method="single").astype(int)

    # Quasi-diagonal ordering: unwind the linkage tree into leaf order.
    srt = pd.Series([Z[-1, 0], Z[-1, 1]])
    num = Z[-1, 3]
    while srt.max() >= num:
        srt.index = range(0, srt.shape[0] * 2, 2)
        df0 = srt[srt >= num]
        i, j = df0.index, df0.values - num
        srt[i] = Z[j, 0]
        srt = pd.concat([srt, pd.Series(Z[j, 1], index=i + 1)]).sort_index()
    order = [int(x) for x in srt.tolist()]

    def cluster_var(idx: List[int]) -> float:
        sub = cov[np.ix_(idx, idx)]
        w = inverse_variance(sub)
        return float(w @ sub @ w)

    w = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[a:b] for c in clusters
                    for a, b in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters) - 1, 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = cluster_var(c0), cluster_var(c1)
            alpha = 1.0 - v0 / (v0 + v1) if (v0 + v1) > 1e-18 else 0.5
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    total = w.sum()
    return w / total if total > 1e-12 else np.full(n, 1.0 / n)


# ── Method registry ───────────────────────────────────────────────────────────
#
# One record per shipped weighting method. Everything the UI needs to describe,
# label, chart and log a method lives HERE — the app branches on registry fields
# rather than on `method == "EQUAL"`, so adding a style never again means
# hunting down scattered if/else.
#
# `family`      accumulation | balanced | preservation | baseline
# `uses_clusters`  whether the cluster diagnostic explains this method's weights
# `uses_cvg` whether the weights read Pragati's two tapes (the grid-state columns)
# `rc_target`   the risk-contribution pattern the method AIMS for, which is what
#               the risk charts must be scored against. "equal" means the method
#               targets identical risk shares; "cluster" balances between the
#               halves of a correlation-ordered list (HRP's bisection); "none"
#               means it does not manage risk contribution at
#               all.
# `needs_covariance`  whether the WEIGHTS are computed from the covariance. This
#               is an eligibility rule, not a description: a style that reads the
#               covariance can only hold names that have one, so it is confined
#               to the estimation universe (>= MIN_COVERAGE of the window). Equal
#               weight reads nothing, so it allocates over every priced symbol —
#               excluding a recently listed name from a 1/N book would be a rule
#               with no statistic behind it. See compute_nco_portfolio.
# `evidence`    one measured sentence, shown in the UI. No claim without a number.

METHOD_SPECS = {
    "EQUAL": {
        "label": "Equal Weight",
        "short": "EQUAL",
        "family": "baseline",
        "formula": "1 / N",
        "tagline": "Identical share per holding — the default, and the bar",
        "uses_clusters": False,
        "uses_momentum": False,
        "uses_cvg": False,
        "rc_target": "none",
        "needs_covariance": False,
        "evidence": ("The default because nothing beat it reproducibly. Across 36 candidate allocators "
                     "on three universes, no method delivered a reproducible return "
                     "improvement: ERC gave up 0.51%/yr on Nifty 50 and 1.48% on Dow 30. "
                     "Managed Momentum (v12.1) led it in every era tested, not significantly "
                     "and partly on late index entrants. Lowest turnover of any style."),
        # The evidence above in one line, against Equal Weight — the form the
        # Analytics tab's Style Comparison quotes under a one-window table.
        "long_run": "the bar: no allocator tested on three universes beat it reproducibly on return",
        "sip_default": True,
    },
    "ERC": {
        "label": "Equal Risk Contribution",
        "short": "ERC",
        "family": "preservation",
        "formula": "w_i * (Cov w)_i identical for every holding",
        "tagline": "Every holding contributes the same share of variance",
        "uses_clusters": False,
        "uses_momentum": False,
        "uses_cvg": False,
        "rc_target": "equal",
        "needs_covariance": True,
        "evidence": ("The preferred risk-reduction style on return: it beats HRP on the "
                     "any-date hit rate in 6 of 6 cells across both stock universes (the "
                     "36-candidate search, not in this repository). On the in-repo harness "
                     "(research/style_search.py: monthly, every name held, net of costs, "
                     "2007-26) it does NOT beat equal weight on return (-0.49%/yr Nifty 50, "
                     "-1.27% Dow 30), at lower volatility (20.1 vs 22.2, 15.3 vs 16.5) and "
                     "about 0.6x HRP's turnover (0.42x/yr vs 0.72x on Nifty 50). "
                     "Shrinkage: Ledoit-Wolf (2004) constant correlation, with its rho term "
                     "since v12.2."),
        "long_run": ("-0.49%/yr on Nifty 50 and -1.27% on Dow 30 (2007-26, every name held), "
                     "at volatility 20.1 / 15.3 against Equal Weight's 22.2 / 16.5"),
        "sip_default": False,
    },
    "HRP": {
        "label": "Risk Parity (HRP)",
        "short": "HRP",
        "family": "preservation",
        "formula": "recursive bisection on cluster variance",
        "tagline": "Clusters by correlation, splits capital by cluster variance",
        # False: HRP bisects its own single-linkage leaf order, which the Ward panel is
        # not (its first split cuts a Ward cluster every month), so the panel is a
        # diagnostic for HRP too.
        "uses_clusters": False,
        "uses_momentum": False,
        "uses_cvg": False,
        "rc_target": "cluster",
        "needs_covariance": True,
        "evidence": ("The deepest volatility and drawdown cut of the styles on the in-repo "
                     "harness (research/style_search.py: monthly, every name held, net of "
                     "costs, 2007-26, v12.2's staggered windows): volatility 18.8 against "
                     "Equal Weight's 22.2 and ERC's 20.1 on Nifty 50, 14.2 against 16.5 / 15.3 "
                     "on Dow 30; max drawdown -49.3% vs -56.5% and -36.4% vs -39.9%. It pays "
                     "on return (-0.54%/yr Nifty 50, -2.47% Dow 30 against Equal Weight) at "
                     "about 1.7x ERC's turnover (0.72 vs 0.42x/yr; the staggered windows cut "
                     "it 40%), and won 0 of 115 five-year SIP streams. Measured holding every "
                     "name: at the default 30 positions on Nifty 50 the book is the 30 "
                     "lowest-variance names (19.11%/yr against 19.56, turnover 0.96x)."),
        "long_run": ("-0.54%/yr on Nifty 50 and -2.47% on Dow 30 (2007-26, every name held), "
                     "the deepest volatility and drawdown cut: volatility 18.8 / 14.2, "
                     "max drawdown -49.3% / -36.4%"),
        "sip_default": False,
    },
    # ── Conviction-Value Grid · the 3 × 3 state book ─────────────────────────
    # pragati.pine read through both of its tapes (both on D · W) —
    # conviction (who controls: the rows) and value (rich or cheap against the
    # macro drivers, Samanvaya's engine: the columns) — each name placed in one
    # of nine states and sized by its state. The pane's histogram runs the rows:
    # a name changes row only when the push is behind the change (cvgrid.py).
    # Reads NO covariance: like Equal Weight it allocates over every priced
    # symbol, and the risk diagnostics are a mirror rather than a target.
    "CVG": {
        "label": "Conviction-Value Grid",
        "short": "CVG",
        "family": "accumulation",
        "formula": ("graded 3 × 3 — cell units up: turned 3 · building 1.5 · paid 0.75 · "
                    "faint: basing 1.5 · idle 1 · stalling 0.75 · down: dislocated 4 · "
                    "fading 1.5 · distribution 0.25; shaded within each cell by the tapes' "
                    "intensity; rows move only on a confirmed push"),
        "tagline": "Nine states from conviction × value; the histogram moves the rows",
        "uses_clusters": False,
        "uses_momentum": False,
        "uses_cvg": True,
        "rc_target": "none",
        "needs_covariance": False,
        "evidence": ("Units re-measured in v8 (research/cvg_reweight.py; monthly, every name "
                     "held, net of 10bp India / 3bp US costs; chosen before 2018, confirmed "
                     "after). Against the seed units it adds +0.42%/yr (t 0.6) then +0.98% "
                     "(t 1.5) on Nifty 50 and +0.55% (t 1.0) then +0.89% (t 1.0) on Dow 30, "
                     "at lower turnover. Against Equal Weight: Nifty +0.83%/yr (t 2.1) then "
                     "+0.47% (t 1.3); Dow -0.23% then +0.90% (t 2.4). The ETF book (1-27 "
                     "funds from 2012) is too thin to test. v12: Dislocated 3 → 4 beat 3 in all "
                     "three eras (Nifty +0.29 / +0.13 / +0.10 %/yr, Dow +0.04 / +0.02 / +0.07); "
                     "since v12.2 the conviction tape reads D · W, the tape these figures were "
                     "measured on (v12.0-12.1 read Ladder down, whose scale shrank as intraday "
                     "rungs were added). "
                     "Several designs were tried on these panels, so read every t as directional."),
        "long_run": ("vs Equal Weight, before / after 2018: Nifty 50 +0.83% / +0.47%/yr, "
                     "Dow 30 -0.23% / +0.90%/yr — at 1.2-1.3x monthly turnover"),
        "sip_default": False,
    },
    # ── Managed Momentum · the grid plus a crash-managed 12-1 overlay (v12.1) ─
    # CVG's weights, plus λ · rank(12-1 momentum) / N. The overlay stands down
    # (strength 0) while the equal-weighted market's 24-month return is
    # negative, shrinks while its own volatility runs above its long-run median
    # and grows (up to MMOM_SCALE_CAP) while it runs below. Every name keeps at
    # least MMOM_FLOOR of its CVG weight. Reads no
    # covariance, like the grid it is built on; reads a long close history for
    # the overlay (backdata.fetch_close_history), and stands down to the grid
    # when that fetch fails (the estimation panel cannot read the gate).
    "MMOM": {
        "label": "Managed Momentum",
        "short": "MMOM",
        "family": "accumulation",
        "formula": ("CVG weights + λ · rank(12-1 momentum) / N, λ = 1 · off while the market's "
                    "24-month return is negative · scaled by min(1.5, median / current) overlay "
                    "volatility · no name below ¼ of its CVG weight"),
        "tagline": "The grid plus a 12-1 momentum overlay that stands down in bear markets",
        "uses_clusters": False,
        "uses_momentum": True,
        "uses_cvg": True,
        "rc_target": "none",
        "needs_covariance": False,
        "evidence": ("Found by the v12.1 style search (research/style_search*.py: five families, "
                     "43 configurations, chosen on 2007-19, run once on 2020+) and re-measured "
                     "as shipped in v12.2 (research/mmom_ship.py --app-history: the app's close "
                     "history from 2006, monthly, every name held, net of 10bp India / 3bp US "
                     "costs, yfinance's unadjusted demergers repaired, the two-sided volatility "
                     "scale). Against the best of the eight earlier styles and blends in each era "
                     "(2007-13 / 2014-19 / 2020+): Nifty 50 +0.33 / +1.72 / +1.59 %/yr, Dow 30 "
                     "+0.77 / +0.57 / +0.29; +2.01 %/yr over Equal Weight on the 27-fund ETF book "
                     "(19 months). Full span: Nifty 22.36% vs CVG 20.99%, Dow 16.42% vs 15.80%, "
                     "at CVG's volatility and 1.3x its turnover. None of it is significant: the "
                     "largest per-era t over the best earlier style is 1.07; over the full span "
                     "Nifty leads CVG by +1.37 %/yr (t 1.17) and Equal Weight by +2.27 (t 2.11), "
                     "nominal. The Nifty 2007-13 lead (+0.33 over the staggered HRP) is thin. The "
                     "shipped form is a post-holdout variant (λ, floor, month-to-date volatility, "
                     "the two-sided scale) of one of 43 tries, so none of it survives a "
                     "family-wise correction, nor the survivorship of today's constituents: the "
                     "2020+ edge sits in a few names, led by late index entrants (BSE, TRENT, BEL, "
                     "ADANIENT; NVDA, AMZN, CRM). On a point-in-time Dow, its overlay reading only "
                     "that day's members, it trails CVG by 0.38 %/yr (t -0.45); no point-in-time "
                     "Nifty was tested. Its crash gate reads today's constituents, a laxer market "
                     "than an index (an index gate measured worse), at the month's first session. "
                     "In a book cut below the universe the names held are mostly the 12-1 "
                     "leaders: ahead of the grid on Nifty 50 at 30 positions, 1.8-4.3 %/yr behind "
                     "it on the Dow 2020+ at 25-10 positions and 1.9-5.0 behind on a "
                     "point-in-time Dow (measured on v12.1). In a full book expect CVG-like "
                     "results, not a reliable premium."),
        "long_run": ("vs the best of the eight earlier styles and blends, 2007-13 / 2014-19 / 2020+: "
                     "Nifty 50 +0.33% / +1.72% / +1.59%/yr, Dow 30 +0.77% / +0.57% / +0.29%/yr — "
                     "none significant (largest per-era t 1.07; full-span Nifty vs CVG t 1.17), "
                     "every-name books; -0.38%/yr vs CVG on a point-in-time Dow"),
        "sip_default": False,
    },
    # ── Implemented, deliberately NOT surfaced in the UI ─────────────────────
    # ERC + momentum was carried as the lead ship candidate until a defect was
    # found in the ERC solver (it renormalised inside the descent loop, so it
    # was solving for something between inverse-volatility and equal risk).
    # Re-measured against a CORRECT ERC the result inverted: the tilt went from
    # beating equal weight in 98-100% of 60-month SIP streams to 0 of 115, and
    # its Nifty lump-sum excess fell from +1.79%/yr to +0.19%/yr. It stays here
    # because the code is validated and the research harness uses it, but it is
    # excluded from METHOD_ORDER and cannot be selected.
    "ERC_MOM": {
        "label": "ERC + Momentum",
        "short": "ERC+MOM",
        "family": "experimental",
        "formula": "equal risk contribution x (1 + 0.5 z_momentum)",
        "tagline": "Research only — did not survive the ERC solver fix",
        "uses_clusters": False,
        "uses_momentum": True,
        "uses_cvg": False,
        "rc_target": "equal",
        "needs_covariance": True,
        "evidence": ("NOT SHIPPED. Against a corrected ERC base it beat equal weight in 0 "
                     "of 115 five-year SIP streams and by +0.19%/yr on Nifty lump-sum "
                     "(alpha t = 1.68). The earlier 98-100% SIP hit rate was an artifact "
                     "of a defect in the ERC solver."),
        "long_run": "+0.19%/yr on Nifty 50 lump-sum (t 1.68), 0 of 115 SIP streams won",
        "sip_default": False,
    },
}

# Selectable styles, in display order: the default first, then the
# risk-reduction family ordered by how well it does its job per unit of trading,
# then the styles that read the tape: the grid, and the grid with its momentum
# overlay. ERC_MOM is implemented but intentionally absent — see its spec above.
METHOD_ORDER = ("EQUAL", "ERC", "HRP", "CVG", "MMOM")
METHODS = METHOD_ORDER

# Momentum tilt strength. 0.5 is the measured setting; the parameter surface is
# a plateau over roughly 0.3-1.0, and 1.0 raised turnover materially for a
# smaller and less stable gain.
MOMENTUM_LAMBDA = 0.5
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21

# ── Managed Momentum · the grid plus a crash-managed 12-1 momentum overlay ────
# Found by the v12.1 style search (research/style_search*.py) and re-measured as
# shipped in research/mmom_ship.py. The overlay is the literature's, at its own
# defaults: 12-1 momentum (Jegadeesh & Titman 1993) added on top of CVG's
# weights — value and momentum are negatively correlated everywhere (Asness,
# Moskowitz & Pedersen 2013) — switched OFF while the equal-weighted market's
# 24-month return is negative, the state in which momentum crashes (Daniel &
# Moskowitz 2016), and scaled DOWN while the overlay's own six-month volatility
# runs above its long-run median, and UP, to 1.5x, while it runs below (Barroso & Santa-Clara
# 2015; two-sided since v12.2).
MMOM_LAMBDA = 1.0          # overlay strength: weight_i = cvg_i + λ · rank_i / N
MMOM_LOOK = 252            # 12-1: the total return from t-252 to t-21
MMOM_SKIP = 21
MMOM_GATE = 504            # bars of market return the bear gate reads (24 months)
MMOM_VOL_WIN = 126         # bars of overlay return its volatility is measured over
MMOM_FLOOR = 0.25          # no name below this share of its CVG weight: the grid's own floor
                           # to neutral (Distribution 0.25 : Idle 1), so every weight stays positive
                           # and the book always fills its count (held only if top-N reaches it)
MMOM_MIN_RANKED = 10       # fewer momentum-scored names than this: no overlay
MMOM_MIN_VOL_MONTHS = 7    # month-start volatility readings before the scale may act
MMOM_HISTORY_START = "2006-01-01"   # the close history app.py fetches for the overlay
MMOM_MIN_HISTORY = MMOM_GATE + 1    # rows before the overlay may act, whatever the source (MM-B5)
# The volatility scale is min(MMOM_SCALE_CAP, median / now): the overlay shrinks while its own
# volatility runs above its long-run median and grows, up to 1.5x, while it runs below — volatility
# targeting in both directions (Barroso & Santa-Clara 2015; Moreira & Muir 2017). Pre-registered
# (research/audit_mmom.py, MM-O1 BSC_UP15) and re-measured on the v12.2 code and data: ahead of the
# one-sided scale (cap 1) in all six era cells — Nifty 50 +0.28 / +0.14 / +0.22 %/yr, Dow 30 +0.05 /
# +0.18 / +0.01 — and level on the point-in-time Dow (+0.006); none significant.
MMOM_SCALE_CAP = 1.5
# The windows above count ROWS of a 5-day calendar. A panel with > 300 rows in its trailing
# 365 days (Crypto, or a Custom List mixing 7-day and 5-day calendars) reads the same spans
# in 7-day rows; read as 5-day rows they were ~8-1 momentum, a 16.5-month gate and a 4-month
# volatility window, and the Crypto gate read shut at -22% on a +26% market (MM-B2).
MMOM_WINDOWS_7D = dict(look=365, skip=30, gate=730, vol_win=183, ann=365.0)


def mmom_windows(index) -> dict:
    """The overlay's row windows for the panel's own calendar (rows in the trailing 365 days)."""
    ix = pd.DatetimeIndex(index) if len(index) else pd.DatetimeIndex([])
    bpy = int((ix > ix[-1] - pd.Timedelta(days=365)).sum()) if len(ix) else 0
    if bpy > 300:
        return dict(MMOM_WINDOWS_7D, bpy=bpy, calendar="7-day")
    return dict(look=MMOM_LOOK, skip=MMOM_SKIP, gate=MMOM_GATE, vol_win=MMOM_VOL_WIN, ann=252.0,
                bpy=bpy, calendar="5-day")


def mmom_ranks(score: pd.Series, index) -> pd.Series:
    """Centred cross-sectional rank in [-1, 1] over the names with a score; the rest sit at 0."""
    s = pd.to_numeric(score.reindex(index), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < MMOM_MIN_RANKED:
        return pd.Series(0.0, index=index)
    r = s.rank(method="average")
    return (2.0 * (r - 1.0) / (len(s) - 1.0) - 1.0).reindex(index).fillna(0.0)


def mmom_momentum(closes: pd.DataFrame, names, look: int = MMOM_LOOK,
                  skip: int = MMOM_SKIP) -> pd.Series:
    """12-1 total return as of the panel's last row. `closes` carried over gaps of <= 5 sessions."""
    if closes is None or len(closes) < look + 1:
        return pd.Series(np.nan, index=list(names))
    p1 = closes.iloc[-1 - skip].reindex(names)
    p0 = closes.iloc[-1 - look].reindex(names)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (p1 / p0 - 1.0).replace([np.inf, -np.inf], np.nan)


def mmom_gate(prices: pd.DataFrame, gate: int = MMOM_GATE,
              look: int = MMOM_LOOK) -> Tuple[float, float]:
    """(gate, market return): 0 while the equal-weighted market's 24-month return is negative.

    The market is the mean daily return of every name priced that day. With less than a year
    of history the gate cannot be read and stays open (1); with less than 24 months it reads
    what there is — app.py logs the span either way.
    """
    # Closes carried over gaps of <= 5 sessions first, as momentum and the research panel read
    # them: uncarried, every return across a missing quote was lost (MM-B1).
    mkt = prices.ffill(limit=5).pct_change(fill_method=None).mean(axis=1).dropna()
    n = min(gate, len(mkt))
    if n < look:
        return 1.0, float("nan")
    ret = float(np.prod(1.0 + mkt.iloc[-n:].to_numpy()) - 1.0)
    return (0.0 if ret < 0.0 else 1.0), ret


def mmom_scale(prices: pd.DataFrame, look: int = MMOM_LOOK, skip: int = MMOM_SKIP,
               vol_win: int = MMOM_VOL_WIN, ann: float = 252.0) -> Tuple[float, float, float, int]:
    """(scale, current vol, median vol, months): Barroso & Santa-Clara's scale for the overlay.

    The unit overlay — the rank / N weights re-formed at every month start and held to the next,
    the current month to date included — is priced from the closes; its 126-day realised
    volatility today is set against the median of that volatility at every month start up to
    and including today (an expanding median: nothing after today enters it). The scale is
    min(MMOM_SCALE_CAP, median / today): it shrinks the overlay while its volatility runs above
    the median and grows it, up to 1.5x, while it runs below. Below MMOM_MIN_VOL_MONTHS
    readings it stays at 1.
    """
    if prices is None or len(prices) == 0 or not isinstance(prices.index, pd.DatetimeIndex):
        return 1.0, float("nan"), float("nan"), 0
    idx = prices.index
    starts = list(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first())
    if not starts:
        return 1.0, float("nan"), float("nan"), 0
    pos = {d: i for i, d in enumerate(idx)}
    carried = prices.ffill(limit=5)
    rets = carried.pct_change(fill_method=None)                # MM-B1: carried, as the gate
    bounds = list(zip(starts[:-1], starts[1:]))
    if idx[-1] > starts[-1]:
        bounds.append((starts[-1], idx[-1]))                 # the month to date
    pieces = []
    for m0, m1 in bounds:
        i0, i1 = pos[m0], pos[m1]
        if i0 < look:
            continue
        row = prices.iloc[i0]
        names = prices.columns[row.notna() & (row > 0)]
        u = mmom_ranks(mmom_momentum(carried.iloc[: i0 + 1], names, look, skip), names)
        if not (u != 0).any():
            continue
        rr = rets.iloc[i0 + 1: i1 + 1].reindex(columns=names).fillna(0.0)
        pieces.append(pd.Series(rr.to_numpy() @ (u.to_numpy() / len(names)), index=rr.index))
    if not pieces:
        return 1.0, float("nan"), float("nan"), 0
    sig = pd.concat(pieces).rolling(vol_win).std(ddof=1) * np.sqrt(ann)
    at = sig.reindex([d for d in starts if d in sig.index]).dropna()
    now = float(sig.iloc[-1])
    if len(at) < MMOM_MIN_VOL_MONTHS or not np.isfinite(now) or now <= 0:
        return 1.0, now, float(at.median()) if len(at) else float("nan"), int(len(at))
    target = float(at.median())
    return float(min(MMOM_SCALE_CAP, target / now)), now, target, int(len(at))


def _close_panel(prices: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """A caller's close panel made safe to slice by date: a naive DatetimeIndex, sorted, one row
    per day and one column per symbol (the last of each, as the snapshot panel keeps). None when
    it is empty or its index is not dates."""
    if prices is None or len(prices) == 0:
        return None
    try:
        ix = pd.DatetimeIndex(pd.to_datetime(prices.index))
    except (TypeError, ValueError):
        return None
    if ix.tz is not None:
        ix = ix.tz_localize(None)
    p = prices.set_axis(ix, axis=0)
    p = p.loc[:, ~p.columns.duplicated(keep="last")]
    return p[~p.index.duplicated(keep="last")].sort_index()


def mmom_overlay(prices: pd.DataFrame, names) -> Tuple[pd.Series, pd.Series, dict]:
    """(rank, 12-1 momentum, diagnostics) — the overlay Managed Momentum adds to CVG.

    `prices` is a wide close panel, date × symbol, ending on the rebalance date: every name in
    the universe (the gate's market), not only `names` (the ranks).
    """
    names = list(names)
    if prices is None:
        prices = pd.DataFrame()
    if len(prices) and not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices = prices.set_axis(pd.DatetimeIndex(pd.to_datetime(prices.index)), axis=0)
        except (TypeError, ValueError):
            prices = pd.DataFrame()
    if len(prices):
        # A close <= 0 is not a price (CL=F printed -37.63 on 2020-04-20): read as one it made
        # -306% and -127% "returns" that shut the gate and doubled the overlay's vol (MM-B7).
        prices = prices.where(prices > 0)
    win = mmom_windows(prices.index)
    mom = mmom_momentum(prices.ffill(limit=5), names, win["look"], win["skip"])
    rank = mmom_ranks(mom, names)
    # The bear state is formed monthly (Daniel & Moskowitz), as every measured book read it: the
    # gate reads the market as of the first session of the run date's month and holds through
    # the month. Read daily, a book built mid-month near a flip swung between momentum and the
    # pure grid from one day to the next (36.5% one-day turnover on Nifty 50, 2026-09-28; MM-B6).
    # Momentum and the volatility scale still read the run date.
    m0 = None
    if len(prices):
        t = prices.index[-1]
        m0 = prices.index[(prices.index.year == t.year) & (prices.index.month == t.month)][0]
    head = prices.loc[:m0] if m0 is not None else prices
    gate, mkt = mmom_gate(head, win["gate"], win["look"])
    scale, vol, vol_med, vol_months = (mmom_scale(prices, win["look"], win["skip"], win["vol_win"],
                                                  win["ann"])
                                       if gate > 0 else (1.0, float("nan"), float("nan"), 0))
    return rank, mom, {
        "gate": gate, "market_24m": mkt, "scale": scale, "strength": MMOM_LAMBDA * gate * scale,
        "overlay_vol": vol, "overlay_vol_median": vol_med, "vol_months": vol_months,
        "ranked": int(mom.notna().sum()), "history_days": int(len(prices)),
        "history_start": prices.index[0] if len(prices) else None,
        "windows": win, "history_needed": int(win["gate"] + 1), "gate_read_on": m0,
        "gate_rows": int(len(head)),
    }


def method_spec(method: str) -> dict:
    """Registry lookup that never raises — unknown methods degrade to Equal Weight."""
    return METHOD_SPECS.get(str(method).upper(), METHOD_SPECS["EQUAL"])


def _apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Renormalize to 1 with every weight <= cap, by waterfall redistribution.

    The naive loop — clip to the cap, then divide by the new sum — does NOT
    converge: dividing by a sum below 1 pushes the capped names straight back
    above the cap, and the iteration oscillates until it runs out of passes and
    returns weights that violate the very bound it was enforcing. Measured here
    before the fix: a min-variance solution concentrated in 8 names came out at
    12.50% each against a 10% cap.

    The procedure here fixes capped names at the cap and fills the shortfall into
    the HEADROOM (cap - w) of the uncapped names, not pro rata to their weight:
    the smallest weights gain the most, which pulls the uncapped tail toward
    equal weight ([0.5, 0.3, 0.2] at a 0.4 cap gives [0.4, 0.333, 0.267], where
    pro rata gives [0.4, 0.36, 0.24]). Pro rata was measured within noise (|Δ| <=
    0.12%/yr on any style or era, top-15 to every name; research/audit_cvg.py
    CVG-B11, audit_hrp.py D6), so the headroom rule is kept. Feasibility is checked first: capping n names at `cap` can only
    reach 100% when n * cap >= 1, so when the allocator concentrates into too
    few names the cap is relaxed to the tightest value that is satisfiable.
    """
    n = len(w)
    if n == 0:
        return w
    cap_eff = max(float(cap), 1.0 / n)          # n * cap_eff >= 1 by construction
    w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, None)
    if w.sum() <= 1e-12:
        return np.full(n, 1.0 / n)
    w = w / w.sum()

    # Clip first, then fill the shortfall into the HEADROOM (cap - w) of names
    # that are still below the cap. Because every addition is bounded by that
    # headroom and the result is clipped again, no weight can end above the cap
    # — the guarantee holds by construction rather than by convergence.
    #
    # A trailing `w / w.sum()` would break exactly that guarantee: after
    # clipping, the sum is below 1, so dividing by it scales the capped names
    # right back over the line. Measured before this fix: 12.500070% against a
    # 12.5% cap.
    w = np.minimum(w, cap_eff)
    for _ in range(50):
        shortfall = 1.0 - float(w.sum())
        if shortfall <= 1e-12:
            break
        headroom = np.clip(cap_eff - w, 0.0, None)
        total_head = float(headroom.sum())
        if total_head <= 1e-12:
            break                                # everything already at the cap
        w = w + shortfall * (headroom / total_head)
        w = np.minimum(w, cap_eff)
    return w


def build_returns_matrix(history: Sequence[Tuple[object, pd.DataFrame]],
                         symbols: Optional[List[str]] = None,
                         lookback: int = 252,
                         ) -> pd.DataFrame:
    """Daily simple returns from a (date, snapshot) history, wide by symbol.

    Symbols with less than MIN_COVERAGE of the window are dropped BEFORE any
    row-wise NaN drop. Doing it the other way round discards a date whenever any
    single symbol is missing, which on a universe whose members listed at
    different times throws away most of the sample — measured, it collapsed a
    38-period backtest to 14.
    """
    rows: Dict[object, pd.Series] = {}
    for dt, df in history:
        if df is None or df.empty or "symbol" not in df.columns or "price" not in df.columns:
            continue
        s = pd.to_numeric(df.set_index("symbol")["price"], errors="coerce")
        rows[dt] = s[~s.index.duplicated(keep="last")]
    if not rows:
        return pd.DataFrame()

    px = pd.DataFrame(rows).T.sort_index()
    if symbols:
        px = px.reindex(columns=[c for c in symbols if c in px.columns])
    if px.empty or px.shape[1] == 0:
        return pd.DataFrame()

    # A close <= 0 is not a price, and a dead quote (>= _DEAD_QUOTE_RUN repeats of
    # the same close) is not a return series: both are unpriced BEFORE returns are
    # taken. The live estimation panel arrives unmasked (generate_historical_data);
    # a frozen name otherwise enters the covariance with variance ~0, and HRP gave
    # it ~100% of raw weight (J&KBANK, NIFTY SMLCAP 250, Oct 2016 - Feb 2017).
    px = px.where(px > 0)
    _same = px.diff().eq(0)
    _run = _same.apply(lambda c: c.groupby((~c).cumsum()).transform("sum"))
    _dead = _same & (_run >= _DEAD_QUOTE_RUN)
    dead_counts = {str(c): int(k) for c, k in _dead.sum().items() if k}
    if dead_counts:
        px = px.mask(_dead)
    # fill_method=None: the pandas default pads every interior gap into zero
    # returns, which re-carries a masked or suspended name at a frozen price.
    rets = px.pct_change(fill_method=None).tail(lookback)
    if rets.empty:
        return pd.DataFrame()
    cover = rets.notna().sum()
    keep = [c for c in rets.columns if cover[c] >= MIN_COVERAGE * len(rets)]
    if not keep:
        return pd.DataFrame()
    out = rets[keep].dropna(how="any")
    # Degenerate columns (non-finite or near-zero variance over the rows kept) are
    # left out of the estimation universe rather than floored: a floor hands the
    # frozen name 37-87% of HRP's raw weight, and ERC solves it to the cap too.
    _v = out.var(ddof=1)
    _zero = out.eq(0.0).mean() if len(out) else pd.Series(0.0, index=out.columns)
    _fin = np.isfinite(_v.to_numpy(dtype=float))
    _med = float(np.median(_v.to_numpy(dtype=float)[_fin])) if _fin.any() else 0.0
    degenerate = {str(c): (float(_v[c] / _med) if _med > 0 and np.isfinite(_v[c]) else float("nan"))
                  for c in out.columns
                  if not np.isfinite(_v[c]) or _v[c] <= 0 or _zero[c] >= _DEGENERATE_ZERO_SHARE
                  or _v[c] < _DEGENERATE_VAR_RATIO * _med}
    if degenerate:
        out = out.drop(columns=list(degenerate))
    # Record WHAT was dropped and by how much, so a book built on fewer names
    # than the declared universe can be explained rather than guessed at. A
    # recently listed ETF cannot have a 252-day covariance estimate; excluding
    # it is correct, but it should never be silent.
    out.attrs["excluded"] = {
        c: {"obs": int(cover[c]), "window": int(len(rets)),
            "coverage": float(cover[c] / len(rets))}
        for c in rets.columns if c not in keep
    }
    out.attrs["coverage_window"] = int(len(rets))
    out.attrs["coverage_required"] = float(MIN_COVERAGE)
    out.attrs["dead"] = dead_counts              # dead quotes unpriced, per symbol
    out.attrs["degenerate"] = degenerate         # symbol -> variance / median variance
    return out


def build_price_matrix(history: Sequence[Tuple[object, pd.DataFrame]],
                       symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Wide price panel from the same (date, snapshot) history.

    Momentum needs LEVELS, not the returns matrix: the returns matrix is
    truncated to the covariance lookback and row-wise NaN-dropped, which would
    silently shorten the 12-month momentum window. This reads the full panel and
    lets `momentum_scores` decide what it has enough history for.
    """
    rows: Dict[object, pd.Series] = {}
    for dt, df in history:
        if df is None or df.empty or "symbol" not in df.columns or "price" not in df.columns:
            continue
        s = pd.to_numeric(df.set_index("symbol")["price"], errors="coerce")
        rows[dt] = s[~s.index.duplicated(keep="last")]
    if not rows:
        return pd.DataFrame()
    px = pd.DataFrame(rows).T.sort_index()
    if symbols:
        px = px.reindex(columns=[c for c in symbols if c in px.columns])
    return px


# Snapshot column → the name the book carries it under. Numeric readings first,
# then the two text fields.
CVG_FIELDS = {
    "conv tape": "conviction",               # the conviction tape (D · W)
    "conv daily": "conviction_daily",   # its daily rung (the pane's trace)
    "conv weekly": "conviction_weekly", # its reconstructed weekly rung
    "conv ladder down": "ladder_down",       # 1 = the tape read Ladder down (off since v12.2), 0 = D · W
    "conv hist": "hist",                     # the pane's histogram, native
    "conv push": "push",                     # the histogram as drawn, −1 … +1
    "value tape": "value_tape",              # the value tape, D · W (+ rich)
    "value daily": "value_daily",       # Samanvaya's reading on the chart
    "value hedge": "hedge",                   # share of the macro hedge applied
    "conv push gate": "push_gate",           # +1 / −1 a confirmed push, 0 none
    "cvg state days": "state_days",         # days in the current state
    "cvg held": "held_row",                 # 1 = row held against the tape
}
CVG_TEXT = {
    "cvg state": "state",                   # one of cvgrid.STATES
    "conv push tier": "push_tier",           # e.g. "up · impulse · quiet"
    "value drivers": "drivers",               # the hedge's drivers in force
}


def cvg_readings(history: Sequence[Tuple[object, pd.DataFrame]],
                    symbols: List[str]) -> pd.DataFrame:
    """Each symbol's grid readings as of the LAST snapshot in `history`.

    backdata computes them over each symbol's full OHLCV history, warm-up
    included — which is why they arrive as snapshot columns rather than being
    rebuilt here: the snapshots carry close prices only, and start after the
    warm-up both tapes need. A name with no reading — listed too recently, or
    a panel cached before the columns existed — is UNREAD, never guessed.
    """
    out = pd.DataFrame(np.nan, index=list(symbols), columns=list(CVG_FIELDS.values()))
    for name in CVG_TEXT.values():
        out[name] = None
    if history:
        snap = history[-1][1]
        if snap is not None and not snap.empty and "symbol" in snap.columns:
            snap = snap.drop_duplicates("symbol", keep="last").set_index("symbol")
            for col, name in CVG_FIELDS.items():
                if col in snap.columns:
                    out[name] = pd.to_numeric(snap[col], errors="coerce").reindex(out.index)
            for col, name in CVG_TEXT.items():
                if col in snap.columns:
                    out[name] = snap[col].reindex(out.index)
    st = out["state"].where(out["state"].isin(list(STATE_UNITS)), "UNREAD")
    out["state"] = st.fillna("UNREAD").astype(str)
    return out


# Read the map GRADED (each name shaded within its cell, as the Pine draws the
# tapes) rather than as flat cells. The research harness switches it off to
# measure what the grading itself adds.
CVG_GRADED: bool = True


def cvg_units(readings: pd.DataFrame) -> pd.Series:
    """Each name's weight in units on the 3 × 3 map (cvgrid.graded_units).

    Flat cell units when CVG_GRADED is off. The histogram has already done
    its work in deciding the state; on the graded map it also decides how
    firmly a HELD row keeps its cell.
    """
    if not CVG_GRADED:
        return r_units_flat(readings)
    conv = pd.to_numeric(readings["conviction"], errors="coerce")
    val = pd.to_numeric(readings["value_tape"], errors="coerce")
    push = pd.to_numeric(readings["push"], errors="coerce")
    return pd.Series([graded_units(str(s), c, v, p) for s, c, v, p in
                      zip(readings["state"], conv, val, push)], index=readings.index, dtype=float)


def r_units_flat(readings: pd.DataFrame) -> pd.Series:
    """Each name's cell units, unshaded."""
    return readings["state"].map(STATE_UNITS).fillna(STATE_UNITS["UNREAD"]).astype(float)


def cvg_weights(readings: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Weights from each name's place on the 3 × 3 map, and the fill order.

    Selection follows the weight, as for every allocator here: the heaviest
    names first. Ties fall back to the states' own order (units, then the grid)
    and then to the room between control and price, `conviction − value`, the
    one ordering both tapes agree on. Every unit is strictly positive, so the
    book always holds the N it was asked for: a punished name sits at the
    floor, it is never dropped.
    """
    r = readings.copy()
    units = cvg_units(r)
    room = (pd.to_numeric(r["conviction"], errors="coerce")
            - pd.to_numeric(r["value_tape"], errors="coerce")).fillna(0.0)
    key = pd.DataFrame({"units": -units,
                        "order": r["state"].map(STATE_ORDER).fillna(len(STATE_ORDER)),
                        "room": -room}, index=r.index)
    order = list(key.sort_values(["units", "order", "room"], kind="stable").index)
    w = units.reindex(order).to_numpy(dtype=float)
    return w / w.sum(), order


def _is_priced(price: object) -> bool:
    """A usable, positive, finite price — the only input equal weight requires."""
    try:
        p = float(price)          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(p)) and p > 0


# HRP's leaf order is re-drawn from scratch every month, and dropping 2 of 252 days changes it in
# 99% of trials: most of HRP's turnover is estimation noise, not a change of view. HRP is therefore
# the mean of its fits on HRP_WINDOWS windows of `lookback` rows ending 0, HRP_STEP, 2·HRP_STEP
# sessions back, inside the last HRP_PANEL sessions (the app's estimation panel holds ~400).
# Pre-registered and measured through this code on the v12.2 inputs (research/audit_hrp.py,
# O1-A, re-measured on the final v12.2 panels): higher net CAGR in all six era cells (+0.12 to
# +0.46 %/yr, none significant), ~40% less turnover (Nifty 50 1.23 -> 0.72x/yr, Dow 0.92 -> 0.53),
# point-in-time Dow +0.37, ETF book -0.25 over 19 months. A panel too short for a
# window simply uses fewer; on a single window it is exactly the one-fit HRP.
HRP_WINDOWS = 3
HRP_STEP = 21
HRP_PANEL = 400


def hrp_staggered(history: Sequence[Tuple[object, pd.DataFrame]], est_names: List[str],
                  cov: np.ndarray, corr: np.ndarray, lookback: int = 252) -> Tuple[np.ndarray, List[int]]:
    """(weights over est_names, the windows used): the mean of hrp_weights over the staggered
    windows. Window 0 is today's (`cov`, `corr`); an earlier window is built over today's
    estimation names with the same rules, and skipped when it is not estimable. A name an
    earlier window could not estimate is averaged over the windows that could."""
    hist = list(history)[-HRP_PANEL:]
    parts = [pd.Series(hrp_weights(cov, corr), index=est_names)]
    used = [0]
    for j in range(1, HRP_WINDOWS):
        end = len(hist) - HRP_STEP * j
        if end < lookback + 1:
            continue
        Rj = build_returns_matrix(hist[:end], symbols=list(est_names), lookback=lookback)
        nj = Rj.shape[1]
        if Rj.empty or nj < 2 or len(Rj) < MIN_OBS or len(Rj) < MIN_OBS_PER_ASSET * nj:
            continue
        Xj = Rj.to_numpy(dtype=float)
        cj = np.cov(Xj, rowvar=False)
        rj = np.nan_to_num(np.corrcoef(Xj, rowvar=False), nan=0.0)
        parts.append(pd.Series(hrp_weights(cj, rj), index=Rj.columns).reindex(est_names))
        used.append(j)
    W = pd.concat(parts, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    tot = float(W.sum())
    return (W / tot).to_numpy(dtype=float) if tot > 0 else parts[0].to_numpy(dtype=float), used


def compute_nco_portfolio(history: Sequence[Tuple[object, pd.DataFrame]],
                          prices: Dict[str, float],
                          capital: float,
                          num_positions: int,
                          method: str = "HRP",
                          max_pos_pct: float = 0.10,
                          lookback: int = 252,
                          price_history: Optional[pd.DataFrame] = None,
                          ) -> pd.DataFrame:
    """Curate a portfolio with one of the registered styles (METHOD_SPECS).

    Selection AND weighting both come from the allocator: weights are computed
    over every eligible symbol, the top `num_positions` by weight are kept, and
    those are renormalized. Equal Weight, ERC and HRP forecast nothing (ERC and
    HRP weight from the covariance); the Conviction-Value Grid sizes by the
    tape's state, and Managed Momentum adds a 12-1 momentum overlay to it,
    reading `price_history` — a wide close panel, date × symbol, cut here to the
    book's date — and standing down to the grid without it.

    ELIGIBILITY IS PER-STYLE, because it is a property of the weight formula.
    Styles that read the covariance (`needs_covariance` in METHOD_SPECS) can only
    hold names that HAVE one, so they are confined to the estimation universe —
    symbols with at least MIN_COVERAGE of the window. Equal weight reads nothing,
    so it allocates over every priced symbol; a coverage rule protects a
    covariance, and 1/N has none to protect. The risk diagnostics are always
    computed on the estimation universe, and `nco_rc_coverage` records the share
    of book weight they describe (1.0 for every covariance-driven style).

    A per-position cap is applied (relaxed to 1/n when n makes it infeasible),
    but NO book-level floor: a floor would fight the method. (Managed Momentum's
    MMOM_FLOOR is inside its weight formula — no name below a quarter of its grid
    weight — not a floor on the book.) The entire point is that a
    redundant asset — one whose risk is already carried by a cluster peer —
    SHOULD receive a small weight. Forcing it up to 1% would re-introduce the
    concentration the clustering exists to remove.

    Returns a DataFrame with symbol / price / weightage_pct / units / value plus
    `cluster` and `risk_contribution`, and carries `.attrs` describing the fit
    (`nco_method`, `nco_clusters`, `nco_silhouette`, `nco_obs`, `max_pos_pct_eff`).
    Holdings with no covariance estimate carry NaN in every risk column rather
    than a fabricated number. Returns an empty frame when a covariance-driven
    style has no covariance it can trust.
    """
    empty = pd.DataFrame()
    if not prices or capital <= 0 or num_positions <= 0:
        return empty

    _m = str(method).upper()
    if _m not in METHOD_SPECS:
        # method_spec() degrades an unknown name to Equal Weight's spec for the UI;
        # here that ran HRP under EQUAL's eligibility and crashed or mislabelled.
        raise ValueError(f"unknown method {method!r}; registered: {', '.join(METHOD_SPECS)}")
    _spec = method_spec(_m)
    # Does this style's WEIGHT FORMULA read the covariance? The answer decides
    # which names it is allowed to hold — see the eligibility split below.
    _needs_cov = bool(_spec.get("needs_covariance", True))

    rets = build_returns_matrix(history, symbols=list(prices.keys()), lookback=lookback)
    est_names = list(rets.columns)
    n_est = len(est_names)
    # Is the covariance estimable? Enough observations outright, and at least
    # MIN_OBS_PER_ASSET (one) per asset. Behaviour is unchanged from the old
    # `4.0 * n / 4.0`; T/n is recorded so a book estimated near 1 is visible.
    cov_ok = (
        not rets.empty
        and n_est >= 2
        and len(rets) >= MIN_OBS
        and len(rets) >= MIN_OBS_PER_ASSET * n_est
    )
    # A style that allocates FROM the covariance cannot proceed without one. A
    # style that does not, can — and must: refusing to build an equal-weight book
    # because a matrix it never looks at was not estimable is a defect dressed as
    # a safeguard.
    if _needs_cov and not cov_ok:
        # Say WHY, with the numbers: the app's empty-book message reads these.
        empty.attrs.update(nco_obs=int(len(rets)), nco_n_est=int(n_est),
                           nco_coverage_window=int(rets.attrs.get("coverage_window", 0)))
        return empty

    # ── Two universes, not one ────────────────────────────────────────────────
    # ESTIMATION universe: the names carrying at least MIN_COVERAGE of the
    # window, i.e. the names that HAVE a covariance estimate. Every risk number
    # this module reports — cluster, risk contribution, volatility, correlation
    # to the book, ex-ante volatility — is computed over these and only these.
    #
    # ALLOCATION universe: the names capital is actually spread across. For a
    # covariance-driven style the two sets are identical, because a weight for a
    # name with no estimate is undefined. Equal weight estimates nothing, so its
    # allocation universe is every priced symbol: the coverage rule exists to
    # protect a covariance, and 1/N has no covariance to protect. Excluding a
    # recently listed ETF from a 1/N book was a rule with no statistic behind it.
    #
    # The estimation set is still computed on an equal-weight run — the risk
    # diagnostics are the reason to look at the book — but it no longer decides
    # what the book holds.
    if _needs_cov:
        alloc_names = list(est_names)
    else:
        alloc_names = [s for s in prices if _is_priced(prices.get(s))]
    if not alloc_names:
        return empty

    est_pos = {s: i for i, s in enumerate(est_names)}

    if cov_ok:
        R = rets.to_numpy(dtype=float)
        cov = np.cov(R, rowvar=False)
        corr = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
        # Clustered ONCE and reused for both the count/silhouette report and the
        # per-holding labels. Two separate calls cost twice as much for the same
        # answer, and would silently disagree if the routine ever stopped being
        # deterministic.
        labels, k, sil = cluster_assets(corr)
        lab_by_name = {s: int(labels[i]) for i, s in enumerate(est_names)}
    else:
        R = np.zeros((0, 0))
        cov = None
        corr = None
        labels, k, sil = np.zeros(0, dtype=int), 0, 0.0
        lab_by_name = {}

    # Every style is computed HERE rather than short-circuited earlier, so all of
    # them travel the identical pipeline — same selection, same position cap,
    # same risk decomposition. That makes the styles genuinely comparable on
    # screen: any difference the user sees is the allocator. The one thing they
    # do NOT share is the eligibility rule above, which is a property of the
    # weight formula rather than a stylistic choice.
    mom = pd.Series(np.nan, index=alloc_names)
    _mmom: Optional[dict] = None
    _hrp_windows: Optional[List[int]] = None
    _floored_names: set = set()
    # The covariance the ALLOCATOR optimised against. The convergence diagnostic
    # below must be measured on this matrix, not on the sample covariance used
    # for reporting: ERC solves on the shrunk estimate, so scoring its solution
    # against the raw sample matrix reports a dispersion of ~0.07 for a solver
    # that in fact converged exactly. Two different matrices, two different
    # questions — keep them apart.
    solver_cov = cov
    # The grid's readings, per allocation name. Read only on a CVG run: every
    # other style's output columns stay empty, so no chart can imply a reading
    # the book did not use.
    _uses_dh = bool(_spec.get("uses_cvg", False))
    dh: Optional[pd.DataFrame] = None

    if _m == "EQUAL":
        w = np.full(len(alloc_names), 1.0 / len(alloc_names))
    elif _m == "CVG":
        # Sized by STATE from the two tapes; no covariance is read. The
        # allocation names are REORDERED into the order the book fills in, so
        # the stable top-N below takes core names first and the floor last.
        _read = cvg_readings(history, alloc_names)
        w, alloc_names = cvg_weights(_read)
        dh = _read.reindex(alloc_names)
    elif _m == "MMOM":
        # The grid's weights, then the momentum overlay on top: + λ · rank / N,
        # its strength set by the bear gate and the volatility scale, no name
        # below MMOM_FLOOR of its CVG weight. The overlay reads `price_history`
        # (the long close panel) when given, else the estimation panel — whose
        # ~400 sessions (~19 months) cannot read the gate's 24 months, so the
        # overlay then stands down to the grid (nco_mmom_stood_down says why).
        _read = cvg_readings(history, alloc_names)
        _c, alloc_names = cvg_weights(_read)
        dh = _read.reindex(alloc_names)
        # Never past the book's own date: a close history cached to a later day
        # must not lend the overlay tomorrow's market.
        _ph = _close_panel(price_history)
        if _ph is not None and not _ph.empty and history:
            _ph = _ph.loc[:pd.Timestamp(history[-1][0]).normalize()]
        _from_close = _ph is not None and not _ph.empty
        _px = _ph if _from_close else build_price_matrix(history)
        _rank, _mom, _mmom = mmom_overlay(_px, alloc_names)
        _last = (_px.ffill(limit=5).iloc[-1] if len(_px) else pd.Series(dtype=float))
        _mmom["coverage"] = float(np.mean([np.isfinite(float(_last.get(s, np.nan)))
                                           for s in alloc_names])) if alloc_names else 0.0
        _mmom["source"] = "close history" if _from_close else "estimation panel"
        _mmom["stood_down"] = None
        # A history short of the gate's 24 months (the ~400-session estimation panel, a run
        # date before ~2008, a Custom List of recent listings) cannot read its own crash
        # guard: the overlay stands down to the grid rather than run at full strength
        # unguarded, whichever history it read (MM-B5; it used to test the fallback only).
        if _mmom["gate_rows"] < _mmom["history_needed"]:
            _mmom.update(strength=0.0, stood_down=f"{_mmom['source']} too short for the 24-month gate")
        _cw = pd.Series(_c, index=alloc_names)
        _tilted = _cw + _mmom["strength"] * _rank / len(_cw)
        _floored = _tilted < MMOM_FLOOR * _cw
        _floored_names = set(_floored.index[_floored.to_numpy()])
        w = np.maximum(_tilted, MMOM_FLOOR * _cw).to_numpy(dtype=float)
        mom = _mom.reindex(alloc_names)
        _mmom["floored"] = int(len(_floored_names))
    elif cov is None or corr is None:
        # Unreachable as the registry stands: every covariance-driven style
        # returned above when the covariance was not estimable. Kept as a hard
        # guard so a style added later without `needs_covariance` set cannot
        # silently dereference a matrix that was never built.
        return empty
    elif _m == "HRP":
        w, _hrp_windows = hrp_staggered(history, est_names, cov, corr, lookback)
    elif _m in ("ERC", "ERC_MOM"):
        solver_cov = ledoit_wolf(R)
        w = erc_weights(solver_cov)
        if _m == "ERC_MOM":
            px_panel = build_price_matrix(history, symbols=alloc_names)
            mom = momentum_scores(px_panel, MOMENTUM_LOOKBACK, MOMENTUM_SKIP)
            mom = mom.reindex(alloc_names)
            if mom.notna().sum() >= 2:
                w = apply_momentum_tilt(w, mom, MOMENTUM_LAMBDA)
            # Too few names carry a full 12-1 window to rank meaningfully; the
            # book falls back to plain ERC rather than tilting on noise. The
            # caller can see this happened via the `nco_momentum_names` attr.
    else:
        raise ValueError(f"no weight rule for registered method {_m!r}")

    w = np.nan_to_num(w, nan=0.0)
    if w.sum() <= 1e-12:
        return empty
    w = w / w.sum()

    # Risk balance as SOLVED, over the full eligible set and before any selection
    # or cap. Kept separately from the realised figure below because the two
    # answer different questions: this one says whether the optimiser converged,
    # the realised one says what the book the user actually holds looks like
    # after top-N selection and the position cap have moved it. Conflating them
    # makes a correct ERC solve look like a failed one.
    #
    # Undefined for a style that solves nothing: 1/N has no solution to converge
    # to, and its weight vector does not even span the estimation universe.
    if _needs_cov and cov_ok and solver_cov is not None:
        _rc_solved = risk_contributions(w, solver_cov)
        _rc_solved_disp = (float(np.std(_rc_solved) / np.mean(_rc_solved))
                           if np.mean(_rc_solved) > 1e-18 else 0.0)
    else:
        _rc_solved_disp = float("nan")

    # Sorted by weight, and STABLY. Equal weight makes every entry a tie, so an
    # unstable sort would leave quicksort's partition order to decide which N of
    # them the book holds — arbitrary, and liable to change under an unrelated
    # pandas upgrade. A stable sort keeps ties in universe order, which for an
    # index universe is its published constituent order.
    ser = pd.Series(w, index=alloc_names).sort_values(ascending=False, kind="stable")
    # Drop names the allocator zeroed out BEFORE selecting. Corner-solution
    # optimisers (minimum variance, maximum diversification) put most names at
    # exactly 0, and carrying those into the top-N selection would fill the book
    # with zero-weight rows that the cap logic then has to reason about.
    #
    # This is also why Max Diversification was withdrawn as a shipped style: it
    # routinely zeroed enough names that the book came back SHORTER than the
    # position count the user asked for (10 of a requested 15 on this universe).
    # A style that silently re-decides how many positions you hold is not a
    # weighting method. `nco_positions_short` below makes any recurrence visible
    # rather than silent.
    n_nonzero = int((ser > 1e-9).sum())
    ser = ser[ser > 1e-9]
    if ser.empty:
        return empty
    chosen = ser.head(min(num_positions, len(ser)))
    chosen = chosen / chosen.sum()

    wv = _apply_cap(chosen.to_numpy(dtype=float), max_pos_pct)
    n = len(wv)
    cap_eff = max(max_pos_pct, 1.0 / n)

    sel = list(chosen.index)
    if _mmom is not None:
        # The floor is counted over the universe before top-N; this is how many of those the
        # book actually holds: usually none below the universe size, though a floored
        # Dislocated name can outweigh an unfloored Idle one and make the cut.
        _mmom["floored_held"] = int(sum(s in _floored_names for s in sel))
    # Which HOLDINGS carry a covariance estimate. Identical to `sel` for every
    # covariance-driven style. On an equal-weight book it can be a strict subset,
    # and the diagnostics below are then reported over that subset with the share
    # of book weight they cover recorded in `nco_rc_coverage` — rather than
    # quietly implying they describe the whole book.
    covered = np.array([bool(cov_ok and s in est_pos) for s in sel])

    rc = np.full(n, np.nan)
    ann_vol = np.full(n, np.nan)
    corr_to_book = np.full(n, np.nan)
    cluster_col = np.full(n, np.nan)
    port_var = float("nan")
    rc_coverage = 0.0

    if covered.any() and cov is not None:
        cidx = [est_pos[s] for s, ok in zip(sel, covered) if ok]
        sub_cov = cov[np.ix_(cidx, cidx)]
        w_cov = wv[covered]
        rc_coverage = float(w_cov.sum())
        # Renormalised WITHIN the covered sub-book, so every figure below reads
        # as "of the risk this book's measurable part carries". At full coverage
        # — every style except a partially estimable equal-weight run — the
        # renormalisation is by 1.0 and these are exactly the same numbers as
        # before the estimation set and the allocation set were separated.
        wc = (w_cov / w_cov.sum() if w_cov.sum() > 1e-18
              else np.full(len(cidx), 1.0 / len(cidx)))
        port_var = float(wc @ sub_cov @ wc)
        # Marginal risk contribution, normalised to sum to 1 — shows whether the
        # clustering actually balanced risk or merely balanced capital.
        _rc = wc * (sub_cov @ wc)
        _rc = (_rc / _rc.sum() if abs(_rc.sum()) > 1e-18
               else np.full(len(cidx), 1.0 / len(cidx)))
        rc[covered] = _rc
        # Per-asset diagnostics for the risk heatmap: annualized volatility, and
        # each holding's correlation to the finished book (how much it moves WITH
        # the portfolio, i.e. how little it diversifies).
        ann_vol[covered] = np.sqrt(np.diag(sub_cov)) * np.sqrt(252)
        book_ret = R[:, cidx] @ wc
        corr_to_book[covered] = np.nan_to_num(np.array([
            float(np.corrcoef(R[:, j], book_ret)[0, 1]) if np.std(R[:, j]) > 1e-12 else 0.0
            for j in cidx
        ], dtype=float), nan=0.0)
        cluster_col[covered] = [float(lab_by_name.get(s, 0))
                                for s, ok in zip(sel, covered) if ok]

    px_arr = np.array([float(prices.get(s, np.nan)) for s in sel], dtype=float)
    # Grid readings per holding, for the same reason momentum is carried: the
    # table and charts show the reading that set the weight, not a
    # re-derivation. Empty for every other style.
    _dh_sel = (dh.reindex(sel) if dh is not None
               else pd.DataFrame(index=sel, columns=[*CVG_FIELDS.values(),
                                                     *CVG_TEXT.values()]))
    # Momentum is carried per-holding so the risk profile chart can show the
    # tilt that was actually applied, not a re-derivation of it. Methods that do
    # not use momentum emit NaN, and the chart drops the row.
    _mom_sel = mom.reindex(sel) if _spec["uses_momentum"] else pd.Series(
        np.nan, index=sel)
    _mom_z = (rank_z(_mom_sel) if _spec["uses_momentum"] and _mom_sel.notna().sum() >= 2
              else pd.Series(np.nan, index=sel))
    out = pd.DataFrame({
        "symbol": sel,
        "price": px_arr,
        "weightage_pct": wv * 100.0,
        "cluster": cluster_col,
        "risk_contribution": rc,
        "volatility": ann_vol,
        "corr_to_book": corr_to_book,
        "momentum": _mom_sel.to_numpy(dtype=float),
        "momentum_z": _mom_z.to_numpy(dtype=float),
        **{name: pd.to_numeric(_dh_sel[name], errors="coerce").to_numpy(dtype=float)
           for name in CVG_FIELDS.values()},
        **{name: _dh_sel[name].to_numpy(dtype=object) for name in CVG_TEXT.values()},
        "state_units": (_dh_sel["state"].map(STATE_UNITS).to_numpy(dtype=float)
                        if dh is not None else np.full(len(sel), np.nan)),
        "cvg_units": (cvg_units(_dh_sel).to_numpy(dtype=float)
                         if dh is not None else np.full(len(sel), np.nan)),
    })
    out = out[out["price"].notna() & (out["price"] > 0)].copy()
    if out.empty:
        return empty
    out["weightage_pct"] = out["weightage_pct"] / out["weightage_pct"].sum() * 100.0
    out["units"] = np.floor((capital * out["weightage_pct"] / 100.0) / out["price"])
    # Flooring to whole shares alone left up to ~14% of capital idle on a Rs 5L, 50-name
    # book (91.6% invested on average 2020-26, a -1.6%/yr cash drag on CVG); every
    # measured figure assumes a fully invested book. The leftover is spent one share at a
    # time on the holding furthest below its target value, never past the position cap,
    # until no further share is affordable (research/audit_cvg.py, CVG-B3).
    _units, _px_u = out["units"].to_numpy(dtype=float), out["price"].to_numpy(dtype=float)
    _target = capital * out["weightage_pct"].to_numpy(dtype=float) / 100.0
    _cap_value = max(cap_eff, 1.0 / len(out)) * capital + 1e-9
    _cash = float(capital - (_units * _px_u).sum())
    _topped = 0
    # Greedy in bulk: the holding furthest below target buys as many shares as bring it level
    # with the next-furthest (at least one), capped by the cash and the cap. One share at a time
    # stalled on a sub-cent coin (SHIB ~1.2e-5) for 200,000 steps with whole shares affordable.
    for _ in range(10_000):
        _ok = (_px_u <= _cash + 1e-9) & ((_units + 1.0) * _px_u <= _cap_value)
        if not _ok.any():
            break
        _gap = np.where(_ok, _target - _units * _px_u, -np.inf)
        _i = int(np.argmax(_gap))
        _rest = np.delete(_gap, _i)
        _most = int(min(np.floor((_cash + 1e-9) / _px_u[_i]),
                        np.floor((_cap_value - _units[_i] * _px_u[_i]) / _px_u[_i])))
        _k = (_most if not _rest.size or not np.isfinite(_rest.max())
              else int(np.clip(np.ceil((_gap[_i] - _rest.max()) / _px_u[_i]), 1, _most)))
        _units[_i] += _k
        _cash -= _k * _px_u[_i]
        _topped += _k
    out["units"] = _units
    out["value"] = out["units"] * out["price"]
    # Rows that hold no share even after the top-up (the price exceeds the cash left or the
    # cap): in the book, unfunded. Any
    # style can produce them (a small weight on an expensive name); the grid's floor and
    # Managed Momentum's make them likelier. Recorded so the app can say so, never hidden.
    _unfunded = out.loc[out["units"] <= 0, "symbol"].astype(str).tolist()

    out.attrs["nco_method"] = _m
    out.attrs["nco_method_label"] = _spec["label"]
    out.attrs["nco_method_family"] = _spec["family"]
    out.attrs["nco_method_formula"] = _spec["formula"]
    out.attrs["nco_rc_target"] = _spec["rc_target"]
    out.attrs["nco_uses_clusters"] = bool(_spec["uses_clusters"])
    out.attrs["nco_uses_momentum"] = bool(_spec["uses_momentum"])
    out.attrs["nco_needs_covariance"] = bool(_needs_cov)
    out.attrs["nco_cov_estimable"] = bool(cov_ok)
    out.attrs["nco_clusters"] = int(k)
    out.attrs["nco_silhouette"] = float(sil)
    out.attrs["nco_obs"] = int(len(rets))
    out.attrs["nco_obs_per_asset"] = float(len(rets) / n_est) if n_est else float("nan")
    if _hrp_windows is not None:
        # The staggered windows HRP averaged (sessions back = HRP_STEP × each entry).
        out.attrs["nco_hrp_windows"] = list(_hrp_windows)
    out.attrs["nco_dead_quotes"] = dict(rets.attrs.get("dead", {}))
    out.attrs["nco_degenerate"] = dict(rets.attrs.get("degenerate", {}))
    # The set the ALLOCATOR worked over, and the (possibly smaller) set the risk
    # numbers were estimated on. They differ only when a style that needs no
    # covariance holds a name that has none.
    out.attrs["nco_universe"] = int(len(alloc_names))
    out.attrs["nco_estimation_universe"] = int(n_est)
    out.attrs["nco_port_vol_ann"] = (float(np.sqrt(max(port_var, 0.0)) * np.sqrt(252))
                                     if np.isfinite(port_var) else float("nan"))
    # What share of the book's weight the risk figures above actually describe.
    # 1.0 for every covariance-driven style, by construction.
    out.attrs["nco_rc_coverage"] = float(rc_coverage)
    out.attrs["nco_positions_uncovered"] = int(out["risk_contribution"].isna().sum())
    out.attrs["max_pos_pct_eff"] = float(cap_eff)
    out.attrs["min_pos_pct_eff"] = 0.0
    # Risk-balance diagnostics. `rc_dispersion` is the coefficient of variation
    # of the risk contributions: 0 is perfect equal-risk, and it is the number
    # that says whether ERC actually achieved what it targets. `rc_concentration`
    # is the heaviest holding's risk share against its equal share — the header
    # card's "Risk Concentration".
    _rc_fin = rc[np.isfinite(rc)]
    _rc_sel = _rc_fin[_rc_fin > 0] if (_rc_fin > 0).any() else _rc_fin
    out.attrs["nco_rc_dispersion"] = (float(np.std(_rc_sel) / np.mean(_rc_sel))
                                      if len(_rc_sel) and np.mean(_rc_sel) > 1e-18
                                      else float("nan"))
    out.attrs["nco_rc_dispersion_solved"] = float(_rc_solved_disp)
    out.attrs["nco_rc_concentration"] = (float(np.nanmax(rc) * len(_rc_fin))
                                         if len(_rc_fin) else float("nan"))
    # Position-count contract: the book must hold exactly what the user asked
    # for, unless the ELIGIBLE UNIVERSE itself is smaller. `nco_positions_short`
    # separates the two causes — a short book because the universe ran out is a
    # data condition the user can act on; a short book because the allocator
    # zeroed names out is an allocator defect.
    #
    # Eligibility, carried through so the UI can say why the book was built from
    # fewer names than the universe declares. `nco_universe_excluded` is what the
    # coverage rule kept OUT OF THE BOOK — empty for a style that excludes
    # nothing — while `nco_diagnostic_excluded` is what it kept out of the RISK
    # NUMBERS, which is the same set for a covariance-driven style and a
    # strictly milder statement for equal weight.
    _excl = dict(rets.attrs.get("excluded", {})) if not rets.empty else {}
    _degen = dict(rets.attrs.get("degenerate", {})) if not rets.empty else {}
    out.attrs["nco_universe_requested"] = int(n_est + len(_excl) + len(_degen) if _needs_cov
                                              else len(alloc_names))
    out.attrs["nco_universe_excluded"] = dict(_excl) if _needs_cov else {}
    out.attrs["nco_diagnostic_excluded"] = dict(_excl)
    out.attrs["nco_coverage_required"] = float(rets.attrs.get("coverage_required", MIN_COVERAGE))
    out.attrs["nco_coverage_window"] = int(rets.attrs.get("coverage_window", len(rets)))
    out.attrs["nco_positions_requested"] = int(num_positions)
    out.attrs["nco_positions_delivered"] = int(len(out))
    out.attrs["nco_positions_nonzero"] = int(n_nonzero)
    out.attrs["nco_positions_short"] = max(0, int(num_positions) - int(len(out)))
    out.attrs["nco_positions_unfunded"] = int(len(_unfunded))
    out.attrs["nco_cash"] = float(max(_cash, 0.0))
    # n x cap <= 1: the cap pins every holding at exactly 1/n (10 positions at 10%, or any
    # count where the cap was relaxed to 1/n), so the style chose the names, not the weights.
    out.attrs["nco_flat_by_cap"] = bool(len(out) * max(cap_eff, 1.0 / len(out)) <= 1.0 + 1e-9)
    out.attrs["nco_topup_shares"] = int(_topped)
    out.attrs["nco_unfunded_symbols"] = list(_unfunded)
    out.attrs["nco_short_cause"] = (
        "none" if len(out) >= num_positions
        else "universe" if len(alloc_names) <= num_positions or n_nonzero >= num_positions
        else "allocator_zeroed")
    out.attrs["nco_momentum_names"] = int(_mom_sel.notna().sum())
    out.attrs["nco_momentum_lambda"] = (float(_mmom["strength"]) if _mmom is not None
                                        else float(MOMENTUM_LAMBDA) if _spec["uses_momentum"]
                                        else 0.0)
    out.attrs["nco_momentum_applied"] = bool(
        _spec["uses_momentum"] and _mom_sel.notna().sum() >= 2
        and (_mmom is None or (_mmom["strength"] > 0 and _mmom["ranked"] >= MMOM_MIN_RANKED)))
    # Managed Momentum's overlay, as applied today: the bear gate (1 open, 0 shut) and the
    # market return it read, the volatility scale and the readings behind it, the strength
    # that results (λ · gate · scale), how many names were ranked and how many sat at the
    # floor, and which history the overlay read. Absent for every other style.
    if _mmom is not None:
        for _k, _v in _mmom.items():
            out.attrs[f"nco_mmom_{_k}"] = _v
        out.attrs["nco_mmom_lambda"] = float(MMOM_LAMBDA)
        out.attrs["nco_mmom_floor"] = float(MMOM_FLOOR)
        out.attrs["nco_mmom_history_short"] = bool(_mmom["gate_rows"] < _mmom["history_needed"])
    # The grid, over the WHOLE allocation universe as well as the book: the states
    # every name sits in (the census), the readings behind them (for the conviction-value
    # map and the watchlist, which show names the book holds at the floor or not
    # at all), and how far the value tape's macro hedge was earned. `applied` is
    # False when no name had a reading — every name is then UNREAD at the same unit
    # (1), the book is 1/N, and the UI must say so. Beside read names 1 unit is BELOW
    # the read average (~1.29 graded units on Nifty and Dow, 1.34 on the ETF book): a
    # young listing or fund is held at ~0.78x until both its tapes calibrate (CVG-B13).
    out.attrs["nco_uses_cvg"] = _uses_dh
    if _uses_dh and dh is not None:
        _held = set(out["symbol"])
        _uni = dh.assign(symbol=list(dh.index), held=[s in _held for s in dh.index])
        _uni["weight_pct"] = _uni["symbol"].map(
            dict(zip(out["symbol"], out["weightage_pct"]))).fillna(0.0)
        _uni["units"] = cvg_units(_uni).to_numpy(dtype=float)
        _read = int((_uni["state"] != "UNREAD").sum())
        out.attrs["nco_cvg_universe"] = _uni.reset_index(drop=True)
        out.attrs["nco_cvg_names"] = _read
        out.attrs["nco_cvg_census"] = {k: int(v) for k, v in
                                          _uni["state"].value_counts().items()}
        out.attrs["nco_cvg_book_census"] = {k: int(v) for k, v in
                                               out["state"].value_counts().items()}
        # The histogram's part: how many names it could read, how many it
        # confirms each way, how many it cannot (turning or quiet), and how
        # many rows it is holding against their tape right now.
        _g = pd.to_numeric(_uni["push_gate"], errors="coerce")
        out.attrs["nco_cvg_push_read"] = int(_g.notna().sum())
        out.attrs["nco_cvg_confirm_up"] = int((_g > 0).sum())
        out.attrs["nco_cvg_confirm_down"] = int((_g < 0).sum())
        out.attrs["nco_cvg_unconfirmed"] = int((_g == 0).sum())
        out.attrs["nco_cvg_quiet"] = int(_uni["push_tier"].astype(str).str.contains("quiet").sum())
        out.attrs["nco_cvg_held"] = int((pd.to_numeric(_uni["held_row"], errors="coerce") > 0).sum())
        # The conviction ladder each name read today: DOWN (intraday rungs) or D · W ↺.
        _ld = pd.to_numeric(_uni["ladder_down"] if "ladder_down" in _uni else pd.Series(dtype=float), errors="coerce")
        out.attrs["nco_cvg_ladder_down"] = int((_ld == 1).sum())
        out.attrs["nco_cvg_ladder_up"] = int((_ld == 0).sum())
        out.attrs["nco_cvg_graded"] = CVG_GRADED
        _h = pd.to_numeric(_uni["hedge"], errors="coerce").dropna()
        out.attrs["nco_cvg_hedge_median"] = float(_h.median()) if len(_h) else float("nan")
        out.attrs["nco_cvg_applied"] = _read > 0
    else:
        out.attrs["nco_cvg_names"] = 0
        out.attrs["nco_cvg_applied"] = False
    # Full correlation matrix plus the cluster ordering, so the UI can draw the
    # correlation structure of the estimation universe. Ordering by the Ward
    # clusters is what makes the block structure visible; it is a diagnostic for
    # every style — HRP bisects its own single-linkage leaf order, not this one.
    # Absent when there was no estimable covariance to
    # draw, which the UI reads as "skip the Risk Structure section".
    if cov_ok:
        _order = [est_names[i] for i in np.argsort(labels, kind="stable")]
        out.attrs["corr_matrix"] = pd.DataFrame(
            corr, index=est_names, columns=est_names).loc[_order, _order]
        out.attrs["cluster_order"] = _order
        out.attrs["cluster_labels"] = dict(lab_by_name)
    # Stable, so ties keep the order the allocator filled them in: universe
    # order for 1/N, state-then-room for the grid.
    return out.sort_values("weightage_pct", ascending=False, kind="stable").reset_index(drop=True)


__all__ = [
    "METHODS",
    "METHOD_ORDER",
    "METHOD_SPECS",
    "method_spec",
    "MOMENTUM_LAMBDA",
    "MOMENTUM_LOOKBACK",
    "MOMENTUM_SKIP",
    "MMOM_LAMBDA",
    "MMOM_FLOOR",
    "MMOM_GATE",
    "MMOM_HISTORY_START",
    "MMOM_MIN_HISTORY",
    "mmom_windows",
    "hrp_staggered",
    "HRP_WINDOWS",
    "mmom_overlay",
    "mmom_gate",
    "mmom_scale",
    "mmom_momentum",
    "mmom_ranks",
    "CVG_FIELDS",
    "CVG_TEXT",
    "correlation_distance",
    "inverse_variance",
    "ledoit_wolf",
    "cluster_assets",
    "risk_contributions",
    "hrp_weights",
    "erc_weights",
    "cvg_readings",
    "cvg_units",
    "cvg_weights",
    "momentum_scores",
    "rank_z",
    "apply_momentum_tilt",
    "build_returns_matrix",
    "build_price_matrix",
    "compute_nco_portfolio",
]
