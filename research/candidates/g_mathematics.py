"""
research/candidates/g_mathematics.py — FAMILY G: FIVE STYLES FROM MATHEMATICS FIRST (v14 style search)

PRE-REGISTRATION (written 2026-11-22, committed before any configuration below was run on any data)
═══════════════════════════════════════════════════════════════════════════════════════════════════

A NOTE ON THE BRIEF
    The brief asked for this family "at the level of a mathematician who has solved the Riemann
    hypothesis". It has not been solved here, and nothing below depends on it: the zeros of ζ carry no
    information about which of 50 stocks to hold. What mathematics can offer is structure the earlier
    searches did not use. Those tried: Equal Weight, ERC, HRP, NCO, minimum variance, maximum
    diversification, inverse volatility (the 36-candidate search); reversal, momentum, volatility power,
    excess-growth / fractional Kelly, anomaly tilts and style ensembles (the v12.1 search, families A-E).
    Each candidate here comes from a different branch, with its source:

CANDIDATES (5 functions, 8 configurations; the declared default marked *)
    G1 rmt_erc(window)          RANDOM-MATRIX THEORY. A correlation matrix of N names from T days is mostly
                                noise below the Marchenko-Pastur edge λ₊ = (1 + √(N/T))²: eigenvalues under it
                                are replaced by their mean (trace kept; Laloux, Cizeau, Bouchaud & Potters
                                1999), the diagonal reset to 1, Σ = D C̃ D with D the sample st. dev., then
                                equal risk contribution (nco.erc_weights). window ∈ {252*, 504}.
    G2 max_enb()                MINIMUM-TORSION BETS (Meucci, Santangelo & Deguest 2015). The minimum-torsion
                                transform t turns the N correlated returns into N uncorrelated factors that
                                stay closest to the originals; a book's variance then splits into non-
                                negative shares pₖ = bₖ²·(tΣtᵀ)ₖₖ / wᵀΣw, b = t⁻ᵀw. The book maximises the
                                effective number of bets exp(−Σ pₖ ln pₖ): diversification across the
                                independent sources of risk rather than across tickers. Σ = Ledoit-Wolf on
                                252 days; SLSQP from the ERC point; 1/(2N) ≤ wᵢ ≤ 10%. One configuration.
    G3 network_periphery(lam)   CORRELATION NETWORKS (Pozzi, Di Matteo & Aste 2013: the periphery of a stock
                                correlation network outperformed its centre, at lower risk). On the minimum
                                spanning tree of d = √(2(1 − ρ)) over 252 days (scipy.sparse.csgraph), a
                                hybrid centrality hᵢ = the mean rank of MST degree, MST closeness (inverse
                                mean path length) and eigenvector centrality of ρ⁺ = max(ρ, 0). Weights
                                wᵢ ∝ exp(λ·zᵢ), z = nco.rank_z(−h) (centred ranks, unit spread): toward the
                                periphery. λ ∈ {0.25, 0.5*}.
    G4 exp_gradient(eta)        ONLINE LEARNING (Helmbold, Schapire, Singer & Warmuth 1998; Cover 1991). The
                                exponentiated-gradient portfolio, updated on every past month's price
                                relatives xᵢ from the first month of the panel: wᵢ ← wᵢ·exp(η·xᵢ / w·x),
                                renormalised. Its log-wealth trails the best constant-rebalanced book in
                                hindsight by at most O(√(T log N)) — a guarantee no estimator carries. A name
                                entering is admitted at the mean weight; a name with no relative that month
                                is given the book's own (no update). η ∈ {0.05, 0.3*} — 0.3 is the bound's
                                own order, 2·x_min·√(2 ln N / T) ≈ 0.3 at x_min ≈ 0.7, N = 50, T ≈ 150.
    G5 spectral_parity()        SPECTRAL GRAPH THEORY (Ng, Jordan & Weiss 2001; von Luxburg 2007). Affinity
                                A = max(ρ, 0) on 252 days (zero diagonal); normalised Laplacian
                                L = I − D^−½ A D^−½; k ∈ [2, 8] at the largest eigengap; the k leading
                                eigenvectors row-normalised and k-means'd (sklearn, n_init 10, seed 0). Capital
                                splits equally across the k clusters, equally within: parity over the
                                universe's actual independent groups (~3 here: the eigenvalue participation
                                ratio is 2.95), not over its tickers. One configuration.

FIXED DESIGN (declared, not tuned)
    * The estimation set: the priced names with ≥ 95% of the window (nco.MIN_COVERAGE), rows with any gap
      dropped; a priced name outside it is held at 1/N and the rest share the remaining mass. Fewer than
      10 names in the set (or T < N): the whole book is 1/N that month.
    * G4 reads only price relatives between the harness's own rebalance dates, up to the rebalance date.
    * Weights are raw; the harness renormalises and applies the 10% cap and (for top-N) the cut.

PRE-STATED EXPECTATION
    G1 and G2 are risk allocators on better risk estimates: expect ERC-like books, lower risk than 1/N and a
    return cost, i.e. the known tension's low-risk end. G3 leans to low-correlation names, which in
    2007-19 includes today's late entrants (survivorship, as C2 found). G4 is a slow momentum learner and
    will be flattered by survivorship; the point-in-time Dow is its real test. G5 is the most likely to
    sit near 1/N: three clusters of roughly equal size give each name about its equal share.

RESULT (2026-11-22; tables in research/style_search_v14.py) — ALL FIVE REJECTED.
    G1 and G2 behaved as risk allocators should: lower volatility than 1/N (Nifty 20.3 and 19.9 against
    22.2), at a return cost in most cells; G2 has Nifty's best return per unit of risk after HRP (1.02
    against 1.05). G3 came closest — 3/4 discovery cells at λ = 0.5, the finalist (λ = 0.25) 2/4, +0.45 %/yr
    over 1/N on Nifty over the full span — but lost every E3 cell and the point-in-time Dow. G4 at η = 0.05
    barely leaves 1/N (+0.04 %/yr, t 0.69, the search's largest). G5 won 3/4 discovery cells and then lost
    all three 2020+ cells (−2.68 Nifty, −1.27 Dow, −0.88 point-in-time Dow): the overfit the holdout is for.
    None reaches bar 3; the mathematics is sound and the universes are too correlated for any weighting
    rule to move the book far from 1/N (every finalist 0.98-1.00 correlated with it, month by month).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))                    # research/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root

import nco                                                   # noqa: E402
from f_pragyam import MIN_SET, assemble, equal, returns_set  # noqa: E402

CAP = 0.10
_EG_CACHE: dict = {}


def _corr(R: pd.DataFrame) -> np.ndarray:
    C = np.corrcoef(R.to_numpy(dtype=float), rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return (C + C.T) / 2.0


def _usable(keep, R) -> bool:
    return len(keep) >= MIN_SET and len(R) >= max(nco.MIN_OBS, len(keep))


# ── G1 ─────────────────────────────────────────────────────────────────────────────────────────────
def mp_clip(C: np.ndarray, T: int) -> np.ndarray:
    """Marchenko-Pastur eigenvalue clipping (Laloux et al. 1999): trace kept, unit diagonal restored."""
    N = C.shape[0]
    lam_plus = (1.0 + np.sqrt(N / T)) ** 2
    vals, vecs = np.linalg.eigh(C)
    noise = vals < lam_plus
    if noise.any():
        vals = vals.copy()
        vals[noise] = vals[noise].mean()
    Ct = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(Ct), 1e-20, None))
    Ct = Ct / np.outer(d, d)
    return (Ct + Ct.T) / 2.0


def rmt_erc(ctx, window: int = 252) -> pd.Series:
    keep, R, out = returns_set(ctx, window)
    if not _usable(keep, R):
        return equal(ctx)
    sd = R.std(ddof=1).to_numpy(dtype=float)
    Ct = mp_clip(_corr(R), len(R))
    return assemble(ctx, keep, nco.erc_weights(Ct * np.outer(sd, sd)), out)


# ── G2 ─────────────────────────────────────────────────────────────────────────────────────────────
def min_torsion(S: np.ndarray, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
    """Meucci, Santangelo & Deguest (2015), Appendix: the minimum-torsion matrix t, tΣtᵀ diagonal."""
    sig = np.sqrt(np.diag(S))
    C = S / np.outer(sig, sig)
    c = np.real(sqrtm(C))
    n = len(sig)
    d = np.ones(n)
    f_prev = None
    pi = np.eye(n)
    for _ in range(max_iter):
        U = np.diag(d) @ c @ c @ np.diag(d)
        u = np.real(sqrtm(U))
        q = np.linalg.solve(u, np.diag(d) @ c)
        d = np.diag(q @ c)
        pi = np.diag(d) @ q
        f = np.linalg.norm(c - pi, "fro")
        if f_prev is not None and abs(f - f_prev) / max(f, 1e-30) / n <= tol:
            break
        f_prev = f
    x = pi @ np.linalg.inv(c)
    return np.diag(sig) @ x @ np.diag(1.0 / sig)


def enb_shares(w: np.ndarray, S: np.ndarray, t: np.ndarray, t_inv_T: np.ndarray, fvar: np.ndarray) -> np.ndarray:
    b = t_inv_T @ w
    v = float(w @ S @ w)
    return (b * b * fvar) / v if v > 0 else np.full(len(w), 1.0 / len(w))


def max_enb(ctx) -> pd.Series:
    keep, R, out = returns_set(ctx, 252)
    if not _usable(keep, R):
        return equal(ctx)
    S = nco.ledoit_wolf(R.to_numpy(dtype=float))
    n = len(keep)
    t = min_torsion(S)
    t_inv_T = np.linalg.inv(t).T
    fvar = np.diag(t @ S @ t.T)

    def neg_entropy(w):
        p = np.clip(enb_shares(w, S, t, t_inv_T, fvar), 1e-300, None)
        p = p / p.sum()
        return float(np.sum(p * np.log(p)))

    x0 = nco.erc_weights(S)
    lo = 1.0 / (2 * n)
    x0 = np.clip(x0, lo, CAP)
    x0 = x0 / x0.sum()
    res = minimize(neg_entropy, x0, method="SLSQP", bounds=[(lo, CAP)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 300, "ftol": 1e-10})
    w = res.x if res.success and np.isfinite(res.x).all() else x0
    return assemble(ctx, keep, w, out)


# ── G3 ─────────────────────────────────────────────────────────────────────────────────────────────
def hybrid_centrality(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    D = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))
    np.fill_diagonal(D, 0.0)
    D = np.where((D <= 0) & ~np.eye(n, dtype=bool), 1e-9, D)     # csgraph reads 0 as "no edge"
    mst = minimum_spanning_tree(D).toarray()
    mst = mst + mst.T
    degree = (mst > 0).sum(axis=1).astype(float)
    paths = shortest_path(mst, directed=False)
    closeness = (n - 1) / np.where(paths.sum(axis=1) > 0, paths.sum(axis=1), np.nan)
    A = np.clip(C, 0.0, None)
    np.fill_diagonal(A, 0.0)
    vals, vecs = np.linalg.eigh(A)
    eig = np.abs(vecs[:, -1])
    ranks = [pd.Series(x).rank(pct=True).to_numpy() for x in (degree, np.nan_to_num(closeness), eig)]
    return np.mean(ranks, axis=0)


def network_periphery(ctx, lam: float = 0.5) -> pd.Series:
    keep, R, out = returns_set(ctx, 252)
    if not _usable(keep, R):
        return equal(ctx)
    h = hybrid_centrality(_corr(R))
    z = nco.rank_z(pd.Series(-h, index=keep)).to_numpy(dtype=float)
    return assemble(ctx, keep, np.exp(lam * z), out)


# ── G4 ─────────────────────────────────────────────────────────────────────────────────────────────
def eg_path(d: dict, eta: float) -> dict:
    """{rebalance date: EG weights after every relative up to that date}, built once per panel and η."""
    key = (d["u"], len(d["months"]), round(eta, 6))
    if key in _EG_CACHE:
        return _EG_CACHE[key]
    px, months = d["px"], d["months"]
    out, w = {}, pd.Series(dtype=float)
    for i, a in enumerate(months):
        pa = px.loc[a]
        names = pa.index[pa.notna() & (pa > 0)]
        if i > 0:
            prev = months[i - 1]
            pp_ = px.loc[prev]
            held = w.index[w > 0]
            x = (pa.reindex(held) / pp_.reindex(held)).replace([np.inf, -np.inf], np.nan)
            wx = float((w[held] * x.fillna(0.0)).sum() / max(w[held][x.notna()].sum(), 1e-12)) if x.notna().any() else 1.0
            x = x.fillna(wx)                                   # no relative → the book's own: no update
            w = w[held] * np.exp(eta * x / wx)
        w = w.reindex(names)
        new = w.isna()
        if new.any():
            w[new] = w[~new].mean() if (~new).any() else 1.0
        w = w / w.sum()
        out[a] = w.copy()
    _EG_CACHE[key] = out
    return out


def exp_gradient(ctx, eta: float = 0.3) -> pd.Series:
    w = eg_path(ctx._d, eta).get(ctx.date)
    if w is None or w.empty:
        return equal(ctx)
    return w.reindex(ctx.priced).fillna(w.mean())


# ── G5 ─────────────────────────────────────────────────────────────────────────────────────────────
def spectral_labels(C: np.ndarray, kmax: int = 8) -> np.ndarray:
    from sklearn.cluster import KMeans
    n = C.shape[0]
    A = np.clip(C, 0.0, None)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0))
    L = np.eye(n) - (A * dinv[:, None]) * dinv[None, :]
    vals, vecs = np.linalg.eigh(L)
    ks = range(2, min(kmax, n - 1) + 1)
    k = max(ks, key=lambda j: vals[j] - vals[j - 1])          # gap after the j-th smallest eigenvalue
    U = vecs[:, :k]
    U = U / np.clip(np.linalg.norm(U, axis=1, keepdims=True), 1e-12, None)
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(U)


def spectral_parity(ctx) -> pd.Series:
    keep, R, out = returns_set(ctx, 252)
    if not _usable(keep, R):
        return equal(ctx)
    lab = spectral_labels(_corr(R))
    k = len(np.unique(lab))
    sizes = pd.Series(lab).value_counts()
    w = np.array([1.0 / (k * sizes[c]) for c in lab])
    return assemble(ctx, keep, w, out)


CANDIDATES = {
    "rmt_erc": (rmt_erc, {"window": [252, 504]}),
    "max_enb": (max_enb, {}),
    "network_periphery": (network_periphery, {"lam": [0.25, 0.5]}),
    "exp_gradient": (exp_gradient, {"eta": [0.05, 0.3]}),
    "spectral_parity": (spectral_parity, {}),
}
DEFAULTS = {"rmt_erc": {"window": 252}, "max_enb": {}, "network_periphery": {"lam": 0.5},
            "exp_gradient": {"eta": 0.3}, "spectral_parity": {}}
