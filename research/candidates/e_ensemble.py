"""
research/candidates/e_ensemble.py — FAMILY E: ensembles, style rotation and regime-conditioned blends.

PRE-REGISTRATION (written 2026-10-03, before any candidate below was run)
─────────────────────────────────────────────────────────────────────────
WHY THIS FAMILY. A static blend of the shipped styles lands at its members' midpoint (measured,
style_blends.py), so it cannot beat its best member. The best existing style differs by cell —
HRP (low risk) won Nifty 50 2007-13, a crash-and-recovery era; CVG won both 2014-19 cells and is
0.2 %/yr behind EW in Dow 30 2007-13. Only a TIME-VARYING mix of the books can, in principle, beat
the best member in every cell: hold the low-risk book (HRP) when the market is stressed and the
return book (CVG) otherwise, or hold whichever style has been winning. Every candidate here is
built only from the shipped styles' raw weights (ctx.raw) plus one causal market or performance
signal; nothing inside a style is changed.

LITERATURE BASIS
    Faber, M. (2007), "A Quantitative Approach to Tactical Asset Allocation", Journal of Wealth
        Management 9(4): a 10-month simple moving average (≈ the 200-day) as a trend filter
        across asset classes since 1901, cutting drawdowns at a small cost in return.
    Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012), "Time Series Momentum", Journal of
        Financial Economics 104(2): the 12-month look-back as the canonical trend signal.
    Ang, A. & Bekaert, G. (2004), "How Regimes Affect Asset Allocation", Financial Analysts
        Journal 60(2): equity returns have a high-volatility, high-correlation, bear regime; a
        regime-switching allocation dominated static ones out of sample in their data.
    Moreira, A. & Muir, T. (2017), "Volatility-Managed Portfolios", Journal of Finance 72(4):
        scale exposure by c / (last month's realized variance). Used here ONLY as a relative tilt
        between the return book and the low-risk book, since the book must stay fully invested.
    Barroso, P. & Santa-Clara, P. (2015), "Momentum Has Its Moments", Journal of Financial
        Economics 116(1): the same scaling on a 6-month (126-day) realized-variance window.
    Cederburg, S., O'Doherty, M., Wang, F. & Yan, X. (2020), "On the Performance of
        Volatility-Managed Portfolios", Journal of Financial Economics 138(1): with a real-time
        scaling constant, volatility management mostly FAILS out of sample (103 strategies) — the
        reason the scaling constant here is the expanding-window median, never a full-sample one.
    Gupta, T. & Kelly, B. (2019), "Factor Momentum Everywhere", Journal of Portfolio Management
        45(3): factors are timed by their own past return, 1- to 12-month look-backs.
    Ehsani, S. & Linnainmaa, J. (2022), "Factor Momentum and the Momentum Factor", Journal of
        Finance 77(3): factors are autocorrelated (the average factor earns ~1 bp/month after a
        losing year vs ~53 bp after a winning year); 12-month formation.
    Arnott, R., Clements, M., Kalesnik, V. & Linnainmaa, J. (2023), "Factor Momentum", Review of
        Financial Studies 36(8): cross-sectional factor momentum (buy recent winners among
        factors) subsumes industry momentum; strongest at short (1-month) formation.
    Daniel, K. & Moskowitz, T. (2016), "Momentum Crashes", Journal of Financial Economics
        122(2): momentum-type rules crash in post-panic rebounds — the main failure mode of a
        rule that holds last year's winner (here: HRP after a crash) into the recovery.
    Asness, C., Chandra, S., Ilmanen, A. & Israel, R. (2017), "Contrarian Factor Timing is
        Deceptively Difficult", Journal of Portfolio Management 43(5): value-timing of factors
        adds little and can hurt a diversified multi-style book. Timmermann, A. (2006), "Forecast
        Combinations", Handbook of Economic Forecasting 1, and DeMiguel, Garlappi & Uppal (2009),
        "Optimal Versus Naive Diversification", Review of Financial Studies 22(5): estimated,
        time-varying combination weights rarely beat equal weights out of sample.
    PRIOR: the evidence that style/factor timing survives out of sample is weak (Asness et al.
    2017; Cederburg et al. 2020; DeMiguel et al. 2009). The prior that any candidate here beats
    all four discovery cells, let alone the holdout, is low.

MECHANISM. HRP beats CVG/EW when the market falls (low-volatility names fall less) and lags them
in rallies. If stress is persistent enough to be detected causally (trend breaks, volatility
clusters — Ang & Bekaert; Moreira & Muir), a rule that holds HRP only in the stressed months keeps
most of HRP's crash protection and most of CVG's upside, which neither book does alone. Style
momentum gets there without a market model: if the HRP-vs-CVG-vs-EW spread is autocorrelated
(Ehsani & Linnainmaa; Arnott et al.), the recent winner keeps winning. Switching costs turnover
(HRP ≈ 1.1x/yr, CVG ≈ 1.5x/yr on their own); a full HRP↔CVG switch costs roughly half a unit of
one-way turnover (≈ 5 bp India, ≈ 1.5 bp US).

COMMON CONVENTIONS (fixed now)
    • Return book R = CVG; low-risk book L = HRP (the shipped raw weights, ctx.raw).
    • Market signal = the equal-weighted index of the universe, built inside fn from ctx.px only:
      the daily cross-sectional mean of the names' close-to-close returns (pct_change without
      fill; a name without both closes is skipped that day), chain-linked.
    • A hard switch returns ctx.raw["HRP"] or ctx.raw["CVG"] unchanged, so in a given month it
      IS that shipped book. A partial mix (V_REGIME only) blends the two CAPPED books,
      λ·cut(HRP) + (1−λ)·cut(CVG) (style_blends "·b" reading), so the mix does not leak HRP's
      uncapped raw weights past the 10% cap.
    • Until a candidate's signal window is full, it holds R (CVG). No signal = no timing.

CANDIDATES (3; 3-point grids; 9 configurations in all)
 1. T_SWITCH — trend regime switch (Faber 2007; Moskowitz-Ooi-Pedersen 2012).
        risk-off ⇔ EW-index close on ctx.date < its simple moving average over the last
        L × 21 trading days. Risk-off → HRP book; risk-on → CVG book.
        Grid: L ∈ {6, 10, 12} months   (10 = Faber's default; 12 = MOP's look-back).
        Plain function t_switch(ctx, L).
 2. V_REGIME — volatility-managed relative tilt (Moreira & Muir 2017; Barroso & Santa-Clara
        2015; real-time constant per Cederburg et al. 2020; regime reading per Ang & Bekaert 2004).
        σ²_t = annualised mean squared daily EW-index return over the last W trading days;
        c_t  = median of that rolling series over ALL history ≤ ctx.date (expanding, real time).
        Weight on CVG λ_R = min(1, c_t / σ²_t); weight on HRP 1 − λ_R. Calm months (σ² ≤ median)
        hold pure CVG; stressed months shade into HRP in proportion to the variance spike.
        Requires ≥ 21 values of the rolling series, else CVG.
        Grid: W ∈ {21, 63, 126} trading days (21 = Moreira-Muir's previous month;
        126 = Barroso-Santa-Clara's 6 months). Plain function v_regime(ctx, W).
 3. S_MOM — style momentum, cross-sectional (Arnott et al. 2023; Gupta & Kelly 2019; Ehsani &
        Linnainmaa 2022). Each month hold the ONE book among {EW, HRP, CVG} with the highest
        trailing K-month return, its gross monthly returns rebuilt from the stored raw weights
        of PAST rebalances only (option (ii) of the brief): for each completed period
        m_i → m_{i+1} with m_{i+1} ≤ ctx.date, the capped book sb.cut(d["raw"][s][m_i]) is priced
        from ctx.px (which never extends past ctx.date). Ties → CVG, then EW. Fewer than K
        completed months → CVG.
        Grid: K ∈ {1, 6, 12} months (1 = Arnott et al.'s short formation; 12 = Ehsani-Linnainmaa).
        FACTORY: make_s_mom(d, K) → fn(ctx). Pass it the same data dict given to ss.run — the
        discovery dict here, the full dict in the holdout run; it reads d["months"] and
        d["raw"][s][m] only for m < ctx.date (asserted).

DECISION (from the brief): finalists = configurations that beat the best existing style in all
four discovery cells (Nifty/Dow × E1/E2), ranked by smallest margin; if none, the single
configuration with the largest smallest-margin. At most 2. No tuning beyond the grids above; any
later addition is marked POST-HOC and counted.

Usage:
    import sys; sys.path.insert(0, "/home/user/Pragyam/research"); sys.path.insert(0, "/home/user/Pragyam/research/candidates")
    import style_search as ss, e_ensemble as E
    d = ss.load("nifty_50")
    r = ss.run(E.build(d, "T_SWITCH", L=10), d)          # build() handles plain functions and factories

RESULT (2026-10-03, discovery only) — NO configuration beats the best existing style in all four
cells. 9 configurations declared, 9 run, on both universes (every-name book); none added post hoc.
Net CAGR % and margin vs the best existing style in that cell (bar: Nifty E1 20.21 HRP, Nifty E2
19.68 CVG, Dow E1 14.39 EW, Dow E2 18.50 CVG):

                      CAGR                               margin vs best                     min
                      Nif-E1 Nif-E2 Dow-E1 Dow-E2        Nif-E1 Nif-E2 Dow-E1 Dow-E2      margin
    T_SWITCH(L=6)     18.89  17.73  14.81  17.54         -1.32  -1.94  +0.42  -0.96      -1.94
    T_SWITCH(L=10)    18.08  18.11  13.77  17.97         -2.13  -1.57  -0.61  -0.53      -2.13
    T_SWITCH(L=12)    17.98  18.57  13.47  17.97         -2.23  -1.11  -0.92  -0.53      -2.23
    V_REGIME(W=21)    18.23  19.02  14.10  18.00         -1.98  -0.66  -0.29  -0.49      -1.98
    V_REGIME(W=63)    19.80  19.23  13.54  17.83         -0.41  -0.45  -0.85  -0.66      -0.85
    V_REGIME(W=126)   19.46  19.50  13.13  18.32         -0.75  -0.18  -1.26  -0.18      -1.26
    S_MOM(K=1)        18.97  19.89  13.87  17.83         -1.24  +0.21  -0.52  -0.67      -1.24
    S_MOM(K=6)        19.48  18.30  12.24  17.68         -0.73  -1.37  -2.14  -0.82      -2.14
    S_MOM(K=12)       18.39  16.94  12.82  18.39         -1.82  -2.74  -1.57  -0.10      -2.74

    vol %/yr (Nif-E1/E2, Dow-E1/E2), maxDD % (same), turnover x/yr (same):
    T_SWITCH(L=6)     27.1/14.6/16.9/12.5   -51.9/-16.2/-35.9/-11.3   1.65/1.73/1.47/1.38
    T_SWITCH(L=10)    26.9/14.7/17.0/12.6   -53.7/-16.9/-35.9/-11.3   1.57/1.57/1.41/1.31
    T_SWITCH(L=12)    26.9/14.7/17.0/12.6   -53.7/-16.9/-36.8/-11.3   1.58/1.55/1.43/1.31
    V_REGIME(W=21)    28.2/14.8/17.4/12.4   -56.3/-15.3/-36.8/-10.1   1.60/1.44/1.32/1.26
    V_REGIME(W=63)    28.6/14.9/17.5/12.5   -54.8/-15.9/-37.2/-10.5   1.47/1.39/1.28/1.20
    V_REGIME(W=126)   28.4/15.0/17.7/12.4   -55.1/-15.6/-37.6/-11.1   1.41/1.39/1.25/1.16
    S_MOM(K=1)        26.0/14.5/16.7/11.7   -52.7/-14.4/-37.5/ -9.5   2.21/1.99/1.98/1.60
    S_MOM(K=6)        26.2/14.3/17.0/12.0   -52.6/-14.1/-38.2/ -9.5   1.68/1.55/1.42/0.97
    S_MOM(K=12)       25.4/13.8/17.0/12.2   -54.4/-12.6/-36.8/-11.3   1.50/1.50/1.30/0.72

    Time in the low-risk book: T_SWITCH 13-26% of months (14-32 switches over 155 months);
    V_REGIME(W=63) average HRP weight 0.14/0.04 (Nifty E1/E2), 0.21/0.13 (Dow); S_MOM holds HRP
    23-50% of months, K=1 switching 92-102 times.

FINALIST (rule: none passes all four → the single configuration with the largest smallest
margin): V_REGIME, W=63 — v_regime(ctx, W=63), plain function, this file. Smallest margin
-0.85 %/yr (Dow E1); paired t vs the best style per cell +0.32 / -1.66 / -1.06 / -1.82. It is
a FAILED candidate carried forward only because the protocol names one; it does not beat the bar
in any discovery cell. Full span: Nifty 19.53 %/yr, vol 23.2, maxDD -54.8, turnover 1.43;
Dow 15.52 %/yr, vol 15.3, maxDD -37.2, turnover 1.24. (Runner-up, not a finalist: S_MOM(K=1),
min margin -1.24, the only configuration to beat the bar in Nifty E2, +0.21.)

WHY IT FAILED. (1) The premise is only half true. HRP's Nifty E1 lead over CVG (+1.05 %/yr) did
not come from the crash alone: by calendar year HRP beat CVG by 7 pts in 2008 but lost 24 pts in
2009, and won 2010 (+6.6), 2011 (+4.3) and 2013 (+5.4) — calm or mildly weak years that no market
trend or volatility state flags. (2) Every switch rule pays the Daniel-Moskowitz rebound cost: the
trend filter entered HRP late in 2008 (-47.5 vs HRP -43.7), held it into the 2009 rebound (+128.8
vs CVG +140.7) and one whipsaw month cost 10 pts in 2012 (39.8 vs CVG 50.2); the Dow switch
lost 7 pts to CVG in 2009. (3) In Dow 30 HRP trails CVG/EW by 2-3 %/yr in both eras, so any time
spent there must be very well timed; only the fast T_SWITCH(L=6) cleared Dow E1 (+0.42) — and it
is the worst in both Nifty cells. (4) Style momentum on three books whose monthly returns correlate 0.97-0.998 is
mostly noise: K=1 turns over twice a year and wins only Nifty E2. Consistent with Asness et al.
(2017) and Cederburg et al. (2020): the timing signal is too weak to pay for being wrong at turns.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style_blends as sb                                   # noqa: E402

RISK_ON, RISK_OFF = "CVG", "HRP"
STYLES_MOM = ("CVG", "EW", "HRP")                           # tie order: CVG, then EW, then HRP


# ── shared pieces ────────────────────────────────────────────────────────────────────────────
def _ew_index_returns(px: pd.DataFrame) -> pd.Series:
    """Daily returns of the equal-weighted universe index, from closes ≤ ctx.date only."""
    r = px.pct_change(fill_method=None).iloc[1:]
    return r.mean(axis=1, skipna=True).dropna()


def _capped(w: pd.Series) -> pd.Series:
    return sb.cut(pd.Series(w, dtype=float).sort_values(ascending=False, kind="stable"), None)


def _mix(ctx, lam_off: float) -> pd.Series:
    """λ_off·cut(HRP) + (1−λ_off)·cut(CVG); a pure end returns the shipped raw book unchanged."""
    if lam_off <= 0.0:
        return ctx.raw[RISK_ON]
    if lam_off >= 1.0:
        return ctx.raw[RISK_OFF]
    h, c = _capped(ctx.raw[RISK_OFF]), _capped(ctx.raw[RISK_ON])
    idx = h.index.union(c.index)
    return lam_off * h.reindex(idx, fill_value=0.0) + (1.0 - lam_off) * c.reindex(idx, fill_value=0.0)


# ── 1. T_SWITCH ──────────────────────────────────────────────────────────────────────────────
def t_switch(ctx, L: int = 10) -> pd.Series:
    """Faber (2007) 10-month-SMA filter on the EW index: below → HRP book, above → CVG book."""
    ri = _ew_index_returns(ctx.px)
    n = int(L) * 21
    if len(ri) < n:
        return ctx.raw[RISK_ON]
    lvl = (1.0 + ri).cumprod()
    off = lvl.iloc[-1] < lvl.iloc[-n:].mean()
    return ctx.raw[RISK_OFF] if off else ctx.raw[RISK_ON]


# ── 2. V_REGIME ──────────────────────────────────────────────────────────────────────────────
def v_regime(ctx, W: int = 21) -> pd.Series:
    """Moreira-Muir relative tilt: weight on CVG = min(1, real-time median variance / recent variance)."""
    ri = _ew_index_returns(ctx.px)
    rv = (ri ** 2).rolling(int(W)).mean().dropna() * 252.0
    if len(rv) < 21 or rv.iloc[-1] <= 0:
        return ctx.raw[RISK_ON]
    lam_on = min(1.0, float(rv.median()) / float(rv.iloc[-1]))
    return _mix(ctx, 1.0 - lam_on)


# ── 3. S_MOM (factory) ───────────────────────────────────────────────────────────────────────
def make_s_mom(d: dict, K: int = 12):
    """Hold the best trailing-K-month book among EW/HRP/CVG. Reads d['raw'] only at past rebalances."""
    months = list(d["months"])
    pos = {m: i for i, m in enumerate(months)}
    cache: dict = {}

    def period_ret(ctx, s: str, i: int) -> float:
        a, b = months[i], months[i + 1]
        assert a < ctx.date and b <= ctx.date, "look-ahead"
        key = (s, i)
        if key not in cache:
            w = _capped(d["raw"][s][a])
            r = (ctx.px.loc[b].reindex(w.index) / ctx.px.loc[a].reindex(w.index) - 1.0).fillna(0.0)
            cache[key] = float((w * r).sum())
        return cache[key]

    def fn(ctx) -> pd.Series:
        j = pos[ctx.date]
        if j < int(K):
            return ctx.raw[RISK_ON]
        perf = {s: float(np.prod([1.0 + period_ret(ctx, s, i) for i in range(j - int(K), j)]))
                for s in STYLES_MOM}
        best = max(STYLES_MOM, key=lambda s: (perf[s], -STYLES_MOM.index(s)))
        return ctx.raw[best]

    return fn


CANDIDATES = {
    "T_SWITCH": (t_switch, {"L": [6, 10, 12]}),             # plain function
    "V_REGIME": (v_regime, {"W": [21, 63, 126]}),           # plain function
    "S_MOM":    (make_s_mom, {"K": [1, 6, 12]}),            # FACTORY: make_s_mom(d, K) -> fn
}
FACTORIES = {"S_MOM"}


def build(d: dict, name: str, **params):
    """fn(ctx) for one configuration — works on the discovery or the full data dict alike."""
    f, _ = CANDIDATES[name]
    if name in FACTORIES:
        return f(d, **params)
    return lambda ctx: f(ctx, **params)
