"""
research/candidates/b_momentum.py — FAMILY B: MOMENTUM DONE RIGHT (style search, research/style_search.py)

PRE-REGISTRATION (written 2026-10-03, BEFORE any candidate below was run on any data)
═══════════════════════════════════════════════════════════════════════════════════════

WHAT IS ALREADY KNOWN HERE (from the brief; not re-measured)
    Plain 12-1 momentum as a tercile tilt (3 / 1 / 0.25 units) was inconsistent: it beat CVG on Nifty in
    2014-19 but lost -2.82 %/yr before 2014 (the 2008 crash / 2009 rebound era). Momentum-styled CVG maps
    lost to CVG in every era. Cross-sectional ICs here are ~0-0.04. CVG's edge over EW is small and
    reversion-flavoured (it overweights capitulation: cheap names with sellers in control). So the
    question is not "does momentum pay here" (plain momentum: no, not consistently) but whether the
    literature's refinements, which target the specific reasons plain momentum fails, change that.

LITERATURE BASIS
    [1] Blitz, Huij & Martens (2011), "Residual momentum", J. Empirical Finance 18(3) 506-521. Total-return
        momentum carries large time-varying factor exposures; ranking on residual returns (36-month factor
        regression; score = residual return over months t-12..t-2 divided by its st.dev.) earns ~2x the
        risk-adjusted profit, is more consistent over time and less concentrated in the extremes.
    [2] Gutierrez & Prinsky (2007), "Momentum, reversal, and the trading behaviors of institutions",
        J. Financial Markets 10(1) 48-75. Momentum in ABNORMAL (residual) returns persists; momentum in
        RELATIVE (total) returns reverses -> residual momentum should be less reversal-contaminated.
    [3] Chaves (2016), "Idiosyncratic momentum: U.S. and international evidence", J. Investing. A
        MARKET-ONLY (CAPM) residual is enough; the effect holds in 21 countries, including Japan where plain
        momentum fails. (Justifies the single-factor residual used here: no Fama-French factors exist in
        ctx, and ctx is the only data a candidate may read.)
    [4] Grundy & Martin (2001), RFS: the dynamic beta of total-return momentum is what loses on market
        turns; [5] Daniel & Moskowitz (2016), "Momentum crashes", JFE 122(2) 221-247: crashes are partly
        forecastable — they occur in "panic" states, after market declines (bear = past 2-year market return
        < 0) and when volatility is high, contemporaneous with rebounds (past-loser betas > 3, winner betas
        < 0.5); a dynamic strategy roughly doubles the Sharpe ratio; robust internationally.
    [6] Barroso & Santa-Clara (2015), "Momentum has its moments", JFE 116(1) 111-120: momentum's risk is
        predictable from its own realised variance (daily, previous 6 months); scaling exposure inversely
        to it nearly eliminates crashes and nearly doubles the Sharpe ratio.
    [7] George & Hwang (2004), "The 52-week high and momentum investing", J. Finance 59(5) 2145-2176:
        nearness to the 52-week high (PTH = P / 52-week max) dominates past returns in forecasting, and its
        forecasts do NOT reverse in the long run (anchoring/under-reaction, not over-reaction).
        Liu, Liu & Ma (2011), J. Int. Money & Finance 30(1) 180-204: the 52-week-high, the recent-past and
        the intermediate (12-7) effects are all prevalent across international markets.
    [8] Novy-Marx (2012), JFE: 12-7 dominates 6-2 in the US — but Goyal & Wahal (2015), JFQA 50(6): no
        robust echo in 37 non-US countries (US echo is carry-over of month-2 reversal). => 12-7 NOT chosen.
    [9] Asness, Moskowitz & Pedersen (2013), "Value and momentum everywhere", J. Finance 68(3): value and
        momentum are negatively correlated in every market studied; a combination dominates either alone.
        This is the case for momentum ON TOP OF CVG (a value/capitulation-flavoured book) rather than
        instead of it.
    India: Sehgal & Jain (2011, J. Advances in Management Research 8(1)): momentum profits in India,
        stronger at 6-6 than 12-12, partly sectoral, unexplained by CAPM/FF; Ansari & Khan (2012, Managerial
        Finance 38(2)): strong momentum in India 1995-2006, linked to idiosyncratic risk (behavioural).
        Practitioner evidence (2004-2023) reports a 52-week-high premium in Indian equities. These are broad
        cross-sections; momentum is known to be weaker in large caps (Hong, Lim & Stein 2000, JF), and
        Nifty 50 / Dow 30 are the largest of large caps.
    Decay: McLean & Pontiff (2016, J. Finance 71(1)): anomaly returns are ~26% lower out-of-sample and ~58%
        lower post-publication. Momentum (1993), 52WH (2004), residual (2011), managed (2015/16) all
        pre-date most or all of the discovery window's end, so the 2020+ holdout is post-publication for all.

MECHANISM — why each might beat BOTH 1/N and CVG here
    The bar needs +1.05 %/yr over CVG on Nifty 2007-13 (HRP's crash-era win) while not losing to CVG in
    either 2014-19 cell nor to EW on Dow 2007-13 (+0.22 over CVG). All three candidates therefore ADD a
    zero-sum momentum overlay to CVG's own weights, so the book keeps CVG's capitulation/value bets and
    adds an (ideally negatively correlated, [9]) momentum bet. Each candidate targets one stated failure
    of plain momentum here:
      A resid_mom   — the 2009-type crash and the reversal contamination come from the total-return
                      signal's market beta [1][2][4]; strip beta, keep the stock-specific trend.
      B high52      — the noise/over-reaction problem: an anchoring signal whose forecasts don't reverse
                      [7]; in drawdowns the names nearest their highs are the resilient ones (protection the
                      Nifty E1 cell rewards).
      C managed_mom — keep the classic 12-1 signal but switch it off / shrink it exactly in the states where
                      momentum crashes: DM bear-market gate [5] x Barroso-Santa-Clara volatility scaling [6].

COMMON CONSTRUCTION (fixed)
    c        = CVG raw weights (ctx.raw["CVG"]) over ctx.priced, normalised to sum 1 (CVG covers every
               priced name in both universes).
    score    = the candidate's signal from ctx.px only (closes up to and including ctx.date).
    u        = centred cross-sectional rank of score in [-1, +1] over names with a valid score (needs >= 10
               valid names; a name without a valid score — too little history — gets u = 0, i.e. CVG weight).
    w        = max(0, c + lam_t * u / N), N = number of priced names. The harness renormalises, applies
               the 10% cap, holds monthly, charges 10bp / 3bp per unit one-way turnover.
    Additive (not multiplicative) so the momentum bet is independent of CVG's bet [9]; on a 1/N base it
    is identical to w = (1/N)(1 + lam*u). Expected active return ~ lam * 0.58 * cs-vol * IC, so lam=1 with
    IC 0.03 is ~1.5-2 %/yr: lam in {0.5, 1, 2} spans "too weak to matter" to "strong but long-only".
    Data constraint: ctx.px starts 2006-10-23, so every 12-month signal first exists ~Nov 2007; before
    that each candidate IS CVG (u = 0). Universes are today's constituents (survivorship applies to all
    styles alike; it flatters trend-following somewhat — survivors are past winners).
    Returns are close-to-close daily from ctx.px; the "market" is the equal-weighted average daily return
    of the names priced that day (the universe's own EW index).

CANDIDATES (3) — literature-default parameters; grid = lam only, 3 points each => 9 configurations
    A resid_mom(ctx, lam)   [1][2][3]. Daily CAPM regression r_i = a_i + b_i*mkt + e over the trailing
        756 trading days (36 months; expanding with a 252-day minimum while history is shorter).
        Score = sum(e) / std(e) over the formation window = returns of days t-251 .. t-21 (months t-12..t-2,
        skipping the latest month); needs >= 80% valid days. lam in [0.5, 1.0, 2.0].
    B high52(ctx, lam)      [7]. PTH = P_t / max(close over the last 252 trading days, incl. today); needs
        >= 90% of that window priced. No skip month (PTH is a price level; G&H report results with and
        without a skip). lam in [0.5, 1.0, 2.0].
    C managed_mom(ctx, lam) [5][6]. Signal: classic 12-1 total return P(t-21)/P(t-252) - 1.
        lam_t = lam * gate_t * scale_t.
        gate_t  (Daniel-Moskowitz bear state): 0 if the EW market's cumulative return over the last 504
                trading days (24 months; >= 252 while history is shorter) is negative, else 1.
        scale_t (Barroso-Santa-Clara): the realised daily return of the momentum overlay (u/N formed at each
                past month start, held to the next) over the last 126 trading days gives sig_t (annualised);
                scale_t = min(1, sig* / sig_t), sig* = median of sig over all past month starts (an
                expanding, look-ahead-free target in place of BSC's fixed 12%, because the overlay's units
                differ from a decile WML; de-risk only, never lever). scale = 1 until 126 overlay days and
                6 past estimates exist.
        lam in [0.5, 1.0, 2.0].

DECISION (as the brief): a finalist must beat the best existing style in all 4 discovery cells (Nifty E1/E2,
Dow E1/E2), ranked by smallest margin; if none does, the configuration with the largest smallest-margin.
At most 2 finalists. No tuning beyond the grid; no new candidates after results (any would be marked
POST-HOC and counted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOK, SKIP = 252, 21                  # 12 months, skip the latest month
BETA_WIN, BETA_MIN = 756, 252         # 36-month CAPM window, 12-month minimum
BEAR_WIN = 504                        # Daniel-Moskowitz: 24-month market return
VOL_WIN = 126                         # Barroso-Santa-Clara: 6 months of daily returns
MIN_NAMES = 10


# ── shared pieces ─────────────────────────────────────────────────────────────────────────────────
def _ranks(score: pd.Series, index: pd.Index) -> pd.Series:
    """Centred cross-sectional rank in [-1, 1]; names without a valid score get 0."""
    s = score.reindex(index).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < MIN_NAMES:
        return pd.Series(0.0, index=index)
    r = s.rank(method="average")
    return (2.0 * (r - 1.0) / (len(s) - 1.0) - 1.0).reindex(index).fillna(0.0)


def _overlay(ctx, u: pd.Series, lam: float) -> pd.Series:
    c = ctx.raw["CVG"].reindex(ctx.priced).fillna(0.0).clip(lower=0.0)
    c = c / c.sum() if c.sum() > 0 else pd.Series(1.0 / len(ctx.priced), index=ctx.priced)
    w = c + lam * u.reindex(c.index).fillna(0.0) / len(c)
    return w.clip(lower=0.0)


def _rets(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change(fill_method=None)


def _mom121(pxf: pd.DataFrame, names: pd.Index) -> pd.Series:
    """12-1 total return as of the last row of pxf (pxf = closes forward-filled ≤ 5 sessions)."""
    if len(pxf) < LOOK + 1:
        return pd.Series(np.nan, index=names)
    p0, p1 = pxf.iloc[-1 - LOOK].reindex(names), pxf.iloc[-1 - SKIP].reindex(names)
    return p1 / p0 - 1.0


# ── A: residual momentum (Blitz, Huij & Martens 2011; Chaves 2016) ────────────────────────────────
def _resid_score(px: pd.DataFrame, names: pd.Index) -> pd.Series:
    r = _rets(px).iloc[1:]
    out = pd.Series(np.nan, index=names)
    if len(r) < LOOK:
        return out
    mkt = r.mean(axis=1)                                  # EW market of names priced each day
    est, m_est = r.iloc[-BETA_WIN:], mkt.iloc[-BETA_WIN:]
    form, m_form = r.iloc[-LOOK:-SKIP], mkt.iloc[-LOOK:-SKIP]
    x_all = m_est.to_numpy()
    xf = m_form.to_numpy()
    for s in names:
        if s not in r.columns:
            continue
        y = est[s].to_numpy()
        ok = np.isfinite(y) & np.isfinite(x_all)
        if ok.sum() < BETA_MIN:
            continue
        x, yy = x_all[ok], y[ok]
        vx = x.var()
        if vx <= 0:
            continue
        b = ((x - x.mean()) * (yy - yy.mean())).mean() / vx
        a = yy.mean() - b * x.mean()
        yf = form[s].to_numpy()
        okf = np.isfinite(yf) & np.isfinite(xf)
        if okf.sum() < 0.8 * len(yf):
            continue
        e = yf[okf] - a - b * xf[okf]
        sd = e.std(ddof=1)
        if sd > 0:
            out[s] = e.sum() / sd
    return out


def resid_mom(ctx, lam: float = 1.0) -> pd.Series:
    u = _ranks(_resid_score(ctx.px, ctx.priced), ctx.priced)
    return _overlay(ctx, u, lam)


# ── B: 52-week-high momentum (George & Hwang 2004) ────────────────────────────────────────────────
def _pth(px: pd.DataFrame, names: pd.Index) -> pd.Series:
    if len(px) < LOOK:
        return pd.Series(np.nan, index=names)
    win = px.iloc[-LOOK:].reindex(columns=names)
    hi, cnt = win.max(), win.notna().sum()
    p = px.ffill(limit=5).iloc[-1].reindex(names)
    pth = p / hi
    return pth.where(cnt >= 0.9 * LOOK)


def high52(ctx, lam: float = 1.0) -> pd.Series:
    u = _ranks(_pth(ctx.px, ctx.priced), ctx.priced)
    return _overlay(ctx, u, lam)


# ── C: crash-managed 12-1 momentum (Daniel & Moskowitz 2016 gate x Barroso & Santa-Clara 2015 scale) ─
def _bear_gate(r: pd.DataFrame) -> float:
    mkt = r.mean(axis=1).dropna()
    n = min(BEAR_WIN, len(mkt))
    if n < LOOK:
        return 1.0
    return 0.0 if float(np.prod(1.0 + mkt.iloc[-n:].to_numpy()) - 1.0) < 0.0 else 1.0


def _bsc_scale(px: pd.DataFrame, r: pd.DataFrame) -> float:
    idx = px.index
    starts = list(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first())
    pxf = px.ffill(limit=5)
    pos = {d: i for i, d in enumerate(idx)}
    pieces = []
    for m0, m1 in zip(starts[:-1], starts[1:]):           # complete months up to ctx.date
        i0, i1 = pos[m0], pos[m1]
        if i0 < LOOK:
            continue
        names = px.columns[px.iloc[i0].notna() & (px.iloc[i0] > 0)]
        u = _ranks(_mom121(pxf.iloc[: i0 + 1], names), names)
        if not (u != 0).any():
            continue
        rr = r.iloc[i0 + 1: i1 + 1].reindex(columns=names).fillna(0.0)
        pieces.append(pd.Series(rr.to_numpy() @ (u.to_numpy() / len(names)), index=rr.index))
    if not pieces:
        return 1.0
    f = pd.concat(pieces)
    sig = f.rolling(VOL_WIN).std(ddof=1) * np.sqrt(252.0)
    at_starts = sig.reindex([d for d in starts if d in sig.index]).dropna()
    if len(at_starts) < 7 or not np.isfinite(sig.iloc[-1]):
        return 1.0
    now, target = float(sig.iloc[-1]), float(at_starts.median())   # median includes today's estimate
    return float(min(1.0, target / now)) if now > 0 else 1.0


def managed_mom(ctx, lam: float = 1.0) -> pd.Series:
    px = ctx.px
    r = _rets(px)
    u = _ranks(_mom121(px.ffill(limit=5), ctx.priced), ctx.priced)
    lam_t = lam * _bear_gate(r) * _bsc_scale(px, r)
    return _overlay(ctx, u, lam_t)


CANDIDATES = {
    "resid_mom": (resid_mom, {"lam": [0.5, 1.0, 2.0]}),
    "high52": (high52, {"lam": [0.5, 1.0, 2.0]}),
    "managed_mom": (managed_mom, {"lam": [0.5, 1.0, 2.0]}),
}
