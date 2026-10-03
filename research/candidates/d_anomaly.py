"""
research/candidates/d_anomaly.py — style search, FAMILY D: price-based anomaly tilts on CVG.

PRE-REGISTRATION (written 2026-10-03, before any candidate below was run on any data)
──────────────────────────────────────────────────────────────────────────────────────
Harness: research/style_search.py (discovery files only: Nifty 50 and Dow 30, every name held,
monthly, net of 10bp / 3bp per unit of one-way turnover, 10% cap). Eras E1 < 2014, E2 2014-2019.

LITERATURE BASIS
  · Lottery demand / MAX — Bali, Cakici & Whitelaw (2011, JFE 99, "Maxing out: Stocks as lotteries
    and the cross-section of expected returns"): the maximum daily return over the past month
    predicts returns negatively; lowest-minus-highest MAX decile > 1%/month (US 1962-2005), robust
    to size, B/M, momentum, short-term reversal, liquidity, skewness, and MAX reverses the IVOL
    puzzle. Bali, Brown, Murray & Tang (2017, JFQA 52, "A lottery-demand-based explanation of the
    beta anomaly") use MAX = mean of the FIVE highest daily returns in the month, and show the
    beta anomaly disappears once lottery demand is neutralised — i.e. MAX is the part of "low
    risk" that is mispricing rather than risk.
  · Idiosyncratic volatility — Ang, Hodrick, Xing & Zhang (2006, JF 61, "The cross-section of
    volatility and expected returns"; 2009, JFE 91, "High idiosyncratic volatility and low
    returns: International and further U.S. evidence"): one-month daily residual volatility
    predicts returns negatively; extreme-quintile spread −1.31%/month across 23 developed markets,
    significant in every G7 country. Stambaugh, Yu & Yuan (2015, JF 70, "Arbitrage asymmetry and
    the idiosyncratic volatility puzzle"): the IVOL effect is negative among overpriced and
    positive among underpriced stocks, net negative because shorting is harder than buying —
    long-only books can still harvest the overpriced leg by underweighting.
  · Realized skewness — Amaya, Christoffersen, Jacobs & Vasquez (2015, JFE 118, "Does realized
    skewness predict the cross-section of equity returns?"): low-minus-high realized-skewness
    decile earns 19bp/week (t 3.70) from intraday data; here only daily closes exist, so the
    signal is the skewness of the past month's daily returns (the SKEW control of BCW 2011).
  · Return seasonality — Heston & Sadka (2008, JFE 87, "Seasonality in the cross-section of stock
    returns"): same-calendar-month returns persist at annual lags up to 20 years, across sizes
    and industries; international: Heston & Sadka (2010, JFQA 45) in Canada, Japan, 12 European
    markets; Keloharju, Linnainmaa & Nyberg (2016, JF 71, "Return seasonalities"): 13%/yr from
    historical same-calendar-month returns, largely a seasonality in common factors.
  · Composite — Stambaugh, Yu & Yuan (2012, JFE 104, "The short of it"; 2015 JF): the mispricing
    score is the equal average of the anomalies' percentile ranks. Only price-based components
    exist here (no fundamentals, no volume).
  · Betting against beta — Frazzini & Pedersen (2014, JFE 111): NOT a candidate. Long-only and
    unlevered, BAB is a low-beta tilt; this repo already measured that low-risk tilts (ERC/HRP)
    lose return outside the 2007-13 Nifty crash era.
  DECAY AND LOCAL EVIDENCE (read before choosing; lowers the prior):
  · McLean & Pontiff (2016, JF 71): anomaly returns are 26% lower out-of-sample and 58% lower
    post-publication. Discovery (2007-2019) is post-publication for AHXZ (2006) and HS (2008) and
    largely for BCW (WP 2009, JFE 2011).
  · Hou, Xue & Zhang (2020, RFS 33, "Replicating anomalies"): with NYSE breakpoints and value
    weights, IVOL, total volatility, MAX and short-term reversal fail standalone replication —
    the effects live mostly outside large caps. Nifty 50 and Dow 30 are mega-caps.
  · India: Ali, Hasan & Östermark (2020, IREF 70) — IVOL and MAX effects are POSITIVE in the
    Indian market overall (short-sale constrained) but SIGNIFICANTLY NEGATIVE among large firms,
    the side that matters for Nifty 50. Joshipura & Joshipura (low-volatility effect, NSE top
    500, 2001-2018): low-vol earns higher risk-adjusted returns, strongest in large caps.
    Li, Zhang & Zheng (2018, J. Empirical Finance 49): across 42 markets, cross-sectional
    seasonality is economically significant in advanced markets but NOT in emerging markets.

MECHANISM — WHY IT MIGHT BEAT BOTH 1/N AND CVG HERE
  The bar per cell is: Nifty E1 20.21 (HRP), Nifty E2 19.68 (CVG), Dow E1 14.39 (EW), Dow E2 18.50
  (CVG). Starting from CVG the tilt must add +1.05 (Nifty E1), ≥ 0 (Nifty E2), +0.22 (Dow E1),
  ≥ 0 (Dow E2) %/yr; starting from 1/N it must add +2.28 / +0.85 / ≥ 0 / +0.18. CVG is the nearer
  base, so every candidate is a bounded multiplicative tilt ON CVG:

        w ∝ CVG_raw × (1 + λ·z),   z = cross-sectional rank of the signal mapped to [−1, +1]
                                   (best name +1, worst −1, missing signal 0)

  λ = 1 sets the worst name to zero and doubles the best; λ = 0.5 spans 0.5×-1.5×. Ranks make the
  tilt bounded and scale-free. The hypothesis: lottery / high-IVOL names are overpriced by
  lottery demand (BCW 2011, BBMT 2017) and become most overpriced into speculative peaks; a
  ONE-MONTH lottery signal underweights the names that just spiked — in 2008 those are the
  high-beta names, so it should give part of HRP's crash protection in Nifty E1 — while, unlike a
  252-day covariance low-vol book, it does not hold a permanent low-beta bias, so it should give up
  less in the 2014-19 bull eras where low-risk lost (BBMT: the beta anomaly IS lottery demand, so
  the mispricing piece can be separated from the risk piece). On top of CVG it is a second,
  near-orthogonal bet (CVG sizes on tape control × value; this sizes on lottery-ness). The
  composite adds a risk-unrelated signal (seasonality) to diversify noise in one-month estimates.

SIGNALS (literature defaults; all from ctx.px up to and including ctx.date, never after)
  daily returns  close-to-close over the last 21 trading days (≈ the past month); a name needs
                 ≥ 15 valid daily returns or its signal is missing (z = 0)
  MAX5           mean of the 5 highest daily returns in that window (BBMT 2017 default)   low = good
  IVOL           std of residuals of the 21 daily returns on the equal-weighted mean return of
                 the priced universe (CAPM version of AHXZ 2006; the brief's market)        low = good
  RSKEW          skewness of the 21 daily returns (daily stand-in for ACJV 2015)            low = good
  SEASON         mean return in the HOLDING calendar month (month of ctx.date) over all prior
                 years available, up to 20 (HS 2008 / KLN 2016), from month-end closes;
                 ≥ 1 prior year required                                                   high = good

CANDIDATES (3) AND GRIDS (2 points each — 6 configurations in total; no other value will be run)
  max_tilt        z from −MAX5                                          λ ∈ {0.5, 1.0}
  ivol_tilt       z from −IVOL                                          λ ∈ {0.5, 1.0}
  composite_tilt  z = rank of the SYY-style equal average of the four
                  component ranks (−MAX5, −IVOL, −RSKEW, +SEASON),
                  over the components available for each name          λ ∈ {0.5, 1.0}
  Momentum (12-1) is deliberately excluded from the composite: the brief records it as already
  measured inconsistent across eras in this repo. Base is CVG for all six (bar arithmetic above).

SELECTION (fixed): run all six on nifty_50 and dow_30 (n=None). Finalists (≤ 2): configurations
beating the best existing style in all 4 discovery cells, ranked by smallest margin; if none, the
configuration with the largest smallest-margin across the 4 cells.

PRIOR (stated before running): low. HXZ 2020 and McLean-Pontiff point to little or no MAX/IVOL
premium in mega-caps after 2006; the universes are today's constituents, which favours names that
were volatile and went up (a structural headwind for any lottery-avoidance tilt); seasonality is
weak in emerging markets. The most likely failure: the tilt reproduces part of HRP's Nifty E1 gain
but loses in Nifty E2 / Dow E2, i.e. it is a low-vol tilt by another name. I would put the chance
that any of the six clears all four discovery cells at roughly 10-15%.

RESULT (2026-10-03, discovery only) — REJECTED: none of the six beats the best existing style in
ANY of the four cells. All six trail plain CVG in every cell too. The loss grows with λ.
Six configurations run, exactly the declared grid, nothing else. Every-name book, net of costs.
Data note: the discovery px panel starts 2006-10, so SEASON is missing until the Nov 2007 holding
month. Before that the composite averages its three lottery ranks, as declared.

  CAGR % net (margin vs best existing in that cell)          vol % per cell, turnover /yr
                     Nifty E1       Nifty E2       Dow E1         Dow E2         Nifty E1/E2 vol  Dow E1/E2 vol  turn N / D
  BAR (best)         20.21 HRP      19.68 CVG      14.39 EW       18.50 CVG
  CVG (control)      19.16 (-1.05)  19.68 ( 0.00)  14.17 (-0.21)  18.50 ( 0.00)  30.3 / 15.0      19.4 / 12.8    1.48 / 1.35
  max λ=0.5          18.22 (-1.99)  19.31 (-0.37)  13.53 (-0.86)  18.05 (-0.45)  28.8 / 14.9      18.3 / 12.3    2.62 / 2.33
  max λ=1.0          17.15 (-3.05)  18.88 (-0.80)  12.80 (-1.58)  17.72 (-0.78)  27.6 / 14.8      17.3 / 12.0    4.03 / 3.46
  ivol λ=0.5         18.58 (-1.63)  19.37 (-0.31)  13.63 (-0.76)  17.48 (-1.02)  28.6 / 15.0      18.5 / 12.4    2.27 / 2.01
  ivol λ=1.0         17.86 (-2.35)  18.98 (-0.70)  13.05 (-1.34)  16.41 (-2.08)  27.1 / 15.1      17.7 / 12.0    3.63 / 3.21
  composite λ=0.5    18.06 (-2.15)  19.47 (-0.21)  13.52 (-0.87)  18.05 (-0.45)  29.1 / 14.9      18.6 / 12.5    2.61 / 2.40
  composite λ=1.0    16.85 (-3.35)  19.17 (-0.51)  12.78 (-1.60)  17.62 (-0.88)  28.1 / 14.8      18.0 / 12.2    4.16 / 3.79
  maxDD (E1 cells, monthly NAV): CVG −58.5 / −39.3 · max λ1 −56.2 / −37.6 · ivol λ1 −52.0 / −37.2 ·
  composite λ1 −55.2 / −37.4. The tilts cut vol and drawdown a little, at a larger cost in return.
  Paired gap to CVG, t: −0.35 to −1.90 across the 24 config-cells. None is positive.

  FINALIST (the rule's fallback, since none passes): ivol_tilt(lam=0.5). It has the largest
  smallest-margin, −1.63 (Nifty E1), and is named only because the protocol asks for one. It
  trails CVG in all four cells and is not a credible beat-all candidate. Runner-up max λ0.5 −1.99.

  WHY IT FAILED (read-only diagnostics after the run; they informed no selection):
  · It is not costs. The λ1 gap to CVG is −1.1 to −1.5 %/yr GROSS, against −0.06 to −0.27 of
    extra cost.
  · The 2009 rebound did it. Gross gap vs CVG in 2009: −11.4 (max), −13.8 (ivol), −10.2 (comp)
    %/yr on Nifty; −5.5 to −7.4 on Dow. The names that just spiked or were most idiosyncratic
    in the crash led the V-shaped recovery, the same junk-rally pattern as a momentum crash.
    IVOL did protect in 2007-08 on Nifty (+5.2 %/yr gross over 23 months, about +10 points)
    but lost about 14 points in 2009 alone.
    In 2010-13 and 2014-19 the gap is about 0 to −2.
  · The lottery signals carry no premium here. Mean monthly rank IC with next-month return:
    −MAX5 −0.002 (Nifty) / −0.015 (Dow); −IVOL −0.005 / −0.022; −RSKEW −0.002 / 0.000. All
    |t| ≤ 1.2, and the Dow signs are wrong. This matches HXZ 2020 (no MAX/IVOL premium in large
    caps) and post-publication decay. SEASON was the only positive signal: IC +0.015 (t 1.0)
    Nifty, +0.022 (t 1.2) Dow, concentrated in Dow E2 (+0.063, t 2.4) and ≈ 0 in Nifty E1. It
    was not run alone. Adding it now would be post-hoc. A fresh pre-registration elsewhere is
    the honest test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW, MIN_OBS, TOP_K, MAX_YEARS = 21, 15, 5, 20


# ── signals ──────────────────────────────────────────────────────────────────────────────────────
def _rets(ctx) -> pd.DataFrame:
    """Daily close-to-close returns of the priced names over the last WINDOW sessions (≤ ctx.date)."""
    px = ctx.px[ctx.priced].iloc[-(WINDOW + 1):]
    r = px.pct_change(fill_method=None).iloc[1:]
    return r.loc[:, r.notna().sum() >= MIN_OBS]


def _max5(ctx) -> pd.Series:
    r = _rets(ctx)
    a = np.sort(r.fillna(-np.inf).to_numpy(), axis=0)[::-1][:TOP_K]
    return pd.Series(a.mean(axis=0), index=r.columns)


def _ivol(ctx) -> pd.Series:
    r = _rets(ctx)
    mkt = ctx.px[ctx.priced].iloc[-(WINDOW + 1):].pct_change(fill_method=None).iloc[1:].mean(axis=1)
    out = {}
    for s in r.columns:
        m = r[s].notna() & mkt.notna()
        y, x = r[s][m].to_numpy(), mkt[m].to_numpy()
        if len(y) < MIN_OBS or np.var(x) <= 0:
            continue
        b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        e = y - (y.mean() - b * x.mean()) - b * x
        out[s] = float(np.std(e, ddof=1))
    return pd.Series(out, dtype=float)


def _rskew(ctx) -> pd.Series:
    return _rets(ctx).skew()


def _season(ctx) -> pd.Series:
    px = ctx.px[ctx.priced]
    px = px[px.index < pd.Timestamp(ctx.date.year, ctx.date.month, 1)]   # completed months only
    if px.empty:
        return pd.Series(dtype=float)
    me = px.resample("M").last()
    mr = me.pct_change(fill_method=None)
    same = mr[(mr.index.month == ctx.date.month) & (mr.index.year < ctx.date.year)].tail(MAX_YEARS)
    return same.mean(skipna=True).dropna()


def _z(sig: pd.Series) -> pd.Series:
    """Rank of an oriented signal (higher = better) mapped to [−1, +1]; fewer than 2 names → empty."""
    s = pd.Series(sig, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return pd.Series(dtype=float)
    return 2.0 * (s.rank(method="average") - 1.0) / (len(s) - 1.0) - 1.0


def _tilt(ctx, z: pd.Series, lam: float) -> pd.Series:
    base = ctx.raw["CVG"].reindex(ctx.priced).fillna(0.0)
    zz = z.reindex(ctx.priced).fillna(0.0).clip(-1.0, 1.0)
    w = base * (1.0 + lam * zz)
    return w if w.sum() > 0 else base


# ── candidates ───────────────────────────────────────────────────────────────────────────────────
def max_tilt(ctx, lam: float = 1.0) -> pd.Series:
    """CVG × (1 + λ·z), z = rank of −MAX5 (BCW 2011; BBMT 2017)."""
    return _tilt(ctx, _z(-_max5(ctx)), lam)


def ivol_tilt(ctx, lam: float = 1.0) -> pd.Series:
    """CVG × (1 + λ·z), z = rank of −IVOL, one-month CAPM residual volatility (AHXZ 2006, 2009)."""
    return _tilt(ctx, _z(-_ivol(ctx)), lam)


def composite_tilt(ctx, lam: float = 1.0) -> pd.Series:
    """CVG × (1 + λ·z), z = rank of the equal average of the component ranks (SYY 2012/2015 style)."""
    parts = [_z(-_max5(ctx)), _z(-_ivol(ctx)), _z(-_rskew(ctx)), _z(_season(ctx))]
    avg = pd.concat(parts, axis=1).reindex(ctx.priced).mean(axis=1, skipna=True)
    return _tilt(ctx, _z(avg), lam)


CANDIDATES = {
    "max_tilt": (max_tilt, {"lam": [0.5, 1.0]}),
    "ivol_tilt": (ivol_tilt, {"lam": [0.5, 1.0]}),
    "composite_tilt": (composite_tilt, {"lam": [0.5, 1.0]}),
}
