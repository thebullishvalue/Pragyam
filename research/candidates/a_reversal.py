"""
research/candidates/a_reversal.py — FAMILY A: REVERSAL & CAPITULATION (style search, discovery only)

PRE-REGISTRATION (written 2026-10-03, before any configuration below was run)
─────────────────────────────────────────────────────────────────────────────
LITERATURE BASIS
  · Jegadeesh (1990, JF) and Lehmann (1990, QJE): last month's (week's) losers beat last month's
    winners next month — the short-term reversal.
  · Avramov, Chordia & Goyal (2006, JF 61(5)): reversals are largest where non-informational demand for
    immediacy moves prices (illiquid, high-turnover stocks); raw contrarian profits are smaller than
    likely trading costs. → a TILT, not a long-short rank book.
  · Da, Liu & Schaumburg (2014, Mgmt Sci): only the residual, "non-fundamental" part of last month's
    return reverses; on the long side the reversal is a liquidity shock (fire sales demand liquidity).
  · Blitz, Huij, Lansdorp & Verbeek (2013, J. Financial Markets 16(3)): a reversal on residual returns
    (factor betas stripped, scaled by residual volatility) has no dynamic factor exposure, earns about
    twice the risk-adjusted return of the raw reversal, and survives costs among LARGE caps post-1990.
    The raw reversal, by contrast, loads on whatever factor (beta) just fell — exactly the 2008 risk.
  · Nagel (2012, RFS 25(7) "Evaporating Liquidity"): reversal returns are the returns to liquidity
    provision; they are strongly time-varying and predictable with the VIX, very large in 2007-09
    turmoil, small in calm markets. Hameed & Mian (2015, JFQA 50) find (intra-industry) reversals in
    large, liquid stocks too, stronger after market declines and in volatile times. Butt, Högholm &
    Sadaqat (2021, J. Multinational Financial Mgmt 59): in emerging markets the reversal return is
    higher when market volatility is high.
  · India: studies on NSE / Nifty constituents report overreaction — losers beating winners over 1-2
    year formation (2008-2016) and reversals after large monthly moves — alongside 6-12m momentum; the
    evidence is mostly from small academic journals and uses today-style constituent lists.
  · Decay: McLean & Pontiff (2016, JF) — anomaly returns are 26% lower out of sample and 58% lower
    post-publication. Reversal specifically has weakened post-2000 as market making became electronic
    (decimalisation, HFT), most of all in large caps; long-term reversal (De Bondt & Thaler 1985, JF)
    is reported to attenuate for large caps after 1977-2017 samples.

MECHANISM AND WHY IT MIGHT BEAT BOTH 1/N AND CVG HERE
  CVG's measured edge over 1/N is capitulation: it puts 4 units on DISLOCATED (sellers in control at a
  cheap price) behind a histogram gate — a slow, value-anchored liquidity-provision bet. A residual
  reversal is the fast, price-only version of the same bet (buy the names that just fell more than
  their beta explains), and it is the part of the reversal literature that survives in large caps and
  carries no beta load in a crash. Laid as a multiplicative tilt on CVG it keeps CVG's E2 return and
  could add (i) the reversal premium where it is largest — in stress (Nagel), i.e. the 2008-09 / 2011
  episodes where HRP set the Nifty E1 bar — and (ii) a sharper pick inside CVG's DOWN row, where the
  fire-sale names are. CVG is the base, not 1/N, because CVG is already the best in 2 of the 4 cells and
  closer than 1/N to the bar in the other two (Nifty E1 −1.05 vs −2.28; Dow E1 −0.22 vs 0).

SHARED SIGNAL (fixed; literature defaults)
  Daily simple returns from ctx.px (stale closes are NaN and drop out). Market m = equal-weighted mean
  daily return of the priced names. For each name, over the trailing 252 trading days ending at
  ctx.date (min 60 valid days, else neutral): β = cov(r, m)/var(m); daily residual e = r − β·m (no
  intercept). Formation = the last 21 trading days (one month, no skip). Standardised residual
  (Blitz et al. 2013): E = Σ₂₁ e / (std(e) · √21). Reversal score z = −(cross-sectional z-score of E),
  winsorised at ±3; names without a score get z = 0.
  Tilt map (MSCI tilt-index convention, keeps every name held, positive): S(x) = 1 + x for x ≥ 0,
  1 / (1 − x) for x < 0. Weights w = CVG_raw × S(λ_eff · z); the harness renormalises and caps at 10%.

CANDIDATES (3; 9 configurations)
  A1 resid_rev_cvg     residual 1-month reversal tilt on CVG (Blitz et al. 2013; Da, Liu & Schaumburg
                       2014). λ_eff = lam.                              grid lam ∈ {0.25, 0.5, 1.0}
  A2 stress_rev_cvg    the same tilt, switched on by market stress (Nagel 2012; Hameed & Mian 2015;
                       Butt et al. 2021). s_t = percentile of the current 21-day realised vol of the
                       equal-weighted universe return within its own history up to ctx.date
                       (expanding; VIX proxy, since no VIX is in ctx); λ_eff = lam · max(0, 2·s_t − 1)
                       — off at or below median stress, full lam at the top.  grid lam ∈ {0.5, 1.0, 2.0}
  A3 capit_rev_cvg     the tilt applied ONLY inside CVG's held DOWN row (cvg state ∈ DISLOCATED /
                       FADING / DISTRIBUTION — sellers in control, confirmed by CVG's histogram gate);
                       multiplier 1 elsewhere. Da et al.: the long-side reversal is the fire sale; this
                       sharpens CVG's capitulation bet toward the names whose fall was non-fundamental.
                       λ_eff = lam on the row.                          grid lam ∈ {0.5, 1.0, 2.0}

  Rejected a priori (not run, not counted): long-term reversal 36-60m (De Bondt & Thaler 1985) — the
  price history starts Oct 2006, so the signal does not exist in most of E1; weekly reversal (Lehmann
  1990) — a one-week signal held for a month is mostly decayed and only adds turnover; distance from
  the 52-week low — George & Hwang (2004, JF) find names far from their 52-week high UNDERperform (the
  anchoring/momentum side), so the evidence points the other way.

COUNT: 9 configurations here + 1 already run (the harness demo: 1-month rank-reversal tilt on 1/N,
  Nifty 17.44 / 19.33) = 10 for this family.

SURVIVORSHIP: the universes are today's Nifty 50 / Dow 30, so names that fell and were dropped are
  missing. "Buy last month's loser" is the most flattered bet in the search: in history the losers
  that kept losing left the index (and the panel). Any discovery edge here is an upper bound.

Run:  python research/candidates/a_reversal.py      (discovery only; both universes, every-name book)
"""
from __future__ import annotations

import os
import sys
from functools import partial

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

BETA_WIN, MIN_OBS, FORM, VOL_WIN, WINSOR = 252, 60, 21, 21, 3.0
DOWN_ROW = ("DISLOCATED", "FADING", "DISTRIBUTION")


def _tilt(x: pd.Series) -> pd.Series:
    """MSCI tilt-index map: 1 + x above zero, 1 / (1 − x) below — positive, every name kept."""
    return pd.Series(np.where(x >= 0, 1.0 + x, 1.0 / (1.0 - x)), index=x.index)


def _rev_score(ctx) -> pd.Series:
    """−z of the standardised 21-day residual return (Blitz et al. 2013), winsorised ±3, over ctx.priced."""
    px = ctx.px[ctx.priced].iloc[-(BETA_WIN + 1):]
    r = px.pct_change(fill_method=None).iloc[1:]
    m = r.mean(axis=1)
    E = {}
    for s in r.columns:
        x = r[s]
        ok = x.notna() & m.notna()
        if ok.sum() < MIN_OBS:
            continue
        xs, ms = x[ok], m[ok]
        vm = ms.var()
        if not vm > 0:
            continue
        beta = ((xs - xs.mean()) * (ms - ms.mean())).sum() / ((len(ms) - 1) * vm)
        e = xs - beta * ms
        sd = e.std()
        last = e[e.index > r.index[-FORM - 1]] if len(r) > FORM else e
        if not sd > 0 or len(last) < FORM // 2:
            continue
        E[s] = last.sum() / (sd * np.sqrt(FORM))
    E = pd.Series(E, dtype=float)
    z = pd.Series(0.0, index=ctx.priced)
    if len(E) >= 5 and E.std() > 0:
        z.loc[E.index] = (-(E - E.mean()) / E.std()).clip(-WINSOR, WINSOR)
    return z


def _stress(ctx) -> float:
    """Percentile of today's 21-day realised vol of the equal-weighted universe return in its own history."""
    r = ctx.px.pct_change(fill_method=None).iloc[1:]
    v = r.mean(axis=1).rolling(VOL_WIN, min_periods=VOL_WIN).std().dropna()
    if len(v) < VOL_WIN:
        return 0.5
    return float((v <= v.iloc[-1]).mean())


def _cvg(ctx) -> pd.Series:
    return ctx.raw["CVG"].reindex(ctx.priced).fillna(0.0)


def resid_rev_cvg(ctx, lam: float = 0.5) -> pd.Series:
    """A1 — residual 1-month reversal tilt on CVG."""
    return _cvg(ctx) * _tilt(lam * _rev_score(ctx))


def stress_rev_cvg(ctx, lam: float = 1.0) -> pd.Series:
    """A2 — the residual reversal tilt, on only when market stress is above its median (Nagel 2012)."""
    g = max(0.0, 2.0 * _stress(ctx) - 1.0)
    return _cvg(ctx) * _tilt(lam * g * _rev_score(ctx))


def capit_rev_cvg(ctx, lam: float = 1.0) -> pd.Series:
    """A3 — the residual reversal tilt inside CVG's held DOWN row only (the fire-sale names)."""
    st = ctx.snap["cvg state"].reindex(ctx.priced).astype(str).str.upper()
    gate = st.isin(DOWN_ROW).astype(float)
    return _cvg(ctx) * _tilt(lam * gate * _rev_score(ctx))


CANDIDATES = {
    "A1_resid_rev_cvg": (resid_rev_cvg, {"lam": [0.25, 0.5, 1.0]}),
    "A2_stress_rev_cvg": (stress_rev_cvg, {"lam": [0.5, 1.0, 2.0]}),
    "A3_capit_rev_cvg": (capit_rev_cvg, {"lam": [0.5, 1.0, 2.0]}),
}


def main() -> None:
    import pickle
    import style_search as ss

    out_dir = os.environ.get("A_REV_OUT")                 # optional: where to pickle the table
    rows = []
    for u in ("nifty_50", "dow_30"):
        d = ss.load(u)                                    # discovery only
        base = ss.baselines(d)
        runs = {}
        for name, (fn, grid) in CANDIDATES.items():
            for k, vals in grid.items():
                for v in vals:
                    runs[f"{name}[{k}={v}]"] = ss.run(partial(fn, **{k: v}), d)
        rep = ss.report(runs, d, base, show_base=False)
        for key, r in runs.items():
            m = ss.metrics(r)
            rows.append(dict(u=u, cfg=key, era="ALL", **m))
        rows += [dict(u=u, cfg=x.style, era=x.era, cagr=x.cagr, vs_best=x.vs_best, best=x.best, vol=x.vol,
                      maxdd=x.maxdd, turnover=x.turnover) for x in rep.itertuples()]
    res = pd.DataFrame(rows)
    if out_dir:
        pickle.dump(res, open(os.path.join(out_dir, "a_reversal_disc_results.pkl"), "wb"))
    print(res.to_string())


if __name__ == "__main__":
    main()
