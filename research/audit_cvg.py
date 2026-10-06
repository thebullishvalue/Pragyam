"""
research/audit_cvg.py — CVG opportunity test: a weekly review with a per-name no-trade band (CVG-O1).

PRE-REGISTRATION (copied from the coordinator's brief without changes, before any variant was run;
no variants beyond V1 and V2. The first session writing this file was cut off by a usage limit
before the run; it was resumed with the pre-registration, the implementation notes and the bar
unchanged — only the two extra zero-change checks marked "added on resume" below are new)
───────────────────────────────────────────────────────────────────────────────────────────────────
id      CVG-O1
title   Weekly review with a per-name no-trade band on the every-name CVG book

mechanism
    CVG's heaviest cells are short spells: Dislocated lasts 3.2 days, Basing 2.2 and Turned 2.1 on
    Nifty, and names change state 33 times a year. On Nifty E1/E2 the tilt's active return also fades
    within days (decay2.log: E1 +1.40 at lag 0, +0.82 at lag 1, +0.20 at lag 3). A first-trading-day
    monthly book samples each state on one close. A weekly review catches more of them. A per-name
    band trades a name only when |target − drifted weight| exceeds the band, which keeps turnover
    down. The style search never saw this lever: its harness is monthly-only, and every rejected
    candidate was a monthly weight map or tilt, so this is not a repeat. Prior evidence, stated so the
    result is read honestly (5-phase averages, same-close execution, vs a phase-averaged monthly
    book): band 0.5pt Nifty +2.03/+0.29/+0.11 and Dow +1.53/+0.23/+0.44 at 2.81x/3.32x turnover; PIT
    Dow E3 +0.36. Against the first-trading-day monthly book in the same daily-NAV harness (Nifty
    19.57/20.24/22.96, Dow 14.30/18.21/15.27) it is only -0.03 Nifty E2, -0.10 Nifty E3 and -0.02
    Dow E2. With T+1 execution it is -0.07 Nifty E3 and +0.04 Dow E2. Nifty E3 shows no fast decay
    (lag 5-10 stronger than lag 0). The gain sits almost entirely in 2007-13, so E2/E3 are the
    binding cells and the prior is a likely fail. It is pre-registered because it is the only open,
    mechanism-backed lever, and the result decides whether the README should call CVG a fast signal
    that monthly review under-uses. Shipping would need current holdings as an app input:
    compute_nco_portfolio has none.

literature
    Gârleanu & Pedersen (2013), 'Dynamic Trading with Predictable Returns and Transaction Costs',
    JF 68(6): trade speed is set by alpha decay vs costs, with a no-trade region. Leland (2000);
    Donohue & Yip (2003) on rebalancing bands. Lehmann (1990, QJE) and Jegadeesh (1990, JF) on
    short-horizon reversal. Nagel (2012), 'Evaporating Liquidity', RFS 25(7): reversal returns
    concentrate in high-volatility eras, consistent with the gain sitting in E1. Hoffstein, Faber &
    Braun (2020), SSRN 3673910, on rebalance timing luck (why the schedule is anchored to the
    shipped one).

variants
    V1: Target = the shipped every-name CVG weights (nco.cvg_weights from the snapshot readings at
        the review date's close; units panel as in scratchpad lib.py / units_{u}.pkl, which match
        stored raw CVG to 4e-17), with no top-N and the harness cap. Schedule anchored to the
        shipped one: review on each month's first trading day and then on every 5th trading session
        after it within the same month (sessions 0, 5, 10, 15, 20 of the month). At every review
        after the first, a name moves to target only if |target − drifted weight| > 0.005 (absolute
        weight). Names inside the band keep their drifted weight, and the remainder is spread over
        out-of-band names pro rata to target (exactly dbt_band in scratchpad/lp_wband_lib.py;
        entries and exits go through the same rule). Costs 10bp India / 3bp US per unit one-way
        turnover.
    V2: Identical to V1 with band 0.010 (absolute weight).

bar
    Ruled (pass requires all three): (1) In the daily-NAV harness (scratchpad/lp_wband_lib.dbt_band,
    ss.load(u, holdout=True), identical windows for both arms, on the snapshot readings in force; if
    CVG-B1's LADDER='up' fix lands first, both arms use the regenerated readings), the variant's net
    CAGR is above the shipped arm in E1, E2 and E3 on Nifty 50 AND Dow 30 under the harness's
    same-close convention. The shipped arm is monthly, first trading day, no band, every name.
    Compare within the daily harness only, never against ss.run's absolute 19.16/19.68/22.59 and
    14.17/18.50/15.06. (2) Cadence guard, added because a faster cadence gains disproportionately
    from same-close execution of a decaying signal, and the app's book is computed at the close and
    traded the next session: with both arms executed at the next session's close on the same signal
    (T+1), the variant is not below the shipped arm in any of the six cells. (3) Point-in-time Dow
    E3 (style_search_pit membership, as in scratchpad/lp_pit.py) is not below the shipped arm, both
    same-close and T+1. Reported, not ruled: 2x costs, turnover, 5-phase averages for both arms,
    ETF27, and the gross-vs-net split. Two variants are tested; read t-stats accordingly. The tester
    may not change the schedule, the bands, the band rule or the costs.

needs_regeneration   false

IMPLEMENTATION NOTES (fixed before the run; how the ruled and reported items are computed)
───────────────────────────────────────────────────────────────────────────────────────────────────
  readings   the snapshot readings in force (research/cvg_reweight_{u}.pkl through sb.snapshots,
             stale closes unpriced). CVG-B1's LADDER='up' fix has NOT landed in the repo (git clean
             at the run), so both arms use the current readings.
  target     units = cvgrid.graded_units(cvg state, conv tape, value tape, conv push) per name per
             session (as scratchpad lib.units_panel); a priced name with no state = UNREAD = 1.0
             unit (nco.cvg_readings maps it so). Renormalised over the priced names, then sb.cut
             with no top-N (the 10% cap) — what nco.compute_nco_portfolio(method="CVG") hands the
             harness.
  dbt_band   copied verbatim below from scratchpad/lp_wband_lib.py (checked identical on a run when
             that file is present). A band of 0 returns the target exactly, so the shipped arm is
             dbt_band(months, band=0).
  checks     (added on resume) the units panel's dates must equal the px calendar (T+1 reads the
             previous row), and it is compared with scratchpad units_{u}.pkl when that is present.
  shipped    dates = d["months"] (first trading day of each month; its last entry is the end).
  variant    sessions 0, 5, 10, 15, 20 of each calendar month on the px calendar, from months[0] to
             the end date months[-1] (the end is a valuation date, not a review).
  T+1        each review date moved one session forward (traded at the next session's close); the
             target is the units of the review date itself (the session before the trade); the band
             compares it with the drifted weight on the trade date; the priced set is the trade
             date's; the end date is unchanged. Applied identically to both arms.
  eras       daily returns with index > start and <= end (cagr_w, 252-session annualisation as in
             lp_wband_lib — inflates Nifty's absolute CAGR ~+0.4 since India trades ~246 sessions a
             year; both arms share the window, so the sign of every gap is unaffected).
  paired t   the two arms' calendar-month compounded net returns inside the era, differenced;
             mean / (sd / sqrt n).
  turnover   one-way turnover summed over the reviews dated inside the era, per calendar year.
  PIT        style_search_pit's PKL (px, months, membership) and the readings from its snapshots
             (cvg_reweight_dow_pit.pkl, stale closes unpriced); the book holds priced members only
             (membership read on the trade date); window from the first session of 2019 (seeds
             turnover), E3 scored from 2020-01-01.
  phases     5-phase averages (reported only): shipped arm reviewed on month session k ∈
             {0, 4, 8, 12, 16} (lp_wband's phase-averaged monthly book); variant on sessions
             {k, k+5, k+10, k+15, k+20}, k = 0..4; every phase's window starts on months[0].
  2x / gross same-close, phase 0, cost doubled / cost zero, both arms.
  ETF27      ss.load("etf_27") (27 funds, Mar 2025 →), same-close and T+1; reported, not ruled.
  trials     every variant configuration run (V1/V2 × universe × convention/cost/phase) is counted;
             the shipped arm's runs are controls and listed separately.

Run:  AUDIT_CVG_CACHE=<dir> python research/audit_cvg.py      (cache is optional; ~10 min, 1 core)

RESULT (2026-10-04) — BOTH VARIANTS FAIL THE BAR. CVG stays a monthly book.
(v12.2, 2026-10-06: the committee's CVG bugs are fixed, and CVG-B1 shipped — pragati.LADDER = "up",
the conviction tape reads D · W — with CVG-B4's 8-year tape window; see CHANGELOG 12.2.0. The
figures below were measured before that, on the Ladder-down snapshots.)
───────────────────────────────────────────────────────────────────────────────────────────────────
zero-change (reproduces the shipped CVG exactly)
    target from the units panel vs stored raw CVG (= nco.compute_nco_portfolio(method="CVG")) at
    every monthly rebalance: max |Δw| 2.8e-17 Nifty, 4.2e-17 Dow, 2.8e-17 ETF27, 4.2e-17 PIT Dow
    (the same after the cap). At 12 intra-month review dates (6 for ETF27/PIT) vs a fresh
    compute_nco_portfolio(CVG) on the 253-snapshot window: max 2.8e-17. The shipped arm, period by
    period, vs ss.run's CVG: gross max 9.2e-16. Net max 5.1e-5, because the harness charges the cost on
    the first session's return and ss.run on the month's return. Turnover max 1e-16, except Nifty
    2008-04-01 (5.1e-4): that review follows the one excluded period, where BAJAJ-AUTO (demerged) is
    unpriced at the period end. dbt_band is identical to scratchpad lp_wband_lib (0.0), and the
    units panel is identical to scratchpad units_{u}.pkl (0.0). Shipped arm in the daily harness:
    Nifty 19.57/20.24/22.96, Dow 14.30/18.21/15.27, which are the brief's numbers.

ruled cells — net CAGR %, shipped → variant (gap, paired t of the monthly gap), turnover/yr over the
whole span (shipped → variant)
                      E1                      E2                      E3                   TO
  Nifty same-close
    V1 0.005   19.57→20.83 (+1.25 t+1.4)  20.24→19.97 (−0.28 t−0.6)  22.96→23.19 (+0.24 t+0.4)  1.47→2.86
    V2 0.010   19.57→19.79 (+0.22 t−0.2)  20.24→20.58 (+0.33 t+0.6)  22.96→23.40 (+0.44 t+0.8)  1.47→1.49
  Dow same-close
    V1 0.005   14.30→15.66 (+1.36 t+1.9)  18.21→18.57 (+0.36 t+0.8)  15.27→15.64 (+0.37 t+0.4)  1.41→3.48
    V2 0.010   14.30→15.55 (+1.25 t+1.9)  18.21→18.39 (+0.17 t+0.4)  15.27→16.02 (+0.76 t+0.9)  1.41→2.31
  Nifty T+1
    V1 0.005   19.76→20.27 (+0.52 t+0.5)  20.20→19.98 (−0.22 t−0.4)  23.15→23.13 (−0.02 t+0.1)  1.47→2.80
    V2 0.010   19.76→19.43 (−0.33 t−0.7)  20.20→19.19 (−1.01 t−2.0)  23.15→23.08 (−0.07 t−0.1)  1.47→1.37
  Dow T+1
    V1 0.005   14.47→15.46 (+0.99 t+1.5)  18.00→18.00 (−0.00 t−0.0)  15.57→15.04 (−0.53 t−0.8)  1.41→3.41
    V2 0.010   14.47→15.37 (+0.90 t+1.3)  18.00→18.11 (+0.12 t+0.3)  15.57→15.20 (−0.37 t−0.6)  1.41→2.23
  PIT Dow E3   same-close: V1 11.67→11.95 (+0.28 t+0.3), V2 11.67→12.12 (+0.44 t+0.6); TO 1.55→3.82/2.61
               T+1:        V1 11.84→11.54 (−0.30 t−0.5), V2 11.84→11.76 (−0.08 t−0.2)
  verdict  V1: (1) fails (Nifty E2 −0.28); (2) fails (Nifty E2 −0.22, E3 −0.02; Dow E2 −0.001,
               E3 −0.53); (3) fails (T+1 −0.30).
           V2: (1) PASSES (all six cells above, but t ≤ +0.9 except Dow E1); (2) fails (Nifty E1 −0.33,
               E2 −1.01, E3 −0.07; Dow E3 −0.37); (3) fails (T+1 −0.08).

reported, not ruled
  gross (gap before costs)    Nifty V1 +1.45/−0.11/+0.39, V2 +0.24/+0.33/+0.42; Dow V1 +1.43/+0.42/+0.45,
                              V2 +1.28/+0.20/+0.79. Costs take ≤ 0.2 off V1's gap and ~0 off V2's, so
                              the sign pattern comes from the signal, not from costs.
  2x costs (same-close)       Nifty V1 +1.05/−0.44/+0.08, V2 +0.20/+0.33/+0.45; Dow V1 +1.28/+0.29/+0.30,
                              V2 +1.21/+0.15/+0.72.
  5-phase (same-close)        shipped Nifty 18.52/19.92/22.76 (TO 1.45), Dow 14.51/17.96/15.34 (TO 1.42).
                              Nifty V1 +1.94/+0.21/+0.11 (TO 2.72), V2 +1.06/+0.26/+0.30 (TO 1.42);
                              Dow V1 +1.47/+0.26/+0.54 (TO 3.28), V2 +1.16/+0.31/+0.43 (TO 2.16).
                              The phase ranges cross zero in E2 and E3 in every case except Dow V2 E2
                              (+0.15..+0.67). E1's range stays above zero in every case.
  ETF27 (Mar 2025 →, 19 mo)   same-close V1 17.28→15.99 (−1.29 t−1.4), V2 →16.43 (−0.85 t−0.9);
                              T+1 V1 −1.12, V2 −0.96. TO 1.32→3.11 / 1.94.
  POST-HOC (added after the verdict; not pre-registered; cannot change it): 5-phase averages for the
  cells the bar rules on as single schedules.
      T+1 Nifty  V1 +1.45/−0.09/−0.07 (E2 3/5 phases below), V2 +0.73/−0.12/+0.31
      T+1 Dow    V1 +1.18/+0.10/+0.29, V2 +1.13/+0.14/+0.00 (E3 4/5 phases below)
      PIT Dow E3 same-close V1 +0.44, V2 +0.05; T+1 V1 +0.20, V2 +0.06 (3/5 phases below for both).
      The shipped arm's own timing luck on PIT E3 is 11.67 (phase 0) vs 12.16 (5-phase mean).

reading
    The weekly band adds return in 2007-13 and nothing reliable after 2013. In E1, V1 gains +0.5 to
    +1.9 on Nifty and both variants gain +0.9 to +1.5 on Dow in every convention (Dow t ≈ 1.9
    same-close). V2 on Nifty E1 is weaker: +0.22 same-close, −0.33 at T+1, and +0.7 to +1.1 on
    5-phase means. E2/E3 gaps sit within about ±0.8 of zero, change sign between same-close and T+1
    and between phases, and have |t| ≤ 1. The one exception is V2 Nifty E2 at T+1 (t −2.0, one path;
    its 5-phase mean is −0.12). This is the prior's
    pattern (gain concentrated in E1, the high-volatility era; Nagel 2012). The cadence guard did its
    job: V2 clears every same-close cell and fails at T+1. For the README: do not call CVG a fast
    signal that monthly review under-uses. At most, a weekly band added return in 2007-13 and has been
    timing noise since. No code change, and compute_nco_portfolio needs no holdings input.

trials  72 variant configurations: 40 pre-registered (V1/V2 × Nifty and Dow × same-close, T+1, 2x,
        gross and 4 extra same-close phases, + ETF27 × 2 conventions, + PIT × 2 conventions) and 32
        post-hoc phase runs. There were also 40 shipped-arm control runs. The pre-registered part was
        run twice; the second run, with the post-hoc section added, reproduced the first exactly.
        One smoke test repeated the Dow V1 same-close configuration.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import cvgrid                                     # noqa: E402
import style_blends as sb                         # noqa: E402
import style_search as ss                         # noqa: E402
import style_search_pit as PIT                    # noqa: E402

CACHE = os.environ.get("AUDIT_CVG_CACHE")
SCRATCH_LIB = os.environ.get("AUDIT_CVG_SCRATCH")          # optional: dir holding lp_wband_lib.py
KEY = {"nifty_50": "nifty_50", "dow_30": "dow_30", "etf_27": "etf_book"}
VARIANTS = {"V1": 0.005, "V2": 0.010}
ERAS = ss.ERAS
PIT_START = pd.Timestamp("2019-01-01")
E3 = ("E3", "2020-01-01", None)
COUNT = {"variant": 0, "control": 0}


# ── readings and the units panel ───────────────────────────────────────────────────────────────
def _readings_from(snaps: list) -> dict:
    long = pd.concat([s.drop_duplicates("symbol", keep="last").assign(_d=pd.Timestamp(dt))
                      for dt, s in snaps], ignore_index=True)
    out = {}
    for c in ("conv tape", "value tape", "conv push", "price"):
        long[c] = pd.to_numeric(long[c], errors="coerce")
        out[c] = long.pivot(index="_d", columns="symbol", values=c).sort_index()
    out["state"] = long.pivot(index="_d", columns="symbol", values="cvg state").sort_index()
    return out


def units_panel(R: dict) -> pd.DataFrame:
    S = R["state"]
    C, V, Q = (R[c].reindex_like(S).to_numpy(float) for c in ("conv tape", "value tape", "conv push"))
    A = S.to_numpy(object)
    out = np.full(A.shape, np.nan)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            s = A[i, j]
            if isinstance(s, str):
                out[i, j] = cvgrid.graded_units(s, C[i, j], V[i, j], Q[i, j])
    return pd.DataFrame(out, index=S.index, columns=S.columns)


def _cached(name: str, build):
    if CACHE:
        p = os.path.join(CACHE, f"audit_cvg_{name}.pkl")
        if os.path.exists(p):
            return pickle.load(open(p, "rb"))
        x = build()
        os.makedirs(CACHE, exist_ok=True)
        pickle.dump(x, open(p, "wb"))
        return x
    return build()


def load_universe(u: str):
    d = ss.load(u, holdout=True)
    snaps = sb.snapshots(KEY[u])
    if u == "etf_27":
        snaps = [(dt, s[~s["symbol"].isin(ss.ETF_YOUNG)].reset_index(drop=True)) for dt, s in snaps]
    U = _cached(f"units_{u}", lambda: units_panel(_readings_from(snaps)))
    assert U.index.equals(d["px"].index), "units panel dates must equal the px calendar"
    return d, snaps, U


def load_pit():
    d = pickle.load(open(PIT.PKL, "rb"))
    snaps = sb.unstale(pickle.load(open(os.path.join(HERE, "cvg_reweight_dow_pit.pkl"), "rb")))
    U = _cached("units_dow_pit", lambda: units_panel(_readings_from(snaps)))
    assert U.index.equals(d["px"].index), "units panel dates must equal the px calendar"
    return d, snaps, U


def check_scratch_units(u: str, U: pd.DataFrame) -> float | None:
    """Max |U − scratchpad units_{u}.pkl| over the shared cells (added on resume; None when absent)."""
    p = os.path.join(SCRATCH_LIB, f"units_{u}.pkl") if SCRATCH_LIB else None
    if not p or not os.path.exists(p):
        return None
    V = pickle.load(open(p, "rb"))
    A, B = U.align(V, join="inner")
    both = A.notna() & B.notna()
    one = A.notna() ^ B.notna()
    return float(max((A - B).abs()[both].max().max(), 0.0) + (np.inf if one.to_numpy().any() else 0.0))


# ── the daily-NAV harness: verbatim from scratchpad/lp_wband_lib.py ────────────────────────────
def dbt_band(d, wfn, dates, band, cost=None):
    px = d["px"]; cost = d["cost"] if cost is None else cost
    out = []; tos = {}; cur = None
    for a, b in zip(dates[:-1], dates[1:]):
        p = px.loc[a]; priced = p.index[p.notna() & (p > 0)]
        t = pd.Series(wfn(a, priced), dtype=float); t = t[t.index.isin(priced)].clip(lower=0).fillna(0)
        t = sb.cut(t.sort_values(ascending=False, kind="stable"), None)
        if cur is None:
            w = t; to = 0.0
        else:
            c = cur.reindex(t.index).fillna(0.0); c = c / c.sum()
            inb = (t - c).abs() <= band
            w = t.copy(); w[inb] = c[inb]
            rest = 1.0 - w[inb].sum()
            if (~inb).any() and t[~inb].sum() > 0: w[~inb] = t[~inb] * rest / t[~inb].sum()
            else: w = w / w.sum()
            idx = w.index.union(cur.index)
            to = float(0.5 * (w.reindex(idx, fill_value=0) - cur.reindex(idx, fill_value=0)).abs().sum())
        seg = px.loc[a:b, w.index]; rel = (seg / p.reindex(w.index)).ffill().fillna(1.0)
        nav = rel.mul(w, axis=1).sum(axis=1); dr = nav.pct_change().iloc[1:]
        if len(dr): dr.iloc[0] -= to * cost / 1e4
        tos[a] = to; out.append(dr)
        drift = w * rel.iloc[-1]; cur = drift / drift.sum()
    return pd.concat(out), pd.Series(tos)


def cagr_w(r, a, z):
    m = np.ones(len(r), bool)
    if a: m &= r.index > pd.Timestamp(a)
    if z: m &= r.index <= pd.Timestamp(z)
    x = r[m]; return ((1 + x).prod() ** (252 / len(x)) - 1) * 100


# ── schedules ──────────────────────────────────────────────────────────────────────────────────
def window(d, start=None):
    cal = d["px"].index
    lo = d["months"][0] if start is None else max(pd.Timestamp(start), d["months"][0])
    return cal[(cal >= lo) & (cal <= d["months"][-1])]


def month_sessions(cal: pd.DatetimeIndex, ks) -> list:
    """Sessions ks (0-based) of every calendar month on `cal`, plus the window's start and end."""
    grp = pd.Series(cal, index=cal).groupby([cal.year, cal.month])
    dates = [g.iloc[k] for _, g in grp for k in ks if k < len(g)]
    return sorted(set([cal[0]] + [x for x in dates if x < cal[-1]] + [cal[-1]]))


def shipped_dates(d, cal) -> list:
    return [m for m in d["months"] if cal[0] <= m <= cal[-1]] if cal[0] == d["months"][0] else \
        month_sessions(cal, [0])


def t_plus_1(dates: list, cal: pd.DatetimeIndex) -> list:
    pos = {x: i for i, x in enumerate(cal)}
    end = dates[-1]
    moved = [cal[pos[x] + 1] for x in dates[:-1] if pos[x] + 1 < len(cal) and cal[pos[x] + 1] < end]
    return sorted(set(moved)) + [end]


def run_arm(d, U, dates, band, cost=None, lag=0, members=None, kind="variant"):
    COUNT[kind] += 1
    pos = {x: i for i, x in enumerate(U.index)}

    def wfn(a, priced):
        names = [s for s in priced if members(s, a)] if members else list(priced)
        return U.iloc[pos[a] - lag].reindex(names).fillna(1.0)
    return dbt_band(d, wfn, dates, band, cost)


# ── statistics ─────────────────────────────────────────────────────────────────────────────────
def _mask(idx, a, z):
    m = np.ones(len(idx), bool)
    if a:
        m &= idx > pd.Timestamp(a)
    if z:
        m &= idx <= pd.Timestamp(z)
    return m


def _monthly(r):
    return (1 + r).groupby([r.index.year, r.index.month]).prod() - 1


def cell(rv, tv, rs, ts, a, z) -> dict:
    """Variant (rv, tv) against shipped (rs, ts) in one era."""
    assert rv.index.equals(rs.index), "the two arms must share the window"
    m = _mask(rv.index, a, z)
    x, y = rv[m], rs[m]
    dd = _monthly(x) - _monthly(y)
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if dd.std(ddof=1) > 0 else np.nan
    lo, hi = x.index[0], x.index[-1]
    yrs = max((hi - lo).days, 1) / 365.25

    def to(ts_):
        k = ts_[(ts_.index >= lo - pd.Timedelta(days=0)) & (ts_.index < hi)]
        return float(k.sum() / yrs)
    cv, cs = cagr_w(rv, a, z), cagr_w(rs, a, z)
    return dict(v=cv, s=cs, gap=cv - cs, t=t, months=len(dd), to_v=to(tv), to_s=to(ts),
                vol_v=x.std(ddof=1) * np.sqrt(252) * 100, vol_s=y.std(ddof=1) * np.sqrt(252) * 100)


def line(lbl, cells: dict) -> str:
    return f"  {lbl:30s} " + " | ".join(
        f"{e} {c['s']:6.2f}→{c['v']:6.2f} ({c['gap']:+.2f}, t{c['t']:+.1f}) TO {c['to_s']:.2f}→{c['to_v']:.2f}"
        for e, c in cells.items())


# ── zero-change reproduction ───────────────────────────────────────────────────────────────────
def check_zero_change(u, d, snaps, U, members=None, n_sample=12) -> dict:
    out = {}
    # (a) target at every monthly rebalance vs stored raw CVG (= nco.compute_nco_portfolio, CVG)
    errs, errs_cut = [], []
    for a in d["months"][:-1]:
        raw = d["raw"]["CVG"][a]
        p = d["px"].loc[a]
        priced = [s for s in p.index[p.notna() & (p > 0)] if (members(s, a) if members else True)]
        t = U.loc[a].reindex(priced).fillna(1.0)
        t = t / t.sum()
        idx = t.index.union(raw.index)
        errs.append(float((t.reindex(idx, fill_value=0) - raw.reindex(idx, fill_value=0)).abs().max()))
        ct = sb.cut(t.sort_values(ascending=False, kind="stable"), None)
        cr = sb.cut(raw.sort_values(ascending=False, kind="stable"), None)
        errs_cut.append(float((ct.reindex(idx, fill_value=0) - cr.reindex(idx, fill_value=0)).abs().max()))
    out["monthly_target_vs_raw"] = max(errs)
    out["monthly_cut_vs_raw"] = max(errs_cut)
    # (b) intra-month review dates vs a fresh nco.compute_nco_portfolio(method="CVG") call
    cal = window(d, PIT_START if members else None)
    rev = [x for x in month_sessions(cal, [5, 10, 15, 20]) if x not in set(d["months"])]
    pick = [rev[i] for i in np.linspace(0, len(rev) - 2, n_sample).astype(int)]
    pos = {pd.Timestamp(x): i for i, (x, _) in enumerate(snaps)}
    e2 = []
    for a in pick:
        i = pos[a]
        hist = snaps[max(0, i - 252): i + 1]
        if members:
            hist = [(x, s[s["symbol"].map(lambda q: members(q, a))].reset_index(drop=True)) for x, s in hist]
        raw = sb.raw(hist, "CVG")
        p = d["px"].loc[a]
        priced = [s for s in p.index[p.notna() & (p > 0)] if (members(s, a) if members else True)]
        t = U.loc[a].reindex(priced).fillna(1.0)
        t = t / t.sum()
        idx = t.index.union(raw.index)
        e2.append(float((t.reindex(idx, fill_value=0) - raw.reindex(idx, fill_value=0)).abs().max()))
    out["intramonth_target_vs_nco"] = max(e2)
    out["intramonth_dates"] = len(pick)
    return out


def check_vs_ss_run(u, d, U) -> dict:
    """The daily harness's shipped arm, per holding period, against ss.run's CVG."""
    base = ss.run(lambda c: c.raw["CVG"], d)
    cal = window(d)
    dates = shipped_dates(d, cal)
    r0, t0 = run_arm(d, U, dates, 0.0, cost=0.0, kind="control")
    rn, tn = run_arm(d, U, dates, 0.0, kind="control")
    g, n, tt = [], [], []
    miss = 0
    for a, b in zip(dates[:-1], dates[1:]):
        k = (r0.index > a) & (r0.index <= b)
        gr = float((1 + r0[k]).prod() - 1)
        nr = float((1 + rn[k]).prod() - 1)
        p_b = d["px"].loc[b].reindex(d["raw"]["CVG"][a].index)
        p_a = d["px"].loc[a].reindex(d["raw"]["CVG"][a].index)
        gap_names = bool((p_a.notna() & p_b.isna()).any())
        miss += gap_names
        if not gap_names:
            g.append(abs(gr - base.loc[a, "gross"]))
            n.append(abs(nr - base.loc[a, "ret"]))
        tb = base.loc[a, "to"]
        if not np.isnan(tb) and not gap_names:
            tt.append(abs(t0[a] - tb))
    return dict(periods=len(dates) - 1, periods_with_unpriced_end=miss, gross_max=max(g), net_max=max(n),
                to_max=max(tt) if tt else np.nan,
                ss_run_cagr={e: ss.metrics(ss._era(base, a, z))["cagr"] for e, a, z in ERAS},
                daily_cagr={e: cagr_w(rn, a, z) for e, a, z in ERAS})


def check_dbt_identical(d, U) -> float | None:
    if not SCRATCH_LIB or not os.path.exists(os.path.join(SCRATCH_LIB, "lp_wband_lib.py")):
        return None
    cwd = os.getcwd()
    sys.path.insert(0, SCRATCH_LIB)
    os.chdir(SCRATCH_LIB)
    try:
        import lp_wband_lib as L
    finally:
        os.chdir(cwd)
    cal = window(d)
    dates = month_sessions(cal, [0, 5, 10, 15, 20])
    pos = {x: i for i, x in enumerate(U.index)}
    f = lambda a, pr: U.iloc[pos[a]].reindex(pr).fillna(1.0)     # noqa: E731
    r1, t1 = L.dbt_band(d, f, dates, 0.005)
    r2, t2 = dbt_band(d, f, dates, 0.005)
    return float(max((r1 - r2).abs().max(), (t1 - t2).abs().max()))


# ── the test ───────────────────────────────────────────────────────────────────────────────────
def stock_universe(u: str, res: dict) -> None:
    t0 = time.time()
    d, snaps, U = load_universe(u)
    print(f"\n== {d['name']}  ({u})  readings {U.index[0]:%Y-%m-%d} → {U.index[-1]:%Y-%m-%d}, "
          f"window {d['months'][0]:%Y-%m-%d} → {d['months'][-1]:%Y-%m-%d}, cost {d['cost']}bp", flush=True)
    z = check_zero_change(u, d, snaps, U)
    z.update(check_vs_ss_run(u, d, U))
    if u == "dow_30":
        z["dbt_band_vs_scratch"] = check_dbt_identical(d, U)
    z["units_vs_scratch"] = check_scratch_units(u, U)
    res[u] = {"zero": z}
    print(f"  zero-change: target vs stored raw CVG max {z['monthly_target_vs_raw']:.1e} (after cut "
          f"{z['monthly_cut_vs_raw']:.1e}); {z['intramonth_dates']} intra-month review dates vs a fresh "
          f"nco.compute_nco_portfolio(CVG) max {z['intramonth_target_vs_nco']:.1e}", flush=True)
    print(f"  shipped arm per period vs ss.run CVG: gross max {z['gross_max']:.1e}, net max {z['net_max']:.1e}, "
          f"turnover max {z['to_max']:.1e} over {z['periods'] - z['periods_with_unpriced_end']} periods "
          f"({z['periods_with_unpriced_end']} periods with a held name unpriced at the period end excluded: "
          f"ss.run books it at 0%, the daily harness at its last close)", flush=True)
    print("  shipped arm CAGR, ss.run monthly vs daily harness: " + " | ".join(
        f"{e} {z['ss_run_cagr'][e]:.2f} vs {z['daily_cagr'][e]:.2f}" for e, _, _ in ERAS)
        + (f"  · dbt_band vs scratchpad copy max diff {z['dbt_band_vs_scratch']:.1e}"
           if z.get("dbt_band_vs_scratch") is not None else "")
        + (f"  · units vs scratchpad units_{u}.pkl max diff {z['units_vs_scratch']:.1e}"
           if z.get("units_vs_scratch") is not None else ""), flush=True)

    cal = window(d)
    pos = {x: i for i, x in enumerate(U.index)}
    S0 = shipped_dates(d, cal)
    W0 = month_sessions(cal, [0, 5, 10, 15, 20])
    out = {}
    # ruled: same-close and T+1, phase 0
    for conv, S, W, lag in (("same", S0, W0, 0), ("t1", t_plus_1(S0, cal), t_plus_1(W0, cal), 1)):
        rs, ts = run_arm(d, U, S, 0.0, lag=lag, kind="control")
        out[(conv, "shipped")] = (rs, ts)
        for v, band in VARIANTS.items():
            rv, tv = run_arm(d, U, W, band, lag=lag)
            out[(conv, v)] = (rv, tv)
    # reported: 2x costs, gross, phases
    for tag, cost in (("2x", 2 * d["cost"]), ("gross", 0.0)):
        out[(tag, "shipped")] = run_arm(d, U, S0, 0.0, cost=cost, kind="control")
        for v, band in VARIANTS.items():
            out[(tag, v)] = run_arm(d, U, W0, band, cost=cost)
    ph = {"shipped": [], "V1": [], "V2": []}
    for k in range(5):
        Sk = month_sessions(cal, [4 * k]) if k else S0
        Wk = month_sessions(cal, [k + 5 * j for j in range(5)]) if k else W0
        ph["shipped"].append(out[("same", "shipped")] if k == 0 else run_arm(d, U, Sk, 0.0, kind="control"))
        for v, band in VARIANTS.items():
            ph[v].append(out[("same", v)] if k == 0 else run_arm(d, U, Wk, band))
    cells = {}
    for key in ("same", "t1", "2x", "gross"):
        rs, ts = out[(key, "shipped")]
        for v in VARIANTS:
            rv, tv = out[(key, v)]
            cells[(key, v)] = {e: cell(rv, tv, rs, ts, a, z_) for e, a, z_ in ERAS + (("ALL", None, None),)}
    res[u].update(cells=cells, phases={k: [[cagr_w(r, a, z_) for _, a, z_ in ERAS + (("ALL", None, None),)]
                                           for r, _ in v] for k, v in ph.items()},
                  phase_to={k: [t.sum() / max((r.index[-1] - r.index[0]).days / 365.25, 1e-9) for r, t in v]
                            for k, v in ph.items()})
    for key, lbl in (("same", "same-close"), ("t1", "T+1"), ("2x", "same-close, 2x costs"), ("gross", "gross")):
        for v in VARIANTS:
            print(line(f"{v} band {VARIANTS[v]:.3f} {lbl}", cells[(key, v)]), flush=True)
    P = {k: np.array(v) for k, v in res[u]["phases"].items()}
    labs = [e for e, _, _ in ERAS] + ["ALL"]
    print("  5-phase averages (same-close): shipped " + " | ".join(f"{l} {x:.2f}" for l, x in zip(labs, P["shipped"].mean(0)))
          + f"  TO {np.mean(res[u]['phase_to']['shipped']):.2f}", flush=True)
    for v in VARIANTS:
        dd = P[v] - P["shipped"]
        print(f"      {v}: " + " | ".join(f"{l} {x:.2f} ({y:+.2f}, phases {lo:+.2f}..{hi:+.2f})" for l, x, y, lo, hi in
                                  zip(labs, P[v].mean(0), dd.mean(0), dd.min(0), dd.max(0)))
              + f"  TO {np.mean(res[u]['phase_to'][v]):.2f}", flush=True)
    print(f"  ({time.time() - t0:.0f}s)", flush=True)


def etf(res: dict) -> None:
    d, snaps, U = load_universe("etf_27")
    z = check_zero_change("etf_27", d, snaps, U, n_sample=6)
    z["units_vs_scratch"] = check_scratch_units("etf_27", U)
    cal = window(d)
    S0, W0 = shipped_dates(d, cal), month_sessions(cal, [0, 5, 10, 15, 20])
    cells = {}
    for conv, S, W, lag in (("same", S0, W0, 0), ("t1", t_plus_1(S0, cal), t_plus_1(W0, cal), 1)):
        rs, ts = run_arm(d, U, S, 0.0, lag=lag, kind="control")
        for v, band in VARIANTS.items():
            rv, tv = run_arm(d, U, W, band, lag=lag)
            cells[(conv, v)] = {"ALL": cell(rv, tv, rs, ts, None, None)}
    res["etf_27"] = {"zero": z, "cells": cells}
    print(f"\n== ETF book (27)  window {d['months'][0]:%Y-%m-%d} → {d['months'][-1]:%Y-%m-%d}  zero-change: target vs "
          f"stored raw CVG max {z['monthly_target_vs_raw']:.1e}, intra-month vs nco {z['intramonth_target_vs_nco']:.1e}"
          + (f", units vs scratchpad {z['units_vs_scratch']:.1e}" if z.get("units_vs_scratch") is not None else ""),
          flush=True)
    for (conv, v), c in cells.items():
        print(line(f"{v} band {VARIANTS[v]:.3f} {'same-close' if conv == 'same' else 'T+1'}", c), flush=True)


def pit(res: dict) -> None:
    d, snaps, U = load_pit()
    z = check_zero_change("dow_pit", d, snaps, U, members=PIT.member, n_sample=6)
    cal = window(d, PIT_START)
    S0, W0 = shipped_dates(d, cal), month_sessions(cal, [0, 5, 10, 15, 20])
    cells = {}
    for conv, S, W, lag in (("same", S0, W0, 0), ("t1", t_plus_1(S0, cal), t_plus_1(W0, cal), 1)):
        rs, ts = run_arm(d, U, S, 0.0, lag=lag, members=PIT.member, kind="control")
        for v, band in VARIANTS.items():
            rv, tv = run_arm(d, U, W, band, lag=lag, members=PIT.member)
            cells[(conv, v)] = {"E3": cell(rv, tv, rs, ts, E3[1], E3[2])}
    res["dow_pit"] = {"zero": z, "cells": cells}
    print(f"\n== Dow 30 point-in-time, E3 (window {cal[0]:%Y-%m-%d} → {cal[-1]:%Y-%m-%d})  zero-change: target vs stored "
          f"raw CVG (members) max {z['monthly_target_vs_raw']:.1e}, intra-month vs nco {z['intramonth_target_vs_nco']:.1e}",
          flush=True)
    for (conv, v), c in cells.items():
        print(line(f"{v} band {VARIANTS[v]:.3f} {'same-close' if conv == 'same' else 'T+1'}", c), flush=True)


def verdict(res: dict) -> dict:
    out = {}
    for v in VARIANTS:
        c1 = all(res[u]["cells"][("same", v)][e]["gap"] > 0 for u in ("nifty_50", "dow_30") for e, _, _ in ERAS)
        c2 = all(res[u]["cells"][("t1", v)][e]["gap"] >= 0 for u in ("nifty_50", "dow_30") for e, _, _ in ERAS)
        c3 = all(res["dow_pit"]["cells"][(conv, v)]["E3"]["gap"] >= 0 for conv in ("same", "t1"))
        out[v] = dict(same_close=c1, t_plus_1=c2, pit=c3, passes=c1 and c2 and c3)
        print(f"  {v}: (1) same-close all six cells above: {c1} · (2) T+1 none below: {c2} · (3) PIT E3 not below "
              f"(same-close and T+1): {c3}  →  {'PASS' if out[v]['passes'] else 'FAIL'}", flush=True)
    return out


def posthoc_phases(res: dict) -> None:
    """POST-HOC, NOT PRE-REGISTERED, NOT RULED — added after the verdict above was printed.

    The ruled T+1 and PIT cells are single schedules (phase 0), and a band rule is path-dependent, so
    one session's shift can move a cell by more than the signal does. This repeats the 5-phase
    averaging the brief reports for same-close (shipped arm on month session 4k, variant on sessions
    k, k+5, ..., k+20, k = 0..4) for T+1 on Nifty 50 and Dow 30, and for both conventions on the
    point-in-time Dow E3. Phase 0 is the ruled run, reused. It cannot change the verdict.
    """
    labs = [e for e, _, _ in ERAS]
    print("\n== POST-HOC (not pre-registered, not ruled): 5-phase averages where the ruled cell is one phase",
          flush=True)
    for u in ("nifty_50", "dow_30"):
        d, _, U = load_universe(u)
        cal = window(d)
        S = {"shipped": [[res[u]["cells"][("t1", "V1")][e]["s"] for e in labs]]}
        S.update({v: [[res[u]["cells"][("t1", v)][e]["v"] for e in labs]] for v in VARIANTS})
        for k in range(1, 5):
            r, _ = run_arm(d, U, t_plus_1(month_sessions(cal, [4 * k]), cal), 0.0, lag=1, kind="control")
            S["shipped"].append([cagr_w(r, a, z) for _, a, z in ERAS])
            Wk = t_plus_1(month_sessions(cal, [k + 5 * j for j in range(5)]), cal)
            for v, band in VARIANTS.items():
                r, _ = run_arm(d, U, Wk, band, lag=1)
                S[v].append([cagr_w(r, a, z) for _, a, z in ERAS])
        P = {k: np.array(x) for k, x in S.items()}
        res[u]["posthoc_t1_phases"] = P
        print(f"  {u} T+1 5-phase: shipped " + " | ".join(f"{l} {x:.2f}" for l, x in zip(labs, P["shipped"].mean(0))),
              flush=True)
        for v in VARIANTS:
            dd = P[v] - P["shipped"]
            print(f"      {v}: " + " | ".join(f"{l} {x:.2f} ({y:+.2f}, phases {lo:+.2f}..{hi:+.2f}, {int((q < 0).sum())}/5 below)"
                                      for l, x, y, lo, hi, q in zip(labs, P[v].mean(0), dd.mean(0), dd.min(0), dd.max(0),
                                                                    dd.T)), flush=True)
    d, _, U = load_pit()
    cal = window(d, PIT_START)
    out = {}
    for conv, lag in (("same", 0), ("t1", 1)):
        mv = (lambda x: t_plus_1(x, cal)) if lag else (lambda x: x)
        S = {"shipped": [res["dow_pit"]["cells"][(conv, "V1")]["E3"]["s"]]}
        S.update({v: [res["dow_pit"]["cells"][(conv, v)]["E3"]["v"]] for v in VARIANTS})
        for k in range(1, 5):
            r, _ = run_arm(d, U, mv(month_sessions(cal, [4 * k])), 0.0, lag=lag, members=PIT.member, kind="control")
            S["shipped"].append(cagr_w(r, E3[1], E3[2]))
            Wk = mv(month_sessions(cal, [k + 5 * j for j in range(5)]))
            for v, band in VARIANTS.items():
                r, _ = run_arm(d, U, Wk, band, lag=lag, members=PIT.member)
                S[v].append(cagr_w(r, E3[1], E3[2]))
        out[conv] = {k: np.array(x) for k, x in S.items()}
        P = out[conv]
        print(f"  dow_pit E3 {'same-close' if conv == 'same' else 'T+1'} 5-phase: shipped {P['shipped'].mean():.2f}  "
              + "  ".join(f"{v} {P[v].mean():.2f} ({(P[v] - P['shipped']).mean():+.2f}, phases "
                          f"{(P[v] - P['shipped']).min():+.2f}..{(P[v] - P['shipped']).max():+.2f}, "
                          f"{int(((P[v] - P['shipped']) < 0).sum())}/5 below)" for v in VARIANTS), flush=True)
    res["dow_pit"]["posthoc_phases"] = out


def main() -> None:
    res = {}
    for u in ("nifty_50", "dow_30"):
        stock_universe(u, res)
    etf(res)
    pit(res)
    print("\n== VERDICT (the bar as pre-registered)")
    res["verdict"] = verdict(res)
    posthoc_phases(res)
    res["count"] = dict(COUNT)
    print(f"  backtests: {COUNT['variant']} variant configurations, {COUNT['control']} control (shipped-arm) runs",
          flush=True)
    if CACHE:
        pickle.dump(res, open(os.path.join(CACHE, "audit_cvg_results.pkl"), "wb"))


if __name__ == "__main__":
    main()
