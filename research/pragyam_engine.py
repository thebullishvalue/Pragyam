"""
research/pragyam_engine.py — the Pragyam risk engine, copied verbatim from
thebullishvalue/Indicator research/pragyam.py (commit f0a82d4), the reference implementation of
pragyam.pine v2.6. VAAYDO carries the same copy (vaaydo/pragyam.py). Nothing below is changed;
research/pragyam_panels.py runs it per symbol for the v14 style search.

Pragyam's own research: single-chart direction is not forecastable, risk is (next-10-bar variance
out of sample, log R² 0.71; QLIKE 0.354 against 0.509 for a 20-bar standard deviation).
"""
import numpy as np, pandas as pd
from statistics import NormalDist
WARM, RN = 252, 500
def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values           # ta.ema: alpha 2/(n+1), seeded with the first value
def lag(x, j): return np.r_[np.full(j, np.nan), x[:-j]]
def run(d, H=10, cover=0.80, pstop=0.20, eta=0.02, cap=1.0, floor=0.5, power=1.0, tau=0.10, cost=0.0005, conformal=True, regimes=True, trendf=True, ladf=False, band=0.10):
    """d: DataFrame open/high/low/close. floor=1 -> no trend throttle. Returns every series the script draws."""
    o, h, l, c = (d[k].values.astype(float) for k in ("open", "high", "low", "close")); n = len(c)
    lc, lo = np.log(c), np.log(o); r = np.r_[np.nan, np.diff(lc)]
    gp = lo - np.r_[np.nan, lc[:-1]]; co = lc - lo; hl = np.log(h) - np.log(l)
    gk = gp ** 2 + 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    v = np.where((hl > 0) & (gk > 0), gk, r ** 2)                                      # Garman-Klass with the gap; r^2 where the range is absent
    S, M, L, rL = ema(v, 5), ema(v, 22), ema(v, 252), ema(r ** 2, 252)
    k = np.clip(rL / L, 0.5, 2.0)
    sig = np.sqrt(k * (0.25 * S + 0.40 * M + 0.35 * L)); sig[:WARM] = np.nan; sig[~(sig > 0)] = np.nan   # per-bar risk forecast
    slope = 0.5 * np.log(S / L); slope[np.isnan(sig)] = np.nan                         # risk momentum: short-horizon risk over long, log
    # risk regime: ta.percentrank(sig, RN) cut in thirds - each regime keeps its own widths (Mondrian conformal)
    reg = np.ones(n, dtype=int); prk = np.full(n, np.nan)
    if n > RN:
        W = np.lib.stride_tricks.sliding_window_view(sig, RN + 1)
        p = (W[:, :-1] <= W[:, -1:]).sum(axis=1) / RN * 100.0; bad = np.isnan(W).any(axis=1)
        prk[RN:] = np.where(bad, np.nan, p)
        if regimes: reg[RN:] = np.where(bad, 1, np.where(p < 100 / 3, 0, np.where(p < 200 / 3, 1, 2)))
    T = sum((lc - lag(lc, j)) / (sig * np.sqrt(j)) for j in (63, 126, 252)) / 3        # trend, in sigma
    g = floor + (1 - floor) * (0.5 + 0.5 * np.tanh(T))
    # the horizon ladder: every rung settles its own forecast and learns its own widths
    RG = [max(2, H // 2), H, 2 * H, 4 * H]; NR = len(RG)
    U = [(lc - lag(lc, j)) / (lag(sig, j) * np.sqrt(j)) for j in RG]
    sH = sig * np.sqrt(H)
    wS = (np.log(pd.Series(l).rolling(H).min().values) - lag(lc, H)) / lag(sH, H); wR = (np.log(pd.Series(h).rolling(H).max().values) - lag(lc, H)) / lag(sH, H)
    a = (1 - cover) / 2; z0 = NormalDist().inv_cdf(1 - a); s0 = NormalDist().inv_cdf(1 - pstop / 2)
    # learned widths, in logs: one term per risk regime plus one per trend state (and optionally per ladder state), for each side
    lz, ls = np.log(z0), np.log(s0)
    bU = [[lz] * 3 for _ in RG]; bD = [[lz] * 3 for _ in RG]; bS = [ls] * 3; bR = [ls] * 3            # by risk regime
    tU = [[0.0] * 3 for _ in RG]; tD = [[0.0] * 3 for _ in RG]; tS = [0.0] * 3; tR = [0.0] * 3        # by trend state
    dU = [[0.0] * 3 for _ in RG]; dD = [[0.0] * 3 for _ in RG]; dS = [0.0] * 3; dR = [0.0] * 3        # by ladder state
    trs = np.where(T > 0.5, 2, np.where(T < -0.5, 0, 1)) if trendf else np.ones(n, dtype=int); lads = np.ones(n, dtype=int)
    QU = np.full((NR, n), np.nan); QD = np.full((NR, n), np.nan); qS = np.full(n, np.nan); qR = np.full(n, np.nan)
    MU = np.zeros((NR, n)); MD = np.zeros((NR, n)); OK = np.zeros((NR, n)); X = np.full((NR, n), np.nan)
    mS = np.zeros(n); mR = np.zeros(n)
    st = np.full(n, -1); y = np.clip((lc - lag(lo, H - 1)) / lag(sH, H), -6, 6)        # ledger outcome: open after the signal bar -> close H bars after it
    cnt = np.zeros(9); mean = np.zeros(9); neff = np.zeros(9); last = np.full(9, -10 ** 9); gn = 0; gm = 0.0; gq = 0.0
    post = np.full(n, np.nan); zpost = np.full(n, np.nan); edge = np.full(n, np.nan)
    for t in range(n):
        for i, j in enumerate(RG):
            if t >= j and not np.isnan(U[i][t]) and not np.isnan(QU[i][t - j]):        # the forecast made j bars ago has now been settled
                qu, qd = QU[i][t - j], QD[i][t - j]; uu = U[i][t]
                MU[i][t] = uu > qu; MD[i][t] = uu < -qd; OK[i][t] = 1; X[i][t] = uu / qu if uu >= 0 else uu / qd
                if conformal:
                    jr, jt, jl = reg[t - j], trs[t - j], lads[t - j]; eu, ed = eta * (MU[i][t] - a), eta * (MD[i][t] - a)
                    bU[i][jr] += eu; bD[i][jr] += ed
                    if trendf: tU[i][jt] += eu; tD[i][jt] += ed
                    if ladf: dU[i][jl] += eu; dD[i][jl] += ed
        if OK[1][t]:
            mS[t] = wS[t] < -qS[t - H]; mR[t] = wR[t] > qR[t - H]
            if conformal:
                jr, jt, jl = reg[t - H], trs[t - H], lads[t - H]; es, er = eta * (mS[t] - pstop), eta * (mR[t] - pstop)
                bS[jr] += es; bR[jr] += er
                if trendf: tS[jt] += es; tR[jt] += er
                if ladf: dS[jl] += es; dR[jl] += er
            s = st[t - H]
            if s >= 0 and not np.isnan(y[t]):
                cnt[s] += 1; mean[s] += (y[t] - mean[s]) / cnt[s]; gn += 1
                if t - H - last[s] >= H: neff[s] += 1; last[s] = t - H                 # independent samples: signal bars at least H apart
                dlt = y[t] - gm; gm += dlt / gn; gq += dlt * (y[t] - gm)
        if ladf and OK[:, t].sum() == NR: lads[t] = 0 if MD[:, t].sum() >= 3 else 2 if MU[:, t].sum() >= 3 else 1
        if not np.isnan(sig[t]):
            jr, jt, jl = reg[t], trs[t], lads[t]
            for i in range(NR): QU[i][t], QD[i][t] = np.exp(bU[i][jr] + tU[i][jt] + dU[i][jl]), np.exp(bD[i][jr] + tD[i][jt] + dD[i][jl])
            qS[t], qR[t] = np.exp(bS[jr] + tS[jt] + dS[jl]), np.exp(bR[jr] + tR[jt] + dR[jl])
        if OK[1][t] and not np.isnan(T[t]):
            st[t] = 3 * (2 if T[t] > 0.5 else 0 if T[t] < -0.5 else 1) + (2 if MU[1][t] else 0 if MD[1][t] else 1)
            s = st[t]
            if neff[s] >= 5 and gn > 30 * H:
                se2 = (gq / (gn - 1)) / neff[s]; B = tau ** 2 / (tau ** 2 + se2); dd = mean[s] - gm
                edge[t] = dd; post[t] = B * dd; zpost[t] = dd * np.sqrt(B / se2)
    # exposure: the instrument's own average risk over its risk now, optionally throttled by trend
    star = np.nancumsum(sig) / np.maximum(np.cumsum(~np.isnan(sig)), 1)
    dialR = np.minimum(cap, (star / sig) ** power); E = dialR * g; E[np.isnan(T)] = np.nan
    if band > 0:                                                                       # act only when the dial has moved by a full step
        held = np.nan
        for t in range(n):
            if not np.isnan(E[t]) and (np.isnan(held) or abs(E[t] - held) >= band): held = E[t]
            E[t] = held if not np.isnan(E[t]) else np.nan
    ret = np.r_[np.nan, c[1:] / c[:-1] - 1]; E2 = lag(E, 2); E3 = lag(E, 3)
    pf = E2 * ret - cost * np.abs(E2 - np.nan_to_num(E3, nan=0.0)); ph = np.where(np.isnan(E2), np.nan, ret)   # decided at close t-2, filled at close t-1
    allok = OK.sum(axis=0) == NR
    out = dict(trs=trs, reg=reg, prk=prk, slope=slope, sig=sig, T=T, u=U[1], qU=QU[1], qD=QD[1], qS=qS, qR=qR, mU=MU[1], mD=MD[1], mS=mS, mR=mR, nF=OK[1], st=st, y=y,
               edge=edge, post=post, zpost=zpost, dialR=dialR, dialT=g, E=E, pf=pf, ph=ph, upper=c * np.exp(QU[1] * sH), lower=c * np.exp(-QD[1] * sH),
               stopL=c * np.exp(-qS * sH), stopS=c * np.exp(qR * sH),
               ladX=np.where(allok, X.mean(axis=0), np.nan), ladUp=np.where(allok, MU.sum(axis=0), np.nan), ladDn=np.where(allok, MD.sum(axis=0), np.nan),
               ladMom=np.where(allok, X[0] - X[NR - 1], np.nan))
    for i in range(NR): out[f"u{i}"] = U[i]; out[f"ok{i}"] = OK[i]; out[f"mU{i}"] = MU[i]; out[f"mD{i}"] = MD[i]; out[f"qU{i}"] = QU[i]; out[f"qD{i}"] = QD[i]
    return pd.DataFrame(out, index=d.index)
