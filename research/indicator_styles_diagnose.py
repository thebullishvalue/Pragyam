"""research/indicator_styles_diagnose.py — why the indicator styles trail: rank IC with next month and correlation with past returns; gross vs net (RESULT in indicator_styles.py)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/Pragyam"); sys.path.insert(0, "/home/user/Pragyam/research")
import numpy as np, pandas as pd
import style_search as ss, style_blends as sb, strategies as ST
from scipy.stats import spearmanr
for u in ("nifty_50", "dow_30"):
    d = ss.load(u, holdout=True); snaps = sb.snapshots(u); pos = {pd.Timestamp(t): i for i, (t, _) in enumerate(snaps)}
    px = d["px"]; months = d["months"]
    rows = []
    for a, b in zip(months[:-1], months[1:]):
        snap = snaps[pos[a]][1]; syms = list(snap.drop_duplicates("symbol", keep="last")["symbol"])
        p0 = px.loc[a].reindex(syms); p1 = px.loc[b].reindex(syms)
        i = px.index.get_loc(a); pm = px.iloc[max(0, i - 21)].reindex(syms); p12 = px.iloc[max(0, i - 252)].reindex(syms)
        fwd, past1 = p1 / p0 - 1, p0 / pm - 1
        mom121 = pm / p12 - 1
        for m in ("MMAST", "VSURF"):
            sc = ST.strategy_scores([(a, snap)], syms, m)["score"]
            ok = fwd.notna() & sc.notna() & past1.notna()
            if ok.sum() < 10: continue
            rows.append(dict(m=m, a=a, ic=spearmanr(sc[ok], fwd[ok])[0], c_past1=spearmanr(sc[ok], past1[ok])[0],
                             c_mom121=spearmanr(sc[ok & mom121.notna()], mom121[ok & mom121.notna()])[0]))
        ok = fwd.notna() & past1.notna()
        rows.append(dict(m="past 1-month return", a=a, ic=spearmanr(past1[ok], fwd[ok])[0], c_past1=1.0, c_mom121=np.nan))
        ok2 = fwd.notna() & mom121.notna()
        rows.append(dict(m="12-1 momentum", a=a, ic=spearmanr(mom121[ok2], fwd[ok2])[0], c_past1=np.nan, c_mom121=1.0))
    D = pd.DataFrame(rows)
    g = D.groupby("m").agg(ic=("ic", "mean"), ic_sd=("ic", "std"), n=("ic", "size"), corr_past1m=("c_past1", "mean"), corr_12_1=("c_mom121", "mean"))
    g["ic_t"] = g["ic"] / (g["ic_sd"] / np.sqrt(g["n"]))
    print(f"== {u}: mean monthly rank IC with next month's return, and the score's rank correlation with past returns")
    print(g[["ic", "ic_t", "corr_past1m", "corr_12_1", "n"]].round(3).to_string())
    # gross vs net
    R = {}
    for m in ("MMAST", "VSURF"):
        W = {a: sb.raw([(a, snaps[pos[a]][1])], m) for a in months[:-1]}
        r = ss.run(lambda c, W=W: W[c.date], d)
        g_ = r["gross"]; nav = (1 + g_).prod() ** (12 / len(g_)) - 1
        print(f"   {m}: gross CAGR {nav*100:.2f} · net {ss.metrics(r)['cagr']:.2f} · cost drag {nav*100 - ss.metrics(r)['cagr']:.2f} %/yr · turnover {r['to'].mean()*12:.2f}x")
