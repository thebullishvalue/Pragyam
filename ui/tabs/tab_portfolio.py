"""
PRAGYAM — The curated book, read through its risk structure.

Holdings first, then the structure that produced them: what each name is,
what share of capital and of variance it carries, and where the two disagree.

Author: @thebullishvalue
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from ui.components import (
    render_chart_panel,
    render_interpretation_card,
    render_note,
    render_section_header,
    render_table_panel,
)
from ui.shared import (CVG_CHIP, CVG_TONE, NCO_STYLES, REGIME_FACTOR_ORDER,
                       STYLE_LABELS, num, style_spec)
from ui.components import render_kpi_strip
from cvgrid import STATE_LABEL, STATE_UNITS, STATES
from pragati import INNER_ZONE
from samanvaya import THETA_OSC
import html as html_module

import streamlit.components.v1 as components

# Charts are optional: a missing plotly must degrade this tab to its tables
# and readouts rather than take the whole app down on import.
try:
    from charts import (
        create_cluster_correlation_heatmap,
        create_risk_allocation_heatmap,
        create_risk_contribution_chart,
        create_conviction_value_map,
    )
    CHARTS_AVAILABLE = True
except ImportError:                                     # pragma: no cover
    CHARTS_AVAILABLE = False


def _render_portfolio_tab(portfolio: pd.DataFrame, current_df: pd.DataFrame, capital: float):
    """Tab 1 — the curated book, read through its risk structure.

    This replaces the old conviction-signal overlay. That overlay described a
    score which no longer exists, and which — while it did — had no measurable
    cross-sectional predictive power on this universe (IC ~0.00-0.04, sign
    unstable across horizons). What drives the book now is the covariance
    structure, so that is what the table and heatmaps show.
    """
    _rc = st.session_state.get("run_context") or {}
    _label = style_spec(_rc)["label"]
    render_section_header(
        "Positions",
        f"{len(portfolio)} holdings · {_label}",
        icon="briefcase", accent="accent",
    )

    if portfolio is None or portfolio.empty:
        render_interpretation_card(
            title="NO PORTFOLIO",
            body="Run an analysis to curate a book.",
            color="warning")
        return

    df = portfolio.copy()
    for c in ("cluster", "risk_contribution", "volatility", "corr_to_book"):
        if c not in df.columns:
            df[c] = np.nan
    # Sorted by weight, largest first — the order a holder reads a book in.
    # (The risk heatmap below still groups by cluster, where the block structure
    # is the point.)
    df = df.sort_values("weightage_pct", ascending=False)

    n = len(df)
    eq_share = 100.0 / n if n else 0.0

    # One table primitive for the whole app. This block used to hand-build a
    # <table> with its own <style> inside a components.html iframe — 110 lines
    # of markup that could not inherit a single token from the stylesheet,
    # which is exactly why the theme switch could never reach it.
    view = pd.DataFrame({
        "Symbol": df["symbol"].astype(str),
        "Units": pd.to_numeric(df["units"], errors="coerce"),
        "Price": pd.to_numeric(df["price"], errors="coerce"),
        "Weight %": pd.to_numeric(df["weightage_pct"], errors="coerce"),
        "Value": pd.to_numeric(df["value"], errors="coerce"),
        # Cluster is a LABEL, not a quantity: "C3" is not three of anything, and
        # right-aligning it as a number would invite exactly that reading. An
        # unestimated holding has no cluster at all.
        "Cluster": ["—" if pd.isna(c) else f"C{int(c)}" for c in df["cluster"]],
        "Risk Share %": pd.to_numeric(df["risk_contribution"], errors="coerce") * 100,
        "Risk − Wt": (pd.to_numeric(df["risk_contribution"], errors="coerce") * 100
                      - pd.to_numeric(df["weightage_pct"], errors="coerce")),
        "Vol %": pd.to_numeric(df["volatility"], errors="coerce") * 100,
        "Indep": 1.0 - pd.to_numeric(df["corr_to_book"], errors="coerce").abs(),
    })
    # On a Conviction-Value Grid run the weight IS the state, so the readings that placed each
    # holding sit right after its weight — before the risk columns, which for
    # this style are a mirror, not a target. "Value tape" rather than "Value":
    # that header is already the position's rupee value.
    _uses_dh = bool((portfolio.attrs or {}).get("nco_uses_cvg", False))
    if _uses_dh:
        for c in ("state", "conviction", "value_tape", "state_days", "push_tier", "held_row",
                  "cvg_units", "state_units"):
            if c not in df.columns:
                df[c] = np.nan
        _at_w = view.columns.get_loc("Weight %") + 1
        _held = pd.to_numeric(df["held_row"], errors="coerce").fillna(0) > 0
        for off, (col, vals) in enumerate((
            ("State", [STATE_LABEL.get(str(x), "—") + (" · held" if h else "")
                       for x, h in zip(df["state"], _held)]),
            ("Map units", pd.to_numeric(df["cvg_units"], errors="coerce").to_numpy()),
            ("Push", [str(x) if isinstance(x, str) else "—" for x in df["push_tier"]]),
            ("Conv", pd.to_numeric(df["conviction"], errors="coerce").to_numpy()),
            ("Value tape", pd.to_numeric(df["value_tape"], errors="coerce").to_numpy()),
            ("Days", pd.to_numeric(df["state_days"], errors="coerce").to_numpy()),
        )):
            view.insert(_at_w + off, col, vals)
    render_table_panel(
        view, "holdings", context=f"{n} holdings · sorted by weight",
        show_index=False,
        label_col="Symbol",
        col_precision={"Units": 0, "Price": 2, "Weight %": 2, "Value": 0,
                       "Risk Share %": 2, "Risk − Wt": 2, "Vol %": 1, "Indep": 2,
                       "Conv": 0, "Value tape": 0, "Days": 0, "Map units": 2},
        sign_color_cols={"Conv"} if _uses_dh else None,
        lower_is_better_cols={"Risk − Wt", "Value tape"} if _uses_dh else {"Risk − Wt"},
        max_height=520,
    )

    render_note(
        f"**Risk Share** is each holding's contribution to portfolio variance; **Weight** is its "
        f"share of capital. Equal capital does not mean equal risk — **Risk − Wt** is that gap, "
        f"and controlling it is what this allocator does. It is the one column where lower is "
        f"better, so green marks a holding carrying *less* variance than its capital share and "
        f"red marks one carrying more. **Indep** is 1 − |correlation to the "
        f"book|, so higher means the holding diversifies rather than duplicates. An equal share "
        f"at this position count would be {eq_share:.2f}%."
        + (" **State** is the holding's cell in the Conviction-Value Grid — conviction × value — and "
           "**Map units** its weight on the graded map: the cell's units, shaded toward the "
           "neighbouring cell by how intensely each tape is drawn, exactly as the Pine grades "
           "its colours. **held** means the histogram is holding its row against the tape "
           "because the push is not yet behind the change. **Push** is the pane's histogram "
           "as drawn: which way conviction is being pushed, and whether that push is an impulse, "
           "building, decelerating or turning (\"quiet\" when the reading is an amplified calm). "
           "**Conv** is the conviction tape (green: buyers control) and **Value tape** the value "
           "tape (green: cheap, red: rich), each read on the daily and weekly frames; **Days** is "
           "how long the holding has been in its state. The risk columns are a mirror here — "
           "this style does not read the covariance." if _uses_dh else "")
    )

    if _uses_dh:
        _render_cvg_map(portfolio)

    if not CHARTS_AVAILABLE:
        return

    _at = portfolio.attrs if hasattr(portfolio, "attrs") else {}
    _rc_target = _at.get("nco_rc_target", "none")
    _mspec = style_spec(st.session_state.get("run_context") or {})
    render_section_header(
        "Risk Profile",
        f"Per-holding, row-relative · scored against {_mspec['short']}'s own objective",
        icon="activity", accent="emerald")
    render_chart_panel(create_risk_allocation_heatmap(df), "risk-alloc",
                       context=f"{len(df)} holdings · row-relative percentile")
    # What "green" means is method-dependent, so the caption must be too. Telling
    # an HRP user that green means "risk share near 1/N" would be wrong: HRP
    # balances across clusters, not holdings.
    render_note(
        "Each row is scored against its own peers, oriented so **green is the calm, "
        "diversifying end** — low volatility, high independence. "
        + ("Because this style targets **equal risk contribution**, green on the risk row "
           "means a holding sitting *at* its equal share; any red is a name the selection "
           "or the position cap pulled off target."
           if _rc_target == "equal" else
           "This style does not target equal risk contribution, so green on the risk row "
           "simply means a holding carrying *less* variance than its capital share — the gap "
           "between the Weight and Risk rows is the risk this method leaves unbalanced."
           + (" The **Conviction** and **Value** rows are the two tapes the weights were sized "
              "from — green where buyers control, and where price is cheap."
              if _at.get("nco_uses_cvg") else ""))
    )
    render_section_header(
        "Risk Contribution", "Capital share vs variance share, on one scale",
        icon="bar-chart-2", accent="accent")
    render_chart_panel(create_risk_contribution_chart(df), "risk-contrib",
                       context=f"capital vs variance · equal share {eq_share:.2f}%")
    render_note(
        {
            "equal": ("Grey is capital, coloured is variance. This style solves for **flat** "
                      "risk bars on the dashed equal-share line — the header reports the "
                      "solver's own dispersion (target 0.000) separately from the realised "
                      "figure, because top-N selection and the position cap move the held "
                      "book away from the solution."),
            "cluster": ("Grey is capital, coloured is variance. Uneven risk bars are **expected** "
                        "here: HRP balances risk across *clusters*, not across individual "
                        "holdings, so a small risk share inside a large cluster is a correct "
                        "outcome rather than an imbalance."),
        }.get(_rc_target,
              "Grey is capital, coloured is variance. This style does not manage risk "
              "contribution at all — the spread of the coloured bars is simply where the "
              "covariance puts the variance, and the distance between the two series is the "
              "argument for risk-based weighting.")
    )

    corr = portfolio.attrs.get("corr_matrix")
    labels = portfolio.attrs.get("cluster_labels")
    if corr is not None and not getattr(corr, "empty", True):
        k = portfolio.attrs.get("nco_clusters", 0)
        sil = portfolio.attrs.get("nco_silhouette", 0.0)
        _clusters_drive = bool(portfolio.attrs.get("nco_uses_clusters", False))
        render_section_header(
            "Cluster Structure",
            f"Correlation matrix ordered by cluster · {k} clusters · silhouette {sil:.2f}"
            + ("" if _clusters_drive else " · diagnostic only"),
            icon="layers", accent="violet")
        render_chart_panel(create_cluster_correlation_heatmap(corr, labels),
                           "cluster-corr",
                           context=f"{k} clusters · silhouette {sil:.2f}")
        render_note(
            "Blocks along the diagonal are groups that move together — one bet wearing several "
            "tickers. The gaps cut through the field mark the cluster boundaries. Crisp blocks "
            "mean the clustering "
            "found real structure; a uniformly warm matrix means the universe is effectively a "
            "single bet, which no allocator can fix. "
            + ("These are the boundaries the allocator actually **used** to split capital."
               if _clusters_drive else
               "This style does **not** allocate from the cluster tree — the matrix is shown so "
               "you can see the structure the weights were computed against.")
        )


def _render_cvg_map(portfolio: pd.DataFrame) -> None:
    """The grid's view of the universe: the state census, the map, the watchlist.

    The Conviction-Value Grid is sized from these readings, so this section is to it what
    Cluster Structure is to HRP — what the allocator saw. It covers the WHOLE
    universe, not just the holdings: a name at the floor, or one the book was
    not asked to hold, is part of the reading.
    """
    at = portfolio.attrs or {}
    uni = at.get("nco_cvg_universe")
    census = at.get("nco_cvg_census") or {}
    n_uni = int(sum(census.values())) or len(portfolio)
    render_section_header(
        "Conviction-Value Map",
        f"Conviction × value on D · W · {at.get('nco_cvg_names', 0)} of {n_uni} names read",
        icon="compass", accent="emerald")

    # The census, in allocation order, with each state's weight in units: the
    # whole allocation rule, readable at a glance.
    items = []
    for code, units, label, meaning in STATES:
        cnt = int(census.get(code, 0))
        if cnt == 0:
            continue          # a state earns a card only when a name is in it
        items.append({"label": label, "value": str(cnt),
                      "subtext": f"{units:g} unit{'s' if units != 1 else ''} · {meaning}",
                      "color_class": CVG_CHIP[CVG_TONE[code]]})
    render_kpi_strip(items, max_cols=5, key="cvg-census")

    if CHARTS_AVAILABLE and uni is not None and not getattr(uni, "empty", True):
        render_chart_panel(create_conviction_value_map(uni), "cvg-map",
                           context=f"{n_uni} names · filled = held, sized by weight")
    render_note(
        f"Across: the **conviction tape** — who controls, and how firmly. Up: the **value tape** — "
        f"rich or cheap against what the macro drivers and the home market explain. The dotted "
        f"lines are each tape's own knee — conviction's inner zone at ±{INNER_ZONE:.0f}, where the "
        f"Pine's tape turns from faint to clear, and value's θ at ±{THETA_OSC:.0f} — and cut the "
        f"plane into the grid's nine states. Right of the band buyers control: core weight while "
        f"price is fair or cheap, less once it is rich. Inside it control is undecided; left of "
        f"it sellers control — watched when cheap, punished to the floor when rich. The "
        f"**histogram runs the rows**: a name only changes row once the push is behind it, so a "
        f"point coloured for a region it does not sit in is a row being held. The map is "
        f"**graded**: inside its cell a name's weight moves toward the neighbouring cell by how "
        f"intensely its tapes are drawn — a tape just past its knee sits partway, a solid one "
        f"earns the full cell — so marker size varies within a region. Every name is held: the "
        f"reading sets how much, never whether."
    )

    if uni is None or getattr(uni, "empty", True):
        return
    watch = uni[uni["state"].isin(["BASING", "DISLOCATED"])]
    if watch.empty:
        return
    w = watch.assign(_c=pd.to_numeric(watch["conviction"], errors="coerce")).sort_values(
        "_c", ascending=False)
    view = pd.DataFrame({
        "Symbol": w["symbol"].astype(str),
        "State": [STATE_LABEL.get(str(x), "—") for x in w["state"]],
        "Conv": pd.to_numeric(w["conviction"], errors="coerce"),
        "To turn": INNER_ZONE - pd.to_numeric(w["conviction"], errors="coerce"),
        "Push": w["push_tier"].fillna("—").astype(str),
        "Value tape": pd.to_numeric(w["value_tape"], errors="coerce"),
        "Days": pd.to_numeric(w["state_days"], errors="coerce"),
        "Weight %": pd.to_numeric(w["weight_pct"], errors="coerce"),
        "Hedged vs": w["drivers"].fillna("—").astype(str),
    })
    render_section_header(
        "Watchlist",
        f"{len(view)} cheap without buyers in control · basing "
        f"{STATE_UNITS['BASING']:g} · dislocated {STATE_UNITS['DISLOCATED']:g} units until they turn",
        icon="eye", accent="cyan")
    render_table_panel(
        view, "cvg-watchlist", context="sorted by conviction · closest to turning first",
        show_index=False, label_col="Symbol",
        col_precision={"Conv": 0, "To turn": 0, "Value tape": 0, "Days": 0, "Weight %": 2},
        sign_color_cols={"Conv"}, lower_is_better_cols={"Value tape"},
        max_height=320,
    )
    render_note(
        f"Promotion is the grid itself. When a watchlist name's conviction clears "
        f"+{INNER_ZONE:.0f} **and** the histogram confirms a push up (not turning, not quiet), it "
        f"becomes **{STATE_LABEL['TURNED']}** if value is still cheap or "
        f"**{STATE_LABEL['BUILDING']}** if it has recovered to fair — core weight either way. "
        f"**To turn** is how far the conviction tape still has to travel; **Push** is whether the "
        f"histogram is behind it yet."
    )
