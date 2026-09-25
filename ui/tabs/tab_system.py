"""
PRAGYAM — The run's settings, and the method that produced the book.

Everything here comes from the FROZEN run context - the settings this book
was actually built under, never the live sidebar.

Author: @thebullishvalue
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from ui.components import (
    render_kv_table,
    render_note,
    render_section_header,
)
from ui.shared import NCO_STYLES, REGIME_FACTOR_ORDER, STYLE_LABELS, num, style_spec
from cvgrid import STATE_LABEL, STATES
from samanvaya import DEFAULT_BASKET
from ui.theme import VERSION


def _render_system_tab(training_window: List):
    """Tab — System configuration + methodology reference (Obsidian Quant)."""
    # ── Configuration — the run's settings as a clean KV readout ───────────────
    render_section_header("Run Settings", "What this book was curated under",
                          icon="settings", accent="cyan")
    # Everything here comes from the FROZEN run_context — the settings this book
    # was actually built under, never the live sidebar. Browsing after a run
    # must not relabel a curated portfolio.
    _ctx = st.session_state.get("run_context") or {}
    _pf = st.session_state.get("portfolio")
    _at: Dict[Any, Any] = _pf.attrs if _pf is not None and hasattr(_pf, "attrs") else {}
    _style = _ctx.get("investment_style", "—")
    _spec = style_spec(_ctx)
    _max_eff_at = _at.get("max_pos_pct_eff")
    _max_eff = float(_max_eff_at if _max_eff_at is not None
                     else st.session_state.max_pos_pct)
    _max_relaxed = abs(_max_eff - st.session_state.max_pos_pct) > 1e-9
    _disp = num(_at.get("nco_rc_dispersion"))
    _solved = num(_at.get("nco_rc_dispersion_solved"))
    _conc = num(_at.get("nco_rc_concentration"))
    _vol = num(_at.get("nco_port_vol_ann"))
    # Two universes now, and the readout has to keep them apart: what the
    # allocator SPREAD CAPITAL over, and what the risk numbers were ESTIMATED
    # on. They diverge only for a style that needs no covariance (Equal Weight),
    # which is precisely the case where reporting one as the other would be a
    # lie about which names were considered. See nco.compute_nco_portfolio.
    _needs_cov = bool(_at.get("nco_needs_covariance", True))
    _cov_ok = bool(_at.get("nco_cov_estimable", True))
    _n_alloc = int(_at.get("nco_universe", 0) or 0)
    _n_est = int(_at.get("nco_estimation_universe", _n_alloc) or 0)
    _diag_excl = _at.get("nco_diagnostic_excluded") or {}
    _rc_cov = num(_at.get("nco_rc_coverage"))
    _uncovered = int(_at.get("nco_positions_uncovered", 0) or 0)
    details = {
        "Version": VERSION,
        "Portfolio Style": _style,
        "Curation Method": f"{_spec['label']} ({_spec['family']})",
        "Weight Formula": _spec["formula"],
        "Risk Clusters": ("—" if not _cov_ok else
                          f"{_at.get('nco_clusters', '—')} "
                          f"(silhouette {num(_at.get('nco_silhouette')) or 0:.2f})"
                          + ("" if _spec["uses_clusters"] else " — diagnostic only")),
        "Risk Balance": (
            f"dispersion {_disp:.3f}"
            + (f" (solved {_solved:.3f})" if _solved is not None
               and _spec["rc_target"] == "equal" else "")
            + (" · target 0.000" if _spec["rc_target"] == "equal" else " · not targeted")
            if _disp is not None else "—"),
        "Risk Concentration": (f"{_conc:.2f}x equal share" if _conc is not None else "—"),
        "Positions": (
            f"{_at.get('nco_positions_delivered', 0)} of "
            f"{_at.get('nco_positions_requested', '-')} requested"
            + ("" if not _at.get("nco_positions_short")
               else f" - {_at['nco_positions_short']} short ("
                    + ("eligible universe exhausted"
                       if _at.get("nco_short_cause") == "universe"
                       else "allocator zeroed names") + ")")),
        "Universe": (
            f"{_n_alloc} eligible"
            + (f" of {_at['nco_universe_requested']} in universe"
               if _at.get("nco_universe_requested") else "")
            + (f" · {len(_at.get('nco_universe_excluded') or {})} excluded"
               f" (<{num(_at.get('nco_coverage_required')) or 0.8:.0%} history)"
               if _at.get("nco_universe_excluded") else
               " · nothing excluded (reads no covariance)" if not _needs_cov else "")),
        "Risk Estimation": (
            f"{_n_est} of {_n_alloc} names"
            + (f" · {len(_diag_excl)} below "
               f"{num(_at.get('nco_coverage_required')) or 0.8:.0%} history"
               if _diag_excl else "")
            + (f" · covers {_rc_cov:.0%} of book weight"
               if _rc_cov is not None and _rc_cov < 0.999 else "")),
        "Estimation Window": f"{_at.get('nco_obs', 0)} daily observations",
        "Ex-ante Volatility": (f"{_vol:.2%}" if _vol is not None else "—")
                              + (f" (over {_rc_cov:.0%} of book weight)"
                                 if _vol is not None and _rc_cov is not None
                                 and _rc_cov < 0.999 else ""),
        "Max Position": f"{_max_eff*100:.1f}%" + (" (relaxed)" if _max_relaxed else ""),
        "Data Source": "yfinance (NSE)",
        "Lookback Period": f"{len(training_window)} days",
    }
    if _at.get("nco_uses_cvg"):
        # What the weights were read from, over the whole allocation universe.
        # Only on a Conviction-Value Grid run — no other style reads these, and a row for them
        # would imply otherwise.
        _census = _at.get("nco_cvg_census") or {}
        details["Grid Readings"] = (
            f"{_at.get('nco_cvg_names', 0)} of {_n_alloc} names read on both tapes"
            + ("" if _at.get("nco_cvg_applied") else " · NONE — every name unread, book is 1/N"))
        details["Grid States"] = " · ".join(
            f"{STATE_LABEL[c]} {_census[c]}" for c, *_ in STATES if _census.get(c))
        details["Histogram"] = (
            f"runs the rows · {_at.get('nco_cvg_push_read', 0)} read · "
            f"{_at.get('nco_cvg_confirm_up', 0)} confirm up · "
            f"{_at.get('nco_cvg_confirm_down', 0)} confirm down · "
            f"{_at.get('nco_cvg_unconfirmed', 0)} cannot (turning or quiet) · "
            f"{_at.get('nco_cvg_held', 0)} rows held against the tape")
        details["Map"] = ("graded — each name shaded within its cell by the tapes' drawn "
                          "intensity (faint 0.12–0.45, step, bright 0.65–1.0)"
                          if _at.get("nco_cvg_graded") else "flat cells")
        _hm = num(_at.get("nco_cvg_hedge_median"))
        details["Value Hedge"] = (
            f"{DEFAULT_BASKET} basket · median hedge applied "
            + (f"{_hm:.0%}" if _hm is not None else "—")
            + " (weighed by its own out-of-sample skill)")
    render_kv_table(details)
    if _at.get("nco_uses_cvg") and not _at.get("nco_cvg_applied"):
        render_note(
            "No name carried a calibrated reading on both tapes, so every name is UNREAD at "
            "the neutral unit and this book is Equal Weight. Each tape needs about a year of "
            "daily history per name; a panel cached before the grid's columns existed also "
            "reads this way until it is refetched."
        )
    if _uncovered:
        # Holding a name the covariance cannot see is correct for 1/N and a
        # contradiction for anything else, so say which one this is rather than
        # leaving a reader to infer it from blank cells in the holdings table.
        render_note(
            f"{_uncovered} holding(s) have less than "
            f"{num(_at.get('nco_coverage_required')) or 0.8:.0%} of the estimation window "
            f"and carry no covariance estimate. {_spec['label']} does not read one — it "
            "sizes them by the same rule as everything else — but every risk figure above, and "
            "every risk column in the holdings table, is computed WITHOUT them"
            + (f", over {_rc_cov:.0%} of book weight." if _rc_cov is not None else ".")
        )
    if not _cov_ok:
        render_note(
            "No covariance was estimable for this window, so the cluster, risk and "
            f"correlation diagnostics are unavailable. The book itself is unaffected: "
            f"{_spec['label']} does not read them."
        )
    if _max_relaxed:
        render_note(
            f"Cap relaxed from the nominal "
            f"{st.session_state.max_pos_pct*100:.0f}% because the selected position count "
            "made them mathematically infeasible (too few/many positions to satisfy both "
            "the cap and 100% allocation)."
        )

    # ── Methodology ───────────────────────────────────────────────────────────
    render_section_header("Curation Method", "How a portfolio is built, and what it targets",
                          icon="target", accent="emerald")
    _m_spec = style_spec(st.session_state.get("run_context") or {})
    # The Conviction-Value Grid is the one style that reads the tape, so the statements every other
    # style can make — "allocated from the covariance", "nothing here
    # forecasts", "why not forecast" — are not true of it, and the card says
    # what IS true instead.
    _reads_tape = bool(_m_spec.get("uses_cvg", False))
    _lede = (
        'Capital is allocated from two readings of every name, and nothing else: '
        'Pragati&rsquo;s conviction tape &mdash; who controls, and how firmly &mdash; and '
        'its value tape &mdash; rich or cheap against what the macro drivers and the home '
        'market explain. Together they place the name in a state, and the state sets its '
        'weight. The covariance is not read; the risk panels are its mirror.'
        if _reads_tape else
        'Capital is allocated from the return covariance structure. Nothing here '
        'forecasts returns &mdash; the book is built to spread risk across genuinely '
        'distinct exposures, not to predict which holding will win.'
    )
    _bound_label = "The honest bound" if _reads_tape else "Why not forecast"
    _bound_body = (
        'Reading the tape is forecasting, and Grinold\'s Fundamental Law caps what any '
        'forecast can earn here: <code>IR = IC &times; &radic;BR &times; TC</code>, ~1%/yr on '
        '~1.9 independent bets. The book is therefore held whole &mdash; every name at no less '
        'than the floor, core at 12&times; it &mdash; so a wrong reading costs weight, never a '
        'position. Read the measured sentence above as the size of what the tapes carry.'
        if _reads_tape else
        'Grinold\'s Fundamental Law caps forecast-driven excess return at '
        '<code>IR = IC &times; &radic;BR &times; TC</code>. At &rho; 0.52 these '
        '30 ETFs are only ~1.9 independent bets, so that ceiling is ~1%/yr '
        'however good the signal. Covariance is estimable where expected '
        'returns are not.'
    )
    method_html = (
        '<div class="intel-method-card">'
            '<div class="intel-method-header">'
                '<div class="intel-method-title">Curation Pipeline</div>'
                '<div class="intel-method-pill">'
                    + ('read &rarr; confirm &rarr; classify &rarr; size' if _reads_tape
                       else 'cluster &rarr; allocate &rarr; size')
                + '</div>'
            '</div>'
            '<div class="intel-method-lede">'
                + _lede
            + '</div>'
            '<div class="intel-method-grid">'

                '<div class="intel-method-tile">'
                    + ('<div class="tile-label">Read</div>'
                       '<div class="tile-body">'
                           'Two tapes per name, each on the ladder D &middot; W. '
                           '<b>Conviction</b>: <code>100 &middot; tanh(mean z)</code> of '
                           'participation-weighted agreement <code>&Sigma;c&middot;w / '
                           '&Sigma;|c|&middot;w</code>, <code>c = &Delta;C / TR</code>. '
                           '<b>Value</b>: Samanvaya&rsquo;s blend of the hedged return spread '
                           'and seven market-strength views, the hedge fitted on up to three '
                           'drivers and applied only as far as it has earned out of sample. '
                           'Weekly rungs are rebuilt from the forming week.'
                       '</div>'
                       if _reads_tape else
                       '<div class="tile-label">Cluster</div>'
                       '<div class="tile-body">'
                           'Holdings are grouped by <code>d = sqrt(0.5(1 - &rho;))</code> correlation '
                           'distance using Ward linkage, with the cluster count chosen by silhouette '
                           'score. Typically resolves to ~3 groups &mdash; matching the eigenvalue '
                           'participation ratio of the same matrix. Computed over the names carrying '
                           'at least 80% of the estimation window: a shorter-lived holding is sized, '
                           'but has no covariance to be clustered by.'
                       '</div>')
                + '</div>'

                '<div class="intel-method-tile">'
                    '<div class="tile-label">Allocate</div>'
                    '<div class="tile-body">'
                        + {
                            "EQUAL": ('Equal weight: every selected holding receives an identical '
                                      '<code>1/N</code> share, ignoring the covariance entirely. '
                                      'Because it estimates nothing, it is also not bound by the '
                                      'covariance eligibility rule &mdash; every priced symbol is '
                                      'eligible, including one too recently listed for the other '
                                      'styles to hold. Shown alongside the cluster structure so the '
                                      'risk it leaves unbalanced is visible.'),
                            "ERC": ('Equal Risk Contribution: weights are solved by cyclical '
                                    'coordinate descent so that <code>w<sub>i</sub> &times; '
                                    '(&Sigma;w)<sub>i</sub></code> is identical for every holding '
                                    '&mdash; each name contributes the same share of portfolio '
                                    'variance. Nothing is inverted, and the Risk Contribution chart '
                                    'shows directly whether the solver reached its target.'),
                            "HRP": ('Hierarchical Risk Parity: recursive bisection splits capital '
                                    'between sub-clusters in inverse proportion to their variance. '
                                    'No matrix is inverted, which is what makes it robust when '
                                    'correlations are high and the sample is short.'),
                            "CVG": ('Conviction-Value Grid, 3 &times; 3: rows are conviction &mdash; UP past '
                                       '+30, FAINT, DOWN past &minus;30 &mdash; columns are value '
                                       '&mdash; CHEAP, FAIR, RICH at &plusmn;&theta;. Units: turned '
                                       '<b>3</b>, building <b>3</b>, paid <b>1.5</b> &middot; basing '
                                       '<b>1.5</b>, idle <b>1</b>, stalling <b>0.75</b> &middot; '
                                       'dislocated <b>1</b>, fading <b>0.5</b>, distribution '
                                       '<b>0.25</b>. The pane&rsquo;s <b>histogram runs the rows</b>: '
                                       'a name changes row only when the push confirms it &mdash; on '
                                       'the side of the move, not turning, not quiet. The map is '
                                       '<b>graded</b>: inside its cell a name&rsquo;s weight moves toward '
                                       'the neighbouring cell by the tapes&rsquo; drawn intensity, as the '
                                       'Pine shades them; a held row keeps between half and all of its '
                                       'cell, by how intensely the push holding it is drawn. Every name '
                                       'is held.'),
                          }.get(_m_spec["short"] if _m_spec["short"] in ("ERC", "HRP", "EQUAL", "CVG")
                                else "EQUAL", _m_spec["formula"])
                    + '</div>'
                '</div>'

                '<div class="intel-method-tile">'
                    '<div class="tile-label">What it targets</div>'
                    '<div class="tile-body">'
                        + _m_spec["evidence"].replace("--", "&mdash;")
                    + '</div>'
                '</div>'

                '<div class="intel-method-tile">'
                    f'<div class="tile-label">{_bound_label}</div>'
                    '<div class="tile-body">'
                        + _bound_body
                    + '</div>'
                '</div>'

            '</div>'
        '</div>'
    )
    st.markdown(method_html, unsafe_allow_html=True)
