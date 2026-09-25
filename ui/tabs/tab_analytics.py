"""
PRAGYAM — The book against a benchmark, its own shadow, and the other styles.

Three comparisons, one question each. The benchmark answers "did it beat the
market?". The equal-weight shadow of the SAME holdings answers the narrower
and more actionable "with the names fixed, did the weights help?". The other
styles' books — each curated from this run's own inputs — answer "would a
different style have done better from this date?", with how far each book
overlaps this one and whether the gap clears the noise stated beside it.

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
    render_kpi_strip,
    render_note,
    render_section_header,
    render_table_panel,
)
from ui.shared import NCO_STYLES, REGIME_FACTOR_ORDER, STYLE_LABELS, num, style_spec
from nco import METHOD_ORDER, METHOD_SPECS
import html as html_module
from datetime import date, datetime

import streamlit.components.v1 as components

from logger_config import get_console

# Charts are optional: a missing plotly must degrade this tab to its tables
# and readouts rather than take the whole app down on import.
try:
    from charts import (create_benchmark_comparison_chart)
    CHARTS_AVAILABLE = True
except ImportError:                                     # pragma: no cover
    CHARTS_AVAILABLE = False

log = get_console()

# The equal-weight shadow's name everywhere it appears. It must not read
# "Equal Weight": that is a STYLE, whose book can hold different names, and
# both now sit on the same chart.
SHADOW_LABEL = "EW Shadow"

# The dash each style's book is drawn with as a comparison line. Fixed per
# style rather than by position, so a style looks the same whichever one ran.
PEER_DASH = {"EQUAL": "solid", "ERC": "dashdot", "HRP": "longdash", "CVG": "longdashdot"}

# Below this many daily returns the gap between two books is not read for
# noise at all: a t-statistic over a couple of weeks is itself mostly noise.
NOISE_MIN_DAYS = 20
# |t| at or above this and the gap is called beyond noise.
NOISE_T = 2.0

PeerUnits = Tuple[Tuple[str, Tuple[str, ...], Tuple[float, ...]], ...]


@st.cache_data(ttl=1800, show_spinner=False)
def _analytics_series_cached(
    symbols: Tuple[str, ...], units: Tuple[float, ...],
    anchor_iso: str, days_back: int,
    bench_ticker: str, bench_name: str,
    alt_units: Optional[Tuple[float, ...]] = None,
    peer_units: Optional[PeerUnits] = None,
):
    """Cached wrapper around analytics.build_return_series.

    Keyed on the exact (symbols, units, anchor, benchmark, alt_units,
    peer_units) tuple so the yfinance fetch runs ONCE per unique window and
    every subsequent render/tab-switch hits cache — no repeated downloads.
    Returns (port_value, port_returns, bench_returns, err, unpriced, alt_value,
    bench_value, peers). Compute stays in analytics.py; caching lives here (the
    Streamlit boundary), mirroring _load_historical_data / _detect_regime_cached.

    `alt_units` is the equal-weight shadow book over the SAME symbols. It rides
    along on this one call (rather than a second cached call with a different
    unit vector) so the comparison costs zero extra downloads and both series
    are guaranteed to share one price panel and one start date.

    `peer_units` is ``((style code, symbols, units), ...)`` — the other styles'
    books from this run. Their names join the same download for the same
    reason.
    """
    from analytics import build_return_series
    _port = pd.DataFrame({"symbol": list(symbols), "units": list(units)})
    anchor_dt = datetime.fromisoformat(anchor_iso)
    _alt = dict(zip(symbols, alt_units)) if alt_units else None
    _peers = {code: dict(zip(syms, us)) for code, syms, us in (peer_units or ())}
    # Logged from INSIDE the cached body, so the terminal shows this step only
    # when it actually costs a download. A line on every rerun would say nothing
    # about the run and bury the lines that do.
    with log.task("Performance series",
                  f"{days_back}d · {len(symbols)} holdings · vs {bench_name}") as _t:
        _t.item("Anchor", anchor_dt.strftime("%Y-%m-%d"))
        if _alt:
            _t.detail("valuing the equal-weight shadow book on the same price panel")
        if _peers:
            _t.detail(f"valuing {len(_peers)} other style book(s) on the same price panel")
        result = build_return_series(
            _port, days_back, bench_ticker, bench_name,
            anchor_date=anchor_dt, alt_quantities=_alt, peer_quantities=_peers,
        )
        _port_value, _, _bench_returns, _err, _unpriced, _, _, _peer_out = result
        if _err:
            _t.fail(_err)
        else:
            for _code, _entry in _peer_out.items():
                if _entry.get("reason"):
                    _t.note(f"{METHOD_SPECS[_code]['label']} not compared — {_entry['reason']}")
            if _unpriced:
                # A dropped holding under-represents the book rather than
                # failing it, which is exactly the kind of quiet distortion that
                # has to be named.
                _t.note(f"{len(_unpriced)} holding(s) could not be priced: "
                        + ", ".join(str(s) for s in _unpriced[:8])
                        + (" …" if len(_unpriced) > 8 else ""))
            if _bench_returns is None:
                _t.note(f"no {bench_name} series — the benchmark comparison will be empty")
            _n_points = len(_port_value) if _port_value is not None else 0
            _t.item("Window", f"{_port_value.index[0]:%Y-%m-%d} → {_port_value.index[-1]:%Y-%m-%d}"
                    if _n_points else "empty")
            _t.ok(f"{_n_points} daily points"
                  + (f" · book {float(_port_value.iloc[-1] / _port_value.iloc[0] - 1):+.2%}"
                     if _n_points > 1 and float(_port_value.iloc[0]) > 0 else ""))
        return result


def _reads_covariance(method: str) -> bool:
    """Whether a style's weight formula reads the covariance (registry flag)."""
    return bool(style_spec(method).get("needs_covariance", True)) if method else False


def _book_weights(book: pd.DataFrame) -> Dict[str, float]:
    """Capital share per name as HELD at the anchor — units × price after
    integer-lot flooring, not the target weight the lots were cut from."""
    if "value" in book.columns:
        v = pd.to_numeric(book["value"], errors="coerce")
    else:
        v = (pd.to_numeric(book["units"], errors="coerce")
             * pd.to_numeric(book["price"], errors="coerce"))
    vals = v.fillna(0.0).to_numpy(dtype=float)
    total = float(vals.sum())
    if total <= 0:
        return {}
    return {str(s): float(x) / total for s, x in zip(book["symbol"], vals)}


def _gap_read(book_r: pd.Series, peer_r: pd.Series) -> Tuple[float, str]:
    """The t-statistic of the mean daily return gap (book minus peer), and
    how to read it.

    The gap is set against how far the two books drift apart day to day — the
    standard error of their mean daily difference — so two books that move
    together can clear the bar with a small gap, and two that wander apart
    cannot clear it with a large one. Identical books have no gap to test.
    """
    d = (book_r - peer_r).dropna()
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    if np.isfinite(sd) and sd < 1e-10:
        return float("nan"), "same book"
    if n < NOISE_MIN_DAYS or not np.isfinite(sd):
        return float("nan"), f"under {NOISE_MIN_DAYS} days"
    t = float(d.mean()) / (sd / np.sqrt(n))
    return t, ("beyond noise" if abs(t) >= NOISE_T else "within noise")


def _render_style_comparison(
    *, portfolio: pd.DataFrame, style: str, method: str, m: Dict[str, Any],
    port_returns: pd.Series, bench_returns: Optional[pd.Series], rf: float,
    peer_books: Dict[str, pd.DataFrame], peer_out: Dict[str, Dict[str, Any]],
    has_peers_key: bool, peer_notes: Dict[str, str], has_shadow: bool,
    requested: int,
) -> None:
    """Style Comparison — this book against the book each OTHER style builds
    from the same run.

    The third question the tab answers, after the market and the shadow:
    "would a different style have done better from this date?". Each peer was
    curated at run time from this run's inputs, so the style is the one thing
    that differs; each is priced on this book's calendar from its first date.

    Two columns keep the table from crowning a winner it cannot support.
    OVERLAP is the capital two books hold in common: at a high overlap they
    cannot end far apart, whatever the style names suggest. The t column sets
    the gap against the two books' day-to-day drift apart, so a window of a
    few weeks reads as the noise it almost always is.
    """
    from analytics import compute_metrics

    if not has_peers_key:
        # A book curated before this comparison existed carries no peers; say
        # so rather than show an empty section that reads as "no other style
        # could build a book".
        render_section_header("Style Comparison",
                              f"{style} vs the book each other style builds from the same run",
                              icon="layers", accent="violet")
        render_note("This book was curated before the style comparison existed. Run the "
                    "analysis again to compare it with the book each other style builds.")
        return
    if not peer_books and not peer_notes:
        return

    render_section_header("Style Comparison",
                          f"{style} vs the book each other style builds from the same run",
                          icon="layers", accent="violet")

    w_book = _book_weights(portfolio)
    n_book = len(w_book)
    book_ret = float(m.get("total_return", 0.0))
    rows: List[Dict[str, Any]] = [{
        "Style": style,
        "Return": book_ret,
        "Volatility": m.get("volatility", np.nan),
        "Sharpe": m.get("sharpe", np.nan),
        "Max DD": m.get("max_drawdown", np.nan),
        "Book Edge": np.nan,
        "t": np.nan,
        "Read": "this book",
        "Overlap": 100.0,
        "Shared": f"{n_book}/{n_book}",
    }]
    left_out: List[str] = []
    partial: List[str] = []
    for code in METHOD_ORDER:
        if code == method:
            continue
        label = html_module.escape(str(METHOD_SPECS[code]["label"]))
        if code in peer_notes:
            left_out.append(f"**{label}** built no book ({html_module.escape(peer_notes[code])})")
            continue
        if code not in peer_books:
            continue
        entry = peer_out.get(code) or {}
        v = entry.get("value")
        if v is None or len(v) < 2 or float(v.iloc[0]) == 0:
            left_out.append(f"**{label}** "
                            f"({html_module.escape(entry.get('reason') or 'not valued over this window')})")
            continue
        r = v.pct_change(fill_method=None).dropna()
        pm = compute_metrics(r, bench_returns, rf)
        ret = (float(v.iloc[-1]) / float(v.iloc[0]) - 1.0) * 100.0
        t, read = _gap_read(port_returns, r)
        w_peer = _book_weights(peer_books[code])
        rows.append({
            "Style": str(METHOD_SPECS[code]["label"]),
            "Return": ret,
            "Volatility": pm.get("volatility", np.nan),
            "Sharpe": pm.get("sharpe", np.nan),
            "Max DD": pm.get("max_drawdown", np.nan),
            "Book Edge": book_ret - ret,
            "t": t,
            "Read": read,
            "Overlap": sum(min(w_book.get(s, 0.0), w) for s, w in w_peer.items()) * 100.0,
            "Shared": f"{len(set(w_book) & set(w_peer))}/{len(w_peer)}",
        })
        if entry.get("unpriced"):
            partial.append(f"**{label}** without "
                           + html_module.escape(", ".join(entry["unpriced"][:6]))
                           + (" …" if len(entry["unpriced"]) > 6 else ""))

    if len(rows) > 1:
        render_table_panel(
            pd.DataFrame(rows), "style-comparison",
            context=f"Same run inputs · {requested} positions requested · held from the anchor",
            meta=f"{len(port_returns)} trading days",
            show_index=False,
            label_col="Style",
            precision=2,
            col_precision={"t": 1, "Overlap": 0},
            sign_color_cols={"Book Edge"},
            col_labels={"Return": "Return %", "Volatility": "Vol %", "Max DD": "Max DD %",
                        "Book Edge": "Book edge %", "Overlap": "Overlap %",
                        "Shared": "Names shared"},
            max_height=240,
        )

    _style_h = html_module.escape(style)
    # Why the Equal Weight STYLE and the EW Shadow can hold different names.
    # Two causes, and the note names the one that applies: a covariance style
    # cannot hold names without an estimate while 1/N holds every priced name
    # (Equal Weight's names are a strict superset), or the position count is
    # below the universe and Equal Weight keeps the first names in listing
    # order where this style chose its own.
    _ew_names = set(_book_weights(peer_books["EQUAL"])) if "EQUAL" in peer_books else set()
    _ew_why = ""
    if has_shadow and _ew_names and _ew_names != set(w_book):
        if _ew_names > set(w_book) and _reads_covariance(method):
            _ew_why = (f"it holds every priced name, while {_style_h} can hold only names "
                       f"with enough history for a covariance estimate")
        else:
            _ew_why = ("below the universe's size it keeps the first names in listing "
                       "order, where this style chose its own")
    render_note(
        f"Each row is the book that style builds from this run's own inputs — the same "
        f"date, universe, prices, {requested} positions requested, capital and cap — held "
        f"unchanged from "
        f"the anchor and priced on this book's calendar. **Book edge** is this book's return "
        f"minus that style's, so green means {_style_h} did better. **t** sets the daily gap "
        f"between the two books against how far they drift apart day to day: under "
        f"{NOISE_T:g} it is *within noise*, and under {NOISE_MIN_DAYS} trading days it is not "
        f"read at all. **Overlap** is the capital the two books hold in common, so books "
        f"that overlap heavily cannot end far apart."
        + (f" **Equal Weight** is the style as it would have run, so it holds different "
           f"names from the **{SHADOW_LABEL}**, which splits this book's own names 1/N: "
           f"{_ew_why}." if _ew_why else "")
        + (" Valued on the priced remainder: " + "; ".join(partial) + "." if partial else "")
        + (" Not compared: " + "; ".join(left_out) + "." if left_out else "")
    )
    # One window from one date is a single draw. The record each style was
    # chosen or rejected on is the long run, and it belongs beside the window
    # so the window is not read as a ranking.
    render_note(
        "**One window from one date ranks nothing.** The long-run record against Equal "
        "Weight, from monthly rebalancing over years on three universes:"
        + "".join(f"<br>**{html_module.escape(str(METHOD_SPECS[c]['label']))}** · "
                  f"{html_module.escape(str(METHOD_SPECS[c].get('long_run', '')))}"
                  for c in METHOD_ORDER if METHOD_SPECS[c].get("long_run"))
    )


def _render_analytics_tab(portfolio: pd.DataFrame):
    """Tab — Portfolio Analytics: track the curated book vs a universe-matched
    benchmark (adapted from the SWING Analysis engine, re-themed to Obsidian Quant).

    Anchored to the analysis date (metrics run anchor → today). Shows a normalized
    portfolio-vs-benchmark chart plus risk-adjusted, risk, and benchmark-comparison
    metric cards. Uses the LIVE curated portfolio (no upload); the yfinance fetch is
    cached (see _analytics_series_cached).

    The chart carries a THIRD line on every style but Equal Weight: the same
    selected names weighted 1/N (the EW Shadow). The benchmark measures the book
    against the market; the shadow measures the weighting decision alone, with
    selection held fixed. Omitted on Equal Weight runs, where it would duplicate
    the portfolio line.

    The other styles' books follow as legend-only lines and a Style Comparison
    table: each curated from this run's inputs at run time (run_context
    "peers"), valued off the same download and calendar as this book.
    """
    from analytics import (CAGR_MIN_DAYS, resolve_benchmark, resolve_risk_free_rate,
                           compute_metrics)
    from charts import create_benchmark_comparison_chart

    # Scope comes from the FROZEN run_context — the universe this book was
    # actually curated under. Browsing the sidebar after a run must not resolve
    # the benchmark against a different universe than the holdings came from.
    _ctx = st.session_state.get("run_context") or {}
    universe = _ctx.get("universe") or st.session_state.get("selected_universe") or "default"
    selected_index = _ctx.get("selected_index") or st.session_state.get("selected_index")
    bench_ticker, bench_name = resolve_benchmark(universe, selected_index)
    RISK_FREE_RATE = resolve_risk_free_rate(bench_ticker)

    # Guard: needs a curated portfolio with priced units.
    if portfolio is None or portfolio.empty or "symbol" not in portfolio.columns or "units" not in portfolio.columns:
        render_interpretation_card(
            title="NO CURATED PORTFOLIO",
            body=(
                "Run an analysis first — analytics track the live curated portfolio's "
                "performance against the benchmark, so there is nothing to measure yet."
            ),
            color="warning",
        )
        return

    # ── ANCHOR = the analysis date THIS PORTFOLIO was curated under (frozen in
    #    run_context — see _intel_context's docstring), NOT the live sidebar
    #    date picker. Browsing the sidebar to a different date after a run must
    #    not silently re-anchor the already-curated book's performance window.
    #    Metrics run anchor -> today; the window is dictated by the anchor (no
    #    user timeframe picker). Handle edge cases. ──
    _run_ctx = st.session_state.get("run_context") or {}
    _sel = _run_ctx.get("anchor_date") or st.session_state.get("selected_date")
    anchor_date = _sel if isinstance(_sel, date) else (
        _sel.date() if isinstance(_sel, datetime) else datetime.now().date()
    )
    today = datetime.now().date()

    # Edge: anchor is today or in the future → no forward history to measure.
    if anchor_date >= today:
        render_interpretation_card(
            title="ANCHORED TO TODAY",
            body=(
                f"The analysis date is <strong>{anchor_date.strftime('%d %b %Y')}</strong>, so there "
                "is no forward performance history yet. Analytics measure the curated book from the "
                "analysis date to the present — pick an earlier analysis date (with at least a few "
                "trading days elapsed) to see metrics."
            ),
            color="info",
        )
        return

    _elapsed_days = (today - anchor_date).days
    # Fetch enough calendar days to cover the anchor window (+buffer for alignment);
    # build_return_series then clips precisely to anchor → today.
    days_back = _elapsed_days + 5
    anchor_dt = datetime.combine(anchor_date, datetime.min.time())

    # ── Fetch + compute (CACHED) ───────────────────────────────────────────────
    #  The heavy yfinance fetch is behind _analytics_series_cached, keyed on the
    #  (symbols, units, anchor, benchmark) tuple, so it runs at most ONCE per
    #  unique window and every tab-switch / cosmetic rerun hits cache. Metrics
    #  render immediately on opening the tab — a scoped spinner only shows during
    #  the genuine first (cache-miss) fetch.
    _symbols = tuple(str(s) for s in portfolio["symbol"].tolist())
    _units = tuple(float(u or 0) for u in portfolio["units"].tolist())

    # ── Equal-weight shadow book ───────────────────────────────────────────────
    # A third reference line. The benchmark answers "did the book beat the
    # market?"; this answers the narrower and more actionable question "did the
    # WEIGHTS earn their complexity?" — THIS book's holdings, same anchor, same
    # capital, split 1/N instead of by the style's own weights. Integer-lot
    # flooring included, so it is a real alternative book rather than an
    # idealized fractional one.
    #
    # It isolates the WEIGHTING, not the style: a genuine Equal Weight run is no
    # longer confined to the covariance-eligible names (see
    # nco.compute_nco_portfolio), so it can select a different set. Holding the
    # holdings fixed is what makes this a clean read on the weights.
    #
    # Suppressed on Equal Weight runs, where the trace would draw the portfolio
    # line twice.
    _style = _run_ctx.get("investment_style", "Equal Weight")
    _eq_capital = float(_run_ctx.get("capital") or st.session_state.get("capital") or 0.0)
    _alt_units: Optional[Tuple[float, ...]] = None
    # Drawn for every style EXCEPT equal weight itself, where the trace would
    # draw the portfolio line twice. Keyed off the registry code rather than the
    # display label so renaming a style cannot silently re-enable the duplicate.
    if _run_ctx.get("curation") != "EQUAL" and _eq_capital > 0 and "price" in portfolio.columns:
        _n = len(portfolio)
        _per_pos = _eq_capital / _n if _n else 0.0
        _prices = pd.to_numeric(portfolio["price"], errors="coerce")
        if _per_pos > 0 and _prices.notna().all() and (_prices > 0).all():
            _alt_units = tuple(float(np.floor(_per_pos / p)) for p in _prices)
            # An equal slice that can't buy a single share of even one name
            # makes the comparison meaningless rather than merely approximate.
            if not any(u > 0 for u in _alt_units):
                _alt_units = None

    # ── The other styles' books ────────────────────────────────────────────────
    # Curated at run time from this run's inputs and frozen in run_context
    # (see the "Comparison books" step in app.py). Registry order, so the table
    # reads the same way on every run.
    _peer_books: Dict[str, pd.DataFrame] = _run_ctx.get("peers") or {}
    _peer_units: Optional[PeerUnits] = tuple(
        (code,
         tuple(str(s) for s in _peer_books[code]["symbol"]),
         tuple(float(u or 0) for u in _peer_books[code]["units"]))
        for code in METHOD_ORDER if code in _peer_books
    ) or None

    with st.spinner(f"Loading performance history · {bench_name} benchmark…"):
        (port_value, port_returns, bench_returns, err, unpriced,
         alt_value, bench_value, peer_out) = _analytics_series_cached(
            _symbols, _units, anchor_dt.isoformat(), days_back, bench_ticker, bench_name,
            alt_units=_alt_units, peer_units=_peer_units,
        )

    if err:
        render_interpretation_card(
            title="DATA UNAVAILABLE",
            body=f"Could not build the performance series: {html_module.escape(err)}",
            color="danger",
        )
        return

    # Surface any held symbols that couldn't be priced/matched — the metrics below
    # exclude them, so the reported performance is for the priced remainder only.
    if unpriced:
        _shown = ", ".join(html_module.escape(s) for s in unpriced[:12])
        _more = f" (+{len(unpriced) - 12} more)" if len(unpriced) > 12 else ""
        render_interpretation_card(
            title="SOME HOLDINGS NOT PRICED",
            body=(
                f"<strong>{len(unpriced)}</strong> held symbol(s) could not be priced and are "
                f"<strong>excluded</strong> from these metrics: {_shown}{_more}. "
                "The performance below reflects only the priced holdings."
            ),
            color="warning",
        )

    # Edge: too few trading days since the anchor to compute meaningful metrics.
    if len(port_returns) < 2:
        render_interpretation_card(
            title="NOT ENOUGH HISTORY YET",
            body=(
                f"Only <strong>{len(port_returns)}</strong> trading day(s) have elapsed since "
                f"<strong>{anchor_date.strftime('%d %b %Y')}</strong>. Risk and benchmark metrics "
                "need at least a few daily returns — check back after more trading days, or use an "
                "earlier analysis date."
            ),
            color="warning",
        )
        return

    m = compute_metrics(port_returns, bench_returns, RISK_FREE_RATE)

    # ── Relative performance: header → anchor-window chip → normalized chart ────
    _has_alt = alt_value is not None and len(alt_value) > 1 and float(alt_value.iloc[0]) != 0
    # The other styles' value series that could be priced over this window.
    _peer_lines = [
        (code, str(METHOD_SPECS[code]["label"]), peer_out[code]["value"])
        for code in METHOD_ORDER
        if code in peer_out and len(peer_out[code]["value"]) > 1
        and float(peer_out[code]["value"].iloc[0]) != 0
    ]
    _rel_sub = (
        f"Portfolio vs {bench_name}"
        + (f" vs {SHADOW_LABEL}" if _has_alt else "")
        + " · indexed to 100"
        + (" · other styles in the legend" if _peer_lines else "")
    )
    render_section_header("Relative Performance", _rel_sub, icon="activity", accent="accent")
    # Normalize the benchmark from its PRICE series on the portfolio's own
    # calendar. (1 + returns).cumprod() would start a bar late and rebase there,
    # under-reporting the benchmark and disagreeing with the cards below.
    _bench_series = bench_value if (bench_value is not None and len(bench_value) > 1) else None
    if CHARTS_AVAILABLE and len(port_value) > 0:
        fig = create_benchmark_comparison_chart(
            port_value, _bench_series, bench_name, m.get("total_return", 0.0),
            alt_series=alt_value if _has_alt else None,
            alt_label=SHADOW_LABEL,
            peer_series=[(label, series, PEER_DASH.get(code, "dot"))
                         for code, label, series in _peer_lines],
        )
        # The anchor window belongs in the PANEL HEADER, which is the app's
        # slot for "which instrument, which window" — not in a chip and a note
        # stacked between the section header and the chart. Those two lines
        # were three near-empty rows deep before the plot started, each holding
        # one short phrase across a 1900px measure, and the panel header was
        # already saying the same thing in fewer words one row further down.
        render_chart_panel(
            fig, "benchmark",
            context=f"Anchored {anchor_date.strftime('%d %b %Y')} → today · "
                    f"rebased to 100 · vs {bench_name}",
            meta=f"{len(port_returns)} trading days · {_elapsed_days} calendar",
        )

    # Read the allocation decision out loud: the chart shows three lines, this
    # states the one number the third exists to produce — what the risk-based
    # allocator added, or cost, versus splitting the same holdings evenly.
    _peer_hint = (" The other styles' books are in the legend: click one to draw it."
                  if _peer_lines else "")
    if not _has_alt:
        render_note(
            f"All series are indexed to 100 at the anchor date, so the vertical gap between "
            f"lines is cumulative relative performance. **{bench_name}** is the market; the "
            f"portfolio line is the curated book." + _peer_hint
        )
    if _has_alt:
        _eq_ret = (float(alt_value.iloc[-1]) / float(alt_value.iloc[0]) - 1.0) * 100.0
        _edge = m.get("total_return", 0.0) - _eq_ret
        _edge_cls = "ink-long" if _edge > 0 else "ink-short" if _edge < 0 else ""
        # What to expect of the gap depends on what the style's weights are
        # FOR. A risk allocator gives up return for the volatility it removes;
        # the grid sizes by state, and measured over years its weighting ran
        # within half a percent a year of 1/N either way.
        _sspec = style_spec(_run_ctx)
        _expect = (
            "Expect this to be negative as often as not: the allocator targets risk, and the "
            "return it gives up is the price of the volatility it removes."
            if _sspec.get("needs_covariance", True) else
            "Expect this to be small and to swing either way: the grid sizes by state, not "
            "risk, and over years of monthly rebalancing its weighting ran within half a "
            "percent a year of 1/N."
            if _sspec.get("uses_cvg") else ""
        )
        # The one caption tier, which takes markup: the three emphasised values
        # are coloured by the classes that read the same tokens the chart marks
        # do, so the sentence and the lines it describes cannot disagree about
        # which green they mean.
        render_note(
            f'<strong class="ink-violet">{SHADOW_LABEL}</strong> — the same '
            f'{len(portfolio)} holdings, same anchor, same capital, split 1/N instead of by '
            f'{html_module.escape(_style)}\'s weights — returned '
            f'<strong>{_eq_ret:+.2f}%</strong>. {html_module.escape(_style)} therefore added '
            f'<strong class="{_edge_cls}">{_edge:+.2f}%</strong> on return. {_expect}'
            + _peer_hint
        )

    # ── Head-to-head comparison ───────────────────────────────────────────────
    # One table, three books, read horizontally. This replaced four stacked
    # 6-card rows: every number was present but answering "how does my book
    # compare?" meant scanning disconnected blocks and holding figures in
    # memory. Only statistics that exist for a single book go here; genuinely
    # pairwise ones (beta, capture, tracking error) follow below.
    _cagr_ok = m.get("cagr_meaningful", True)
    render_section_header(
        "Comparative Statistics",
        f"{_style}" + (f" vs {SHADOW_LABEL}" if _has_alt else "") + f" vs {bench_name}"
        + ("" if _cagr_ok else " · CAGR hidden, window too short to annualize"),
        icon="zap", accent="emerald",
    )

    _alt_m = None
    if _has_alt:
        _alt_r = alt_value.pct_change(fill_method=None).dropna()
        _alt_m = compute_metrics(_alt_r, bench_returns, RISK_FREE_RATE)
        _alt_total = (float(alt_value.iloc[-1]) / float(alt_value.iloc[0]) - 1.0) * 100.0
    _bench_m = None
    if bench_returns is not None and len(bench_returns) > 2:
        _bench_m = compute_metrics(bench_returns, bench_returns, RISK_FREE_RATE)

    def _col(metric, fmt="{:+.2f}%", src=None):
        vals = []
        for mm in (m, _alt_m, _bench_m):
            if mm is None:
                vals.append(None)
            else:
                vals.append(mm.get(metric))
        if src is not None:
            vals[1] = src
        return vals

    # (row label, metric key, format, higher_is_better, show?)
    rows_spec = [
        ("Period Return",   "total_return",  "{:+.2f}%", True,  True),
        ("CAGR",            "cagr",          "{:+.2f}%", True,  _cagr_ok),
        ("Volatility",      "volatility",    "{:.2f}%",  False, True),
        ("Sharpe",          "sharpe",        "{:.2f}",   True,  True),
        ("Sortino",         "sortino",       "{:.2f}",   True,  True),
        ("Max Drawdown",    "max_drawdown",  "{:.2f}%",  True,  True),
        ("Calmar",          "calmar",        "{:.2f}",   True,  _cagr_ok),
        ("VaR (95%)",       "var_95",        "{:.2f}%",  True,  True),
        ("CVaR (95%)",      "cvar_95",       "{:.2f}%",  True,  True),
        ("Win Rate",        "win_rate",      "{:.0f}%",  True,  True),
    ]

    # Built as a frame and handed to the one table primitive. This was 60
    # lines of hand-built <table> markup with its own <style> inside an
    # iframe — a second table system, in a second typeface, that no token
    # could reach. `best_in_row` keeps the one thing that markup did carry:
    # the winning cell per row, with the polarity stated per row because this
    # table mixes returns with drawdowns.
    # THE SHADOW COLUMN ONLY EXISTS WHEN THERE IS A SHADOW.
    #
    # It used to be emitted unconditionally, which broke this table on the
    # DEFAULT style. On an Equal Weight run the shadow is suppressed — it would
    # be the same book twice — so that column was all-NaN, and worse, its
    # header was the string "Equal Weight", which is also `_style`. Two columns
    # with one name makes `view[c]` return a DataFrame instead of a Series, so
    # every cell rendered as a stringified pandas object:
    # "Equal Weight 1.091768 Equal Weight NaN Name: 0, dtype: object", and the
    # per-row winner compared a frame against a float.
    #
    # Dropping the column when it carries nothing fixes both at once, and the
    # dedupe below means no future style name colliding with the benchmark's
    # can reintroduce it.
    _cols = [(_style, 0)]
    if _has_alt:
        _cols.append((SHADOW_LABEL, 1))
    if _bench_m is not None:
        _cols.append((bench_name, 2))
    _seen: dict[str, int] = {}
    heads = []
    for name, _ in _cols:
        _seen[name] = _seen.get(name, 0) + 1
        heads.append(name if _seen[name] == 1 else f"{name} ({_seen[name]})")

    hh_rows, hh_polarity, hh_precision = [], [], {}
    for label, key, fmt, hib, show in rows_spec:
        if not show:
            continue
        vals = _col(key)
        # Equal-weight period return comes from the shadow series directly, so
        # it matches the chart legend exactly rather than being recomputed.
        if key == "total_return" and _has_alt:
            vals[1] = _alt_total
        hh_rows.append([label] + [
            (vals[i] if (vals[i] is not None and np.isfinite(vals[i])) else np.nan)
            for _, i in _cols])
        hh_polarity.append(bool(hib))
        hh_precision[label] = 0 if fmt.endswith("{:.0f}%") else 2

    hh = pd.DataFrame(hh_rows, columns=["Metric"] + heads)
    render_table_panel(
        hh, "head-to-head",
        context=" · ".join(heads),
        show_index=False,
        label_col="Metric",
        precision=2,
        best_in_row=hh_polarity,
        max_height=560,
    )
    render_note(
        f"Green marks the best value in each row. **{_style}** is the curated book"
        + (f"; **{SHADOW_LABEL}** is the same {len(portfolio)} holdings split 1/N — the "
           "like-for-like test of the weights" if _has_alt else "")
        + f"; **{bench_name}** is the market. Max Drawdown, VaR and CVaR are "
        f"negative numbers, so *higher is better* — the least negative wins."
        + (" Expect the allocator to lead on volatility and drawdown while trailing on "
           "return: that is the trade it makes, not a fault."
           if style_spec(_run_ctx).get("needs_covariance", True) else "")
    )

    _render_style_comparison(
        portfolio=portfolio, style=_style, method=str(_run_ctx.get("curation") or ""),
        m=m, port_returns=port_returns, bench_returns=bench_returns, rf=RISK_FREE_RATE,
        peer_books=_peer_books, peer_out=peer_out, has_peers_key="peers" in _run_ctx,
        peer_notes=_run_ctx.get("peer_notes") or {}, has_shadow=_has_alt,
        requested=int(_run_ctx.get("num_positions") or len(portfolio)),
    )

    # ── Relationship to benchmark ─────────────────────────────────────────────
    # These have no meaning for a single book — every one is a statistic ABOUT
    # the pairing — so they cannot live in the table above.
    render_section_header("Benchmark Relationship", f"How the book moves with {bench_name}",
                          icon="compass", accent="cyan")
    # One strip, not six hand-placed columns: the strip owns the wrapping rule,
    # so this row reflows to two rows of three on a tablet instead of six
    # columns squeezed to 90px each.
    _b = m.get("beta", 1)
    # Alpha reports the ANNUALIZED CAPM residual once the window can carry one,
    # and the PERIOD residual before that. Both answer "did the book beat what
    # its beta entitled it to"; only the first also claims a per-annum rate,
    # which is the part a six-week window cannot support. The card previously
    # printed a dash and "Window too short" for the whole metric, which read as
    # a broken feature rather than a withheld annualization — the residual was
    # always computable and is the number a reader actually wants.
    # alpha_days is 0 only when the benchmark series never overlapped the book's
    # window — no pairing, so no residual. That is the one case with genuinely
    # nothing to print, and it is a different statement from a short window.
    _a_days = int(m.get("alpha_days", 0) or 0)
    _a_paired = _a_days > 0
    _a_annual = _a_paired and _cagr_ok and _a_days >= CAGR_MIN_DAYS
    _a = m.get("alpha", 0) if _a_annual else m.get("alpha_period", 0)
    _uc = m.get("up_capture", 100)
    _dc = m.get("down_capture", 100)
    render_kpi_strip([
        {"label": "Beta", "value": f"{_b:.2f}", "subtext": "Market sensitivity",
         "color_class": "warning" if _b > 1.2 else "info" if _b < 0.8 else "neutral"},
        {"label": "Alpha" if _a_annual or not _a_paired else "Alpha · Period",
         "value": f"{_a:+.2f}%" if _a_paired else "—",
         "subtext": ("CAPM excess, annualized" if _a_annual else
                     f"CAPM excess over {_a_days} trading days · "
                     f"annualized from {CAGR_MIN_DAYS}") if _a_paired else
                    f"No overlap with {bench_name}",
         "color_class": ("success" if _a > 0 else "danger" if _a < 0 else "neutral")
                        if _a_paired else "neutral"},
        {"label": "Correlation", "value": f"{m.get('correlation', 0):.2f}",
         "subtext": f"R² {m.get('r_squared', 0):.2f}", "color_class": "info"},
        {"label": "Tracking Error", "value": f"{m.get('tracking_error', 0):.1f}%",
         "subtext": "Annualized", "color_class": "info"},
        {"label": "Up Capture", "value": f"{_uc:.0f}%", "subtext": "In rising markets",
         "color_class": "success" if _uc > 100 else "warning"},
        {"label": "Down Capture", "value": f"{_dc:.0f}%", "subtext": "In falling markets",
         "color_class": "success" if _dc < 100 else "danger"},
    ], max_cols=6, key="benchmark-rel")
    render_note(
        f"**Beta** is the book's sensitivity to {bench_name}; **Alpha** is return beyond what that "
        f"beta explains"
        + ("." if _a_annual or not _a_paired else
           f" — stated here **over the window itself**, not per annum, because "
           f"{_a_days} trading days is under the {CAGR_MIN_DAYS} an annualized figure needs. "
           f"Extrapolating a window this short multiplies its noise by the same factor it "
           f"multiplies the return, so the rate is withheld while the excess it is built from "
           f"is not.")
        + f" **Up/Down Capture** are the share of the benchmark's rise and fall the book "
        f"participates in — the ideal pairing is above 100% up and below 100% down. **Tracking "
        f"Error** is the volatility of the difference, so it measures how far the book is allowed "
        f"to wander from the market, not whether it wandered profitably."
    )
