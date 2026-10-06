"""
PRAGYAM — Presentation helpers shared by the app shell and every tab.
प्रज्ञम् (Pragyam) — "Discernment / Wisdom"

The pieces of the UI layer that need to know something about the DOMAIN — how
a style is named, what an undefined figure looks like, which order the regime
factors are read in. They live here rather than in app.py because the tab
modules need them too, and a tab importing the shell it is rendered by is a
cycle waiting to happen.

Nothing here renders. Anything that emits markup belongs in ui/components.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from nco import (METHOD_ORDER, METHOD_SPECS, MMOM_FLOOR, MMOM_LAMBDA, MMOM_MIN_HISTORY,
                 MMOM_MIN_RANKED, MMOM_MIN_VOL_MONTHS, method_spec)

# Portfolio styles, derived from nco.METHOD_SPECS rather than hardcoded here.
# Every style travels the identical pipeline — same clustering diagnostics,
# same risk decomposition — so any difference on screen is the allocator and
# nothing else. Built from the registry so adding or retiring a style is a
# one-line change in nco.py.
NCO_STYLES = {str(METHOD_SPECS[k]["label"]): k for k in METHOD_ORDER}
STYLE_LABELS = list(NCO_STYLES.keys())

# The eight regime factors, in the order they are read: registry key, display
# name, and the key holding that factor's own verdict ("STRONG_UPTREND",
# "EXPANSION", …). Shared by the Regime tab and the run log so the terminal
# trace and the screen cannot drift into naming or ordering the same eight
# factors differently.
REGIME_FACTOR_ORDER = [
    ("momentum", "Momentum", "strength"),
    ("trend", "Trend", "quality"),
    ("breadth", "Breadth", "quality"),
    ("velocity", "Velocity", "acceleration"),
    ("extremes", "Extremes", "type"),
    ("volatility", "Volatility", "regime"),
    ("acceptance", "Acceptance", "state"),
    ("correlation", "Correlation", "regime"),
]


# Grid states → the app's semantic tones. One mapping for the conviction-value map, the
# holdings table and the census, so a state is the same colour everywhere. Each
# tone names the state's action (grid v8): emerald Buy (a turn, capitulation — 3
# units), cyan Accumulate (a base, a washout — 1½), amber Hold or Trim (building,
# stalling, paid), slate Wait / unread (no statement), rose Exit (distribution).
CVG_TONE = {
    "TURNED": "emerald", "DISLOCATED": "emerald",      # 3 units — a turn, capitulation
    "BUILDING": "amber", "PAID": "amber", "STALLING": "amber",
    "BASING": "cyan", "FADING": "cyan",                 # 1½ — base, washout
    "IDLE": "slate", "UNREAD": "slate",
    "DISTRIBUTION": "rose",
}
# The same, in render_chip's vocabulary.
CVG_CHIP = {"emerald": "success", "amber": "warning", "cyan": "info",
               "slate": "neutral", "rose": "danger"}


def style_spec(ctx_or_method) -> dict:
    """Registry record for a run context, a method code, or a style label."""
    if isinstance(ctx_or_method, dict):
        key = ctx_or_method.get("curation", "EQUAL")
    else:
        key = ctx_or_method
    key = NCO_STYLES.get(str(key), str(key))
    return method_spec(key)


def num(value) -> Optional[float]:
    """A finite float, or None — NaN and non-numeric both read as 'no value'.

    The allocator emits NaN wherever a figure is genuinely undefined (a holding
    with no covariance estimate, a book with no estimable covariance at all).
    Formatting those straight into an f-string prints "nan%", which reads as a
    broken number rather than an absent one, so every display path funnels
    through here and renders an em dash instead.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def mmom_state(attrs) -> Optional[dict]:
    """Managed Momentum's overlay, read once from a book's attrs; None for any other book.

    The run log, the System tab and the Holdings card all describe the same
    overlay, so they read it through here rather than each re-parsing the
    `nco_mmom_*` attrs with its own defaults. Strength is λ × gate × scale; the
    rest is what each factor read, and which history it read it from.
    """
    at = attrs or {}
    if "nco_mmom_strength" not in at:
        return None
    days = int(at.get("nco_mmom_history_days", 0) or 0)
    # Rows the bear gate read: the history up to the run month's first session (v12.2).
    gate_days = int(at.get("nco_mmom_gate_rows", days) or 0)
    short = bool(at.get("nco_mmom_history_short"))
    start = at.get("nco_mmom_history_start")
    months = int(at.get("nco_mmom_vol_months", 0) or 0)
    ranked = int(at.get("nco_mmom_ranked", 0) or 0)
    strength = num(at.get("nco_mmom_strength")) or 0.0
    n_uni = int(at.get("nco_universe", 0) or 0)
    coverage = num(at.get("nco_mmom_coverage"))
    held = at.get("nco_mmom_floored_held")
    return {
        "strength": strength,
        # Whether the weights actually moved: a strength above 0 tilts nothing
        # when too few names carry a 12-1 return to rank. nco records the same
        # test as `nco_momentum_applied`; read it where the book carries it.
        "tilted": (bool(at["nco_momentum_applied"]) if "nco_momentum_applied" in at
                   else strength > 0 and ranked >= MMOM_MIN_RANKED),
        # Set when the overlay could not read its own crash guard and stood down
        # to the grid (strength 0) — the reason, else None.
        "stood_down": (str(at["nco_mmom_stood_down"]) if at.get("nco_mmom_stood_down")
                       else None),
        # Rows the 24-month gate needs on this history's calendar (505 on a 5-day one,
        # 731 on a 7-day one); a book built before nco recorded it read 5-day rows.
        "gate_needs": int(at.get("nco_mmom_history_needed") or MMOM_MIN_HISTORY),
        "calendar": str((at.get("nco_mmom_windows") or {}).get("calendar", "5-day")),
        "lam": num(at.get("nco_mmom_lambda")) or MMOM_LAMBDA,
        "gate": num(at.get("nco_mmom_gate")),            # 1 open, 0 shut
        "market": num(at.get("nco_mmom_market_24m")),    # None: under a year, gate held open
        "scale": num(at.get("nco_mmom_scale")),
        "vol": num(at.get("nco_mmom_overlay_vol")),
        "vol_median": num(at.get("nco_mmom_overlay_vol_median")),
        "months": months,
        "scale_acts": months >= MMOM_MIN_VOL_MONTHS,
        "ranked": ranked,
        "ranks_enough": ranked >= MMOM_MIN_RANKED,
        # The floor is counted over the universe BEFORE top-N; `floored_held` is
        # how many of those the book holds (None on a book built before nco
        # recorded it). Below the universe size floored names usually fall outside the
        # book, though a floored Dislocated name can outweigh an unfloored Idle one.
        "floored": int(at.get("nco_mmom_floored", 0) or 0),
        "floored_held": int(held) if held is not None else None,
        "floor": num(at.get("nco_mmom_floor")) or MMOM_FLOOR,
        # N in strength × rank / N: the names allocated over, not the positions.
        "universe": n_uni,
        "whole": holds_universe(at),
        # Share of the allocated names with a close in the history the overlay
        # read; the rest carry no rank and are absent from the gate's market.
        "coverage": coverage,
        "no_close": (int(round((1.0 - coverage) * n_uni)) if coverage is not None else 0),
        "days": days,
        "gate_days": gate_days,
        "short": short,
        "source": str(at.get("nco_mmom_source") or "—"),
        "fell_back": at.get("nco_mmom_source") == "estimation panel",
        "start": pd.Timestamp(start) if start is not None else None,
        "window": f"{days} sessions (24 months wanted)" if short else "24 months",
    }


def holds_universe(attrs) -> bool:
    """Whether the book holds every name it allocated over.

    True only when the positions requested reach the universe. Below that,
    top-N selection runs after the weights: the grid's floor (and Managed
    Momentum's) keeps every WEIGHT positive, so the book always fills the count
    asked for, but the lowest-weighted names — floored names first — are cut.
    "Every name stays held" may be said only when this is True.
    """
    at = attrs or {}
    n = int(at.get("nco_universe", 0) or 0)
    return n > 0 and int(at.get("nco_positions_requested", 0) or 0) >= n


def mmom_floor_text(s: dict) -> str:
    """The floor as one line — what the book holds there, and what the universe had.

    "0 held at 25% of their grid weight · 9 of 50 at the floor before top-N — the
    lowest weights, cut first", or, when the book holds the whole universe, "9 of
    50 names at 25% of their grid weight, all held". One phrasing for the run log
    and the System tab.
    """
    fl, n, pct = s["floored"], s["universe"], f"{s['floor']:.0%}"
    if fl == 0:
        return f"none of {n} names at {pct} of their grid weight"
    if s["whole"]:
        return f"{fl} of {n} names at {pct} of their grid weight, all held"
    held = s["floored_held"]
    return ((f"{held} held" if held is not None else "— held")
            + f" at {pct} of their grid weight · {fl} of {n} at the floor before top-N"
            + (f" — {fl - held} cut by top-N" if held is not None and fl > held else ""))


def mmom_history_caveat(s: Optional[dict]) -> Optional[tuple]:
    """(title, body) when the overlay stood down or read less history than it asks for.

    One statement for the notice rail, the run log and the System tab. A
    stood-down overlay is said outright — strength 0, the book is the grid's —
    never as a gate that "read what there was". None when nothing is owed.
    """
    if s is None:
        return None
    days, needs = s["days"], s["gate_needs"]
    if s["stood_down"]:
        return ("Momentum overlay stood down",
                ("The long close history was unavailable" if s["fell_back"]
                 else "The overlay's history was too short")
                + f", so the overlay stood down — {s['stood_down']} ({s['gate_days']} sessions "
                f"to the month's first, {needs} needed). Its strength is 0 and this book's "
                "weights are the grid's.")
    if not (s["fell_back"] or s["short"]):
        return None
    lead = (f"The long close history was unavailable, so the overlay read the {days}-session "
            "estimation panel" if s["fell_back"]
            else f"The close history holds {days} sessions to this date")
    if not s["short"]:
        return ("Momentum overlay read the estimation panel", lead + ".")
    gate, mkt = s["gate"], s["market"]
    tail = (f", under the {needs} sessions its 24-month bear gate needs, so the gate "
            + ("read what there was" if mkt is not None
               else "could not be read (under a year) and stayed open"))
    if gate == 0 and mkt is not None:
        tail += (f" — market {mkt:+.1%} — and shut it: the overlay is off and this book's "
                 "weights are the grid's.")
    elif s["tilted"] and not s["scale_acts"]:
        tail += (f"; the volatility scale could not act ({s['months']} of {MMOM_MIN_VOL_MONTHS} "
                 f"month-start readings), so the overlay ran unscaled at λ {s['lam']:g}.")
    else:
        tail += "."
    return (("Momentum overlay read the estimation panel" if s["fell_back"]
             else "Momentum overlay read a short history"), lead + tail)


def mmom_coverage_caveat(s: Optional[dict]) -> Optional[tuple]:
    """(title, body) when some allocated names have no close in the history the overlay read.

    None on a stood-down overlay: nothing was ranked, so the gap changes nothing.
    """
    if s is None or s["stood_down"] or s["coverage"] is None or s["no_close"] <= 0:
        return None
    k, n = s["no_close"], s["universe"]
    return (f"{k} of {n} names have no close history",
            "They carry no momentum rank and are absent from the gate's market; the grid "
            "sizes them as usual.")


def unfunded_symbols(attrs) -> list:
    """Holdings whose weight buys less than one share at the run's capital: 0 units.

    Any style can produce them — a small weight on an expensive name — and the
    floors make them likelier. They are rows of the book that hold nothing, so
    every surface names them (Broker Sync already skips a 0-unit row).
    """
    at = attrs or {}
    return [str(x) for x in (at.get("nco_unfunded_symbols") or [])]
