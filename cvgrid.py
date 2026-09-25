"""
PRAGYAM — the Conviction-Value Grid (CVG)
══════════════════════════════════════════════════════════════════════════════

A portfolio style read from Pragati's two tapes, each on the ladder D · W:

    CONVICTION   who controls, and how firmly — pragati.py
    VALUE        where price stands against what the drivers explain, rich or
                 cheap — samanvaya.py

The pair places every name in one of NINE STATES — a full 3 × 3 — and the
state is its weight. The pane's HISTOGRAM runs the engine: a name changes ROW
(who controls) only when the push is behind the change. Nothing here reads a
signal — no ▲▼, no ◆, no divergence, no crossing event.

The 3 × 3
─────────
Each tape is split at its own shading knee. The conviction tape's hue says who
controls and its intensity how firmly — faint inside the inner zone (±30), "a
clear step brighter past it". The value tape turns rich or cheap at θ (±42.9 on
its ±100 scale), Samanvaya's threshold:

                      value CHEAP        value FAIR        value RICH
    UP     (≥ +30)    TURNED      3      BUILDING    3     PAID          1.5
    FAINT             BASING      1.5    IDLE        1     STALLING      0.75
    DOWN   (≤ −30)    DISLOCATED  1      FADING      0.5   DISTRIBUTION  0.25

  TURNED        cheap, and buyers now in control — a dislocation that turned
  BUILDING      buyers in control while price is still fair — the core
  PAID          buyers in control of a price already rich — held, not added to
  BASING        cheap, control not yet decided — the setup before a turn
  IDLE          fair, control not yet decided
  STALLING      rich, and the control that made it rich has faded
  DISLOCATED    cheap, sellers still in control — the watchlist
  FADING        sellers in control at a fair price
  DISTRIBUTION  sellers in control of a rich price — the floor

Units are the weight at each state's centre, CHOSEN from what it means, not
fitted. Every name is held: the state sets how much, never whether.

The map is GRADED, as the Pine draws it: within its cell a name's weight moves
toward one neighbouring cell per axis by how intensely each reading is shaded —
a conviction tape just past +30 sits partway toward the faint row, a solid one
earns the full cell; a barely-rich value sits partway back toward fair. The
weight is continuous inside a cell and steps where the Pine's shading steps,
at the knees (`graded_units`).

The histogram runs the rows
───────────────────────────
Value moves a name between COLUMNS freely — price is where it is. Control is
different: the tape says where control is going; the pane's histogram says
whether the push is behind it. So a name moves UP a row (DOWN → FAINT → UP) only
while the histogram confirms a push UP, and DOWN a row only while it confirms a
push DOWN; otherwise it keeps its row, whatever the tape reads.

A push CONFIRMS when all three of the histogram's channels agree it is real:

    hue          the column is on the side of the move
    lightness    it is IMPULSE, BUILDING or DECELERATING — not TURNING, where
                 the trace itself is already giving ground
    saturation   the regime is not QUIET — an amplified reading is not a reading

A name whose histogram is not yet calibrated follows its tape directly: a
guessed confirmation is not a neutral one. `cvg held` marks a name whose row is
being held against its tape.

Formerly the "Dhṛti Two-Tape" style; renamed with its indicator.

Author: @thebullishvalue
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import pragati
import samanvaya
from pragati import INNER_ZONE, OUTER_ZONE, compute_conviction

STATE_COLUMNS = ("cvg state", "cvg state days", "cvg held")
COLUMNS = pragati.COLUMNS + samanvaya.VALUE_COLUMNS + STATE_COLUMNS

# ── The nine states (and UNREAD), in allocation order, with their units ─────
# Units are CHOSEN from what each state means, not fitted — the Pine's own
# warning applies: "Do not tune this to a backtest." Order is by units, and
# within equal units by the grid (top row first, cheap before rich); it is the
# order the book fills in when N is below the universe.
STATES = (
    ("TURNED",       3.00, "Turned",        "cheap, and buyers now in control"),
    ("BUILDING",     3.00, "Building",      "buyers in control at a fair price"),
    ("PAID",         1.50, "Paid",          "buyers in control, price already rich"),
    ("BASING",       1.50, "Basing",        "cheap, control not yet decided"),
    ("IDLE",         1.00, "Idle",          "fair price, control not yet decided"),
    ("DISLOCATED",   1.00, "Dislocated",    "cheap, sellers still in control"),
    ("UNREAD",       1.00, "Unread",        "a tape not yet calibrated"),
    ("STALLING",     0.75, "Stalling",      "rich, and control has faded"),
    ("FADING",       0.50, "Fading",        "sellers in control at a fair price"),
    ("DISTRIBUTION", 0.25, "Distribution",  "sellers in control of a rich price"),
)
# (row, column) → state. Rows: +1 UP, 0 FAINT, −1 DOWN. Columns: 0 cheap,
# 1 fair, 2 rich.
GRID = {
    (1, 0): "TURNED",     (1, 1): "BUILDING", (1, 2): "PAID",
    (0, 0): "BASING",     (0, 1): "IDLE",     (0, 2): "STALLING",
    (-1, 0): "DISLOCATED", (-1, 1): "FADING", (-1, 2): "DISTRIBUTION",
}
STATE_UNITS = {code: u for code, u, _, _ in STATES}
STATE_LABEL = {code: lab for code, _, lab, _ in STATES}
STATE_MEANING = {code: m for code, _, _, m in STATES}
STATE_ORDER = {code: i for i, (code, _, _, _) in enumerate(STATES)}
CORE_STATES = ("TURNED", "BUILDING")
FLOOR_STATES = ("FADING", "DISTRIBUTION")
CELL = {code: rc for rc, code in GRID.items()}          # state → (row, column)

# ── The graded map ───────────────────────────────────────────────────────────
# The Pine shades both tapes on the same two ramps (section 9, "the ladder
# tapes"): faint inside the knee, transparency 88 → 55; a visible step at it;
# then 35 → 0, solid at the far level. Read back as ink = 1 − T/100:
#     inside the knee   0.12 → 0.45        past it   0.65 → 1.00
VALUE_SOLID = 70.0        # the value tape is solid "toward ±70"
INK_FLOOR = 0.12          # the faintest a ready tape is ever drawn


def tape_ink(x, knee: float, solid: float):
    """A tape reading's ink, exactly as the Pine grades its colour."""
    a = np.abs(np.asarray(x, dtype=float))
    faint = INK_FLOOR + 0.33 * np.clip(a / knee, 0.0, 1.0)
    bright = 0.65 + 0.35 * np.clip((a - knee) / (solid - knee), 0.0, 1.0)
    return np.where(a >= knee, bright, faint)


def graded_units(state: str, conv: float, value: float, push: float) -> float:
    """A name's weight in units on the GRADED map: its cell, shaded.

    Each cell's units are its centre. The name's position inside the cell —
    read on the Pine's own shading ramps — blends it toward ONE neighbouring
    cell per axis, and never past it:

      ROWS   (conviction tape, ink c)
        UP / DOWN row, tape agreeing   c of the cell, 1 − c of the FAINT row:
                                       at the knee (0.65) a third of the way
                                       to Idle's row; solid (1.0) the full cell
        FAINT row                      leans toward the row the tape's sign
                                       points to by c − 0.12 (0 at C = 0, a
                                       third at the knee) — so the Pine's step
                                       at ±30 is a step in weight too
        a row HELD by the histogram    the push holding it earns the cell: its
                                       drawn intensity h (direction × ink ×
                                       saturation, toward the held side) keeps
                                       ½ + ½h of the cell, the rest leans one
                                       row toward the tape — a held row gives
                                       up at most half, never the whole move
      COLUMNS (value tape, ink v)      the same two rules on the value ramp,
                                       FAIR leaning toward the side V is on

    The two axes combine as a bilinear blend of the 3 × 3 units. UNREAD, or a
    missing tape, is its flat unit.
    """
    if state not in CELL or not (np.isfinite(conv) and np.isfinite(value)):
        return float(STATE_UNITS.get(state, STATE_UNITS["UNREAD"]))
    row, col = CELL[state]
    target = 1 if conv >= INNER_ZONE else (-1 if conv <= -INNER_ZONE else 0)
    if target != row:                                   # held by the histogram
        side = 1.0 if row > target else -1.0
        h = float(np.clip(push * side, 0.0, 1.0)) if np.isfinite(push) else 0.0
        rows = ((row, 0.5 + 0.5 * h), (row + (1 if target > row else -1), 0.5 - 0.5 * h))
    elif row != 0:
        c = float(tape_ink(conv, INNER_ZONE, OUTER_ZONE))
        rows = ((row, c), (0, 1.0 - c))
    else:
        lean = float(tape_ink(conv, INNER_ZONE, OUTER_ZONE)) - INK_FLOOR
        rows = ((0, 1.0 - lean), (1 if conv > 0 else -1, lean))
    th = samanvaya.THETA_OSC
    if col != 1:
        v = float(tape_ink(value, th, VALUE_SOLID))
        cols = ((col, v), (1, 1.0 - v))
    else:
        lean = float(tape_ink(value, th, VALUE_SOLID)) - INK_FLOOR
        cols = ((1, 1.0 - lean), (2 if value > 0 else 0, lean))
    return float(sum(p * q * STATE_UNITS[GRID[(r, k)]] for r, p in rows for k, q in cols))


def classify_states(conv: pd.Series, value: pd.Series,
                    gate: Optional[pd.Series] = None) -> pd.DataFrame:
    """The 3 × 3 engine: place each bar in its state; count its days there.

    COLUMN from the value tape, freely. ROW from the conviction tape — UP past
    +30, DOWN past −30, FAINT between — but a name only moves to the tape's row
    when the histogram's gate confirms a push in that direction (+1 to move up,
    −1 to move down). Otherwise it keeps its row, and `cvg held` is 1. With
    no gate reading (histogram not calibrated) the row follows the tape. A bar
    missing either tape is UNREAD, and the row memory starts again after it.
    """
    c = pd.to_numeric(conv, errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(value, errors="coerce").to_numpy(dtype=float)
    g = (pd.to_numeric(gate, errors="coerce").reindex(conv.index).to_numpy(dtype=float)
         if gate is not None else np.full(len(c), np.nan))
    th = samanvaya.THETA_OSC
    n = len(c)
    state = np.empty(n, dtype=object)
    days = np.zeros(n, dtype=int)
    held = np.zeros(n, dtype=int)
    row: Optional[int] = None
    prev = None
    for t in range(n):
        ct, vt, gt = c[t], v[t], g[t]
        if not (np.isfinite(ct) and np.isfinite(vt)):
            s = "UNREAD"
            row = None
        else:
            target = 1 if ct >= INNER_ZONE else (-1 if ct <= -INNER_ZONE else 0)
            if row is None or not np.isfinite(gt):
                row = target
            elif target > row and gt > 0:
                row = target
            elif target < row and gt < 0:
                row = target
            held[t] = int(row != target)
            col = 0 if vt <= -th else (2 if vt >= th else 1)
            s = GRID[(row, col)]
        state[t] = s
        days[t] = days[t - 1] + 1 if t > 0 and prev == s else 1
        prev = s
    return pd.DataFrame({"cvg state": state, "cvg state days": days,
                         "cvg held": held}, index=conv.index)


def compute_readings(df: pd.DataFrame, driver_closes: Optional[pd.DataFrame] = None,
                   symbol: str = "") -> pd.DataFrame:
    """Both of Pragati's tapes and the grid state, for one name — every
    column in COLUMNS.

    `driver_closes` carries the macro drivers (backdata fetches them once per
    panel). Without them the value engine still runs: every factor is empty,
    the hedge never earns skill, and the RV leg is the name's own path — the
    Pine's "Macro hedge: Off", which is where Auto lands anyway when the
    drivers explain nothing.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    df = df.sort_index()
    conv = compute_conviction(df).reindex(df.index)
    val = samanvaya.compute_value(df, driver_closes, symbol).reindex(df.index)
    st = classify_states(conv["conv tape"], val["value tape"], conv["conv push gate"])
    return pd.concat([conv, val, st], axis=1)[list(COLUMNS)]

__all__ = [
    "COLUMNS",
    "STATE_COLUMNS",
    "STATES",
    "STATE_UNITS",
    "STATE_LABEL",
    "STATE_MEANING",
    "STATE_ORDER",
    "CORE_STATES",
    "FLOOR_STATES",
    "GRID",
    "CELL",
    "tape_ink",
    "graded_units",
    "classify_states",
    "compute_readings",
]
