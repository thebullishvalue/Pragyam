"""
The family console: terminal output for the Pragyam family.

Shared, byte for byte, by Tattva, Nishkarsh, Nivesa, Pragyam, Arthagati, Samhita, Swing,
Sanket, VAAYDO, Vyāpti and Anumaan, as the fidelity block of ``theme.css`` is for the page:
one grammar for everything an app writes to the terminal that runs it. Edit the family copy
and apply it to every member; a change made in one member alone is a fork.

The grammar
-----------
    banner        once per process: the product, its version, what it is      ══ rules
    header        a run opens: its title, version, run id and start time       ══ rules
    main_header   the run's inputs, one labelled row each                     ── rule
    start_phase   PHASE 2/5 · MŪLA ENGINE, timed; end_phase closes it         ── rule
    section       an unnumbered block inside a phase                         title, hairline
    task          a numbered step: header, its working lines, the outcome and its time
    item          Label               value      (labels in one column, values wrap under it)
    detail        → a working line
    success · warning · error · info     ✓ ⚠ ✗ ℹ, the glyph in its colour
    summary       ┌─ a card of results ─ … └─
    end_run       the closing line: ✓ RUN COMPLETE · time · warnings · errors

Everything is 72 columns wide, so a run reads in an 80-column terminal. Times read in the
unit that suits them (740ms, 3.64s, 12.4s, 2m 05s). Colour is used only when the output is a
terminal (``NO_COLOR`` / ``FORCE_COLOR`` honoured), so a captured log is plain text.

Third-party output (yfinance's HTTP errors, Python warnings from pandas) is routed into the
run as one dimmed ``⚠ source · message`` line each, de-duplicated within a run, instead of
breaking the trace with raw stderr: ``capture_library_output()`` once at start-up.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import secrets
import sys
import textwrap
import threading
import time
import unicodedata
from datetime import datetime
from typing import Any, Callable, Iterator, TextIO

__all__ = [
    "Colors", "ConsoleOutput", "StepHandle", "console", "get_console", "generate_run_id",
    "get_run_id", "SESSION_RUN_IDENTIFIER", "strip_ansi", "fmt_elapsed", "capture_library_output",
    "ConsoleLogHandler", "route_logging", "unroute_logging", "WIDTH",
]

#: Every rule and every wrapped line fits this many columns.
WIDTH = 72
#: The column an item's value starts at, after its label.
_LABEL_COLS = 22

if os.name == "nt":  # ANSI on Windows 10+ consoles; a no-op elsewhere
    try:
        import colorama
        colorama.just_fix_windows_console()
    except Exception:  # noqa: BLE001
        pass

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _cols(text: str) -> int:
    """Terminal columns a string occupies: combining marks take none, wide glyphs two."""
    n = 0
    for ch in strip_ansi(text):
        if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
            continue
        n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return n


def _label(text: Any) -> str:
    """A label as the family writes it: its first letter capitalised, nothing else touched
    (so "call as issued" reads "Call as issued" and "OHLCV" or "R²" stay as written). An
    identifier ("yahoo:history", "t_total") is left exactly as written."""
    s = str(text).strip().rstrip(":").rstrip()
    first = s.split(" ")[0]
    return s if re.search(r"[:_./]", first) else s[:1].upper() + s[1:]


def fmt_elapsed(secs: float) -> str:
    """A duration in the unit that reads at a glance: 740ms, 3.64s, 12.4s, 2m 05s."""
    if secs < 1:
        return f"{secs * 1000:.0f}ms"
    if secs < 10:
        return f"{secs:.2f}s"
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(round(secs)), 60)
    return f"{m}m {s:02d}s"


def generate_run_id() -> str:
    """A new run id ``YYYYmmdd_HHMMSS_xxxxxxxx`` (pure: it does not start a run)."""
    # From ``secrets``, never ``uuid``: an app's own ids come from uuid4 (and its tests stub it
    # to make them deterministic), so the console must not draw from that stream.
    return f"{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"


#: The process's own id, for output written outside any run.
SESSION_RUN_IDENTIFIER = generate_run_id()


class Colors:
    """ANSI codes. Callers may embed them; they are stripped when colour is off."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    # roles
    ACCENT = CYAN          # the product, a run's title
    PHASE = BLUE           # phase and section headings
    LABEL = GRAY           # labels, rules, times, ids
    # glyphs
    SUCCESS = "✓"
    WARNING = "⚠"
    ERROR = "✗"
    INFO = "ℹ"
    ARROW = "→"


Sink = TextIO | Callable[[str], Any]

#: A value printed in these words takes their colour wherever it appears as a whole value.
_STATUS_INK = {
    "SUCCESS": Colors.GREEN, "OK": Colors.GREEN, "PASS": Colors.GREEN, "PASSED": Colors.GREEN,
    "COMPLETE": Colors.GREEN, "DONE": Colors.GREEN, "READY": Colors.GREEN,
    "PARTIAL": Colors.YELLOW, "DEGRADED": Colors.YELLOW, "WARN": Colors.YELLOW, "STALE": Colors.YELLOW,
    "SKIPPED": Colors.YELLOW,
    "FAILED": Colors.RED, "FAIL": Colors.RED, "ERROR": Colors.RED,
}


def _emit(sink: Sink, text: str) -> None:
    if callable(getattr(sink, "write", None)):
        try:
            sink.write(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            enc = getattr(sink, "encoding", None) or "utf-8"
            sink.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
        flush = getattr(sink, "flush", None)
        if callable(flush):
            flush()
    else:
        sink(text)


class StepHandle:
    """The live step yielded by :meth:`ConsoleOutput.task`.

    Carries the outcome, so the closing line says what the step produced, not only that it
    finished. It defaults to "done": a step that forgets to report still closes honestly.
    """

    def __init__(self, out: "ConsoleOutput") -> None:
        self._out = out
        self.status = "ok"
        self.result = ""

    def detail(self, message: str) -> None:
        self._out.detail(message)

    def item(self, label: str, value: Any) -> None:
        self._out.item(label, value)

    def note(self, message: str) -> None:
        """A warning that does not change the step's outcome."""
        self._out.warning(message)

    def ok(self, result: str) -> None:
        self.status, self.result = "ok", result

    def warn(self, result: str) -> None:
        self.status, self.result = "warn", result

    def fail(self, result: str) -> None:
        self.status, self.result = "fail", result


class ConsoleOutput:
    """Styled, sectioned terminal output for one app's runs.

    Args:
        sink: a text stream or a ``str -> None`` callable; ``None`` resolves ``sys.stdout``
            at write time (so test capture works).
        color: force ANSI on or off; ``None`` decides by whether the sink is a terminal.
    """

    def __init__(self, sink: Sink | None = None, color: bool | None = None) -> None:
        self._sink = sink
        self._color = color
        self._extra: list[Sink] = []
        self._lock = threading.RLock()
        #: False silences everything (studies and tests run many targets).
        self.enabled = True
        self.run_id: str | None = None
        self._run_started = 0.0
        self._run_open = False            # a header was printed for this run
        self._warnings = 0
        self._errors = 0
        self._step_num = 0
        self._task_depth = 0
        self._section_depth = 0
        self._phase_timers: dict[str, tuple[float, int, int]] = {}
        self._seen: set[str] = set()      # library lines already shown in this run
        self._banners: set[str] = set()
        self._written = 0                 # lines written, so a block can tell it opens a phase
        self._heading_at = -1
        self._tls = threading.local()     # per-thread hold buffer and quiet flag (see ``hold``)

    # ── sinks and colour ─────────────────────────────────────────────────
    @property
    def sink(self) -> Sink:
        return self._sink if self._sink is not None else sys.stdout

    def use_color(self) -> bool:
        if self._color is not None:
            return self._color
        if os.environ.get("FORCE_COLOR"):
            return True
        if os.environ.get("NO_COLOR"):
            return False
        isatty = getattr(self.sink, "isatty", None)
        try:
            return bool(isatty()) if callable(isatty) else False
        except Exception:  # noqa: BLE001
            return False

    @contextlib.contextmanager
    def tee(self, sink: Sink) -> Iterator[Sink]:
        """Copy everything written inside the block to ``sink``, as plain text."""
        with self._lock:
            self._extra.append(sink)
        try:
            yield sink
        finally:
            with self._lock:
                if sink in self._extra:
                    self._extra.remove(sink)

    # ── replays: hold, release, drop ─────────────────────────────────────
    # Streamlit reruns the whole script on every interaction, and an app whose pipeline is
    # cached replays it in milliseconds. A replay is not a run, and printing its trace again
    # on every click buries the real one. ``hold`` buffers this thread's lines until the app
    # knows: ``release`` prints them (work was done), ``drop`` discards them and keeps the
    # rest of this rerun quiet (a replay). Per thread, so one user's rerun never silences
    # another user's run.
    def hold(self) -> None:
        self._tls.buf = []

    def release(self) -> None:
        buf, self._tls.buf = getattr(self._tls, "buf", None), None
        for message, end in buf or ():
            self._write(message, end)

    def drop(self) -> None:
        self._tls.buf = None
        self._tls.quiet = True

    def resume(self) -> None:
        """Start a rerun clean: nothing held, nothing quiet."""
        self._tls.buf = None
        self._tls.quiet = False

    def _write(self, message: str = "", end: str = "\n") -> None:
        if not self.enabled or getattr(self._tls, "quiet", False):
            return
        buf = getattr(self._tls, "buf", None)
        if buf is not None:
            buf.append((message, end))
            return
        text = f"{message}{end}"
        plain = strip_ansi(text)
        with self._lock:
            self._written += 1
            try:
                _emit(self.sink, text if self.use_color() else plain)
            except Exception:  # noqa: BLE001 — a closed stream never breaks a run
                pass
            for extra in list(self._extra):
                _emit(extra, plain)

    # ── layout helpers ───────────────────────────────────────────────────
    def _indent(self, base: int) -> int:
        """Inside a step every line shares the step's content column; outside, ``base``."""
        return base if not self._task_depth else 6 + 2 * (self._task_depth - 1)

    def _lines(self, text: str, first: str, hang: int, ink: str = "", cont: str | None = None) -> None:
        """Write ``text`` after the prefix ``first``, wrapped to the width under column ``hang``;
        ``ink`` colours the text itself, applied after wrapping so a coloured line wraps too;
        ``cont`` prefixes continuation lines (a card's bar), spaces to ``hang`` by default."""
        text = str(text)
        room = max(24, WIDTH - hang)
        out: list[str] = []
        for para in text.split("\n") or [""]:
            if "\x1b[" in para or _cols(para) <= room:
                out.append(para)
            else:
                out.extend(textwrap.wrap(para, room, break_long_words=True, break_on_hyphens=False)
                           or [""])
        dye = (lambda t: f"{ink}{t}{Colors.RESET}") if ink else (lambda t: t)
        self._write(first + dye(out[0]))
        for rest in out[1:]:
            self._write((" " * hang if cont is None else cont) + dye(rest))

    def _right(self, left: str, right: str, indent: int = 2) -> str:
        gap = WIDTH - indent - _cols(left) - _cols(right)
        return " " * indent + left + " " * max(2, gap) + right

    @staticmethod
    def _ink_value(value: Any) -> str:
        s = str(value)
        ink = _STATUS_INK.get(s.strip().upper())
        return f"{ink}{s}{Colors.RESET}" if ink else s

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _elapsed(self, start: "datetime | float") -> str:
        secs = (datetime.now() - start).total_seconds() if isinstance(start, datetime) \
            else time.perf_counter() - start
        return fmt_elapsed(secs)

    # ── run identity ─────────────────────────────────────────────────────
    def generate_run_id(self) -> str:
        return generate_run_id()

    def get_run_id(self) -> str:
        return self.run_id or SESSION_RUN_IDENTIFIER

    def begin_run(self, run_id: str | None = None) -> str:
        """Start a run: one id, a clock, fresh step numbers and counters. Prints nothing."""
        self.run_id = run_id or generate_run_id()
        self._run_started = time.perf_counter()
        self._run_open = False
        self._warnings = self._errors = 0
        self._step_num = 0
        self._task_depth = 0
        self._section_depth = 0
        self._phase_timers.clear()
        self._seen.clear()
        self._tls.step = None
        return self.run_id

    def end_run(self, status: str | None = None) -> None:
        """Close the run with its status, time and counts (printed when it was opened)."""
        if self.run_id is None:
            return
        if self._run_open:
            # A run fails only when its caller says so: an error line (one symbol that would
            # not download) is counted, not promoted to the run's verdict.
            failed = (status or "").upper() in ("FAILED", "FAIL", "ERROR")
            glyph, ink, word = ((Colors.ERROR, Colors.RED, "RUN FAILED") if failed else
                                (Colors.SUCCESS, Colors.GREEN, "RUN COMPLETE"))
            w, e = self._warnings, self._errors
            warns = f"{Colors.YELLOW if w else Colors.LABEL}{w} warning{'s' * (w != 1)}{Colors.RESET}"
            errs = f"{Colors.RED if e else Colors.LABEL}{e} error{'s' * (e != 1)}{Colors.RESET}"
            sep = f"{Colors.LABEL} · {Colors.RESET}"
            left = (f"{ink}{glyph} {Colors.BOLD}{word}{Colors.RESET}  "
                    f"{Colors.LABEL}{fmt_elapsed(time.perf_counter() - self._run_started)}{Colors.RESET}"
                    f"{sep}{warns}{sep}{errs}")
            self._write()
            self.line("═")
            self._write(self._right(left, f"{Colors.LABEL}{self._timestamp()}{Colors.RESET}"))
            self.line("═")
            self._write()
        self.run_id = None
        self._run_open = False

    @contextlib.contextmanager
    def run(self, title: str, version: str = "", details: dict[str, Any] | None = None) -> Iterator[str]:
        """``with console.run("Analysis", VERSION, {...}):`` — header, inputs, closing line.

        An exception closes the run as failed and propagates; Streamlit's own stop and rerun
        signals close it as complete (they are control flow, not failures).
        """
        run_id = self.begin_run()
        self.header(title, version)
        if details:
            self.main_header("Inputs", details)
        try:
            yield run_id
        except BaseException as exc:
            if type(exc).__name__ in ("StopException", "RerunException"):
                self.end_run()
            else:
                self.error(f"{type(exc).__name__}: {exc}")
                self.end_run("FAILED")
            raise
        self.end_run()

    # ── mastheads ────────────────────────────────────────────────────────
    def line(self, char: str = "─", length: int = WIDTH) -> None:
        """A rule. Long rules all take the family width, so every rule lines up."""
        n = WIDTH if length >= 40 else length
        self._write(f"{Colors.LABEL}{char * n}{Colors.RESET}")

    def banner(self, product: str, version: str = "", tagline: str = "",
               details: dict[str, Any] | None = None) -> None:
        """The product's masthead, once per process (a Streamlit rerun does not repeat it)."""
        if product in self._banners:
            return
        self._banners.add(product)
        self._write()
        self.line("═")
        ver = f"{Colors.LABEL}v{str(version).lstrip('v')}{Colors.RESET}" if version else ""
        self._write(self._right(f"{Colors.BOLD}{Colors.ACCENT}{product.upper()}{Colors.RESET}", ver))
        if tagline:
            self._lines(tagline, "  ", 2)
        self._write(f"  {Colors.LABEL}session {SESSION_RUN_IDENTIFIER} · "
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}{Colors.RESET}")
        if details:
            self.line("─")
            for k, v in details.items():
                self.item(k, v, indent=2)
        self.line("═")

    def header(self, title: str, version: str = "", run_id: str | None = None) -> None:
        """A run's masthead: starts a run unless ``begin_run`` just started one, so a run has one
        id (a run that ended without ``end_run`` — an ``st.stop()`` — does not leak into the next)."""
        if run_id is not None or self.run_id is None or self._run_open:
            self.begin_run(run_id)
        self._run_open = True
        self._write()
        self.line("═")
        ver = f"{Colors.LABEL}v{str(version).lstrip('v')}{Colors.RESET}" if version else ""
        self._write(self._right(f"{Colors.BOLD}{Colors.ACCENT}{title.upper()}{Colors.RESET}", ver))
        self._write(f"  {Colors.LABEL}run {self.run_id} · {datetime.now():%Y-%m-%d %H:%M:%S}"
                    f"{Colors.RESET}")
        self.line("═")

    def main_header(self, title: str, details: dict[str, Any]) -> None:
        """The run's inputs. Opens the run with ``title`` as its masthead if none was printed."""
        if not self._run_open:
            self.header(title)
            title = "Inputs"
        self.reset_steps()
        self._write()
        self._write(f"  {Colors.BOLD}{Colors.PHASE}{title.upper()}{Colors.RESET}")
        self.line("─")
        for key, value in details.items():
            if str(key).strip().lower().rstrip(":") in ("run id", "run"):
                continue          # the masthead carries the run id; never a second one
            self.item(key, value)

    def section(self, title: str, phase: str = "") -> None:
        """A heading. With ``phase`` it is a phase-level heading, else a block inside one."""
        if not (self._written == self._heading_at and not phase):
            self._write()             # no blank line between a phase's rule and its first block
        if phase:
            self._write(f"  {Colors.BOLD}{Colors.PHASE}{phase.upper()} · {title.upper()}{Colors.RESET}")
            self.line("─")
            self._heading_at = self._written
        else:
            pad = " " * self._indent(2)
            self._write(f"{pad}{Colors.BOLD}{title}{Colors.RESET}")
            self._write(f"{pad}{Colors.LABEL}{'┄' * min(_cols(title), WIDTH - len(pad))}{Colors.RESET}")
        self._section_depth += 1

    def start_phase(self, phase: str, num: int = 0, total: int = 0) -> None:
        """Open a numbered, timed phase: PHASE 2/5 · MŪLA ENGINE."""
        self._phase_timers[phase] = (time.perf_counter(), num, total)
        label = f"PHASE {num}/{total} · {phase.upper()}" if num and total else f"PHASE · {phase.upper()}"
        self._write()
        self._write(self._right(f"{Colors.BOLD}{Colors.PHASE}{label}{Colors.RESET}",
                                f"{Colors.LABEL}{self._timestamp()}{Colors.RESET}"))
        self.line("─")
        self._heading_at = self._written

    def end_phase(self, phase: str) -> None:
        """Close a phase with its time."""
        started = self._phase_timers.pop(phase, None)
        if started is None:
            return
        t0, num, total = started
        tag = f"PHASE {num}/{total} · " if num and total else "PHASE · "
        self._write(f"  {Colors.GREEN}{Colors.SUCCESS}{Colors.RESET} {Colors.LABEL}{tag}{phase.upper()}"
                    f"  {fmt_elapsed(time.perf_counter() - t0)}{Colors.RESET}")

    # ── steps ────────────────────────────────────────────────────────────
    def reset_steps(self) -> None:
        self._step_num = 0

    def _step_head(self, title: str, detail: str = "") -> None:
        """``Step N  Title  detail``; a detail that would run past the width wraps under the title."""
        pad = self._indent(2)
        head = f"{' ' * pad}{Colors.BOLD}Step {self._step_num}{Colors.RESET}  {Colors.BOLD}{title}{Colors.RESET}"
        if detail and _cols(head) + 2 + _cols(detail) <= WIDTH:
            self._write(f"{head}  {Colors.LABEL}{detail}{Colors.RESET}")
            return
        self._write(head)
        if detail:
            self._lines(detail, " " * (pad + 4), pad + 4, Colors.LABEL)

    def step(self, num: int, title: str) -> None:
        """A numbered step header without its own timing (``task`` times itself)."""
        self._write(f"{' ' * self._indent(2)}{Colors.BOLD}Step {num}{Colors.RESET}  {Colors.BOLD}{title}"
                    f"{Colors.RESET}")

    @contextlib.contextmanager
    def task(self, title: str, detail: str = "") -> Iterator[StepHandle]:
        """A numbered step: header, its working lines, then its outcome and time.

            with console.task("Historical panel", "100-day lookback") as t:
                t.detail("cache MISS — downloading")
                t.ok(f"{len(rows)} trading days")

        Numbering runs through the whole run. An exception closes the step as failed and
        propagates, so a crashed step is never reported as complete.
        """
        self._step_num += 1
        self._step_head(title, detail)
        handle = StepHandle(self)
        started = time.perf_counter()
        self._task_depth += 1
        try:
            yield handle
        except BaseException as exc:
            if type(exc).__name__ in ("StopException", "RerunException"):
                self._task_depth -= 1
                raise
            self._errors += 1
            self._lines(f"{type(exc).__name__}: {exc}  {Colors.LABEL}{self._elapsed(started)}{Colors.RESET}",
                        f"{' ' * self._indent(4)}{Colors.RED}{Colors.ERROR}{Colors.RESET} ", self._indent(4) + 2)
            self._task_depth -= 1
            raise
        ink, glyph = {"ok": (Colors.GREEN, Colors.SUCCESS), "warn": (Colors.YELLOW, Colors.WARNING),
                      "fail": (Colors.RED, Colors.ERROR)}[handle.status]
        if handle.status == "warn":
            self._warnings += 1
        elif handle.status == "fail":
            self._errors += 1
        self._write(f"{' ' * self._indent(4)}{ink}{glyph}{Colors.RESET} {handle.result or 'done'}"
                    f"  {Colors.LABEL}{self._elapsed(started)}{Colors.RESET}")
        self._task_depth -= 1

    def begin_step(self, title: str, detail: str = "") -> None:
        """A numbered step opened from a callback (an engine's progress hook), where a ``with``
        block cannot reach. Opening the next step closes this one as done, with its time;
        ``end_step`` closes the last."""
        if getattr(self._tls, "step", None) is not None:
            self.end_step()
        self._step_num += 1
        self._step_head(title, detail)
        self._tls.step = time.perf_counter()
        self._task_depth += 1

    def end_step(self, result: str = "", status: str = "ok") -> None:
        started = getattr(self._tls, "step", None)
        if started is None:
            return
        ink, glyph = {"ok": (Colors.GREEN, Colors.SUCCESS), "warn": (Colors.YELLOW, Colors.WARNING),
                      "fail": (Colors.RED, Colors.ERROR)}[status]
        if status == "warn":
            self._warnings += 1
        elif status == "fail":
            self._errors += 1
        self._write(f"{' ' * self._indent(4)}{ink}{glyph}{Colors.RESET} {result or 'done'}"
                    f"  {Colors.LABEL}{self._elapsed(started)}{Colors.RESET}")
        self._task_depth = max(0, self._task_depth - 1)
        self._tls.step = None

    # ── lines ────────────────────────────────────────────────────────────
    def item(self, label: str, value: Any, indent: int = 4) -> None:
        """``Label               value``: labels in one column, long values wrapping under it."""
        pad = self._indent(indent)
        label = _label(label)
        if _cols(label) > _LABEL_COLS - 2:
            # A label too long for the column takes its own line; the value keeps the column.
            self._write(f"{' ' * pad}{Colors.LABEL}{label}{Colors.RESET}")
            self._lines(self._ink_value(value), " " * (pad + _LABEL_COLS), pad + _LABEL_COLS)
            return
        gap = _LABEL_COLS - _cols(label)
        self._lines(self._ink_value(value), f"{' ' * pad}{Colors.LABEL}{label}{Colors.RESET}{' ' * gap}",
                    pad + _LABEL_COLS)

    def detail(self, message: str) -> None:
        pad = self._indent(4)
        self._lines(message, f"{' ' * pad}{Colors.CYAN}{Colors.ARROW}{Colors.RESET} ", pad + 2)

    def text(self, message: str = "", indent: int = 4) -> None:
        """A verbatim line (a traceback, quoted output), dimmed."""
        pad = self._indent(indent)
        for ln in str(message).split("\n"):
            self._write(f"{' ' * pad}{Colors.LABEL}{ln}{Colors.RESET}")

    def _mark(self, ink: str, glyph: str, message: str, base: int = 4, text_ink: str = "") -> None:
        pad = self._indent(base)
        self._lines(message, f"{' ' * pad}{ink}{glyph}{Colors.RESET} ", pad + 2, text_ink)

    def success(self, message: str) -> None:
        self._mark(Colors.GREEN, Colors.SUCCESS, message)

    def warning(self, message: str) -> None:
        self._warnings += 1
        self._mark(Colors.YELLOW, Colors.WARNING, message)

    def error(self, message: str) -> None:
        self._errors += 1
        self._mark(Colors.RED, Colors.ERROR, message, text_ink=Colors.RED)

    def info(self, message: str) -> None:
        self._mark(Colors.BLUE, Colors.INFO, message)

    def failure(self, step: str, error: str) -> None:
        self.error(step)
        self.item("reason", error, indent=self._indent(4) + 2)

    def issue(self, issue_type: str, location: str, description: str) -> None:
        self.warning(f"[{issue_type}] {location}")
        self.text(description, indent=self._indent(4) + 2)

    def checkpoint(self, name: str, status: str = "OK") -> None:
        ok = str(status).upper() == "OK"
        ink, glyph = (Colors.GREEN, Colors.SUCCESS) if ok else (Colors.RED, Colors.ERROR)
        if not ok:
            self._errors += 1
        self._write(f"{' ' * self._indent(4)}{ink}{glyph}{Colors.RESET} {name}  "
                    f"{Colors.LABEL}checkpoint · {status}{Colors.RESET}")

    def library(self, source: str, message: str) -> None:
        """A third-party line (yfinance, a Python warning), dimmed and shown once per run."""
        message = re.sub(r"\s+", " ", str(message)).strip()
        key = f"{source}|{message}"
        if not message or key in self._seen:
            return
        self._seen.add(key)
        self._warnings += 1
        self._mark(Colors.YELLOW, Colors.WARNING, f"{source} · {message}", text_ink=Colors.LABEL)

    def summary(self, title: str, data: dict[str, Any]) -> None:
        """A card of results."""
        pad = " " * self._indent(2)
        inner = WIDTH - len(pad)
        self._write()
        head = f"┌─ {title} "
        self._write(f"{pad}{Colors.LABEL}┌─ {Colors.RESET}{Colors.BOLD}{title}{Colors.RESET} "
                    f"{Colors.LABEL}{'─' * max(2, inner - _cols(head))}{Colors.RESET}")
        bar = f"{pad}{Colors.LABEL}│{Colors.RESET}  "
        for key, value in data.items():
            label = _label(key)
            cont = bar + " " * _LABEL_COLS
            if _cols(label) > _LABEL_COLS - 2:
                self._write(f"{bar}{Colors.LABEL}{label}{Colors.RESET}")
                self._lines(self._ink_value(value), cont, len(pad) + 3 + _LABEL_COLS, cont=cont)
                continue
            gap = _LABEL_COLS - _cols(label)
            self._lines(self._ink_value(value), f"{bar}{Colors.LABEL}{label}{Colors.RESET}{' ' * gap}",
                        len(pad) + 3 + _LABEL_COLS, cont=cont)
        self._write(f"{pad}{Colors.LABEL}└{'─' * (inner - 1)}{Colors.RESET}")


# ── third-party output and logging → the console ───────────────────────────

class ConsoleLogHandler(logging.Handler):
    """Forward ``logging`` records into a console.

    ``library=True`` renders them as dimmed ``⚠ source · message`` lines, once per run
    (third-party chatter); otherwise by level as error, warning or detail.
    """

    def __init__(self, out: ConsoleOutput, level: int = logging.INFO, library: bool = False) -> None:
        super().__init__(level)
        self.out = out
        self.library = library

    @staticmethod
    def _tidy(record: logging.LogRecord) -> tuple[str, str]:
        msg = record.getMessage()
        source = record.name.split(".")[0]
        if record.name == "py.warnings":
            m = re.search(r"(\w+Warning): (.*?)(?:\n|$)", msg)
            pkg = re.search(r"site-packages[/\\]([A-Za-z0-9_]+)", msg)
            source = pkg.group(1) if pkg else "python"
            msg = f"{m.group(1)}: {m.group(2)}" if m else msg.splitlines()[0]
        desc = re.search(r'"description":"([^"]+)"', msg)
        if desc:                                   # an API's JSON error body: keep its sentence
            msg = msg[: msg.find("{")].strip() + " — " + desc.group(1)
        # yfinance names a failed symbol three ways ("$X: …", "N Failed downloads:", "['X']: …");
        # one form, and the bare count header dropped, so a failure reads once.
        msg = re.sub(r"\['([^\]]*)'\]", lambda m: m.group(1).replace("', '", ", "), msg.strip())
        if re.fullmatch(r"\d+ Failed downloads?:", msg):
            msg = ""
        return source, msg.lstrip("$").strip()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            source, msg = self._tidy(record)
            if self.library:
                self.out.library(source, msg)
            elif record.levelno >= logging.ERROR:
                self.out.error(f"{source} · {msg}")
            elif record.levelno >= logging.WARNING:
                self.out.warning(f"{source} · {msg}")
            else:
                self.out.detail(f"{source} · {msg}")
        except Exception:  # noqa: BLE001
            self.handleError(record)


#: Libraries whose own console output is routed into the run instead of raw stderr.
LIBRARY_LOGGERS = ("yfinance", "py.warnings", "urllib3", "peewee", "curl_cffi", "nsepython")


def capture_library_output(out: ConsoleOutput | None = None,
                           names: tuple[str, ...] = LIBRARY_LOGGERS) -> None:
    """Route third-party loggers and Python warnings into the console, once per process."""
    out = out or console
    logging.captureWarnings(True)
    for n in names:
        lg = logging.getLogger(n)
        if any(isinstance(h, ConsoleLogHandler) for h in lg.handlers):
            continue
        lg.addHandler(ConsoleLogHandler(out, logging.WARNING, library=True))
        lg.propagate = False
        if lg.level == logging.NOTSET or lg.level > logging.WARNING:
            lg.setLevel(logging.WARNING)


def route_logging(out: ConsoleOutput | None = None, names: tuple[str, ...] = ("data",),
                  level: int = logging.INFO) -> ConsoleLogHandler:
    """Route an app's own loggers into ``out``; pass the handler to ``unroute_logging`` after.

    Idempotent: a logger already routed into ``out`` keeps its one handler, so calling this
    on every Streamlit rerun never doubles a line.
    """
    out = out or console
    h = ConsoleLogHandler(out, level)
    h._routed_names = tuple(names)  # type: ignore[attr-defined]
    for n in names:
        lg = logging.getLogger(n)
        if any(isinstance(x, ConsoleLogHandler) and x.out is out and not x.library for x in lg.handlers):
            continue
        lg.addHandler(h)
        if lg.level == logging.NOTSET or lg.level > level:
            lg.setLevel(level)
    return h


def unroute_logging(handler: ConsoleLogHandler) -> None:
    for n in getattr(handler, "_routed_names", getattr(handler, "_nishkarsh_names", ())):
        logging.getLogger(n).removeHandler(handler)


# ── the process's console ───────────────────────────────────────────────────

console = ConsoleOutput()


def get_console() -> ConsoleOutput:
    return console


def get_run_id() -> str:
    return console.get_run_id()
