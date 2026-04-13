"""
PRAGYAM - Direct Console Output System
══════════════════════════════════════════════════════════════════════════════

Bypasses Python logging entirely - writes directly to stdout.
This is the ONLY way to get clean output in Streamlit.

Author: @thebullishvalue
Version: 4.2.0
"""

import sys
import uuid
import os
from datetime import datetime
from typing import Optional, Dict, Any

# ══════════════════════════════════════════════════════════════════════════════
# ENABLE ANSI ON WINDOWS - Using colorama for reliability
# ══════════════════════════════════════════════════════════════════════════════

try:
    import colorama
    colorama.init()
except ImportError:
    # Fallback for Windows without colorama
    if os.name == 'nt':
        from ctypes import windll, byref, c_ulong
        STD_OUTPUT_HANDLE = -11
        hConsole = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = c_ulong()
        windll.kernel32.GetConsoleMode(hConsole, byref(mode))
        mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        windll.kernel32.SetConsoleMode(hConsole, mode)

# ── UTF-8 stdout on Windows ────────────────────────────────────────────────────
# Windows default console encoding (cp1252) cannot represent box-drawing
# characters (═, ─, →, ┌, │, └) used in the logger.  Reconfigure stdout
# to UTF-8 so these render correctly.  Falls back silently if reconfigure
# is unavailable (Python < 3.7) or stdout has been replaced by Streamlit.
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# RUN IDENTIFIER - Session-level fallback
# ══════════════════════════════════════════════════════════════════════════════

_SESSION_RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
_SESSION_UUID = str(uuid.uuid4())[:8]
SESSION_RUN_IDENTIFIER = f"{_SESSION_RUN_ID}_{_SESSION_UUID}"

def generate_run_id() -> str:
    """Generate a unique Run ID for each analysis run."""
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_uuid = str(uuid.uuid4())[:8]
    return f"{run_id}_{run_uuid}"


# ══════════════════════════════════════════════════════════════════════════════
# ANSI COLOR CODES - Windows Compatible
# ══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes that work on Windows 10+."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'
    
    # Symbols
    SUCCESS = '✓'
    WARNING = '⚠'
    ERROR = '✗'
    INFO = 'ℹ'


# ══════════════════════════════════════════════════════════════════════════════
# DIRECT CONSOLE OUTPUT - Bypasses Python logging
# ══════════════════════════════════════════════════════════════════════════════

class ConsoleOutput:
    """Direct console output - no logging module."""
    
    def __init__(self):
        self._section_depth = 0
    
    def _write(self, message: str = '', end: str = '\n'):
        """Write directly to stdout, safe on narrow-encoding Windows consoles."""
        text = f"{message}{end}"
        try:
            sys.stdout.write(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Encode to UTF-8 bytes then decode with 'replace' so box-drawing
            # characters become '?' rather than raising on cp1252 consoles.
            safe = text.encode('utf-8', errors='replace').decode(
                sys.stdout.encoding or 'utf-8', errors='replace'
            )
            sys.stdout.write(safe)
        sys.stdout.flush()
    
    def line(self, char: str = '─', length: int = 60):
        """Print a separator line."""
        self._write(f"{Colors.GRAY}{char * length}{Colors.RESET}")

    def header(self, title: str, version: str = ""):
        """Print run header."""
        self._write()
        self.line('═', 70)
        self._write(f"  {Colors.BOLD}{Colors.CYAN}{title} {version}{Colors.RESET}")
        self._write(f"  {Colors.GRAY}Run ID: {SESSION_RUN_IDENTIFIER}{Colors.RESET}")
        self._write(f"  {Colors.GRAY}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        self.line('═', 70)
        self._write()

    def main_header(self, title: str, details: Dict[str, Any]):
        """Print main run header with title and key details."""
        self._write()
        self.line('═', 70)
        self._write(f"  {Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET}")
        self.line('─', 70)
        for key, value in details.items():
            self._write(f"  {Colors.GRAY}{key}:{Colors.RESET} {value}")
        self.line('═', 70)
        self._write()

    def section(self, title: str, phase: str = ""):
        """Print section header."""
        self._write()
        if phase:
            self.line('═', 60)
            self._write(f"  {Colors.BOLD}{Colors.BLUE}{phase}: {title}{Colors.RESET}")
            self.line('═', 60)
        else:
            self._write(f"{Colors.BOLD}{title}{Colors.RESET}")
            self._write(Colors.GRAY + '─' * len(title) + Colors.RESET)
        self._section_depth += 1
    
    def step(self, num: int, title: str):
        """Print numbered step."""
        self._write(f"  {Colors.BOLD}Step {num}:{Colors.RESET} {title}")
    
    def item(self, label: str, value: Any, indent: int = 4):
        """Print labeled item."""
        self._write(f"{' ' * indent}{Colors.GRAY}{label}:{Colors.RESET} {value}")
    
    def detail(self, message: str):
        """Print detailed information."""
        self._write(f"    {Colors.CYAN}→{Colors.RESET} {message}")
    
    def success(self, message: str):
        """Print success message."""
        self._write(f"  {Colors.GREEN}{Colors.SUCCESS} SUCCESS:{Colors.RESET} {message}")
    
    def warning(self, message: str):
        """Print warning message."""
        self._write(f"  {Colors.YELLOW}{Colors.WARNING} WARNING:{Colors.RESET} {message}")
    
    def error(self, message: str):
        """Print error message."""
        self._write(f"  {Colors.RED}{Colors.ERROR} ERROR:{Colors.RESET} {message}")

    def summary(self, title: str, data: Dict[str, Any]):
        """Print summary box."""
        self._write()
        self._write(f"  {Colors.GRAY}┌─ {title}{Colors.RESET}")
        for key, value in data.items():
            self._write(f"  {Colors.GRAY}│   {key}:{Colors.RESET} {value}")
        self._write(f"  {Colors.GRAY}└─{Colors.RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

# Single global instance
console = ConsoleOutput()


def get_console() -> ConsoleOutput:
    """Get the global console instance."""
    return console
