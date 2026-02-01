# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-many-lines,too-many-arguments,too-many-positional-arguments

from __future__ import annotations

import collections as _collections
import ctypes as _ctypes
import os as _os
import platform as _platform
import shutil as _shutil
import sys as _sys
import time as _time
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import final as _final
from typing import overload as _overload


if _TYPE_CHECKING:
    from typing import Protocol, TypeVar, Union
    from typing_extensions import Buffer as ReadOnlyBuffer  # Python 3.12+
    from typing_extensions import TypeAlias  # Python 3.10+

    ChType: TypeAlias = Union[str, bytes, int]

    _T_co = TypeVar('_T_co', covariant=True)
    _T_contra = TypeVar('_T_contra', contravariant=True)

    class SupportsRead(Protocol[_T_co]):  # pylint: disable=too-few-public-methods
        def read(self, length: int = ..., /) -> _T_co:
            raise NotImplementedError

    class SupportsWrite(Protocol[_T_contra]):  # pylint: disable=too-few-public-methods
        def write(self, s: _T_contra, /) -> object:
            raise NotImplementedError


def _char_width(ch: str) -> int:
    """Get display width of a character (0 for combining, 2 for wide, 1 otherwise).

    Uses wcwidth if available, falls back to unicodedata heuristics.
    Consistent with widestring.utf_char_width(): treats non-printable/unassigned (wcwidth=-1) as width 1.
    """
    try:
        import wcwidth  # pylint: disable=import-outside-toplevel

        w = wcwidth.wcwidth(ch)
        if w < 0:
            return 1  # control/unassigned chars treated as width 1 (consistent with widestring)
        if w == 0:
            return 0  # combining characters
        return min(w, 2)  # wide chars (w >= 2 clamped to 2)
    except ImportError:
        import unicodedata  # pylint: disable=import-outside-toplevel

        # Fallback heuristic
        if unicodedata.combining(ch):
            return 0
        if unicodedata.east_asian_width(ch) in ('F', 'W'):
            return 2
        return 1


# =============================================================================
# Return codes
# =============================================================================
ERR: int = -1  # General error return value (matches standard curses)
OK: int = 0  # General success return value

# =============================================================================
# Text attributes (bit-mask values for combining with | operator)
# These control text appearance: style, color pair, and character extraction
# =============================================================================
A_ATTRIBUTES: int = 0xFFFFFF00  # Mask for all attribute bits
A_NORMAL: int = 0  # Normal display (no attributes)
A_STANDOUT: int = 1 << 16  # Terminal's best highlighting mode
A_UNDERLINE: int = 1 << 17  # Underlined text
A_REVERSE: int = 1 << 18  # Reverse video (swap fg/bg colors)
A_BLINK: int = 1 << 19  # Blinking text
A_DIM: int = 1 << 20  # Half-bright/dim text
A_BOLD: int = 1 << 21  # Bold/extra-bright text
A_ALTCHARSET: int = 1 << 22  # Alternate character set (for line drawing)
A_INVIS: int = 1 << 23  # Invisible text
A_PROTECT: int = 1 << 24  # Protected mode
A_CHARTEXT: int = 0xFF  # Mask for character bits
A_COLOR: int = 0xFF00  # Mask for color pair bits
# Extended attributes (less commonly used)
A_HORIZONTAL: int = 1 << 25  # Horizontal highlight
A_LEFT: int = 1 << 26  # Left highlight
A_LOW: int = 1 << 27  # Low highlight
A_RIGHT: int = 1 << 28  # Right highlight
A_TOP: int = 1 << 29  # Top highlight
A_VERTICAL: int = 1 << 30  # Vertical highlight
A_ITALIC: int = 1 << 31  # Italic text (if supported)

# =============================================================================
# Standard colors (indices 0-7 for use with init_pair)
# =============================================================================
COLOR_BLACK: int = 0
COLOR_RED: int = 1
COLOR_GREEN: int = 2
COLOR_YELLOW: int = 3
COLOR_BLUE: int = 4
COLOR_MAGENTA: int = 5
COLOR_CYAN: int = 6
COLOR_WHITE: int = 7

# =============================================================================
# Mouse button event constants (bit-mask values returned by getmouse)
# Each button has PRESSED, RELEASED, CLICKED, DOUBLE_CLICKED, TRIPLE_CLICKED
# BUTTON4 = scroll up, BUTTON5 = scroll down
# =============================================================================
BUTTON1_RELEASED: int = 1 << 0
BUTTON1_PRESSED: int = 1 << 1
BUTTON1_CLICKED: int = 1 << 2
BUTTON1_DOUBLE_CLICKED: int = 1 << 3
BUTTON1_TRIPLE_CLICKED: int = 1 << 4
BUTTON2_RELEASED: int = 1 << 6
BUTTON2_PRESSED: int = 1 << 7
BUTTON2_CLICKED: int = 1 << 8
BUTTON2_DOUBLE_CLICKED: int = 1 << 9
BUTTON2_TRIPLE_CLICKED: int = 1 << 10
BUTTON3_RELEASED: int = 1 << 12
BUTTON3_PRESSED: int = 1 << 13
BUTTON3_CLICKED: int = 1 << 14
BUTTON3_DOUBLE_CLICKED: int = 1 << 15
BUTTON3_TRIPLE_CLICKED: int = 1 << 16
BUTTON4_RELEASED: int = 1 << 18
BUTTON4_PRESSED: int = 1 << 19
BUTTON4_CLICKED: int = 1 << 20
BUTTON4_DOUBLE_CLICKED: int = 1 << 21
BUTTON4_TRIPLE_CLICKED: int = 1 << 22
# BUTTON5 constants for scroll-down
# Note: Scroll wheel events only fire PRESSED events in practice. Due to 32-bit
# mouse mask constraints (modifiers start at bit 24), only bit 23 is available
# for BUTTON5. The other event types are aliased to PRESSED since scroll wheels
# don't generate RELEASED/CLICKED/DOUBLE/TRIPLE events.
BUTTON5_RELEASED: int = 1 << 23  # Aliased to PRESSED for scroll
BUTTON5_PRESSED: int = 1 << 23  # Scroll down event
BUTTON5_CLICKED: int = 1 << 23  # Aliased to PRESSED for scroll
BUTTON5_DOUBLE_CLICKED: int = 1 << 23  # Aliased to PRESSED for scroll
BUTTON5_TRIPLE_CLICKED: int = 1 << 23  # Aliased to PRESSED for scroll
# Mouse button modifier flags (combined with button events via | operator)
BUTTON_CTRL: int = 1 << 24  # Ctrl key held during mouse event
BUTTON_SHIFT: int = 1 << 25  # Shift key held during mouse event
BUTTON_ALT: int = 1 << 26  # Alt key held during mouse event
# Mouse tracking configuration
ALL_MOUSE_EVENTS: int = (1 << 27) - 1  # Report all mouse events (bits 0-26)
REPORT_MOUSE_POSITION: int = 1 << 27  # Report mouse position changes

# =============================================================================
# Special key constants (returned by getch when keypad mode is enabled)
# Values 256+ represent special keys that don't have ASCII equivalents
# =============================================================================
# Navigation keys
KEY_BREAK: int = 257  # Break key
KEY_DOWN: int = 258  # Down arrow
KEY_UP: int = 259  # Up arrow
KEY_LEFT: int = 260  # Left arrow
KEY_RIGHT: int = 261  # Right arrow
KEY_HOME: int = 262  # Home key
KEY_BACKSPACE: int = 263  # Backspace key
# Function keys (F0-F63, though most keyboards only have F1-F12)
KEY_F0: int = 264  # Function key F0 (F1 = KEY_F0 + 1, etc.)
KEY_F1: int = 265
KEY_F2: int = 266
KEY_F3: int = 267
KEY_F4: int = 268
KEY_F5: int = 269
KEY_F6: int = 270
KEY_F7: int = 271
KEY_F8: int = 272
KEY_F9: int = 273
KEY_F10: int = 274
KEY_F11: int = 275
KEY_F12: int = 276
KEY_F13: int = 277
KEY_F14: int = 278
KEY_F15: int = 279
KEY_F16: int = 280
KEY_F17: int = 281
KEY_F18: int = 282
KEY_F19: int = 283
KEY_F20: int = 284
KEY_F21: int = 285
KEY_F22: int = 286
KEY_F23: int = 287
KEY_F24: int = 288
KEY_F25: int = 289
KEY_F26: int = 290
KEY_F27: int = 291
KEY_F28: int = 292
KEY_F29: int = 293
KEY_F30: int = 294
KEY_F31: int = 295
KEY_F32: int = 296
KEY_F33: int = 297
KEY_F34: int = 298
KEY_F35: int = 299
KEY_F36: int = 300
KEY_F37: int = 301
KEY_F38: int = 302
KEY_F39: int = 303
KEY_F40: int = 304
KEY_F41: int = 305
KEY_F42: int = 306
KEY_F43: int = 307
KEY_F44: int = 308
KEY_F45: int = 309
KEY_F46: int = 310
KEY_F47: int = 311
KEY_F48: int = 312
KEY_F49: int = 313
KEY_F50: int = 314
KEY_F51: int = 315
KEY_F52: int = 316
KEY_F53: int = 317
KEY_F54: int = 318
KEY_F55: int = 319
KEY_F56: int = 320
KEY_F57: int = 321
KEY_F58: int = 322
KEY_F59: int = 323
KEY_F60: int = 324
KEY_F61: int = 325
KEY_F62: int = 326
KEY_F63: int = 327
# Editing keys
KEY_DL: int = 328  # Delete line
KEY_IL: int = 329  # Insert line
KEY_DC: int = 330  # Delete character
KEY_IC: int = 331  # Insert character / enter insert mode
KEY_EIC: int = 332  # Exit insert mode
KEY_CLEAR: int = 333  # Clear screen
KEY_EOS: int = 334  # Clear to end of screen
KEY_EOL: int = 335  # Clear to end of line
KEY_SF: int = 336  # Scroll forward (down)
KEY_SR: int = 337  # Scroll backward (up)
KEY_NPAGE: int = 338  # Next page (Page Down)
KEY_PPAGE: int = 339  # Previous page (Page Up)
KEY_STAB: int = 340  # Set tab
KEY_CTAB: int = 341  # Clear tab
KEY_CATAB: int = 342  # Clear all tabs
KEY_ENTER: int = 343  # Enter key
KEY_SRESET: int = 344  # Soft reset
KEY_RESET: int = 345  # Hard reset
KEY_PRINT: int = 346  # Print key
KEY_LL: int = 347  # Lower-left key (home down)
# Keypad keys (3x3 grid: A1-A3, B2, C1-C3)
KEY_A1: int = 348  # Upper left of keypad
KEY_A3: int = 349  # Upper right of keypad
KEY_B2: int = 350  # Center of keypad
KEY_C1: int = 351  # Lower left of keypad
KEY_C3: int = 352  # Lower right of keypad
KEY_BTAB: int = 353  # Back tab (Shift+Tab)
# Action keys
KEY_BEG: int = 354  # Beginning key
KEY_CANCEL: int = 355  # Cancel key
KEY_CLOSE: int = 356  # Close key
KEY_COMMAND: int = 357  # Command key
KEY_COPY: int = 358  # Copy key
KEY_CREATE: int = 359  # Create key
KEY_END: int = 360  # End key
KEY_EXIT: int = 361  # Exit key
KEY_FIND: int = 362  # Find key
KEY_HELP: int = 363  # Help key
KEY_MARK: int = 364  # Mark key
KEY_MESSAGE: int = 365  # Message key
KEY_MOVE: int = 366  # Move key
KEY_NEXT: int = 367  # Next key
KEY_OPEN: int = 368  # Open key
KEY_OPTIONS: int = 369  # Options key
KEY_PREVIOUS: int = 370  # Previous key
KEY_REDO: int = 371  # Redo key
KEY_REFERENCE: int = 372  # Reference key
KEY_REFRESH: int = 373  # Refresh key
KEY_REPLACE: int = 374  # Replace key
KEY_RESTART: int = 375  # Restart key
KEY_RESUME: int = 376  # Resume key
KEY_SAVE: int = 377  # Save key
# Shifted action keys (S prefix = Shift modifier)
KEY_SBEG: int = 378  # Shifted beginning key
KEY_SCANCEL: int = 379  # Shifted cancel
KEY_SCOMMAND: int = 380  # Shifted command
KEY_SCOPY: int = 381  # Shifted copy
KEY_SCREATE: int = 382  # Shifted create
KEY_SDC: int = 383  # Shifted delete character
KEY_SDL: int = 384  # Shifted delete line
KEY_SELECT: int = 385  # Select key
KEY_SEND: int = 386  # Shifted end
KEY_SEOL: int = 387  # Shifted clear to end of line
KEY_SEXIT: int = 388  # Shifted exit
KEY_SFIND: int = 389  # Shifted find
KEY_SHELP: int = 390  # Shifted help
KEY_SHOME: int = 391  # Shifted home
KEY_SIC: int = 392  # Shifted insert character
KEY_SLEFT: int = 393  # Shifted left arrow
KEY_SMESSAGE: int = 394  # Shifted message
KEY_SMOVE: int = 395  # Shifted move
KEY_SNEXT: int = 396  # Shifted next
KEY_SOPTIONS: int = 397  # Shifted options
KEY_SPREVIOUS: int = 398  # Shifted previous
KEY_SPRINT: int = 399  # Shifted print
KEY_SREDO: int = 400  # Shifted redo
KEY_SREPLACE: int = 401  # Shifted replace
KEY_SRIGHT: int = 402  # Shifted right arrow
KEY_SRSUME: int = 403  # Shifted resume
KEY_SSAVE: int = 404  # Shifted save
KEY_SSUSPEND: int = 405  # Shifted suspend
KEY_SUNDO: int = 406  # Shifted undo
KEY_SUSPEND: int = 407  # Suspend key
KEY_UNDO: int = 408  # Undo key
# Special pseudo-keys
KEY_MOUSE: int = 409  # Mouse event occurred (call getmouse for details)
KEY_RESIZE: int = 410  # Terminal was resized
# Key code range bounds
KEY_MIN: int = 257  # Minimum special key value
KEY_MAX: int = 511  # Maximum special key value


# =============================================================================
# ACS (Alternative Character Set) - Line drawing and special characters
# These use Unicode box-drawing characters for consistent cross-platform display
# In real curses, these are initialized after initscr() based on terminal type
# ACS_* are defined as (Unicode codepoint << 32) | A_ALTCHARSET so addch() can detect them
# This avoids clobbering color-pair bits (8-15) when ACS_* is ORed with color attributes.
# =============================================================================
# ACS encoding helper
def _acs(codepoint: int) -> int:
    return (codepoint << 32) | A_ALTCHARSET


# Box drawing corners
ACS_ULCORNER: int = _acs(ord('┌'))  # Upper left corner
ACS_LLCORNER: int = _acs(ord('└'))  # Lower left corner
ACS_URCORNER: int = _acs(ord('┐'))  # Upper right corner
ACS_LRCORNER: int = _acs(ord('┘'))  # Lower right corner
# Box drawing tees (T-junctions)
ACS_LTEE: int = _acs(ord('├'))  # Left tee (pointing right)
ACS_RTEE: int = _acs(ord('┤'))  # Right tee (pointing left)
ACS_BTEE: int = _acs(ord('┴'))  # Bottom tee (pointing up)
ACS_TTEE: int = _acs(ord('┬'))  # Top tee (pointing down)
# Box drawing lines
ACS_HLINE: int = _acs(ord('─'))  # Horizontal line
ACS_VLINE: int = _acs(ord('│'))  # Vertical line
ACS_PLUS: int = _acs(ord('┼'))  # Plus / crossover
# Scan lines (horizontal lines at different vertical positions)
ACS_S1: int = _acs(ord('⎺'))  # Scan line 1 (top)
ACS_S3: int = _acs(ord('─'))  # Scan line 3
ACS_S7: int = _acs(ord('─'))  # Scan line 7
ACS_S9: int = _acs(ord('⎽'))  # Scan line 9 (bottom)
# Special characters
ACS_DIAMOND: int = _acs(ord('◆'))  # Diamond
ACS_CKBOARD: int = _acs(ord('▒'))  # Checkerboard (stipple pattern)
ACS_DEGREE: int = _acs(ord('°'))  # Degree symbol
ACS_PLMINUS: int = _acs(ord('±'))  # Plus/minus
ACS_BULLET: int = _acs(ord('·'))  # Bullet / middle dot
ACS_BLOCK: int = _acs(ord('█'))  # Solid block
ACS_BOARD: int = _acs(ord('#'))  # Board of squares
ACS_LANTERN: int = _acs(ord('#'))  # Lantern symbol
# Arrows
ACS_LARROW: int = _acs(ord('<'))  # Left arrow
ACS_RARROW: int = _acs(ord('>'))  # Right arrow
ACS_DARROW: int = _acs(ord('v'))  # Down arrow
ACS_UARROW: int = _acs(ord('^'))  # Up arrow
# Mathematical symbols
ACS_LEQUAL: int = _acs(ord('≤'))  # Less than or equal
ACS_GEQUAL: int = _acs(ord('≥'))  # Greater than or equal
ACS_NEQUAL: int = _acs(ord('≠'))  # Not equal
ACS_PI: int = _acs(ord('π'))  # Pi
ACS_STERLING: int = _acs(ord('£'))  # UK pound sterling
# Alternative ACS names (using BSSB notation: B=blank, S=solid for each side)
# These map to the same characters as above for compatibility
ACS_BSSB: int = ACS_ULCORNER  # Upper left corner (blank-solid-solid-blank)
ACS_SSBB: int = ACS_LLCORNER  # Lower left corner
ACS_BBSS: int = ACS_URCORNER  # Upper right corner
ACS_SBBS: int = ACS_LRCORNER  # Lower right corner
ACS_SBSS: int = ACS_LTEE  # Left tee
ACS_SSSB: int = ACS_RTEE  # Right tee
ACS_SSBS: int = ACS_BTEE  # Bottom tee
ACS_BSSS: int = ACS_TTEE  # Top tee
ACS_BSBS: int = ACS_HLINE  # Horizontal line
ACS_SBSB: int = ACS_VLINE  # Vertical line
ACS_SSSS: int = ACS_PLUS  # Plus / crossover

# =============================================================================
# Module version info
# =============================================================================
version: bytes = b'nvitop-curses-emulation'  # pylint: disable=invalid-name


class _TerminalState:  # pylint: disable=too-many-instance-attributes
    """Manages global terminal state for the curses emulation layer."""

    def __init__(self) -> None:
        self._stdout = _sys.__stdout__
        self._stdin = _sys.__stdin__

        # Terminal dimensions
        size = _shutil.get_terminal_size(fallback=(80, 24))
        self.lines: int = size.lines
        self.cols: int = size.columns

        # Terminal modes
        self.echo_mode: bool = True
        self.cbreak_mode: bool = False
        self.keypad_mode: bool = False
        self.nodelay_mode: bool = False
        self.leaveok_mode: bool = False
        self.cursor_visible: int = 1

        # Color support
        self.colors: int = 256
        self.color_pairs: int = 256
        self.color_table: dict[int, tuple[int, int, int]] = {
            # Standard 8 colors (0-7) in curses RGB (0-1000)
            0: (0, 0, 0),  # black
            1: (1000, 0, 0),  # red
            2: (0, 1000, 0),  # green
            3: (1000, 1000, 0),  # yellow
            4: (0, 0, 1000),  # blue
            5: (1000, 0, 1000),  # magenta
            6: (0, 1000, 1000),  # cyan
            7: (1000, 1000, 1000),  # white
        }
        self.pair_table: dict[int, tuple[int, int]] = {0: (-1, -1)}  # pair 0 = default colors
        self.default_colors_enabled: bool = False

        # Current attribute state
        self.current_attr: int = 0

        # Input queue for ungetch
        self.input_queue: _collections.deque[int] = _collections.deque()

        # Mouse state
        self.mouse_mask: int = 0
        self.mouse_interval: int = 200
        self.mouse_queue: _collections.deque[tuple[int, int, int, int, int]] = _collections.deque()

        # Screen buffer: list of rows, each row is list of (char, attr) tuples
        self.screen_buffer: list[list[tuple[str, int]]] = []
        self.cursor_y: int = 0
        self.cursor_x: int = 0

        # Windows console state (for restoration)
        self._windows_console_handle: int | None = None
        self._windows_original_mode: int | None = None

        # Unix terminal state (for restoration)
        self._unix_original_termios: list | None = None
        self._unix_stdin_fd: int | None = None

        # Platform-specific setup
        self._platform_init()

    def _platform_init(self) -> None:
        """Platform-specific initialization."""
        if _platform.system() == 'Windows':
            self._init_windows()
        else:
            self._init_unix()

    def _init_windows(self) -> None:
        """Initialize Windows console for ANSI support."""
        try:
            from colorama import just_fix_windows_console  # pylint: disable=import-outside-toplevel

            just_fix_windows_console()
        except ImportError:
            # Fall back to enabling ANSI manually via Windows API
            try:
                kernel32 = _ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
                # Enable ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004)
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                mode = _ctypes.c_ulong()
                if kernel32.GetConsoleMode(handle, _ctypes.byref(mode)):
                    # Save original mode for restoration in endwin()
                    self._windows_console_handle = handle
                    self._windows_original_mode = mode.value
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            except (AttributeError, OSError):
                pass  # not a Windows console, ignore

    def restore_windows_console(self) -> None:
        """Restore original Windows console mode."""
        if self._windows_console_handle is not None and self._windows_original_mode is not None:
            try:
                kernel32 = _ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
                kernel32.SetConsoleMode(self._windows_console_handle, self._windows_original_mode)
            except (AttributeError, OSError):
                pass

    def _init_unix(self) -> None:
        """Initialize Unix terminal and save original settings."""
        # Save original terminal settings for restoration in endwin()
        if _sys.stdin is not None and hasattr(_sys.stdin, 'fileno'):
            try:
                import termios  # pylint: disable=import-outside-toplevel

                fd = _sys.stdin.fileno()
                self._unix_stdin_fd = fd
                self._unix_original_termios = termios.tcgetattr(fd)
            except (ImportError, OSError, ValueError):
                pass  # not a tty or termios not available

    def set_cbreak_unix(self, enable: bool) -> None:
        """Configure Unix terminal for cbreak mode (no line buffering, no echo)."""
        if self._unix_stdin_fd is None:
            return

        try:
            import termios  # pylint: disable=import-outside-toplevel
            import tty  # pylint: disable=import-outside-toplevel

            if enable:
                # Set cbreak mode: no line buffering, pass signals, no echo
                tty.setcbreak(self._unix_stdin_fd)
            elif self._unix_original_termios is not None:
                # Restore original settings
                termios.tcsetattr(
                    self._unix_stdin_fd,
                    termios.TCSADRAIN,
                    self._unix_original_termios,
                )
        except (ImportError, OSError, termios.error):
            pass  # ignore errors (e.g., not a tty)

    def restore_unix_terminal(self) -> None:
        """Restore original Unix terminal settings."""
        if self._unix_stdin_fd is not None and self._unix_original_termios is not None:
            try:
                import termios  # pylint: disable=import-outside-toplevel

                termios.tcsetattr(
                    self._unix_stdin_fd,
                    termios.TCSADRAIN,
                    self._unix_original_termios,
                )
            except (ImportError, OSError):
                pass

    def write(self, data: str) -> None:
        """Write data to stdout."""
        if self._stdout is not None:
            self._stdout.write(data)
            self._stdout.flush()

    def update_size(self) -> None:
        """Update terminal dimensions and resize screen buffer if needed."""
        size = _shutil.get_terminal_size(fallback=(80, 24))
        old_lines, old_cols = self.lines, self.cols
        self.lines = size.lines
        self.cols = size.columns

        # Resize screen buffer if dimensions changed
        if self.screen_buffer and (old_lines != self.lines or old_cols != self.cols):
            self._resize_screen_buffer()

    def _resize_screen_buffer(self) -> None:
        """Resize the screen buffer, preserving content where possible."""
        old_lines = len(self.screen_buffer)
        new_buffer: list[list[tuple[str, int]]] = []
        for y in range(self.lines):
            if y < old_lines:
                # Copy existing row, adjusting column count
                old_row = self.screen_buffer[y]
                if self.cols <= len(old_row):
                    new_row = old_row[: self.cols]
                else:
                    new_row = old_row + [(' ', 0) for _ in range(self.cols - len(old_row))]
            else:
                # New row
                new_row = [(' ', 0) for _ in range(self.cols)]
            new_buffer.append(new_row)
        self.screen_buffer = new_buffer

        # Ensure cursor is within new bounds
        self.cursor_y = min(self.cursor_y, self.lines - 1) if self.lines > 0 else 0
        self.cursor_x = min(self.cursor_x, self.cols - 1) if self.cols > 0 else 0

    def init_screen_buffer(self) -> None:
        """Initialize the screen buffer."""
        self.screen_buffer = [[(' ', 0) for _ in range(self.cols)] for _ in range(self.lines)]

    def get_ansi_color_code(self, color: int, is_fg: bool) -> str:
        """Convert curses color number to ANSI escape code."""
        if color == -1:  # default color
            return '39' if is_fg else '49'

        base = 30 if is_fg else 40
        if 0 <= color < 8:
            return str(base + color)
        if 8 <= color < 16:
            return str(base + 60 + (color - 8))  # bright colors
        # 256-color mode
        return f'{"38" if is_fg else "48"};5;{color}'

    def get_ansi_attr_codes(self, attr: int) -> list[str]:
        """Convert curses attribute to ANSI codes."""
        codes: list[str] = []

        if attr & A_BOLD:
            codes.append('1')
        if attr & A_DIM:
            codes.append('2')
        if attr & A_ITALIC:
            codes.append('3')
        if attr & A_UNDERLINE:
            codes.append('4')
        if attr & A_BLINK:
            codes.append('5')
        # A_STANDOUT and A_REVERSE both render as reverse video (SGR 7)
        if attr & (A_STANDOUT | A_REVERSE):
            codes.append('7')
        if attr & A_INVIS:
            codes.append('8')

        # Extract color pair from attr
        pair_num = (attr & A_COLOR) >> 8
        if pair_num in self.pair_table:
            fg, bg = self.pair_table[pair_num]
            codes.append(self.get_ansi_color_code(fg, is_fg=True))
            codes.append(self.get_ansi_color_code(bg, is_fg=False))

        return codes

    def apply_attr(self, attr: int) -> str:
        """Generate ANSI escape sequence for attribute."""
        codes = self.get_ansi_attr_codes(attr)
        if codes:
            return f'\033[0;{";".join(codes)}m'
        return '\033[0m'

    def move_cursor(self, y: int, x: int) -> str:
        """Generate ANSI escape sequence to move cursor."""
        return f'\033[{y + 1};{x + 1}H'

    def read_key_nonblocking(self) -> int:
        """Read a key without blocking. Returns -1 if no key available."""
        # Check input queue first
        if self.input_queue:
            return self.input_queue.popleft()

        if _platform.system() == 'Windows':
            return self._read_key_windows()
        return self._read_key_unix()

    def _read_key_windows(self) -> int:  # pylint: disable=too-many-return-statements,too-many-branches
        """Read key on Windows using msvcrt."""
        import msvcrt  # pylint: disable=import-outside-toplevel,import-error

        # When mouse tracking is enabled, check stdin for ANSI sequences first
        # (Windows Terminal and modern consoles send mouse events via ANSI)
        if self.mouse_mask != 0 and _sys.stdin is not None:
            try:
                import select  # pylint: disable=import-outside-toplevel

                rlist, _, _ = select.select([_sys.stdin], [], [], 0)
                if rlist:
                    return self._read_key_unix()
            except (ImportError, OSError):
                pass  # select not available or not supported on this stdin

        if not msvcrt.kbhit():  # type: ignore[attr-defined]
            return -1

        ch = msvcrt.getwch()  # type: ignore[attr-defined]
        code = ord(ch)

        # Handle escape sequences (for Windows Terminal ANSI support)
        if code == 27:  # ESC
            if msvcrt.kbhit():  # type: ignore[attr-defined]
                next_ch = msvcrt.getwch()  # type: ignore[attr-defined]
                if next_ch == '[':
                    # CSI sequence - read complete sequence until terminator
                    csi_seq = '['
                    max_seq_len = 32
                    deadline = _time.monotonic() + 0.1
                    found_terminator = False
                    for _ in range(max_seq_len):
                        if _time.monotonic() > deadline:
                            break
                        if not msvcrt.kbhit():  # type: ignore[attr-defined]
                            _time.sleep(0.001)
                            continue
                        csi_ch = msvcrt.getwch()  # type: ignore[attr-defined]
                        if csi_ch == '<':
                            # SGR mouse sequence
                            return self._parse_sgr_mouse_windows()
                        csi_seq += csi_ch
                        # Check for sequence terminator (letter A-Z, a-z, or ~)
                        if csi_ch.isalpha() or csi_ch == '~':
                            found_terminator = True
                            break
                    if found_terminator:
                        return self._map_escape_sequence(csi_seq)
                    # Incomplete sequence (timeout) - queue consumed bytes, return ESC
                    for ch in csi_seq:
                        self.input_queue.append(ord(ch))
                if next_ch == 'O':
                    # SS3 sequence (F1-F4 keys)
                    deadline = _time.monotonic() + 0.1
                    while _time.monotonic() < deadline:
                        if msvcrt.kbhit():  # type: ignore[attr-defined]
                            ss3_ch = msvcrt.getwch()  # type: ignore[attr-defined]
                            return self._map_escape_sequence('O' + ss3_ch)
                        _time.sleep(0.001)
                    # Timeout waiting for SS3 char, queue 'O' and return ESC
                    self.input_queue.append(ord('O'))
                    return 27
                # Alt+key: queue the consumed character, return ESC
                self.input_queue.append(ord(next_ch))
            return 27

        # Handle special keys (arrows, function keys, etc.)
        if code in (0, 0xE0):  # extended key prefix
            if msvcrt.kbhit():  # type: ignore[attr-defined]
                ext = ord(msvcrt.getwch())  # type: ignore[attr-defined]
                return self._map_windows_key(ext)
            return -1

        return code

    def _parse_sgr_mouse_windows(self) -> int:
        """Parse SGR mouse sequence on Windows via msvcrt."""
        import msvcrt  # pylint: disable=import-outside-toplevel,import-error

        # Read until 'M' (press) or 'm' (release)
        params: list[str] = []
        current = ''
        # Use deadline-based timeout (200ms) to handle slow terminals
        deadline = _time.monotonic() + 0.2
        max_chars = 20  # prevent infinite loops from malformed input

        chars_read = 0
        while _time.monotonic() < deadline and chars_read < max_chars:
            if not msvcrt.kbhit():  # type: ignore[attr-defined]
                _time.sleep(0.001)
                continue

            ch = msvcrt.getwch()  # type: ignore[attr-defined]
            chars_read += 1

            if ch == ';':
                params.append(current)
                current = ''
            elif ch in ('M', 'm'):
                params.append(current)
                is_release = ch == 'm'
                break
            elif ch.isdigit():
                current += ch
            else:
                return -1  # invalid sequence
        else:
            return -1  # timeout or max chars exceeded

        if len(params) != 3:
            return -1

        try:
            button_code = int(params[0])
            x = int(params[1]) - 1  # convert to 0-indexed
            y = int(params[2]) - 1
        except ValueError:
            return -1

        # Decode button and modifiers
        bstate = self._decode_mouse_button(button_code, is_release)

        # Queue the mouse event
        self.mouse_queue.append((0, x, y, 0, bstate))

        return KEY_MOUSE

    def _read_key_unix(self) -> int:  # pylint: disable=too-many-branches,too-many-return-statements
        """Read key on Unix using select."""
        import select  # pylint: disable=import-outside-toplevel

        if _sys.stdin is None:
            return -1

        # Guard against non-TTY stdin (IDE, piped input) where select() may fail
        try:
            _sys.stdin.fileno()
        except (OSError, ValueError, AttributeError):
            return -1

        try:
            rlist, _, _ = select.select([_sys.stdin], [], [], 0)
        except (OSError, ValueError):
            return -1
        if not rlist:
            return -1

        ch = _sys.stdin.read(1)
        if not ch:
            return -1

        code = ord(ch)

        # Handle escape sequences
        if code == 27:  # ESC # pylint: disable=too-many-nested-blocks
            # Use deadline-based timeout (100ms total) to handle slow connections (SSH, etc.)
            deadline = _time.monotonic() + 0.1
            remaining = 0.1
            rlist, _, _ = select.select([_sys.stdin], [], [], remaining)
            if rlist:
                seq = _sys.stdin.read(1)
                if seq == '[':
                    # CSI sequence - could be mouse or key
                    # Read complete sequence until terminator (letter or ~)
                    # Limit iterations to prevent DoS from malformed input
                    csi_seq = '['
                    max_seq_len = 32  # CSI sequences are typically <20 chars
                    found_terminator = False
                    for _ in range(max_seq_len):
                        remaining = deadline - _time.monotonic()
                        if remaining <= 0:
                            break
                        rlist, _, _ = select.select([_sys.stdin], [], [], remaining)
                        if not rlist:
                            break
                        next_ch = _sys.stdin.read(1)
                        if not next_ch:
                            break
                        csi_seq += next_ch
                        if next_ch == '<':
                            # SGR mouse sequence: \033[<button;x;y[Mm]
                            return self._parse_sgr_mouse(_sys.stdin)
                        # Check for sequence terminator (letter A-Z, a-z, or ~)
                        if next_ch.isalpha() or next_ch == '~':
                            found_terminator = True
                            break
                    if found_terminator:
                        return self._map_escape_sequence(csi_seq)
                    # Incomplete sequence (timeout) - queue consumed bytes, return ESC
                    for ch in csi_seq:
                        self.input_queue.append(ord(ch))
                if seq == 'O':
                    # SS3 sequence (F1-F4 keys)
                    remaining = deadline - _time.monotonic()
                    if remaining > 0:
                        rlist, _, _ = select.select([_sys.stdin], [], [], remaining)
                        if rlist:
                            next_ch = _sys.stdin.read(1)
                            if next_ch:
                                return self._map_escape_sequence('O' + next_ch)
                    # Timeout or no char, queue 'O' and return ESC
                    self.input_queue.append(ord('O'))
                    return 27
                # Alt+key: queue the consumed character, return ESC
                if seq:
                    self.input_queue.append(ord(seq))
            return 27

        return code

    def _parse_sgr_mouse(self, stdin: SupportsRead[str]) -> int:  # pylint: disable=too-many-return-statements
        """Parse SGR extended mouse sequence and queue the event."""
        import select  # pylint: disable=import-outside-toplevel

        # Read until 'M' (press) or 'm' (release)
        params: list[str] = []
        current = ''
        max_chars = 20  # prevent infinite loops from malformed input

        chars_read = 0
        while chars_read < max_chars:
            rlist, _, _ = select.select([stdin], [], [], 0.1)
            if not rlist:
                return -1  # timeout

            ch = stdin.read(1)
            if not ch:
                return -1

            chars_read += 1

            if ch == ';':
                params.append(current)
                current = ''
            elif ch in ('M', 'm'):
                params.append(current)
                is_release = ch == 'm'
                break
            elif ch.isdigit():
                current += ch
            else:
                return -1  # invalid sequence
        else:
            return -1  # max chars exceeded

        if len(params) != 3:
            return -1

        try:
            button_code = int(params[0])
            x = int(params[1]) - 1  # convert to 0-indexed
            y = int(params[2]) - 1
        except ValueError:
            return -1

        # Decode button and modifiers from SGR button code
        bstate = self._decode_mouse_button(button_code, is_release)

        # Queue the mouse event: (id, x, y, z, bstate)
        self.mouse_queue.append((0, x, y, 0, bstate))

        return KEY_MOUSE

    def _decode_mouse_button(self, button_code: int, is_release: bool) -> int:  # pylint: disable=too-many-branches
        """Decode SGR button code to curses button state."""
        # Validate button_code is within reasonable bounds (0-255 for SGR encoding)
        if not isinstance(button_code, int) or button_code < 0 or button_code > 255:
            return 0  # return no button state for invalid codes

        # SGR button encoding:
        # bits 0-1: button (0=left, 1=middle, 2=right, 3=release/none)
        # bit 2: shift
        # bit 3: meta/alt
        # bit 4: control
        # bit 5: motion (drag)
        # bits 6-7: additional buttons (64=scroll up, 65=scroll down)

        button = button_code & 0x03
        shift = bool(button_code & 0x04)
        alt = bool(button_code & 0x08)
        ctrl = bool(button_code & 0x10)
        motion = bool(button_code & 0x20)
        scroll = button_code & 0x40

        bstate = 0

        if scroll:
            # Scroll wheel events (button_code 64=up, 65=down)
            if button == 0:
                bstate = BUTTON4_PRESSED  # scroll up
            elif button == 1:
                bstate = BUTTON5_PRESSED  # scroll down
        elif motion:
            # Motion/drag events - report as button pressed
            if button == 0:
                bstate = BUTTON1_PRESSED
            elif button == 1:
                bstate = BUTTON2_PRESSED
            elif button == 2:
                bstate = BUTTON3_PRESSED
        elif is_release:
            # Release events
            if button == 0:
                bstate = BUTTON1_RELEASED
            elif button == 1:
                bstate = BUTTON2_RELEASED
            elif button == 2:
                bstate = BUTTON3_RELEASED
            else:
                bstate = BUTTON1_RELEASED  # default for button 3 (no button)
        else:
            # Press events
            if button == 0:
                bstate = BUTTON1_PRESSED
            elif button == 1:
                bstate = BUTTON2_PRESSED
            elif button == 2:
                bstate = BUTTON3_PRESSED

        # Add modifier flags
        if shift:
            bstate |= BUTTON_SHIFT
        if alt:
            bstate |= BUTTON_ALT
        if ctrl:
            bstate |= BUTTON_CTRL

        return bstate

    def _map_windows_key(self, ext: int) -> int:
        """Map Windows extended key codes to curses key codes."""
        key_map = {
            72: KEY_UP,
            80: KEY_DOWN,
            75: KEY_LEFT,
            77: KEY_RIGHT,
            71: KEY_HOME,
            79: KEY_END,
            73: KEY_PPAGE,
            81: KEY_NPAGE,
            82: KEY_IC,
            83: KEY_DC,
            59: KEY_F1,
            60: KEY_F2,
            61: KEY_F3,
            62: KEY_F4,
            63: KEY_F5,
            64: KEY_F6,
            65: KEY_F7,
            66: KEY_F8,
            67: KEY_F9,
            68: KEY_F10,
            133: KEY_F11,
            134: KEY_F12,
        }
        return key_map.get(ext, -1)

    def _map_escape_sequence(self, seq: str) -> int:
        """Map ANSI escape sequences to curses key codes."""
        seq_map = {
            # Arrow keys
            '[A': KEY_UP,
            '[B': KEY_DOWN,
            '[C': KEY_RIGHT,
            '[D': KEY_LEFT,
            # Navigation keys
            '[H': KEY_HOME,
            '[F': KEY_END,
            '[5~': KEY_PPAGE,
            '[6~': KEY_NPAGE,
            '[2~': KEY_IC,
            '[3~': KEY_DC,
            '[1~': KEY_HOME,  # alternative home
            '[4~': KEY_END,  # alternative end
            '[7~': KEY_HOME,  # rxvt home
            '[8~': KEY_END,  # rxvt end
            # Function keys (SS3 format)
            'OP': KEY_F1,
            'OQ': KEY_F2,
            'OR': KEY_F3,
            'OS': KEY_F4,
            # Function keys (CSI format)
            '[15~': KEY_F5,
            '[17~': KEY_F6,
            '[18~': KEY_F7,
            '[19~': KEY_F8,
            '[20~': KEY_F9,
            '[21~': KEY_F10,
            '[23~': KEY_F11,
            '[24~': KEY_F12,
            # Alternative function keys
            '[11~': KEY_F1,
            '[12~': KEY_F2,
            '[13~': KEY_F3,
            '[14~': KEY_F4,
        }
        return seq_map.get(seq, 27)


# Global terminal state instance
_terminal: _TerminalState | None = None  # pylint: disable=invalid-name
_initscr_called: bool = False  # pylint: disable=invalid-name
_setupterm_called: bool = False  # pylint: disable=invalid-name
_current_window: CursesWindow | None = None  # pylint: disable=invalid-name

# Module-level attributes set after initialization
LINES: int = 24
COLS: int = 80
COLORS: int = 256
COLOR_PAIRS: int = 256


def baudrate() -> int:
    raise NotImplementedError


def beep() -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    # Write BEL directly to stdout (addch sanitizes control characters)
    _terminal.write('\x07')


def can_change_color() -> bool:
    raise NotImplementedError


def cbreak(flag: bool = True, /) -> None:
    if not _setupterm_called:
        raise error('must call setupterm() first')
    assert _terminal is not None
    _terminal.cbreak_mode = flag
    # Actually configure the terminal on Unix
    if _platform.system() != 'Windows':
        _terminal.set_cbreak_unix(flag)


def color_content(color_number: int, /) -> tuple[int, int, int]:
    raise NotImplementedError


def color_pair(pair_num: int, /) -> int:
    return (pair_num & 0xFF) << 8


def curs_set(visibility: int, /) -> int:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    old_visibility = _terminal.cursor_visible
    _terminal.cursor_visible = visibility
    if visibility == 0:
        _terminal.write('\033[?25l')  # hide cursor
    else:
        _terminal.write('\033[?25h')  # show cursor
    return old_visibility


def def_prog_mode() -> None:
    raise NotImplementedError


def def_shell_mode() -> None:
    raise NotImplementedError


def delay_output(ms: int, /) -> None:
    raise NotImplementedError


def doupdate() -> None:
    raise NotImplementedError


def echo(flag: bool = True, /) -> None:
    if not _setupterm_called:
        raise error('must call setupterm() first')
    assert _terminal is not None
    _terminal.echo_mode = flag


def endwin() -> None:
    global _initscr_called, _current_window  # pylint: disable=global-statement

    if not _initscr_called:
        return

    assert _terminal is not None

    # Reset terminal state
    _terminal.write('\033[0m')  # reset attributes
    _terminal.write('\033[?25h')  # show cursor
    _terminal.write('\033[?1049l')  # switch back to normal screen buffer
    _terminal.write('\033[?1000l')  # disable mouse tracking

    # Restore platform-specific terminal settings
    _terminal.restore_windows_console()
    _terminal.restore_unix_terminal()

    _initscr_called = False
    _current_window = None


def erasechar() -> bytes:
    raise NotImplementedError


def filter() -> None:  # pylint: disable=redefined-builtin
    raise NotImplementedError


def flash() -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    # Visual bell - briefly invert colors
    _terminal.write('\033[?5h')  # enable reverse video
    _time.sleep(0.1)
    _terminal.write('\033[?5l')  # disable reverse video


def flushinp() -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    _terminal.input_queue.clear()

    # Also flush any pending input from the OS
    if _platform.system() == 'Windows':
        import msvcrt  # pylint: disable=import-outside-toplevel,import-error

        while msvcrt.kbhit():  # type: ignore[attr-defined]
            msvcrt.getwch()  # type: ignore[attr-defined]
    else:
        import select  # pylint: disable=import-outside-toplevel

        if _sys.stdin is not None:
            try:
                _sys.stdin.fileno()  # verify stdin supports fileno
                while select.select([_sys.stdin], [], [], 0)[0]:
                    _sys.stdin.read(1)
            except (OSError, ValueError, AttributeError):
                pass  # stdin not suitable for select (IDE, piped input, etc.)


def getmouse() -> tuple[int, int, int, int, int]:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    if not _terminal.mouse_queue:
        raise error('no mouse event available')
    return _terminal.mouse_queue.popleft()


def getsyx() -> tuple[int, int]:
    raise NotImplementedError


def getwin(file: SupportsRead[bytes], /) -> CursesWindow:
    raise NotImplementedError


def get_escdelay() -> int:
    raise NotImplementedError


def get_tabsize() -> int:
    raise NotImplementedError


def halfdelay(tenths: int, /) -> None:
    raise NotImplementedError


def has_colors() -> bool:
    """Check if the terminal supports colors.

    Returns False for:
    - Non-TTY output (piped/redirected)
    - TERM=dumb

    Note: On Windows, we assume color support is available since Windows 10+
    has native VT/ANSI support, and colorama (if installed) handles older versions.
    """
    # Check if __stdout__ is a TTY (output goes to __stdout__, not stdout which may be redirected)
    if (
        _sys.__stdout__ is None
        or not hasattr(_sys.__stdout__, 'isatty')
        or not _sys.__stdout__.isatty()
    ):
        return False

    # Check for dumb terminal
    term = _os.environ.get('TERM', '')
    return term.lower() != 'dumb'


def has_ic() -> bool:
    raise NotImplementedError


def has_il() -> bool:
    raise NotImplementedError


def has_key(key: int, /) -> bool:
    raise NotImplementedError


def has_extended_color_support() -> bool:
    raise NotImplementedError


def init_color(color_number: int, r: int, g: int, b: int, /) -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    if color_number < 0 or color_number >= _terminal.colors:
        raise error(f'invalid color number: {color_number}')
    _terminal.color_table[color_number] = (r, g, b)


def init_pair(pair_num: int, fg: int, bg: int, /) -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    if pair_num < 1 or pair_num >= _terminal.color_pairs:
        raise error(f'invalid color pair number: {pair_num}')
    _terminal.pair_table[pair_num] = (fg, bg)


def initscr() -> CursesWindow:
    global _current_window, _initscr_called  # pylint: disable=global-statement

    if _initscr_called:
        assert _current_window is not None
        return _current_window

    # Ensure setupterm is called
    if not _setupterm_called:
        setupterm()

    assert _terminal is not None

    try:
        # Switch to alternate screen buffer
        _terminal.write('\033[?1049h')  # enable alternate screen buffer
        _terminal.write('\033[2J')  # clear screen
        _terminal.write('\033[H')  # move cursor to home position

        # Initialize screen buffer
        _terminal.init_screen_buffer()

        _current_window = CursesWindow()
        _initscr_called = True
    except Exception:
        # Restore terminal state on initialization failure
        _terminal.write('\033[?1049l')  # switch back to normal screen buffer
        _terminal.write('\033[0m')  # reset attributes
        raise

    return _current_window


def intrflush(flag: bool, /) -> None:
    raise NotImplementedError


def is_term_resized(nlines: int, ncols: int, /) -> bool:
    raise NotImplementedError


def isendwin() -> bool:
    raise NotImplementedError


def keyname(key: int, /) -> bytes:
    raise NotImplementedError


def killchar() -> bytes:
    raise NotImplementedError


def longname() -> bytes:
    raise NotImplementedError


def meta(yes: bool, /) -> None:
    raise NotImplementedError


def mouseinterval(interval: int, /) -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    _terminal.mouse_interval = interval


def mousemask(newmask: int, /) -> tuple[int, int]:
    """Set mouse event mask and return (avail-mask, old-mask).

    Returns:
        Tuple of (avail-mask, old-mask) where avail-mask is the mouse events that will
        actually be reported (intersection of requested mask and supported events), and
        old-mask is the previous mouse mask before this call.
    """
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    old_mask = _terminal.mouse_mask
    _terminal.mouse_mask = newmask

    # Enable/disable mouse tracking via ANSI escape sequences
    if newmask != 0:
        # Enable mouse tracking (X10 compatibility mode + SGR extended mode)
        _terminal.write('\033[?1000h')  # enable basic mouse tracking
        _terminal.write('\033[?1006h')  # enable SGR extended mouse mode
    else:
        _terminal.write('\033[?1000l')  # disable mouse tracking
        _terminal.write('\033[?1006l')  # disable SGR extended mouse mode

    # Return (avail_mask, old_mask) per curses API contract
    # avail_mask = events we can actually report (all mouse events via SGR protocol)
    avail_mask = newmask & (ALL_MOUSE_EVENTS | REPORT_MOUSE_POSITION)
    return (avail_mask, old_mask)


def napms(ms: int, /) -> int:
    raise NotImplementedError


def newpad(nlines: int, ncols: int, /) -> CursesWindow:
    raise NotImplementedError


def newwin(nlines: int, ncols: int, begin_y: int = 0, begin_x: int = 0, /) -> CursesWindow:
    raise NotImplementedError


def nl(flag: bool = True, /) -> None:
    raise NotImplementedError


def nocbreak() -> None:
    cbreak(False)


def noecho() -> None:
    echo(False)


def nonl() -> None:
    raise NotImplementedError


def noqiflush() -> None:
    raise NotImplementedError


def noraw() -> None:
    raise NotImplementedError


def pair_content(pair_num: int, /) -> tuple[int, int]:
    raise NotImplementedError


def pair_number(attr: int, /) -> int:
    raise NotImplementedError


def putp(string: ReadOnlyBuffer, /) -> None:
    raise NotImplementedError


def qiflush(flag: bool = True, /) -> None:
    raise NotImplementedError


def raw(flag: bool = True, /) -> None:
    raise NotImplementedError


def reset_prog_mode() -> None:
    raise NotImplementedError


def reset_shell_mode() -> None:
    raise NotImplementedError


def resetty() -> None:
    raise NotImplementedError


def resize_term(nlines: int, ncols: int, /) -> None:
    raise NotImplementedError


def resizeterm(nlines: int, ncols: int, /) -> None:
    raise NotImplementedError


def savetty() -> None:
    raise NotImplementedError


def setsyx(y: int, x: int, /) -> None:
    raise NotImplementedError


def set_escdelay(ms: int, /) -> None:
    raise NotImplementedError


def set_tabsize(size: int, /) -> None:
    raise NotImplementedError


def setupterm(term: str | None = None, fd: int = -1) -> None:  # pylint: disable=unused-argument
    global _terminal, _setupterm_called, LINES, COLS  # pylint: disable=global-statement

    if _setupterm_called:
        return

    _terminal = _TerminalState()
    _setupterm_called = True

    # Update module-level constants
    LINES = _terminal.lines
    COLS = _terminal.cols


def start_color() -> None:
    global COLORS, COLOR_PAIRS  # pylint: disable=global-statement

    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None

    # Update module-level color constants
    COLORS = _terminal.colors
    COLOR_PAIRS = _terminal.color_pairs


def assume_default_colors(fg: int, bg: int, /) -> None:
    raise NotImplementedError


def termattrs() -> int:
    raise NotImplementedError


def termname() -> bytes:
    raise NotImplementedError


def tigetflag(capname: str, /) -> int:
    raise NotImplementedError


def tigetnum(capname: str, /) -> int:
    raise NotImplementedError


def tigetstr(capname: str, /) -> bytes | None:
    raise NotImplementedError


def tparm(
    str: ReadOnlyBuffer,  # pylint: disable=redefined-builtin
    i1: int = 0,
    i2: int = 0,
    i3: int = 0,
    i4: int = 0,
    i5: int = 0,
    i6: int = 0,
    i7: int = 0,
    i8: int = 0,
    i9: int = 0,
    /,
) -> bytes:
    raise NotImplementedError


def typeahead(fd: int, /) -> None:
    raise NotImplementedError


def unctrl(ch: ChType, /) -> bytes:
    raise NotImplementedError


def unget_wch(ch: int | str, /) -> None:
    raise NotImplementedError


def ungetch(ch: ChType, /) -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None

    if isinstance(ch, str):
        ch = ord(ch)
    elif isinstance(ch, bytes):
        ch = ch[0]
    _terminal.input_queue.appendleft(ch)


def ungetmouse(
    id: int,  # pylint: disable=redefined-builtin
    x: int,
    y: int,
    z: int,
    bstate: int,
    /,
) -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    _terminal.mouse_queue.appendleft((id, x, y, z, bstate))


def update_lines_cols() -> None:
    global LINES, COLS  # pylint: disable=global-statement

    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None

    _terminal.update_size()
    LINES = _terminal.lines
    COLS = _terminal.cols


def use_default_colors() -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _terminal is not None
    _terminal.default_colors_enabled = True
    # Allow -1 to mean "default color"
    _terminal.pair_table[0] = (-1, -1)


def use_env(flag: bool, /) -> None:
    raise NotImplementedError


class error(Exception):  # noqa: N801,N818 # pylint: disable=invalid-name
    pass


@_final
class CursesWindow:  # pylint: disable=too-many-public-methods
    def __init__(self) -> None:
        self.encoding = 'utf-8'
        if _platform.system() == 'Windows':
            try:
                code_page = _ctypes.windll.kernel32.GetConsoleOutputCP()  # type: ignore[attr-defined,unused-ignore]
            except (AttributeError, OSError):
                self.encoding = 'utf-8'
            else:
                self.encoding = f'cp{code_page}'

        # Window state
        self._keypad_mode: bool = False
        self._nodelay_mode: bool = False
        self._leaveok_mode: bool = False
        self._current_attr: int = 0

    # pylint: disable-next=too-many-branches,too-many-statements
    def _write_char_to_buffer(
        self,
        y: int,
        x: int,
        char: str,
        attr: int,
        func_name: str,
    ) -> tuple[int, int, bool]:
        r"""Write a single character to screen buffer, handling width and special chars.

        Args:
            y, x: Current cursor position
            char: Character to write (single char, already sanitized for control chars except \n/\t)
            attr: Attribute to apply
            func_name: Calling function name for error messages

        Returns:
            (new_y, new_x, wrote_char): Updated position and whether a char was written.
            wrote_char=False for newline/tab (cursor moved but nothing in buffer).

        Raises:
            error: If writing would scroll off screen.
        """
        assert _terminal is not None

        # Handle newline - move to start of next line
        if char == '\n':
            y += 1
            x = 0
            if y >= _terminal.lines:
                raise error(f'{func_name}() would scroll off screen (scrollok not supported)')
            return (y, x, False)

        # Handle tab - advance to next tab stop
        if char == '\t':
            next_tab = ((x // 8) + 1) * 8
            if next_tab >= _terminal.cols:
                x = 0
                y += 1
                if y >= _terminal.lines:
                    raise error(f'{func_name}() would scroll off screen (scrollok not supported)')
            else:
                x = next_tab
            return (y, x, False)

        # Sanitize control characters to prevent ANSI escape injection
        if ord(char) < 32 or char == '\x7f':
            char = '?'

        width = _char_width(char)

        if width == 0:
            # Combining character: overlay on previous cell (don't advance cursor).
            # If at position (0, 0), there's no previous cell - the combining char is dropped.
            target_y, target_x = y, x - 1
            if target_x < 0 < target_y:
                target_y -= 1
                target_x = _terminal.cols - 1
            if target_x >= 0:
                prev_char, prev_attr = _terminal.screen_buffer[target_y][target_x]
                # If previous cell is wide-char placeholder, attach to the wide char
                if prev_char == '\x00' and target_x > 0:
                    target_x -= 1
                    prev_char, prev_attr = _terminal.screen_buffer[target_y][target_x]
                if prev_char != '\x00':
                    _terminal.screen_buffer[target_y][target_x] = (prev_char + char, prev_attr)
            return (y, x, True)

        if width == 2:
            # Wide character: needs 2 cells
            if _terminal.cols < 2:
                # Terminal too narrow - replace with space
                _terminal.screen_buffer[y][x] = (' ', attr)
                x += 1
            elif x + 1 >= _terminal.cols:
                # Wide char at right edge - wrap to next line
                x = 0
                y += 1
                if y >= _terminal.lines:
                    raise error(f'{func_name}() would scroll off screen (scrollok not supported)')
                _terminal.screen_buffer[y][x] = (char, attr)
                _terminal.screen_buffer[y][x + 1] = ('\x00', attr)
                x += 2
            else:
                _terminal.screen_buffer[y][x] = (char, attr)
                _terminal.screen_buffer[y][x + 1] = ('\x00', attr)
                x += 2
        else:
            # Normal width-1 character
            _terminal.screen_buffer[y][x] = (char, attr)
            x += 1

        # Handle line wrapping
        if x >= _terminal.cols:
            x = 0
            y += 1
            if y >= _terminal.lines:
                raise error(f'{func_name}() would scroll off screen (scrollok not supported)')

        return (y, x, True)

    @_overload
    def addch(self, ch: ChType, attr: int = ...) -> None: ...

    @_overload
    def addch(self, y: int, x: int, ch: ChType, attr: int = ...) -> None: ...

    # pylint: disable-next=too-many-branches
    def addch(self, *args: int | ChType, attr: int | None = None) -> None:  # type: ignore[misc]
        assert _terminal is not None

        # Parse arguments: addch(ch, [attr]) or addch(y, x, ch, [attr])
        if len(args) >= 3 and isinstance(args[0], int) and isinstance(args[1], int):
            y, x = args[0], args[1]
            ch = args[2]
            if len(args) >= 4 and attr is None:
                attr = int(args[3])
        else:
            y, x = _terminal.cursor_y, _terminal.cursor_x
            ch = args[0] if args else ' '
            if len(args) >= 2 and attr is None:
                attr = int(args[1])

        # Convert ch to string, handling chtype (character | attributes)
        # Type semantics:
        #   str: Unicode character, used as-is
        #   int: chtype format, with special handling for A_ALTCHARSET (ACS_* values)
        #   bytes: decoded to string
        embedded_attr = 0
        if isinstance(ch, int):
            if ch & A_ALTCHARSET:
                # ACS_* value: Unicode codepoint in bits 32+, attributes in bits 0-31
                embedded_attr = (ch & A_ATTRIBUTES) & ~A_ALTCHARSET
                codepoint = (ch >> 32) & 0x10FFFF
                if codepoint == 0:
                    # Backward-compat for older ACS_* encoding (ord(char) | A_ALTCHARSET)
                    codepoint = ch & 0xFFFF
                char = chr(codepoint) if 0 <= codepoint <= 0x10FFFF else '?'
            else:
                # Standard chtype: character in bits 0-7, attributes in bits 8+
                embedded_attr = ch & ~A_CHARTEXT
                codepoint = ch & A_CHARTEXT
                char = chr(codepoint) if codepoint <= 127 else '?'
        elif isinstance(ch, bytes):
            char = ch.decode(self.encoding, errors='replace')
        else:
            char = str(ch)

        # Take only first character
        char = char[0] if char else ' '

        # Merge attributes: start with window's current, overlay embedded, then explicit
        final_attr = self._current_attr
        if embedded_attr:
            final_attr |= embedded_attr
        if attr is not None:
            final_attr = attr | embedded_attr
        attr = final_attr

        # Validate bounds
        if not (0 <= y < _terminal.lines and 0 <= x < _terminal.cols):
            raise error(
                f'addch() at ({y}, {x}) outside window (0-{_terminal.lines - 1}, 0-{_terminal.cols - 1})',
            )

        # Write character using helper
        new_y, new_x, _ = self._write_char_to_buffer(y, x, char, attr, 'addch')
        _terminal.cursor_y = new_y
        _terminal.cursor_x = new_x

    @_overload
    def addnstr(
        self,
        str: str,  # pylint: disable=redefined-builtin
        n: int,
        attr: int = ...,
    ) -> None: ...

    @_overload
    def addnstr(
        self,
        y: int,
        x: int,
        str: str,  # pylint: disable=redefined-builtin
        n: int,
        attr: int = ...,
    ) -> None: ...

    def addnstr(self, *args: int | str, attr: int | None = None) -> None:  # type: ignore[misc]
        assert _terminal is not None

        # Parse arguments: addnstr(str, n, [attr]) or addnstr(y, x, str, n, [attr])
        if len(args) >= 4 and isinstance(args[0], int) and isinstance(args[1], int):
            y, x = args[0], args[1]
            text = str(args[2])
            n = int(args[3])
            if len(args) >= 5 and attr is None:
                attr = int(args[4])
        else:
            y, x = _terminal.cursor_y, _terminal.cursor_x
            text = str(args[0]) if args else ''
            n = int(args[1]) if len(args) >= 2 else -1
            if len(args) >= 3 and attr is None:
                attr = int(args[2])

        # Set default n if not specified
        if n < 0:
            n = len(text)

        if attr is None:
            attr = self._current_attr

        # Validate initial bounds
        if not (0 <= y < _terminal.lines and 0 <= x < _terminal.cols):
            raise error(
                f'addnstr() at ({y}, {x}) outside window (0-{_terminal.lines - 1}, 0-{_terminal.cols - 1})',
            )

        # Truncate text to n characters
        text = text[:n]

        # Write characters using helper
        for char in text:
            if 0 <= y < _terminal.lines and 0 <= x < _terminal.cols:
                y, x, _ = self._write_char_to_buffer(y, x, char, attr, 'addnstr')

        _terminal.cursor_y = y
        _terminal.cursor_x = min(x, _terminal.cols - 1) if _terminal.cols > 0 else 0

    @_overload
    def addstr(
        self,
        str: str,  # pylint: disable=redefined-builtin
        attr: int = ...,
    ) -> None: ...

    @_overload
    def addstr(
        self,
        y: int,
        x: int,
        str: str,  # pylint: disable=redefined-builtin
        attr: int = ...,
    ) -> None: ...

    def addstr(self, *args: int | str, attr: int | None = None) -> None:  # type: ignore[misc]
        assert _terminal is not None

        # Parse arguments: addstr(str, [attr]) or addstr(y, x, str, [attr])
        if len(args) >= 3 and isinstance(args[0], int) and isinstance(args[1], int):
            y, x = args[0], args[1]
            text = str(args[2])
            if len(args) >= 4 and attr is None:
                attr = int(args[3])
        else:
            y, x = _terminal.cursor_y, _terminal.cursor_x
            text = str(args[0]) if args else ''
            if len(args) >= 2 and attr is None:
                attr = int(args[1])

        if attr is None:
            attr = self._current_attr

        # Validate initial bounds
        if not (0 <= y < _terminal.lines and 0 <= x < _terminal.cols):
            raise error(
                f'addstr() at ({y}, {x}) outside window (0-{_terminal.lines - 1}, 0-{_terminal.cols - 1})',
            )

        # Write characters using helper
        for char in text:
            if 0 <= y < _terminal.lines and 0 <= x < _terminal.cols:
                y, x, _ = self._write_char_to_buffer(y, x, char, attr, 'addstr')

        _terminal.cursor_y = y
        _terminal.cursor_x = min(x, _terminal.cols - 1) if _terminal.cols > 0 else 0

    def attroff(self, attr: int, /) -> None:
        raise NotImplementedError

    def attron(self, attr: int, /) -> None:
        raise NotImplementedError

    def attrset(self, attr: int, /) -> None:
        self._current_attr = attr
        assert _terminal is not None
        _terminal.current_attr = attr

    def bkgd(self, ch: ChType, attr: int = 0, /) -> None:
        raise NotImplementedError

    def bkgdset(self, ch: ChType, attr: int = 0, /) -> None:
        raise NotImplementedError

    def border(
        self,
        ls: ChType = 0,
        rs: ChType = 0,
        ts: ChType = 0,
        bs: ChType = 0,
        tl: ChType = 0,
        tr: ChType = 0,
        bl: ChType = 0,
        br: ChType = 0,
    ) -> None:
        raise NotImplementedError

    @_overload
    def box(self) -> None: ...

    @_overload
    def box(self, vertch: ChType = 0, horch: ChType = 0) -> None: ...

    def box(self, *vhch: ChType) -> None:  # type: ignore[misc]
        raise NotImplementedError

    @_overload
    def chgat(self, attr: int) -> None: ...

    @_overload
    def chgat(self, num: int, attr: int) -> None: ...

    @_overload
    def chgat(self, y: int, x: int, attr: int) -> None: ...

    @_overload
    def chgat(self, y: int, x: int, num: int, attr: int) -> None: ...

    def chgat(self, *args: int) -> None:  # type: ignore[misc]
        assert _terminal is not None

        # Parse arguments based on length:
        # chgat(attr) - change attr for rest of line from cursor
        # chgat(num, attr) - change attr for num chars from cursor
        # chgat(y, x, attr) - change attr for rest of line from (y, x)
        # chgat(y, x, num, attr) - change attr for num chars from (y, x)
        if len(args) == 1:
            y, x = _terminal.cursor_y, _terminal.cursor_x
            num = _terminal.cols - x
            attr = args[0]
        elif len(args) == 2:
            y, x = _terminal.cursor_y, _terminal.cursor_x
            num, attr = args[0], args[1]
        elif len(args) == 3:
            y, x, attr = args[0], args[1], args[2]
            num = _terminal.cols - x
        elif len(args) >= 4:
            y, x, num, attr = args[0], args[1], args[2], args[3]
        else:
            return

        # Change attributes in screen buffer
        for i in range(num):
            if 0 <= y < _terminal.lines and 0 <= x + i < _terminal.cols:
                char, _ = _terminal.screen_buffer[y][x + i]
                _terminal.screen_buffer[y][x + i] = (char, attr)

    def clear(self) -> None:
        raise NotImplementedError

    def clearok(self, yes: int) -> None:
        raise NotImplementedError

    def clrtobot(self) -> None:
        raise NotImplementedError

    def clrtoeol(self) -> None:
        raise NotImplementedError

    def cursyncup(self) -> None:
        raise NotImplementedError

    @_overload
    def delch(self) -> None: ...

    @_overload
    def delch(self, y: int, x: int) -> None: ...

    def delch(self, *yx: int) -> None:  # type: ignore[misc]
        raise NotImplementedError

    def deleteln(self) -> None:
        raise NotImplementedError

    @_overload
    def derwin(self, begin_y: int, begin_x: int) -> CursesWindow: ...

    @_overload
    def derwin(self, nlines: int, ncols: int, begin_y: int, begin_x: int) -> CursesWindow: ...

    def derwin(self, *args: int) -> CursesWindow:  # type: ignore[misc]
        raise NotImplementedError

    def echochar(self, ch: ChType, attr: int = 0, /) -> None:
        raise NotImplementedError

    def enclose(self, y: int, x: int, /) -> bool:
        raise NotImplementedError

    def erase(self) -> None:
        assert _terminal is not None
        # Clear screen buffer (cursor position intentionally NOT reset to match standard curses)
        _terminal.screen_buffer = [
            [(' ', 0) for _ in range(_terminal.cols)] for _ in range(_terminal.lines)
        ]

    def getbegyx(self) -> tuple[int, int]:
        raise NotImplementedError

    def getbkgd(self) -> tuple[int, int]:
        raise NotImplementedError

    @_overload
    def getch(self) -> int: ...

    @_overload
    def getch(self, y: int, x: int) -> int: ...

    # pylint: disable-next=too-many-branches
    def getch(self, *yx: int) -> int:  # type: ignore[misc]
        assert _terminal is not None

        # Move cursor if position specified (with bounds check like move())
        if len(yx) == 2:
            y, x = yx[0], yx[1]
            if not (0 <= y < _terminal.lines and 0 <= x < _terminal.cols):
                raise error(
                    f'getch() at ({y}, {x}) outside window '
                    f'(0-{_terminal.lines - 1}, 0-{_terminal.cols - 1})',
                )
            _terminal.cursor_y = y
            _terminal.cursor_x = x
            _terminal.write(_terminal.move_cursor(y, x))

        # Non-blocking mode
        if self._nodelay_mode:
            return _terminal.read_key_nonblocking()

        # Blocking mode - use platform-appropriate blocking I/O
        use_select = False
        if _platform.system() != 'Windows' and _sys.stdin is not None:
            # Unix: try to use select() for efficient blocking wait
            # But guard against invalid fileno (piped stdin, some IDEs, etc.)
            try:
                import select  # pylint: disable=import-outside-toplevel

                # Test if stdin has a valid fileno for select
                _sys.stdin.fileno()
                use_select = True
            except (ValueError, OSError, AttributeError, ImportError):
                # stdin doesn't support fileno() or it's invalid - fall back to polling
                use_select = False

        if use_select:
            import select  # pylint: disable=import-outside-toplevel

            while True:
                # Check input queue first
                if _terminal.input_queue:
                    return _terminal.input_queue.popleft()
                try:
                    # Block until input is available (with 100ms timeout for responsiveness)
                    rlist, _, _ = select.select([_sys.stdin], [], [], 0.1)
                    if rlist:
                        key = _terminal.read_key_nonblocking()
                        if key != -1:
                            return key
                except (ValueError, OSError):
                    # stdin became invalid during wait - fall back to polling
                    break

        # Windows or fallback: polling with adaptive sleep
        # Note: Windows msvcrt doesn't support select() on console input
        if _platform.system() != 'Windows':
            # Non-Windows without valid select() means stdin is non-interactive
            # (e.g., piped input, IDE console). Check queue first for ungetch()'d keys,
            # then return ERR to avoid infinite loop.
            if _terminal.input_queue:
                return _terminal.input_queue.popleft()
            return ERR

        sleep_time = 0.001  # start with 1ms for responsiveness
        max_sleep = 0.05  # Cap at 50ms to avoid perceived lag
        while True:
            key = _terminal.read_key_nonblocking()
            if key != -1:
                return key
            _time.sleep(sleep_time)
            # Gradually increase sleep time to reduce CPU usage during idle
            if sleep_time < max_sleep:
                sleep_time = min(sleep_time * 1.5, max_sleep)

    @_overload
    def get_wch(self) -> int | str: ...

    @_overload
    def get_wch(self, y: int, x: int) -> int | str: ...

    def get_wch(self, *yx: int) -> int | str:  # type: ignore[misc]
        raise NotImplementedError

    @_overload
    def getkey(self) -> str: ...

    @_overload
    def getkey(self, y: int, x: int) -> str: ...

    def getkey(self, *yx: int) -> str:  # type: ignore[misc]
        raise NotImplementedError

    def getmaxyx(self) -> tuple[int, int]:
        assert _terminal is not None
        return (_terminal.lines, _terminal.cols)

    def getparyx(self) -> tuple[int, int]:
        raise NotImplementedError

    @_overload
    def getstr(self) -> bytes: ...

    @_overload
    def getstr(self, n: int) -> bytes: ...

    @_overload
    def getstr(self, y: int, x: int) -> bytes: ...

    @_overload
    def getstr(self, y: int, x: int, n: int) -> bytes: ...

    def getstr(self, *args: int) -> bytes:  # type: ignore[misc]
        raise NotImplementedError

    def getyx(self) -> tuple[int, int]:
        assert _terminal is not None
        return (_terminal.cursor_y, _terminal.cursor_x)

    @_overload
    def hline(self, ch: ChType, n: int) -> None: ...

    @_overload
    def hline(self, y: int, x: int, ch: ChType, n: int) -> None: ...

    def hline(self, *args: int | ChType) -> None:  # type: ignore[misc]
        raise NotImplementedError

    def idcok(self, flag: bool) -> None:
        raise NotImplementedError

    def idlok(self, yes: bool) -> None:
        raise NotImplementedError

    def immedok(self, flag: bool) -> None:
        raise NotImplementedError

    @_overload
    def inch(self) -> int: ...

    @_overload
    def inch(self, y: int, x: int) -> int: ...

    def inch(self, *yx: int) -> int:  # type: ignore[misc]
        raise NotImplementedError

    @_overload
    def insch(self, ch: ChType, attr: int = 0) -> None: ...

    @_overload
    def insch(self, y: int, x: int, ch: ChType, attr: int = 0) -> None: ...

    def insch(self, *args: int | ChType, attr: int = 0) -> None:  # type: ignore[misc]
        raise NotImplementedError

    def insdelln(self, nlines: int) -> None:
        raise NotImplementedError

    def insertln(self) -> None:
        raise NotImplementedError

    @_overload
    def insnstr(
        self,
        str: str,  # pylint: disable=redefined-builtin
        n: int,
        attr: int = ...,
    ) -> None: ...

    @_overload
    def insnstr(
        self,
        y: int,
        x: int,
        str: str,  # pylint: disable=redefined-builtin
        n: int,
        attr: int = ...,
    ) -> None: ...

    def insnstr(self, *args: int | str, attr: int = 0) -> None:  # type: ignore[misc]
        raise NotImplementedError

    @_overload
    def insstr(
        self,
        str: str,  # pylint: disable=redefined-builtin
        attr: int = ...,
    ) -> None: ...

    @_overload
    def insstr(
        self,
        y: int,
        x: int,
        str: str,  # pylint: disable=redefined-builtin
        attr: int = ...,
    ) -> None: ...

    def insstr(self, *args: int | str, attr: int = 0) -> None:  # type: ignore[misc]
        raise NotImplementedError

    @_overload
    def instr(self, n: int = 2047) -> bytes: ...

    @_overload
    def instr(self, y: int, x: int, n: int = 2047) -> bytes: ...

    def instr(self, *args: int) -> bytes:  # type: ignore[misc]
        raise NotImplementedError

    def is_linetouched(self, line: int, /) -> bool:
        raise NotImplementedError

    def is_wintouched(self) -> bool:
        raise NotImplementedError

    def keypad(self, yes: bool, /) -> None:
        self._keypad_mode = yes
        assert _terminal is not None
        _terminal.keypad_mode = yes

    def leaveok(self, yes: bool) -> None:
        self._leaveok_mode = yes
        assert _terminal is not None
        _terminal.leaveok_mode = yes

    def move(self, new_y: int, new_x: int) -> None:
        assert _terminal is not None
        if not (0 <= new_y < _terminal.lines and 0 <= new_x < _terminal.cols):
            raise error(
                f'move() to ({new_y}, {new_x}) outside window '
                f'(0-{_terminal.lines - 1}, 0-{_terminal.cols - 1})',
            )
        _terminal.cursor_y = new_y
        _terminal.cursor_x = new_x

    def mvderwin(self, y: int, x: int) -> None:
        raise NotImplementedError

    def mvwin(self, new_y: int, new_x: int) -> None:
        raise NotImplementedError

    def nodelay(self, yes: bool) -> None:
        self._nodelay_mode = yes
        assert _terminal is not None
        _terminal.nodelay_mode = yes

    def notimeout(self, yes: bool) -> None:
        raise NotImplementedError

    @_overload
    def noutrefresh(self) -> None: ...

    @_overload
    def noutrefresh(
        self,
        pminrow: int,
        pmincol: int,
        sminrow: int,
        smincol: int,
        smaxrow: int,
        smaxcol: int,
    ) -> None: ...

    def noutrefresh(self, *args: int) -> None:  # type: ignore[misc]
        raise NotImplementedError

    @_overload
    def overlay(self, destwin: CursesWindow) -> None: ...

    @_overload
    def overlay(
        self,
        destwin: CursesWindow,
        sminrow: int,
        smincol: int,
        dminrow: int,
        dmincol: int,
        dmaxrow: int,
        dmaxcol: int,
    ) -> None: ...

    def overlay(self, destwin: CursesWindow, *args: int) -> None:  # type: ignore[misc]
        raise NotImplementedError

    @_overload
    def overwrite(self, destwin: CursesWindow) -> None: ...

    @_overload
    def overwrite(
        self,
        destwin: CursesWindow,
        sminrow: int,
        smincol: int,
        dminrow: int,
        dmincol: int,
        dmaxrow: int,
        dmaxcol: int,
    ) -> None: ...

    def overwrite(self, destwin: CursesWindow, *args: int) -> None:  # type: ignore[misc]
        raise NotImplementedError

    def putwin(self, file: SupportsWrite[bytes], /) -> None:
        raise NotImplementedError

    def redrawln(self, beg: int, num: int, /) -> None:
        raise NotImplementedError

    def redrawwin(self) -> None:
        raise NotImplementedError

    @_overload
    def refresh(self) -> None: ...

    @_overload
    def refresh(
        self,
        pminrow: int,
        pmincol: int,
        sminrow: int,
        smincol: int,
        smaxrow: int,
        smaxcol: int,
    ) -> None:
        raise NotImplementedError

    def refresh(self, *args: int) -> None:  # type: ignore[misc] # pylint: disable=unused-argument
        assert _terminal is not None

        # Validate screen buffer before iteration
        if not _terminal.screen_buffer:
            _terminal.init_screen_buffer()

        # Build output string
        output: list[str] = []
        output.append('\033[H')  # move to home position
        output.append('\033[0m')  # reset attributes

        current_attr = 0
        try:
            # Take a shallow copy of the row list to prevent iteration errors if
            # screen_buffer is reassigned during refresh (e.g., by resize operations).
            # Note: This does NOT provide thread-safety or signal-safety - individual rows
            # can still be modified during iteration. This is acceptable for typical
            # single-threaded curses usage where refresh() is called from the main loop.
            buffer_snapshot = list(_terminal.screen_buffer)
            for row in buffer_snapshot:
                for char, attr in row:
                    # Skip wide-char placeholder cells (second cell of wide characters)
                    if char == '\x00':
                        continue
                    if attr != current_attr:
                        output.append(_terminal.apply_attr(attr))
                        current_attr = attr
                    output.append(char)
                output.append('\033[K')  # clear to end of line (for any trailing content)
                output.append('\n')
        except (TypeError, ValueError, IndexError):
            # Handle corrupted buffer - reinitialize and retry
            _terminal.init_screen_buffer()
            output = ['\033[H', '\033[0m', '\033[2J']  # home, reset, clear screen

        # Remove the last newline
        if output and output[-1] == '\n':
            output.pop()

        output.append('\033[0m')  # reset at end

        # Restore cursor position unless leaveok is enabled
        if not self._leaveok_mode:
            output.append(_terminal.move_cursor(_terminal.cursor_y, _terminal.cursor_x))

        _terminal.write(''.join(output))

    def resize(self, nlines: int, ncols: int) -> None:
        raise NotImplementedError

    def scroll(self, lines: int = 1) -> None:
        raise NotImplementedError

    def scrollok(self, flag: bool) -> None:
        raise NotImplementedError

    def setscrreg(self, top: int, bottom: int, /) -> None:
        raise NotImplementedError

    def standend(self) -> None:
        raise NotImplementedError

    def standout(self) -> None:
        raise NotImplementedError

    @_overload
    def subpad(self, begin_y: int, begin_x: int) -> CursesWindow: ...

    @_overload
    def subpad(self, nlines: int, ncols: int, begin_y: int, begin_x: int) -> CursesWindow: ...

    def subpad(self, *args: int) -> CursesWindow:  # type: ignore[misc]
        raise NotImplementedError

    @_overload
    def subwin(self, begin_y: int, begin_x: int) -> CursesWindow: ...

    @_overload
    def subwin(self, nlines: int, ncols: int, begin_y: int, begin_x: int) -> CursesWindow: ...

    def subwin(self, *args: int) -> CursesWindow:  # type: ignore[misc]
        raise NotImplementedError

    def syncdown(self) -> None:
        raise NotImplementedError

    def syncok(self, flag: bool) -> None:
        raise NotImplementedError

    def syncup(self) -> None:
        raise NotImplementedError

    def timeout(self, delay: int) -> None:
        raise NotImplementedError

    def touchline(self, start: int, count: int, changed: bool = True) -> None:
        raise NotImplementedError

    def touchwin(self) -> None:
        raise NotImplementedError

    def untouchwin(self) -> None:
        raise NotImplementedError

    @_overload
    def vline(self, ch: ChType, n: int) -> None: ...

    @_overload
    def vline(self, y: int, x: int, ch: ChType, n: int) -> None: ...

    def vline(self, *args: int | ChType) -> None:  # type: ignore[misc]
        raise NotImplementedError
