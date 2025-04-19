# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import colorsys
import contextlib
import curses
import locale
import os
import signal
from typing import TYPE_CHECKING, Any, ClassVar, Tuple, Union

from nvitop.tui.library.history import GRAPH_SYMBOLS


if TYPE_CHECKING:
    from collections.abc import Generator
    from typing_extensions import TypeAlias  # Python 3.10+


__all__ = ['CursesShortcuts', 'libcurses', 'setlocale_utf8']


LIGHT_THEME: bool = False
DEFAULT_FOREGROUND: int = curses.COLOR_WHITE
DEFAULT_BACKGROUND: int = curses.COLOR_BLACK
COLOR_PAIRS: dict[tuple[int, int], int] = {}
TRUE_COLORS: dict[str | tuple[int, int, int], int] = {
    'black': 0,
    'red': 1,
    'green': 2,
    'yellow': 3,
    'blue': 4,
    'magenta': 5,
    'cyan': 6,
    'white': 7,
    'bright black': 8,
    'bright red': 9,
    'bright green': 10,
    'bright yellow': 11,
    'bright blue': 12,
    'bright magenta': 13,
    'bright cyan': 14,
    'bright white': 15,
    **{f'preserved {i:02d}': i for i in range(16, 64)},
}


BASE_ATTR: int = 0


def _init_color_theme(light_theme: bool = False) -> None:
    """Set the default fg/bg colors."""
    global LIGHT_THEME, DEFAULT_FOREGROUND, DEFAULT_BACKGROUND  # pylint: disable=global-statement

    LIGHT_THEME = light_theme
    if LIGHT_THEME:
        DEFAULT_FOREGROUND = curses.COLOR_BLACK
        DEFAULT_BACKGROUND = curses.COLOR_WHITE
    else:
        DEFAULT_FOREGROUND = curses.COLOR_WHITE
        DEFAULT_BACKGROUND = curses.COLOR_BLACK


def _colormap(x: float, levels: int = 160) -> tuple[int, int, int]:
    # pylint: disable=invalid-name
    h = 0.5 * (1.0 - x) - 0.15
    h = (round(h * levels) / levels) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.8)
    return (round(1000.0 * r), round(1000.0 * g), round(1000.0 * b))


def _get_true_color(rgb: tuple[int, int, int]) -> int:
    if rgb not in TRUE_COLORS:
        try:
            curses.init_color(len(TRUE_COLORS), *rgb)
        except curses.error:
            return -1
        TRUE_COLORS[rgb] = len(TRUE_COLORS)
    return TRUE_COLORS[rgb]


Color: TypeAlias = Union[str, int, float, Tuple[int, int, int]]


def _get_color(fg: Color, bg: Color) -> int:
    """Return the curses color pair for the given fg/bg combination."""
    global COLOR_PAIRS  # pylint: disable=global-statement,global-variable-not-assigned

    if isinstance(fg, str):
        fg = getattr(curses, f'COLOR_{fg.upper()}', -1)
    elif isinstance(fg, tuple):
        fg = _get_true_color(fg)
    elif isinstance(fg, float):
        fg = _get_true_color(_colormap(fg))
    if isinstance(bg, str):
        bg = getattr(curses, f'COLOR_{bg.upper()}', -1)
    elif isinstance(bg, tuple):
        bg = _get_true_color(bg)
    elif isinstance(bg, float):
        bg = _get_true_color(_colormap(bg))

    key = (fg, bg)
    if key not in COLOR_PAIRS:
        new_id = len(COLOR_PAIRS) + 1
        try:
            curses.init_pair(new_id, fg, bg)
        except curses.error:
            # If curses.use_default_colors() failed during the initialization
            # of curses, then using -1 as fg or bg will fail as well, which
            # we need to handle with fallback-defaults:
            if fg == -1:  # -1 is the "default" color
                fg = DEFAULT_FOREGROUND
            if bg == -1:  # -1 is the "default" color
                bg = DEFAULT_BACKGROUND

            try:
                curses.init_pair(new_id, fg, bg)
            except curses.error:
                # If this fails too, colors are probably not supported
                pass
        COLOR_PAIRS[key] = new_id

    return COLOR_PAIRS[key]


def setlocale_utf8() -> bool:
    for code in ('C.UTF-8', 'en_US.UTF-8', '', 'C'):
        try:
            code = locale.setlocale(locale.LC_ALL, code)
        except locale.Error:  # noqa: PERF203
            continue
        else:
            if 'utf8' in code.lower() or 'utf-8' in code.lower():
                return True

    return False


@contextlib.contextmanager
def libcurses(colorful: bool = False, light_theme: bool = False) -> Generator[curses.window]:
    os.environ.setdefault('ESCDELAY', '25')
    setlocale_utf8()

    win = curses.initscr()
    win.nodelay(True)
    win.leaveok(True)
    win.keypad(True)

    curses.noecho()
    curses.cbreak()
    curses.curs_set(False)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    curses.mouseinterval(0)
    curses.ungetmouse(0, 0, 0, 0, 0)

    _init_color_theme(light_theme)

    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass

    if colorful:
        try:
            CursesShortcuts.TERM_256COLOR = curses.COLORS >= 256
        except AttributeError:
            pass

    # Push a Ctrl+C (ascii value 3) to the curses getch stack
    def interrupt_handler(*_: Any) -> None:  # pylint: disable=unused-argument
        curses.ungetch(3)

    # Simulate a ^C press in curses when an interrupt is caught
    signal.signal(signal.SIGINT, interrupt_handler)

    try:
        yield win
    finally:
        curses.endwin()


class CursesShortcuts:
    """This class defines shortcuts to facilitate operations with curses.

    color(*keys) -- sets the color associated with the keys from
        the current colorscheme.
    color_at(y, x, width, *keys) -- sets the color at the given position
    color_reset() -- resets the color to the default
    addstr(*args) -- failsafe version of self.win.addstr(*args)
    """

    ASCII_TRANSTABLE: ClassVar[dict[int, int]] = str.maketrans(
        '═─╴╒╤╕╪╘╧╛┌┬┐┼└┴┘│╞╡├┤▏▎▍▌▋▊▉█░▲▼␤' + GRAPH_SYMBOLS,
        '=--++++++++++++++||||||||||||||^v?' + '=' * len(GRAPH_SYMBOLS),
    )
    TERM_256COLOR: ClassVar[bool] = False

    def __init__(self) -> None:
        self.win: curses.window | None = None
        self.no_unicode: bool = False

    def addstr(self, *args: str | int | Color, **kwargs: str | int | Color) -> None:
        if self.no_unicode:
            args = [  # type: ignore[assignment]
                arg.translate(self.ASCII_TRANSTABLE) if isinstance(arg, str) else arg
                for arg in args
            ]

        assert self.win is not None
        try:
            self.win.addstr(*args, **kwargs)  # type: ignore[arg-type]
        except curses.error:
            pass

    def addnstr(self, *args: str | int | Color, **kwargs: str | int | Color) -> None:
        if self.no_unicode:
            args = [  # type: ignore[assignment]
                arg.translate(self.ASCII_TRANSTABLE) if isinstance(arg, str) else arg
                for arg in args
            ]

        assert self.win is not None
        try:
            self.win.addnstr(*args, **kwargs)  # type: ignore[arg-type]
        except curses.error:
            pass

    def addch(self, *args: str | int | Color, **kwargs: str | int | Color) -> None:
        if self.no_unicode:
            args = [  # type: ignore[assignment]
                arg.translate(self.ASCII_TRANSTABLE) if isinstance(arg, str) else arg
                for arg in args
            ]

        assert self.win is not None
        try:
            self.win.addch(*args, **kwargs)  # type: ignore[arg-type]
        except curses.error:
            pass

    def color(self, fg: Color = -1, bg: Color = -1, attr: str | int = 0) -> int:
        """Change the colors from now on."""
        return self.set_fg_bg_attr(fg, bg, attr)

    def color_reset(self) -> int:
        """Change the colors to the default colors."""
        return self.color()

    def color_at(
        self,
        y: int,
        x: int,
        width: int,
        *args: str | int | Color,
        **kwargs: str | int | Color,
    ) -> None:
        """Change the colors at the specified position."""
        assert self.win is not None
        try:
            self.win.chgat(y, x, width, self.get_fg_bg_attr(*args, **kwargs))  # type: ignore[arg-type]
        except curses.error:
            pass

    @staticmethod
    def get_fg_bg_attr(fg: Color = -1, bg: Color = -1, attr: str | int = 0) -> int:
        """Return the curses attribute for the given fg/bg/attr combination."""
        if fg == -1 and bg == -1 and attr == 0:
            return BASE_ATTR

        if isinstance(attr, str):
            attr_strings = map(str.strip, attr.split('|'))
            attr = 0
            for s in attr_strings:
                attr |= getattr(curses, f'A_{s.upper()}', 0)

        # Tweak for light themes
        if (
            LIGHT_THEME
            and attr & curses.A_REVERSE != 0
            and bg == -1
            and fg not in {DEFAULT_FOREGROUND, -1}
        ):
            bg = DEFAULT_FOREGROUND

        if fg == -1 and bg == -1:
            return attr | BASE_ATTR
        return curses.color_pair(_get_color(fg, bg)) | attr | BASE_ATTR

    def set_fg_bg_attr(self, fg: Color = -1, bg: Color = -1, attr: str | int = 0) -> int:
        assert self.win is not None
        try:
            attr = self.get_fg_bg_attr(fg, bg, attr)
            self.win.attrset(attr)
        except curses.error:
            return 0
        return attr

    def update_size(self, termsize: tuple[int, int] | None = None) -> tuple[int, int]:
        if termsize is not None:
            return termsize

        self.update_lines_cols()
        assert self.win is not None
        return self.win.getmaxyx()

    @staticmethod
    def update_lines_cols() -> None:
        curses.update_lines_cols()

    @staticmethod
    def beep() -> None:
        curses.beep()

    @staticmethod
    def flash() -> None:
        curses.flash()

    @staticmethod
    def set_base_attr(attr: str | int = 0) -> None:
        global BASE_ATTR  # pylint: disable=global-statement

        if isinstance(attr, str):
            attr_strings = map(str.strip, attr.split('|'))
            attr = 0
            for s in attr_strings:
                attr |= getattr(curses, f'A_{s.upper()}', 0)

        BASE_ATTR = attr
