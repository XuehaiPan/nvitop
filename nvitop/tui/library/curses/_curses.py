# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import contextlib
import platform
from typing import TYPE_CHECKING, overload

from nvitop.tui.library.curses import ascii


try:
    from typing import final  # Python 3.8+
except ImportError:

    def final(x):
        return x


if TYPE_CHECKING:
    from collections.abc import Buffer as ReadOnlyBuffer
    from typing import TypeVar
    from typing_extensions import Protocol, TypeAlias

    ChType: TypeAlias = str | bytes | int

    _T_co = TypeVar('_T_co', covariant=True)
    _T_contra = TypeVar('_T_contra', contravariant=True)

    class SupportsRead(Protocol[_T_co]):
        def read(self, length: int = ..., /) -> _T_co: ...

    class SupportsWrite(Protocol[_T_contra]):
        def write(self, s: _T_contra, /) -> object: ...


ERR: int = 1
OK: int = 0
A_ATTRIBUTES: int = 4294967040
A_NORMAL: int = 0
A_STANDOUT: int = 65536
A_UNDERLINE: int = 131072
A_REVERSE: int = 262144
A_BLINK: int = 524288
A_DIM: int = 1048576
A_BOLD: int = 2097152
A_ALTCHARSET: int = 4194304
A_INVIS: int = 8388608
A_PROTECT: int = 16777216
A_CHARTEXT: int = 255
A_COLOR: int = 65280
A_HORIZONTAL: int = 33554432
A_LEFT: int = 67108864
A_LOW: int = 134217728
A_RIGHT: int = 268435456
A_TOP: int = 536870912
A_VERTICAL: int = 1073741824
A_ITALIC: int = 2147483648
COLOR_BLACK: int = 0
COLOR_RED: int = 1
COLOR_GREEN: int = 2
COLOR_YELLOW: int = 3
COLOR_BLUE: int = 4
COLOR_MAGENTA: int = 5
COLOR_CYAN: int = 6
COLOR_WHITE: int = 7
BUTTON1_PRESSED: int = 2
BUTTON1_RELEASED: int = 1
BUTTON1_CLICKED: int = 4
BUTTON1_DOUBLE_CLICKED: int = 8
BUTTON1_TRIPLE_CLICKED: int = 16
BUTTON2_PRESSED: int = 128
BUTTON2_RELEASED: int = 64
BUTTON2_CLICKED: int = 256
BUTTON2_DOUBLE_CLICKED: int = 512
BUTTON2_TRIPLE_CLICKED: int = 1024
BUTTON3_PRESSED: int = 8192
BUTTON3_RELEASED: int = 4096
BUTTON3_CLICKED: int = 16384
BUTTON3_DOUBLE_CLICKED: int = 32768
BUTTON3_TRIPLE_CLICKED: int = 65536
BUTTON4_PRESSED: int = 524288
BUTTON4_RELEASED: int = 262144
BUTTON4_CLICKED: int = 1048576
BUTTON4_DOUBLE_CLICKED: int = 2097152
BUTTON4_TRIPLE_CLICKED: int = 4194304
BUTTON_SHIFT: int = 33554432
BUTTON_CTRL: int = 16777216
BUTTON_ALT: int = 67108864
ALL_MOUSE_EVENTS: int = 134217727
REPORT_MOUSE_POSITION: int = 134217728
KEY_BREAK: int = 257
KEY_DOWN: int = 258
KEY_UP: int = 259
KEY_LEFT: int = 260
KEY_RIGHT: int = 261
KEY_HOME: int = 262
KEY_BACKSPACE: int = 263
KEY_F0: int = 264
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
KEY_DL: int = 328
KEY_IL: int = 329
KEY_DC: int = 330
KEY_IC: int = 331
KEY_EIC: int = 332
KEY_CLEAR: int = 333
KEY_EOS: int = 334
KEY_EOL: int = 335
KEY_SF: int = 336
KEY_SR: int = 337
KEY_NPAGE: int = 338
KEY_PPAGE: int = 339
KEY_STAB: int = 340
KEY_CTAB: int = 341
KEY_CATAB: int = 342
KEY_ENTER: int = 343
KEY_SRESET: int = 344
KEY_RESET: int = 345
KEY_PRINT: int = 346
KEY_LL: int = 347
KEY_A1: int = 348
KEY_A3: int = 349
KEY_B2: int = 350
KEY_C1: int = 351
KEY_C3: int = 352
KEY_BTAB: int = 353
KEY_BEG: int = 354
KEY_CANCEL: int = 355
KEY_CLOSE: int = 356
KEY_COMMAND: int = 357
KEY_COPY: int = 358
KEY_CREATE: int = 359
KEY_END: int = 360
KEY_EXIT: int = 361
KEY_FIND: int = 362
KEY_HELP: int = 363
KEY_MARK: int = 364
KEY_MESSAGE: int = 365
KEY_MOVE: int = 366
KEY_NEXT: int = 367
KEY_OPEN: int = 368
KEY_OPTIONS: int = 369
KEY_PREVIOUS: int = 370
KEY_REDO: int = 371
KEY_REFERENCE: int = 372
KEY_REFRESH: int = 373
KEY_REPLACE: int = 374
KEY_RESTART: int = 375
KEY_RESUME: int = 376
KEY_SAVE: int = 377
KEY_SBEG: int = 378
KEY_SCANCEL: int = 379
KEY_SCOMMAND: int = 380
KEY_SCOPY: int = 381
KEY_SCREATE: int = 382
KEY_SDC: int = 383
KEY_SDL: int = 384
KEY_SELECT: int = 385
KEY_SEND: int = 386
KEY_SEOL: int = 387
KEY_SEXIT: int = 388
KEY_SFIND: int = 389
KEY_SHELP: int = 390
KEY_SHOME: int = 391
KEY_SIC: int = 392
KEY_SLEFT: int = 393
KEY_SMESSAGE: int = 394
KEY_SMOVE: int = 395
KEY_SNEXT: int = 396
KEY_SOPTIONS: int = 397
KEY_SPREVIOUS: int = 398
KEY_SPRINT: int = 399
KEY_SREDO: int = 400
KEY_SREPLACE: int = 401
KEY_SRIGHT: int = 402
KEY_SRSUME: int = 403
KEY_SSAVE: int = 404
KEY_SSUSPEND: int = 405
KEY_SUNDO: int = 406
KEY_SUSPEND: int = 407
KEY_UNDO: int = 408
KEY_MOUSE: int = 409
KEY_RESIZE: int = 410
KEY_MIN: int = 257
KEY_MAX: int = 511


_initscr_called: bool = False
_setupterm_called: bool = False
_current_window: CursesWindow | None = None


def baudrate() -> int: ...


def beep() -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    _current_window.addch(ascii.BEL)


def can_change_color() -> bool: ...


def cbreak(flag: bool = True, /) -> None:
    raise NotImplementedError


def color_content(color_number: int, /) -> tuple[int, int, int]: ...


def color_pair(pair_number: int, /) -> int:
    raise NotImplementedError


def curs_set(visibility: int, /) -> int:
    raise NotImplementedError


def def_prog_mode() -> None: ...
def def_shell_mode() -> None: ...
def delay_output(ms: int, /) -> None: ...
def doupdate() -> None: ...


def echo(flag: bool = True, /) -> None:
    raise NotImplementedError


def endwin() -> None:
    raise NotImplementedError


def erasechar() -> bytes: ...
def filter() -> None: ...


def flash() -> None:
    raise NotImplementedError


def flushinp() -> None:
    raise NotImplementedError


def getmouse() -> tuple[int, int, int, int, int]:
    raise NotImplementedError


def getsyx() -> tuple[int, int]: ...
def getwin(file: SupportsRead[bytes], /) -> CursesWindow: ...
def halfdelay(tenths: int, /) -> None: ...
def has_colors() -> bool: ...


def has_ic() -> bool: ...
def has_il() -> bool: ...
def has_key(key: int, /) -> bool: ...


def init_color(color_number: int, r: int, g: int, b: int, /) -> None:
    raise NotImplementedError


def init_pair(pair_number: int, fg: int, bg: int, /) -> None:
    raise NotImplementedError


def initscr() -> CursesWindow:
    global _current_window, _initscr_called, _setupterm_called

    if _initscr_called:
        assert _current_window is not None
        return _current_window

    _initscr_called = _setupterm_called = True

    _current_window = CursesWindow()
    return _current_window


def intrflush(flag: bool, /) -> None: ...
def is_term_resized(nlines: int, ncols: int, /) -> bool: ...
def isendwin() -> bool: ...
def keyname(key: int, /) -> bytes: ...
def killchar() -> bytes: ...
def longname() -> bytes: ...
def meta(yes: bool, /) -> None: ...


def mouseinterval(interval: int, /) -> None:
    raise NotImplementedError


def mousemask(newmask: int, /) -> tuple[int, int]:
    raise NotImplementedError


def napms(ms: int, /) -> int: ...
def newpad(nlines: int, ncols: int, /) -> CursesWindow: ...
def newwin(nlines: int, ncols: int, begin_y: int = ..., begin_x: int = ..., /) -> CursesWindow: ...
def nl(flag: bool = True, /) -> None: ...


def nocbreak() -> None:
    raise NotImplementedError


def noecho() -> None:
    raise NotImplementedError


def nonl() -> None: ...
def noqiflush() -> None: ...
def noraw() -> None: ...
def pair_content(pair_number: int, /) -> tuple[int, int]: ...
def pair_number(attr: int, /) -> int: ...
def putp(string: ReadOnlyBuffer, /) -> None: ...
def qiflush(flag: bool = True, /) -> None: ...
def raw(flag: bool = True, /) -> None: ...
def reset_prog_mode() -> None: ...
def reset_shell_mode() -> None: ...
def resetty() -> None: ...
def resize_term(nlines: int, ncols: int, /) -> None: ...
def resizeterm(nlines: int, ncols: int, /) -> None: ...
def savetty() -> None: ...
def setsyx(y: int, x: int, /) -> None: ...


def setupterm(term: str | None = None, fd: int = -1) -> None:
    raise NotImplementedError


def start_color() -> None:
    raise NotImplementedError


def termattrs() -> int: ...
def termname() -> bytes: ...
def tigetflag(capname: str, /) -> int: ...
def tigetnum(capname: str, /) -> int: ...
def tigetstr(capname: str, /) -> bytes | None: ...
def tparm(
    str: ReadOnlyBuffer,
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
) -> bytes: ...
def typeahead(fd: int, /) -> None: ...
def unctrl(ch: ChType, /) -> bytes: ...
def unget_wch(ch: int | str, /) -> None: ...


def ungetch(ch: ChType, /) -> None:
    raise NotImplementedError


def ungetmouse(id: int, x: int, y: int, z: int, bstate: int, /) -> None:
    raise NotImplementedError


def update_lines_cols() -> None:
    raise NotImplementedError


def use_default_colors() -> None:
    raise NotImplementedError


def use_env(flag: bool, /) -> None: ...


class error(Exception):  # noqa: N801,N818
    pass


@final
class CursesWindow:
    def __init__(self) -> None:
        self.encoding = 'utf-8'
        if platform.system() == 'Windows':
            import ctypes

            with contextlib.suppress(AttributeError, OSError):
                self.encoding = f'cp{ctypes.windll.kernel32.GetConsoleOutputCP()}'

    @overload
    def addch(self, ch: ChType, attr: int = ...) -> None: ...
    @overload
    def addch(self, y: int, x: int, ch: ChType, attr: int = ...) -> None: ...

    def addch(self, *args, **kwargs):
        raise NotImplementedError

    @overload
    def addnstr(self, str: str, n: int, attr: int = ...) -> None: ...
    @overload
    def addnstr(self, y: int, x: int, str: str, n: int, attr: int = ...) -> None: ...

    def addnstr(self, *args, **kwargs):
        raise NotImplementedError

    @overload
    def addstr(self, str: str, attr: int = ...) -> None: ...
    @overload
    def addstr(self, y: int, x: int, str: str, attr: int = ...) -> None: ...

    def addstr(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def attroff(self, attr: int, /) -> None: ...
    def attron(self, attr: int, /) -> None: ...

    def attrset(self, attr: int, /) -> None:
        raise NotImplementedError

    def bkgd(self, ch: ChType, attr: int = ..., /) -> None: ...
    def bkgdset(self, ch: ChType, attr: int = ..., /) -> None: ...
    def border(
        self,
        ls: ChType = ...,
        rs: ChType = ...,
        ts: ChType = ...,
        bs: ChType = ...,
        tl: ChType = ...,
        tr: ChType = ...,
        bl: ChType = ...,
        br: ChType = ...,
    ) -> None: ...
    @overload
    def box(self) -> None: ...
    @overload
    def box(self, vertch: ChType = ..., horch: ChType = ...) -> None: ...

    @overload
    def chgat(self, attr: int) -> None: ...
    @overload
    def chgat(self, num: int, attr: int) -> None: ...
    @overload
    def chgat(self, y: int, x: int, attr: int) -> None: ...
    @overload
    def chgat(self, y: int, x: int, num: int, attr: int) -> None: ...

    def chgat(self, *args, **kwargs):
        raise NotImplementedError

    def clear(self) -> None: ...
    def clearok(self, yes: int) -> None: ...
    def clrtobot(self) -> None: ...
    def clrtoeol(self) -> None: ...
    def cursyncup(self) -> None: ...
    @overload
    def delch(self) -> None: ...
    @overload
    def delch(self, y: int, x: int) -> None: ...
    def deleteln(self) -> None: ...
    @overload
    def derwin(self, begin_y: int, begin_x: int) -> CursesWindow: ...
    @overload
    def derwin(self, nlines: int, ncols: int, begin_y: int, begin_x: int) -> CursesWindow: ...
    def echochar(self, ch: ChType, attr: int = ..., /) -> None: ...
    def enclose(self, y: int, x: int, /) -> bool: ...

    def erase(self) -> None:
        raise NotImplementedError

    def getbegyx(self) -> tuple[int, int]: ...
    def getbkgd(self) -> tuple[int, int]: ...
    @overload
    def getch(self) -> int: ...
    @overload
    def getch(self, y: int, x: int) -> int: ...

    def getch(self, *args) -> int:
        raise NotImplementedError

    @overload
    def get_wch(self) -> int | str: ...
    @overload
    def get_wch(self, y: int, x: int) -> int | str: ...
    @overload
    def getkey(self) -> str: ...
    @overload
    def getkey(self, y: int, x: int) -> str: ...

    def getmaxyx(self) -> tuple[int, int]:
        raise NotImplementedError

    def getparyx(self) -> tuple[int, int]: ...
    @overload
    def getstr(self) -> bytes: ...
    @overload
    def getstr(self, n: int) -> bytes: ...
    @overload
    def getstr(self, y: int, x: int) -> bytes: ...
    @overload
    def getstr(self, y: int, x: int, n: int) -> bytes: ...
    def getyx(self) -> tuple[int, int]: ...
    @overload
    def hline(self, ch: ChType, n: int) -> None: ...
    @overload
    def hline(self, y: int, x: int, ch: ChType, n: int) -> None: ...
    def idcok(self, flag: bool) -> None: ...
    def idlok(self, yes: bool) -> None: ...
    def immedok(self, flag: bool) -> None: ...
    @overload
    def inch(self) -> int: ...
    @overload
    def inch(self, y: int, x: int) -> int: ...
    @overload
    def insch(self, ch: ChType, attr: int = ...) -> None: ...
    @overload
    def insch(self, y: int, x: int, ch: ChType, attr: int = ...) -> None: ...
    def insdelln(self, nlines: int) -> None: ...
    def insertln(self) -> None: ...
    @overload
    def insnstr(self, str: str, n: int, attr: int = ...) -> None: ...
    @overload
    def insnstr(self, y: int, x: int, str: str, n: int, attr: int = ...) -> None: ...
    @overload
    def insstr(self, str: str, attr: int = ...) -> None: ...
    @overload
    def insstr(self, y: int, x: int, str: str, attr: int = ...) -> None: ...
    @overload
    def instr(self, n: int = ...) -> bytes: ...
    @overload
    def instr(self, y: int, x: int, n: int = ...) -> bytes: ...
    def is_linetouched(self, line: int, /) -> bool: ...
    def is_wintouched(self) -> bool: ...

    def keypad(self, yes: bool, /) -> None:
        raise NotImplementedError

    def leaveok(self, yes: bool) -> None:
        raise NotImplementedError

    def move(self, new_y: int, new_x: int) -> None: ...
    def mvderwin(self, y: int, x: int) -> None: ...
    def mvwin(self, new_y: int, new_x: int) -> None: ...

    def nodelay(self, yes: bool) -> None:
        raise NotImplementedError

    def notimeout(self, yes: bool) -> None: ...
    @overload
    def noutrefresh(self) -> None: ...
    @overload
    def noutrefresh(
        self,
        pminrow: int,
        pmincol: int,
        sminrow: int,
        smincol: int,
        smaxrow: int,
        smaxcol: int,
    ) -> None: ...
    @overload
    def overlay(self, destwin: CursesWindow) -> None: ...
    @overload
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
    @overload
    def overwrite(self, destwin: CursesWindow) -> None: ...
    @overload
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
    def putwin(self, file: SupportsWrite[bytes], /) -> None: ...
    def redrawln(self, beg: int, num: int, /) -> None: ...
    def redrawwin(self) -> None: ...
    @overload
    def refresh(self) -> None: ...
    @overload
    def refresh(
        self,
        pminrow: int,
        pmincol: int,
        sminrow: int,
        smincol: int,
        smaxrow: int,
        smaxcol: int,
    ) -> None: ...

    def refresh(self, *args) -> None:
        raise NotImplementedError

    def resize(self, nlines: int, ncols: int) -> None: ...
    def scroll(self, lines: int = ...) -> None: ...
    def scrollok(self, flag: bool) -> None: ...
    def setscrreg(self, top: int, bottom: int, /) -> None: ...
    def standend(self) -> None: ...
    def standout(self) -> None: ...
    @overload
    def subpad(self, begin_y: int, begin_x: int) -> CursesWindow: ...
    @overload
    def subpad(self, nlines: int, ncols: int, begin_y: int, begin_x: int) -> CursesWindow: ...
    @overload
    def subwin(self, begin_y: int, begin_x: int) -> CursesWindow: ...
    @overload
    def subwin(self, nlines: int, ncols: int, begin_y: int, begin_x: int) -> CursesWindow: ...
    def syncdown(self) -> None: ...
    def syncok(self, flag: bool) -> None: ...
    def syncup(self) -> None: ...
    def timeout(self, delay: int) -> None: ...
    def touchline(self, start: int, count: int, changed: bool = ...) -> None: ...
    def touchwin(self) -> None: ...
    def untouchwin(self) -> None: ...
    @overload
    def vline(self, ch: ChType, n: int) -> None: ...
    @overload
    def vline(self, y: int, x: int, ch: ChType, n: int) -> None: ...
