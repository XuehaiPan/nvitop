# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-many-lines,too-many-arguments,too-many-positional-arguments

from __future__ import annotations

from typing import TYPE_CHECKING, final, overload

# pylint: disable-next=redefined-builtin
from nvitop.tui.library.curses import ascii  # type: ignore[no-redef]


if TYPE_CHECKING:
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


def baudrate() -> int:
    raise NotImplementedError


def beep() -> None:
    if not _initscr_called:
        raise error('must call initscr() first')
    assert _current_window is not None
    _current_window.addch(ascii.BEL)


def can_change_color() -> bool:
    raise NotImplementedError


def cbreak(flag: bool = True, /) -> None:
    raise NotImplementedError('must implement')


def color_content(color_number: int, /) -> tuple[int, int, int]:
    raise NotImplementedError


def color_pair(pair_number: int, /) -> int:  # pylint: disable=redefined-outer-name
    raise NotImplementedError('must implement')


def curs_set(visibility: int, /) -> int:
    raise NotImplementedError('must implement')


def def_prog_mode() -> None:
    raise NotImplementedError


def def_shell_mode() -> None:
    raise NotImplementedError


def delay_output(ms: int, /) -> None:
    raise NotImplementedError


def doupdate() -> None:
    raise NotImplementedError


def echo(flag: bool = True, /) -> None:
    raise NotImplementedError('must implement')


def endwin() -> None:
    raise NotImplementedError('must implement')


def erasechar() -> bytes:
    raise NotImplementedError


def filter() -> None:  # pylint: disable=redefined-builtin
    raise NotImplementedError


def flash() -> None:
    raise NotImplementedError('must implement')


def flushinp() -> None:
    raise NotImplementedError('must implement')


def getmouse() -> tuple[int, int, int, int, int]:
    raise NotImplementedError('must implement')


def getsyx() -> tuple[int, int]:
    raise NotImplementedError


def getwin(file: SupportsRead[bytes], /) -> CursesWindow:
    raise NotImplementedError


def halfdelay(tenths: int, /) -> None:
    raise NotImplementedError


def has_colors() -> bool:
    raise NotImplementedError


def has_ic() -> bool:
    raise NotImplementedError


def has_il() -> bool:
    raise NotImplementedError


def has_key(key: int, /) -> bool:
    raise NotImplementedError


def init_color(color_number: int, r: int, g: int, b: int, /) -> None:
    raise NotImplementedError('must implement')


def init_pair(pair_number: int, fg: int, bg: int, /) -> None:  # pylint: disable=redefined-outer-name
    raise NotImplementedError('must implement')


def initscr() -> CursesWindow:
    global _current_window, _initscr_called, _setupterm_called  # pylint: disable=global-statement

    if _initscr_called:
        assert _current_window is not None
        return _current_window

    _initscr_called = _setupterm_called = True

    _current_window = CursesWindow()
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
    raise NotImplementedError('must implement')


def mousemask(newmask: int, /) -> tuple[int, int]:
    raise NotImplementedError('must implement')


def napms(ms: int, /) -> int:
    raise NotImplementedError


def newpad(nlines: int, ncols: int, /) -> CursesWindow:
    raise NotImplementedError


def newwin(nlines: int, ncols: int, begin_y: int = ..., begin_x: int = ..., /) -> CursesWindow:
    raise NotImplementedError


def nl(flag: bool = True, /) -> None:
    raise NotImplementedError


def nocbreak() -> None:
    raise NotImplementedError('must implement')


def noecho() -> None:
    raise NotImplementedError('must implement')


def nonl() -> None:
    raise NotImplementedError


def noqiflush() -> None:
    raise NotImplementedError


def noraw() -> None:
    raise NotImplementedError


def pair_content(pair_number: int, /) -> tuple[int, int]:  # pylint: disable=redefined-outer-name
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


def setupterm(term: str | None = None, fd: int = -1) -> None:
    raise NotImplementedError('must implement')


def start_color() -> None:
    raise NotImplementedError('must implement')


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
    raise NotImplementedError('must implement')


def ungetmouse(
    id: int,  # pylint: disable=redefined-builtin
    x: int,
    y: int,
    z: int,
    bstate: int,
    /,
) -> None:
    raise NotImplementedError('must implement')


def update_lines_cols() -> None:
    raise NotImplementedError('must implement')


def use_default_colors() -> None:
    raise NotImplementedError('must implement')


def use_env(flag: bool, /) -> None:
    raise NotImplementedError


class error(Exception):  # noqa: N801,N818 # pylint: disable=invalid-name
    pass


@final
class CursesWindow:  # pylint: disable=too-many-public-methods
    def __init__(self) -> None:
        import platform  # pylint: disable=import-outside-toplevel

        self.encoding = 'utf-8'
        if platform.system() == 'Windows':
            import ctypes  # pylint: disable=import-outside-toplevel

            try:
                code_page = ctypes.windll.kernel32.GetConsoleOutputCP()  # type: ignore[attr-defined,unused-ignore]
            except (AttributeError, OSError):
                self.encoding = 'utf-8'
            else:
                self.encoding = f'cp{code_page}'

    @overload
    def addch(self, ch: ChType, attr: int = ...) -> None: ...

    @overload
    def addch(self, y: int, x: int, ch: ChType, attr: int = ...) -> None: ...

    def addch(self, *args: int | ChType, attr: int = 0) -> None:  # type: ignore[misc]
        raise NotImplementedError('must implement')

    @overload
    def addnstr(
        self,
        str: str,  # pylint: disable=redefined-builtin
        n: int,
        attr: int = ...,
    ) -> None: ...

    @overload
    def addnstr(
        self,
        y: int,
        x: int,
        str: str,  # pylint: disable=redefined-builtin
        n: int,
        attr: int = ...,
    ) -> None: ...

    def addnstr(self, *args: int, attr: int = ...) -> None:  # type: ignore[misc]
        raise NotImplementedError('must implement')

    @overload
    def addstr(
        self,
        str: str,  # pylint: disable=redefined-builtin
        attr: int = ...,
    ) -> None: ...

    @overload
    def addstr(
        self,
        y: int,
        x: int,
        str: str,  # pylint: disable=redefined-builtin
        attr: int = ...,
    ) -> None: ...

    def addstr(self, *args: int, attr: int = ...) -> None:  # type: ignore[misc]
        raise NotImplementedError('must implement')

    def attroff(self, attr: int, /) -> None:
        raise NotImplementedError

    def attron(self, attr: int, /) -> None:
        raise NotImplementedError

    def attrset(self, attr: int, /) -> None:
        raise NotImplementedError('must implement')

    def bkgd(self, ch: ChType, attr: int = ..., /) -> None:
        raise NotImplementedError

    def bkgdset(self, ch: ChType, attr: int = ..., /) -> None:
        raise NotImplementedError

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
    ) -> None:
        raise NotImplementedError

    @overload
    def box(self) -> None: ...

    @overload
    def box(self, vertch: ChType = ..., horch: ChType = ...) -> None: ...

    def box(self, *vhch: ChType) -> None:  # type: ignore[misc]
        raise NotImplementedError

    @overload
    def chgat(self, attr: int) -> None: ...

    @overload
    def chgat(self, num: int, attr: int) -> None: ...

    @overload
    def chgat(self, y: int, x: int, attr: int) -> None: ...

    @overload
    def chgat(self, y: int, x: int, num: int, attr: int) -> None: ...

    def chgat(self, *args: int) -> None:  # type: ignore[misc]
        raise NotImplementedError('must implement')

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

    @overload
    def delch(self) -> None: ...

    @overload
    def delch(self, y: int, x: int) -> None: ...

    def delch(self, *yx: int) -> None:  # type: ignore[misc]
        raise NotImplementedError

    def deleteln(self) -> None:
        raise NotImplementedError

    @overload
    def derwin(self, begin_y: int, begin_x: int) -> CursesWindow: ...

    @overload
    def derwin(self, nlines: int, ncols: int, begin_y: int, begin_x: int) -> CursesWindow: ...

    def derwin(self, *args: int) -> CursesWindow:  # type: ignore[misc]
        raise NotImplementedError

    def echochar(self, ch: ChType, attr: int = ..., /) -> None:
        raise NotImplementedError

    def enclose(self, y: int, x: int, /) -> bool:
        raise NotImplementedError

    def erase(self) -> None:
        raise NotImplementedError('must implement')

    def getbegyx(self) -> tuple[int, int]:
        raise NotImplementedError

    def getbkgd(self) -> tuple[int, int]:
        raise NotImplementedError

    @overload
    def getch(self) -> int: ...

    @overload
    def getch(self, y: int, x: int) -> int: ...

    def getch(self, *yx: int) -> int:  # type: ignore[misc]
        raise NotImplementedError('must implement')

    @overload
    def get_wch(self) -> int | str: ...

    @overload
    def get_wch(self, y: int, x: int) -> int | str: ...

    def get_wch(self, *yx: int) -> int | str:  # type: ignore[misc]
        raise NotImplementedError

    @overload
    def getkey(self) -> str: ...

    @overload
    def getkey(self, y: int, x: int) -> str: ...

    def getkey(self, *yx: int) -> int | str:  # type: ignore[misc]
        raise NotImplementedError

    def getmaxyx(self) -> tuple[int, int]:
        raise NotImplementedError('must implement')

    def getparyx(self) -> tuple[int, int]:
        raise NotImplementedError

    @overload
    def getstr(self) -> bytes: ...

    @overload
    def getstr(self, n: int) -> bytes: ...

    @overload
    def getstr(self, y: int, x: int) -> bytes: ...

    @overload
    def getstr(self, y: int, x: int, n: int) -> bytes: ...

    def getstr(self, *args: int) -> bytes:  # type: ignore[misc]
        raise NotImplementedError

    def getyx(self) -> tuple[int, int]:
        raise NotImplementedError

    @overload
    def hline(self, ch: ChType, n: int) -> None: ...

    @overload
    def hline(self, y: int, x: int, ch: ChType, n: int) -> None: ...

    def hline(self, *args: int | ChType) -> None:  # type: ignore[misc]
        raise NotImplementedError

    def idcok(self, flag: bool) -> None:
        raise NotImplementedError

    def idlok(self, yes: bool) -> None:
        raise NotImplementedError

    def immedok(self, flag: bool) -> None:
        raise NotImplementedError

    @overload
    def inch(self) -> int: ...

    @overload
    def inch(self, y: int, x: int) -> int: ...

    def inch(self, *yx: int) -> int:  # type: ignore[misc]
        raise NotImplementedError

    @overload
    def insch(self, ch: ChType, attr: int = ...) -> None: ...

    @overload
    def insch(self, y: int, x: int, ch: ChType, attr: int = ...) -> None: ...

    def insch(self, *args: int | ChType, attr: int = ...) -> None:  # type: ignore[misc]
        raise NotImplementedError

    def insdelln(self, nlines: int) -> None:
        raise NotImplementedError

    def insertln(self) -> None:
        raise NotImplementedError

    @overload
    def insnstr(
        self,
        str: str,  # pylint: disable=redefined-builtin
        n: int,
        attr: int = ...,
    ) -> None: ...

    @overload
    def insnstr(
        self,
        y: int,
        x: int,
        str: str,  # pylint: disable=redefined-builtin
        n: int,
        attr: int = ...,
    ) -> None: ...

    def insnstr(self, *args: int | str, attr: int = ...) -> None:  # type: ignore[misc]
        raise NotImplementedError

    @overload
    def insstr(
        self,
        str: str,  # pylint: disable=redefined-builtin
        attr: int = ...,
    ) -> None: ...

    @overload
    def insstr(
        self,
        y: int,
        x: int,
        str: str,  # pylint: disable=redefined-builtin
        attr: int = ...,
    ) -> None: ...

    def insstr(self, *args: int | str, attr: int = ...) -> None:  # type: ignore[misc]
        raise NotImplementedError

    @overload
    def instr(self, n: int = ...) -> bytes: ...

    @overload
    def instr(self, y: int, x: int, n: int = ...) -> bytes: ...

    def instr(self, *args: int) -> bytes:  # type: ignore[misc]
        raise NotImplementedError

    def is_linetouched(self, line: int, /) -> bool:
        raise NotImplementedError

    def is_wintouched(self) -> bool:
        raise NotImplementedError

    def keypad(self, yes: bool, /) -> None:
        raise NotImplementedError('must implement')

    def leaveok(self, yes: bool) -> None:
        raise NotImplementedError('must implement')

    def move(self, new_y: int, new_x: int) -> None:
        raise NotImplementedError

    def mvderwin(self, y: int, x: int) -> None:
        raise NotImplementedError

    def mvwin(self, new_y: int, new_x: int) -> None:
        raise NotImplementedError

    def nodelay(self, yes: bool) -> None:
        raise NotImplementedError('must implement')

    def notimeout(self, yes: bool) -> None:
        raise NotImplementedError

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

    def noutrefresh(self, *args: int) -> None:  # type: ignore[misc]
        raise NotImplementedError

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

    def overlay(self, destwin: CursesWindow, *args: int) -> None:  # type: ignore[misc]
        raise NotImplementedError

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

    def overwrite(self, destwin: CursesWindow, *args: int) -> None:  # type: ignore[misc]
        raise NotImplementedError

    def putwin(self, file: SupportsWrite[bytes], /) -> None:
        raise NotImplementedError

    def redrawln(self, beg: int, num: int, /) -> None:
        raise NotImplementedError

    def redrawwin(self) -> None:
        raise NotImplementedError

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
    ) -> None:
        raise NotImplementedError

    def refresh(self, *args: int) -> None:  # type: ignore[misc]
        raise NotImplementedError('must implement')

    def resize(self, nlines: int, ncols: int) -> None:
        raise NotImplementedError

    def scroll(self, lines: int = ...) -> None:
        raise NotImplementedError

    def scrollok(self, flag: bool) -> None:
        raise NotImplementedError

    def setscrreg(self, top: int, bottom: int, /) -> None:
        raise NotImplementedError

    def standend(self) -> None:
        raise NotImplementedError

    def standout(self) -> None:
        raise NotImplementedError

    @overload
    def subpad(self, begin_y: int, begin_x: int) -> CursesWindow: ...

    @overload
    def subpad(self, nlines: int, ncols: int, begin_y: int, begin_x: int) -> CursesWindow: ...

    def subpad(self, *args: int) -> CursesWindow:  # type: ignore[misc]
        raise NotImplementedError

    @overload
    def subwin(self, begin_y: int, begin_x: int) -> CursesWindow: ...

    @overload
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

    def touchline(self, start: int, count: int, changed: bool = ...) -> None:
        raise NotImplementedError

    def touchwin(self) -> None:
        raise NotImplementedError

    def untouchwin(self) -> None:
        raise NotImplementedError

    @overload
    def vline(self, ch: ChType, n: int) -> None: ...

    @overload
    def vline(self, y: int, x: int, ch: ChType, n: int) -> None: ...

    def vline(self, *args: int | ChType) -> None:  # type: ignore[misc]
        raise NotImplementedError
