"""Constants and membership tests for ASCII characters"""

# Copied from the CPython repository.
# https://github.com/python/cpython/blob/HEAD/Lib/curses/ascii.py

# pylint: disable=missing-function-docstring

from __future__ import annotations

from typing import overload


NUL = 0x00  # ^@
SOH = 0x01  # ^A
STX = 0x02  # ^B
ETX = 0x03  # ^C
EOT = 0x04  # ^D
ENQ = 0x05  # ^E
ACK = 0x06  # ^F
BEL = 0x07  # ^G
BS = 0x08  # ^H
TAB = 0x09  # ^I
HT = 0x09  # ^I
LF = 0x0A  # ^J
NL = 0x0A  # ^J
VT = 0x0B  # ^K
FF = 0x0C  # ^L
CR = 0x0D  # ^M
SO = 0x0E  # ^N
SI = 0x0F  # ^O
DLE = 0x10  # ^P
DC1 = 0x11  # ^Q
DC2 = 0x12  # ^R
DC3 = 0x13  # ^S
DC4 = 0x14  # ^T
NAK = 0x15  # ^U
SYN = 0x16  # ^V
ETB = 0x17  # ^W
CAN = 0x18  # ^X
EM = 0x19  # ^Y
SUB = 0x1A  # ^Z
ESC = 0x1B  # ^[
FS = 0x1C  # ^\
GS = 0x1D  # ^]
RS = 0x1E  # ^^
US = 0x1F  # ^_
SP = 0x20  # space
DEL = 0x7F  # delete

controlnames = [
    'NUL',
    'SOH',
    'STX',
    'ETX',
    'EOT',
    'ENQ',
    'ACK',
    'BEL',
    'BS',
    'HT',
    'LF',
    'VT',
    'FF',
    'CR',
    'SO',
    'SI',
    'DLE',
    'DC1',
    'DC2',
    'DC3',
    'DC4',
    'NAK',
    'SYN',
    'ETB',
    'CAN',
    'EM',
    'SUB',
    'ESC',
    'FS',
    'GS',
    'RS',
    'US',
    'SP',
]


def _ctoi(c: int | str) -> int:
    if isinstance(c, str):
        return ord(c)
    return c


def isalnum(c: int | str) -> bool:
    return isalpha(c) or isdigit(c)


def isalpha(c: int | str) -> bool:
    return isupper(c) or islower(c)


def isascii(c: int | str) -> bool:
    return 0 <= _ctoi(c) <= 127


def isblank(c: int | str) -> bool:
    return _ctoi(c) in (9, 32)


def iscntrl(c: int | str) -> bool:
    return 0 <= _ctoi(c) <= 31 or _ctoi(c) == 127


def isdigit(c: int | str) -> bool:
    return 48 <= _ctoi(c) <= 57


def isgraph(c: int | str) -> bool:
    return 33 <= _ctoi(c) <= 126


def islower(c: int | str) -> bool:
    return 97 <= _ctoi(c) <= 122


def isprint(c: int | str) -> bool:
    return 32 <= _ctoi(c) <= 126


def ispunct(c: int | str) -> bool:
    return isgraph(c) and not isalnum(c)


def isspace(c: int | str) -> bool:
    return _ctoi(c) in (9, 10, 11, 12, 13, 32)


def isupper(c: int | str) -> bool:
    return 65 <= _ctoi(c) <= 90


def isxdigit(c: int | str) -> bool:
    return isdigit(c) or (65 <= _ctoi(c) <= 70) or (97 <= _ctoi(c) <= 102)


def isctrl(c: int | str) -> bool:
    return 0 <= _ctoi(c) < 32


def ismeta(c: int | str) -> bool:
    return _ctoi(c) > 127


@overload
def ascii(c: int) -> int: ...  # pylint: disable=redefined-builtin


@overload
def ascii(c: str) -> str: ...


def ascii(c: int | str) -> int | str:
    if isinstance(c, str):
        return chr(_ctoi(c) & 0x7F)
    return _ctoi(c) & 0x7F


@overload
def ctrl(c: int) -> int: ...


@overload
def ctrl(c: str) -> str: ...


def ctrl(c: int | str) -> int | str:
    if isinstance(c, str):
        return chr(_ctoi(c) & 0x1F)
    return _ctoi(c) & 0x1F


@overload
def alt(c: int) -> int: ...


@overload
def alt(c: str) -> str: ...


def alt(c: int | str) -> int | str:
    if isinstance(c, str):
        return chr(_ctoi(c) | 0x80)
    return _ctoi(c) | 0x80


@overload
def unctrl(c: int) -> str: ...


@overload
def unctrl(c: str) -> str: ...


def unctrl(c: int | str) -> str:
    bits = _ctoi(c)
    if bits == 0x7F:
        rep = '^?'
    elif isprint(bits & 0x7F):
        rep = chr(bits & 0x7F)
    else:
        rep = '^' + chr(((bits & 0x7F) | 0x20) + 0x20)
    if bits & 0x80:
        return '!' + rep
    return rep
