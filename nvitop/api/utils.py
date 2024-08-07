# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2024 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities of nvitop APIs."""

# pylint: disable=invalid-name

from __future__ import annotations

import ctypes
import datetime
import functools
import math
import os
import re
import sys
import time
from collections.abc import KeysView
from typing import Any, Callable, Generator, Iterable, Iterator, TypeVar

from psutil import WINDOWS


__all__ = [
    'NA',
    'NaType',
    'NotApplicable',
    'NotApplicableType',
    'UINT_MAX',
    'ULONGLONG_MAX',
    'KiB',
    'MiB',
    'GiB',
    'TiB',
    'PiB',
    'SIZE_UNITS',
    'bytes2human',
    'human2bytes',
    'timedelta2human',
    'utilization2string',
    'colored',
    'set_color',
    'boolify',
    'Snapshot',
]


if WINDOWS:
    try:
        from colorama import init
    except ImportError:
        pass
    else:
        init()

try:
    from termcolor import colored as _colored
except ImportError:

    def _colored(  # type: ignore[misc] # pylint: disable=unused-argument,too-many-arguments
        text: str,
        color: str | None = None,
        on_color: str | None = None,
        attrs: Iterable[str] | None = None,
        *,
        no_color: bool | None = None,
        force_color: bool | None = None,
    ) -> str:
        return text


COLOR: bool = sys.stdout.isatty()


def set_color(value: bool) -> None:
    """Force enable text coloring."""
    global COLOR  # pylint: disable=global-statement
    COLOR = bool(value)
    if COLOR:
        os.environ['FORCE_COLOR'] = '1'
        os.environ.pop('NO_COLOR', None)
    else:
        os.environ.pop('FORCE_COLOR', None)
        os.environ['NO_COLOR'] = '1'


def colored(
    text: str,
    color: str | None = None,
    on_color: str | None = None,
    attrs: Iterable[str] | None = None,
) -> str:
    """Colorize text with ANSI color escape codes.

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.

    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.

    Examples:
        >>> colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        >>> colored('Hello, World!', 'green')
    """
    if COLOR:
        return _colored(text, color=color, on_color=on_color, attrs=attrs)  # type: ignore[arg-type]
    return text


class NaType(str):
    """A singleton (:const:`str: 'N/A'`) class represents a not applicable value.

    The :const:`NA` instance behaves like a :class:`str` instance (:const:`'N/A'`) when doing string
    manipulation (e.g. concatenation). For arithmetic operations, for example ``NA / 1024 / 1024``,
    it acts like the :data:`math.nan`.

    Examples:
        >>> NA
        'N/A'

        >>> 'memory usage: {}'.format(NA)  # NA is an instance of `str`
        'memory usage: N/A'
        >>> NA.lower()                     # NA is an instance of `str`
        'n/a'
        >>> NA.ljust(5)                    # NA is an instance of `str`
        'N/A  '
        >>> NA + ' str'                    # string contamination if the operand is a string
        'N/A str'

        >>> float(NA)                      # explicit conversion to float (`math.nan`)
        nan
        >>> NA + 1                         # auto-casting to float if the operand is a number
        nan
        >>> NA * 1024                      # auto-casting to float if the operand is a number
        nan
        >>> NA / (1024 * 1024)             # auto-casting to float if the operand is a number
        nan
    """

    # NOTE: Decorate this class with `@final` and remove `noqa` when we drop Python 3.7 support.
    def __new__(cls) -> NaType:  # noqa: PYI034
        """Get the singleton instance (:const:`nvitop.NA`)."""
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, 'N/A')
        return cls._instance

    def __bool__(self) -> bool:
        """Convert :const:`NA` to :class:`bool` and return :data:`False`.

        >>> bool(NA)
        False
        """
        return False

    def __int__(self) -> int:
        """Convert :const:`NA` to :class:`int` and return :const:`0`.

        >>> int(NA)
        0
        """
        return 0

    def __float__(self) -> float:
        """Convert :const:`NA` to :class:`float` and return :data:`math.nan`.

        >>> float(NA)
        nan
        >>> float(NA) is math.nan
        True
        """
        return math.nan

    def __add__(self, other: object) -> str | float:  # type: ignore[override]
        """Return :data:`math.nan` if the operand is a number or uses string concatenation if the operand is a string (``NA + other``).

        A special case is when the operand is :const:`nvitop.NA` itself, the result is
        :data:`math.nan` instead of :const:`'N/AN/A'`.

        >>> NA + ' str'
        'N/A str'
        >>> NA + NA
        nan
        >>> NA + 1
        nan
        >>> NA + 1.0
        nan
        """  # pylint: disable=line-too-long
        if isinstance(other, (int, float)):
            return float(self) + other
        if other is NA:
            return float(self)
        return super().__add__(other)  # type: ignore[operator]

    def __radd__(self, other: object) -> str | float:
        """Return :data:`math.nan` if the operand is a number or uses string concatenation if the operand is a string (``other + NA``).

        >>> 'str' + NA
        'strN/A'
        >>> 1 + NA
        nan
        >>> 1.0 + NA
        nan
        """  # pylint: disable=line-too-long
        if isinstance(other, (int, float)):
            return other + float(self)
        return NotImplemented

    def __sub__(self, other: object) -> float:
        """Return :data:`math.nan` if the operand is a number (``NA - other``).

        >>> NA - 'str'
        TypeError: unsupported operand type(s) for -: 'NaType' and 'str'
        >>> NA - NA
        'N/AN/A'
        >>> NA + 1
        nan
        >>> NA + 1.0
        nan
        """
        if isinstance(other, (int, float)):
            return float(self) - other
        if other is NA:
            return float(self)
        return NotImplemented

    def __rsub__(self, other: object) -> float:
        """Return :data:`math.nan` if the operand is a number (``other - NA``).

        >>> 'str' - NA
        TypeError: unsupported operand type(s) for -: 'str' and 'NaType'
        >>> 1 - NA
        nan
        >>> 1.0 - NA
        nan
        """
        if isinstance(other, (int, float)):
            return other - float(self)
        return NotImplemented

    def __mul__(self, other: object) -> float:  # type: ignore[override]
        """Return :data:`math.nan` if the operand is a number (``NA * other``).

        A special case is when the operand is :const:`nvitop.NA` itself, the result is also :data:`math.nan`.

        >>> NA * 1024
        nan
        >>> NA * 1024.0
        nan
        >>> NA * NA
        nan
        """
        if isinstance(other, (int, float)):
            return float(self) * other
        if other is NA:
            return float(self)
        return NotImplemented

    def __rmul__(self, other: object) -> float:  # type: ignore[override]
        """Return :data:`math.nan` if the operand is a number (``other * NA``).

        >>> 1024 * NA
        nan
        >>> 1024.0 * NA
        nan
        """
        if isinstance(other, (int, float)):
            return other * float(self)
        return NotImplemented

    def __truediv__(self, other: object) -> float:
        """Return :data:`math.nan` if the operand is a number (``NA / other``).

        >>> NA / 1024
        nan
        >>> NA / 1024.0
        nan
        >>> NA / 0
        ZeroDivisionError: float division by zero
        >>> NA / 0.0
        ZeroDivisionError: float division by zero
        >>> NA / NA
        nan
        """
        if isinstance(other, (int, float)):
            return float(self) / other
        if other is NA:
            return float(self)
        return NotImplemented

    def __rtruediv__(self, other: object) -> float:
        """Return :data:`math.nan` if the operand is a number (``other / NA``).

        >>> 1024 / NA
        nan
        >>> 1024.0 / NA
        nan
        """
        if isinstance(other, (int, float)):
            return other / float(self)
        return NotImplemented

    def __floordiv__(self, other: object) -> float:
        """Return :data:`math.nan` if the operand is a number (``NA // other``).

        >>> NA // 1024
        nan
        >>> NA // 1024.0
        nan
        >>> NA / 0
        ZeroDivisionError: float division by zero
        >>> NA / 0.0
        ZeroDivisionError: float division by zero
        >>> NA // NA
        nan
        """
        if isinstance(other, (int, float)):
            return float(self) // other
        if other is NA:
            return float(self)
        return NotImplemented

    def __rfloordiv__(self, other: object) -> float:
        """Return :data:`math.nan` if the operand is a number (``other // NA``).

        >>> 1024 // NA
        nan
        >>> 1024.0 // NA
        nan
        """
        if isinstance(other, (int, float)):
            return other // float(self)
        return NotImplemented

    def __mod__(self, other: object) -> float:  # type: ignore[override]
        """Return :data:`math.nan` if the operand is a number (``NA % other``).

        >>> NA % 1024
        nan
        >>> NA % 1024.0
        nan
        >>> NA % 0
        ZeroDivisionError: float modulo
        >>> NA % 0.0
        ZeroDivisionError: float modulo
        """
        if isinstance(other, (int, float)):
            return float(self) % other
        if other is NA:
            return float(self)
        return NotImplemented

    def __rmod__(self, other: object) -> float:
        """Return :data:`math.nan` if the operand is a number (``other % NA``).

        >>> 1024 % NA
        nan
        >>> 1024.0 % NA
        nan
        """
        if isinstance(other, (int, float)):
            return other % float(self)
        return NotImplemented

    def __divmod__(self, other: object) -> tuple[float, float]:
        """The pair ``(NA // other, NA % other)`` (``divmod(NA, other)``).

        >>> divmod(NA, 1024)
        (nan, nan)
        >>> divmod(NA, 1024.0)
        (nan, nan)
        >>> divmod(NA, 0)
        ZeroDivisionError: float floor division by zero
        >>> divmod(NA, 0.0)
        ZeroDivisionError: float floor division by zero
        """
        return (self // other, self % other)

    def __rdivmod__(self, other: object) -> tuple[float, float]:
        """The pair ``(other // NA, other % NA)`` (``divmod(other, NA)``).

        >>> divmod(1024, NA)
        (nan, nan)
        >>> divmod(1024.0, NA)
        (nan, nan)
        """
        return (other // self, other % self)

    def __pos__(self) -> float:
        """Return :data:`math.nan` (``+NA``).

        >>> +NA
        nan
        """
        return +float(self)

    def __neg__(self) -> float:
        """Return :data:`math.nan` (``-NA``).

        >>> -NA
        nan
        """
        return -float(self)

    def __abs__(self) -> float:
        """Return :data:`math.nan` (``abs(NA)``).

        >>> abs(NA)
        nan
        """
        return abs(float(self))

    def __round__(self, ndigits: int | None = None) -> int | float:
        """Round :const:`nvitop.NA` to ``ndigits`` decimal places, defaulting to :const:`0`.

        If ``ndigits`` is omitted or :data:`None`, returns :const:`0`, otherwise returns :data:`math.nan`.

        >>> round(NA)
        0
        >>> round(NA, 0)
        nan
        >>> round(NA, 1)
        nan
        """
        if ndigits is None:
            return int(self)
        return round(float(self), ndigits)

    def __lt__(self, x: object) -> bool:
        """The :const:`nvitop.NA` is always greater than any number, or uses the dictionary order for string."""
        if isinstance(x, (int, float)):
            return False
        return super().__lt__(x)  # type: ignore[operator]

    def __le__(self, x: object) -> bool:
        """The :const:`nvitop.NA` is always greater than any number, or uses the dictionary order for string."""
        if isinstance(x, (int, float)):
            return False
        return super().__le__(x)  # type: ignore[operator]

    def __gt__(self, x: object) -> bool:
        """The :const:`nvitop.NA` is always greater than any number, or uses the dictionary order for string."""
        if isinstance(x, (int, float)):
            return True
        return super().__gt__(x)  # type: ignore[operator]

    def __ge__(self, x: object) -> bool:
        """The :const:`nvitop.NA` is always greater than any number, or uses the dictionary order for string."""
        if isinstance(x, (int, float)):
            return True
        return super().__ge__(x)  # type: ignore[operator]

    def __format__(self, format_spec: str) -> str:
        """Format :const:`nvitop.NA` according to ``format_spec``."""
        try:
            return super().__format__(format_spec)
        except ValueError:
            return format(math.nan, format_spec)


NotApplicableType = NaType

# isinstance(NA, str) -> True
# NA == 'N/A'         -> True
# NA is NaType()      -> True (`NaType` is a singleton class)
NA = NaType()
NA.__doc__ = """The singleton instance of :class:`NaType`. The actual value is :const:`str: 'N/A'`."""  # pylint: disable=attribute-defined-outside-init

NotApplicable = NA

UINT_MAX: int = ctypes.c_uint(-1).value  # 0xFFFFFFFF
"""The maximum value of :class:`ctypes.c_uint`."""
ULONGLONG_MAX: int = ctypes.c_ulonglong(-1).value  # 0XFFFFFFFFFFFFFFFF
"""The maximum value of :class:`ctypes.c_ulonglong`."""

KiB: int = 1 << 10
"""Kibibyte (1024)"""

MiB: int = 1 << 20
"""Mebibyte (1024 * 1024)"""

GiB: int = 1 << 30
"""Gibibyte (1024 * 1024 * 1024)"""

TiB: int = 1 << 40
"""Tebibyte (1024 * 1024 * 1024 * 1024)"""

PiB: int = 1 << 50
"""Pebibyte (1024 * 1024 * 1024 * 1024 * 1024)"""

SIZE_UNITS: dict[str | None, int] = {
    None: 1,
    '': 1,
    'B': 1,
    'KiB': KiB,
    'MiB': MiB,
    'GiB': GiB,
    'TiB': TiB,
    'PiB': PiB,
    'KB': 1000,
    'MB': 1000**2,
    'GB': 1000**3,
    'TB': 1000**4,
    'PB': 1000**5,
}
"""Units of storage and memory measurements."""
SIZE_PATTERN: re.Pattern = re.compile(
    r'^\s*\+?\s*(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTP]i?B?|B?)\s*$',
    flags=re.IGNORECASE,
)
"""The regex pattern for human readable size."""


# pylint: disable-next=too-many-return-statements,too-many-branches
def bytes2human(
    b: int | float | NaType,  # noqa: PYI041
    *,
    min_unit: int = 1,
) -> str:
    """Convert bytes to a human readable string."""
    if b == NA:
        return NA

    if not isinstance(b, int):
        try:
            b = round(float(b))
        except ValueError:
            return NA

    if b < KiB and min_unit < KiB:
        return f'{b}B'
    if b < MiB and min_unit <= KiB:
        return f'{round(b / KiB)}KiB'
    if b < 100 * MiB and min_unit <= MiB:
        return f'{round(b / MiB, 2):.2f}MiB'
    if b < 1000 * MiB and min_unit <= MiB:
        return f'{round(b / MiB, 1):.1f}MiB'
    if b < 20 * GiB and min_unit <= MiB:
        return f'{round(b / MiB)}MiB'
    if b < 100 * GiB and min_unit <= GiB:
        return f'{round(b / GiB, 2):.2f}GiB'
    if b < 1000 * GiB and min_unit <= GiB:
        return f'{round(b / GiB, 1):.1f}GiB'
    if b < 100 * TiB and min_unit <= TiB:
        return f'{round(b / TiB, 2):.2f}TiB'
    if b < 1000 * TiB and min_unit <= TiB:
        return f'{round(b / TiB, 1):.1f}TiB'
    if b < 100 * PiB:
        return f'{round(b / PiB, 2):.2f}PiB'
    return f'{round(b / PiB, 1):.1f}PiB'


def human2bytes(s: int | str) -> int:
    """Convert a human readable size string (*case insensitive*) to bytes.

    Raises:
        ValueError:
            If cannot convert the given size string.

    Examples:
        >>> human2bytes('500B')
        500
        >>> human2bytes('10k')
        10000
        >>> human2bytes('10ki')
        10240
        >>> human2bytes('1M')
        1000000
        >>> human2bytes('1MiB')
        1048576
        >>> human2bytes('1.5GiB')
        1610612736
    """
    if isinstance(s, int):
        if s >= 0:
            return s
        raise ValueError(f'Cannot convert {s!r} to bytes.')

    match = SIZE_PATTERN.match(s)
    if match is None:
        raise ValueError(f'Cannot convert {s!r} to bytes.')
    size, unit = match.groups()
    unit = unit.upper().replace('I', 'i').replace('B', '') + 'B'
    return int(float(size) * SIZE_UNITS[unit])


def timedelta2human(
    dt: int | float | datetime.timedelta | NaType,  # noqa: PYI041
    *,
    round: bool = False,  # pylint: disable=redefined-builtin
) -> str:
    """Convert a number in seconds or a :class:`datetime.timedelta` instance to a human readable string."""
    if isinstance(dt, (int, float)):
        dt = datetime.timedelta(seconds=dt)

    if not isinstance(dt, datetime.timedelta):
        return NA

    if dt.days >= 4 or (round and dt.days >= 1):
        return f'{dt.days + dt.seconds / 86400:.1f} days'

    hours, seconds = divmod(86400 * dt.days + dt.seconds, 3600)
    if hours > 0:
        return '{:d}:{:02d}:{:02d}'.format(hours, *divmod(seconds, 60))
    return '{:d}:{:02d}'.format(*divmod(seconds, 60))


def utilization2string(utilization: int | float | NaType) -> str:  # noqa: PYI041
    """Convert a utilization rate to string."""
    if utilization != NA:
        if isinstance(utilization, int):
            return f'{utilization}%'
        if isinstance(utilization, float):
            return f'{utilization:.1f}%'
    return NA


def boolify(string: str, default: Any = None) -> bool:
    """Convert the given value, usually a string, to boolean."""
    if string.lower() in {'true', 'yes', 'on', 'enabled', '1'}:
        return True
    if string.lower() in {'false', 'no', 'off', 'disabled', '0'}:
        return False
    if default is not None:
        return bool(default)
    return bool(string)


class Snapshot:
    """A dict-like object holds the snapshot values.

    The value can be accessed by ``snapshot.name`` or ``snapshot['name']`` syntax.
    The Snapshot can also be converted to a dictionary by ``dict(snapshot)`` or ``{**snapshot}``.

    Missing attributes will be automatically fetched from the original object.
    """

    def __init__(self, real: Any, **items: Any) -> None:
        """Initialize a new :class:`Snapshot` object with the given attributes."""
        self.real = real
        self.timestamp = time.time()
        for key, value in items.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Return a string representation of the snapshot."""
        keys = set(self.__dict__.keys()).difference({'real', 'timestamp'})
        keys = ['real', *sorted(keys)]
        keyvals = []
        for key in keys:
            value = getattr(self, key)
            keyval = f'{key}={value!r}'
            if isinstance(value, Snapshot):
                keyval = keyval.replace('\n', '\n    ')  # extra indentation for nested snapshots
            keyvals.append(keyval)
        return '{}{}(\n    {},\n)'.format(
            self.real.__class__.__name__,
            self.__class__.__name__,
            ',\n    '.join(keyvals),
        )

    def __hash__(self) -> int:
        """Return a hash value of the snapshot."""
        return hash((self.real, self.timestamp))

    def __getattr__(self, name: str) -> Any:
        """Get a member from the instance.

        If the attribute is not defined, fetches from the original object and makes a function call.
        """
        try:
            return super().__getattr__(name)  # type: ignore[misc]
        except AttributeError:
            attribute = getattr(self.real, name)
            if callable(attribute):
                attribute = attribute()

            setattr(self, name, attribute)
            return attribute

    def __getitem__(self, name: str) -> Any:
        """Support ``snapshot['name']`` syntax."""
        try:
            return getattr(self, name)
        except AttributeError as ex:
            raise KeyError(name) from ex

    def __setitem__(self, name: str, value: Any) -> None:
        """Support ``snapshot['name'] = value`` syntax."""
        setattr(self, name, value)

    def __iter__(self) -> Iterator[str]:
        """Support ``for name in snapshot`` syntax and ``*`` tuple unpack ``[*snapshot]`` syntax."""

        def gen() -> Generator[str]:
            yield from (name for name in self.__dict__ if name not in {'real', 'timestamp'})

        return gen()

    def keys(self) -> Iterable[str]:
        # pylint: disable-next=line-too-long
        """Support ``**`` dictionary unpack ``{**snapshot}`` / ``dict(**snapshot)`` syntax and ``dict(snapshot)`` dictionary conversion."""
        return KeysView(self)  # type: ignore[arg-type]


Method = TypeVar('Method', bound=Callable[..., Any])


# Modified from psutil (https://github.com/giampaolo/psutil)
def memoize_when_activated(method: Method) -> Method:
    """A memoize decorator which is disabled by default.

    It can be activated and deactivated on request. For efficiency reasons it can be used only
    against class methods accepting no arguments.
    """

    @functools.wraps(method)
    def wrapped(self: object, *args: Any, **kwargs: Any) -> Any:
        try:
            # case 1: we previously entered oneshot() ctx
            # pylint: disable-next=protected-access
            ret = self._cache[method]  # type: ignore[attr-defined]
        except AttributeError:
            # case 2: we never entered oneshot() ctx
            return method(self, *args, **kwargs)
        except KeyError:
            # case 3: we entered oneshot() ctx but there's no cache
            # for this entry yet
            ret = method(self, *args, **kwargs)
            try:
                # pylint: disable-next=protected-access
                self._cache[method] = ret  # type: ignore[attr-defined]
            except AttributeError:
                # multi-threading race condition, see:
                # https://github.com/giampaolo/psutil/issues/1948
                pass
        return ret

    def cache_activate(self: object) -> None:
        """Activate cache.

        Expects an instance. Cache will be stored as a "_cache" instance attribute.
        """
        if not hasattr(self, '_cache'):
            # pylint: disable-next=protected-access
            self._cache = {}  # type: ignore[attr-defined]

    def cache_deactivate(self: object) -> None:
        """Deactivate and clear cache."""
        try:
            # pylint: disable-next=protected-access
            del self._cache  # type: ignore[attr-defined]
        except AttributeError:
            pass

    wrapped.cache_activate = cache_activate  # type: ignore[attr-defined]
    wrapped.cache_deactivate = cache_deactivate  # type: ignore[attr-defined]
    return wrapped  # type: ignore[return-value]
