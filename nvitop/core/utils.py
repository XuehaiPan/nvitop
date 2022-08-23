# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Utilities of nvitop APIs."""

# pylint: disable=invalid-name

import datetime
import functools
import math
import re
import sys
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Union

from psutil import WINDOWS


__all__ = [
    'NA',
    'NaType',
    'NotApplicable',
    'NotApplicableType',
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

    def _colored(  # pylint: disable=unused-argument
        text: str,
        color: Optional[str] = None,
        on_color: Optional[str] = None,
        attrs: Iterable[str] = None,
    ) -> str:
        return text


COLOR = sys.stdout.isatty()


def set_color(value: bool) -> None:
    """Force enables text coloring."""

    global COLOR  # pylint: disable=global-statement
    COLOR = bool(value)


def colored(
    text: str,
    color: Optional[str] = None,
    on_color: Optional[str] = None,
    attrs: Iterable[str] = None,
) -> str:
    """Colorizes text.

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
        return _colored(text, color=color, on_color=on_color, attrs=attrs)
    return text


class NaType(str):
    """A singleton (:const:`str: 'N/A'`) class represents a not applicable value.

    The :const:`NA` instance behaves like a :class:`str` instance (:const:`'N/A'`) when doing string
    manipulation (e.g. concatenation). For arithmetic operations, for example :code:`NA / 1024 / 1024`,
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
        >>> NA + 'str'                     # string contamination if the operand is a string
        'N/Astr'

        >>> float(NA)                      # explicit conversion to float (`math.nan`)
        nan
        >>> NA + 1                         # auto-casting to float if the operand is a number
        nan
        >>> NA * 1024                      # auto-casting to float if the operand is a number
        nan
        >>> NA / (1024 * 1024)             # auto-casting to float if the operand is a number
        nan
    """

    def __new__(cls) -> 'NaType':
        """Gets the singleton instance (:const:`nvitop.NA`)."""

        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, 'N/A')
        return cls._instance

    def __bool__(self) -> bool:
        """Converts :const:`NA` to :class:`bool`. Returns :data:`False`.

        >>> bool(NA)
        False
        """

        return False

    def __int__(self) -> int:
        """Converts :const:`NA` to :class:`int`. Returns :const:`0`.

        >>> int(NA)
        0
        """

        return 0

    def __float__(self) -> float:
        """Converts :const:`NA` to :class:`float`. Returns :data:`math.nan`.

        >>> float(NA)
        nan
        >>> float(NA) is math.nan
        True
        """

        return math.nan

    def __add__(self, other: object) -> Union[str, float]:
        """:const:`nvitop.NA` + other: Returns :data:`math.nan` if the operand is a number or uses
        string concatenation if the operand is a string. A special case is when the operand is
        :const:`nvitop.NA` itself, the result is :data:`math.nan` instead of :const:`'N/AN/A'`.

        >>> NA + 'str'
        'N/Astr'
        >>> NA + NA
        nan
        >>> NA + 1
        nan
        >>> NA + 1.0
        nan
        """

        if isinstance(other, (int, float)) or other is NA:
            return float(self) + other
        return super().__add__(other)

    def __radd__(self, other: object) -> Union[str, float]:
        """other + :const:`nvitop.NA`: Returns :data:`math.nan` if the operand is a number.

        >>> 'str' + NA
        'strN/A'
        >>> 1 + NA
        nan
        >>> 1.0 + NA
        nan
        """

        if isinstance(other, (int, float)):
            return other + float(self)
        return NotImplemented

    def __sub__(self, other: object) -> float:
        """:const:`nvitop.NA` - other: Returns :data:`math.nan` if the operand is a number.

        >>> NA - 'str'
        TypeError: unsupported operand type(s) for -: 'NaType' and 'str'
        >>> NA - NA
        'N/AN/A'
        >>> NA + 1
        nan
        >>> NA + 1.0
        nan
        """

        if isinstance(other, (int, float)) or other is NA:
            return float(self) - other
        return NotImplemented

    def __rsub__(self, other: object) -> float:
        """other - :const:`nvitop.NA`: Returns :data:`math.nan` if the operand is a number.

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

    def __mul__(self, other: object) -> float:
        """:const:`nvitop.NA` * other: Returns :data:`math.nan` if the operand is a number. A special
        case is when the operand is :const:`nvitop.NA` itself, the result is also :data:`math.nan`.

        >>> NA * 1024
        nan
        >>> NA * 1024.0
        nan
        >>> NA * NA
        nan
        """

        if isinstance(other, (int, float)) or other is NA:
            return float(self) * other
        return NotImplemented

    def __rmul__(self, other: object) -> float:
        """other * :const:`nvitop.NA`: Returns :data:`math.nan` if the operand is a number.

        >>> 1024 * NA
        nan
        >>> 1024.0 * NA
        nan
        """

        if isinstance(other, (int, float)):
            return other * float(self)
        return NotImplemented

    def __truediv__(self, other: object) -> float:
        """:const:`nvitop.NA` / other: Returns :data:`math.nan` if the operand is a number.

        >>> NA / 1024
        nan
        >>> NA / 1024.0
        nan
        >>> NA / 0
        ZeroDivisionError: float division by zero
        >>> NA / 0.0
        ZeroDivisionError: float division by zero
        """

        if isinstance(other, (int, float)):
            return float(self) / other
        return NotImplemented

    def __rtruediv__(self, other: object) -> float:
        """other / :const:`nvitop.NA`: Returns :data:`math.nan` if the operand is a number.

        >>> 1024 / NA
        nan
        >>> 1024.0 / NA
        nan
        """

        if isinstance(other, (int, float)):
            return other / float(self)
        return NotImplemented

    def __floordiv__(self, other: object) -> float:
        """:const:`nvitop.NA` // other: Returns :data:`math.nan` if the operand is a number.

        >>> NA // 1024
        nan
        >>> NA // 1024.0
        nan
        >>> NA / 0
        ZeroDivisionError: float division by zero
        >>> NA / 0.0
        ZeroDivisionError: float division by zero
        """

        if isinstance(other, (int, float)):
            return float(self) // other
        return NotImplemented

    def __rfloordiv__(self, other: object) -> float:
        """other // :const:`nvitop.NA`: Returns :data:`math.nan` if the operand is a number.

        >>> 1024 // NA
        nan
        >>> 1024.0 // NA
        nan
        """

        if isinstance(other, (int, float)):
            return other // float(self)
        return NotImplemented

    def __mod__(self, other: object) -> float:
        """:const:`nvitop.NA` % other: Returns :data:`math.nan` if the operand is a number.

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
        return NotImplemented

    def __rmod__(self, other: object) -> float:
        """other % :const:`nvitop.NA`: Returns :data:`math.nan` if the operand is a number.

        >>> 1024 % NA
        nan
        >>> 1024.0 % NA
        nan
        """

        if isinstance(other, (int, float)):
            return other % float(self)
        return NotImplemented

    def __divmod__(self, other: object) -> Tuple[float, float]:
        """divmod(:const:`nvitop.NA`, other): The pair (:const:`nvitop.NA` // other, :const:`nvitop.NA` % other).

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

    def __rdivmod__(self, other: object) -> Tuple[float, float]:
        """divmod(other, :const:`nvitop.NA`): The pair (other // :const:`nvitop.NA`, other % :const:`nvitop.NA`).

        >>> divmod(1024, NA)
        (nan, nan)
        >>> divmod(1024.0, NA)
        (nan, nan)
        """

        return (other // self, other % self)

    def __pos__(self) -> float:
        """+:const:`nvitop.NA`: Returns :data:`math.nan`.

        >>> +NA
        nan
        """

        return +float(self)

    def __neg__(self) -> float:
        """+:const:`nvitop.NA`: Returns :data:`math.nan`.

        >>> -NA
        nan
        """

        return -float(self)

    def __abs__(self) -> float:
        """abs(NA): Returns :data:`math.nan`.

        >>> abs(NA)
        nan
        """

        return abs(float(self))

    def __round__(self, ndigits: Optional[int] = None) -> Union[int, float]:
        """Rounds :const:`nvitop.NA` to ``ndigits`` decimal places, defaulting to :const:`0`.

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
        """The :const:`nvitop.NA` is always greater than any number. Use the dictionary order for string."""

        if isinstance(x, (int, float)):
            return False
        return super().__lt__(x)

    def __le__(self, x: object) -> bool:
        """The :const:`nvitop.NA` is always greater than any number. Use the dictionary order for string."""

        if isinstance(x, (int, float)):
            return False
        return super().__le__(x)

    def __gt__(self, x: object) -> bool:
        """The :const:`nvitop.NA` is always greater than any number. Use the dictionary order for string."""

        if isinstance(x, (int, float)):
            return True
        return super().__gt__(x)

    def __ge__(self, x: object) -> bool:
        """The :const:`nvitop.NA` is always greater than any number. Use the dictionary order for string."""

        if isinstance(x, (int, float)):
            return True
        return super().__ge__(x)

    def __format__(self, format_spec: str) -> str:
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

KiB = 1 << 10
"""Kibibyte (1024)"""

MiB = 1 << 20
"""Mebibyte (1024 * 1024)"""

GiB = 1 << 30
"""Gibibyte (1024 * 1024 * 1024)"""

TiB = 1 << 40
"""Tebibyte (1024 * 1024 * 1024 * 1024)"""

PiB = 1 << 50
"""Pebibyte (1024 * 1024 * 1024 * 1024 * 1024)"""

SIZE_UNITS = {
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
    'PB': 1000**4,
}
"""Units of storage and memory measurements."""
SIZE_PATTERN = re.compile(
    r'^\s*\+?\s*(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTP]i?B?|B?)\s*$', flags=re.IGNORECASE
)
"""The regex pattern for human readable size."""


def bytes2human(b: Union[int, float, NaType]) -> str:  # pylint: disable=too-many-return-statements
    """Converts bytes to a human readable string."""

    if b == NA:
        return NA

    if not isinstance(b, int):
        try:
            b = round(float(b))
        except ValueError:
            return NA

    if b < KiB:
        return '{}B'.format(b)
    if b < MiB:
        return '{}KiB'.format(round(b / KiB))
    if b <= 20 * GiB:
        return '{}MiB'.format(round(b / MiB))
    if b < 100 * GiB:
        return '{:.2f}GiB'.format(round(b / GiB, 2))
    if b < 1000 * GiB:
        return '{:.1f}GiB'.format(round(b / GiB, 1))
    if b < 100 * TiB:
        return '{:.2f}TiB'.format(round(b / TiB, 2))
    if b < 1000 * TiB:
        return '{:.1f}TiB'.format(round(b / TiB, 1))
    if b < 100 * PiB:
        return '{:.2f}PiB'.format(round(b / PiB, 2))
    return '{:.1f}PiB'.format(round(b / PiB, 1))


def human2bytes(s: Union[int, str]) -> int:
    """Converts a human readable size string (*case insensitive*) to bytes.

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
        raise ValueError('Cannot convert {!r} to bytes.'.format(s))

    match = SIZE_PATTERN.match(s)
    if match is None:
        raise ValueError('Cannot convert {!r} to bytes.'.format(s))
    size, unit = match.groups()
    unit = unit.upper().replace('I', 'i').replace('B', '') + 'B'
    return int(float(size) * SIZE_UNITS[unit])


def timedelta2human(dt: Union[int, float, datetime.timedelta, NaType]) -> str:
    """Converts a number in seconds or a :class:`datetime.timedelta` instance to a human readable string."""

    if isinstance(dt, (int, float)):
        dt = datetime.timedelta(seconds=dt)

    if not isinstance(dt, datetime.timedelta):
        return NA

    if dt.days >= 4:
        return '{:.1f} days'.format(dt.days + dt.seconds / 86400)

    hours, seconds = divmod(86400 * dt.days + dt.seconds, 3600)
    if hours > 0:
        return '{:d}:{:02d}:{:02d}'.format(hours, *divmod(seconds, 60))
    return '{:d}:{:02d}'.format(*divmod(seconds, 60))


def utilization2string(utilization: Union[int, float, NaType]) -> str:
    """Converts a utilization rate to string."""

    if utilization != NA:
        if isinstance(utilization, int):
            return '{}%'.format(utilization)
        if isinstance(utilization, float):
            return '{:.1f}%'.format(utilization)
    return NA


def boolify(string: str, default: Any = None) -> bool:
    """Converts the given value, usually a string, to boolean."""

    if string.lower() in ('true', 'yes', 'on', 'enabled', '1'):
        return True
    if string.lower() in ('false', 'no', 'off', 'disabled', '0'):
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

    def __init__(self, real: Any, **items) -> None:
        self.real = real
        self.timestamp = time.time()
        for key, value in items.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        keys = set(self.__dict__.keys()).difference({'real', 'timestamp'})
        keys = ['real', *sorted(keys)]
        keyvals = []
        for key in keys:
            value = getattr(self, key)
            keyval = '{}={!r}'.format(key, value)
            if isinstance(value, Snapshot):
                keyval = keyval.replace('\n', '\n    ')  # extra indentation for nested snapshots
            keyvals.append(keyval)
        return '{}{}(\n    {}\n)'.format(
            self.real.__class__.__name__, self.__class__.__name__, ',\n    '.join(keyvals)
        )

    __repr__ = __str__

    def __hash__(self) -> int:
        return hash((self.real, self.timestamp))

    def __getattr__(self, name: str) -> Any:
        """Gets a member from the instance.
        If the attribute is not defined, fetches from the original object and makes a function call.
        """

        try:
            return super().__getattr__(name)
        except AttributeError:
            attribute = getattr(self.real, name)
            if callable(attribute):
                attribute = attribute()

            setattr(self, name, attribute)
            return attribute

    def __getitem__(self, name: str) -> Any:
        """Supports ``snapshot['name']`` syntax."""

        try:
            return getattr(self, name)
        except AttributeError as ex:
            raise KeyError(name) from ex

    def __setitem__(self, name: str, value: Any) -> None:
        """Supports ``snapshot['name'] = value`` syntax."""

        setattr(self, name, value)

    def __iter__(self) -> Iterable[str]:
        """Supports ``for name in snapshot`` syntax and ``*`` tuple unpack ``[*snapshot]`` syntax."""

        def gen() -> str:
            for name in self.__dict__:
                if name not in ('real', 'timestamp'):
                    yield name

        return gen()

    def keys(self) -> Iterable[str]:
        """Supports `**`` dictionary unpack ``{**snapshot}`` / ``dict(**snapshot)`` syntax and
        ``dict(snapshot)`` dictionary conversion.
        """

        return iter(self)


# Modified from psutil (https://github.com/giampaolo/psutil)
def memoize_when_activated(method: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """A memoize decorator which is disabled by default. It can be activated and
    deactivated on request. For efficiency reasons it can be used only against
    class methods accepting no arguments.
    """

    @functools.wraps(method)
    def wrapped(self):
        try:
            # case 1: we previously entered oneshot() ctx
            ret = self._cache[method]  # pylint: disable=protected-access
        except AttributeError:
            # case 2: we never entered oneshot() ctx
            return method(self)
        except KeyError:
            # case 3: we entered oneshot() ctx but there's no cache
            # for this entry yet
            ret = method(self)
            try:
                self._cache[method] = ret  # pylint: disable=protected-access
            except AttributeError:
                # multi-threading race condition, see:
                # https://github.com/giampaolo/psutil/issues/1948
                pass
        return ret

    def cache_activate(self):
        """Activate cache. Expects a Process instance. Cache will be stored as
        a "_cache" instance attribute.
        """

        if not hasattr(self, '_cache'):
            setattr(self, '_cache', {})

    def cache_deactivate(self):
        """Deactivate and clear cache."""
        try:
            del self._cache  # pylint: disable=protected-access
        except AttributeError:
            pass

    wrapped.cache_activate = cache_activate
    wrapped.cache_deactivate = cache_deactivate
    return wrapped
