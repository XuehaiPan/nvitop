# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import datetime
import functools
import math
import time


__all__ = ['NA', 'NaType', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB',
           'bytes2human', 'timedelta2human', 'utilization2string', 'boolify',
           'Snapshot']


class NotApplicableType(str):
    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, 'N/A')
        return cls._instance

    def __bool__(self):
        return False

    def __int__(self):
        return 0

    def __float__(self):
        return math.nan

    def __lt__(self, x):
        if isinstance(x, (int, float)):
            return False
        return super().__lt__(x)

    def __le__(self, x):
        if isinstance(x, (int, float)):
            return False
        return super().__le__(x)

    def __gt__(self, x):
        if isinstance(x, (int, float)):
            return True
        return super().__gt__(x)

    def __ge__(self, x):
        if isinstance(x, (int, float)):
            return True
        return super().__ge__(x)

    def __format__(self, format_spec):
        try:
            return super().__format__(format_spec)
        except ValueError:
            return format(math.nan, format_spec)


# isinstance(NA, str)       -> True
# NA == 'N/A'               -> True
# NA is NotApplicableType() -> True (NotApplicableType is a singleton class)
NaType = NotApplicableType
NA = NotApplicable = NotApplicableType()


KiB = 1 << 10
MiB = 1 << 20
GiB = 1 << 30
TiB = 1 << 40
PiB = 1 << 50


def bytes2human(x):  # pylint: disable=too-many-return-statements
    if x is None or x == NA:
        return NA

    if not isinstance(x, int):
        try:
            x = round(float(x))
        except ValueError:
            return NA

    if x < KiB:
        return '{}B'.format(x)
    if x < MiB:
        return '{}KiB'.format(round(x / KiB))
    if x <= 20 * GiB:
        return '{}MiB'.format(round(x / MiB))
    if x < 100 * GiB:
        return '{:.2f}GiB'.format(round(x / GiB, 2))
    if x < 1000 * GiB:
        return '{:.1f}GiB'.format(round(x / GiB, 1))
    if x < 100 * TiB:
        return '{:.2f}TiB'.format(round(x / TiB, 2))
    if x < 1000 * TiB:
        return '{:.1f}TiB'.format(round(x / TiB, 1))
    if x < 100 * PiB:
        return '{:.2f}PiB'.format(round(x / PiB, 2))
    return '{:.1f}PiB'.format(round(x / PiB, 1))


def timedelta2human(dt):
    if not isinstance(dt, datetime.timedelta):
        return NA

    if dt.days >= 4:
        return '{:.1f} days'.format(dt.days + dt.seconds / 86400)

    hours, seconds = divmod(86400 * dt.days + dt.seconds, 3600)
    if hours > 0:
        return '{:d}:{:02d}:{:02d}'.format(hours, *divmod(seconds, 60))
    return '{:d}:{:02d}'.format(*divmod(seconds, 60))


def utilization2string(utilization):
    if utilization != NA:
        if isinstance(utilization, int):
            return '{}%'.format(utilization)
        if isinstance(utilization, float):
            return '{:.1f}%'.format(utilization)
    return NA


def boolify(string, default=None):
    if string.lower() in ('true', 'yes', 'on', 'enabled', '1'):
        return True
    if string.lower() in ('false', 'no', 'off', 'disabled', '0'):
        return False
    if default is not None:
        return bool(default)
    return bool(string)


class Snapshot:
    def __init__(self, real, **items):
        self.real = real
        self.timestamp = time.monotonic()
        for key, value in items.items():
            setattr(self, key, value)

    def __str__(self):
        keys = set(self.__dict__.keys())
        keys.remove('real')
        keys.remove('timestamp')
        return '{}{}(\n    {}\n)'.format(
            self.real.__class__.__name__, self.__class__.__name__,
            ', \n    '.join('{}={!r}'.format(key, getattr(self, key)) for key in ['real', *sorted(keys)])
        )

    __repr__ = __str__

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            attribute = getattr(self.real, name)
            if callable(attribute):
                attribute = attribute()

            setattr(self, name, attribute)
            return attribute

    def __getitem__(self, name):
        try:
            return self.__getattr__(name)
        except AttributeError as e:
            raise KeyError from e

    def __setitem__(self, name, value):
        self.__setattr__(name, value)


# Modified from psutil (https://github.com/giampaolo/psutil)
def memoize_when_activated(func):
    """A memoize decorator which is disabled by default. It can be
    activated and deactivated on request.
    For efficiency reasons it can be used only against class methods
    accepting no arguments.
    """

    @functools.wraps(func)
    def wrapped(self):
        try:
            # case 1: we previously entered oneshot() ctx
            ret = self._cache[func]  # pylint: disable=protected-access
        except AttributeError:
            # case 2: we never entered oneshot() ctx
            return func(self)
        except KeyError:
            # case 3: we entered oneshot() ctx but there's no cache
            # for this entry yet
            ret = func(self)
            try:
                self._cache[func] = ret  # pylint: disable=protected-access
            except AttributeError:
                # multi-threading race condition, see:
                # https://github.com/giampaolo/psutil/issues/1948
                pass
        return ret

    def cache_activate(self):
        """Activate cache. Expects a Process instance. Cache will be
        stored as a "_cache" instance attribute."""

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
