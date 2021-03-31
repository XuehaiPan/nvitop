# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import sys

import pynvml as nvml


try:
    if not sys.stdout.isatty():
        raise ImportError
    from termcolor import colored  # pylint: disable=unused-import
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):  # pylint: disable=unused-argument
        return text


def cut_string(s, maxlen, padstr='...', align='left'):
    assert align in ('left', 'right')

    if not isinstance(s, str):
        s = str(s)

    if len(s) <= maxlen:
        return s
    if align == 'left':
        return s[:maxlen - len(padstr)] + padstr
    else:
        return padstr + s[-(maxlen - len(padstr)):]


def bytes2human(x):
    if x == 'N/A':
        return x

    if not isinstance(x, int):
        x = int(x)

    if x < (1 << 10):
        return '{}B'.format(x)
    if x < (1 << 20):
        return '{}KiB'.format(x >> 10)
    else:
        return '{}MiB'.format(x >> 20)


def timedelta2human(dt):
    if dt == 'N/A':
        return dt

    if dt.days > 1:
        return '{} days'.format(dt.days)
    else:
        hours, seconds = divmod(86400 * dt.days + dt.seconds, 3600)
        if hours > 0:
            return '{:d}:{:02d}:{:02d}'.format(hours, *divmod(seconds, 60))
        else:
            return '{:d}:{:02d}'.format(*divmod(seconds, 60))


def nvml_query(func, *args, **kwargs):
    if isinstance(func, str):
        func = getattr(nvml, func)

    try:
        retval = func(*args, **kwargs)
    except nvml.NVMLError:
        return 'N/A'
    else:
        if isinstance(retval, bytes):
            retval = retval.decode('UTF-8')
        return retval


def nvml_check_return(retval, types=None):
    if types is None:
        return retval != 'N/A'
    return retval != 'N/A' and isinstance(retval, types)


class Snapshot(object):
    def __init__(self, real, **items):
        self.real = real
        for key, value in items.items():
            setattr(self, key, value)

    def __bool__(self):
        return bool(self.__dict__)

    def __str__(self):
        keys = set(self.__dict__.keys())
        keys.remove('real')
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join('{}={!r}'.format(key, getattr(self, key)) for key in ['real', *sorted(keys)])
        )

    __repr__ = __str__
