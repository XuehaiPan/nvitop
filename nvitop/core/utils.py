# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name


def bytes2human(x):
    if x is None or x == 'N/A':
        return 'N/A'

    if not isinstance(x, int):
        x = int(x)

    if x < (1 << 10):
        return '{}B'.format(x)
    if x < (1 << 20):
        return '{}KiB'.format(x >> 10)
    return '{}MiB'.format(x >> 20)


def timedelta2human(dt):
    if dt is None or dt == 'N/A':
        return 'N/A'

    if dt.days >= 4:
        return '{:.1f} days'.format(dt.days + dt.seconds / 86400)

    hours, seconds = divmod(86400 * dt.days + dt.seconds, 3600)
    if hours > 0:
        return '{:d}:{:02d}:{:02d}'.format(hours, *divmod(seconds, 60))
    return '{:d}:{:02d}'.format(*divmod(seconds, 60))


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
