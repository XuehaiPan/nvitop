# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from nvitop.api import NA, host
from nvitop.api.host import AccessDenied, PsutilError


__all__ = [
    'AccessDenied',
    'PsutilError',
    'cpu_percent',
    'getuser',
    'hostname',
    'load_average',
    'reverse_ppid_map',
    'swap_memory',
    'uptime',
    'virtual_memory',
]


def ignore_error(*, fallback):
    """Ignore errors in the function."""

    def wrapper(func):
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:  # noqa: BLE001 # pylint: disable=broad-exception-caught
                return fallback

        return wrapped

    return wrapper


class VirtualMemory(NamedTuple):  # pylint: disable=missing-class-docstring
    total: int
    available: int
    percent: int
    used: int
    free: int


@ignore_error(fallback=VirtualMemory(NA, NA, NA, NA, NA))
def virtual_memory():
    vm = host.virtual_memory()
    return VirtualMemory(
        total=vm.total,
        available=vm.available,
        percent=vm.percent,
        used=vm.used,
        free=vm.free,
    )


class SwapMemory(NamedTuple):  # pylint: disable=missing-class-docstring
    total: int
    used: int
    free: int
    percent: float
    sin: int
    sout: int


@ignore_error(fallback=SwapMemory(NA, NA, NA, NA, NA, NA))
def swap_memory():
    sm = host.swap_memory()
    return SwapMemory(
        total=sm.total,
        used=sm.used,
        free=sm.free,
        percent=sm.percent,
        sin=sm.sin,
        sout=sm.sout,
    )


@ignore_error(fallback=(NA, NA, NA))
def load_average():
    la = host.load_average()
    if la is None:
        return (NA, NA, NA)
    return la


if TYPE_CHECKING:
    from nvitop.api.host import cpu_percent, getuser, hostname, reverse_ppid_map, uptime
else:
    cpu_percent = ignore_error(fallback=NA)(host.cpu_percent)
    getuser = ignore_error(fallback=NA)(host.getuser)
    hostname = ignore_error(fallback=NA)(host.hostname)
    uptime = ignore_error(fallback=NA)(host.uptime)
    reverse_ppid_map = ignore_error(fallback=MappingProxyType({}))(host.reverse_ppid_map)
