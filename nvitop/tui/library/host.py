# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

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


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing_extensions import ParamSpec  # Python 3.10+

    _P = ParamSpec('_P')
    _T = TypeVar('_T')


def ignore_error(*, fallback: Any) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Ignore errors in the function."""

    def wrapper(func: Callable[_P, _T]) -> Callable[_P, _T]:
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            try:
                return func(*args, **kwargs)
            except Exception:  # noqa: BLE001 # pylint: disable=broad-exception-caught
                return fallback

        return wrapped

    return wrapper


class VirtualMemory(NamedTuple):  # pylint: disable=missing-class-docstring
    total: int = NA  # type: ignore[assignment]
    available: int = NA  # type: ignore[assignment]
    percent: int = NA  # type: ignore[assignment]
    used: int = NA  # type: ignore[assignment]
    free: int = NA  # type: ignore[assignment]


@ignore_error(fallback=VirtualMemory())
def virtual_memory() -> VirtualMemory:
    vm = host.virtual_memory()
    return VirtualMemory(
        total=vm.total,
        available=vm.available,
        percent=vm.percent,
        used=vm.used,
        free=vm.free,
    )


class SwapMemory(NamedTuple):  # pylint: disable=missing-class-docstring
    total: int = NA  # type: ignore[assignment]
    used: int = NA  # type: ignore[assignment]
    free: int = NA  # type: ignore[assignment]
    percent: float = NA  # type: ignore[assignment]


@ignore_error(fallback=SwapMemory())
def swap_memory() -> SwapMemory:
    sm = host.swap_memory()
    return SwapMemory(
        total=sm.total,
        used=sm.used,
        free=sm.free,
        percent=sm.percent,
    )


@ignore_error(fallback=(NA, NA, NA))
def load_average() -> tuple[float, float, float]:
    la = host.load_average()
    if la is None:
        return (NA, NA, NA)  # type: ignore[unreachable]
    return la


if TYPE_CHECKING:
    from nvitop.api.host import cpu_percent, getuser, hostname, reverse_ppid_map, uptime
else:
    cpu_percent = ignore_error(fallback=NA)(host.cpu_percent)
    getuser = ignore_error(fallback=NA)(host.getuser)
    hostname = ignore_error(fallback=NA)(host.hostname)
    uptime = ignore_error(fallback=NA)(host.uptime)
    reverse_ppid_map = ignore_error(fallback=MappingProxyType({}))(host.reverse_ppid_map)
