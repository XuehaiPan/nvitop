# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Shortcuts for package ``psutil``.

psutil is a cross-platform library for retrieving information on running processes
and system utilization (CPU, memory, disks, network, sensors) in Python.
"""

import os as _os

import psutil as _psutil
from cachetools.func import ttl_cache as _ttl_cache
from psutil import *  # pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin


__all__ = _psutil.__all__.copy()
__all__.extend(
    (
        'load_average',
        'memory_percent',
        'swap_percent',
        'ppid_map',
        'reverse_ppid_map',
        'WSL',
        'WINDOWS_SUBSYSTEM_FOR_LINUX',
    )
)
__all__[__all__.index('Error')] = 'PsutilError'


PsutilError = Error  # make alias
del Error  # pylint: disable=undefined-variable


cpu_percent = _ttl_cache(ttl=0.25)(_psutil.cpu_percent)
virtual_memory = _ttl_cache(ttl=0.25)(_psutil.virtual_memory)
swap_memory = _ttl_cache(ttl=0.25)(_psutil.swap_memory)


try:
    load_average = _ttl_cache(ttl=2.0)(_psutil.getloadavg)
except AttributeError:

    def load_average():  # pylint: disable=missing-function-docstring
        return None


def memory_percent():
    """The percentage usage of virtual memory, calculated as (total - available) / total * 100."""

    return virtual_memory().percent


def swap_percent():
    """The percentage usage of virtual memory, calculated as used / total * 100."""

    return swap_memory().percent


ppid_map = _psutil._ppid_map  # pylint: disable=protected-access
"""Obtains a ``{pid: ppid, ...}`` dict for all running processes in one shot."""


def reverse_ppid_map():  # pylint: disable=function-redefined
    """Obtains a ``{ppid: [pid, ...], ...}`` dict for all running processes in one shot."""

    from collections import defaultdict  # pylint: disable=import-outside-toplevel

    tree = defaultdict(list)
    for pid, ppid in ppid_map().items():
        tree[ppid].append(pid)

    return tree


if LINUX:
    WSL = _os.getenv('WSL_DISTRO_NAME', default=None)
    if WSL is not None and WSL == '':
        WSL = 'WSL'
else:
    WSL = None
WINDOWS_SUBSYSTEM_FOR_LINUX = WSL
"""The Linux distribution name of the Windows Subsystem for Linux."""


del _os, _ttl_cache
