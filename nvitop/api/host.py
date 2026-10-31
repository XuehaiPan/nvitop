# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2026 Xuehai Pan. All Rights Reserved.
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
"""Shortcuts for package ``psutil``.

``psutil`` is a cross-platform library for retrieving information on running processes and system
utilization (CPU, memory, disks, network, sensors) in Python.
"""

from __future__ import annotations

import os as _os
import threading as _threading

import psutil as _psutil
from psutil import *  # noqa: F403 # pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin
from psutil import (  # noqa: F401
    LINUX,
    MACOS,
    POSIX,
    WINDOWS,
    AccessDenied,
    Error,
    NoSuchProcess,
    Process,
    ZombieProcess,
    boot_time,
    cpu_percent,
    pids,
    swap_memory,
    virtual_memory,
)
from psutil import Error as PsutilError  # pylint: disable=reimported


__all__ = [
    'WINDOWS_SUBSYSTEM_FOR_LINUX',
    'WSL',
    'PsutilError',
    'getuser',
    'hostname',
    'load_average',
    'memory_percent',
    'ppid_map',
    'reverse_ppid_map',
    'swap_percent',
    'uptime',
]
__all__ += [name for name in _psutil.__all__ if not name.startswith('_') and name != 'Error']


del Error  # renamed to PsutilError


if _psutil.WINDOWS:
    # ``psutil.cpu_percent()`` counts the Windows ``interrupt`` and ``dpc`` times as busy time.
    # Both fields are summed from per-CPU counters that only cover the calling thread's
    # processor group; on machines with more than 64 logical CPUs the counters of the other
    # groups are uninitialized, so the sum can hold garbage and produce spurious 100% readings
    # (psutil#2972, psutil#3061). ``psutil.cpu_times().system`` is the kernel time minus idle
    # and already includes interrupt/dpc time, so ``user + system`` over ``user + system +
    # idle`` is both sufficient and immune to those values.
    _cpu_times = _threading.local()

    def cpu_percent(  # pylint: disable=function-redefined
        interval: float | None = None,
        percpu: bool = False,
    ) -> float | list[float]:
        """Return the system CPU utilization as a percentage.

        On Windows, compute the utilization from the user and system times instead of
        :func:`psutil.cpu_percent`, which adds the interrupt and dpc times to the busy time.
        As with ``psutil.cpu_percent``, the first call in each thread returns ``0.0`` because
        there is no previous sample to compare against.
        """
        if interval is not None or percpu:
            return _psutil.cpu_percent(interval=interval, percpu=percpu)

        times = _psutil.cpu_times()
        prev = getattr(_cpu_times, 'times', None)
        _cpu_times.times = times
        if prev is None:
            return 0.0
        busy = (times.user - prev.user) + (times.system - prev.system)
        total = busy + (times.idle - prev.idle)
        if busy < 0.0 or total <= 0.0:
            return 0.0
        return 100.0 * busy / total


def getuser() -> str:
    """Get the current username from the environment or password database."""
    import getpass  # pylint: disable=import-outside-toplevel

    try:
        return getpass.getuser()
    except (ModuleNotFoundError, OSError):
        return _os.getlogin()


def hostname() -> str:
    """Get the hostname of the machine."""
    import platform  # pylint: disable=import-outside-toplevel

    return platform.node()


if hasattr(_psutil, 'getloadavg'):

    def load_average() -> tuple[float, float, float]:
        """Get the system load average."""
        return _psutil.getloadavg()

else:

    def load_average() -> None:  # type: ignore[misc]
        """Get the system load average."""
        return


def uptime() -> float:
    """Get the system uptime."""
    import time as _time  # pylint: disable=import-outside-toplevel

    return _time.time() - boot_time()


def memory_percent() -> float:
    """The percentage usage of virtual memory, calculated as ``(total - available) / total * 100``."""
    return virtual_memory().percent


def swap_percent() -> float:
    """The percentage usage of virtual memory, calculated as ``used / total * 100``."""
    return swap_memory().percent


def ppid_map() -> dict[int, int]:
    """Obtain a ``{pid: ppid, ...}`` dict for all running processes in one shot."""
    ret = {}
    for pid in pids():
        try:
            ret[pid] = Process(pid).ppid()
        except (NoSuchProcess, ZombieProcess):  # noqa: PERF203
            pass
    return ret


try:
    from psutil import _ppid_map as ppid_map  # type: ignore[no-redef] # noqa: F811,RUF100
except ImportError:
    pass


def reverse_ppid_map() -> dict[int, list[int]]:
    """Obtain a ``{ppid: [pid, ...], ...}`` dict for all running processes in one shot."""
    from collections import defaultdict  # pylint: disable=import-outside-toplevel

    ret = defaultdict(list)
    for pid, ppid in ppid_map().items():
        ret[ppid].append(pid)

    return ret


# pylint: disable=invalid-name
if LINUX:
    WSL = _os.getenv('WSL_DISTRO_NAME', default=None)
    if WSL is not None and WSL == '':
        WSL = 'WSL'
else:
    WSL = None
WINDOWS_SUBSYSTEM_FOR_LINUX = WSL
"""The Linux distribution name of the Windows Subsystem for Linux."""
# pylint: enable=invalid-name
