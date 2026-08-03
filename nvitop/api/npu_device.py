# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2026 Xuehai Pan. All Rights Reserved.
# Copyright 2026 YSP. All Rights Reserved.
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
"""The live classes for Ascend NPU devices.

This module mirrors the public API of :mod:`nvitop.api.device` for Huawei Ascend
NPUs.  All metrics are queried through :mod:`nvitop.api.libnpu`, which wraps the
``npu-smi`` command-line tool, so no CANN / ACL Python packages are required.

The core classes are :class:`NpuDevice` and :class:`NpuProcess`.

Examples:
    >>> from nvitop.api.npu_device import NpuDevice
    >>> NpuDevice.count()                       # number of Ascend NPUs in the system
    8
    >>> NpuDevice.driver_version()              # version of the installed NPU driver
    '25.5.1'
    >>> NpuDevice.all()                         # all devices in the system
    [NpuDevice(index=0, name='Ascend 910B3', ...), ...]
    >>> npu0 = NpuDevice(0)                     # -> NpuDevice
    >>> npu0.name()                             # the chip name of the device
    'Ascend 910B3'
    >>> npu0.memory_used_human()                # memory used in human-readable format
    '57.80GiB'
    >>> npu0.gpu_utilization()                  # the AICore utilization in percentage
    0
"""

# pylint: disable=too-many-lines,too-many-instance-attributes

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, ClassVar, NamedTuple

from nvitop.api import libnpu
from nvitop.api.libnpu import NpuError, NpuQueryError, NpuSmiNotFound  # noqa: F401
from nvitop.api.utils import NA, NaType, bytes2human, utilization2string

try:  # psutil is optional: fall back to /proc parsing on Linux when unavailable
    import psutil as _psutil
except ImportError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]


__all__ = [
    'NpuError',
    'NpuQueryError',
    'NpuSmiNotFound',
    'MemoryInfo',
    'ClockInfos',
    'UtilizationRates',
    'NpuDevice',
    'NpuProcess',
]


class MemoryInfo(NamedTuple):  # in bytes
    """The memory usage information of an NPU device."""

    total: int  # total memory in bytes
    used: int  # used memory in bytes
    free: int  # free memory in bytes


class ClockInfos(NamedTuple):  # in MHz
    """The clock speed information of an NPU device."""

    aicore: int | NaType  # the current AICore clock speed in MHz
    aicore_max: int | NaType  # the maximum AICore clock speed in MHz
    hbm: int | NaType  # the HBM clock speed in MHz


class UtilizationRates(NamedTuple):  # in percentage
    """The utilization rates of an NPU device."""

    gpu: int | NaType  # the AICore utilization in percentage
    memory: int | NaType  # the HBM utilization in percentage


# pylint: disable=invalid-name
_LOADING_INTENSITY = ('light', 'moderate', 'heavy')
_INTENSITY_COLORS = {'light': 'green', 'moderate': 'yellow', 'heavy': 'red'}


def _loading_intensity(value: int | float | NaType, thresholds: tuple[int, int]) -> str:
    """Map a numeric value to a loading intensity string using the given thresholds."""
    if isinstance(value, NaType):
        return 'light'
    if value >= thresholds[1]:
        return 'heavy'
    if value >= thresholds[0]:
        return 'moderate'
    return 'light'


def _intensity_color(intensity: str) -> str:
    """Map a loading intensity string to its color name."""
    return _INTENSITY_COLORS.get(intensity, 'green')


# pylint: enable=invalid-name


class NpuDevice:  # pylint: disable=too-many-public-methods
    """The live class for Ascend NPU devices.

    Args:
        index (int): The NPU card id (as shown by ``npu-smi info -m``).

    Raises:
        NpuSmiNotFound: If the ``npu-smi`` executable cannot be found.
        NpuQueryError: If the device is not found for the given index.
    """

    GPU_UTILIZATION_THRESHOLDS: ClassVar[tuple[int, int]] = (10, 75)
    MEMORY_UTILIZATION_THRESHOLDS: ClassVar[tuple[int, int]] = (10, 80)
    TEMPERATURE_THRESHOLDS: ClassVar[tuple[int, int]] = (60, 80)
    POWER_USAGE_THRESHOLDS: ClassVar[tuple[int, int]] = (150000, 300000)  # in mW
    FAN_SPEED_THRESHOLDS: ClassVar[tuple[int, int]] = (50, 80)
    HEALTH_THRESHOLDS: ClassVar[tuple[int, int]] = (1, 2)

    #: The color of the device loading intensity
    _LOADING_COLOR: ClassVar[OrderedDict[str, str]] = OrderedDict(
        light='green',
        moderate='yellow',
        heavy='red',
    )

    def __init__(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f'invalid NPU index: {index!r}.')
        try:
            count = libnpu.npu_device_count()
        except NpuError as ex:
            raise NpuQueryError(f'Failed to query NPU device count: {ex}') from ex
        if index >= count:
            raise NpuQueryError(f'NPU device {index} does not exist (only {count} devices found).')

        self._index = index
        self._chip_name: str | NaType = NA

    # ------------------------------------------------------------------ #
    # Class-level helpers                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the ``npu-smi`` executable is available on this host."""
        return libnpu.npu_smi_path() is not None

    @classmethod
    def driver_version(cls) -> str | NaType:
        """Get the version of the installed Ascend NPU driver (e.g. ``25.5.1``)."""
        return libnpu.npu_driver_version()

    @classmethod
    def count(cls) -> int:
        """Get the number of Ascend NPU devices in the system."""
        return libnpu.npu_device_count()

    @classmethod
    def all(cls) -> list[NpuDevice]:
        """Get all physical devices in the system."""
        return [cls(index) for index in range(cls.count())]

    @classmethod
    def from_index(cls, index: int) -> NpuDevice:
        """Get the device with the given index."""
        return cls(index)

    @classmethod
    def from_indices(cls, indices: Any) -> list[NpuDevice]:
        """Get a list of devices from the given indices."""
        return [cls(index) for index in indices]

    # ------------------------------------------------------------------ #
    # Raw query helpers                                                   #
    # ------------------------------------------------------------------ #

    def _query_kv(self, query_type: str) -> dict[str, str]:
        """Query the detailed key-value information of this device (cached)."""
        return libnpu.npu_query_kv(self._index, query_type)

    def _global_info(self) -> dict[str, Any]:
        """Query the global overview and locate the entry of this device.

        The global query is cached by :mod:`nvitop.api.libnpu` with a short TTL
        (:data:`libnpu._GLOBAL_CACHE_TTL`), so monitor mode sees fresh values
        while all devices share a single ``npu-smi`` invocation.
        """
        data = libnpu.npu_query_global()
        for device in data['devices']:
            if device['index'] == self._index:
                return device
        raise NpuQueryError(f'NPU device {self._index} not found in the global overview.')

    # ------------------------------------------------------------------ #
    # Identifiers                                                         #
    # ------------------------------------------------------------------ #

    @property
    def index(self) -> int:
        """The NPU card id (as shown by ``npu-smi info -m``)."""
        return self._index

    @property
    def physical_index(self) -> int:
        """The physical NPU card id, same as :attr:`index`."""
        return self._index

    def name(self) -> str | NaType:
        """Get the chip name of the device (e.g. ``Ascend 910B3``)."""
        if self._chip_name is NA:
            self._chip_name = libnpu.npu_device_chip_name(self._index)
        return self._chip_name

    def uuid(self) -> str | NaType:
        """Get the UUID of the device.

        ``npu-smi`` does not expose a stable UUID; the PCIe bus id prefixed with
        ``NPU-`` is used as a surrogate identifier.
        """
        bus_id = self.bus_id()
        return f'NPU-{bus_id}' if not isinstance(bus_id, NaType) else NA

    def bus_id(self) -> str | NaType:
        """Get the PCIe bus id of the device (e.g. ``0000:C1:00.0``)."""
        try:
            return self._global_info()['bus_id']
        except NpuError:
            return NA

    def serial(self) -> str | NaType:
        """Get the serial number of the device from the board information."""
        try:
            board = self._query_kv('board')
        except NpuError:
            return NA
        value = board.get('Serial Number', 'NA')
        if value in ('NA', '-'):
            return NA
        return value

    def health(self) -> str | NaType:
        """Get the health status of the device (``OK`` / ``Warning`` / ``Fault``)."""
        try:
            return self._global_info()['health']
        except NpuError:
            return NA

    # ------------------------------------------------------------------ #
    # Memory                                                              #
    # ------------------------------------------------------------------ #

    def memory_info(self) -> MemoryInfo:  # in bytes
        """Get the HBM memory usage information of the device (in bytes)."""
        info = self._global_info()
        total = int(info['memory_total']) * 1024 * 1024  # MB -> bytes
        used = int(info['memory_used']) * 1024 * 1024  # MB -> bytes
        return MemoryInfo(total=total, used=used, free=total - used)

    def memory_total(self) -> int | NaType:  # in bytes
        """Get the total HBM memory of the device (in bytes)."""
        try:
            return self.memory_info().total
        except NpuError:
            return NA

    def memory_used(self) -> int | NaType:  # in bytes
        """Get the used HBM memory of the device (in bytes)."""
        try:
            return self.memory_info().used
        except NpuError:
            return NA

    def memory_free(self) -> int | NaType:  # in bytes
        """Get the free HBM memory of the device (in bytes)."""
        try:
            return self.memory_info().free
        except NpuError:
            return NA

    def memory_total_human(self) -> str | NaType:  # in human-readable
        """Get the total HBM memory of the device in human-readable format."""
        return bytes2human(self.memory_total())

    def memory_used_human(self) -> str | NaType:  # in human-readable
        """Get the used HBM memory of the device in human-readable format."""
        return bytes2human(self.memory_used())

    def memory_free_human(self) -> str | NaType:  # in human-readable
        """Get the free HBM memory of the device in human-readable format."""
        return bytes2human(self.memory_free())

    def memory_percent(self) -> float | NaType:  # in percentage
        """Get the HBM memory usage percentage of the device."""
        total = self.memory_total()
        used = self.memory_used()
        if isinstance(total, NaType) or isinstance(used, NaType) or total <= 0:
            return NA
        return used / total * 100.0

    def memory_usage(self) -> str:  # string of used memory over total memory (in human-readable)
        """Get the memory usage string of the device (e.g. ``57.80GiB / 64.00GiB``)."""
        try:
            total = self.memory_info().total
            used = self.memory_info().used
            return '{} / {}'.format(bytes2human(used), bytes2human(total))
        except NpuError:
            return 'NA / NA'

    # ------------------------------------------------------------------ #
    # Utilization                                                         #
    # ------------------------------------------------------------------ #

    def utilization_rates(self) -> UtilizationRates:  # in percentage
        """Get the utilization rates of the device."""
        return UtilizationRates(gpu=self.gpu_utilization(), memory=self.memory_utilization())

    def gpu_utilization(self) -> int | NaType:  # in percentage
        """Get the AICore utilization of the device in percentage."""
        try:
            return self._global_info()['aicore']
        except NpuError:
            return NA

    def memory_utilization(self) -> int | NaType:  # in percentage
        """Get the HBM utilization of the device in percentage.

        The value is derived from the used / total HBM memory of the global
        overview (``npu-smi info``), which matches the ``HBM Usage Rate(%)``
        reported by ``npu-smi info -t usages`` without an extra invocation.
        """
        try:
            info = self._global_info()
            total = int(info['memory_total'])
            used = int(info['memory_used'])
        except (NpuError, TypeError, ValueError):
            return NA
        if total <= 0:
            return NA
        return int(used * 100 // total)

    def utilization_strings(self) -> tuple[str, str]:
        """Get the utilization strings of the device (AICore and HBM)."""
        gpu = utilization2string(self.gpu_utilization())
        memory = utilization2string(self.memory_utilization())
        return gpu, memory

    # ------------------------------------------------------------------ #
    # Clocks                                                              #
    # ------------------------------------------------------------------ #

    def clock_infos(self) -> ClockInfos:  # in MHz
        """Get the clock speed information of the device."""
        aicore: int | NaType = NA
        aicore_max: int | NaType = NA
        hbm: int | NaType = NA
        try:
            common = self._query_kv('common')
            aicore = _parse_int_field(common.get('Aicore curFreq(MHZ)'))
            aicore_max = _parse_int_field(common.get('Aicore Freq(MHZ)'))
        except NpuError:
            pass
        try:
            memory = self._query_kv('memory')
            hbm = _parse_int_field(memory.get('HBM Clock Speed(MHz)'))
        except NpuError:
            pass
        return ClockInfos(aicore=aicore, aicore_max=aicore_max, hbm=hbm)

    def aicore_clock(self) -> int | NaType:  # in MHz
        """Get the current AICore clock speed of the device in MHz."""
        return self.clock_infos().aicore

    def memory_clock(self) -> int | NaType:  # in MHz
        """Get the HBM clock speed of the device in MHz."""
        return self.clock_infos().hbm

    # ------------------------------------------------------------------ #
    # Temperature / Power / Fan                                           #
    # ------------------------------------------------------------------ #

    def temperature(self) -> int | NaType:  # in Celsius
        """Get the temperature of the device in Celsius."""
        try:
            return self._global_info()['temperature']
        except NpuError:
            return NA

    def power_usage(self) -> int | NaType:  # in milliwatts (mW)
        """Get the real-time power usage of the device in milliwatts."""
        try:
            power = self._global_info()['power']
        except NpuError:
            return NA
        if isinstance(power, NaType):
            return NA
        return int(power * 1000)  # W -> mW

    def power_limit(self) -> int | NaType:  # in milliwatts (mW)
        """Get the power limit of the device.

        ``npu-smi`` does not expose the power limit; :data:`NA` is returned.
        """
        return NA

    def power_status(self) -> str:  # string of power usage in watts (W)
        """Get the power usage string of the device (e.g. ``99.6W``)."""
        usage = self.power_usage()
        if isinstance(usage, NaType):
            return 'N/A'
        return '{:.1f}W'.format(usage / 1000)

    def fan_speed(self) -> int | NaType:  # in percentage
        """Get the fan speed of the device.

        Ascend NPUs are passively cooled; :data:`NA` is returned.
        """
        return NA

    # ------------------------------------------------------------------ #
    # Loading intensities and colors                                      #
    # ------------------------------------------------------------------ #

    def gpu_loading_intensity(self) -> str:
        """Get the loading intensity of the AICore utilization."""
        return _loading_intensity(self.gpu_utilization(), self.GPU_UTILIZATION_THRESHOLDS)

    def memory_loading_intensity(self) -> str:
        """Get the loading intensity of the memory utilization."""
        return _loading_intensity(self.memory_utilization(), self.MEMORY_UTILIZATION_THRESHOLDS)

    def loading_intensity(self) -> str:
        """Get the overall loading intensity of the device."""
        intensities = [self.gpu_loading_intensity(), self.memory_loading_intensity()]
        if 'heavy' in intensities:
            return 'heavy'
        if 'moderate' in intensities:
            return 'moderate'
        return 'light'

    def utilization_color(self, utilization: int | float | NaType) -> str:
        """Get the color of the given utilization value."""
        return _intensity_color(
            _loading_intensity(utilization, self.GPU_UTILIZATION_THRESHOLDS),
        )

    def memory_color(self, memory_percent: int | float | NaType) -> str:
        """Get the color of the given memory percentage."""
        return _intensity_color(
            _loading_intensity(memory_percent, self.MEMORY_UTILIZATION_THRESHOLDS),
        )

    def temperature_color(self, temperature: int | float | NaType) -> str:
        """Get the color of the given temperature value."""
        return _intensity_color(_loading_intensity(temperature, self.TEMPERATURE_THRESHOLDS))

    def power_color(self, power_usage: int | float | NaType) -> str:
        """Get the color of the given power usage (in milliwatts)."""
        return _intensity_color(_loading_intensity(power_usage, self.POWER_USAGE_THRESHOLDS))

    def fan_speed_color(self, fan_speed: int | float | NaType) -> str:
        """Get the color of the given fan speed value."""
        return _intensity_color(_loading_intensity(fan_speed, self.FAN_SPEED_THRESHOLDS))

    def health_color(self, health: str | NaType) -> str:
        """Get the color of the given health status."""
        if isinstance(health, NaType):
            return 'yellow'
        if health == 'OK':
            return 'green'
        if health in ('Warning', 'Degraded'):
            return 'yellow'
        return 'red'

    # ------------------------------------------------------------------ #
    # Snapshot                                                            #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        """Take a one-time snapshot of the device as a plain dict."""
        return {
            'index': self.index,
            'name': self.name(),
            'bus_id': self.bus_id(),
            'serial': self.serial(),
            'health': self.health(),
            'memory_total': self.memory_total(),
            'memory_used': self.memory_used(),
            'memory_free': self.memory_free(),
            'memory_total_human': self.memory_total_human(),
            'memory_used_human': self.memory_used_human(),
            'memory_free_human': self.memory_free_human(),
            'memory_percent': self.memory_percent(),
            'memory_usage': self.memory_usage(),
            'gpu_utilization': self.gpu_utilization(),
            'memory_utilization': self.memory_utilization(),
            'aicore_clock': self.aicore_clock(),
            'memory_clock': self.memory_clock(),
            'temperature': self.temperature(),
            'power_usage': self.power_usage(),
            'power_status': self.power_status(),
            'fan_speed': self.fan_speed(),
            'loading_intensity': self.loading_intensity(),
        }

    def as_snapshot(self) -> dict[str, Any]:
        """Alias of :meth:`snapshot`."""
        return self.snapshot()

    # ------------------------------------------------------------------ #
    # Dunder methods                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        name = self.name()
        bus_id = self.bus_id()
        return 'NpuDevice(index={}, name={!r}, bus_id={!r})'.format(
            self.index,
            name if not isinstance(name, NaType) else None,
            bus_id if not isinstance(bus_id, NaType) else None,
        )

    def __str__(self) -> str:
        name = self.name()
        if isinstance(name, NaType):
            return f'NpuDevice(index={self.index})'
        return f'NpuDevice(index={self.index}, name={name})'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NpuDevice) and self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)


def _parse_int_field(value: str | None) -> int | NaType:
    """Parse an integer field from a raw ``npu-smi`` value."""
    if value is None or value in ('NA', '-'):
        return NA
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return NA


class NpuProcess:
    """The live class for processes running on Ascend NPU devices.

    Args:
        pid (int): The process id.
        device (Optional[NpuDevice]): The NPU device the process is running on.
        used_memory (Optional[int]): The NPU memory used by the process (in bytes).
        name (Optional[str]): The process name as reported by ``npu-smi``.
    """

    def __init__(
        self,
        pid: int,
        device: NpuDevice | None = None,
        used_memory: int | None = None,
        name: str | None = None,
    ) -> None:
        self.pid = int(pid)
        self.device = device
        self.used_memory = int(used_memory) if used_memory is not None else None
        self._name = name
        self._psutil_process = _psutil.Process(self.pid) if _psutil is not None else None

    #: The context type of NPU processes (compute).
    type: ClassVar[str] = 'C'

    def name(self) -> str | NaType:
        """Get the process name."""
        if self._name is not None:
            return self._name
        try:
            if self._psutil_process is not None:
                return self._psutil_process.name()
        except Exception:  # pylint: disable=broad-except
            pass
        return self._proc_name()
    def _proc_name(self) -> str | NaType:
        """Fallback: read the process name from ``/proc/<pid>/comm``."""
        try:
            with open(f'/proc/{self.pid}/comm', 'r', encoding='utf-8') as file:  # noqa: PTH123
                return file.read().strip()
        except OSError:
            return NA

    def username(self) -> str | NaType:
        """Get the username of the process owner."""
        try:
            if self._psutil_process is not None:
                return self._psutil_process.username()
        except Exception:  # pylint: disable=broad-except
            pass
        # Fallback: parse the UID from /proc/<pid>/status and map it via pwd
        try:
            with open(f'/proc/{self.pid}/status', 'r', encoding='utf-8') as file:  # noqa: PTH123
                for line in file:
                    if line.startswith('Uid:'):
                        uid = int(line.split()[1])
                        import pwd  # pylint: disable=import-outside-toplevel

                        try:
                            return pwd.getpwuid(uid).pw_name
                        except KeyError:
                            return str(uid)
        except (OSError, ValueError):
            return NA
        return NA

    def cmdline(self) -> str | NaType:
        """Get the full command line of the process."""
        try:
            if self._psutil_process is not None:
                parts = self._psutil_process.cmdline()
                if parts:
                    return ' '.join(parts)
        except Exception:  # pylint: disable=broad-except
            pass
        # Fallback: read the command line from /proc/<pid>/cmdline
        try:
            with open(f'/proc/{self.pid}/cmdline', 'rb') as file:  # noqa: PTH123
                raw = file.read().replace(b'\x00', b' ').strip()
                return raw.decode('utf-8', errors='replace')
        except OSError:
            return NA

    def create_time(self) -> float | NaType:
        """Get the creation time of the process as a Unix timestamp."""
        try:
            if self._psutil_process is not None:
                return self._psutil_process.create_time()
        except Exception:  # pylint: disable=broad-except
            pass
        # Fallback: parse the start time from /proc/<pid>/stat (jiffies since boot)
        try:
            import time  # pylint: disable=import-outside-toplevel

            with open(f'/proc/{self.pid}/stat', 'r', encoding='utf-8') as file:  # noqa: PTH123
                fields = file.read().rsplit(')', 1)[1].split()
            start_jiffies = int(fields[19])
            clock_ticks = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
            with open('/proc/uptime', 'r', encoding='utf-8') as file:  # noqa: PTH123
                uptime_secs = float(file.read().split()[0])
            running_secs = start_jiffies / clock_ticks
            return time.time() - (uptime_secs - running_secs)
        except (OSError, ValueError, IndexError, KeyError):
            return NA

    def used_memory_human(self) -> str | NaType:
        """Get the NPU memory used by the process in human-readable format."""
        if self.used_memory is None:
            return NA
        return bytes2human(self.used_memory)

    def cpu_percent(self) -> float:
        """Get the CPU utilization of the process (0.0 if psutil is unavailable)."""
        try:
            if self._psutil_process is not None:
                return self._psutil_process.cpu_percent(interval=None)
        except Exception:  # pylint: disable=broad-except
            pass
        return 0.0

    @classmethod
    def from_pid(cls, pid: int, device: NpuDevice | None = None) -> NpuProcess:
        """Create an :class:`NpuProcess` instance from a pid."""
        return cls(pid, device=device)

    def __repr__(self) -> str:
        return 'NpuProcess(pid={}, name={!r})'.format(self.pid, self.name())

    def __str__(self) -> str:
        name = self.name()
        if isinstance(name, NaType):
            return f'NpuProcess(pid={self.pid})'
        return f'NpuProcess(pid={self.pid}, name={name})'
