# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
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
"""Utilities for querying MetaX GPUs through ``mx-smi``."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, replace

from nvitop.api.utils import MiB, NA, NaType


__all__ = [
    'DeviceInfo',
    'MxSmiError',
    'MxSmiDeviceNotFound',
    'MxSmiNotFound',
    'MxSmiSnapshot',
    'ProcessInfo',
    'clear_cache',
    'device_count',
    'driver_version',
    'get_device',
    'is_available',
    'is_forced',
    'maca_version',
    'processes',
    'snapshot',
]


@dataclass(frozen=True)
class DeviceInfo:
    """MetaX GPU device information collected from ``mx-smi``."""

    index: int
    name: str | NaType = NA
    uuid: str | NaType = NA
    bus_id: str | NaType = NA
    state: str | NaType = NA
    persistence_mode: str | NaType = NA
    performance_state: str | NaType = NA
    memory_total: int | NaType = NA
    memory_used: int | NaType = NA
    memory_free: int | NaType = NA
    gpu_utilization: int | NaType = NA
    memory_utilization: int | NaType = NA
    temperature: int | NaType = NA
    power_usage: int | NaType = NA
    power_limit: int | NaType = NA
    fan_speed: int | NaType = NA


@dataclass(frozen=True)
class ProcessInfo:
    """MetaX GPU process information collected from ``mx-smi``."""

    gpu_index: int
    pid: int
    name: str | NaType = NA
    used_memory: int | NaType = NA


@dataclass(frozen=True)
class MxSmiSnapshot:
    """A single ``mx-smi`` sample."""

    devices: dict[int, DeviceInfo]
    processes: list[ProcessInfo]
    driver_version: str | NaType = NA
    maca_version: str | NaType = NA
    mxsmi_version: str | NaType = NA


class MxSmiError(RuntimeError):
    """Base exception for ``mx-smi`` query errors."""


class MxSmiNotFound(MxSmiError):
    """Raised when the ``mx-smi`` executable is not available."""


class MxSmiDeviceNotFound(MxSmiError):
    """Raised when a MetaX GPU device cannot be found."""


_BACKEND_ENVVAR = 'NVITOP_GPU_BACKEND'
_CACHE_TTL = 0.25
_CACHE_LOCK = threading.RLock()
_CACHE: MxSmiSnapshot | None = None
_CACHE_EXPIRES_AT = 0.0

# Inventory data (UUID / name / bus_id) from ``mx-smi -L`` changes very rarely,
# so we keep it in a separate cache with a much longer TTL to avoid spawning an
# extra subprocess on every 0.25 s snapshot refresh.
_LIST_CACHE_TTL = 60.0
_LIST_CACHE_LOCK = threading.RLock()
_LIST_CACHE: dict[int, DeviceInfo] | None = None
_LIST_CACHE_VERSION: str | NaType = NA
_LIST_CACHE_EXPIRES_AT = 0.0

_LIST_RE = re.compile(
    r'^GPU#(?P<index>\d+)\s+'
    r'(?P<name>.+?)\s+'
    r'(?P<bus_id>[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\s+'
    r'(?P<state>.*?)\s+'
    r'\(UUID:\s*(?P<uuid>[^)]+)\)\s*$',
)
_MXSMI_VERSION_RE = re.compile(r'\bmx-smi\s+version:\s*(?P<version>\S+)', flags=re.IGNORECASE)
_DRIVER_VERSION_RE = re.compile(r'Kernel Mode Driver Version:\s*(?P<version>[^\s|]+)')
_MACA_VERSION_RE = re.compile(r'MACA Version:\s*(?P<version>[^\s|]+)')
_SUMMARY_FIRST_RE = re.compile(
    r'^(?P<index>\d+)\s+(?P<name>.+?)\s+(?P<persistence>On|Off|Enable|Disable|Enabled|Disabled)\s*$',
)
_GPU_UTIL_RE = re.compile(r'(?P<util>\d+(?:\.\d+)?)\s*%')
_SUMMARY_SECOND_RE = re.compile(
    r'^(?P<temperature>\d+(?:\.\d+)?)C\s+'
    r'(?P<power_usage>\d+(?:\.\d+)?)W\s*/\s*'
    r'(?P<power_limit>\d+(?:\.\d+)?)W\s+'
    r'(?P<performance_state>\S+)',
)
_MEMORY_RE = re.compile(
    r'(?P<used>\d+(?:\.\d+)?)\s*/\s*(?P<total>\d+(?:\.\d+)?)\s*MiB',
    flags=re.IGNORECASE,
)
_PROCESS_RE = re.compile(
    r'^\|\s*(?P<gpu_index>\d+)\s+'
    r'(?P<pid>\d+)\s+'
    r'(?P<name>.*?)\s+'
    r'(?P<used_memory>\d+(?:\.\d+)?|N/A)\s*\|\s*$',
)


def is_forced() -> bool:
    """Return whether the MetaX backend was explicitly requested."""
    backend = os.getenv(_BACKEND_ENVVAR, default='').strip().lower().replace('_', '-')
    return backend in {'mx-smi', 'mxsmi', 'metax'}


def is_available() -> bool:
    """Return whether ``mx-smi`` can see at least one MetaX GPU."""
    if shutil.which('mx-smi') is None:
        return False
    try:
        return device_count() > 0
    except MxSmiError:
        return False


def device_count() -> int:
    """Return the number of MetaX GPUs visible to ``mx-smi``."""
    return len(snapshot().devices)


def driver_version() -> str | NaType:
    """Return the MetaX kernel mode driver version."""
    return snapshot().driver_version


def maca_version() -> str | NaType:
    """Return the MACA runtime version reported by ``mx-smi``."""
    return snapshot().maca_version


def get_device(
    *,
    index: int | bytes | None = None,
    uuid: str | bytes | None = None,
    bus_id: str | bytes | None = None,
) -> DeviceInfo:
    """Return a MetaX device by index, UUID, or PCI bus ID."""
    if sum(arg is not None for arg in (index, uuid, bus_id)) != 1:
        raise TypeError('get_device() expects exactly one identifier.')

    devices = snapshot().devices
    if index is not None:
        try:
            return devices[int(index)]
        except (KeyError, TypeError, ValueError) as ex:
            raise MxSmiDeviceNotFound(f'MetaX GPU index {index!r} was not found.') from ex

    identifier = _normalize_identifier(uuid if uuid is not None else bus_id)
    for device in devices.values():
        if identifier in {_normalize_identifier(device.uuid), _normalize_identifier(device.bus_id)}:
            return device

    raise MxSmiDeviceNotFound(f'MetaX GPU {identifier!r} was not found.')


def processes(index: int) -> list[ProcessInfo]:
    """Return processes reported by ``mx-smi`` for the given GPU index."""
    return [process for process in snapshot().processes if process.gpu_index == index]


def snapshot(*, ttl: float = _CACHE_TTL) -> MxSmiSnapshot:
    """Take or return a cached ``mx-smi`` snapshot."""
    global _CACHE, _CACHE_EXPIRES_AT  # pylint: disable=global-statement

    now = time.monotonic()
    with _CACHE_LOCK:
        if _CACHE is not None and now < _CACHE_EXPIRES_AT:
            return _CACHE

    current = _take_snapshot()

    with _CACHE_LOCK:
        _CACHE = current
        _CACHE_EXPIRES_AT = time.monotonic() + ttl
        return _CACHE


def clear_cache() -> None:
    """Clear the cached ``mx-smi`` snapshot and device inventory."""
    global _CACHE, _CACHE_EXPIRES_AT  # pylint: disable=global-statement
    global _LIST_CACHE, _LIST_CACHE_VERSION, _LIST_CACHE_EXPIRES_AT  # pylint: disable=global-statement

    with _CACHE_LOCK:
        _CACHE = None
        _CACHE_EXPIRES_AT = 0.0

    with _LIST_CACHE_LOCK:
        _LIST_CACHE = None
        _LIST_CACHE_VERSION = NA
        _LIST_CACHE_EXPIRES_AT = 0.0


def _get_inventory_cache() -> tuple[dict[int, DeviceInfo], str | NaType]:
    """Return the cached ``mx-smi -L`` inventory, refreshing when stale."""
    global _LIST_CACHE, _LIST_CACHE_VERSION, _LIST_CACHE_EXPIRES_AT  # pylint: disable=global-statement

    now = time.monotonic()
    with _LIST_CACHE_LOCK:
        if _LIST_CACHE is not None and now < _LIST_CACHE_EXPIRES_AT:
            return _LIST_CACHE, _LIST_CACHE_VERSION

    listed_devices, mxsmi_version = _parse_list_output(_run_mxsmi('-L'))

    with _LIST_CACHE_LOCK:
        _LIST_CACHE = listed_devices
        _LIST_CACHE_VERSION = mxsmi_version
        _LIST_CACHE_EXPIRES_AT = time.monotonic() + _LIST_CACHE_TTL
        return _LIST_CACHE, _LIST_CACHE_VERSION


def _take_snapshot() -> MxSmiSnapshot:
    listed_devices, listed_mxsmi_version = _get_inventory_cache()
    summary = _parse_summary_output(_run_mxsmi())

    devices = listed_devices.copy()
    for index, device in summary.devices.items():
        base = devices.get(index, DeviceInfo(index=index))
        devices[index] = replace(
            base,
            name=device.name if device.name is not NA else base.name,
            bus_id=device.bus_id if device.bus_id is not NA else base.bus_id,
            state=device.state if device.state is not NA else base.state,
            persistence_mode=(
                device.persistence_mode
                if device.persistence_mode is not NA
                else base.persistence_mode
            ),
            performance_state=(
                device.performance_state
                if device.performance_state is not NA
                else base.performance_state
            ),
            memory_total=device.memory_total,
            memory_used=device.memory_used,
            memory_free=device.memory_free,
            gpu_utilization=device.gpu_utilization,
            memory_utilization=device.memory_utilization,
            temperature=device.temperature,
            power_usage=device.power_usage,
            power_limit=device.power_limit,
            fan_speed=device.fan_speed,
        )

    return MxSmiSnapshot(
        devices=devices,
        processes=summary.processes,
        driver_version=summary.driver_version,
        maca_version=summary.maca_version,
        mxsmi_version=summary.mxsmi_version if summary.mxsmi_version is not NA else listed_mxsmi_version,
    )


def _run_mxsmi(*args: str) -> str:
    executable = shutil.which('mx-smi')
    if executable is None:
        raise MxSmiNotFound('The `mx-smi` executable was not found.')

    command = [executable, *args]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as ex:
        raise MxSmiError(f'Failed to run `{_command_to_string(command)}`.') from ex

    if completed.returncode != 0:
        output = completed.stdout.strip()
        message = f'`{_command_to_string(command)}` exited with status {completed.returncode}.'
        if output:
            message = f'{message}\n{output}'
        raise MxSmiError(message)

    return completed.stdout


def _parse_list_output(output: str) -> tuple[dict[int, DeviceInfo], str | NaType]:
    devices: dict[int, DeviceInfo] = {}
    mxsmi_version: str | NaType = NA

    for line in output.splitlines():
        version_match = _MXSMI_VERSION_RE.search(line)
        if version_match is not None:
            mxsmi_version = version_match.group('version')
            continue

        match = _LIST_RE.match(line.strip())
        if match is None:
            continue

        index = int(match.group('index'))
        devices[index] = DeviceInfo(
            index=index,
            name=match.group('name').strip(),
            uuid=match.group('uuid').strip(),
            bus_id=match.group('bus_id').strip(),
            state=match.group('state').strip() or NA,
        )

    return devices, mxsmi_version


def _parse_summary_output(output: str) -> MxSmiSnapshot:
    devices: dict[int, DeviceInfo] = {}
    processes: list[ProcessInfo] = []
    driver_version: str | NaType = NA
    maca_version: str | NaType = NA
    mxsmi_version: str | NaType = NA
    lines = output.splitlines()

    for lineno, line in enumerate(lines):
        version_match = _MXSMI_VERSION_RE.search(line)
        if version_match is not None:
            mxsmi_version = version_match.group('version')

        driver_match = _DRIVER_VERSION_RE.search(line)
        if driver_match is not None:
            driver_version = driver_match.group('version')

        maca_match = _MACA_VERSION_RE.search(line)
        if maca_match is not None:
            maca_version = maca_match.group('version')

        parts = _split_table_line(line)
        if len(parts) != 3:
            continue

        first_match = _SUMMARY_FIRST_RE.match(parts[0])
        if first_match is None:
            continue

        try:
            next_parts = _split_table_line(lines[lineno + 1])
        except IndexError:
            continue
        if len(next_parts) != 3:
            continue

        second_match = _SUMMARY_SECOND_RE.match(next_parts[0])
        memory_match = _MEMORY_RE.search(next_parts[1])
        if second_match is None or memory_match is None:
            continue

        index = int(first_match.group('index'))
        memory_used = _mib_to_bytes(memory_match.group('used'))
        memory_total = _mib_to_bytes(memory_match.group('total'))
        memory_free = (
            memory_total - memory_used
            if isinstance(memory_total, int) and isinstance(memory_used, int)
            else NA
        )
        devices[index] = DeviceInfo(
            index=index,
            name=first_match.group('name').strip(),
            bus_id=parts[1].strip(),
            state=next_parts[2].strip() or NA,
            persistence_mode=_normalize_mode(first_match.group('persistence')),
            performance_state=second_match.group('performance_state'),
            memory_total=memory_total,
            memory_used=memory_used,
            memory_free=memory_free,
            gpu_utilization=_percent_to_int(parts[2]),
            temperature=round(float(second_match.group('temperature'))),
            power_usage=_watts_to_milliwatts(second_match.group('power_usage')),
            power_limit=_watts_to_milliwatts(second_match.group('power_limit')),
        )

    in_process_table = False
    for line in lines:
        if '| Process:' in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        if 'no process found' in line.lower():
            continue

        process_match = _PROCESS_RE.match(line)
        if process_match is None:
            continue

        processes.append(
            ProcessInfo(
                gpu_index=int(process_match.group('gpu_index')),
                pid=int(process_match.group('pid')),
                name=process_match.group('name').strip() or NA,
                used_memory=_mib_to_bytes(process_match.group('used_memory')),
            ),
        )

    return MxSmiSnapshot(
        devices=devices,
        processes=processes,
        driver_version=driver_version,
        maca_version=maca_version,
        mxsmi_version=mxsmi_version,
    )


def _split_table_line(line: str) -> list[str]:
    if not line.startswith('|'):
        return []
    return [part.strip() for part in line.strip().strip('|').split('|')]


def _mib_to_bytes(value: str) -> int | NaType:
    if value.upper() == 'N/A':
        return NA
    return round(float(value) * MiB)


def _watts_to_milliwatts(value: str) -> int:
    return round(float(value) * 1000)


def _percent_to_int(value: str) -> int | NaType:
    match = _GPU_UTIL_RE.search(value)
    if match is None:
        return NA
    return round(float(match.group('util')))


def _normalize_mode(value: str) -> str | NaType:
    normalized = value.strip().lower()
    if normalized in {'on', 'enable', 'enabled'}:
        return 'Enabled'
    if normalized in {'off', 'disable', 'disabled'}:
        return 'Disabled'
    return NA


def _normalize_identifier(value: str | bytes | NaType | None) -> str:
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    if value is None or value is NA:
        return ''
    return str(value).strip().lower()


def _command_to_string(command: list[str]) -> str:
    return ' '.join(command)
