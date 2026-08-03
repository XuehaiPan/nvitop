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
"""Utilities for the Ascend NPU management (`npu-smi`).

This module provides a lightweight, dependency-free wrapper around the Huawei
Ascend ``npu-smi`` command-line tool. It plays the same role for Ascend NPUs as
:mod:`nvitop.api.libnvml` plays for NVIDIA GPUs: all NPU metrics (utilization,
memory, temperature, power, clocks, health and running processes) are obtained
by executing and parsing the ``npu-smi`` text output.

The ``npu-smi`` tool is part of the Huawei Ascend CANN toolkit / NPU driver
package and is available at ``/usr/local/sbin/npu-smi`` on Ascend servers.
"""

# pylint: disable=too-many-lines,invalid-name

from __future__ import annotations

import atexit as _atexit
import concurrent.futures as _futures
import os as _os
import re as _re
import shutil as _shutil
import subprocess as _subprocess
import threading as _threading
import time as _time
from typing import Any as _Any

from nvitop.api.utils import NA, NaType  # noqa: F401  # pylint: disable=unused-import


__all__ = [
    'NpuError',
    'NpuSmiNotFound',
    'NpuQueryError',
    'npu_init',
    'npu_shutdown',
    'npu_smi_path',
    'npu_driver_version',
    'npu_device_count',
    'npu_device_chip_name',
    'npu_query_kv',
    'npu_query_kv_batch',
    'npu_query_proc_mem',
    'npu_query_global',
    'clear_cache',
]

# Common paths of the `npu-smi` executable on Ascend servers #########################################
NPU_SMI_CANDIDATES = (
    '/usr/local/sbin/npu-smi',
    '/usr/local/bin/npu-smi',
    '/usr/bin/npu-smi',
    '/usr/sbin/npu-smi',
)

# Time-to-live of the cached detailed queries (usages / memory / common / temp / power / board)
_DETAIL_CACHE_TTL = 30.0  # in seconds
# Time-to-live of the cached global overview (`npu-smi info`).
# One invocation covers all devices and processes, so it is cached with a short
# TTL and shared by every device query; monitor mode refreshes it periodically.
_GLOBAL_CACHE_TTL = 5.0  # in seconds

_KV_PATTERN = _re.compile(r'^\s*(?P<key>[^:]*?)\s*:\s*(?P<value>.*?)\s*$')
_PROC_MEM_PATTERN = _re.compile(
    r'Process id:(?P<pid>\d+)\s+Process name:(?P<name>\S+)\s+Process memory\(MB\):(?P<mem>\d+)',
)
_DRIVER_VERSION_PATTERN = _re.compile(r'^\s*\|?\s*npu-smi\s+(\S+)\s+Version:\s+(\S+)')
_NUMBER_PATTERN = _re.compile(r'\d+(?:\.\d+)?')
_FRACTION_PATTERN = _re.compile(r'(\d+(?:\.\d+)?)\s*/\s*(\d+)')

# Runtime state ######################################################################################
_NPU_SMI: str | None = None
_NPU_SMI_LOCK: _threading.RLock = _threading.RLock()
_INITIALIZED = False

_CACHE: dict[tuple[_Any, ...], tuple[float, _Any]] = {}
_CACHE_LOCK: _threading.RLock = _threading.RLock()


class NpuError(Exception):
    """Base exception class for NPU query errors."""


class NpuSmiNotFound(NpuError):
    """Raised when the ``npu-smi`` executable cannot be found.

    Usually the Ascend NPU driver / CANN toolkit is not installed on this host.
    """


class NpuQueryError(NpuError):
    """Raised when an ``npu-smi`` query fails (e.g. invalid card id)."""


def npu_smi_path() -> str | None:
    """Locate the ``npu-smi`` executable and cache the result."""
    global _NPU_SMI  # pylint: disable=global-statement

    with _NPU_SMI_LOCK:
        if _NPU_SMI is not None:
            return _NPU_SMI

        for candidate in NPU_SMI_CANDIDATES:
            if _os.path.isfile(candidate) and _os.access(candidate, _os.X_OK):
                _NPU_SMI = candidate
                return _NPU_SMI

        found = _shutil.which('npu-smi')
        if found is not None:
            _NPU_SMI = found
            return _NPU_SMI

        return None


def npu_init() -> None:
    """Verify that ``npu-smi`` is available on this host.

    Raises:
        NpuSmiNotFound: If the ``npu-smi`` executable cannot be found.
        NpuQueryError: If the ``npu-smi`` executable cannot be executed.
    """
    global _INITIALIZED  # pylint: disable=global-statement

    if _INITIALIZED:
        return

    path = npu_smi_path()
    if path is None:
        raise NpuSmiNotFound(
            'Cannot find the `npu-smi` executable. '
            'Please make sure the Ascend NPU driver / CANN toolkit is installed.',
        )

    try:
        npu_query_raw('info')
    except (OSError, _subprocess.SubprocessError) as ex:
        raise NpuQueryError(f'Failed to execute `{path}`: {ex}') from ex

    _INITIALIZED = True
    _atexit.register(npu_shutdown)


def npu_shutdown() -> None:
    """Reset the initialization state and clear all cached query results."""
    global _INITIALIZED  # pylint: disable=global-statement

    _INITIALIZED = False
    clear_cache()


def clear_cache() -> None:
    """Clear all cached ``npu-smi`` query results."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _cache_get(key: _Any, ttl: float = _DETAIL_CACHE_TTL) -> _Any:
    with _CACHE_LOCK:
        cached = _CACHE.get(key, None)
        if cached is None:
            return None
        timestamp, value = cached
        if _time.monotonic() - timestamp <= ttl:
            return value
        _CACHE.pop(key, None)
        return None


def _cache_set(key: _Any, value: _Any) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (_time.monotonic(), value)


def npu_query_raw(*args: str) -> str:
    """Execute ``npu-smi <args...>`` and return the raw stdout text.

    Raises:
        NpuSmiNotFound: If the ``npu-smi`` executable cannot be found.
        NpuQueryError: If the ``npu-smi`` process exits with a non-zero status or times out.
    """
    path = npu_smi_path()
    if path is None:
        raise NpuSmiNotFound('Cannot find the `npu-smi` executable.')

    env = dict(_os.environ)
    env['LC_ALL'] = 'C'  # force English output for stable parsing
    try:
        completed = _subprocess.run(
            [path, *args],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
            check=False,
        )
    except _subprocess.TimeoutExpired as ex:
        raise NpuQueryError(f'`npu-smi {" ".join(args)}` timed out.') from ex
    except OSError as ex:
        raise NpuQueryError(f'Failed to execute `{path}`: {ex}') from ex

    if completed.returncode != 0:
        stderr = (completed.stderr or '').strip()
        detail = f': {stderr}' if stderr else ''
        raise NpuQueryError(f'`npu-smi {" ".join(args)}` failed{detail}')

    return completed.stdout or ''


def npu_driver_version() -> str | NaType:
    """Get the version of the installed Ascend NPU driver (e.g. ``25.5.1``).

    The version line is part of the global ``npu-smi info`` output, so the
    cached global query is reused (no extra ``npu-smi`` invocation).
    """
    try:
        output = _global_raw()
    except NpuError:
        return NA
    return _parse_driver_version(output)


def _parse_driver_version(output: str) -> str | NaType:
    """Parse the driver version from the ``npu-smi ... Version: ...`` line."""
    for line in output.splitlines():
        match = _DRIVER_VERSION_PATTERN.match(line.strip())
        if match is not None:
            return match.group(2)
    return NA


def npu_device_count() -> int:
    """Get the number of Ascend NPU devices in the system.

    The device list is obtained from ``npu-smi info -m`` by counting the chips
    whose chip name contains ``Ascend`` (the MCU companion chips are excluded).
    The result is cached briefly because every :class:`NpuDevice` construction
    performs this query.
    """
    cache_key = ('device-count',)
    cached = _cache_get(cache_key, ttl=_GLOBAL_CACHE_TTL)
    if cached is not None:
        return cached

    try:
        output = npu_query_raw('info', '-m')
    except NpuError:
        return 0
    count = _parse_device_count(output)
    _cache_set(cache_key, count)
    return count


def _parse_device_count(output: str) -> int:
    """Parse the number of Ascend NPU devices from ``npu-smi info -m`` output."""
    return sum(
        1
        for line in output.splitlines()
        if 'Ascend' in line and len(line.split()) >= 2 and line.split()[0].isdigit()
    )


def npu_device_chip_name(card_id: int) -> str | NaType:
    """Get the chip name (e.g. ``Ascend 910B3``) of the given NPU card.

    The ``npu-smi info -m`` output covers every card, so the mapping is cached
    and shared by all devices.
    """
    cache_key = ('chip-names',)
    cached = _cache_get(cache_key, ttl=_GLOBAL_CACHE_TTL)
    if cached is None:
        try:
            output = npu_query_raw('info', '-m')
        except NpuError:
            return NA
        cached = _parse_chip_names(output)
        _cache_set(cache_key, cached)
    return cached.get(card_id, NA)


def _parse_chip_names(output: str) -> dict[int, str]:
    """Parse the ``{card_id: chip_name}`` mapping from ``npu-smi info -m`` output."""
    names: dict[int, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and 'Ascend' in line:
            names[int(parts[0])] = ' '.join(parts[3:]) if len(parts) > 3 else ' '.join(parts[1:])
    return names


def _parse_kv_output(output: str) -> dict[str, str]:
    """Parse the ``Key : value`` lines of an ``npu-smi info -t <type>`` output into a dict.

    Lines without a colon are skipped; values like ``-`` or ``NA`` are kept as-is
    and should be converted by the caller.  When a key appears more than once
    (e.g. ``Temperature(C)`` is reported both for the NPU chip and the MCU
    companion chip), the first occurrence (the NPU chip) wins.
    """
    ret: dict[str, str] = {}
    for line in output.splitlines():
        match = _KV_PATTERN.match(line)
        if match is not None:
            key = match.group('key').strip()
            value = match.group('value').strip()
            if key and key not in ret:
                ret[key] = value
    return ret


def npu_query_kv(card_id: int, query_type: str, *, use_cache: bool = True) -> dict[str, str]:
    """Query detailed information of one NPU card via ``npu-smi info -t <type> -i <id>``.

    Supported query types (validated against npu-smi 25.5.1):
        ``usages``, ``memory``, ``common``, ``temp``, ``power``, ``board``

    Args:
        card_id (int): The NPU card id (as shown by ``npu-smi info -m``).
        query_type (str): One of the supported query types.
        use_cache (bool): Whether to use the cached result (default: ``True``).
            Pass ``False`` in monitor mode to always get fresh data.

    Returns:
        dict: A mapping of field names to string values parsed from the output.

    Raises:
        NpuSmiNotFound: If the ``npu-smi`` executable cannot be found.
        NpuQueryError: If the query fails (e.g. invalid card id).
    """
    cache_key = ('kv', card_id, query_type)
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    output = npu_query_raw('info', '-t', query_type, '-i', str(card_id))
    parsed = _parse_kv_output(output)
    if use_cache:
        _cache_set(cache_key, parsed)
    return parsed


def npu_query_kv_batch(
    card_ids: list[int] | tuple[int, ...],
    query_type: str,
    *,
    use_cache: bool = True,
    max_workers: int = 8,
) -> dict[int, dict[str, str]]:
    """Query detailed information of multiple NPU cards in parallel.

    ``npu-smi`` has a startup cost of roughly half a second per invocation, so
    querying the cards one by one is slow.  This helper runs the per-card
    queries concurrently (each one still goes through the shared TTL cache) and
    returns a ``{card_id: parsed dict}`` mapping.

    Args:
        card_ids: The NPU card ids to query.
        query_type: One of the supported query types (see :func:`npu_query_kv`).
        use_cache (bool): Whether to use the cached results (default: ``True``).
        max_workers (int): The maximum number of concurrent ``npu-smi`` processes.

    Returns:
        dict: A mapping of card ids to the parsed key-value dicts.
    """
    with _futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(npu_query_kv, card_id, query_type, use_cache=use_cache): card_id
            for card_id in card_ids
        }
        results: dict[int, dict[str, str]] = {}
        for future in _futures.as_completed(futures):
            card_id = futures[future]
            try:
                results[card_id] = future.result()
            except NpuError:
                results[card_id] = {}
    return results


def npu_query_proc_mem(card_id: int, *, use_cache: bool = True) -> list[dict[str, str]]:
    """Get the process memory table of one NPU card via ``npu-smi info -t proc-mem -i <id>``.

    Args:
        card_id (int): The NPU card id.
        use_cache (bool): Whether to use the cached result (default: ``True``).
            Pass ``False`` in monitor mode to always get fresh process memory data.

    Returns:
        list: A list of dicts with keys ``pid``, ``name`` and ``mem`` (in MB).
    """
    cache_key = ('proc-mem', card_id)
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    try:
        output = npu_query_raw('info', '-t', 'proc-mem', '-i', str(card_id))
    except NpuQueryError:
        return []
    parsed = _parse_proc_mem(output)
    if use_cache:
        _cache_set(cache_key, parsed)
    return parsed


def _parse_proc_mem(output: str) -> list[dict[str, str]]:
    """Parse the process memory rows from ``npu-smi info -t proc-mem`` output."""
    return [
        {'pid': match.group('pid'), 'name': match.group('name'), 'mem': match.group('mem')}
        for line in output.splitlines()
        for match in [_PROC_MEM_PATTERN.search(line)]
        if match is not None
    ]


def _split_table_row(line: str) -> list[str]:
    """Split a ``npu-smi`` table row (``| a | b |``) into stripped cells."""
    return [cell.strip() for cell in line.split('|')][1:-1]


def _parse_compound_field(text: str) -> tuple[list[float], list[tuple[int, int]]]:
    """Parse a whitespace-separated compound field of ``npu-smi``.

    The device rows of ``npu-smi info`` pack several metrics into one table
    column separated by multiple spaces, e.g. ``99.5        33      0    / 0``
    (power / temperature / hugepages) or ``0      0    / 0      59189/ 65536``
    (AICore / DDR / HBM).  This helper extracts the plain numbers (in order)
    and the ``used/total`` fraction pairs separately.

    Returns:
        tuple: ``(plain_numbers, fractions)`` where ``fractions`` is a list of
        ``(used, total)`` tuples parsed from all ``a/b`` patterns.
    """
    fractions = [
        (int(used), int(total))
        for used, total in _FRACTION_PATTERN.findall(text)
    ]
    without_fractions = _FRACTION_PATTERN.sub(' ', text)
    plain = [float(num) for num in _NUMBER_PATTERN.findall(without_fractions)]
    return plain, fractions


def _global_raw(*, use_cache: bool = True) -> str:
    """Get the raw output of the global ``npu-smi info`` (cached with a short TTL).

    The global overview is the primary monitoring source: one invocation covers
    every device (utilization, memory, power, temperature, health) and every
    running process with its NPU memory.  The raw text is cached so that all
    consumers share a single ``npu-smi`` invocation.
    """
    cache_key = ('global-raw',)
    if use_cache:
        cached = _cache_get(cache_key, ttl=_GLOBAL_CACHE_TTL)
        if cached is not None:
            return cached

    output = npu_query_raw('info')
    # A forced refresh becomes the new shared snapshot for device and process readers.
    _cache_set(cache_key, output)
    return output


def npu_query_global(*, use_cache: bool = True) -> dict[str, _Any]:
    """Query the global overview of all NPU devices via ``npu-smi info``.

    Args:
        use_cache (bool): Whether to use the cached result (default: ``True``).
            The global overview is cached for :data:`_GLOBAL_CACHE_TTL` seconds;
            pass ``False`` in monitor mode to always get fresh data.

    Returns:
        dict: A dict with two keys:

            - ``devices``: a list of per-device dicts with keys ``index``, ``name``,
              ``health``, ``power`` (W, float), ``temperature`` (C, int), ``bus_id``,
              ``aicore`` (percent, int), ``memory_used`` / ``memory_total`` (MB, int)
              and ``ddr_used`` / ``ddr_total`` (MB, int).
            - ``processes``: a list of per-process dicts with keys ``npu``, ``chip``,
              ``pid``, ``name`` and ``mem`` (MB, int).

    Raises:
        NpuSmiNotFound: If the ``npu-smi`` executable cannot be found.
        NpuQueryError: If the query fails.
    """
    output = _global_raw(use_cache=use_cache)
    return _parse_global(output)


def _parse_global(output: str) -> dict[str, _Any]:
    """Parse the global overview (devices and processes) from ``npu-smi info`` output."""
    devices: list[dict[str, _Any]] = []
    processes: list[dict[str, _Any]] = []

    lines = output.splitlines()
    section = 'header'  # one of 'header', 'devices', 'processes'
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith('+'):
            i += 1
            continue

        cells = _split_table_row(line)
        if not cells:
            i += 1
            continue

        # Detect section headers
        if len(cells) > 1 and 'Health' in cells[1]:
            section = 'devices'
            i += 1
            continue
        if len(cells) > 1 and 'Process id' in cells[1]:
            section = 'processes'
            i += 1
            continue

        if section == 'devices' and len(cells) >= 3:
            # First row of a device: | NPU Name | Health | Power Temp Hugepages |
            first_field = cells[0].split()
            if len(first_field) >= 2 and first_field[0].isdigit():
                plain, fractions = _parse_compound_field(cells[2])
                device: dict[str, _Any] = {
                    'index': int(first_field[0]),
                    'name': ' '.join(first_field[1:]),
                    'health': cells[1],
                    'power': float(plain[0]) if len(plain) > 0 else NA,
                    'temperature': int(plain[1]) if len(plain) > 1 else NA,
                    'hugepages_used': fractions[0][0] if len(fractions) > 0 else 0,
                    'hugepages_total': fractions[0][1] if len(fractions) > 0 else 0,
                }
                # Second row of the same device: | Chip | Bus-Id | AICore DDR HBM |
                if i + 1 < n and lines[i + 1].startswith('|'):
                    cells2 = _split_table_row(lines[i + 1])
                    if len(cells2) >= 3:
                        plain2, fractions2 = _parse_compound_field(cells2[2])
                        device['chip'] = _parse_int(cells2[0])
                        device['bus_id'] = cells2[1]
                        device['aicore'] = int(plain2[0]) if len(plain2) > 0 else NA
                        ddr = fractions2[0] if len(fractions2) > 0 else (0, 0)
                        hbm = fractions2[1] if len(fractions2) > 1 else (0, 0)
                        device['ddr_used'], device['ddr_total'] = ddr
                        device['memory_used'], device['memory_total'] = hbm
                    i += 1
                devices.append(device)

        elif section == 'processes' and len(cells) >= 4:
            # | NPU     Chip              | Process id    | Process name | Process memory(MB) |
            npu_chip = cells[0].split()
            if len(npu_chip) >= 2 and npu_chip[0].isdigit():
                processes.append(
                    {
                        'npu': int(npu_chip[0]),
                        'chip': _parse_int(npu_chip[1]),
                        'pid': _parse_int(cells[1]),
                        'name': cells[2],
                        'mem': _parse_int(cells[3]),
                    },
                )
        i += 1

    return {'devices': devices, 'processes': processes}


def _parse_int(value: str | int | float | None, default: int | NaType = NA) -> int | NaType:
    """Parse an integer from a string, returning ``NA`` for unparseable values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value else default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _parse_float(value: str | int | float | None, default: float | NaType = NA) -> float | NaType:
    """Parse a float from a string, returning ``NA`` for unparseable values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value) if value == value else default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default
