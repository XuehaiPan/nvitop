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
"""Collection and interactive runtime for the Ascend NPU monitor."""

from __future__ import annotations

import os
import select
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any, Iterable, Tuple

import psutil

from nvitop.api import libnpu
from nvitop.api.npu_device import NpuDevice, NpuProcess
from nvitop.api.utils import NaType, colored
from nvitop.npu_ui import (
    MonitorHistory,
    render_device_dashboard,
    render_devices_table,
    render_footer,
    render_header,
    render_help,
    render_history_panel,
    render_monitor_title,
    render_processes_table,
    render_summary,
)


if TYPE_CHECKING:
    import argparse


ProcessEntry = Tuple[NpuProcess, Tuple[int, ...]]
PROCESS_SORTS = ('memory', 'pid', 'user', 'name', 'npu')

_CLEAR_SCREEN = '\x1b[2J\x1b[H'
_HIDE_CURSOR = '\x1b[?25l'
_SHOW_CURSOR = '\x1b[?25h'


def collect_processes(
    devices: Iterable[NpuDevice],
    use_cache: bool = True,
    process_cache: dict[int, NpuProcess] | None = None,
) -> list[ProcessEntry]:
    """Collect and merge process rows from the global NPU overview."""
    indices = {device.index for device in devices}
    data = libnpu.npu_query_global(use_cache=use_cache)

    merged: dict[int, dict[str, Any]] = {}
    for row in data['processes']:
        npu = int(row['npu'])
        if npu not in indices:
            continue
        pid = int(row['pid'])
        memory = int(row['mem']) * 1024 * 1024 if not isinstance(row['mem'], NaType) else 0
        if pid not in merged:
            merged[pid] = {'name': row['name'], 'memory': 0, 'npus': set()}
        merged[pid]['memory'] += memory
        merged[pid]['npus'].add(npu)

    entries = []
    active_processes = {}
    for pid, info in merged.items():
        process = process_cache.get(pid) if process_cache is not None else None
        if process is None:
            process = NpuProcess(pid, used_memory=info['memory'], name=info['name'])
        else:
            process.used_memory = info['memory']
            process._name = info['name']  # pylint: disable=protected-access
        active_processes[pid] = process
        entries.append((process, tuple(sorted(info['npus']))))
    if process_cache is not None:
        process_cache.clear()
        process_cache.update(active_processes)
    return entries


def prefetch_clocks(devices: Iterable[NpuDevice], *, use_cache: bool = True) -> None:
    """Warm per-device clock caches with parallel ``npu-smi`` batches."""
    indices = [device.index for device in devices]
    libnpu.npu_query_kv_batch(indices, 'common', use_cache=use_cache)
    libnpu.npu_query_kv_batch(indices, 'memory', use_cache=use_cache)


def filter_processes(
    processes: list[ProcessEntry],
    *,
    users: set[str] | None = None,
    pids: set[int] | None = None,
) -> list[ProcessEntry]:
    """Filter collected processes by owner and PID."""
    filtered = []
    for process, npus in processes:
        if pids is not None and process.pid not in pids:
            continue
        if users is not None:
            username = process.username()
            if isinstance(username, NaType) or username not in users:
                continue
        filtered.append((process, npus))
    return filtered


def sort_processes(processes: list[ProcessEntry], sort_by: str) -> list[ProcessEntry]:
    """Sort process rows using stable, user-facing keys."""
    if sort_by == 'memory':
        return sorted(
            processes,
            key=lambda item: item[0].used_memory if item[0].used_memory is not None else 0,
            reverse=True,
        )
    if sort_by == 'pid':
        return sorted(processes, key=lambda item: item[0].pid)
    if sort_by == 'user':
        return sorted(processes, key=lambda item: str(item[0].username()).lower())
    if sort_by == 'name':
        return sorted(processes, key=lambda item: str(item[0].name()).lower())
    if sort_by == 'npu':
        return sorted(processes, key=lambda item: item[1])
    raise ValueError(f'unsupported process sort key: {sort_by!r}')


def _configure_terminal_input() -> tuple[int | None, list[Any] | None]:
    """Enable cbreak input when stdin is an interactive POSIX terminal."""
    if not sys.stdin.isatty():
        return None, None
    try:
        import termios  # pylint: disable=import-outside-toplevel
        import tty  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None, None
    try:
        input_fd = sys.stdin.fileno()
        settings = termios.tcgetattr(input_fd)
        tty.setcbreak(input_fd)
    except (OSError, termios.error):
        return None, None
    else:
        return input_fd, settings


def _restore_terminal_input(input_fd: int | None, settings: list[Any] | None) -> None:
    """Restore terminal input settings after monitor mode exits."""
    if input_fd is None or settings is None:
        return
    import termios  # pylint: disable=import-outside-toplevel

    termios.tcsetattr(input_fd, termios.TCSADRAIN, settings)


def _read_key(input_fd: int | None, timeout: float) -> str | None:
    """Read one monitor key, waiting up to ``timeout`` seconds."""
    if input_fd is None:
        time.sleep(timeout)
        return None
    readable, _, _ = select.select([input_fd], [], [], timeout)
    return os.read(input_fd, 1).decode(errors='ignore').lower() if readable else None


def _sample_history(history: MonitorHistory, devices: Iterable[NpuDevice]) -> None:
    """Append one host and aggregate NPU sample to the history window."""
    memory_used = 0.0
    memory_total = 0.0
    utilizations = []
    for device in devices:
        used = device.memory_used()
        total = device.memory_total()
        utilization = device.gpu_utilization()
        if not isinstance(used, NaType) and not isinstance(total, NaType):
            memory_used += float(used)
            memory_total += float(total)
        if not isinstance(utilization, NaType):
            utilizations.append(float(utilization))

    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    load_average = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else os.getloadavg()
    history.append(
        cpu_percent=psutil.cpu_percent(interval=None),
        memory_percent=virtual_memory.percent,
        memory_used=virtual_memory.used,
        swap_percent=swap_memory.percent,
        npu_memory_percent=100.0 * memory_used / memory_total if memory_total else 0.0,
        npu_utilization=sum(utilizations) / len(utilizations) if utilizations else 0.0,
        load_average=load_average,
    )


def run_monitor(
    devices: list[NpuDevice],
    *,
    users: set[str] | None = None,
    pids: set[int] | None = None,
    interval: float,
    mode: str,
    args: argparse.Namespace,
) -> None:
    """Run the interactive monitor loop."""
    term_size = shutil.get_terminal_size()
    compact = mode == 'compact' or (
        mode == 'auto'
        and (term_size.columns < 100 or term_size.lines < 30 + 3 * len(devices))
    )
    paused = False
    show_help = False
    sort_index = PROCESS_SORTS.index(getattr(args, 'sort', 'memory'))
    sort_by = PROCESS_SORTS[sort_index]
    driver = NpuDevice.driver_version()
    history = MonitorHistory()
    process_cache: dict[int, NpuProcess] = {}
    processes: list[ProcessEntry] = []
    summary_text = ''
    devices_text = ''
    dashboard_text = ''
    history_text = ''
    processes_text = ''
    needs_render = True
    refresh_data = True
    input_fd, input_settings = _configure_terminal_input()

    sys.stdout.write(_HIDE_CURSOR)
    try:
        while True:
            if needs_render:
                if refresh_data:
                    processes = sort_processes(
                        filter_processes(
                            collect_processes(
                                devices,
                                use_cache=False,
                                process_cache=process_cache,
                            ),
                            users=users,
                            pids=pids,
                        ),
                        sort_by,
                    )
                    prefetch_clocks(devices)
                    summary_text = render_summary(
                        devices,
                        process_count=len(processes),
                        colorful=args.colorful,
                        no_unicode=args.no_unicode,
                    )
                    devices_text = render_devices_table(
                        devices,
                        colorful=args.colorful,
                        no_unicode=args.no_unicode,
                    )
                    term_size = shutil.get_terminal_size()
                    dashboard_text = render_device_dashboard(
                        devices,
                        colorful=args.colorful,
                        no_unicode=args.no_unicode,
                        width=term_size.columns,
                    )
                    _sample_history(history, devices)
                    history_text = render_history_panel(
                        history,
                        no_unicode=args.no_unicode,
                        width=term_size.columns,
                    )
                    processes_text = render_processes_table(
                        processes,
                        colorful=args.colorful,
                        no_unicode=args.no_unicode,
                        width=term_size.columns,
                    )
                    refresh_data = False

                sys.stdout.write(_CLEAR_SCREEN)
                if show_help:
                    print(
                        render_monitor_title(
                            paused=paused,
                            no_unicode=args.no_unicode,
                            width=term_size.columns,
                        ),
                    )
                    print()
                    print(render_help(no_unicode=args.no_unicode, width=term_size.columns))
                elif not compact:
                    print(
                        render_monitor_title(
                            paused=paused,
                            no_unicode=args.no_unicode,
                            width=term_size.columns,
                        ),
                    )
                    print()
                    print(dashboard_text)
                    if history_text:
                        print()
                        print(history_text)
                    if os.geteuid() == 0:
                        print()
                        print(colored('! CAUTION: SUPERUSER LOGGED-IN.', 'yellow'))
                    if not getattr(args, 'no_processes', False):
                        print()
                        if processes:
                            print(processes_text)
                        else:
                            print(colored('No running processes found.', 'yellow'))
                else:
                    print(
                        render_header(
                            devices,
                            monitor=True,
                            colorful=args.colorful,
                            driver=driver,
                            paused=paused,
                            no_unicode=args.no_unicode,
                        ),
                    )
                    print()
                    print(summary_text)
                    print()
                    print(devices_text)
                print()
                print(
                    render_footer(
                        compact=compact,
                        paused=paused,
                        sort_by=sort_by,
                        no_unicode=args.no_unicode,
                    ),
                )
                sys.stdout.flush()
                needs_render = False

            key = _read_key(input_fd, 0.25 if paused else interval)
            if key == 'q':
                return
            if key == 'h':
                show_help = not show_help
                needs_render = True
            elif key == ' ':
                paused = not paused
                if not paused:
                    refresh_data = True
                needs_render = True
            elif key == 'r':
                refresh_data = True
                needs_render = True
            elif key == 'c':
                compact = not compact
                needs_render = True
            elif key == 's':
                sort_index = (sort_index + 1) % len(PROCESS_SORTS)
                sort_by = PROCESS_SORTS[sort_index]
                processes = sort_processes(processes, sort_by)
                processes_text = render_processes_table(
                    processes,
                    colorful=args.colorful,
                    no_unicode=args.no_unicode,
                    width=term_size.columns,
                )
                needs_render = True
            elif key is None and not paused and not show_help:
                refresh_data = True
                needs_render = True
    finally:
        _restore_terminal_input(input_fd, input_settings)
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()


__all__ = [
    'PROCESS_SORTS',
    'ProcessEntry',
    'collect_processes',
    'filter_processes',
    'prefetch_clocks',
    'run_monitor',
    'sort_processes',
]
