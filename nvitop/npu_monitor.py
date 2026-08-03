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

import select
import sys
import time
from typing import TYPE_CHECKING, Any, Iterable, Tuple

from nvitop.api import libnpu
from nvitop.api.npu_device import NpuDevice, NpuProcess
from nvitop.api.utils import NaType, colored
from nvitop.npu_ui import (
    render_devices_table,
    render_footer,
    render_header,
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

    return [
        (
            NpuProcess(pid, used_memory=info['memory'], name=info['name']),
            tuple(sorted(info['npus'])),
        )
        for pid, info in merged.items()
    ]


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
    return sys.stdin.read(1).lower() if readable else None


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
    compact = mode == 'compact'
    paused = False
    sort_index = PROCESS_SORTS.index(getattr(args, 'sort', 'memory'))
    sort_by = PROCESS_SORTS[sort_index]
    driver = NpuDevice.driver_version()
    processes: list[ProcessEntry] = []
    needs_render = True
    force_refresh = True
    input_fd, input_settings = _configure_terminal_input()

    sys.stdout.write(_HIDE_CURSOR)
    try:
        while True:
            if needs_render:
                if not paused or force_refresh:
                    processes = sort_processes(
                        filter_processes(
                            collect_processes(devices, use_cache=False),
                            users=users,
                            pids=pids,
                        ),
                        sort_by,
                    )
                    prefetch_clocks(devices)
                    force_refresh = False

                sys.stdout.write(_CLEAR_SCREEN)
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
                print(
                    render_summary(
                        devices,
                        process_count=len(processes),
                        colorful=args.colorful,
                        no_unicode=args.no_unicode,
                    ),
                )
                print()
                print(render_devices_table(devices, colorful=args.colorful, no_unicode=args.no_unicode))
                if not compact and not getattr(args, 'no_processes', False):
                    print()
                    if processes:
                        print(
                            render_processes_table(
                                processes,
                                colorful=args.colorful,
                                no_unicode=args.no_unicode,
                            ),
                        )
                    else:
                        print(colored('No running processes found.', 'yellow'))
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
            if key == ' ':
                paused = not paused
                needs_render = True
            elif key == 'r':
                force_refresh = True
                needs_render = True
            elif key == 'c':
                compact = not compact
                needs_render = True
            elif key == 's':
                sort_index = (sort_index + 1) % len(PROCESS_SORTS)
                sort_by = PROCESS_SORTS[sort_index]
                processes = sort_processes(processes, sort_by)
                needs_render = True
            elif key is None and not paused:
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
