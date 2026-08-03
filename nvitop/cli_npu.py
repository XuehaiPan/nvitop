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
"""The interactive Ascend-NPU process viewer (``nvitop-npu``).

A drop-in replacement of ``nvitop`` for Huawei Ascend NPU servers.  It queries
the NPU status through the ``npu-smi`` command-line tool and renders an
``nvidia-smi``-style report (single shot with ``-1`` or continuously refreshed
with ``-m``).

Examples:
    .. code-block:: console

        $ nvitop-npu          # one-shot report of all NPU devices and processes
        $ nvitop-npu -1       # same as above, report query data only once
        $ nvitop-npu -m       # monitor mode, refresh every 2 seconds
        $ nvitop-npu -m compact --interval 5
        $ nvitop-npu -u root -p 12345
        $ nvitop-npu --colorful
"""

# pylint: disable=too-many-lines,too-many-branches,too-many-statements,too-many-locals

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from typing import Any, Callable, Iterable

from nvitop.api import libnpu
from nvitop.api.npu_device import NpuDevice, NpuProcess, NpuQueryError, NpuSmiNotFound
from nvitop.api.utils import NA, NaType, colored, set_color, utilization2string
from nvitop.version import __version__


# Constants #########################################################################################

BLOCK = '█'
EMPTY = '░'
BAR_WIDTH = 8

_DEVICE_COLUMNS = [
    ('NPU', 4, 'index'),
    ('Name', 14, 'name'),
    ('Bus-Id', 12, 'bus_id'),
    ('Health', 7, 'health'),
    ('AICore(%)', 10, 'gpu'),
    ('HBM-Usage', 20, 'memory'),
    ('Power(W)', 8, 'power'),
    ('Temp(C)', 7, 'temp'),
    ('AICoreFreq(MHz)', 13, 'clock'),
]

_PROCESS_COLUMNS = [
    ('NPU', 4, 'npu'),
    ('PID', 8, 'pid'),
    ('Type', 5, 'type'),
    ('Process name', 24, 'name'),
    ('Memory', 12, 'memory'),
    ('User', 12, 'user'),
]

_CLEAR_SCREEN = '\x1b[2J\x1b[H'
_HIDE_CURSOR = '\x1b[?25l'
_SHOW_CURSOR = '\x1b[?25h'


def _border(chars: str, widths: list[int], no_unicode: bool) -> str:
    """Build a table border row.

    ``chars`` is a 3-character string of ``(left_corner, tee, right_corner)``;
    e.g. ``'┌┬┐'`` for the top border.  In ASCII mode all joints are rendered
    as ``+`` with ``-`` segments.
    """
    if no_unicode:
        left, tee, right, hline = '+', '+', '+', '-'
    else:
        left, tee, right, hline = chars[0], chars[1], chars[2], '─'
    return left + tee.join(hline * width for width in widths) + right


def _vline(no_unicode: bool) -> str:
    """Return the vertical line character for the given mode."""
    return '|' if no_unicode else '│'


def _fit_columns(
    columns: list[tuple[str, int, str]],
    term_width: int,
    min_keep: int = 5,
) -> list[tuple[str, int, str]]:
    """Shrink the column definitions to fit the terminal width.

    Columns are dropped from the end first (the least important ones), then the
    remaining columns are narrowed proportionally down to ``min_keep``.
    """
    if term_width <= 0:
        return columns

    def total_width(cols: list[tuple[str, int, str]]) -> int:
        return sum(width for _, width, _ in cols) + len(cols) + 1

    cols = list(columns)
    while len(cols) > min_keep and total_width(cols) > term_width:
        cols.pop()
    if total_width(cols) > term_width:
        # Narrow the remaining columns proportionally
        overflow = total_width(cols) - term_width
        shrinkable = [i for i, (_, width, _) in enumerate(cols) if width > min_keep]
        while overflow > 0 and shrinkable:
            for i in shrinkable:
                if overflow <= 0:
                    break
                if cols[i][1] > min_keep:
                    cols[i] = (cols[i][0], cols[i][1] - 1, cols[i][2])
                    overflow -= 1
    return cols


def _format_cell(text: Any, width: int) -> str:
    """Format and truncate a cell to the given width."""
    text = str(text)
    if len(text) > width:
        if width <= 1:
            return text[:width]
        return text[: width - 1] + '…'
    return text.ljust(width)


def _bar(value: int | float | NaType) -> str:
    """Render a simple progress bar of width :data:`BAR_WIDTH`."""
    if isinstance(value, NaType):
        return EMPTY * BAR_WIDTH
    filled = int(round(max(0.0, min(100.0, float(value))) / 100.0 * BAR_WIDTH))
    return BLOCK * filled + EMPTY * (BAR_WIDTH - filled)


def _colorize_bar(value: int | float | NaType, color: str) -> str:
    """Render a colored progress bar."""
    return colored(_bar(value), color)


def _printable(value: Any) -> str:
    """Convert a value to its printable form, mapping ``NA`` to ``'N/A'``."""
    if isinstance(value, NaType):
        return 'N/A'
    return str(value)


# Device / process collection #######################################################################

def collect_processes(
    devices: Iterable[NpuDevice],
    use_cache: bool = True,
) -> list[tuple[NpuProcess, int]]:
    """Collect the running processes of the given devices.

    The process table of the global ``npu-smi info`` query already contains the
    per-process NPU memory, so a single invocation covers every device.
    Processes sharing the same pid are merged, and the NPU memory of a process
    is the sum over all devices it occupies.

    Returns:
        list: A list of ``(process, npu_index)`` tuples.
    """
    indices = {device.index for device in devices}
    data = libnpu.npu_query_global(use_cache=use_cache)

    merged: dict[int, dict[str, Any]] = {}
    for row in data['processes']:
        npu = int(row['npu'])
        if npu not in indices:
            continue
        pid = int(row['pid'])
        memory = (
            int(row['mem']) * 1024 * 1024  # MB -> bytes
            if not isinstance(row['mem'], NaType)
            else 0
        )
        if pid not in merged:
            merged[pid] = {'name': row['name'], 'memory': 0, 'npu': npu}
        merged[pid]['memory'] += memory
        merged[pid]['npu'] = npu  # keep the last device

    return [
        (NpuProcess(pid, used_memory=info['memory'], name=info['name']), info['npu'])
        for pid, info in merged.items()
    ]


def prefetch_clocks(devices: Iterable[NpuDevice], *, use_cache: bool = True) -> None:
    """Warm the per-device clock caches with parallel batches of ``npu-smi`` calls.

    Each card's clock info lives in per-card ``npu-smi info -t common`` /
    ``-t memory`` queries.  Running them concurrently reduces the total latency
    from ``2 * N * 0.5s`` to roughly two round trips; the results are shared
    through the module-level TTL cache so the per-device queries below hit it.
    """
    indices = [device.index for device in devices]
    libnpu.npu_query_kv_batch(indices, 'common', use_cache=use_cache)
    libnpu.npu_query_kv_batch(indices, 'memory', use_cache=use_cache)


def filter_processes(
    processes: list[tuple[NpuProcess, int]],
    *,
    users: set[str] | None = None,
    pids: set[int] | None = None,
) -> list[tuple[NpuProcess, int]]:
    """Filter the collected processes by the given users and pids."""
    ret = []
    for process, npu in processes:
        if pids is not None and process.pid not in pids:
            continue
        if users is not None:
            username = process.username()
            if isinstance(username, NaType) or username not in users:
                continue
        ret.append((process, npu))
    return ret


# Renderers ##########################################################################################

def render_devices_table(
    devices: list[NpuDevice],
    *,
    colorful: bool = False,
    no_unicode: bool = False,
) -> str:
    """Render the device status table (``nvidia-smi``-style)."""
    term_width = shutil.get_terminal_size().columns
    columns = _fit_columns(_DEVICE_COLUMNS, term_width)
    widths = [width for _, width, _ in columns]
    vline = _vline(no_unicode)
    top = _border('┌┬┐', widths, no_unicode)
    header = vline + vline.join(
        _format_cell(colored(name, attrs=('bold',)), width)
        for (name, width, _) in columns
    ) + vline
    sep = _border('├┼┤', widths, no_unicode)
    bottom = _border('└┴┘', widths, no_unicode)

    lines = [top, header, sep]
    for device in devices:
        gpu = device.gpu_utilization()
        memory = device.memory_utilization()
        temperature = device.temperature()
        power = device.power_usage()
        health = device.health()

        cells = {
            'index': str(device.index),
            'name': _printable(device.name()),
            'bus_id': _printable(device.bus_id()),
            'health': colored(_printable(health), device.health_color(health)),
            'gpu': (
                colored(utilization2string(gpu), device.utilization_color(gpu))
                if colorful
                else utilization2string(gpu)
            ),
            'memory': (
                colored(device.memory_usage(), device.memory_color(device.memory_percent()))
                if colorful
                else device.memory_usage()
            ),
            'power': (
                colored(device.power_status(), device.power_color(power))
                if colorful
                else device.power_status()
            ),
            'temp': (
                colored(_printable(temperature), device.temperature_color(temperature))
                if colorful
                else _printable(temperature)
            ),
            'clock': _printable(device.aicore_clock()),
        }
        lines.append(
            vline + vline.join(_format_cell(cells[key], width) for (_, width, key) in columns) + vline,
        )
    lines.append(bottom)
    return '\n'.join(lines)


def render_processes_table(
    processes: list[tuple[NpuProcess, int]],
    *,
    colorful: bool = False,
    no_unicode: bool = False,
) -> str:
    """Render the process table."""
    term_width = shutil.get_terminal_size().columns
    columns = _fit_columns(_PROCESS_COLUMNS, term_width)
    widths = [width for _, width, _ in columns]
    vline = _vline(no_unicode)
    top = _border('┌┬┐', widths, no_unicode)
    header = vline + vline.join(
        _format_cell(colored(name, attrs=('bold',)), width)
        for (name, width, _) in columns
    ) + vline
    sep = _border('├┼┤', widths, no_unicode)
    bottom = _border('└┴┘', widths, no_unicode)

    lines = [top, header, sep]
    for process, npu in processes:
        name = process.name()
        cells = {
            'npu': str(npu),
            'pid': str(process.pid),
            'type': 'C',
            'name': _printable(name),
            'memory': _printable(process.used_memory_human()),
            'user': _printable(process.username()),
        }
        lines.append(
            vline + vline.join(_format_cell(cells[key], width) for (_, width, key) in columns) + vline,
        )
    lines.append(bottom)
    return '\n'.join(lines)


def render_header(
    devices: list[NpuDevice],
    *,
    monitor: bool = False,
    colorful: bool = False,
) -> str:
    """Render the summary header line."""
    driver = NpuDevice.driver_version()
    driver_str = f', driver {driver}' if not isinstance(driver, NaType) else ''
    names = {str(device.name()) for device in devices}
    name_str = ', '.join(sorted(names)) if names else 'Ascend NPU'
    title = 'Ascend NPU Monitor (nvitop-npu)'
    summary = '{} x {} | {} devices{}'.format(
        len(devices),
        name_str,
        len(devices),
        driver_str,
    )
    parts = [title]
    if not monitor:
        parts.append(summary)
    line = '  '.join(parts)
    if colorful:
        line = colored(line, attrs=('bold',))
    return line


# Argument parsing ##################################################################################

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for ``nvitop-npu``."""
    parser = argparse.ArgumentParser(
        prog='nvitop-npu',
        description='An interactive Ascend-NPU process viewer.',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        '--help',
        '-h',
        dest='help',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.',
    )
    parser.add_argument(
        '--version',
        '-V',
        dest='version',
        action='version',
        version=f'%(prog)s {__version__}',
        help="Show %(prog)s's version number and exit.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--once',
        '-1',
        dest='once',
        action='store_true',
        help='Report query data only once.',
    )
    mode.add_argument(
        '--monitor',
        '-m',
        dest='monitor',
        type=str,
        default=argparse.SUPPRESS,
        nargs='?',
        choices=['auto', 'full', 'compact'],
        help=(
            'Run as a resource monitor. Continuously report query data.\n'
            'If the argument is omitted, the value from `NVITOP_MONITOR_MODE` will be used.\n'
            '(default fallback mode: auto)'
        ),
    )

    parser.add_argument(
        '--interval',
        dest='interval',
        type=posfloat,
        default=None,
        metavar='SEC',
        help='Process status update interval in seconds. (default: 2)',
    )
    parser.add_argument(
        '--no-unicode',
        '--ascii',
        '-U',
        dest='no_unicode',
        action='store_true',
        help='Use ASCII characters only, which is useful for terminals without Unicode support.',
    )
    parser.add_argument(
        '--colorful',
        dest='colorful',
        action='store_true',
        help=(
            'Use gradient colors to get spectrum-like bar charts.\n'
            'Set variable `NVITOP_MONITOR_MODE="colorful"` for convenience.'
        ),
    )
    parser.add_argument(
        '--force-color',
        dest='force_color',
        action='store_true',
        help='Force colorize even when `stdout` is not a TTY terminal.',
    )
    parser.add_argument(
        '--light',
        action='store_true',
        help=(
            'Tweak visual results for light theme terminals in monitor mode.\n'
            'Set variable `NVITOP_MONITOR_MODE="light"` on light terminals for convenience.'
        ),
    )

    device_filtering = parser.add_argument_group('device filtering')
    device_filtering.add_argument(
        '--only',
        '-o',
        dest='only',
        type=int,
        nargs='+',
        metavar='INDEX',
        help='Only show the specified devices.',
    )

    process_filtering = parser.add_argument_group('process filtering')
    process_filtering.add_argument(
        '--compute',
        '-c',
        dest='compute',
        action='store_true',
        help='Only show NPU processes with the compute context. (all NPU processes are compute)',
    )
    process_filtering.add_argument(
        '--user',
        '-u',
        dest='user',
        type=str,
        nargs='*',
        metavar='USERNAME',
        help='Only show processes of the given users (or `$USER` for no argument).',
    )
    process_filtering.add_argument(
        '--pid',
        '-p',
        dest='pid',
        type=int,
        nargs='+',
        metavar='PID',
        help='Only show processes of the given PIDs.',
    )

    args = parser.parse_args()

    if args.interval is not None and args.interval < 0.25:
        parser.error(
            f'the interval {args.interval:0.2g}s is too short, which may cause performance issues. '
            f'Expected 1/4 or higher.',
        )

    if not args.colorful:
        args.colorful = 'colorful' in NVITOP_MONITOR_MODE and 'plain' not in NVITOP_MONITOR_MODE
    if not args.light:
        args.light = 'light' in NVITOP_MONITOR_MODE and 'dark' not in NVITOP_MONITOR_MODE
    if hasattr(args, 'monitor') and args.monitor is None:
        modes = NVITOP_MONITOR_MODE.intersection({'auto', 'full', 'compact'})
        args.monitor = 'auto' if len(modes) != 1 else modes.pop()

    return args


def posfloat(argstring: str) -> float:
    """Parse a positive float argument."""
    num = float(argstring)
    if not math.isfinite(num) or num <= 0:
        raise ValueError
    return num


NVITOP_MONITOR_MODE = set(
    map(
        str.strip,
        os.environ.get('NVITOP_MONITOR_MODE', '').lower().split(','),
    ),
)


# Main ##############################################################################################

def main() -> int:
    """The entry point of ``nvitop-npu``."""
    args = parse_arguments()

    if args.force_color:
        set_color(True)

    try:
        libnpu.npu_init()
    except NpuSmiNotFound as ex:
        print(
            '{} {}'.format(colored('NPU ERROR:', color='red', attrs=('bold',)), ex),
            file=sys.stderr,
        )
        print(
            'HINT: This host has no Ascend NPU or the `npu-smi` tool is not installed. '
            'Please run this tool on an Ascend NPU server (e.g. Atlas 800 / 910B).',
            file=sys.stderr,
        )
        return 1
    except NpuQueryError as ex:
        print(
            '{} {}'.format(colored('NPU ERROR:', color='red', attrs=('bold',)), ex),
            file=sys.stderr,
        )
        return 1

    device_count = NpuDevice.count()
    if device_count == 0:
        print(
            '{} {}'.format(
                colored('NPU ERROR:', color='red', attrs=('bold',)),
                'No Ascend NPU device found on this host.',
            ),
            file=sys.stderr,
        )
        return 1

    if args.only is not None:
        indices = set(args.only)
        invalid_indices = indices.difference(range(device_count))
        indices.intersection_update(range(device_count))
        if len(invalid_indices) > 0:
            print(
                '{} {}'.format(
                    colored('ERROR:', color='red', attrs=('bold',)),
                    f'Invalid device indices: {sorted(invalid_indices)}.',
                ),
                file=sys.stderr,
            )
            return 1
    else:
        indices = set(range(device_count))

    try:
        devices = NpuDevice.from_indices(sorted(indices))
    except NpuQueryError as ex:
        print(
            '{} {}'.format(colored('NPU ERROR:', color='red', attrs=('bold',)), ex),
            file=sys.stderr,
        )
        return 1

    users: set[str] | None = None
    if args.user is not None:
        if len(args.user) == 0:
            users = {os.environ.get('USER', 'root')}
        else:
            users = set(args.user)
    pids: set[int] | None = set(args.pid) if args.pid is not None else None

    interval = args.interval if args.interval is not None else 2.0
    monitor = hasattr(args, 'monitor')
    mode = args.monitor if monitor else 'auto'

    try:
        if monitor and sys.stdout.isatty():
            run_monitor(devices, users=users, pids=pids, interval=interval, mode=mode, args=args)
        else:
            run_once(devices, users=users, pids=pids, args=args)
    except KeyboardInterrupt:
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()
    except NpuQueryError as ex:
        print(
            '{} {}'.format(colored('NPU ERROR:', color='red', attrs=('bold',)), ex),
            file=sys.stderr,
        )
        return 1
    finally:
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()
    return 0


def run_once(
    devices: list[NpuDevice],
    *,
    users: set[str] | None = None,
    pids: set[int] | None = None,
    args: argparse.Namespace,
) -> None:
    """Render a single report and print it to ``stdout``."""
    prefetch_clocks(devices)
    processes = filter_processes(collect_processes(devices), users=users, pids=pids)
    print(render_header(devices, colorful=args.colorful))
    print()
    print(render_devices_table(devices, colorful=args.colorful, no_unicode=args.no_unicode))
    if len(processes) > 0:
        print()
        print(render_processes_table(processes, colorful=args.colorful, no_unicode=args.no_unicode))
    else:
        print()
        print(colored('No running processes found.', 'yellow'))
    print()


def run_monitor(
    devices: list[NpuDevice],
    *,
    users: set[str] | None = None,
    pids: set[int] | None = None,
    interval: float,
    mode: str,
    args: argparse.Namespace,
) -> None:
    """Run the monitor loop, refreshing the report every ``interval`` seconds."""
    compact = mode == 'compact'
    sys.stdout.write(_HIDE_CURSOR)
    try:
        while True:
            sys.stdout.write(_CLEAR_SCREEN)
            # Force fresh data every round: the global overview (one invocation)
            # plus one parallel batch for the per-card clock info.
            processes = filter_processes(collect_processes(devices, use_cache=False), users=users, pids=pids)
            prefetch_clocks(devices, use_cache=False)
            print(render_header(devices, monitor=True, colorful=args.colorful))
            print()
            print(render_devices_table(devices, colorful=args.colorful, no_unicode=args.no_unicode))
            if not compact:
                print()
                if len(processes) > 0:
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
            sys.stdout.flush()
            time.sleep(interval)
    finally:
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()


if __name__ == '__main__':
    sys.exit(main())
