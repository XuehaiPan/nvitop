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

        $ nvitop-npu          # interactive monitor in a terminal
        $ nvitop-npu -1       # report query data only once
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
import sys

from nvitop.api import libnpu
from nvitop.api.npu_device import NpuDevice, NpuQueryError, NpuSmiNotFound
from nvitop.api.utils import colored, set_color
from nvitop.npu_monitor import (
    PROCESS_SORTS,
    collect_processes,
    filter_processes,
    prefetch_clocks,
    run_monitor,
    sort_processes,
)
from nvitop.npu_ui import (
    render_devices_table,
    render_header,
    render_processes_table,
    render_summary,
)
from nvitop.version import __version__


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
        '--sort',
        choices=PROCESS_SORTS,
        default='memory',
        help='Sort processes by the selected field. (default: memory)',
    )
    parser.add_argument(
        '--no-processes',
        action='store_true',
        help='Hide the process table.',
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


def _should_monitor(args: argparse.Namespace, *, is_tty: bool) -> bool:
    """Return whether the CLI should start the interactive monitor."""
    return hasattr(args, 'monitor') or (is_tty and not args.once)


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
        users = {os.environ.get('USER', 'root')} if len(args.user) == 0 else set(args.user)
    pids: set[int] | None = set(args.pid) if args.pid is not None else None

    interval = args.interval if args.interval is not None else 2.0
    monitor = _should_monitor(args, is_tty=sys.stdout.isatty())
    mode = getattr(args, 'monitor', 'auto')

    try:
        if monitor and sys.stdout.isatty():
            run_monitor(devices, users=users, pids=pids, interval=interval, mode=mode, args=args)
        else:
            run_once(devices, users=users, pids=pids, args=args)
    except KeyboardInterrupt:
        pass
    except NpuQueryError as ex:
        print(
            '{} {}'.format(colored('NPU ERROR:', color='red', attrs=('bold',)), ex),
            file=sys.stderr,
        )
        return 1
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
    processes = sort_processes(
        filter_processes(collect_processes(devices), users=users, pids=pids),
        getattr(args, 'sort', 'memory'),
    )
    print(render_header(devices, colorful=args.colorful, no_unicode=args.no_unicode))
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
    if not getattr(args, 'no_processes', False) and len(processes) > 0:
        print()
        print(render_processes_table(processes, colorful=args.colorful, no_unicode=args.no_unicode))
    elif not getattr(args, 'no_processes', False):
        print()
        print(colored('No running processes found.', 'yellow'))
    print()


if __name__ == '__main__':
    sys.exit(main())
