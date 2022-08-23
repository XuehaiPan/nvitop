# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

import argparse
import curses
import os
import sys

from nvitop.core import HostProcess, libnvml
from nvitop.gui import USERNAME, Device, Top, colored, libcurses, set_color, setlocale_utf8
from nvitop.version import __version__


TTY = sys.stdin.isatty() and sys.stdout.isatty()
NVITOP_MONITOR_MODE = set(
    map(str.strip, os.environ.get('NVITOP_MONITOR_MODE', '').lower().split(','))
)


def parse_arguments():  # pylint: disable=too-many-branches,too-many-statements

    coloring_rules = '{} < th1 %% <= {} < th2 %% <= {}'.format(
        colored('light', 'green'), colored('moderate', 'yellow'), colored('heavy', 'red')
    )

    def posint(argstring):
        num = int(argstring)
        if num <= 0:
            raise ValueError
        return num

    posint.__name__ = 'positive integer'

    parser = argparse.ArgumentParser(
        prog='nvitop',
        description='An interactive NVIDIA-GPU process viewer.',
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
        version='%(prog)s {}'.format(__version__),
        help="Show %(prog)s's version number and exit.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--once', '-1', dest='once', action='store_true', help='Report query data only once.'
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
            'Run as a resource monitor. Continuously report query data and handle user inputs.\n'
            'If the argument is omitted, the value from `NVITOP_MONITOR_MODE` will be used.\n'
            '(default fallback mode: auto)'
        ),
    )

    parser.add_argument(
        '--interval',
        dest='interval',
        type=posint,
        default=None,
        metavar='SEC',
        help='Process status update interval in seconds. (default: 2)',
    )
    parser.add_argument(
        '--ascii',
        '--no-unicode',
        '-U',
        dest='ascii',
        action='store_true',
        help='Use ASCII characters only, which is useful for terminals without Unicode support.',
    )

    coloring = parser.add_argument_group('coloring')
    coloring.add_argument(
        '--colorful',
        dest='colorful',
        action='store_true',
        help=(
            'Use gradient colors to get spectrum-like bar charts. This option is only available\n'
            'when the terminal supports 256 colors. You may need to set environment variable\n'
            '`TERM="xterm-256color"`. Note that the terminal multiplexer, such as `tmux`, may\n'
            'override the `TREM` variable.'
        ),
    )
    coloring.add_argument(
        '--force-color',
        dest='force_color',
        action='store_true',
        help='Force colorize even when `stdout` is not a TTY terminal.',
    )
    coloring.add_argument(
        '--light',
        action='store_true',
        help=(
            'Tweak visual results for light theme terminals in monitor mode.\n'
            'Set variable `NVITOP_MONITOR_MODE="light"` on light terminals for convenience.'
        ),
    )
    gpu_thresholds = Device.GPU_UTILIZATION_THRESHOLDS
    coloring.add_argument(
        '--gpu-util-thresh',
        type=int,
        nargs=2,
        choices=range(1, 100),
        metavar=('th1', 'th2'),
        help=(
            'Thresholds of GPU utilization to determine the load intensity.\n'
            'Coloring rules: {}.\n'
            '( 1 <= th1 < th2 <= 99, defaults: {} {} )'
        ).format(coloring_rules, *gpu_thresholds),
    )
    memory_thresholds = Device.MEMORY_UTILIZATION_THRESHOLDS
    coloring.add_argument(
        '--mem-util-thresh',
        type=int,
        nargs=2,
        choices=range(1, 100),
        metavar=('th1', 'th2'),
        help=(
            'Thresholds of GPU memory percent to determine the load intensity.\n'
            'Coloring rules: {}.\n'
            '( 1 <= th1 < th2 <= 99, defaults: {} {} )'
        ).format(coloring_rules, *memory_thresholds),
    )

    device_filtering = parser.add_argument_group('device filtering')
    device_filtering.add_argument(
        '--only',
        '-o',
        dest='only',
        type=int,
        nargs='+',
        metavar='idx',
        help='Only show the specified devices, suppress option `--only-visible`.',
    )
    device_filtering.add_argument(
        '--only-visible',
        '-ov',
        dest='only_visible',
        action='store_true',
        help='Only show devices in the `CUDA_VISIBLE_DEVICES` environment variable.',
    )

    process_filtering = parser.add_argument_group('process filtering')
    process_filtering.add_argument(
        '--compute',
        '-c',
        dest='compute',
        action='store_true',
        help="Only show GPU processes with the compute context. (type: 'C' or 'C+G')",
    )
    process_filtering.add_argument(
        '--no-compute',
        '-C',
        dest='no_compute',
        action='store_true',
        help="Exclude GPU processes with the compute context. (type: 'G' only)",
    )
    process_filtering.add_argument(
        '--graphics',
        '-g',
        dest='graphics',
        action='store_true',
        help="Only show GPU processes with the graphics context. (type: 'G' or 'C+G')",
    )
    process_filtering.add_argument(
        '--no-graphics',
        '-G',
        dest='no_graphics',
        action='store_true',
        help="Exclude GPU processes with the graphics context. (type: 'C' only)",
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

    if not args.colorful:
        args.colorful = 'colorful' in NVITOP_MONITOR_MODE and 'plain' not in NVITOP_MONITOR_MODE
    if not args.light:
        args.light = 'light' in NVITOP_MONITOR_MODE and 'dark' not in NVITOP_MONITOR_MODE
    if args.user is not None and len(args.user) == 0:
        args.user.append(USERNAME)
    if args.gpu_util_thresh is None:
        try:
            gpu_util_thresh = os.getenv('NVITOP_GPU_UTILIZATION_THRESHOLDS', None)
            gpu_util_thresh = list(map(int, gpu_util_thresh.split(',')))[:2]
            if (
                len(gpu_util_thresh) != 2
                or min(gpu_util_thresh) <= 0
                or max(gpu_util_thresh) >= 100
            ):
                raise ValueError
        except (ValueError, AttributeError):
            pass
        else:
            args.gpu_util_thresh = gpu_util_thresh
    if args.mem_util_thresh is None:
        try:
            mem_util_thresh = os.getenv('NVITOP_MEMORY_UTILIZATION_THRESHOLDS', None)
            mem_util_thresh = list(map(int, mem_util_thresh.split(',')))[:2]
            if (
                len(mem_util_thresh) != 2
                or min(mem_util_thresh) <= 0
                or max(mem_util_thresh) >= 100
            ):
                raise ValueError
        except (ValueError, AttributeError):
            pass
        else:
            args.mem_util_thresh = mem_util_thresh

    return args


def main():  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    args = parse_arguments()

    if args.force_color:
        set_color(True)

    messages = []
    if args.once and hasattr(args, 'monitor'):
        messages.append('ERROR: Both `--once` and `--monitor` switches are on.')
        del args.monitor

    if not args.once and not hasattr(args, 'monitor') and TTY:
        args.monitor = None

    if hasattr(args, 'monitor') and not TTY:
        messages.append('ERROR: You must run monitor mode from a TTY terminal.')
        del args.monitor

    if hasattr(args, 'monitor') and args.monitor is None:
        mode = NVITOP_MONITOR_MODE.intersection({'auto', 'full', 'compact'})
        if len(mode) != 1:
            mode = 'auto'
        else:
            mode = mode.pop()
        args.monitor = mode

    if not setlocale_utf8():
        args.ascii = True

    try:
        device_count = Device.count()
    except libnvml.NVMLError_LibraryNotFound:
        return 1
    except libnvml.NVMLError as ex:
        print(
            '{} {}'.format(colored('NVML ERROR:', color='red', attrs=('bold',)), ex),
            file=sys.stderr,
        )
        return 1

    if args.gpu_util_thresh is not None:
        Device.GPU_UTILIZATION_THRESHOLDS = tuple(sorted(args.gpu_util_thresh))
    if args.mem_util_thresh is not None:
        Device.MEMORY_UTILIZATION_THRESHOLDS = tuple(sorted(args.mem_util_thresh))

    if args.only is not None:
        indices = set(args.only)
        invalid_indices = indices.difference(range(device_count))
        indices.intersection_update(range(device_count))
        if len(invalid_indices) > 1:
            messages.append('ERROR: Invalid device indices: {}.'.format(sorted(invalid_indices)))
        elif len(invalid_indices) == 1:
            messages.append('ERROR: Invalid device index: {}.'.format(list(invalid_indices)[0]))
    elif args.only_visible:
        indices = set(
            index if isinstance(index, int) else index[0]
            for index in Device.parse_cuda_visible_devices()
        )
    else:
        indices = range(device_count)
    devices = Device.from_indices(sorted(set(indices)))

    filters = []
    if args.compute:
        filters.append(lambda process: 'C' in process.type)
    if args.no_compute:
        filters.append(lambda process: 'C' not in process.type)
    if args.graphics:
        filters.append(lambda process: 'G' in process.type)
    if args.no_graphics:
        filters.append(lambda process: 'G' not in process.type)
    if args.user is not None:
        users = set(args.user)
        filters.append(lambda process: process.username in users)
    if args.pid is not None:
        pids = set(args.pid)
        filters.append(lambda process: process.pid in pids)

    top = None
    if hasattr(args, 'monitor') and len(devices) > 0:
        try:
            with libcurses(colorful=args.colorful, light_theme=args.light) as win:
                top = Top(
                    devices,
                    filters,
                    ascii=args.ascii,
                    mode=args.monitor,
                    interval=args.interval,
                    win=win,
                )
                top.loop()
        except curses.error as ex:
            if top is not None:
                raise
            messages.append('ERROR: Failed to initialize `curses` ({})'.format(ex))

    if top is None:
        top = Top(devices, filters, ascii=args.ascii)
        if not sys.stdout.isatty():
            parent = HostProcess().parent()
            grandparent = parent.parent() if parent is not None else None
            if grandparent is not None and parent.name() == 'sh' and grandparent.name() == 'watch':
                print(
                    'HINT: You are running `nvitop` under `watch` command. Please try `nvitop -m` directly.',
                    file=sys.stderr,
                )

    top.print()
    top.destroy()

    if len(libnvml.UNKNOWN_FUNCTIONS) > 0:
        unknown_function_messages = [
            'ERROR: Some FunctionNotFound errors occurred while calling:'
            if len(libnvml.UNKNOWN_FUNCTIONS) > 1
            else 'ERROR: A FunctionNotFound error occurred while calling:'
        ]
        unknown_function_messages.extend(
            '    nvmlQuery({.__name__!r}, *args, **kwargs)'.format(func)
            for func, _ in libnvml.UNKNOWN_FUNCTIONS.values()
        )
        unknown_function_messages.append(
            (
                'Please verify whether the `{0}` package is compatible with your NVIDIA driver version.\n'
                'You can check the release history of `{0}` and install the compatible version manually.\n'
                'See {1} for more information.'
            ).format(
                colored('nvidia-ml-py', attrs=('bold',)),
                colored('https://github.com/XuehaiPan/nvitop#installation', attrs=('underline',)),
            )
        )
        message = '\n'.join(unknown_function_messages)
        if (
            'nvmlDeviceGetComputeRunningProcesses' in message
            or 'nvmlDeviceGetGraphicsRunningProcesses' in message
        ) and Device.cuda_driver_version().startswith('10.'):
            message = '\n'.join(
                (
                    message,
                    '',
                    'You are using CUDA 10.x driver (yours is: @VERSION@) which is too old. Please contact',
                    'your system admin to update the NVIDIA driver, or reinstall `nvitop` using:',
                    '    pip3 install "nvitop[cuda10]"',
                )
            ).replace('@VERSION@', Device.driver_version())
        messages.append(message)

    if len(messages) > 0:
        for message in messages:
            if message.startswith('ERROR:'):
                message = message.replace(
                    'ERROR:', colored('ERROR:', color='red', attrs=('bold',)), 1
                )
            print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
