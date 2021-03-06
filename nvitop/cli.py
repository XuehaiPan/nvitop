# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

import argparse
import locale
import sys

from nvitop.core import nvml, Device
from nvitop.gui import Top, libcurses, colored
from nvitop.version import __version__


def parse_arguments():
    coloring_rules = '{} < th1 %% <= {} < th2 %% <= {}'.format(colored('light', 'green'),
                                                               colored('moderate', 'yellow'),
                                                               colored('heavy', 'red'))
    parser = argparse.ArgumentParser(prog='nvitop', description='An interactive NVIDIA-GPU process viewer.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))
    parser.add_argument('--monitor', '-m', dest='monitor', type=str, default=argparse.SUPPRESS,
                        nargs='?', choices=['auto', 'full', 'compact'],
                        help='Run as a resource monitor. Continuously report query data,\n'
                             'rather than the default of just once.\n'
                             'If no argument is given, the default mode `auto` is used.')
    parser.add_argument('--only', '-o', dest='only', type=int, nargs='+', metavar='idx',
                        help='Only show the specified devices, suppress option `--only-visible`.')
    parser.add_argument('--only-visible', '-ov', dest='only_visible', action='store_true',
                        help='Only show devices in environment variable `CUDA_VISIBLE_DEVICES`.')
    parser.add_argument('--compute', '-c', dest='compute', action='store_true',
                        help="Only show GPU processes with the compute context. (type: 'C' or 'C+G')")
    parser.add_argument('--graphics', '-g', dest='graphics', action='store_true',
                        help="Only show GPU processes with the graphics context. (type: 'G' or 'C+G')")
    parser.add_argument('--gpu-util-thresh', type=int, nargs=2, choices=range(1, 100), metavar=('th1', 'th2'),
                        help='Thresholds of GPU utilization to determine the load intensity.\n' +
                             'Coloring rules: {}.\n'.format(coloring_rules) +
                             '( 1 <= th1 < th2 <= 99, defaults: {} {} )'.format(*Device.GPU_UTILIZATION_THRESHOLDS))
    parser.add_argument('--mem-util-thresh', type=int, nargs=2,
                        choices=range(1, 100), metavar=('th1', 'th2'),
                        help='Thresholds of GPU memory utilization to determine the load intensity.\n' +
                             'Coloring rules: {}.\n'.format(coloring_rules) +
                             '( 1 <= th1 < th2 <= 99, defaults: {} {} )'.format(*Device.MEMORY_UTILIZATION_THRESHOLDS))
    parser.add_argument('--ascii', '--no-unicode', '-U', dest='ascii', action='store_true',
                        help='Use ASCII characters only, which is useful for terminals without Unicode support.')
    args = parser.parse_args()
    if hasattr(args, 'monitor') and args.monitor is None:
        args.monitor = 'auto'
    if args.gpu_util_thresh is not None:
        Device.GPU_UTILIZATION_THRESHOLDS = tuple(sorted(args.gpu_util_thresh))
    if args.mem_util_thresh is not None:
        Device.MEMORY_UTILIZATION_THRESHOLDS = tuple(sorted(args.mem_util_thresh))

    return args


def main():  # pylint: disable=too-many-branches
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, '')
        except locale.Error:
            pass

    args = parse_arguments()

    messages = []
    if hasattr(args, 'monitor') and not (sys.stdin.isatty() and sys.stdout.isatty()):
        messages.append('ERROR: You must run monitor mode from a terminal.')
        del args.monitor

    try:
        Device.count()
    except nvml.NVMLError_LibraryNotFound:
        return 1
    except nvml.NVMLError as e:  # pylint: disable=invalid-name
        print('{} {}'.format(colored('NVML ERROR:', color='red', attrs=('bold',)), e), file=sys.stderr)
        return 1

    if args.only is not None:
        devices = Device.from_indices(set(args.only))
    elif args.only_visible:
        devices = Device.from_cuda_visible_devices()
    else:
        devices = Device.all()
    devices.sort(key=lambda device: device.index)

    filters = []
    if args.compute:
        filters.append(lambda process: 'C' in process.type)
    if args.graphics:
        filters.append(lambda process: 'G' in process.type)

    if hasattr(args, 'monitor') and len(devices) > 0:
        with libcurses() as win:
            top = Top(devices, filters, ascii=args.ascii, mode=args.monitor, win=win)
            top.loop()
    else:
        top = Top(devices, filters, ascii=args.ascii)
    top.print()
    top.destroy()

    if len(nvml.UNKNOWN_FUNCTIONS) > 0:
        messages.append('ERROR: A FunctionNotFound error occurred while calling:')
        if len(nvml.UNKNOWN_FUNCTIONS) > 1:
            messages[-1] = messages[-1].replace('A FunctionNotFound error', 'Some FunctionNotFound errors')
        messages.extend([
            *list(map('    nvmlQuery({.__name__!r}, *args, **kwargs)'.format, nvml.UNKNOWN_FUNCTIONS)),
            ('Please verify whether the {0} package is compatible with your NVIDIA driver version.\n'
             'You can check the release history of {0} and install the compatible version manually.\n'
             'See {1} for more information.').format(
                colored('nvidia-ml-py', attrs=('bold',)),
                colored('https://github.com/XuehaiPan/nvitop#installation', attrs=('underline',))
            )
        ])
    if len(messages) > 0:
        for message in messages:
            if message.startswith('ERROR:'):
                message = message.replace('ERROR:', colored('ERROR:', color='red', attrs=('bold',)), 1)
            print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
