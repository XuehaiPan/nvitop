# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import argparse
import os
import sys

import pynvml as nvml

from .device import Device
from .ui import libcurses, Top
from .utils import colored


def parse_arguments():
    coloring_rules = '{} < th1 %% <= {} < th2 %% <= {}'.format(colored('light', 'green'),
                                                               colored('moderate', 'yellow'),
                                                               colored('heavy', 'red'))
    parser = argparse.ArgumentParser(prog='nvitop', description='A interactive NVIDIA-GPU process viewer.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--monitor', type=str, default='notpresented',
                        nargs='?', choices=['auto', 'full', 'compact'],
                        help='Run as a resource monitor. Continuously report query data,\n' +
                             'rather than the default of just once.\n' +
                             'If no argument is given, the default mode `auto` is used.')
    parser.add_argument('-o', '--only', type=int, nargs='+', metavar='idx',
                        help='Only show the specified devices, suppress option `-ov`.')
    parser.add_argument('-ov', '--only-visible', action='store_true',
                        help='Only show devices in environment variable `CUDA_VISIBLE_DEVICES`.')
    parser.add_argument('--gpu-util-thresh', type=int, nargs=2, choices=range(1, 100), metavar=('th1', 'th2'),
                        help='Thresholds of GPU utilization to distinguish load intensity.\n' +
                             'Coloring rules: {}.\n'.format(coloring_rules) +
                             '( 1 <= th1 < th2 <= 99, defaults: {} {} )'.format(*Device.GPU_UTILIZATION_THRESHOLDS))
    parser.add_argument('--mem-util-thresh', type=int, nargs=2,
                        choices=range(1, 100), metavar=('th1', 'th2'),
                        help='Thresholds of GPU memory utilization to distinguish load intensity.\n' +
                             'Coloring rules: {}.\n'.format(coloring_rules) +
                             '( 1 <= th1 < th2 <= 99, defaults: {} {} )'.format(*Device.MEMORY_UTILIZATION_THRESHOLDS))
    args = parser.parse_args()
    if args.monitor is None:
        args.monitor = 'auto'
    if args.gpu_util_thresh is not None:
        Device.GPU_UTILIZATION_THRESHOLDS = tuple(sorted(args.gpu_util_thresh))
    if args.mem_util_thresh is not None:
        Device.MEMORY_UTILIZATION_THRESHOLDS = tuple(sorted(args.mem_util_thresh))

    return args


def main():
    args = parse_arguments()

    if args.monitor != 'notpresented' and not (sys.stdin.isatty() and sys.stdout.isatty()):
        print('ERROR: Must run nvitop monitor mode from terminal.', file=sys.stderr)
        return 1

    try:
        nvml.nvmlInit()
    except nvml.NVMLError_LibraryNotFound:  # pylint: disable=no-member
        print('ERROR: NVIDIA Management Library (NVML) not found.\n'
              'HINT: The NVIDIA Management Library ships with the NVIDIA display driver (available at\n'
              '      https://www.nvidia.com/Download/index.aspx), or can be downloaded as part of the\n'
              '      NVIDIA CUDA Toolkit (available at https://developer.nvidia.com/cuda-downloads).\n'
              '      The lists of OS platforms and NVIDIA-GPUs supported by the NVML library can be\n'
              '      found in the NVML API Reference at https://docs.nvidia.com/deploy/nvml-api.',
              file=sys.stderr)
        return 1

    device_count = nvml.nvmlDeviceGetCount()
    if args.only is not None:
        visible_devices = set(args.only)
    elif args.only_visible:
        try:
            visible_devices = map(str.strip, os.getenv('CUDA_VISIBLE_DEVICES').split(','))
            visible_devices = set(map(int, filter(str.isnumeric, visible_devices)))
        except (ValueError, AttributeError):
            visible_devices = set(range(device_count))
    else:
        visible_devices = set(range(device_count))
    devices = list(map(Device, sorted(set(range(device_count)).intersection(visible_devices))))

    if args.monitor != 'notpresented' and len(devices) > 0:
        with libcurses() as win:
            top = Top(devices, mode=args.monitor, win=win)
            top.loop()
    else:
        top = Top(devices)
    top.print()
    top.destroy()

    nvml.nvmlShutdown()


if __name__ == '__main__':
    sys.exit(main())
