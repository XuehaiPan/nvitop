# This file is part of nvhtop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

import argparse
import contextlib
import curses
import os
import sys
import time

import pynvml as nvml
from .displayable import DisplayableContainer
from .monitor import Device
from .panel import colored, DevicePanel, ProcessPanel


@contextlib.contextmanager
def libcurses():
    win = curses.initscr()
    win.nodelay(True)
    curses.noecho()
    curses.cbreak()
    curses.curs_set(False)

    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass

    try:
        yield win
    finally:
        curses.endwin()


class Top(DisplayableContainer):
    def __init__(self, devices, mode='auto', win=None):
        super(Top, self).__init__(win)

        assert mode in ('auto', 'full', 'compact')
        compact = (mode == 'compact')
        self.mode = mode
        self._compact = compact

        self.devices = devices
        self.device_count = len(self.devices)

        self.win = win
        self.termsize = None

        self.device_panel = DevicePanel(self.devices, compact, win=win)
        self.process_panel = ProcessPanel(self.devices, win=win)
        self.process_panel.y = self.device_panel.height + 1
        self.add_child(self.device_panel)
        self.add_child(self.process_panel)

        self.height = self.device_panel.height + 1 + self.process_panel.height

        self.update_size()

    @property
    def compact(self):
        return self._compact

    @compact.setter
    def compact(self, value):
        if self._compact != value:
            self.need_redraw = True
            self._compact = value

    def update_size(self):
        n_term_lines, _ = termsize = self.win.getmaxyx()
        if self.mode == 'auto':
            self.compact = (n_term_lines < self.device_panel.full_height + 1 + self.process_panel.height)
            self.device_panel.compact = self.compact
            self.process_panel.y = self.device_panel.y + self.device_panel.height + 1
        self.height = self.device_panel.height + 1 + self.process_panel.height
        if self.termsize != termsize:
            self.termsize = termsize
            self.need_redraw = True

    def poke(self):
        super(Top, self).poke()

        self.update_size()

    def draw(self):
        if self.need_redraw:
            self.win.erase()
        super(Top, self).draw()

    def finalize(self):
        DisplayableContainer.finalize(self)
        self.win.refresh()

    def loop(self):
        if self.win is None:
            return

        key = -1
        while True:
            try:
                self.poke()
                self.draw()
                self.finalize()
                for i in range(10):
                    key = self.win.getch()
                    if key == -1 or key == ord('q'):
                        break
                curses.flushinp()
                if key == ord('q'):
                    break
                time.sleep(0.5)
            except KeyboardInterrupt:
                pass

    def print(self):
        self.device_panel.print()
        print()
        self.process_panel.print()


def main():
    coloring_rules = '{} < th1 %% <= {} < th2 %% <= {}'.format(colored('light', 'green'),
                                                               colored('moderate', 'yellow'),
                                                               colored('heavy', 'red'))
    parser = argparse.ArgumentParser(prog='nvhtop', description='A interactive Nvidia-GPU process viewer.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--monitor', type=str, default='notpresented',
                        nargs='?', choices=['auto', 'full', 'compact'],
                        help='Run as a resource monitor. Continuously report query data,\n' +
                             'rather than the default of just once.\n' +
                             'If no argument is specified, the default mode `auto` is used.')
    parser.add_argument('--gpu-util-thresh', type=int, nargs=2, choices=range(1, 100), metavar=('th1', 'th2'),
                        help='Thresholds of GPU utilization to distinguish load intensity.\n' +
                             'Coloring rules: {}.\n'.format(coloring_rules) +
                             '( 1 <= th1 < th2 <= 99, defaults: {} {} )'.format(*Device.GPU_UTILIZATION_THRESHOLDS))
    parser.add_argument('--mem-util-thresh', type=int, nargs=2,
                        choices=range(1, 100), metavar=('th1', 'th2'),
                        help='Thresholds of GPU memory utilization to distinguish load intensity.\n' +
                             'Coloring rules: {}.\n'.format(coloring_rules) +
                             '( 1 <= th1 < th2 <= 99, defaults: {} {} )'.format(*Device.MEMORY_UTILIZATION_THRESHOLDS))
    parser.add_argument('--visible-devices-only', action='store_true',
                        help='Only show devices in environment variable `CUDA_VISIBLE_DEVICES`')
    args = parser.parse_args()
    if args.monitor is None:
        args.monitor = 'auto'
    if args.monitor != 'notpresented' and not (sys.stdin.isatty() and sys.stdout.isatty()):
        print('Error: Must run nvhtop monitor mode from terminal', file=sys.stderr)
        return 1
    if args.gpu_util_thresh is not None:
        Device.GPU_UTILIZATION_THRESHOLDS = tuple(sorted(args.gpu_util_thresh))
    if args.mem_util_thresh is not None:
        Device.MEMORY_UTILIZATION_THRESHOLDS = tuple(sorted(args.mem_util_thresh))

    try:
        nvml.nvmlInit()
    except nvml.NVMLError_LibraryNotFound as error:  # pylint: disable=no-member
        print('Error: {}'.format(error), file=sys.stderr)
        return 1

    device_count = nvml.nvmlDeviceGetCount()
    if args.visible_devices_only:
        try:
            cuda_visible_devices = set(map(int, filter(lambda s: s != '' and not s.isspace(),
                                                       os.getenv('CUDA_VISIBLE_DEVICES').split(','))))
        except (ValueError, AttributeError):
            cuda_visible_devices = set(range(device_count))
        cuda_visible_devices.intersection_update(range(device_count))
    else:
        cuda_visible_devices = set(range(device_count))
    devices = list(map(Device, sorted(cuda_visible_devices)))

    if args.monitor != 'notpresented':
        with libcurses() as win:
            top = Top(devices, mode=args.monitor, win=win)
            top.loop()
    else:
        top = Top(devices)
    top.print()

    nvml.nvmlShutdown()
