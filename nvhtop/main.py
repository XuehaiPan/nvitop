# This file is part of nvhtop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

import argparse
import curses
import sys
import time
from contextlib import contextmanager

import pynvml as nvml
from .displayable import DisplayableContainer
from .monitor import Device
from .panel import DevicePanel, ProcessPanel


@contextmanager
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
    def __init__(self, mode='auto', win=None):
        super(Top, self).__init__(win)

        assert mode in ('auto', 'full', 'compact')
        compact = (mode == 'compact')
        self.mode = mode
        self._compact = compact

        self.device_count = nvml.nvmlDeviceGetCount()
        self.devices = list(map(Device, range(self.device_count)))

        self.win = win
        self.termsize = None

        self.device_panel = DevicePanel(self.devices, compact, win=win)
        self.process_panel = ProcessPanel(self.devices, win=win)
        self.process_panel.y = self.device_panel.height + 1
        self.add_child(self.device_panel)
        self.add_child(self.process_panel)

        self.height = self.device_panel.height + 1 + self.process_panel.height

    @property
    def compact(self):
        return self._compact

    @compact.setter
    def compact(self, value):
        if self._compact != value:
            self.need_redraw = True
            self._compact = value

    def poke(self):
        super(Top, self).poke()

        n_term_lines, _ = self.win.getmaxyx()
        if self.mode == 'auto':
            self.compact = (n_term_lines < 4 + 3 * (self.device_count + 1) + 1 + self.process_panel.height)
            self.device_panel.compact = self.compact
            self.process_panel.y = self.device_panel.y + self.device_panel.height + 1
        self.height = self.device_panel.height + 1 + self.process_panel.height

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
    try:
        nvml.nvmlInit()
    except nvml.NVMLError_LibraryNotFound as error:  # pylint: disable=no-member
        print(error, file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(prog='nvhtop', description='A interactive Nvidia-GPU process viewer.')
    parser.add_argument('-m', '--monitor', type=str, default='notpresented',
                        nargs='?', choices=['auto', 'full', 'compact'],
                        help='Run as a resource monitor. '
                             'Continuously report query data, rather than the default of just once. '
                             'If no argument is specified, the default mode `auto` is used.')
    parser.add_argument('--mem-util-thresh', type=int, nargs=2, choices=range(1, 100), metavar='th',
                        help='Thresholds of GPU memory utilization to distinguish load intensity, '
                        '(1 <= th <= 99, defaults: {} {})'.format(Device.MEMORY_UTILIZATION_THRESHOLD_LIGHT,
                                                                  Device.MEMORY_UTILIZATION_THRESHOLD_MODERATE))
    parser.add_argument('--gpu-util-thresh', type=int, nargs=2, choices=range(1, 100), metavar='th',
                        help='Thresholds of GPU utilization to distinguish load intensity, '
                        '(1 <= th <= 99, defaults: {} {})'.format(Device.GPU_UTILIZATION_THRESHOLD_LIGHT,
                                                                  Device.GPU_UTILIZATION_THRESHOLD_MODERATE))
    args = parser.parse_args()
    if args.monitor is None:
        args.monitor = 'auto'
    if args.mem_util_thresh is not None:
        Device.MEMORY_UTILIZATION_THRESHOLD_LIGHT = min(args.mem_util_thresh)
        Device.MEMORY_UTILIZATION_THRESHOLD_MODERATE = max(args.mem_util_thresh)
    if args.gpu_util_thresh is not None:
        Device.GPU_UTILIZATION_THRESHOLD_LIGHT = min(args.gpu_util_thresh)
        Device.GPU_UTILIZATION_THRESHOLD_MODERATE = max(args.gpu_util_thresh)


    if args.monitor != 'notpresented':
        with libcurses() as win:
            top = Top(mode=args.monitor, win=win)
            top.loop()
    else:
        top = Top()
    top.print()

    nvml.nvmlShutdown()
