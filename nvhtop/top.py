# This file is part of nvhtop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

import curses
import time

from .displayable import DisplayableContainer
from .panel import DevicePanel, ProcessPanel


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

    def print(self):
        self.device_panel.print()
        print()
        self.process_panel.print()
