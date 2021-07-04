# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import curses
import threading
from functools import partial

from nvitop.gui.library import DisplayableContainer
from nvitop.gui.screens.main.device import DevicePanel
from nvitop.gui.screens.main.host import HostPanel
from nvitop.gui.screens.main.process import ProcessPanel


class BreakLoop(Exception):
    pass


class MainScreen(DisplayableContainer):
    NAME = 'main'

    def __init__(self, devices, filters, ascii, mode, win, root):  # pylint: disable=redefined-builtin
        super().__init__(win, root)

        self.width = root.width

        assert mode in ('auto', 'full', 'compact')
        compact = (mode == 'compact')
        self.mode = mode
        self._compact = compact

        self.devices = devices
        self.device_count = len(self.devices)

        self.lock = threading.RLock()

        self.device_panel = DevicePanel(self.devices, compact, win=win, root=self)
        self.device_panel.focused = False
        self.add_child(self.device_panel)

        self.host_panel = HostPanel(self.devices, compact, win=win, root=self)
        self.host_panel.focused = False
        self.add_child(self.host_panel)

        self.process_panel = ProcessPanel(self.devices, compact, filters, win=win, root=self)
        self.process_panel.focused = False
        self.add_child(self.process_panel)

        self.selected = self.process_panel.selected

        self.ascii = ascii
        self.device_panel.ascii = self.ascii
        self.host_panel.ascii = self.ascii
        self.process_panel.ascii = self.ascii
        if ascii:
            self.host_panel.full_height = self.host_panel.height = self.host_panel.compact_height

        self.x, self.y = root.x, root.y
        self.device_panel.x = self.host_panel.x = self.process_panel.x = self.x
        self.device_panel.y = self.y
        self.host_panel.y = self.device_panel.y + self.device_panel.height
        self.process_panel.y = self.host_panel.y + self.host_panel.height
        self.height = self.device_panel.height + self.host_panel.height + self.process_panel.height

    @property
    def compact(self):
        return self._compact

    @compact.setter
    def compact(self, value):
        if self._compact != value:
            self.need_redraw = True
            self._compact = value

    def update_size(self, termsize=None):
        if termsize is None:
            self.update_lines_cols()  # pylint: disable=no-member
            termsize = self.win.getmaxyx()
        n_term_lines, n_term_cols = termsize

        self.width = n_term_cols - self.x
        self.device_panel.width = self.width
        self.host_panel.width = self.width
        self.process_panel.width = self.width

        height = n_term_lines - self.y
        heights = [
            self.device_panel.full_height + self.host_panel.full_height + self.process_panel.full_height,
            self.device_panel.compact_height + self.host_panel.full_height + self.process_panel.full_height,
            self.device_panel.compact_height + self.host_panel.compact_height + self.process_panel.full_height,
        ]
        if self.mode == 'auto':
            self.compact = (height < heights[0])
            self.host_panel.compact = (height < heights[1])
            self.process_panel.compact = (height < heights[-1])
        else:
            self.compact = (self.mode == 'compact')
            self.host_panel.compact = self.compact
            self.process_panel.compact = self.compact
        self.device_panel.compact = self.compact

        self.host_panel.y = self.device_panel.y + self.device_panel.height
        self.process_panel.y = self.host_panel.y + self.host_panel.height
        height = self.device_panel.height + self.host_panel.height + self.process_panel.height

        if self.height != height:
            self.height = height
            self.need_redraw = True

    def poke(self):
        super().poke()

        height = self.device_panel.height + self.host_panel.height + self.process_panel.height
        if self.height != height:
            self.update_size()
            self.need_redraw = True

    def draw(self):
        self.color_reset()

        super().draw()

    def print(self):
        if self.device_count > 0:
            print_width = min(map(lambda panel: panel.print_width(), self.container))
            self.width = max(print_width, min(self.width, 100))
        else:
            self.width = 79
        for panel in self.container:
            panel.width = self.width
            panel.print()

    def init_keybindings(self):
        # pylint: disable=multiple-statements

        def quit(top): raise BreakLoop  # pylint: disable=redefined-builtin

        def change_mode(top, mode):
            top.main_screen.mode = mode
            top.update_size()

        def force_refresh(top):
            top.update_size()
            top.need_redraw = True

        def host_left(top): top.main_screen.process_panel.host_offset -= 2
        def host_right(top): top.main_screen.process_panel.host_offset += 2
        def host_begin(top): top.main_screen.process_panel.host_offset = -1
        def host_end(top): top.main_screen.process_panel.host_offset = 1024

        def select_move(top, direction): top.main_screen.selected.move(direction=direction)
        def select_clear(top): top.main_screen.selected.clear()

        def terminate(top): top.main_screen.selected.terminate()
        def kill(top): top.main_screen.selected.kill()
        def interrupt(top): top.main_screen.selected.interrupt()

        def sort_by(top, order, reverse):
            top.main_screen.process_panel.order = order
            top.main_screen.process_panel.reverse = reverse
            top.update_size()

        def order_previous(top):
            sort_by(top,
                    order=ProcessPanel.ORDERS[top.main_screen.process_panel.order].previous,
                    reverse=False)

        def order_next(top):
            sort_by(top,
                    order=ProcessPanel.ORDERS[top.main_screen.process_panel.order].next,
                    reverse=False)

        def order_reverse(top):
            sort_by(top,
                    order=top.main_screen.process_panel.order,
                    reverse=(not top.main_screen.process_panel.reverse))

        self.root.keymaps.bind('main', 'q', quit)
        self.root.keymaps.copy('main', 'q', 'Q')
        self.root.keymaps.bind('main', 'a', partial(change_mode, mode='auto'))
        self.root.keymaps.bind('main', 'f', partial(change_mode, mode='full'))
        self.root.keymaps.bind('main', 'c', partial(change_mode, mode='compact'))
        self.root.keymaps.bind('main', 'r', force_refresh)
        self.root.keymaps.copy('main', 'r', 'R')
        self.root.keymaps.copy('main', 'r', '<C-r>')
        self.root.keymaps.copy('main', 'r', '<F5>')

        self.root.keymaps.bind('main', '<Left>', host_left)
        self.root.keymaps.copy('main', '<Left>', '[')
        self.root.keymaps.copy('main', '<Left>', '<A-h>')
        self.root.keymaps.bind('main', '<Right>', host_right)
        self.root.keymaps.copy('main', '<Right>', ']')
        self.root.keymaps.copy('main', '<Right>', '<A-l>')
        self.root.keymaps.bind('main', '<C-a>', host_begin)
        self.root.keymaps.copy('main', '<C-a>', '^')
        self.root.keymaps.bind('main', '<C-e>', host_end)
        self.root.keymaps.copy('main', '<C-e>', '$')
        self.root.keymaps.bind('main', '<Up>', partial(select_move, direction=-1))
        self.root.keymaps.copy('main', '<Up>', '<S-Tab>')
        self.root.keymaps.copy('main', '<Up>', '<A-k>')
        self.root.keymaps.bind('main', '<Down>', partial(select_move, direction=+1))
        self.root.keymaps.copy('main', '<Down>', '<Tab>')
        self.root.keymaps.copy('main', '<Down>', '<A-j>')
        self.root.keymaps.bind('main', '<Home>', partial(select_move, direction=-(1 << 20)))
        self.root.keymaps.bind('main', '<End>', partial(select_move, direction=+(1 << 20)))
        self.root.keymaps.bind('main', '<Esc>', select_clear)

        self.root.keymaps.bind('main', 'T', terminate)
        self.root.keymaps.bind('main', 'K', kill)
        self.root.keymaps.bind('main', '<C-c>', interrupt)
        self.root.keymaps.copy('main', '<C-c>', 'I')

        self.root.keymaps.bind('main', ',', order_previous)
        self.root.keymaps.copy('main', ',', '<')
        self.root.keymaps.bind('main', '.', order_next)
        self.root.keymaps.copy('main', '.', '>')
        self.root.keymaps.bind('main', '/', order_reverse)
        for order in ProcessPanel.ORDERS:
            self.root.keymaps.bind('main', 'o' + order[:1].lower(), partial(sort_by, order=order, reverse=False))
            self.root.keymaps.bind('main', 'o' + order[:1].upper(), partial(sort_by, order=order, reverse=True))
