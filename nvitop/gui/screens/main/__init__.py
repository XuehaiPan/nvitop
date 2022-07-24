# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import threading
from functools import partial

from nvitop.gui.library import LARGE_INTEGER, DisplayableContainer, MouseEvent
from nvitop.gui.screens.main.device import DevicePanel
from nvitop.gui.screens.main.host import HostPanel
from nvitop.gui.screens.main.process import ProcessPanel


class BreakLoop(Exception):
    pass


class MainScreen(DisplayableContainer):  # pylint: disable=too-many-instance-attributes
    NAME = 'main'

    # pylint: disable-next=redefined-builtin,too-many-arguments,too-many-locals,too-many-statements
    def __init__(self, devices, filters, ascii, mode, win, root):
        super().__init__(win, root)

        self.width = root.width

        assert mode in ('auto', 'full', 'compact')
        compact = mode == 'compact'
        self.mode = mode
        self._compact = compact

        self.devices = devices
        self.device_count = len(self.devices)

        self.snapshot_lock = threading.Lock()

        self.device_panel = DevicePanel(self.devices, compact, win=win, root=root)
        self.device_panel.focused = False
        self.add_child(self.device_panel)

        self.host_panel = HostPanel(self.device_panel.leaf_devices, compact, win=win, root=root)
        self.host_panel.focused = False
        self.add_child(self.host_panel)

        self.process_panel = ProcessPanel(
            self.device_panel.leaf_devices, compact, filters, win=win, root=root
        )
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
        n_term_lines, n_term_cols = termsize = super().update_size(termsize=termsize)

        self.width = n_term_cols - self.x
        self.device_panel.width = self.width
        self.host_panel.width = self.width
        self.process_panel.width = self.width

        self.y = min(self.y, self.root.y)
        height = n_term_lines - self.y
        heights = [
            self.device_panel.full_height
            + self.host_panel.full_height
            + self.process_panel.full_height,
            self.device_panel.compact_height
            + self.host_panel.full_height
            + self.process_panel.full_height,
            self.device_panel.compact_height
            + self.host_panel.compact_height
            + self.process_panel.full_height,
        ]
        if self.mode == 'auto':
            self.compact = height < heights[0]
            self.host_panel.compact = height < heights[1]
            self.process_panel.compact = height < heights[-1]
        else:
            self.compact = self.mode == 'compact'
            self.host_panel.compact = self.compact
            self.process_panel.compact = self.compact
        self.device_panel.compact = self.compact

        self.device_panel.y = self.y
        self.host_panel.y = self.device_panel.y + self.device_panel.height
        self.process_panel.y = self.host_panel.y + self.host_panel.height
        height = self.device_panel.height + self.host_panel.height + self.process_panel.height

        if self.y < self.root.y and self.y + height < n_term_lines:
            self.y = min(self.root.y + self.root.height - height, self.root.y)
            self.update_size(termsize)
            self.need_redraw = True

        if self.height != height:
            self.height = height
            self.need_redraw = True

    def move(self, direction=0):
        if direction == 0:
            return

        self.y -= direction
        self.update_size()
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

    def __contains__(self, item):
        if self.visible and isinstance(item, MouseEvent):
            return True
        return super().__contains__(item)

    def init_keybindings(self):
        # pylint: disable=too-many-locals,too-many-statements

        def quit():  # pylint: disable=redefined-builtin
            raise BreakLoop

        def change_mode(mode):
            self.mode = mode
            self.root.update_size()

        def force_refresh():
            select_clear()
            host_begin()
            self.y = self.root.y
            self.root.update_size()
            self.root.need_redraw = True

        def host_left():
            self.process_panel.host_offset -= 2

        def host_right():
            self.process_panel.host_offset += 2

        def host_begin():
            self.process_panel.host_offset = -1

        def host_end():
            self.process_panel.host_offset = LARGE_INTEGER

        def select_move(direction):
            self.selected.move(direction=direction)

        def select_clear():
            self.selected.clear()

        def screen_move(direction):
            self.move(direction)

        def terminate():
            self.selected.terminate()

        def kill():
            self.selected.kill()

        def interrupt():
            self.selected.interrupt()

        def sort_by(order, reverse):
            self.process_panel.order = order
            self.process_panel.reverse = reverse
            self.root.update_size()

        def order_previous():
            sort_by(order=ProcessPanel.ORDERS[self.process_panel.order].previous, reverse=False)

        def order_next():
            sort_by(order=ProcessPanel.ORDERS[self.process_panel.order].next, reverse=False)

        def order_reverse():
            sort_by(order=self.process_panel.order, reverse=(not self.process_panel.reverse))

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
        self.root.keymaps.copy('main', '<Left>', '<A-h>')
        self.root.keymaps.bind('main', '<Right>', host_right)
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

        self.root.keymaps.bind('main', '<PageUp>', partial(screen_move, direction=-1))
        self.root.keymaps.copy('main', '<PageUp>', '[')
        self.root.keymaps.copy('main', '<PageUp>', '<A-K>')
        self.root.keymaps.bind('main', '<PageDown>', partial(screen_move, direction=+1))
        self.root.keymaps.copy('main', '<PageDown>', ']')
        self.root.keymaps.copy('main', '<PageDown>', '<A-J>')

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
            self.root.keymaps.bind(
                'main', 'o' + order[:1].lower(), partial(sort_by, order=order, reverse=False)
            )
            self.root.keymaps.bind(
                'main', 'o' + order[:1].upper(), partial(sort_by, order=order, reverse=True)
            )
