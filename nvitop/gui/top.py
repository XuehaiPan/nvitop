# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import curses
import shutil
import sys
import threading
import time
from functools import partial

from .lib import DisplayableContainer, ALT_KEY, KeyBuffer, KeyMaps, MouseEvent
from .panels import DevicePanel, HostPanel, ProcessPanel


class BreakLoop(Exception):
    pass


class Top(DisplayableContainer):
    def __init__(self, devices, ascii=False, mode='auto', win=None):  # pylint: disable=redefined-builtin
        super().__init__(win, root=self)

        self.width = max(79, shutil.get_terminal_size(fallback=(79, 24)).columns)
        if not sys.stdout.isatty():
            self.width = 1024
        self.termsize = None

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

        self.process_panel = ProcessPanel(self.devices, compact, win=win, root=self)
        self.process_panel.focused = False
        self.add_child(self.process_panel)

        self.selected = self.process_panel.selected

        self.ascii = ascii
        self.device_panel.ascii = self.ascii
        self.host_panel.ascii = self.ascii
        self.process_panel.ascii = self.ascii
        if ascii:
            self.host_panel.full_height = self.host_panel.height = self.host_panel.compact_height

        self.x = self.y = 0
        self.device_panel.x = self.host_panel.x = self.process_panel.x = self.x
        self.device_panel.y = self.y
        self.host_panel.y = self.device_panel.y + self.device_panel.height
        self.process_panel.y = self.host_panel.y + self.host_panel.height
        self.height = self.device_panel.height + self.host_panel.height + self.process_panel.height

        if win is not None:
            self.keybuffer = KeyBuffer()
            self.keymaps = KeyMaps(self.keybuffer)
            self.last_input_time = time.monotonic()
            self.init_keybindings()

    @property
    def compact(self):
        return self._compact

    @compact.setter
    def compact(self, value):
        if self._compact != value:
            self.need_redraw = True
            self._compact = value

    def init_keybindings(self):
        # pylint: disable=multiple-statements

        def quit(top): raise BreakLoop  # pylint: disable=redefined-builtin

        def change_mode(top, mode):
            top.mode = mode
            top.update_size()

        def host_left(top): top.process_panel.host_offset -= 1
        def host_right(top): top.process_panel.host_offset += 1
        def host_begin(top): top.process_panel.host_offset = -1
        def host_end(top): top.process_panel.host_offset = 1024

        def select_up(top): top.selected.move(direction=-1)
        def select_down(top): top.selected.move(direction=+1)
        def select_clear(top): top.selected.clear()

        def terminate(top): top.selected.terminate()
        def kill(top): top.selected.kill()
        def interrupt(top): top.selected.interrupt()

        def sort_by(top, order, reverse):
            top.process_panel.order = order
            top.process_panel.reverse = reverse
            top.update_size()

        def order_previous(top):
            sort_by(top, order=top.process_panel.ORDERS[top.process_panel.order].previous, reverse=False)

        def order_next(top):
            sort_by(top, order=top.process_panel.ORDERS[top.process_panel.order].next, reverse=False)

        def order_reverse(top):
            sort_by(top, order=top.process_panel.order, reverse=(not top.process_panel.reverse))

        self.keymaps.bind('root', 'q', quit)
        self.keymaps.copy('root', 'q', 'Q')
        self.keymaps.bind('root', 'a', partial(change_mode, mode='auto'))
        self.keymaps.bind('root', 'f', partial(change_mode, mode='full'))
        self.keymaps.bind('root', 'c', partial(change_mode, mode='compact'))

        self.keymaps.bind('root', '<Left>', host_left)
        self.keymaps.copy('root', '<Left>', '[')
        self.keymaps.bind('root', '<Right>', host_right)
        self.keymaps.copy('root', '<Right>', ']')
        self.keymaps.bind('root', '<Home>', host_begin)
        self.keymaps.copy('root', '<Home>', '<C-a>')
        self.keymaps.copy('root', '<Home>', '^')
        self.keymaps.bind('root', '<End>', host_end)
        self.keymaps.copy('root', '<End>', '<C-e>')
        self.keymaps.copy('root', '<End>', '$')
        self.keymaps.bind('root', '<Up>', select_up)
        self.keymaps.copy('root', '<Up>', '<S-Tab>')
        self.keymaps.bind('root', '<Down>', select_down)
        self.keymaps.copy('root', '<Down>', '<Tab>')
        self.keymaps.bind('root', '<Esc>', select_clear)

        self.keymaps.bind('root', 'T', terminate)
        self.keymaps.bind('root', 'K', kill)
        self.keymaps.bind('root', '<C-c>', interrupt)
        self.keymaps.copy('root', '<C-c>', 'I')

        self.keymaps.bind('root', ',', order_previous)
        self.keymaps.copy('root', ',', '<')
        self.keymaps.bind('root', '.', order_next)
        self.keymaps.copy('root', '.', '>')
        self.keymaps.bind('root', '/', order_reverse)
        for order in ProcessPanel.ORDERS:
            self.keymaps.bind('root', 'o' + order[:1].lower(), partial(sort_by, order=order, reverse=False))
            self.keymaps.bind('root', 'o' + order[:1].upper(), partial(sort_by, order=order, reverse=True))

        self.keymaps.use_keymap('root')

    def update_size(self):
        curses.update_lines_cols()  # pylint: disable=no-member
        n_term_lines, n_term_cols = termsize = self.win.getmaxyx()
        self.width = n_term_cols - self.x
        heights = [
            self.device_panel.full_height + self.host_panel.full_height + self.process_panel.full_height,
            self.device_panel.compact_height + self.host_panel.full_height + self.process_panel.full_height,
            self.device_panel.compact_height + self.host_panel.compact_height + self.process_panel.full_height,
        ]
        if self.mode == 'auto':
            self.compact = (n_term_lines < heights[0])
            self.host_panel.compact = (n_term_lines < heights[1])
            self.process_panel.compact = (n_term_lines < heights[-1])
        else:
            self.compact = (self.mode == 'compact')
            self.host_panel.compact = self.compact
            self.process_panel.compact = self.compact
        self.device_panel.compact = self.compact
        self.host_panel.y = self.device_panel.y + self.device_panel.height
        self.process_panel.y = self.host_panel.y + self.host_panel.height
        self.height = self.device_panel.height + self.process_panel.height
        self.device_panel.width = self.width
        self.host_panel.width = self.width
        self.process_panel.width = self.width
        if self.termsize != termsize:
            self.termsize = termsize
            self.need_redraw = True

    def poke(self):
        super().poke()

        height = self.device_panel.height + self.host_panel.height + self.process_panel.height
        if self.termsize is None or self.height != height:
            self.update_size()

    def draw(self):
        if self.need_redraw:
            self.win.erase()

        if self.width >= 79:
            super().draw()
        elif self.need_redraw:
            n_term_lines, n_term_cols = self.termsize
            message = 'nvitop needs at least a width of 79 to render, the current width is {}.'.format(self.width)
            width = min(max(n_term_cols, 40), n_term_cols, 60) - 10
            lines = ['nvitop']
            for word in message.split()[1:]:
                if len(lines[-1]) + len(word) + 1 <= width:
                    lines[-1] += ' ' + word
                else:
                    lines[-1] = lines[-1].strip()
                    lines.append(word)
            height, width = len(lines) + 4, max(map(len, lines)) + 4
            lines = ['│ {} │'.format(line.ljust(width - 4)) for line in lines]
            lines = [
                '╒' + '═' * (width - 2) + '╕',
                '│' + ' ' * (width - 2) + '│',
                *lines,
                '│' + ' ' * (width - 2) + '│',
                '╘' + '═' * (width - 2) + '╛',
            ]

            y_start, x_start = (n_term_lines - height) // 2, (n_term_cols - width) // 2
            for y, line in enumerate(lines, start=y_start):
                self.addstr(y, x_start, line)

    def finalize(self):
        super().finalize()
        self.win.refresh()

    def redraw(self):
        self.poke()
        self.draw()
        self.finalize()

    def loop(self):
        if self.win is None:
            return

        while True:
            try:
                self.redraw()
                self.handle_input()
                if time.monotonic() - self.last_input_time > 1.0:
                    time.sleep(0.25)
            except BreakLoop:
                break

    def print(self):
        if self.device_count > 0:
            device_panel_width = self.device_panel.print_width()
            host_panel_width = self.host_panel.print_width()
            process_panel_width = self.process_panel.print_width()
            print_width = min(device_panel_width, host_panel_width, process_panel_width)
            self.width = max(print_width, min(self.width, 100))
        else:
            self.width = 79
        self.device_panel.width = self.width
        self.host_panel.width = self.width
        self.process_panel.width = self.width

        self.device_panel.print()
        self.host_panel.print()
        self.process_panel.print()

    def handle_mouse(self):
        """Handles mouse input"""

        try:
            event = MouseEvent(curses.getmouse())
        except curses.error:
            return
        else:
            super().click(event)

    def handle_key(self, key):
        """Handles key input"""

        if key < 0:
            self.keybuffer.clear()
        elif not super().press(key):
            self.keymaps.use_keymap('root')
            self.press(key)

    def handle_keys(self, *keys):
        for key in keys:
            self.handle_key(key)

    def press(self, key):
        keybuffer = self.keybuffer

        keybuffer.add(key)
        if keybuffer.result is not None:
            try:
                keybuffer.result(self)
            finally:
                if keybuffer.finished_parsing:
                    keybuffer.clear()
        elif keybuffer.finished_parsing:
            keybuffer.clear()
            return False
        return True

    def handle_input(self):  # pylint: disable=too-many-branches
        key = self.win.getch()
        if key == curses.ERR:
            return

        self.last_input_time = time.monotonic()
        if key == curses.KEY_ENTER:
            key = ord('\n')
        if key == 27 or (128 <= key < 256):
            # Handle special keys like ALT+X or unicode here:
            keys = [key]
            for _ in range(4):
                getkey = self.win.getch()
                if getkey != -1:
                    keys.append(getkey)
            if len(keys) == 1:
                keys.append(-1)
            elif keys[0] == 27:
                keys[0] = ALT_KEY
            self.handle_keys(*keys)
            curses.flushinp()
        elif key >= 0:
            # Handle simple key presses, CTRL+X, etc here:
            curses.flushinp()
            if key == curses.KEY_MOUSE:
                self.handle_mouse()
            elif key == curses.KEY_RESIZE:
                self.update_size()
            else:
                self.handle_key(key)
