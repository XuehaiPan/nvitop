# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import curses
import shutil
import sys
import threading
import time
from functools import partial

from .library import DisplayableContainer, ALT_KEY, KeyBuffer, KeyMaps, MouseEvent
from .panels import (DevicePanel, HostPanel, ProcessPanel,
                     EnvironPanel, TreeViewPanel, HelpPanel)


class BreakLoop(Exception):
    pass


class Top(DisplayableContainer):
    def __init__(self, devices, filters=(), ascii=False, mode='auto', win=None):  # pylint: disable=redefined-builtin
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

        self.x = self.y = 0
        self.device_panel.x = self.host_panel.x = self.process_panel.x = self.x
        self.device_panel.y = self.y
        self.host_panel.y = self.device_panel.y + self.device_panel.height
        self.process_panel.y = self.host_panel.y + self.host_panel.height
        self.height = self.device_panel.height + self.host_panel.height + self.process_panel.height

        if win is not None:
            self.environ_panel = EnvironPanel(win=win, root=self)
            self.environ_panel.visible = False
            self.environ_panel.ascii = False
            self.environ_panel.x = self.environ_panel.x = 0
            self.add_child(self.environ_panel)

            self.treeview_panel = TreeViewPanel(win=win, root=self)
            self.treeview_panel.visible = False
            self.treeview_panel.ascii = self.ascii
            self.treeview_panel.x = self.treeview_panel.y = 0
            self.add_child(self.treeview_panel)

            self.help_panel = HelpPanel(win=win, root=self)
            self.help_panel.visible = False
            self.help_panel.ascii = self.ascii
            self.help_panel.x = self.help_panel.y = 0
            self.add_child(self.help_panel)

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

        def force_refresh(top):
            top.update_size()
            top.need_redraw = True

        def host_left(top): top.process_panel.host_offset -= 2
        def host_right(top): top.process_panel.host_offset += 2
        def host_begin(top): top.process_panel.host_offset = -2
        def host_end(top): top.process_panel.host_offset = 1024

        def select_move(top, direction): top.selected.move(direction=direction)
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

        def show_environ(top, refresh=False, panel='root'):
            top.device_panel.visible = False
            top.host_panel.visible = False
            top.process_panel.visible = False
            top.treeview_panel.visible = False
            top.help_panel.visible = False

            if refresh:
                panel = top.environ_panel.previous_panel

            top.environ_panel.visible = True
            top.environ_panel.focused = True
            top.environ_panel.previous_panel = panel

            if panel == 'treeview':
                top.environ_panel.process = top.treeview_panel.selected.process
            else:
                top.environ_panel.process = top.process_panel.selected.process

        def environ_left(top): top.environ_panel.x_offset = max(0, top.environ_panel.x_offset - 5)
        def environ_right(top): top.environ_panel.x_offset += 5
        def environ_begin(top): top.environ_panel.x_offset = 0
        def environ_move(top, direction): top.environ_panel.move(direction=direction)

        def environ_return(top):
            if top.environ_panel.previous_panel == 'treeview':
                show_treeview(top)
            else:
                return2top(top)

        def show_treeview(top):
            top.device_panel.visible = False
            top.host_panel.visible = False
            top.process_panel.visible = False
            top.environ_panel.visible = False
            top.help_panel.visible = False

            top.treeview_panel.visible = True
            top.treeview_panel.focused = True

            if not top.treeview_panel.selected.is_set():
                top.treeview_panel.selected.process = top.process_panel.selected.process

        def show_help(top):
            top.device_panel.visible = False
            top.host_panel.visible = False
            top.process_panel.visible = False
            top.environ_panel.visible = False
            top.treeview_panel.visible = False

            top.help_panel.visible = True
            top.help_panel.focused = True

        def return2top(top):
            top.device_panel.visible = True
            top.host_panel.visible = True
            top.process_panel.visible = True
            top.environ_panel.visible = False
            top.treeview_panel.visible = False
            top.help_panel.visible = False

            top.treeview_panel.selected.clear()

        self.keymaps.bind('root', 'q', quit)
        self.keymaps.copy('root', 'q', 'Q')
        self.keymaps.bind('root', 'a', partial(change_mode, mode='auto'))
        self.keymaps.bind('root', 'f', partial(change_mode, mode='full'))
        self.keymaps.bind('root', 'c', partial(change_mode, mode='compact'))
        self.keymaps.bind('root', 'r', force_refresh)
        self.keymaps.copy('root', 'r', 'R')
        self.keymaps.copy('root', 'r', '<C-r>')
        self.keymaps.copy('root', 'r', '<F5>')

        self.keymaps.bind('root', '<Left>', host_left)
        self.keymaps.copy('root', '<Left>', '[')
        self.keymaps.copy('root', '<Left>', '<A-h>')
        self.keymaps.bind('root', '<Right>', host_right)
        self.keymaps.copy('root', '<Right>', ']')
        self.keymaps.copy('root', '<Right>', '<A-l>')
        self.keymaps.bind('root', '<C-a>', host_begin)
        self.keymaps.copy('root', '<C-a>', '^')
        self.keymaps.bind('root', '<C-e>', host_end)
        self.keymaps.copy('root', '<C-e>', '$')
        self.keymaps.bind('root', '<Up>', partial(select_move, direction=-1))
        self.keymaps.copy('root', '<Up>', '<S-Tab>')
        self.keymaps.copy('root', '<Up>', '<A-k>')
        self.keymaps.bind('root', '<Down>', partial(select_move, direction=+1))
        self.keymaps.copy('root', '<Down>', '<Tab>')
        self.keymaps.copy('root', '<Down>', '<A-j>')
        self.keymaps.bind('root', '<Home>', partial(select_move, direction=-(1 << 20)))
        self.keymaps.bind('root', '<End>', partial(select_move, direction=+(1 << 20)))
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

        self.keymaps.bind('root', 'e', partial(show_environ, panel='root'))
        self.keymaps.bind('environ', 'r', partial(show_environ, refresh=True))
        self.keymaps.copy('environ', 'r', 'R')
        self.keymaps.copy('environ', 'r', '<C-r>')
        self.keymaps.copy('environ', 'r', '<F5>')
        self.keymaps.bind('environ', '<Esc>', environ_return)
        self.keymaps.copy('environ', '<Esc>', 'q')
        self.keymaps.copy('environ', '<Esc>', 'Q')
        self.keymaps.bind('environ', '<Left>', environ_left)
        self.keymaps.copy('environ', '<Left>', '[')
        self.keymaps.copy('environ', '<Left>', '<A-h>')
        self.keymaps.bind('environ', '<Right>', environ_right)
        self.keymaps.copy('environ', '<Right>', ']')
        self.keymaps.copy('environ', '<Right>', '<A-l>')
        self.keymaps.bind('environ', '<C-a>', environ_begin)
        self.keymaps.copy('environ', '<C-a>', '^')
        self.keymaps.bind('environ', '<Up>', partial(environ_move, direction=-1))
        self.keymaps.copy('environ', '<Up>', '<S-Tab>')
        self.keymaps.copy('environ', '<Up>', '<A-k>')
        self.keymaps.bind('environ', '<Down>', partial(environ_move, direction=+1))
        self.keymaps.copy('environ', '<Down>', '<Tab>')
        self.keymaps.copy('environ', '<Down>', '<A-j>')
        self.keymaps.bind('environ', '<Home>', partial(environ_move, direction=-(1 << 20)))
        self.keymaps.bind('environ', '<End>', partial(environ_move, direction=+(1 << 20)))

        def tree_left(top): top.treeview_panel.x_offset = max(0, top.treeview_panel.x_offset - 5)
        def tree_right(top): top.treeview_panel.x_offset += 5
        def tree_begin(top): top.treeview_panel.x_offset = 0
        def tree_select_move(top, direction): top.treeview_panel.selected.move(direction=direction)
        def tree_select_clear(top): top.treeview_panel.selected.clear()
        def tree_terminate(top): top.treeview_panel.selected.terminate()
        def tree_kill(top): top.treeview_panel.selected.kill()
        def tree_interrupt(top): top.treeview_panel.selected.interrupt()

        self.keymaps.bind('root', 't', show_treeview)
        self.keymaps.bind('treeview', 't', return2top)
        self.keymaps.copy('treeview', 't', 'q')
        self.keymaps.copy('treeview', 't', 'Q')
        self.keymaps.bind('treeview', '<Left>', tree_left)
        self.keymaps.copy('treeview', '<Left>', '[')
        self.keymaps.copy('treeview', '<Left>', '<A-h>')
        self.keymaps.bind('treeview', '<Right>', tree_right)
        self.keymaps.copy('treeview', '<Right>', ']')
        self.keymaps.copy('treeview', '<Right>', '<A-l>')
        self.keymaps.bind('treeview', '<C-a>', tree_begin)
        self.keymaps.copy('treeview', '<C-a>', '^')
        self.keymaps.bind('treeview', '<Up>', partial(tree_select_move, direction=-1))
        self.keymaps.copy('treeview', '<Up>', '<S-Tab>')
        self.keymaps.copy('treeview', '<Up>', '<A-k>')
        self.keymaps.bind('treeview', '<Down>', partial(tree_select_move, direction=+1))
        self.keymaps.copy('treeview', '<Down>', '<Tab>')
        self.keymaps.copy('treeview', '<Down>', '<A-j>')
        self.keymaps.bind('treeview', '<Home>', partial(tree_select_move, direction=-(1 << 20)))
        self.keymaps.bind('treeview', '<End>', partial(tree_select_move, direction=+(1 << 20)))
        self.keymaps.bind('treeview', '<Esc>', tree_select_clear)
        self.keymaps.bind('treeview', 'e', partial(show_environ, panel='treeview'))

        self.keymaps.bind('treeview', 'T', tree_terminate)
        self.keymaps.bind('treeview', 'K', tree_kill)
        self.keymaps.bind('treeview', '<C-c>', tree_interrupt)
        self.keymaps.copy('treeview', '<C-c>', 'I')

        self.keymaps.bind('root', 'h', show_help)
        self.keymaps.bind('help', '<any>', return2top)

        self.keymaps.use_keymap('root')

    def update_size(self):
        curses.update_lines_cols()  # pylint: disable=no-member
        n_term_lines, n_term_cols = termsize = self.win.getmaxyx()
        n_term_lines -= self.y
        n_term_cols -= self.x
        self.width = n_term_cols
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
        self.environ_panel.height, self.environ_panel.width = n_term_lines, n_term_cols
        self.treeview_panel.height, self.treeview_panel.width = n_term_lines, n_term_cols
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

        self.color_reset()

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
            for y, line in enumerate(lines, start=y_start):  # pylint: disable=invalid-name
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
