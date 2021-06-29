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
from .screens import MainScreen, EnvironScreen, TreeViewScreen, HelpScreen, BreakLoop


class Top(DisplayableContainer):
    def __init__(self, devices, filters=(), ascii=False, mode='auto', win=None):  # pylint: disable=redefined-builtin
        super().__init__(win, root=self)

        self.x = self.y = 0
        self.width = max(79, shutil.get_terminal_size(fallback=(79, 24)).columns)
        if not sys.stdout.isatty():
            self.width = 1024
        self.termsize = None

        self.ascii = ascii

        self.devices = devices
        self.device_count = len(self.devices)

        self.lock = threading.RLock()

        self.main_screen = MainScreen(self.devices, filters, ascii=ascii, mode=mode, win=win, root=self)
        self.main_screen.visible = True
        self.main_screen.focused = False
        self.add_child(self.main_screen)

        if win is not None:
            self.environ_screen = EnvironScreen(win=win, root=self)
            self.environ_screen.visible = False
            self.environ_screen.ascii = False
            self.environ_screen.x = self.environ_screen.x = 0
            self.add_child(self.environ_screen)

            self.treeview_screen = TreeViewScreen(win=win, root=self)
            self.treeview_screen.visible = False
            self.treeview_screen.ascii = self.ascii
            self.treeview_screen.x = self.treeview_screen.y = 0
            self.add_child(self.treeview_screen)

            self.help_screen = HelpScreen(win=win, root=self)
            self.help_screen.visible = False
            self.environ_screen.ascii = False
            self.help_screen.x = self.help_screen.y = 0
            self.add_child(self.help_screen)

            self.keybuffer = KeyBuffer()
            self.keymaps = KeyMaps(self.keybuffer)
            self.last_input_time = time.monotonic()
            self.init_keybindings()

    def init_keybindings(self):
        # pylint: disable=multiple-statements

        for child in self.container:
            if hasattr(child, 'init_keybindings'):
                child.init_keybindings()

        def show_environ(top, screen='main'):
            top.main_screen.visible = False
            top.treeview_screen.visible = False
            top.help_screen.visible = False

            top.environ_screen.visible = True
            top.environ_screen.focused = True
            top.environ_screen.previous_screen = screen

            if screen == 'treeview':
                top.environ_screen.process = top.treeview_screen.selected.process
            else:
                top.environ_screen.process = top.main_screen.selected.process

        def environ_return(top):
            if top.environ_screen.previous_screen == 'treeview':
                show_treeview(top)
            else:
                return2top(top)

        def show_treeview(top):
            top.main_screen.visible = False
            top.environ_screen.visible = False
            top.help_screen.visible = False

            top.treeview_screen.visible = True
            top.treeview_screen.focused = True

            if not top.treeview_screen.selected.is_set():
                top.treeview_screen.selected.process = top.main_screen.selected.process

        def show_help(top):
            top.main_screen.visible = False
            top.environ_screen.visible = False
            top.treeview_screen.visible = False

            top.help_screen.visible = True
            top.help_screen.focused = True

        def return2top(top):
            top.environ_screen.visible = False
            top.treeview_screen.visible = False
            top.help_screen.visible = False

            top.main_screen.visible = True
            top.main_screen.focused = False

            top.treeview_screen.selected.clear()

        self.keymaps.bind('main', 'e', partial(show_environ, screen='main'))
        self.keymaps.bind('environ', '<Esc>', environ_return)
        self.keymaps.copy('environ', '<Esc>', 'q')
        self.keymaps.copy('environ', '<Esc>', 'Q')

        self.keymaps.bind('main', 't', show_treeview)
        self.keymaps.bind('treeview', 't', return2top)
        self.keymaps.copy('treeview', 't', 'q')
        self.keymaps.copy('treeview', 't', 'Q')

        self.keymaps.bind('treeview', 'e', partial(show_environ, screen='treeview'))

        self.keymaps.bind('main', 'h', show_help)
        self.keymaps.bind('help', '<any>', return2top)

        self.keymaps.use_keymap('main')

    def update_size(self):
        curses.update_lines_cols()  # pylint: disable=no-member
        n_term_lines, n_term_cols = termsize = self.win.getmaxyx()
        n_term_lines -= self.y
        n_term_cols -= self.x
        self.width = n_term_cols
        self.main_screen.update_size(termsize)
        self.environ_screen.height, self.environ_screen.width = n_term_lines, n_term_cols
        self.treeview_screen.height, self.treeview_screen.width = n_term_lines, n_term_cols
        if self.termsize != termsize:
            self.termsize = termsize
            self.need_redraw = True

    def poke(self):
        super().poke()

        if self.termsize is None:
            self.update_size()

    def draw(self):
        if self.need_redraw:
            self.win.erase()

        self.color_reset()

        if self.width >= 79:
            super().draw()
            return
        elif not self.need_redraw:
            return

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
        self.main_screen.print()

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
            self.keymaps.use_keymap('main')
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
