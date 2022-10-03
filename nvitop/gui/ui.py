# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import curses
import shutil
import time

from nvitop.gui.library import ALT_KEY, DisplayableContainer, KeyBuffer, KeyMaps, MouseEvent
from nvitop.gui.screens import (
    BreakLoop,
    EnvironScreen,
    HelpScreen,
    MainScreen,
    ProcessMetricsScreen,
    TreeViewScreen,
)


class UI(DisplayableContainer):  # pylint: disable=too-many-instance-attributes
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        devices,
        filters=(),
        ascii=False,  # pylint: disable=redefined-builtin
        mode='auto',
        interval=None,
        win=None,
    ):
        super().__init__(win, root=self)

        self.x = self.y = 0
        self.width = max(79, shutil.get_terminal_size(fallback=(79, 24)).columns - self.x)
        self.termsize = None

        self.ascii = ascii

        self.devices = devices
        self.device_count = len(self.devices)

        self.main_screen = MainScreen(
            self.devices, filters, ascii=ascii, mode=mode, win=win, root=self
        )
        self.main_screen.visible = True
        self.main_screen.focused = False
        self.add_child(self.main_screen)
        self.current_screen = self.previous_screen = self.main_screen

        self._messagebox = None

        if win is not None:
            self.environ_screen = EnvironScreen(win=win, root=self)
            self.environ_screen.visible = False
            self.environ_screen.ascii = False
            self.add_child(self.environ_screen)

            self.treeview_screen = TreeViewScreen(win=win, root=self)
            self.treeview_screen.visible = False
            self.treeview_screen.ascii = self.ascii
            self.add_child(self.treeview_screen)

            self.process_metrics_screen = ProcessMetricsScreen(win=win, root=self)
            self.process_metrics_screen.visible = False
            self.process_metrics_screen.ascii = self.ascii
            self.add_child(self.process_metrics_screen)

            self.help_screen = HelpScreen(win=win, root=self)
            self.help_screen.visible = False
            self.help_screen.ascii = False
            self.add_child(self.help_screen)

            if interval is not None:
                self.main_screen.process_panel.set_snapshot_interval(interval)
                self.treeview_screen.set_snapshot_interval(interval)

            self.keybuffer = KeyBuffer()
            self.keymaps = KeyMaps(self.keybuffer)
            self.last_input_time = time.monotonic()
            self.init_keybindings()

    @property
    def messagebox(self):
        return self._messagebox

    @messagebox.setter
    def messagebox(self, value):
        self.need_redraw = True
        if self._messagebox is not None:
            self.remove_child(self._messagebox)

        self._messagebox = value
        if value is not None:
            self._messagebox.visible = True
            self._messagebox.focused = True
            self._messagebox.ascii = self.ascii
            self._messagebox.previous_focused = self.get_focused_obj()
            self.add_child(self._messagebox)

    def get_focused_obj(self):
        if self.messagebox is not None:
            return self.messagebox
        return super().get_focused_obj()

    def update_size(self, termsize=None):
        n_term_lines, n_term_cols = termsize = super().update_size(termsize=termsize)

        self.width = n_term_cols - self.x
        self.height = n_term_lines - self.y

        for screen in self.container:
            if hasattr(screen, 'update_size'):
                screen.update_size(termsize)

        if self.termsize != termsize:
            self.termsize = termsize
            self.need_redraw = True

        return termsize

    def poke(self):
        super().poke()

        if self.termsize is None:
            self.update_size()

    def draw(self):
        if self.need_redraw:
            self.win.erase()

        self.set_base_attr(attr=0)
        self.color_reset()

        if self.width >= 79:
            if self.messagebox is not None:
                self.set_base_attr(attr='dim')
            super().draw()
            return
        if not self.need_redraw:
            return

        n_term_lines, n_term_cols = self.termsize
        message = 'nvitop needs at least a width of 79 to render, the current width is {}.'.format(
            self.width
        )
        words = iter(message.split())
        width = min(max(n_term_cols, 40), n_term_cols, 60) - 10
        lines = [next(words)]
        for word in words:
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
                keybuffer.result()
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

    def init_keybindings(self):
        # pylint: disable=multiple-statements,too-many-statements

        for screen in self.container:
            if hasattr(screen, 'init_keybindings'):
                screen.init_keybindings()

        def show_screen(screen, focused=None):
            for s in self.container:
                if s is screen:
                    s.visible = True
                    if focused is not None:
                        s.focused = focused
                else:
                    s.visible = False

            self.previous_screen = self.current_screen
            self.current_screen = screen

        def show_main():
            show_screen(self.main_screen, focused=False)

            if self.treeview_screen.selection.is_set():
                self.main_screen.selection.process = self.treeview_screen.selection.process
            self.treeview_screen.selection.clear()
            self.process_metrics_screen.disable()

        def show_environ():
            show_screen(self.environ_screen, focused=True)

            if self.previous_screen is not self.help_screen:
                self.environ_screen.process = self.previous_screen.selection.process

        def environ_return():
            if self.previous_screen is self.treeview_screen:
                show_treeview()
            elif self.previous_screen is self.process_metrics_screen:
                show_process_metrics()
            else:
                show_main()

        def show_treeview():
            if not self.main_screen.process_panel.has_snapshots:
                return

            show_screen(self.treeview_screen, focused=True)

            if not self.treeview_screen.selection.is_set():
                self.treeview_screen.selection.process = self.main_screen.selection.process
            self.main_screen.selection.clear()

        def show_process_metrics():
            if self.current_screen is self.main_screen:
                if self.main_screen.selection.is_set():
                    show_screen(self.process_metrics_screen, focused=True)
                    self.process_metrics_screen.process = self.previous_screen.selection.process
            elif self.current_screen is not self.treeview_screen:
                show_screen(self.process_metrics_screen, focused=True)

        def show_help():
            show_screen(self.help_screen, focused=True)

        def help_return():
            if self.previous_screen is self.treeview_screen:
                show_treeview()
            elif self.previous_screen is self.environ_screen:
                show_environ()
            elif self.previous_screen is self.process_metrics_screen:
                show_process_metrics()
            else:
                show_main()

        self.keymaps.bind('main', 'e', show_environ)
        self.keymaps.bind('environ', 'e', environ_return)
        self.keymaps.copy('environ', 'e', '<Esc>')
        self.keymaps.copy('environ', 'e', 'q')
        self.keymaps.copy('environ', 'e', 'Q')

        self.keymaps.bind('main', 't', show_treeview)
        self.keymaps.bind('treeview', 't', show_main)
        self.keymaps.copy('treeview', 't', 'q')
        self.keymaps.copy('treeview', 't', 'Q')
        self.keymaps.bind('treeview', 'e', show_environ)

        self.keymaps.bind('main', '<Enter>', show_process_metrics)
        self.keymaps.bind('process-metrics', '<Enter>', show_main)
        self.keymaps.copy('process-metrics', '<Enter>', '<Esc>')
        self.keymaps.copy('process-metrics', '<Enter>', 'q')
        self.keymaps.copy('process-metrics', '<Enter>', 'Q')
        self.keymaps.bind('process-metrics', 'e', show_environ)

        for screen in ('main', 'treeview', 'environ', 'process-metrics'):
            self.keymaps.bind(screen, 'h', show_help)
            self.keymaps.copy(screen, 'h', '?')
        self.keymaps.bind('help', '<Esc>', help_return)
        self.keymaps.bind('help', '<any>', help_return)

        self.keymaps.use_keymap('main')
