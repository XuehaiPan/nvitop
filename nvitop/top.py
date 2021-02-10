# This file is part of nvitop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import curses
import signal
import time

import psutil

from .displayable import DisplayableContainer
from .keybinding import ALT_KEY, KeyBuffer, KeyMaps
from .mouse import MouseEvent
from .panel import DevicePanel, ProcessPanel


class BreakLoop(Exception):
    pass


class Top(DisplayableContainer):
    def __init__(self, devices, mode='auto', win=None):
        super(Top, self).__init__(win, self)

        assert mode in ('auto', 'full', 'compact')
        compact = (mode == 'compact')
        self.mode = mode
        self._compact = compact

        self.devices = devices
        self.device_count = len(self.devices)

        self.win = win
        self.termsize = None

        self.process_panel = ProcessPanel(self.devices, win=win)
        self.process_panel.focused = True
        self.add_child(self.process_panel)

        self.device_panel = DevicePanel(self.devices, compact, win=win)
        self.device_panel.focused = False
        self.add_child(self.device_panel)

        self.process_panel.y = self.device_panel.height + 1
        self.height = self.device_panel.height + 1 + self.process_panel.height

        self.keybuffer = KeyBuffer()
        self.keymaps = KeyMaps(self.keybuffer)
        self.last_input_time = time.time()
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
        def quit(top): raise BreakLoop  # pylint: disable=redefined-builtin,multiple-statements

        def cmd_left(top): top.process_panel.offset -= 1  # pylint: disable=multiple-statements

        def cmd_right(top): top.process_panel.offset += 1  # pylint: disable=multiple-statements

        def select_up(top):
            selected = top.process_panel.selected
            with top.process_panel.snapshot_lock:
                snapshots = top.process_panel.snapshots
            if len(snapshots) > 0:
                if not selected.is_set():
                    selected.index = len(snapshots) - 1
                else:
                    selected.index = max(0, selected.index - 1)
                selected.process = snapshots[selected.index]
            else:
                selected.clear()

        def select_down(top):
            selected = top.process_panel.selected
            with top.process_panel.snapshot_lock:
                snapshots = top.process_panel.snapshots
            if len(snapshots) > 0:
                if not selected.is_set():
                    selected.index = 0
                else:
                    selected.index = min(selected.index + 1, len(snapshots) - 1)
                selected.process = snapshots[selected.index]
            else:
                selected.clear()

        def send_signal(top, sig):
            selected = top.process_panel.selected
            if selected.is_set() and selected.process.username == top.process_panel.current_user:
                try:
                    psutil.Process(selected.process.pid).send_signal(sig)
                except psutil.Error:
                    pass
                else:
                    if sig != signal.SIGINT:
                        selected.clear()
                    time.sleep(1.0)

        def kill(top): return send_signal(top, signal.SIGKILL)  # pylint: disable=multiple-statements
        def terminate(top): return send_signal(top, signal.SIGTERM)  # pylint: disable=multiple-statements
        def interrupt(top): return send_signal(top, signal.SIGINT)  # pylint: disable=multiple-statements

        self.keymaps.bind('top', 'q', quit)
        self.keymaps.bind('process', 'q', quit)
        self.keymaps.bind('process', '<left>', cmd_left)
        self.keymaps.bind('process', '<', cmd_left)
        self.keymaps.bind('process', '[', cmd_left)
        self.keymaps.bind('process', '<right>', cmd_right)
        self.keymaps.bind('process', '>', cmd_right)
        self.keymaps.bind('process', ']', cmd_right)
        self.keymaps.bind('process', '<up>', select_up)
        self.keymaps.bind('process', '<down>', select_down)
        self.keymaps.bind('process', 'k', kill)
        self.keymaps.bind('process', 't', terminate)
        self.keymaps.bind('process', '<C-c>', interrupt)

    def update_size(self):
        curses.update_lines_cols()  # pylint: disable=no-member
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

        if self.termsize is None or self.height != self.device_panel.height + 1 + self.process_panel.height:
            self.update_size()

    def draw(self):
        if self.need_redraw:
            self.win.erase()
        super(Top, self).draw()

    def finalize(self):
        super(Top, self).finalize()
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
                if time.time() - self.last_input_time > 1.0:
                    time.sleep(0.5)
            except BreakLoop:
                break

    def print(self):
        self.device_panel.print()
        print()
        self.process_panel.print()

    def handle_mouse(self):
        """Handles mouse input"""

        try:
            event = MouseEvent(curses.getmouse())
        except curses.error:
            return
        else:
            super(Top, self).click(event)

    def handle_key(self, key):
        """Handles key input"""

        if key < 0:
            self.keybuffer.clear()
        elif not super(Top, self).press(key):
            self.keymaps.use_keymap('top')
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
        self.last_input_time = time.time()

        key = self.win.getch()
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
        else:
            # Handle simple key presses, CTRL+X, etc here:
            if key >= 0:
                curses.flushinp()
                if key == curses.KEY_MOUSE:
                    self.handle_mouse()
                elif key == curses.KEY_RESIZE:
                    self.update_size()
                else:
                    self.handle_key(key)
