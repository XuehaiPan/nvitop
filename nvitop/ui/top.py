# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import curses
import time

from .displayable import DisplayableContainer
from .keybinding import ALT_KEY, KeyBuffer, KeyMaps
from .mouse import MouseEvent
from .panels import DevicePanel, ProcessPanel


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
        # pylint: disable=multiple-statements

        def quit(top): raise BreakLoop  # pylint: disable=redefined-builtin

        def cmd_left(top): top.process_panel.offset -= 1
        def cmd_right(top): top.process_panel.offset += 1

        def select_up(top): top.process_panel.selected.move(direction=-1)
        def select_down(top): top.process_panel.selected.move(direction=+1)
        def select_clear(top): top.process_panel.selected.clear()

        def kill(top): top.process_panel.selected.kill()
        def terminate(top): top.process_panel.selected.terminate()
        def interrupt(top): top.process_panel.selected.interrupt()

        self.keymaps.bind('top', 'q', quit)
        self.keymaps.bind('process', 'q', quit)
        self.keymaps.bind('process', '<left>', cmd_left)
        self.keymaps.copy('process', '<left>', '[')
        self.keymaps.bind('process', '<right>', cmd_right)
        self.keymaps.copy('process', '<right>', ']')
        self.keymaps.bind('process', '<up>', select_up)
        self.keymaps.copy('process', '<up>', '<s-tab>')
        self.keymaps.bind('process', '<down>', select_down)
        self.keymaps.copy('process', '<down>', '<tab>')
        self.keymaps.bind('process', '<backspace>', select_clear)
        self.keymaps.bind('process', 'k', kill)
        self.keymaps.bind('process', 't', terminate)
        self.keymaps.bind('process', '<c-c>', interrupt)

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
                    time.sleep(0.25)
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
        key = self.win.getch()
        if key != curses.ERR:
            self.last_input_time = time.time()

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
