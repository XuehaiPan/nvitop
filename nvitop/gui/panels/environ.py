# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import shutil
from collections import OrderedDict

from ...core import host, HostProcess, GpuProcess
from ..lib import Displayable


class EnvironPanel(Displayable):
    def __init__(self, win, root):
        super().__init__(win, root)

        self.this = HostProcess()
        self.selected = root.selected

        self._process = None
        self._environ = None

        self.x_offset = 0
        self._y_offset = 0
        self.scroll_offset = 0

        self._height = 0
        self.width, self.height = shutil.get_terminal_size(fallback=(79, 24))

    @property
    def process(self):
        return self._process

    @process.setter
    def process(self, value):
        if value is None:
            value = self.this

        self._process = value
        try:
            self.environ = self.process.environ().copy()
        except host.PsutilError:
            self.environ = None

    @property
    def environ(self):
        return self._environ

    @environ.setter
    def environ(self, value):
        if value is not None:
            value = OrderedDict([(key, value[key]) for key in sorted(value.keys())])
        self._environ = value
        self.y_offset = 0
        self.scroll_offset = 0

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        try:
            self.y_offset = self.y_offset
        except AttributeError:
            pass

    @property
    def y_offset(self):
        return self._y_offset

    @y_offset.setter
    def y_offset(self, value):
        if self.environ is None:
            self._y_offset = 0
            self.scroll_offset = 0
            return

        n_items = len(self.environ)
        self._y_offset = max(0, min(value, n_items - 1))
        if self.y + 2 + n_items <= self.scroll_offset + self.height:
            self.scroll_offset = max(0, self.y + 2 + n_items - self.height)
        elif self.y + 2 + self.y_offset > self.scroll_offset + self.height - 1:
            self.scroll_offset = self.y + 2 + self.y_offset - self.height + 1
        self.scroll_offset = min(self.scroll_offset, self.y_offset)

    def move(self, direction):
        self.y_offset = self.y_offset + direction

    def draw(self):
        self.color_reset()

        try:
            name = self.process.name()
        except host.PsutilError:
            name = 'N/A'

        if isinstance(self.process, GpuProcess):
            process_type = 'GPU: {}'.format(self.process.type.replace('C', 'Compute').replace('G', 'Graphics'))
        else:
            process_type = 'Host'
        header = 'Environment of process {} ({}) - {}'.format(self.process.pid, process_type, name)

        self.addstr(self.y, self.x, header.ljust(self.width))
        self.addstr(self.y + 1, self.x, '#' * self.width)
        self.color_at(self.y, self.x, width=self.width, fg='cyan', attr='bold')
        self.color_at(self.y + 1, self.x, width=self.width, fg='green', attr='bold')

        if self.environ is None:
            self.addstr(self.y + 2, self.x, 'Could not read process environment.')
            self.color_at(self.y + 2, self.x, width=self.width, fg='cyan', attr='reverse')
            return

        for y, (key, value) in enumerate(self.environ.items(), start=self.y + 2 - self.scroll_offset):
            if not 2 <= y - self.y < self.height:
                continue

            key_length = len(key)
            line = '{}={}'.format(key, value)
            self.addstr(y, self.x, line[self.x_offset:].ljust(self.width))
            if self.x_offset < key_length:
                self.color_at(y, self.x, width=key_length - self.x_offset, fg='blue', attr='bold')
            if self.x_offset < key_length + 1:
                self.color_at(y, self.x + key_length - self.x_offset, width=1, fg='magenta')

            if y == self.y + 2 - self.scroll_offset + self.y_offset:
                self.color_at(y, self.x, width=self.width, fg='cyan', attr='bold | reverse')

    def finalize(self):
        self.need_redraw = False

    def press(self, key):
        self.root.keymaps.use_keymap('environ')
        self.root.press(key)

    def click(self, event):
        direction = event.wheel_direction()
        if event.shift():
            self.x_offset = max(0, self.x_offset + direction)
        else:
            self.move(direction=direction)
        return True
