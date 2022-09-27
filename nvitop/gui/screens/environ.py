# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from collections import OrderedDict
from functools import partial
from itertools import islice

from nvitop.gui.library import Displayable, GpuProcess, HostProcess, WideString, host


class EnvironScreen(Displayable):  # pylint: disable=too-many-instance-attributes
    NAME = 'environ'

    def __init__(self, win, root):
        super().__init__(win, root)

        self.this = HostProcess()

        self._process = None
        self._environ = None
        self.items = None
        self.username = None
        self.command = None

        self.x_offset = 0
        self._y_offset = 0
        self.scroll_offset = 0
        self.y_mouse = None

        self._height = 0
        self.x, self.y = root.x, root.y
        self.width, self.height = root.width, root.height

    @property
    def process(self):
        return self._process

    @process.setter
    def process(self, value):
        if value is None:
            value = self.this

        self._process = value

        with self.process.oneshot():
            try:
                self.environ = self.process.environ().copy()
            except host.PsutilError:
                self.environ = None

            try:
                self.command = self.process.command()
            except host.PsutilError:
                self.command = 'N/A'

            try:
                self.username = self.process.username()
            except host.PsutilError:
                self.username = 'N/A'

        self.command = WideString(self.command)
        self.username = WideString(self.username)

    @property
    def environ(self):
        return self._environ

    @environ.setter
    def environ(self, value):
        newline = '␤' if not self.root.ascii else '?'

        def normalize(s):
            return s.replace('\n', newline)

        if value is not None:
            self.items = [
                (WideString(key), WideString('{}={}'.format(key, normalize(value[key]))))
                for key in sorted(value.keys())
            ]
            value = OrderedDict(self.items)
        else:
            self.items = None
        self._environ = value
        self.x_offset = 0
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
    def display_height(self):
        return self.height - 2

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
        if n_items <= self.scroll_offset + self.display_height:
            self.scroll_offset = max(0, n_items - self.display_height)
        elif self.y_offset > self.scroll_offset + self.display_height - 1:
            self.scroll_offset = self.y_offset - self.display_height + 1
        self.scroll_offset = min(self.scroll_offset, self.y_offset)

    def move(self, direction, wheel=False):
        if self.environ is not None and wheel:
            n_items = len(self.environ)
            old_scroll_offset = self.scroll_offset
            self.scroll_offset = max(
                0, min(self.scroll_offset + direction, n_items - self.display_height)
            )
            direction -= self.scroll_offset - old_scroll_offset
            self._y_offset += self.scroll_offset - old_scroll_offset
        self.y_offset = self.y_offset + direction

    def update_size(self, termsize=None):
        n_term_lines, n_term_cols = termsize = super().update_size(termsize=termsize)

        self.width = n_term_cols - self.x
        self.height = n_term_lines - self.y

        return termsize

    def draw(self):
        self.color_reset()

        if isinstance(self.process, GpuProcess):
            process_type = 'GPU: {}'.format(
                self.process.type.replace('C', 'Compute').replace('G', 'Graphics')
            )
        else:
            process_type = 'Host'
        header_prefix = WideString(
            'Environment of process {} ({}@{}): '.format(
                self.process.pid, self.username, process_type
            )
        )
        offset = max(0, min(self.x_offset, len(self.command) + len(header_prefix) - self.width))
        header = str((header_prefix + self.command[offset:]).ljust(self.width)[: self.width])

        self.addstr(self.y, self.x, header)
        self.addstr(self.y + 1, self.x, '#' * self.width)
        self.color_at(self.y, self.x, width=len(header_prefix) - 1, fg='cyan', attr='bold')
        self.color_at(self.y + 1, self.x, width=self.width, fg='green', attr='bold')

        if self.environ is None:
            self.addstr(self.y + 2, self.x, 'Could not read process environment.')
            self.color_at(self.y + 2, self.x, width=self.width, fg='cyan', attr='reverse')
            return

        items = islice(self.items, self.scroll_offset, self.scroll_offset + self.display_height)
        for y, (key, line) in enumerate(items, start=self.y + 2):
            key_length = len(key)
            line = str(line[self.x_offset :].ljust(self.width)[: self.width])
            self.addstr(y, self.x, line)
            if self.x_offset < key_length:
                self.color_at(y, self.x, width=key_length - self.x_offset, fg='blue', attr='bold')
            if self.x_offset < key_length + 1:
                self.color_at(y, self.x + key_length - self.x_offset, width=1, fg='magenta')

            if y == self.y_mouse:
                self.y_offset = y - (self.y + 2 - self.scroll_offset)

            if y == self.y + 2 - self.scroll_offset + self.y_offset:
                self.color_at(y, self.x, width=self.width, fg='cyan', attr='bold | reverse')

    def finalize(self):
        self.y_mouse = None
        super().finalize()

    def press(self, key):
        self.root.keymaps.use_keymap('environ')
        self.root.press(key)

    def click(self, event):
        if event.pressed(1) or event.pressed(3) or event.clicked(1) or event.clicked(3):
            self.y_mouse = event.y
            return True

        direction = event.wheel_direction()
        if event.shift():
            self.x_offset = max(0, self.x_offset + 2 * direction)
        else:
            self.move(direction=direction, wheel=True)
        return True

    def init_keybindings(self):
        def refresh_environ():
            self.process = self.root.previous_screen.selection.process
            self.need_redraw = True

        def environ_left():
            self.x_offset = max(0, self.x_offset - 5)

        def environ_right():
            self.x_offset += 5

        def environ_begin():
            self.x_offset = 0

        def environ_move(direction):
            self.move(direction=direction)

        keymaps = self.root.keymaps

        keymaps.bind('environ', 'r', refresh_environ)
        keymaps.copy('environ', 'r', 'R')
        keymaps.copy('environ', 'r', '<C-r>')
        keymaps.copy('environ', 'r', '<F5>')
        keymaps.bind('environ', '<Left>', environ_left)
        keymaps.copy('environ', '<Left>', '<A-h>')
        keymaps.bind('environ', '<Right>', environ_right)
        keymaps.copy('environ', '<Right>', '<A-l>')
        keymaps.bind('environ', '<C-a>', environ_begin)
        keymaps.copy('environ', '<C-a>', '^')
        keymaps.bind('environ', '<Up>', partial(environ_move, direction=-1))
        keymaps.copy('environ', '<Up>', '<S-Tab>')
        keymaps.copy('environ', '<Up>', '<A-k>')
        keymaps.copy('environ', '<Up>', '<PageUp>')
        keymaps.copy('environ', '<Up>', '[')
        keymaps.bind('environ', '<Down>', partial(environ_move, direction=+1))
        keymaps.copy('environ', '<Down>', '<Tab>')
        keymaps.copy('environ', '<Down>', '<A-j>')
        keymaps.copy('environ', '<Down>', '<PageDown>')
        keymaps.copy('environ', '<Down>', ']')
        keymaps.bind('environ', '<Home>', partial(environ_move, direction=-(1 << 20)))
        keymaps.bind('environ', '<End>', partial(environ_move, direction=+(1 << 20)))
