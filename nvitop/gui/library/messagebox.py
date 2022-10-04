# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# This file is originally part of ranger, the console file manager. https://github.com/ranger/ranger
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from functools import partial

from nvitop.gui.library.displayable import Displayable
from nvitop.gui.library.keybinding import normalize_keybinding
from nvitop.gui.library.process import host


class MessageBox(Displayable):  # pylint: disable=too-many-instance-attributes
    class Option:  # pylint: disable=too-few-public-methods
        # pylint: disable-next=too-many-arguments
        def __init__(self, name, key, callback, keys=(), attrs=()):
            self.name = name
            self.offset = 0
            self.key = normalize_keybinding(key)
            self.callback = callback
            self.keys = tuple(set(normalize_keybinding(key) for key in keys).difference({self.key}))
            self.attrs = attrs

        def __str__(self):
            return self.name

    # pylint: disable-next=too-many-arguments
    def __init__(self, message, options, default, yes, no, cancel, win, root):
        super().__init__(win, root)

        if default is None:
            default = 0
        if no is None:
            no = cancel

        self.options = options
        self.num_options = len(self.options)

        assert cancel is not None and self.num_options >= 2
        assert 0 <= no < self.num_options
        assert 0 <= cancel < self.num_options
        assert 0 <= default < self.num_options

        self.previous_focused = None
        self.message = message
        self.previous_keymap = root.keymaps.used_keymap
        self.current = default
        self.yes = yes
        self.cancel = cancel
        self.no = no  # pylint: disable=invalid-name

        self.name_len = max(8, max(len(option.name) for option in options))
        for option in self.options:
            option.offset = (self.name_len - len(option.name)) // 2
            option.name = option.name.center(self.name_len)

        self.xy_mouse = None
        self.x, self.y = root.x, root.y
        self.width = (self.name_len + 6) * self.num_options + 6

        self.init_keybindings()

        lines = []
        for msg in self.message.splitlines():
            words = iter(msg.split())
            try:
                lines.append(next(words))
            except StopIteration:
                lines.append('')
                continue
            for word in words:
                if len(lines[-1]) + len(word) + 1 <= self.width - 6:
                    lines[-1] += ' ' + word
                else:
                    lines[-1] = lines[-1].strip()
                    lines.append(word)
        if len(lines) == 1:
            lines[-1] = lines[-1].center(self.width - 6)
        lines = [' │ {} │ '.format(line.ljust(self.width - 6)) for line in lines]
        lines = [
            ' ╒' + '═' * (self.width - 4) + '╕ ',
            ' │' + ' ' * (self.width - 4) + '│ ',
            *lines,
            ' │' + ' ' * (self.width - 4) + '│ ',
            ' │  ' + '  '.join(['┌' + '─' * (self.name_len + 2) + '┐'] * self.num_options) + '  │ ',
            ' │  ' + '  '.join(map('│ {} │'.format, self.options)) + '  │ ',
            ' │  ' + '  '.join(['└' + '─' * (self.name_len + 2) + '┘'] * self.num_options) + '  │ ',
            ' ╘' + '═' * (self.width - 4) + '╛ ',
        ]
        self.lines = lines

    def draw(self):
        self.set_base_attr(attr=0)
        self.color_reset()

        n_term_lines, n_term_cols = self.root.termsize

        height = len(self.lines)
        y_start, x_start = (n_term_lines - height) // 2, (n_term_cols - self.width) // 2
        y_option_start = y_start + height - 3
        for y, line in enumerate(self.lines, start=y_start):
            self.addstr(y, x_start, line)

        for i, option in enumerate(self.options):
            x_option_start = x_start + 6 + i * (self.name_len + 6) + option.offset
            for attr in option.attrs:
                attr = attr.copy()
                y = y_option_start + attr.pop('y')
                x = x_option_start + attr.pop('x')
                self.color_at(y, x, **attr)

        if self.xy_mouse is not None:
            x, y = self.xy_mouse
            if y_option_start - 1 <= y <= y_option_start + 1:
                current = (x - x_start - 3) // (self.name_len + 6)
                x_option_start = x_start + 6 + current * (self.name_len + 6)
                if (
                    0 <= current < self.num_options
                    and x_option_start - 3 <= x < x_option_start + self.name_len + 3
                ):
                    self.current = current
            self.xy_mouse = None

        option = self.options[self.current]
        x_option_start = x_start + 6 + self.current * (self.name_len + 6)
        for y in range(y_option_start - 1, y_option_start + 2):
            self.color_at(
                y,
                x_option_start - 3,
                width=self.name_len + 6,
                attr='standout | bold',
            )
        for attr in option.attrs:
            attr = attr.copy()
            y = y_option_start + attr.pop('y')
            x = x_option_start + option.offset + attr.pop('x')
            attr['fg'], attr['bg'] = attr.get('bg', -1), attr.get('fg', -1)
            attr['attr'] = self.get_fg_bg_attr(attr=attr.get('attr', 0))
            attr['attr'] |= self.get_fg_bg_attr(attr='standout | bold')
            self.color_at(y, x, **attr)

    def press(self, key):
        self.root.keymaps.use_keymap('messagebox')
        self.root.press(key)

    def click(self, event):
        if event.pressed(1) or event.pressed(3) or event.clicked(1) or event.clicked(3):
            self.xy_mouse = (event.x, event.y)
            return True

        direction = event.wheel_direction()
        self.current = (self.current + direction) % self.num_options
        return True

    def init_keybindings(self):  # pylint: disable=too-many-branches
        def apply(index):
            callback = self.options[index].callback
            if callback is not None:
                callback()
            self.root.keymaps.clear_keymap('messagebox')
            self.root.keymaps.use_keymap(self.previous_keymap)
            self.root.need_redraw = True
            self.root.messagebox = None

        def confirm():
            apply(self.current)

        def select_previous():
            self.current = (self.current - 1) % self.num_options

        def select_next():
            self.current = (self.current + 1) % self.num_options

        keymaps = self.root.keymaps
        keymaps.clear_keymap('messagebox')

        for i, option in enumerate(self.options):
            keymaps.bind('messagebox', option.key, partial(apply, index=i))
            for key in option.keys:
                keymaps.copy('messagebox', option.key, key)

        if len(set('0123456789').intersection(keymaps['messagebox'])) and self.num_options <= 9:
            for key_n, option in zip('123456789', self.options):
                keymaps.copy('messagebox', option.key, key_n)

        assert (
            len({'<Enter>', '<Esc>', '<Left>', '<Right>'}.intersection(keymaps['messagebox'])) == 0
        )

        if self.yes is not None and 'y' not in keymaps['messagebox']:
            keymaps.copy('messagebox', self.options[self.yes].key, 'y')
            if 'Y' not in keymaps['messagebox']:
                keymaps.copy('messagebox', self.options[self.yes].key, 'Y')
        if self.no is not None and 'n' not in keymaps['messagebox']:
            keymaps.copy('messagebox', self.options[self.no].key, 'n')
            if 'N' not in keymaps['messagebox']:
                keymaps.copy('messagebox', self.options[self.no].key, 'N')
        if self.cancel is not None:
            keymaps.copy('messagebox', self.options[self.cancel].key, '<Esc>')
            if 'q' not in keymaps['messagebox'] and 'Q' not in keymaps['messagebox']:
                keymaps.copy('messagebox', self.options[self.cancel].key, 'q')
                keymaps.copy('messagebox', self.options[self.cancel].key, 'Q')

        keymaps.bind('messagebox', '<Enter>', confirm)
        if '<Space>' not in keymaps['messagebox']:
            keymaps.copy('messagebox', '<Enter>', '<Space>')

        keymaps.bind('messagebox', '<Left>', select_previous)
        keymaps.bind('messagebox', '<Right>', select_next)
        if ',' not in keymaps['messagebox'] and '.' not in keymaps['messagebox']:
            keymaps.copy('messagebox', '<Left>', ',')
            keymaps.copy('messagebox', '<Right>', '.')
        if '<' not in keymaps['messagebox'] and '>' not in keymaps['messagebox']:
            keymaps.copy('messagebox', '<Left>', '<')
            keymaps.copy('messagebox', '<Right>', '>')
        if '[' not in keymaps['messagebox'] and ']' not in keymaps['messagebox']:
            keymaps.copy('messagebox', '<Left>', '[')
            keymaps.copy('messagebox', '<Right>', ']')
        if '<Tab>' not in keymaps['messagebox'] and '<S-Tab>' not in keymaps['messagebox']:
            keymaps.copy('messagebox', '<Left>', '<S-Tab>')
            keymaps.copy('messagebox', '<Right>', '<Tab>')


def send_signal(signal, panel):
    assert signal in ('terminate', 'kill', 'interrupt')
    default = {'terminate': 0, 'kill': 1, 'interrupt': 2}.get(signal)
    processes = []
    for process in panel.selection.processes():
        try:
            username = process.username()
        except host.PsutilError:
            username = 'N/A'
        processes.append('{}({})'.format(process.pid, username))
    if len(processes) == 0:
        return
    if len(processes) == 1:
        message = 'Send signal to process {}?'.format(processes[0])
    else:
        maxlen = max(map(len, processes))
        processes = [process.ljust(maxlen) for process in processes]
        message = 'Send signal to the following processes?\n\n{}'.format(' '.join(processes))

    panel.root.messagebox = MessageBox(
        message=message,
        options=[
            MessageBox.Option(
                'SIGTERM',
                't',
                panel.selection.terminate,
                keys=('T',),
                attrs=(
                    dict(y=0, x=0, width=7, fg='red'),
                    dict(y=0, x=3, width=1, fg='red', attr='bold | underline'),
                ),
            ),
            MessageBox.Option(
                'SIGKILL',
                'k',
                panel.selection.kill,
                keys=('K',),
                attrs=(
                    dict(y=0, x=0, width=7, fg='red'),
                    dict(y=0, x=3, width=1, fg='red', attr='bold | underline'),
                ),
            ),
            MessageBox.Option(
                'SIGINT',
                'i',
                panel.selection.interrupt,
                keys=('I',),
                attrs=(
                    dict(y=0, x=0, width=6, fg='red'),
                    dict(y=0, x=3, width=1, fg='red', attr='bold | underline'),
                ),
            ),
            MessageBox.Option(
                'Cancel',
                'c',
                None,
                keys=('C',),
                attrs=(dict(y=0, x=0, width=1, attr='bold | underline'),),
            ),
        ],
        default=default,
        yes=None,
        no=3,
        cancel=3,
        win=panel.win,
        root=panel.root,
    )