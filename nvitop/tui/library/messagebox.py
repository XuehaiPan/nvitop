# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# This file is originally part of ranger, the console file manager. https://github.com/ranger/ranger
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import curses
import string
import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Literal

from nvitop.tui.library import host
from nvitop.tui.library.displayable import Displayable
from nvitop.tui.library.keybinding import NAMED_SPECIAL_KEYS, normalize_keybinding
from nvitop.tui.library.utils import cut_string
from nvitop.tui.library.widestring import WideString


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from nvitop.tui.library.mouse import MouseEvent
    from nvitop.tui.screens.base import BaseScreen, BaseSelectableScreen
    from nvitop.tui.tui import TUI


__all__ = ['MessageBox']


DIGITS: frozenset[str] = frozenset(string.digits)


class MessageBox(Displayable):  # pylint: disable=too-many-instance-attributes
    class Option:  # pylint: disable=too-few-public-methods
        # pylint: disable-next=too-many-arguments
        def __init__(
            self,
            name: str,
            key: str,
            callback: Callable[[], None] | None,
            *,
            keys: Iterable[str] = (),
            attrs: tuple[dict[str, int | str], ...] = (),
        ) -> None:
            self.name: WideString = WideString(name)
            self.offset: int = 0
            self.key: str = normalize_keybinding(key)
            self.callback: Callable[[], None] | None = callback
            self.keys: tuple[str, ...] = tuple(
                set(map(normalize_keybinding, keys)).difference({self.key}),
            )
            self.attrs: tuple[dict[str, int | str], ...] = attrs

        def __call__(self) -> None:
            if self.callback is not None:
                self.callback()

        def __str__(self) -> str:
            return str(self.name)

    root: TUI
    parent: TUI

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        message: str,
        options: list[MessageBox.Option],
        *,
        default: int | None,
        yes: int | None,
        no: int | None,
        cancel: int,
        win: curses.window,
        root: TUI,
    ) -> None:
        super().__init__(win, root)

        if default is None:
            default = 0
        if no is None:
            no = cancel

        self.options: list[MessageBox.Option] = options
        self.num_options: int = len(self.options)

        assert cancel is not None
        assert self.num_options >= 2
        assert 0 <= no < self.num_options
        assert 0 <= cancel < self.num_options
        assert 0 <= default < self.num_options

        self.previous_focused: BaseScreen | None = None
        self.message: str = message
        self.previous_keymap: str = root.keymaps.used_keymap  # type: ignore[assignment]
        self.current: int = default
        self.yes: int | None = yes
        self.cancel: int = cancel
        self.no: int = no  # pylint: disable=invalid-name
        self.timestamp: float = time.monotonic()

        self.name_len: int = max(8, *(len(option.name) for option in options))
        for option in self.options:
            option.offset = (self.name_len - len(option.name)) // 2
            option.name = option.name.center(self.name_len)

        self.xy_mouse: tuple[int, int] | None = None
        self.x, self.y = root.x, root.y
        self.width: int = (self.name_len + 6) * self.num_options + 6

        self.init_keybindings()

        lines: list[str | WideString] = []
        for msg in self.message.splitlines():
            words = iter(map(WideString, msg.split()))
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
            lines[-1] = WideString(lines[-1]).center(self.width - 6)
        raw_lines = [f' │ {line.ljust(self.width - 6)} │ ' for line in lines]
        raw_lines = [
            ' ╒' + '═' * (self.width - 4) + '╕ ',
            ' │' + ' ' * (self.width - 4) + '│ ',
            *raw_lines,
            ' │' + ' ' * (self.width - 4) + '│ ',
            ' │  ' + '  '.join(['┌' + '─' * (self.name_len + 2) + '┐'] * self.num_options) + '  │ ',
            ' │  ' + '  '.join(map('│ {} │'.format, self.options)) + '  │ ',
            ' │  ' + '  '.join(['└' + '─' * (self.name_len + 2) + '┘'] * self.num_options) + '  │ ',
            ' ╘' + '═' * (self.width - 4) + '╛ ',
        ]
        self.lines: list[str] = raw_lines

    @property
    def current(self) -> int:
        return self._current

    @current.setter
    def current(self, value: int) -> None:
        self._current = value
        self.timestamp = time.monotonic()

    def draw(self) -> None:  # pylint: disable=too-many-locals
        self.set_base_attr(attr=0)
        self.color_reset()

        assert self.root.termsize is not None
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
                y = y_option_start + attr.pop('y')  # type: ignore[operator]
                x = x_option_start + attr.pop('x')  # type: ignore[operator]
                width: int = attr.pop('width')  # type: ignore[assignment]
                self.color_at(y, x, width, **attr)

        if self.xy_mouse is not None:
            x, y = self.xy_mouse
            if y_option_start - 1 <= y <= y_option_start + 1:
                current = (x - x_start - 3) // (self.name_len + 6)
                x_option_start = x_start + 6 + current * (self.name_len + 6)
                if (
                    0 <= current < self.num_options
                    and x_option_start - 3 <= x < x_option_start + self.name_len + 3
                ):
                    self.apply(current, wait=True)

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
            y = y_option_start + attr.pop('y')  # type: ignore[operator]
            x = x_option_start + option.offset + attr.pop('x')  # type: ignore[operator]
            width = attr.pop('width')  # type: ignore[assignment]
            attr['fg'], attr['bg'] = attr.get('bg', -1), attr.get('fg', -1)
            attr['attr'] = self.get_fg_bg_attr(attr=attr.get('attr', 0))
            attr['attr'] |= self.get_fg_bg_attr(attr='standout | bold')  # type: ignore[operator]
            self.color_at(y, x, width, **attr)

    def finalize(self) -> None:
        self.xy_mouse = None
        super().finalize()

    def press(self, key: int) -> bool:
        self.root.keymaps.use_keymap('messagebox')
        return self.root.press(key)

    def click(self, event: MouseEvent) -> bool:
        if event.pressed(1) or event.pressed(3) or event.clicked(1) or event.clicked(3):
            self.xy_mouse = (event.x, event.y)
            return True

        direction = event.wheel_direction()
        self.current = (self.current + direction) % self.num_options
        return True

    def apply(self, index: int | None = None, wait: bool | None = None) -> None:
        if index is None:
            index = self.current

        assert 0 <= index < self.num_options

        if (index != self.current and wait is None) or wait:
            self.current = index

            def confirm() -> None:
                time.sleep(0.25)
                curses.ungetch(curses.KEY_ENTER)

            threading.Thread(name='messagebox-confirm', target=confirm, daemon=True).start()
            return

        option = self.options[index]
        option()

        self.root.keymaps.clear_keymap('messagebox')
        self.root.keymaps.use_keymap(self.previous_keymap)
        self.root.need_redraw = True
        self.root.messagebox = None

    def init_keybindings(self) -> None:  # pylint: disable=too-many-branches
        def select_previous() -> None:
            self.current = (self.current - 1) % self.num_options

        def select_next() -> None:
            self.current = (self.current + 1) % self.num_options

        keymaps = self.root.keymaps
        keymap = keymaps.clear_keymap('messagebox')

        for i, option in enumerate(self.options):
            keymaps.bind('messagebox', option.key, partial(self.apply, index=i))
            for key in option.keys:
                keymaps.alias('messagebox', option.key, key)

        keymap[keymaps.keybuffer.quantifier_key] = keymaps.keybuffer.QUANTIFIER_KEY_FINISHED  # type: ignore[assignment]
        if len(DIGITS.intersection(keymap)) == 0 and self.num_options <= 9:
            for key_n, option in zip('123456789', self.options):
                keymaps.alias('messagebox', option.key, key_n)

        assert set(keymap).isdisjoint(
            NAMED_SPECIAL_KEYS[key] for key in ('Enter', 'Esc', 'Left', 'Right')
        )

        if self.yes is not None and ord('y') not in keymap:
            keymaps.alias('messagebox', self.options[self.yes].key, 'y')
            if ord('Y') not in keymap:
                keymaps.alias('messagebox', self.options[self.yes].key, 'Y')
        if self.no is not None and ord('n') not in keymap:
            keymaps.alias('messagebox', self.options[self.no].key, 'n')
            if ord('N') not in keymap:
                keymaps.alias('messagebox', self.options[self.no].key, 'N')
        if self.cancel is not None:
            keymaps.bind('messagebox', '<Esc>', partial(self.apply, index=self.cancel, wait=False))
            if ord('q') not in keymap and ord('Q') not in keymap:
                keymaps.alias('messagebox', '<Esc>', 'q')
                keymaps.alias('messagebox', '<Esc>', 'Q')

        keymaps.bind('messagebox', '<Enter>', self.apply)
        if NAMED_SPECIAL_KEYS['Space'] not in keymap:
            keymaps.alias('messagebox', '<Enter>', '<Space>')

        keymaps.bind('messagebox', '<Left>', select_previous)
        keymaps.bind('messagebox', '<Right>', select_next)
        if ord(',') not in keymap and ord('.') not in keymap:
            keymaps.alias('messagebox', '<Left>', ',')
            keymaps.alias('messagebox', '<Right>', '.')
        if ord('<') not in keymap and ord('>') not in keymap:
            keymaps.alias('messagebox', '<Left>', '<')
            keymaps.alias('messagebox', '<Right>', '>')
        if ord('[') not in keymap and ord(']') not in keymap:
            keymaps.alias('messagebox', '<Left>', '[')
            keymaps.alias('messagebox', '<Right>', ']')
        if NAMED_SPECIAL_KEYS['Tab'] not in keymap and NAMED_SPECIAL_KEYS['S-Tab'] not in keymap:
            keymaps.alias('messagebox', '<Left>', '<S-Tab>')
            keymaps.alias('messagebox', '<Right>', '<Tab>')

    @staticmethod
    def confirm_sending_signal_to_processes(
        signal: Literal['terminate', 'kill', 'interrupt'],
        screen: BaseSelectableScreen,
    ) -> None:
        assert signal in ('terminate', 'kill', 'interrupt')
        default = {'terminate': 0, 'kill': 1, 'interrupt': 2}.get(signal)
        processes = []
        for process in screen.selection.processes():
            try:
                username = process.username()
            except host.PsutilError:
                username = 'N/A'
            username = cut_string(username, maxlen=24, padstr='+')
            processes.append(f'{process.pid}({username})')
        if len(processes) == 0:
            return
        if len(processes) == 1:
            message = f'Send signal to process {processes[0]}?'
        else:
            maxlen = max(map(len, processes))
            processes = [process.ljust(maxlen) for process in processes]
            message = 'Send signal to the following processes?\n\n{}'.format(' '.join(processes))

        screen.root.messagebox = MessageBox(
            message=message,
            options=[
                MessageBox.Option(
                    'SIGTERM',
                    't',
                    screen.selection.terminate,
                    keys=('T',),
                    attrs=(
                        {'y': 0, 'x': 0, 'width': 7, 'fg': 'red'},
                        {'y': 0, 'x': 3, 'width': 1, 'fg': 'red', 'attr': 'bold | underline'},
                    ),
                ),
                MessageBox.Option(
                    'SIGKILL',
                    'k',
                    screen.selection.kill,
                    keys=('K',),
                    attrs=(
                        {'y': 0, 'x': 0, 'width': 7, 'fg': 'red'},
                        {'y': 0, 'x': 3, 'width': 1, 'fg': 'red', 'attr': 'bold | underline'},
                    ),
                ),
                MessageBox.Option(
                    'SIGINT',
                    'i',
                    screen.selection.interrupt,
                    keys=('I',),
                    attrs=(
                        {'y': 0, 'x': 0, 'width': 6, 'fg': 'red'},
                        {'y': 0, 'x': 3, 'width': 1, 'fg': 'red', 'attr': 'bold | underline'},
                    ),
                ),
                MessageBox.Option(
                    'Cancel',
                    'c',
                    None,
                    keys=('C',),
                    attrs=({'y': 0, 'x': 0, 'width': 1, 'attr': 'bold | underline'},),
                ),
            ],
            default=default,
            yes=None,
            no=3,
            cancel=3,
            win=screen.win,  # type: ignore[arg-type]
            root=screen.root,
        )
