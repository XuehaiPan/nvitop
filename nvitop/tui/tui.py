# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import curses
import shutil
import time
from typing import TYPE_CHECKING, Literal, Union

from nvitop.tui.library import (
    ALT_KEY,
    Device,
    DisplayableContainer,
    KeyBuffer,
    KeyMaps,
    MessageBox,
    MouseEvent,
    Snapshot,
)
from nvitop.tui.screens import (
    BaseScreen,
    BreakLoop,
    EnvironScreen,
    HelpScreen,
    MainScreen,
    ProcessMetricsScreen,
    TreeViewScreen,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = ['TUI', 'MonitorMode']


MonitorMode = Literal['auto', 'full', 'compact']


class TUI(DisplayableContainer[Union[BaseScreen, MessageBox]]):  # pylint: disable=too-many-instance-attributes
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        devices: list[Device],
        filters: Iterable[Callable[[Snapshot], bool]] = (),
        *,
        no_unicode: bool = False,
        mode: MonitorMode = 'auto',
        interval: float | None = None,
        win: curses.window | None = None,
    ) -> None:
        super().__init__(win, root=self)

        self.x = self.y = 0
        self.width: int = max(79, shutil.get_terminal_size(fallback=(79, 24)).columns - self.x)
        self.height: int = 0
        self.termsize: tuple[int, int] | None = None

        self.no_unicode: bool = no_unicode

        self.devices: list[Device] = devices
        self.device_count: int = len(self.devices)

        self.main_screen: MainScreen = MainScreen(
            self.devices,
            filters,
            no_unicode=no_unicode,
            mode=mode,
            win=win,
            root=self,
        )
        self.main_screen.visible = True
        self.main_screen.focused = False
        self.add_child(self.main_screen)
        self.current_screen: BaseScreen = self.main_screen
        self.previous_screen: BaseScreen = self.main_screen

        self._messagebox: MessageBox | None = None

        if win is not None:
            self.environ_screen: EnvironScreen = EnvironScreen(win=win, root=self)
            self.environ_screen.visible = False
            self.environ_screen.no_unicode = False
            self.add_child(self.environ_screen)

            self.treeview_screen: TreeViewScreen = TreeViewScreen(win=win, root=self)
            self.treeview_screen.visible = False
            self.treeview_screen.no_unicode = self.no_unicode
            self.add_child(self.treeview_screen)

            self.process_metrics_screen: ProcessMetricsScreen = ProcessMetricsScreen(
                win=win,
                root=self,
            )
            self.process_metrics_screen.visible = False
            self.process_metrics_screen.no_unicode = self.no_unicode
            self.add_child(self.process_metrics_screen)

            self.help_screen: HelpScreen = HelpScreen(win=win, root=self)
            self.help_screen.visible = False
            self.help_screen.no_unicode = False
            self.add_child(self.help_screen)

            if interval is not None:
                if interval < 1.0:
                    self.main_screen.device_panel.set_snapshot_interval(interval)
                    self.main_screen.host_panel.set_snapshot_interval(interval)
                if interval < 0.5:
                    self.process_metrics_screen.set_snapshot_interval(interval)
                self.main_screen.process_panel.set_snapshot_interval(interval)
                self.treeview_screen.set_snapshot_interval(interval)

            self.keybuffer: KeyBuffer = KeyBuffer()
            self.keymaps: KeyMaps = KeyMaps(self.keybuffer)
            self.last_input_time: float = time.monotonic()
            self.init_keybindings()

    @property
    def messagebox(self) -> MessageBox | None:
        return self._messagebox

    @messagebox.setter
    def messagebox(self, value: MessageBox | None) -> None:
        self.need_redraw = True
        if self._messagebox is not None:
            self.remove_child(self._messagebox)

        self._messagebox = value
        if value is not None:
            value.visible = True
            value.focused = True
            value.no_unicode = self.no_unicode
            value.previous_focused = self.get_focused_obj()  # type: ignore[assignment]
            self.add_child(value)

    def get_focused_obj(self) -> BaseScreen | MessageBox | None:
        if self.messagebox is not None:
            return self.messagebox
        return super().get_focused_obj()

    def update_size(self, termsize: tuple[int, int] | None = None) -> tuple[int, int]:
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

    def poke(self) -> None:
        super().poke()

        if self.termsize is None:
            self.update_size()

    def draw(self) -> None:
        assert self.win is not None
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

        assert self.termsize is not None
        n_term_lines, n_term_cols = self.termsize
        message = (
            f'nvitop needs at least a width of 79 to render, the current width is {self.width}.'
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
        lines = [f'│ {line.ljust(width - 4)} │' for line in lines]
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

    def finalize(self) -> None:
        assert self.win is not None
        super().finalize()
        self.win.refresh()

    def redraw(self) -> None:
        self.poke()
        self.draw()
        self.finalize()

    def loop(self) -> None:
        if self.win is None:
            return

        try:
            while True:
                self.redraw()
                self.handle_input()
                if time.monotonic() - self.last_input_time > 1.0:
                    time.sleep(0.2)
        except BreakLoop:
            pass

    def print(self) -> None:
        self.main_screen.print()

    def handle_mouse(self) -> None:
        """Handle mouse input."""
        try:
            event = MouseEvent.get()
        except curses.error:
            return
        super().click(event)

    def handle_key(self, key: int) -> None:
        """Handle key input."""
        if key < 0:
            self.keybuffer.clear()
        elif not super().press(key):
            self.keymaps.use_keymap('main')
            self.press(key)

    def handle_keys(self, *keys: int) -> None:
        for key in keys:
            self.handle_key(key)

    def press(self, key: int) -> bool:
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

    def handle_input(self) -> None:  # pylint: disable=too-many-branches
        assert self.win is not None
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

    def init_keybindings(self) -> None:
        # pylint: disable=multiple-statements,too-many-statements

        for screen in self.container:
            if hasattr(screen, 'init_keybindings'):
                screen.init_keybindings()

        def show_screen(screen: BaseScreen, focused: bool | None = None) -> None:
            for s in self.container:
                if s is screen:
                    s.visible = True
                    if focused is not None:
                        s.focused = focused
                else:
                    s.visible = False

            self.previous_screen = self.current_screen
            self.current_screen = screen

        def show_main() -> None:
            target_screen = self.main_screen
            show_screen(target_screen, focused=False)

            if self.treeview_screen.selection.is_set():
                target_screen.selection.process = self.treeview_screen.selection.process
            self.treeview_screen.selection.clear()
            self.process_metrics_screen.disable()

        def show_environ() -> None:
            target_screen = self.environ_screen
            show_screen(target_screen, focused=True)

            if self.previous_screen is not self.help_screen:
                target_screen.process = self.previous_screen.selection.process  # type: ignore[attr-defined]

        def environ_return() -> None:
            if self.previous_screen is self.treeview_screen:
                show_treeview()
            elif self.previous_screen is self.process_metrics_screen:
                show_process_metrics()
            else:
                show_main()

        def show_treeview() -> None:
            if not self.main_screen.process_panel.has_snapshots:
                return

            target_screen = self.treeview_screen
            show_screen(target_screen, focused=True)

            if not target_screen.selection.is_set():
                target_screen.selection.process = self.main_screen.selection.process
            self.main_screen.selection.clear()

        def show_process_metrics() -> None:
            target_screen = self.process_metrics_screen
            if self.current_screen is self.main_screen:
                if self.main_screen.selection.is_set():
                    show_screen(target_screen, focused=True)
                    target_screen.process = self.previous_screen.selection.process  # type: ignore[attr-defined]
            elif self.current_screen is not self.treeview_screen:
                show_screen(target_screen, focused=True)

        def show_help() -> None:
            show_screen(self.help_screen, focused=True)

        def help_return() -> None:
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
        self.keymaps.alias('environ', 'e', '<Esc>')
        self.keymaps.alias('environ', 'e', 'q')
        self.keymaps.alias('environ', 'e', 'Q')

        self.keymaps.bind('main', 't', show_treeview)
        self.keymaps.bind('treeview', 't', show_main)
        self.keymaps.alias('treeview', 't', 'q')
        self.keymaps.alias('treeview', 't', 'Q')
        self.keymaps.bind('treeview', 'e', show_environ)

        self.keymaps.bind('main', '<Enter>', show_process_metrics)
        self.keymaps.bind('process-metrics', '<Enter>', show_main)
        self.keymaps.alias('process-metrics', '<Enter>', '<Esc>')
        self.keymaps.alias('process-metrics', '<Enter>', 'q')
        self.keymaps.alias('process-metrics', '<Enter>', 'Q')
        self.keymaps.bind('process-metrics', 'e', show_environ)

        for screen_name in ('main', 'treeview', 'environ', 'process-metrics'):
            self.keymaps.bind(screen_name, 'h', show_help)
            self.keymaps.alias(screen_name, 'h', '?')
        self.keymaps.bind('help', '<Esc>', help_return)
        self.keymaps.bind('help', '<any>', help_return)

        self.keymaps.use_keymap('main')
