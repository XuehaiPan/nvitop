# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import threading
from functools import partial
from typing import TYPE_CHECKING, ClassVar, NoReturn

from nvitop.tui.library import (
    LARGE_INTEGER,
    Device,
    Displayable,
    MessageBox,
    MouseEvent,
    Selection,
    Snapshot,
)
from nvitop.tui.screens.base import BaseScreen, BaseSelectableScreen
from nvitop.tui.screens.main.panels import DevicePanel, HostPanel, OrderName, ProcessPanel


if TYPE_CHECKING:
    import curses
    from collections.abc import Callable, Iterable

    from nvitop.tui.tui import TUI, MonitorMode


__all__ = ['BreakLoop', 'MainScreen']


class BreakLoop(Exception):  # noqa: N818
    pass


class MainScreen(BaseSelectableScreen):  # pylint: disable=too-many-instance-attributes
    NAME: ClassVar[str] = 'main'

    # pylint: disable-next=too-many-arguments,too-many-locals,too-many-statements
    def __init__(
        self,
        devices: list[Device],
        filters: Iterable[Callable[[Snapshot], bool]],
        *,
        no_unicode: bool,
        mode: MonitorMode,
        win: curses.window | None,
        root: TUI,
    ) -> None:
        super().__init__(win, root)

        self.width: int = root.width

        assert mode in {'auto', 'full', 'compact'}
        compact: bool = mode == 'compact'
        self.mode: MonitorMode = mode
        self._compact: bool = compact

        self.devices: list[Device] = devices
        self.device_count: int = len(self.devices)

        self.snapshot_lock = threading.Lock()

        self.device_panel: DevicePanel = DevicePanel(
            self.devices,
            compact,
            win=win,
            root=root,
        )
        self.device_panel.focused = False
        self.add_child(self.device_panel)

        self.host_panel: HostPanel = HostPanel(
            self.device_panel.leaf_devices,
            compact,
            win=win,
            root=root,
        )
        self.host_panel.focused = False
        self.add_child(self.host_panel)

        self.process_panel: ProcessPanel = ProcessPanel(
            self.device_panel.leaf_devices,
            compact,
            filters,
            win=win,
            root=root,
        )
        self.process_panel.focused = False
        self.add_child(self.process_panel)

        self.selection: Selection = self.process_panel.selection

        self.no_unicode: bool = no_unicode
        self.device_panel.no_unicode = self.no_unicode
        self.host_panel.no_unicode = self.no_unicode
        self.process_panel.no_unicode = self.no_unicode
        if no_unicode:
            self.host_panel.full_height = self.host_panel.height = self.host_panel.compact_height

        self.x, self.y = root.x, root.y
        self.device_panel.x = self.host_panel.x = self.process_panel.x = self.x
        self.device_panel.y = self.y
        self.host_panel.y = self.device_panel.y + self.device_panel.height
        self.process_panel.y = self.host_panel.y + self.host_panel.height
        self.height = self.device_panel.height + self.host_panel.height + self.process_panel.height

    @property
    def compact(self) -> bool:
        return self._compact

    @compact.setter
    def compact(self, value: bool) -> None:
        if self._compact != value:
            self.need_redraw = True
            self._compact = value

    def update_size(self, termsize: tuple[int, int] | None = None) -> tuple[int, int]:
        n_term_lines, n_term_cols = termsize = super().update_size(termsize=termsize)

        self.width = n_term_cols - self.x
        self.device_panel.width = self.width
        self.host_panel.width = self.width
        self.process_panel.width = self.width

        self.y = min(self.y, self.root.y)
        height = n_term_lines - self.y
        heights = [
            self.device_panel.full_height
            + self.host_panel.full_height
            + self.process_panel.full_height,
            self.device_panel.compact_height
            + self.host_panel.full_height
            + self.process_panel.full_height,
            self.device_panel.compact_height
            + self.host_panel.compact_height
            + self.process_panel.full_height,
        ]
        if self.mode == 'auto':
            self.compact = height < heights[0]
            self.host_panel.compact = height < heights[1]
            self.process_panel.compact = height < heights[-1]
        else:
            self.compact = self.mode == 'compact'
            self.host_panel.compact = self.compact
            self.process_panel.compact = self.compact
        self.device_panel.compact = self.compact

        self.device_panel.y = self.y
        self.host_panel.y = self.device_panel.y + self.device_panel.height
        self.process_panel.y = self.host_panel.y + self.host_panel.height
        height = self.device_panel.height + self.host_panel.height + self.process_panel.height

        if self.y < self.root.y and self.y + height < n_term_lines:
            self.y = min(self.root.y + self.root.height - height, self.root.y)
            self.update_size(termsize)
            self.need_redraw = True

        if self.height != height:
            self.height = height
            self.need_redraw = True

        return termsize

    def move(self, direction: int = 0) -> None:
        if direction == 0:
            return

        self.y -= direction
        self.update_size()
        self.need_redraw = True

    def poke(self) -> None:
        super().poke()

        height = self.device_panel.height + self.host_panel.height + self.process_panel.height
        if self.height != height:
            self.update_size()
            self.need_redraw = True

    def draw(self) -> None:
        self.color_reset()

        super().draw()

    def print(self) -> None:
        if self.device_count > 0:
            print_width = min(panel.print_width() for panel in self.container)
            self.width = max(print_width, min(self.width, 100))
        else:
            self.width = 79
        for panel in self.container:
            panel.width = self.width
            panel.print()

    def __contains__(self, item: Displayable | MouseEvent | tuple[int, int]) -> bool:
        if self.visible and isinstance(item, MouseEvent):
            return True
        return super().__contains__(item)

    def init_keybindings(self) -> None:
        # pylint: disable=too-many-locals,too-many-statements

        def quit() -> NoReturn:  # pylint: disable=redefined-builtin
            raise BreakLoop

        def change_mode(mode: MonitorMode) -> None:
            self.mode = mode
            self.root.update_size()

        def force_refresh() -> None:
            select_clear()
            host_begin()
            self.y = self.root.y
            self.root.update_size()
            self.root.need_redraw = True

        def screen_move(direction: int) -> None:
            self.move(direction)

        def host_left() -> None:
            self.process_panel.host_offset -= 2

        def host_right() -> None:
            self.process_panel.host_offset += 2

        def host_begin() -> None:
            self.process_panel.host_offset = -1

        def host_end() -> None:
            self.process_panel.host_offset = LARGE_INTEGER

        def select_move(direction: int) -> None:
            self.selection.move(direction=direction)

        def select_clear() -> None:
            self.selection.clear()

        def tag() -> None:
            self.selection.tag()
            select_move(direction=+1)

        def sort_by(order: OrderName, reverse: bool) -> None:
            self.process_panel.order = order
            self.process_panel.reverse = reverse
            self.root.update_size()

        def order_previous() -> None:
            sort_by(order=ProcessPanel.ORDERS[self.process_panel.order].previous, reverse=False)

        def order_next() -> None:
            sort_by(order=ProcessPanel.ORDERS[self.process_panel.order].next, reverse=False)

        def order_reverse() -> None:
            sort_by(order=self.process_panel.order, reverse=not self.process_panel.reverse)

        keymaps = self.root.keymaps

        keymaps.bind('main', 'q', quit)
        keymaps.alias('main', 'q', 'Q')
        keymaps.bind('main', 'a', partial(change_mode, mode='auto'))
        keymaps.bind('main', 'f', partial(change_mode, mode='full'))
        keymaps.bind('main', 'c', partial(change_mode, mode='compact'))
        keymaps.bind('main', 'r', force_refresh)
        keymaps.alias('main', 'r', 'R')
        keymaps.alias('main', 'r', '<C-r>')
        keymaps.alias('main', 'r', '<F5>')

        keymaps.bind('main', '<PageUp>', partial(screen_move, direction=-1))
        keymaps.alias('main', '<PageUp>', '[')
        keymaps.alias('main', '<PageUp>', '<A-K>')
        keymaps.bind('main', '<PageDown>', partial(screen_move, direction=+1))
        keymaps.alias('main', '<PageDown>', ']')
        keymaps.alias('main', '<PageDown>', '<A-J>')

        keymaps.bind('main', '<Left>', host_left)
        keymaps.alias('main', '<Left>', '<A-h>')
        keymaps.bind('main', '<Right>', host_right)
        keymaps.alias('main', '<Right>', '<A-l>')
        keymaps.bind('main', '<C-a>', host_begin)
        keymaps.alias('main', '<C-a>', '^')
        keymaps.bind('main', '<C-e>', host_end)
        keymaps.alias('main', '<C-e>', '$')
        keymaps.bind('main', '<Up>', partial(select_move, direction=-1))
        keymaps.alias('main', '<Up>', '<S-Tab>')
        keymaps.alias('main', '<Up>', '<A-k>')
        keymaps.bind('main', '<Down>', partial(select_move, direction=+1))
        keymaps.alias('main', '<Down>', '<Tab>')
        keymaps.alias('main', '<Down>', '<A-j>')
        keymaps.bind('main', '<Home>', partial(select_move, direction=-(1 << 20)))
        keymaps.bind('main', '<End>', partial(select_move, direction=+(1 << 20)))
        keymaps.bind('main', '<Esc>', select_clear)
        keymaps.bind('main', '<Space>', tag)

        keymaps.bind(
            'main',
            'T',
            partial(
                MessageBox.confirm_sending_signal_to_processes,
                signal='terminate',
                screen=self,
            ),
        )
        keymaps.bind(
            'main',
            'K',
            partial(
                MessageBox.confirm_sending_signal_to_processes,
                signal='kill',
                screen=self,
            ),
        )
        keymaps.alias('main', 'K', 'k')
        keymaps.bind(
            'main',
            '<C-c>',
            partial(
                MessageBox.confirm_sending_signal_to_processes,
                signal='interrupt',
                screen=self,
            ),
        )
        keymaps.alias('main', '<C-c>', 'I')

        keymaps.bind('main', ',', order_previous)
        keymaps.alias('main', ',', '<')
        keymaps.bind('main', '.', order_next)
        keymaps.alias('main', '.', '>')
        keymaps.bind('main', '/', order_reverse)
        for name, order in ProcessPanel.ORDERS.items():
            keymaps.bind(
                'main',
                f'o{order.bind_key.lower()}',
                partial(sort_by, order=name, reverse=False),
            )
            keymaps.bind(
                'main',
                f'o{order.bind_key.upper()}',
                partial(sort_by, order=name, reverse=True),
            )
