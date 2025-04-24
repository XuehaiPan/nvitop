# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, ClassVar

from nvitop.tui.library import (
    NA,
    BufferedHistoryGraph,
    Device,
    GiB,
    HistoryGraph,
    MigDevice,
    NaType,
    bytes2human,
    colored,
    host,
    make_bar,
    timedelta2human,
)
from nvitop.tui.screens.main.panels.base import BasePanel


if TYPE_CHECKING:
    import curses

    from nvitop.tui.tui import TUI


__all__ = ['HostPanel']


class HostPanel(BasePanel):  # pylint: disable=too-many-instance-attributes
    NAME: ClassVar[str] = 'host'
    SNAPSHOT_INTERVAL: ClassVar[float] = 0.5

    def __init__(
        self,
        devices: list[Device | MigDevice],
        compact: bool,
        *,
        win: curses.window | None,
        root: TUI,
    ) -> None:
        super().__init__(win, root)

        self.devices: list[Device | MigDevice] = devices
        self.device_count: int = len(self.devices)

        if win is not None:
            self.average_gpu_memory_percent: HistoryGraph | None = None
            self.average_gpu_utilization: HistoryGraph | None = None
            self.enable_history()

        self._compact: bool = compact
        self.width: int = max(79, root.width)
        self.full_height: int = 12
        self.compact_height: int = 2
        self.height: int = self.compact_height if compact else self.full_height

        self.cpu_percent: float = NA  # type: ignore[assignment]
        self.load_average: tuple[float, float, float] = (NA, NA, NA)  # type: ignore[assignment]
        self.virtual_memory: host.VirtualMemory = host.VirtualMemory()
        self.swap_memory: host.SwapMemory = host.SwapMemory()
        self._snapshot_daemon = threading.Thread(
            name='host-snapshot-daemon',
            target=self._snapshot_target,
            daemon=True,
        )
        self._daemon_running = threading.Event()

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int) -> None:
        width = max(79, value)
        if self._width != width:
            if self.visible:
                self.need_redraw = True
            graph_width = max(width - 80, 20)
            if self.win is not None:
                self.average_gpu_memory_percent.width = graph_width  # type: ignore[union-attr]
                self.average_gpu_utilization.width = graph_width  # type: ignore[union-attr]
                for device in self.devices:
                    device.memory_percent.history.width = graph_width  # type: ignore[attr-defined]
                    device.gpu_utilization.history.width = graph_width  # type: ignore[attr-defined]
        self._width = width

    @property
    def compact(self) -> bool:
        return self._compact or self.no_unicode

    @compact.setter
    def compact(self, value: bool) -> None:
        value = value or self.no_unicode
        if self._compact != value:
            self.need_redraw = True
            self._compact = value
            self.height = self.compact_height if self.compact else self.full_height

    def enable_history(self) -> None:
        host.cpu_percent = BufferedHistoryGraph(
            interval=1.0,
            width=77,
            height=5,
            upsidedown=False,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format='CPU: {:.1f}%'.format,
        )(host.cpu_percent)
        host.virtual_memory = BufferedHistoryGraph(
            interval=1.0,
            width=77,
            height=4,
            upsidedown=True,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format='{:.1f}%'.format,
        )(host.virtual_memory, get_value=lambda vm: vm.percent)
        host.swap_memory = BufferedHistoryGraph(
            interval=1.0,
            width=77,
            height=1,
            upsidedown=False,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format='{:.1f}%'.format,
        )(host.swap_memory, get_value=lambda sm: sm.percent)

        def percentage(x: float | NaType) -> str:
            return f'{x:.1f}%' if x is not NA else NA

        def enable_history(device: Device) -> None:
            device.memory_percent = BufferedHistoryGraph(  # type: ignore[method-assign]
                interval=1.0,
                width=20,
                height=5,
                upsidedown=False,
                baseline=0.0,
                upperbound=100.0,
                dynamic_bound=False,
                format=lambda x: f'GPU {device.display_index} MEM: {percentage(x)}',
            )(device.memory_percent)
            device.gpu_utilization = BufferedHistoryGraph(  # type: ignore[method-assign]
                interval=1.0,
                width=20,
                height=5,
                upsidedown=True,
                baseline=0.0,
                upperbound=100.0,
                dynamic_bound=False,
                format=lambda x: f'GPU {device.display_index} UTL: {percentage(x)}',
            )(device.gpu_utilization)

        for device in self.devices:
            enable_history(device)

        prefix = 'AVG ' if self.device_count > 1 else ''
        self.average_gpu_memory_percent = BufferedHistoryGraph(
            interval=1.0,
            width=20,
            height=5,
            upsidedown=False,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format=lambda x: f'{prefix}GPU MEM: {percentage(x)}',
        )
        self.average_gpu_utilization = BufferedHistoryGraph(
            interval=1.0,
            width=20,
            height=5,
            upsidedown=True,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format=lambda x: f'{prefix}GPU UTL: {percentage(x)}',
        )

    @classmethod
    def set_snapshot_interval(cls, interval: float) -> None:
        assert interval > 0.0
        interval = float(interval)

        cls.SNAPSHOT_INTERVAL = min(interval / 3.0, 0.5)

    def take_snapshots(self) -> None:
        host.cpu_percent()
        host.virtual_memory()
        host.swap_memory()
        self.load_average = host.load_average()

        self.cpu_percent = host.cpu_percent.history.last_value
        self.virtual_memory = host.virtual_memory.history.last_retval  # type: ignore[attr-defined]
        self.swap_memory = host.swap_memory.history.last_retval  # type: ignore[attr-defined]

        total_memory_used = 0
        total_memory_total = 0
        gpu_utilizations = []
        for device in self.devices:
            memory_used = device.snapshot.memory_used
            memory_total = device.snapshot.memory_total
            gpu_utilization = device.snapshot.gpu_utilization
            if memory_used is not NA and memory_total is not NA:
                total_memory_used += memory_used
                total_memory_total += memory_total
            if gpu_utilization is not NA:
                gpu_utilizations.append(float(gpu_utilization))
        if total_memory_total > 0:
            avg = 100.0 * total_memory_used / total_memory_total
            self.average_gpu_memory_percent.add(avg)  # type: ignore[union-attr]
        if len(gpu_utilizations) > 0:
            avg = sum(gpu_utilizations) / len(gpu_utilizations)
            self.average_gpu_utilization.add(avg)  # type: ignore[union-attr]

    def _snapshot_target(self) -> None:
        self._daemon_running.wait()
        while self._daemon_running.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def frame_lines(self, compact: bool | None = None) -> list[str]:
        if compact is None:
            compact = self.compact
        if compact or self.no_unicode:
            return []

        remaining_width = self.width - 79
        data_line = (
            '│                                                                             │'
        )
        separator_line = (
            '├────────────╴120s├─────────────────────────╴60s├──────────╴30s├──────────────┤'
        )
        if self.width >= 100:
            data_line += ' ' * (remaining_width - 1) + '│'
            separator_line = separator_line[:-1] + '┼' + '─' * (remaining_width - 1) + '┤'

        frame = [
            '╞═══════════════════════════════╧══════════════════════╧══════════════════════╡',
            data_line,
            data_line,
            data_line,
            data_line,
            data_line,
            separator_line,
            data_line,
            data_line,
            data_line,
            data_line,
            data_line,
            '╘═════════════════════════════════════════════════════════════════════════════╛',
        ]
        if self.width >= 100:
            frame[0] = frame[0][:-1] + '╪' + '═' * (remaining_width - 1) + '╡'
            frame[-1] = frame[-1][:-1] + '╧' + '═' * (remaining_width - 1) + '╛'

        return frame

    def poke(self) -> None:
        if not self._daemon_running.is_set():
            self._daemon_running.set()
            self._snapshot_daemon.start()
            self.take_snapshots()

        super().poke()

    def draw(self) -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self.color_reset()

        load_average = 'Load Average: {} {} {}'.format(
            *(f'{value:5.2f}'[:5] if value < 10000.0 else '9999+' for value in self.load_average),
        )

        if self.compact:
            width_right = len(load_average) + 4
            width_left = self.width - 2 - width_right
            cpu_bar = '[ {} ]'.format(
                make_bar(
                    'CPU',
                    self.cpu_percent,
                    width_left - 4,
                    extra_text=f'  UPTIME: {timedelta2human(host.uptime(), round=True)}',
                ),
            )
            memory_bar = '[ {} ]'.format(
                make_bar(
                    'MEM',
                    self.virtual_memory.percent,
                    width_left - 4,
                    extra_text=f'  USED: {bytes2human(self.virtual_memory.used, min_unit=GiB)}',
                ),
            )
            swap_bar = '[ {} ]'.format(make_bar('SWP', self.swap_memory.percent, width_right - 4))
            self.addstr(self.y, self.x, f'{cpu_bar}  ( {load_average} )')
            self.addstr(self.y + 1, self.x, f'{memory_bar}  {swap_bar}')
            self.color_at(self.y, self.x, width=len(cpu_bar), fg='cyan', attr='bold')
            self.color_at(self.y + 1, self.x, width=width_left, fg='magenta', attr='bold')
            self.color_at(self.y, self.x + width_left + 2, width=width_right, attr='bold')
            self.color_at(
                self.y + 1,
                self.x + width_left + 2,
                width=width_right,
                fg='blue',
                attr='bold',
            )
            return

        remaining_width = self.width - 79

        if self.need_redraw:
            for y, line in enumerate(self.frame_lines(), start=self.y - 1):
                self.addstr(y, self.x, line)
            self.color_at(self.y + 5, self.x + 14, width=4, attr='dim')
            self.color_at(self.y + 5, self.x + 45, width=3, attr='dim')
            self.color_at(self.y + 5, self.x + 60, width=3, attr='dim')

            if self.width >= 100:
                for offset, string in (
                    (20, '╴30s├'),
                    (35, '╴60s├'),
                    (66, '╴120s├'),
                    (96, '╴180s├'),
                    (126, '╴240s├'),
                    (156, '╴300s├'),
                ):
                    if offset > remaining_width:
                        break
                    self.addstr(self.y + 5, self.x + self.width - offset, string)
                    self.color_at(
                        self.y + 5,
                        self.x + self.width - offset + 1,
                        width=len(string) - 2,
                        attr='dim',
                    )

        self.color(fg='cyan')
        for y, line in enumerate(host.cpu_percent.history.graph, start=self.y):
            self.addstr(y, self.x + 1, line)

        self.color(fg='magenta')
        for y, line in enumerate(host.virtual_memory.history.graph, start=self.y + 6):  # type: ignore[attr-defined]
            self.addstr(y, self.x + 1, line)

        self.color(fg='blue')
        for y, line in enumerate(host.swap_memory.history.graph, start=self.y + 10):  # type: ignore[attr-defined]
            self.addstr(y, self.x + 1, line)

        if self.width >= 100:
            if self.device_count > 1 and self.parent.selection.is_set():
                device = self.parent.selection.process.device  # type: ignore[union-attr]
                gpu_memory_percent = device.memory_percent.history  # type: ignore[union-attr]
                gpu_utilization = device.gpu_utilization.history  # type: ignore[union-attr]
            else:
                gpu_memory_percent = self.average_gpu_memory_percent
                gpu_utilization = self.average_gpu_utilization

            if self.TERM_256COLOR:
                for i, (y, line) in enumerate(enumerate(gpu_memory_percent.graph, start=self.y)):
                    self.addstr(y, self.x + 79, line, self.get_fg_bg_attr(fg=1.0 - i / 4.0))

                for i, (y, line) in enumerate(enumerate(gpu_utilization.graph, start=self.y + 6)):
                    self.addstr(y, self.x + 79, line, self.get_fg_bg_attr(fg=i / 4.0))
            else:
                self.color(fg=Device.color_of(gpu_memory_percent.last_value, type='memory'))
                for y, line in enumerate(gpu_memory_percent.graph, start=self.y):
                    self.addstr(y, self.x + 79, line)

                self.color(fg=Device.color_of(gpu_utilization.last_value, type='gpu'))
                for y, line in enumerate(gpu_utilization.graph, start=self.y + 6):
                    self.addstr(y, self.x + 79, line)

        self.color_reset()
        self.addstr(self.y, self.x + 1, f' {load_average} ')
        self.addstr(self.y + 1, self.x + 1, f' {host.cpu_percent.history} ')
        self.addstr(
            self.y + 9,
            self.x + 1,
            (
                f' MEM: {bytes2human(self.virtual_memory.used, min_unit=GiB)} '
                f'({host.virtual_memory.history}) '  # type: ignore[attr-defined]
            ),
        )
        self.addstr(
            self.y + 10,
            self.x + 1,
            (
                f' SWP: {bytes2human(self.swap_memory.used, min_unit=GiB)} '
                f'({host.swap_memory.history}) '  # type: ignore[attr-defined]
            ),
        )
        if self.width >= 100:
            self.addstr(self.y, self.x + 79, f' {gpu_memory_percent} ')
            self.addstr(self.y + 10, self.x + 79, f' {gpu_utilization} ')

    def destroy(self) -> None:
        super().destroy()
        self._daemon_running.clear()

    def print_width(self) -> int:
        if self.device_count > 0 and self.width >= 100:
            return self.width
        return 79

    def print(self) -> None:
        self.cpu_percent = host.cpu_percent()
        self.virtual_memory = host.virtual_memory()
        self.swap_memory = host.swap_memory()
        self.load_average = host.load_average()

        load_average = 'Load Average: {} {} {}'.format(
            *(f'{value:5.2f}'[:5] if value < 10000.0 else '9999+' for value in self.load_average),
        )

        width_right = len(load_average) + 4
        width_left = self.width - 2 - width_right
        cpu_bar = '[ {} ]'.format(
            make_bar(
                'CPU',
                self.cpu_percent,
                width_left - 4,
                extra_text=f'  UPTIME: {timedelta2human(host.uptime(), round=True)}',
            ),
        )
        memory_bar = '[ {} ]'.format(
            make_bar(
                'MEM',
                self.virtual_memory.percent,
                width_left - 4,
                extra_text=f'  USED: {bytes2human(self.virtual_memory.used, min_unit=GiB)}',
            ),
        )
        swap_bar = '[ {} ]'.format(make_bar('SWP', self.swap_memory.percent, width_right - 4))

        lines = [
            '{}  {}'.format(
                colored(cpu_bar, color='cyan', attrs=('bold',)),
                colored(f'( {load_average} )', attrs=('bold',)),
            ),
            '{}  {}'.format(
                colored(memory_bar, color='magenta', attrs=('bold',)),
                colored(swap_bar, color='blue', attrs=('bold',)),
            ),
        ]

        lines = '\n'.join(lines)
        if self.no_unicode:
            lines = lines.translate(self.ASCII_TRANSTABLE)

        try:
            print(lines)
        except UnicodeError:
            print(lines.translate(self.ASCII_TRANSTABLE))

    def press(self, key: int) -> bool:
        self.root.keymaps.use_keymap('host')
        return self.root.press(key)
