# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import itertools
import threading
import time
from operator import attrgetter, xor
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

from nvitop.tui.library import (
    HOSTNAME,
    IS_SUPERUSER,
    IS_WINDOWS,
    IS_WSL,
    LARGE_INTEGER,
    USER_CONTEXT,
    USERNAME,
    Device,
    Displayable,
    GpuProcess,
    MigDevice,
    MouseEvent,
    Selection,
    Snapshot,
    WideString,
    colored,
    cut_string,
    ttl_cache,
    wcslen,
)
from nvitop.tui.screens.main.panels.base import BaseSelectablePanel


if TYPE_CHECKING:
    import curses
    from collections.abc import Callable, Iterable, Mapping

    from nvitop.tui.tui import TUI


__all__ = ['OrderName', 'ProcessPanel']


OrderName = Literal[
    'natural',
    'pid',
    'username',
    'gpu_memory',
    'sm_utilization',
    'gpu_memory_utilization',
    'cpu_percent',
    'memory_percent',
    'time',
]


class Order(NamedTuple):
    key: Callable[[Any], Any]
    reverse: bool
    offset: int
    column: str
    previous: OrderName
    next: OrderName
    bind_key: str


class ProcessPanel(BaseSelectablePanel):  # pylint: disable=too-many-instance-attributes
    NAME: ClassVar[str] = 'process'
    SNAPSHOT_INTERVAL: ClassVar[float] = 0.5

    ORDERS: ClassVar[Mapping[OrderName, Order]] = MappingProxyType(
        {
            'natural': Order(
                key=attrgetter(
                    'device.tuple_index',
                    '_gone',
                    'username',
                    'pid',
                ),
                reverse=False,
                offset=3,
                column='ID',
                previous='time',
                next='pid',
                bind_key='n',
            ),
            'pid': Order(
                key=attrgetter(
                    '_gone',
                    'pid',
                    'device.tuple_index',
                ),
                reverse=False,
                offset=10,
                column='PID',
                previous='natural',
                next='username',
                bind_key='p',
            ),
            'username': Order(
                key=attrgetter(
                    '_gone',
                    'username',
                    'pid',
                    'device.tuple_index',
                ),
                reverse=False,
                offset=19,
                column='USER',
                previous='pid',
                next='gpu_memory',
                bind_key='u',
            ),
            'gpu_memory': Order(
                key=attrgetter(
                    '_gone',
                    'gpu_memory',
                    'gpu_sm_utilization',
                    'cpu_percent',
                    'pid',
                    'device.tuple_index',
                ),
                reverse=True,
                offset=25,
                column='GPU-MEM',
                previous='username',
                next='sm_utilization',
                bind_key='g',
            ),
            'sm_utilization': Order(
                key=attrgetter(
                    '_gone',
                    'gpu_sm_utilization',
                    'gpu_memory',
                    'cpu_percent',
                    'pid',
                    'device.tuple_index',
                ),
                reverse=True,
                offset=34,
                column='SM',
                previous='gpu_memory',
                next='gpu_memory_utilization',
                bind_key='s',
            ),
            'gpu_memory_utilization': Order(
                key=attrgetter(
                    '_gone',
                    'gpu_memory_utilization',
                    'gpu_memory',
                    'cpu_percent',
                    'pid',
                    'device.tuple_index',
                ),
                reverse=True,
                offset=38,
                column='GMBW',
                previous='sm_utilization',
                next='cpu_percent',
                bind_key='b',
            ),
            'cpu_percent': Order(
                key=attrgetter(
                    '_gone',
                    'cpu_percent',
                    'memory_percent',
                    'pid',
                    'device.tuple_index',
                ),
                reverse=True,
                offset=44,
                column='%CPU',
                previous='gpu_memory_utilization',
                next='memory_percent',
                bind_key='c',
            ),
            'memory_percent': Order(
                key=attrgetter(
                    '_gone',
                    'memory_percent',
                    'cpu_percent',
                    'pid',
                    'device.tuple_index',
                ),
                reverse=True,
                offset=50,
                column='%MEM',
                previous='cpu_percent',
                next='time',
                bind_key='m',
            ),
            'time': Order(
                key=attrgetter(
                    '_gone',
                    'running_time',
                    'pid',
                    'device.tuple_index',
                ),
                reverse=True,
                offset=56,
                column='TIME',
                previous='memory_percent',
                next='natural',
                bind_key='t',
            ),
        },
    )

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        devices: list[Device | MigDevice],
        compact: bool,
        filters: Iterable[Callable[[Snapshot], bool]],
        *,
        win: curses.window | None,
        root: TUI,
    ) -> None:
        super().__init__(win, root)

        self.devices: list[Device | MigDevice] = devices

        self._compact: bool = compact
        self.width: int = max(79, root.width)
        self.height = self._full_height = self.compact_height = 7

        self.filters: list[Callable[[Snapshot], bool] | None] = [None, *filters]

        self.host_headers: list[str] = ['%CPU', '%MEM', 'TIME', 'COMMAND']

        self.selection: Selection = Selection(self)
        self.host_offset: int = -1
        self.y_mouse: int | None = None

        self._order: OrderName = 'natural'
        self.reverse: bool = False

        self.has_snapshots: int = False
        self._snapshot_buffer: list[Snapshot] | None = None
        self._snapshots: list[Snapshot] = []
        self.snapshot_lock = threading.Lock()
        self._snapshot_daemon = threading.Thread(
            name='process-snapshot-daemon',
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
        if self._width != width and self.visible:
            self.need_redraw = True
        self._width = width

    @property
    def compact(self) -> bool:
        return self._compact or self.order != 'natural'

    @compact.setter
    def compact(self, value: bool) -> None:
        if self._compact != value:
            self.need_redraw = True
            self._compact = value
            processes = self.snapshots
            n_processes, n_devices = (
                len(processes),
                len({p.device.physical_index for p in processes}),
            )
            self.full_height = 1 + max(6, 5 + n_processes + n_devices - 1)
            self.compact_height = 1 + max(6, 5 + n_processes)
            self.height = self.compact_height if self.compact else self.full_height

    @property
    def full_height(self) -> int:
        return self._full_height if self.order == 'natural' else self.compact_height

    @full_height.setter
    def full_height(self, value: int) -> None:
        self._full_height = value

    @property
    def order(self) -> OrderName:
        return self._order

    @order.setter
    def order(self, value: OrderName) -> None:
        if self._order != value:
            self._order = value
            self.height = self.compact_height if self.compact else self.full_height

    @property
    def snapshots(self) -> list[Snapshot]:
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots: list[Snapshot] | None) -> None:
        if snapshots is None:
            return
        self.has_snapshots = True

        time_length = max(4, max((len(p.running_time_human) for p in snapshots), default=4))
        time_header = ' ' * (time_length - 4) + 'TIME'
        info_length = max((len(p.host_info) for p in snapshots), default=0)
        n_processes, n_devices = len(snapshots), len({p.device.physical_index for p in snapshots})
        self.full_height = 1 + max(6, 5 + n_processes + n_devices - 1)
        self.compact_height = 1 + max(6, 5 + n_processes)
        height = self.compact_height if self.compact else self.full_height
        key, reverse, *_ = self.ORDERS[self.order]
        snapshots.sort(key=key, reverse=xor(reverse, self.reverse))

        old_host_offset = self.host_offset
        with self.snapshot_lock:
            self.need_redraw = (
                self.need_redraw or self.height > height or self.host_headers[-2] != time_header
            )
            self._snapshots = snapshots

            self.host_headers[-2] = time_header
            self.height = height
            self.host_offset = max(-1, min(self.host_offset, info_length - self.width + 45))

        if old_host_offset not in {self.host_offset, LARGE_INTEGER}:
            self.beep()

        if self.selection.is_set():
            identity = self.selection.identity
            self.selection.reset()
            for i, process in enumerate(snapshots):
                if process._ident == identity:  # pylint: disable=protected-access
                    self.selection.index = i
                    self.selection.process = process
                    break

    @classmethod
    def set_snapshot_interval(cls, interval: float) -> None:
        assert interval > 0.0
        interval = float(interval)

        cls.SNAPSHOT_INTERVAL = min(interval / 3.0, 1.0)
        cls.take_snapshots = ttl_cache(ttl=interval)(  # type: ignore[method-assign]
            cls.take_snapshots.__wrapped__,  # type: ignore[attr-defined] # pylint: disable=no-member
        )

    def ensure_snapshots(self) -> None:
        if not self.has_snapshots:
            self.snapshots = self.take_snapshots()

    @ttl_cache(ttl=2.0)
    def take_snapshots(self) -> list[Snapshot]:
        snapshots = GpuProcess.take_snapshots(self.processes, failsafe=True)
        for condition in self.filters:
            snapshots = filter(condition, snapshots)  # type: ignore[assignment]
        snapshots = list(snapshots)

        time_length = max(4, max((len(p.running_time_human) for p in snapshots), default=4))
        for snapshot in snapshots:
            snapshot.host_info = WideString(
                '{:>5} {:>5}  {}  {}'.format(
                    snapshot.cpu_percent_string.replace('%', ''),
                    snapshot.memory_percent_string.replace('%', ''),
                    ' ' * (time_length - len(snapshot.running_time_human))
                    + snapshot.running_time_human,
                    snapshot.command,
                ),
            )

        with self.snapshot_lock:
            self._snapshot_buffer = snapshots

        return snapshots

    def _snapshot_target(self) -> None:
        self._daemon_running.wait()
        while self._daemon_running.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def header_lines(self) -> list[str]:
        header = [
            '╒' + '═' * (self.width - 2) + '╕',
            '│ {} │'.format('Processes:'.ljust(self.width - 4)),
            r'│ GPU     PID      USER  GPU-MEM %SM %GMBW  {} │'.format(
                '  '.join(self.host_headers).ljust(self.width - 46),
            ),
            '╞' + '═' * (self.width - 2) + '╡',
        ]
        if len(self.snapshots) == 0:
            if self.has_snapshots:
                message = ' No running processes found{} '.format(' (in WSL)' if IS_WSL else '')
            else:
                message = ' Gathering process status...'
            header.extend(
                [f'│ {message.ljust(self.width - 4)} │', '╘' + '═' * (self.width - 2) + '╛'],
            )
        return header

    @property
    def processes(self) -> list[GpuProcess]:
        return list(
            itertools.chain.from_iterable(device.processes().values() for device in self.devices),  # type: ignore[misc]
        )

    def poke(self) -> None:
        if not self._daemon_running.is_set():
            self._daemon_running.set()
            self._snapshot_daemon.start()

        self.snapshots = self._snapshot_buffer

        self.selection.within_window = False
        if len(self.snapshots) > 0 and self.selection.is_set():
            y = self.y + 5
            prev_device_index: int | None = None
            for process in self.snapshots:
                device_index = process.device.physical_index
                if prev_device_index != device_index:
                    if not self.compact and prev_device_index is not None:
                        y += 1
                    prev_device_index = device_index

                if self.selection.is_same(process):
                    self.selection.within_window = (
                        self.root.y <= y < self.root.termsize[0]  # type: ignore[index]
                        and self.width >= 79
                    )
                    if not self.selection.within_window:
                        if y < self.root.y:
                            self.parent.y += self.root.y - y
                        elif y >= self.root.termsize[0]:  # type: ignore[index]
                            self.parent.y -= y - self.root.termsize[0] + 1  # type: ignore[index]
                        self.parent.update_size(self.root.termsize)
                        self.need_redraw = True
                    break
                y += 1

        super().poke()

    def draw(self) -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self.color_reset()

        if self.need_redraw:
            if IS_SUPERUSER:
                self.addstr(self.y, self.x + 1, '!CAUTION: SUPERUSER LOGGED-IN.')
                self.color_at(self.y, self.x + 1, width=1, fg='red', attr='blink')
                self.color_at(self.y, self.x + 2, width=29, fg='yellow', attr='italic')

            for y, line in enumerate(self.header_lines(), start=self.y + 1):
                self.addstr(y, self.x, line)

            context_width = wcslen(USER_CONTEXT)
            if not IS_WINDOWS or len(USER_CONTEXT) == context_width:
                # Do not support windows-curses with wide characters
                username_width = wcslen(USERNAME)
                hostname_width = wcslen(HOSTNAME)
                offset = self.x + self.width - context_width - 2
                self.addstr(self.y + 2, self.x + offset, USER_CONTEXT)
                self.color_at(self.y + 2, self.x + offset, width=context_width, attr='bold')
                self.color_at(
                    self.y + 2,
                    self.x + offset,
                    width=username_width,
                    fg=('yellow' if IS_SUPERUSER else 'magenta'),
                    attr='bold',
                )
                self.color_at(
                    self.y + 2,
                    self.x + offset + username_width + 1,
                    width=hostname_width,
                    fg='green',
                    attr='bold',
                )

        self.addstr(self.y + 3, self.x + 1, r' GPU     PID      USER  GPU-MEM %SM %GMBW  ')
        host_offset = max(self.host_offset, 0)
        command_offset = max(14 + len(self.host_headers[-2]) - host_offset, 0)
        if command_offset > 0:
            host_headers = '  '.join(self.host_headers)
            self.addstr(
                self.y + 3,
                self.x + 44,
                f'{host_headers[host_offset:].ljust(self.width - 46)}',
            )
        else:
            self.addstr(self.y + 3, self.x + 44, '{}'.format('COMMAND'.ljust(self.width - 46)))

        _, reverse, offset, column, *_ = self.ORDERS[self.order]
        column_width = len(column)
        reverse = xor(reverse, self.reverse)
        indicator = '▼' if reverse else '▲'
        if self.order in {'cpu_percent', 'memory_percent', 'time'}:
            offset -= host_offset
            if self.order == 'time':
                offset += len(self.host_headers[-2]) - 4
            if offset > 44 or host_offset == 0:
                self.addstr(self.y + 3, self.x + offset - 1, column + indicator)
                self.color_at(
                    self.y + 3,
                    self.x + offset - 1,
                    width=column_width,
                    attr='bold | underline',
                )
            elif offset <= 44 < offset + column_width:
                self.addstr(self.y + 3, self.x + 44, (column + indicator)[45 - offset :])
                if offset + column_width >= 46:
                    self.color_at(
                        self.y + 3,
                        self.x + 44,
                        width=offset + column_width - 45,
                        attr='bold | underline',
                    )
            if offset + column_width >= 45:
                self.color_at(self.y + 3, self.x + offset + column_width - 1, width=1, attr='bold')
        elif self.order == 'natural' and not reverse:
            self.color_at(self.y + 3, self.x + 2, width=3, attr='bold')
        else:
            self.addstr(self.y + 3, self.x + offset - 1, column + indicator)
            self.color_at(
                self.y + 3,
                self.x + offset - 1,
                width=column_width,
                attr='bold | underline',
            )
            self.color_at(self.y + 3, self.x + offset + column_width - 1, width=1, attr='bold')

        hint = True
        if self.y_mouse is not None:
            self.selection.reset()
            hint = False

        self.selection.within_window = False
        if len(self.snapshots) > 0:
            y = self.y + 5
            prev_device_index: int | None = None
            prev_device_display_index = None
            color = -1
            for process in self.snapshots:
                device_index = process.device.physical_index
                device_display_index = process.device.display_index
                if prev_device_index != device_index:
                    if not self.compact and prev_device_index is not None:
                        self.addstr(y, self.x, '├' + '─' * (self.width - 2) + '┤')
                        if y == self.y_mouse:
                            self.y_mouse += 1
                        y += 1
                    prev_device_index = device_index
                if prev_device_display_index != device_display_index:
                    color = process.device.snapshot.display_color
                    prev_device_display_index = device_display_index

                host_info = process.host_info
                if self.host_offset < 0:
                    host_info = cut_string(host_info, padstr='..', maxlen=self.width - 45)
                else:
                    host_info = WideString(host_info)[self.host_offset :]
                self.addstr(
                    y,
                    self.x,
                    '│{:>4} {:>7} {} {} {:>8} {:>3} {:>5} {} │'.format(
                        device_display_index,
                        cut_string(process.pid, maxlen=7, padstr='.'),
                        process.type,
                        str(
                            WideString(cut_string(process.username, maxlen=7, padstr='+')).rjust(7),
                        ),
                        process.gpu_memory_human,
                        process.gpu_sm_utilization_string.replace('%', ''),
                        process.gpu_memory_utilization_string.replace('%', ''),
                        WideString(host_info).ljust(self.width - 45)[: self.width - 45],
                    ),
                )
                if self.host_offset > 0:
                    self.addstr(y, self.x + 43, ' ')

                is_zombie = process.is_zombie
                no_permissions = process.no_permissions
                is_gone = process.is_gone
                if (is_zombie or no_permissions or is_gone) and command_offset == 0:
                    self.addstr(y, self.x + 44, process.command)

                if y == self.y_mouse:
                    self.selection.process = process
                    hint = True

                if self.selection.is_same(process):
                    self.color_at(
                        y,
                        self.x + 1,
                        width=self.width - 2,
                        fg='yellow' if self.selection.is_tagged(process) else 'cyan',
                        attr='bold | reverse',
                    )
                    self.selection.within_window = (
                        self.root.y <= y < self.root.termsize[0]  # type: ignore[index]
                        and self.width >= 79
                    )
                else:
                    owned = IS_SUPERUSER or str(process.username) == USERNAME
                    if self.selection.is_same_on_host(process):
                        self.addstr(y, self.x + 1, '=', self.get_fg_bg_attr(attr='bold | blink'))
                    self.color_at(y, self.x + 2, width=3, fg=color)
                    if self.selection.is_tagged(process):
                        self.color_at(
                            y,
                            self.x + 5,
                            width=self.width - 6,
                            fg='yellow',
                            attr='bold' if owned else 'bold | dim',
                        )
                    elif not owned:
                        self.color_at(y, self.x + 5, width=self.width - 6, attr='dim')
                    if is_zombie or no_permissions:
                        self.color_at(y, self.x + 44 + command_offset, width=14, fg='yellow')
                    elif is_gone:
                        self.color_at(y, self.x + 44 + command_offset, width=15, fg='red')
                y += 1

            self.addstr(y, self.x, '╘' + '═' * (self.width - 2) + '╛')
            if not hint:
                self.selection.clear()

        elif self.has_snapshots:
            message = ' No running processes found{} '.format(' (in WSL)' if IS_WSL else '')
            self.addstr(self.y + 5, self.x, f'│ {message.ljust(self.width - 4)} │')

        text_offset = self.x + self.width - 47
        if len(self.selection.tagged) > 0 or (
            self.selection.owned() and self.selection.within_window
        ):
            self.addstr(self.y, text_offset, '(Press ^C(INT)/T(TERM)/K(KILL) to send signals)')
            self.color_at(self.y, text_offset + 7, width=2, fg='magenta', attr='bold | italic')
            self.color_at(self.y, text_offset + 10, width=3, fg='red', attr='bold')
            self.color_at(self.y, text_offset + 15, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y, text_offset + 17, width=4, fg='red', attr='bold')
            self.color_at(self.y, text_offset + 23, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y, text_offset + 25, width=4, fg='red', attr='bold')
        else:
            self.addstr(self.y, text_offset, ' ' * 47)

    def finalize(self) -> None:
        self.y_mouse = None
        super().finalize()

    def destroy(self) -> None:
        super().destroy()
        self._daemon_running.clear()

    def print_width(self) -> int:
        self.ensure_snapshots()
        return min(
            self.width,
            max((45 + len(process.host_info) for process in self.snapshots), default=79),
        )

    def print(self) -> None:
        self.ensure_snapshots()

        lines = ['', *self.header_lines()]
        lines[2] = ''.join(
            (
                lines[2][: -2 - wcslen(USER_CONTEXT)],
                colored(USERNAME, color=('yellow' if IS_SUPERUSER else 'magenta'), attrs=('bold',)),
                colored('@', attrs=('bold',)),
                colored(HOSTNAME, color='green', attrs=('bold',)),
                lines[2][-2:],
            ),
        )

        if len(self.snapshots) > 0:
            key, reverse, *_ = self.ORDERS['natural']
            self.snapshots.sort(key=key, reverse=reverse)
            prev_device_index: int | None = None
            prev_device_display_index = None
            color = None
            for process in self.snapshots:
                device_index = process.device.physical_index
                device_display_index = process.device.display_index
                if prev_device_index != device_index:
                    if prev_device_index is not None:
                        lines.append('├' + '─' * (self.width - 2) + '┤')
                    prev_device_index = device_index
                if prev_device_display_index != device_display_index:
                    color = process.device.snapshot.display_color
                    prev_device_display_index = device_display_index

                host_info = cut_string(process.host_info, padstr='..', maxlen=self.width - 45)

                info = '{:>7} {} {} {:>8} {:>3} {:>5} {}'.format(
                    cut_string(process.pid, maxlen=7, padstr='.'),
                    process.type,
                    str(WideString(cut_string(process.username, maxlen=7, padstr='+')).rjust(7)),
                    process.gpu_memory_human,
                    process.gpu_sm_utilization_string.replace('%', ''),
                    process.gpu_memory_utilization_string.replace('%', ''),
                    WideString(host_info).ljust(self.width - 45)[: self.width - 45],
                )
                if process.is_zombie or process.no_permissions or process.is_gone:
                    info_segments = info.split(process.command)
                    if not IS_SUPERUSER and process.username != USERNAME:
                        info_segments = [colored(item, attrs=('dark',)) for item in info_segments]
                    info = colored(
                        process.command,
                        color=('red' if process.is_gone else 'yellow'),
                    ).join(info_segments)
                elif not IS_SUPERUSER and process.username != USERNAME:
                    info = colored(info, attrs=('dark',))
                lines.append('│{} {} │'.format(colored(f'{device_display_index:>4}', color), info))

            lines.append('╘' + '═' * (self.width - 2) + '╛')

        lines = '\n'.join(lines)
        if self.no_unicode:
            lines = lines.translate(self.ASCII_TRANSTABLE)

        try:
            print(lines)
        except UnicodeError:
            print(lines.translate(self.ASCII_TRANSTABLE))

    def press(self, key: int) -> bool:
        self.root.keymaps.use_keymap('process')
        return self.root.press(key)

    def click(self, event: MouseEvent) -> bool:
        if event.pressed(1) or event.pressed(3) or event.clicked(1) or event.clicked(3):
            self.y_mouse = event.y
            return True

        direction = event.wheel_direction()
        if event.shift():
            self.host_offset += 2 * direction
        else:
            self.selection.move(direction=direction)
        return True

    def __contains__(self, item: Displayable | MouseEvent | tuple[int, int]) -> bool:
        if self.parent.visible and isinstance(item, MouseEvent):
            return True
        return super().__contains__(item)
