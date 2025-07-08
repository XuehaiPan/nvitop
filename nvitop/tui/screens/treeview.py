# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import threading
import time
from collections import deque
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING, Any, ClassVar

from nvitop.tui.library import (
    IS_SUPERUSER,
    IS_WSL,
    NA,
    USERNAME,
    Device,
    GpuProcess,
    HostProcess,
    MessageBox,
    MouseEvent,
    Selection,
    Snapshot,
    WideString,
    host,
    ttl_cache,
)
from nvitop.tui.screens.base import BaseSelectableScreen


if TYPE_CHECKING:
    import curses
    from collections.abc import Iterable
    from typing_extensions import Self  # Python 3.11+

    from nvitop.tui.tui import TUI


__all__ = ['TreeViewScreen']


class TreeNode:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        process: GpuProcess | HostProcess,
        children: Iterable[Self] = (),
    ) -> None:
        self.process: GpuProcess | HostProcess = process
        self.parent: TreeNode | None = None
        self.children: list[Self] = []
        self.devices: set[Device] = set()
        self.children_set: set[Self] = set()
        self.is_root: bool = True
        self.is_last: bool = False
        self.prefix: str = ''
        for child in children:
            self.add(child)

    def add(self, child: Self) -> None:
        if child in self.children_set:
            return
        self.children.append(child)
        self.children_set.add(child)
        child.parent = self
        child.is_root = False

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)  # type: ignore[misc]
        except AttributeError:
            return getattr(self.process, name)

    def __eq__(self, other: object) -> bool:
        # pylint: disable-next=protected-access
        return self.process._ident == other.process._ident  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(self.process)

    def as_snapshot(self) -> None:  # pylint: disable=too-many-branches,too-many-statements
        if not isinstance(self.process, Snapshot):
            with self.process.oneshot():
                try:
                    username = self.process.username()
                except host.PsutilError:
                    username = NA
                try:
                    command = self.process.command()
                    if len(command) == 0:
                        command = 'Zombie Process'
                except host.AccessDenied:
                    command = 'No Permissions'
                except host.PsutilError:
                    command = 'No Such Process'

                cpu_percent_string: str
                try:
                    cpu_percent = self.process.cpu_percent()
                except host.PsutilError:
                    cpu_percent = cpu_percent_string = NA
                else:
                    if cpu_percent is NA:
                        cpu_percent_string = NA
                    elif cpu_percent < 1000.0:
                        cpu_percent_string = f'{cpu_percent:.1f}%'
                    elif cpu_percent < 10000:
                        cpu_percent_string = f'{int(cpu_percent)}%'
                    else:
                        cpu_percent_string = '9999+%'

                memory_percent_string: str
                try:
                    memory_percent = self.process.memory_percent()
                except host.PsutilError:
                    memory_percent = memory_percent_string = NA
                else:
                    memory_percent_string = (
                        f'{memory_percent:.1f}%' if memory_percent is not NA else NA
                    )

                try:
                    num_threads = self.process.num_threads()
                except host.PsutilError:
                    num_threads = NA

                try:
                    running_time_human = self.process.running_time_human()
                except host.PsutilError:
                    running_time_human = NA

            self.process = Snapshot(  # type: ignore[assignment]
                real=self.process,
                pid=self.process.pid,
                username=username,
                command=command,
                cpu_percent=cpu_percent,
                cpu_percent_string=cpu_percent_string,
                memory_percent=memory_percent,
                memory_percent_string=memory_percent_string,
                num_threads=num_threads,
                running_time_human=running_time_human,
            )

        if len(self.children) > 0:
            for child in self.children:
                child.as_snapshot()
            self.children.sort(
                key=lambda node: (
                    node._gone,  # pylint: disable=protected-access
                    node.username,
                    node.pid,
                ),
            )
            for child in self.children:
                child.is_last = False
            self.children[-1].is_last = True

    def set_prefix(self, prefix: str = '') -> None:
        if self.is_root:
            self.prefix = ''
        else:
            self.prefix = prefix + ('└─ ' if self.is_last else '├─ ')
            prefix += '   ' if self.is_last else '│  '
        for child in self.children:
            child.set_prefix(prefix)

    @classmethod
    def merge(  # pylint: disable=too-many-branches
        cls,
        leaves: list[Snapshot | GpuProcess | HostProcess],
    ) -> list[Self]:
        nodes: dict[int, Self] = {}
        for process in leaves:
            real_process = process.real if isinstance(process, Snapshot) else process

            try:
                node = nodes[real_process.pid]
            except KeyError:
                node = nodes[real_process.pid] = cls(real_process)
            finally:
                try:
                    node.devices.add(process.device)
                except AttributeError:
                    pass

        queue = deque(nodes.values())
        while len(queue) > 0:
            node = queue.popleft()
            try:
                with node.process.oneshot():
                    parent_process = node.process.parent()
            except host.PsutilError:
                continue
            if parent_process is None:
                continue

            try:
                parent = nodes[parent_process.pid]
            except KeyError:
                parent = nodes[parent_process.pid] = cls(parent_process)
                queue.append(parent)
            else:
                continue
            finally:
                parent.add(node)

        cpid_map = host.reverse_ppid_map()
        for process in leaves:
            if isinstance(process, Snapshot):
                process = process.real

            node = nodes[process.pid]
            for cpid in cpid_map.get(process.pid, []):
                if cpid not in nodes:
                    nodes[cpid] = child = cls(HostProcess(cpid))
                    node.add(child)

        return sorted(filter(lambda node: node.is_root, nodes.values()), key=lambda node: node.pid)

    @staticmethod
    def freeze(roots: list[TreeNode]) -> list[FreezedTreeNode]:
        for root in roots:
            root.as_snapshot()
            root.set_prefix()
        return roots  # type: ignore[return-value]


class FreezedTreeNode(TreeNode):
    process: Snapshot  # type: ignore[assignment]


def flatten_process_trees(roots: list[FreezedTreeNode]) -> list[FreezedTreeNode]:
    flattened = []
    stack = list(reversed(roots))
    while len(stack) > 0:
        top = stack.pop()
        flattened.append(top)
        stack.extend(reversed(top.children))
    return flattened


class TreeViewScreen(BaseSelectableScreen):  # pylint: disable=too-many-instance-attributes
    NAME: ClassVar[str] = 'treeview'
    SNAPSHOT_INTERVAL: ClassVar[float] = 0.5

    def __init__(self, *, win: curses.window, root: TUI) -> None:
        super().__init__(win, root)

        self.selection: Selection = Selection(self)
        self.x_offset: int = 0
        self.y_mouse: int | None = None

        self._snapshot_buffer: list[Snapshot] = []
        self._snapshots: list[Snapshot] = []
        self.snapshot_lock = threading.Lock()
        self._snapshot_daemon = threading.Thread(
            name='treeview-snapshot-daemon',
            target=self._snapshot_target,
            daemon=True,
        )
        self._daemon_running = threading.Event()

        self.x, self.y = root.x, root.y
        self.width, self.height = root.width, root.height
        self.scroll_offset: int = 0

    @property
    def display_height(self) -> int:
        return self.height - 1

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        if self._visible != value:
            self.need_redraw = True
            self._visible = value
        if self.visible:
            self._daemon_running.set()
            try:
                self._snapshot_daemon.start()
            except RuntimeError:
                pass
            self.snapshots = self.take_snapshots()
        else:
            self._daemon_running.clear()
            self.focused = False

    @property
    def snapshots(self) -> list[Snapshot]:
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots: list[Snapshot]) -> None:
        with self.snapshot_lock:
            self.need_redraw = self.need_redraw or len(self._snapshots) > len(snapshots)
            self._snapshots = snapshots

        if self.selection.is_set():
            identity = self.selection.identity
            self.selection.reset()
            for i, process in enumerate(snapshots):
                if process._ident[:2] == identity[:2]:  # pylint: disable=protected-access
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

    @ttl_cache(ttl=2.0)
    def take_snapshots(self) -> list[Snapshot]:
        self.root.main_screen.process_panel.ensure_snapshots()
        snapshots = (
            self.root.main_screen.process_panel._snapshot_buffer  # pylint: disable=protected-access
        )

        roots = TreeNode.merge(snapshots)  # type: ignore[arg-type]
        roots = TreeNode.freeze(roots)
        nodes = flatten_process_trees(roots)

        snapshots = []
        for node in nodes:
            snapshot = node.process
            snapshot.username = WideString(snapshot.username)
            snapshot.prefix = node.prefix
            snapshot.devices = (
                (
                    'GPU '
                    + ','.join(
                        dev.display_index
                        for dev in sorted(node.devices, key=lambda device: device.tuple_index)
                    )
                )
                if node.devices
                else 'Host'
            )
            snapshots.append(snapshot)

        with self.snapshot_lock:
            self._snapshot_buffer = snapshots

        return snapshots

    def _snapshot_target(self) -> None:
        while True:
            self._daemon_running.wait()
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def update_size(self, termsize: tuple[int, int] | None = None) -> tuple[int, int]:
        n_term_lines, n_term_cols = termsize = super().update_size(termsize=termsize)

        self.width = n_term_cols - self.x
        self.height = n_term_lines - self.y

        return termsize

    def poke(self) -> None:
        if self._daemon_running.is_set():
            self.snapshots = self._snapshot_buffer

        self.selection.within_window = False
        if len(self.snapshots) > 0 and self.selection.is_set():
            for i, process in enumerate(self.snapshots):
                y = self.y + 1 - self.scroll_offset + i
                if self.selection.is_same_on_host(process):
                    self.selection.within_window = (
                        1 <= y - self.y < self.height and self.width >= 79
                    )
                    if not self.selection.within_window:
                        if y < self.y + 1:
                            self.scroll_offset -= self.y + 1 - y
                        elif y >= self.y + self.height:
                            self.scroll_offset += y - self.y - self.height + 1
                    self.scroll_offset = max(
                        min(len(self.snapshots) - self.display_height, self.scroll_offset),
                        0,
                    )
                    break
        else:
            self.scroll_offset = 0

        super().poke()

    def draw(self) -> None:  # pylint: disable=too-many-statements,too-many-locals
        self.color_reset()

        pid_width = max(3, max((len(str(process.pid)) for process in self.snapshots), default=3))
        username_width = max(
            4,
            max((len(process.username) for process in self.snapshots), default=4),
        )
        device_width = max(6, max((len(process.devices) for process in self.snapshots), default=6))
        num_threads_width = max(
            4,
            max((len(str(process.num_threads)) for process in self.snapshots), default=4),
        )
        time_width = max(
            4,
            max((len(process.running_time_human) for process in self.snapshots), default=4),
        )

        header = '  '.join(
            [
                'PID'.rjust(pid_width),
                'USER'.ljust(username_width),
                'DEVICE'.rjust(device_width),
                'NLWP'.rjust(num_threads_width),
                '%CPU',
                '%MEM',
                'TIME'.rjust(time_width),
                'COMMAND',
            ],
        )
        command_offset = len(header) - 7
        if self.x_offset < command_offset:
            self.addstr(
                self.y,
                self.x,
                header[self.x_offset : self.x_offset + self.width].ljust(self.width),
            )
        else:
            self.addstr(self.y, self.x, 'COMMAND'.ljust(self.width))
        self.color_at(self.y, self.x, width=self.width, fg='cyan', attr='bold | reverse')

        if len(self.snapshots) == 0:
            self.addstr(
                self.y + 1,
                self.x,
                'No running GPU processes found{}.'.format(' (in WSL)' if IS_WSL else ''),
            )
            return

        hint = True
        if self.y_mouse is not None:
            self.selection.reset()
            hint = False

        self.selection.within_window = False
        processes = islice(
            self.snapshots,
            self.scroll_offset,
            self.scroll_offset + self.display_height,
        )
        for y, process in enumerate(processes, start=self.y + 1):
            prefix_length = len(process.prefix)
            line = '{}  {}  {}  {} {:>5} {:>5}  {}  {}{}'.format(
                str(process.pid).rjust(pid_width),
                process.username.ljust(username_width),
                process.devices.rjust(device_width),
                str(process.num_threads).rjust(num_threads_width),
                process.cpu_percent_string.replace('%', ''),
                process.memory_percent_string.replace('%', ''),
                process.running_time_human.rjust(time_width),
                process.prefix,
                process.command,
            )

            line = str(WideString(line)[self.x_offset :].ljust(self.width)[: self.width])
            self.addstr(y, self.x, line)

            prefix_length -= max(0, self.x_offset - command_offset)
            if prefix_length > 0:
                self.color_at(
                    y,
                    self.x + max(0, command_offset - self.x_offset),
                    width=prefix_length,
                    fg='green',
                    attr='bold',
                )

            if y == self.y_mouse:
                self.selection.process = process
                hint = True

            owned = IS_SUPERUSER or str(process.username) == USERNAME
            if self.selection.is_same_on_host(process):
                self.color_at(
                    y,
                    self.x,
                    width=self.width,
                    fg='yellow' if self.selection.is_tagged(process) else 'green',
                    attr='bold | reverse',
                )
                self.selection.within_window = 1 <= y - self.y < self.height and self.width >= 79
            elif self.selection.is_tagged(process):
                self.color_at(
                    y,
                    self.x,
                    width=self.width,
                    fg='yellow',
                    attr='bold' if owned else 'bold | dim',
                )
            elif not owned:
                self.color_at(y, self.x, width=self.width, attr='dim')

        if not hint:
            self.selection.clear()

        self.color(fg='cyan', attr='bold | reverse')
        text_offset = self.x + self.width - 47
        if len(self.selection.tagged) > 0 or (
            self.selection.owned() and self.selection.within_window
        ):
            self.addstr(self.y, text_offset - 1, ' (Press ^C(INT)/T(TERM)/K(KILL) to send signals)')
            self.color_at(
                self.y,
                text_offset + 7,
                width=2,
                fg='cyan',
                bg='yellow',
                attr='bold | italic | reverse',
            )
            self.color_at(
                self.y,
                text_offset + 10,
                width=3,
                fg='cyan',
                bg='red',
                attr='bold | reverse',
            )
            self.color_at(
                self.y,
                text_offset + 15,
                width=1,
                fg='cyan',
                bg='yellow',
                attr='bold | italic | reverse',
            )
            self.color_at(
                self.y,
                text_offset + 17,
                width=4,
                fg='cyan',
                bg='red',
                attr='bold | reverse',
            )
            self.color_at(
                self.y,
                text_offset + 23,
                width=1,
                fg='cyan',
                bg='yellow',
                attr='bold | italic | reverse',
            )
            self.color_at(
                self.y,
                text_offset + 25,
                width=4,
                fg='cyan',
                bg='red',
                attr='bold | reverse',
            )

    def finalize(self) -> None:
        self.y_mouse = None
        super().finalize()

    def destroy(self) -> None:
        super().destroy()
        self._daemon_running.clear()

    def press(self, key: int) -> bool:
        self.root.keymaps.use_keymap('treeview')
        return self.root.press(key)

    def click(self, event: MouseEvent) -> bool:
        if event.pressed(1) or event.pressed(3) or event.clicked(1) or event.clicked(3):
            self.y_mouse = event.y
            return True

        direction = event.wheel_direction()
        if event.shift():
            self.x_offset = max(0, self.x_offset + 2 * direction)
        else:
            self.selection.move(direction=direction)
        return True

    def init_keybindings(self) -> None:
        def tree_left() -> None:
            self.x_offset = max(0, self.x_offset - 5)

        def tree_right() -> None:
            self.x_offset += 5

        def tree_begin() -> None:
            self.x_offset = 0

        def select_move(direction: int) -> None:
            self.selection.move(direction=direction)

        def select_clear() -> None:
            self.selection.clear()

        def tag() -> None:
            self.selection.tag()
            select_move(direction=+1)

        keymaps = self.root.keymaps

        keymaps.bind('treeview', '<Left>', tree_left)
        keymaps.alias('treeview', '<Left>', '<A-h>')
        keymaps.bind('treeview', '<Right>', tree_right)
        keymaps.alias('treeview', '<Right>', '<A-l>')
        keymaps.bind('treeview', '<C-a>', tree_begin)
        keymaps.alias('treeview', '<C-a>', '^')
        keymaps.bind('treeview', '<Up>', partial(select_move, direction=-1))
        keymaps.alias('treeview', '<Up>', '<S-Tab>')
        keymaps.alias('treeview', '<Up>', '<A-k>')
        keymaps.alias('treeview', '<Up>', '<PageUp>')
        keymaps.alias('treeview', '<Up>', '[')
        keymaps.bind('treeview', '<Down>', partial(select_move, direction=+1))
        keymaps.alias('treeview', '<Down>', '<Tab>')
        keymaps.alias('treeview', '<Down>', '<A-j>')
        keymaps.alias('treeview', '<Down>', '<PageDown>')
        keymaps.alias('treeview', '<Down>', ']')
        keymaps.bind('treeview', '<Home>', partial(select_move, direction=-(1 << 20)))
        keymaps.bind('treeview', '<End>', partial(select_move, direction=+(1 << 20)))
        keymaps.bind('treeview', '<Esc>', select_clear)
        keymaps.bind('treeview', '<Space>', tag)

        keymaps.bind(
            'treeview',
            'T',
            partial(
                MessageBox.confirm_sending_signal_to_processes,
                signal='terminate',
                screen=self,
            ),
        )
        keymaps.bind(
            'treeview',
            'K',
            partial(
                MessageBox.confirm_sending_signal_to_processes,
                signal='kill',
                screen=self,
            ),
        )
        keymaps.alias('treeview', 'K', 'k')
        keymaps.bind(
            'treeview',
            '<C-c>',
            partial(
                MessageBox.confirm_sending_signal_to_processes,
                signal='interrupt',
                screen=self,
            ),
        )
        keymaps.alias('treeview', '<C-c>', 'I')
