# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import shutil
import threading
import time
from collections import deque

from cachetools.func import ttl_cache

from ...core import host, Snapshot
from ..library import Displayable
from .process import CURRENT_USER, IS_SUPERUSER, Selected


class TreeNode(object):
    def __init__(self, process, children=()):
        self.process = process
        self.parent = None
        self.children = []
        self.devices = set()
        self.children_set = set()
        self.is_root = True
        self.is_last = False
        self.prefix = ''
        for child in children:
            self.add(child)

    def add(self, child):
        if child in self.children_set:
            return
        self.children.append(child)
        self.children_set.add(child)
        child.parent = self
        child.is_root = False

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.process, name)

    def __hash__(self):
        return hash(self.process)

    def freeze(self):
        if not isinstance(self.process, Snapshot):
            try:
                username = self.process.username()
            except host.PsutilError:
                username = 'N/A'
            try:
                command = self.process.command()
                if len(command) == 0:
                    command = 'Zombie Process'
            except host.PsutilError:
                command = 'No Such Process'

            self.process = Snapshot(
                real=self.process,
                pid=self.process.pid,
                username=username,
                command=command
            )

        if len(self.children) > 0:
            self.children.sort(key=lambda node: node.pid)
            for child in self.children:
                child.is_last = False
            self.children[-1].is_last = True
            for child in self.children:
                child.freeze()

    def set_prefix(self, prefix=''):
        if self.is_root:
            self.prefix = ''
        else:
            self.prefix = prefix + ('└─ ' if self.is_last else '├─ ')
            prefix += ('   ' if self.is_last else '│  ')
        for child in self.children:
            child.set_prefix(prefix)

    @classmethod
    def merge(cls, leaves):
        nodes = {}
        for process in leaves:
            try:
                node = nodes[process.pid]
            except KeyError:
                node = nodes[process.pid] = cls(process.real)
            finally:
                node.devices.add(process.device)

        queue = deque(nodes.values())
        while len(queue) > 0:
            node = queue.popleft()
            while True:
                try:
                    parent_process = node.process.parent()
                except host.PsutilError:
                    break
                if parent_process is None:
                    break

                try:
                    parent = nodes[parent_process.pid]
                except KeyError:
                    parent = nodes[parent_process.pid] = cls(parent_process)
                    queue.append(parent)
                else:
                    break
                finally:
                    parent.add(node)

        roots = sorted(filter(lambda node: node.is_root, nodes.values()), key=lambda node: node.pid)
        for root in roots:
            root.freeze()
            root.set_prefix()

        return roots

    @staticmethod
    def flatten(roots):
        flattened = []
        stack = list(reversed(roots))
        while len(stack) > 0:
            top = stack.pop()
            flattened.append(top)
            stack.extend(reversed(top.children))
        return flattened


class TreeViewPanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7

    def __init__(self, win, root):
        super().__init__(win, root)

        self.selected = Selected(panel=self)
        self.x_offset = 0

        self._snapshot_buffer = []
        self._snapshots = []
        self.snapshot_lock = root.lock
        self._snapshot_daemon = threading.Thread(name='treeview-snapshot-daemon',
                                                 target=self._snapshot_target, daemon=True)
        self._daemon_running = threading.Event()

        self.width, self.height = shutil.get_terminal_size(fallback=(79, 24))

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
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
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots):
        n_processes = len(snapshots)
        height = 1 + n_processes

        with self.snapshot_lock:
            self._snapshots = snapshots
            self.need_redraw = (self.need_redraw or self.height > height)
            self.height = height

            if self.selected.is_set():
                identity = self.selected.identity
                self.selected.clear()
                for i, process in enumerate(snapshots):
                    if process._ident[:2] == identity[:2]:  # pylint: disable=protected-access
                        self.selected.index = i
                        self.selected.process = process
                        break

    @ttl_cache(ttl=2.0)
    def take_snapshots(self):
        snapshots = self.root.process_panel._snapshot_buffer  # pylint: disable=protected-access
        nodes = TreeNode.flatten(TreeNode.merge(snapshots))
        snapshots = []
        for node in nodes:
            snapshot = node.process
            snapshot.prefix = node.prefix
            if len(node.devices) > 0:
                snapshot.devices = 'GPU ' + ','.join(map(lambda device: str(device.index),
                                                         sorted(node.devices, key=lambda device: device.index)))
            else:
                snapshot.devices = 'Host'
            snapshots.append(snapshot)

        with self.snapshot_lock:
            self._snapshot_buffer = snapshots

        return snapshots

    def _snapshot_target(self):
        while True:
            self._daemon_running.wait()
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def poke(self):
        if self._daemon_running.is_set():
            with self.snapshot_lock:
                self.snapshots = self._snapshot_buffer

        super().poke()

    def draw(self):
        self.color_reset()

        pid_width = max(3, max([len(str(process.pid)) for process in self.snapshots], default=3))
        username_width = max(4, max([len(str(process.username)) for process in self.snapshots], default=4))
        device_width = max(6, max([len(str(process.devices)) for process in self.snapshots], default=6))
        command_offset = pid_width + username_width + device_width + 6

        header = '  '.join(['PID'.rjust(pid_width), 'USER'.ljust(username_width),
                            'DEVICE'.rjust(device_width), 'COMMAND'])
        if self.x_offset < command_offset:
            self.addstr(self.y, self.x, header[self.x_offset:self.x_offset + self.width].ljust(self.width))
        else:
            self.addstr(self.y, self.x, 'COMMAND'.ljust(self.width))
        self.color_at(self.y, self.x, width=self.width, fg='cyan', attr='bold | reverse')

        self.selected.within_window = False
        for y, process in enumerate(self.snapshots, start=self.y + 1):
            prefix_length = len(process.prefix)
            line = '{}  {}  {}  {}{}'.format(str(process.pid).rjust(pid_width),
                                             process.username.ljust(username_width),
                                             process.devices.rjust(device_width),
                                             process.prefix, process.command)
            self.addstr(y, self.x, line[self.x_offset:self.x_offset + self.width].ljust(self.width))

            prefix_length -= max(0, self.x_offset - command_offset)
            if prefix_length > 0:
                self.color_at(y, self.x + max(0, command_offset - self.x_offset),
                              width=prefix_length, fg='green', attr='bold')

            if self.selected.is_same_on_host(process):
                self.color_at(y, self.x, width=self.width, fg='green', attr='bold | reverse')
                self.selected.within_window = (0 <= y < self.root.termsize[0] and self.width >= 79)
            else:
                if process.username != CURRENT_USER and not IS_SUPERUSER:
                    self.color_at(y, self.x, width=self.width, attr='dim')

        self.color(fg='cyan', attr='bold | reverse')
        text_offset = self.x + self.width - 47
        if self.selected.owned() and self.selected.within_window:
            self.addstr(self.y, text_offset, '(Press ^C(INT)/T(TERM)/K(KILL) to send signals)')
            self.color_at(self.y, text_offset + 7, width=2, fg='cyan', bg='yellow', attr='bold | italic | reverse')
            self.color_at(self.y, text_offset + 10, width=3, fg='cyan', bg='red', attr='bold | reverse')
            self.color_at(self.y, text_offset + 15, width=1, fg='cyan', bg='yellow', attr='bold | italic | reverse')
            self.color_at(self.y, text_offset + 17, width=4, fg='cyan', bg='red', attr='bold | reverse')
            self.color_at(self.y, text_offset + 23, width=1, fg='cyan', bg='yellow', attr='bold | italic | reverse')
            self.color_at(self.y, text_offset + 25, width=4, fg='cyan', bg='red', attr='bold | reverse')
        else:
            self.addstr(self.y, text_offset, ' ' * 47)

    def finalize(self):
        self.need_redraw = False

    def press(self, key):
        self.root.keymaps.use_keymap('treeview')
        self.root.press(key)

    def click(self, event):
        direction = event.wheel_direction()
        if event.shift():
            self.x_offset = max(0, self.x_offset + direction)
        else:
            self.selected.move(direction=direction)
        return True
