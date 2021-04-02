# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import getpass
import signal
import threading
import time
from collections import OrderedDict

import psutil
from cachetools.func import ttl_cache

from ..displayable import Displayable
from ...utils import colored, cut_string, Snapshot


CURRENT_USER = getpass.getuser()
if psutil.WINDOWS:
    import ctypes
    IS_SUPERUSER = bool(ctypes.windll.shell32.IsUserAnAdmin())
else:
    import os
    IS_SUPERUSER = ((os.geteuid() == 0) if hasattr(os, 'geteuid') else False)


class Selected(object):
    def __init__(self, panel):
        self.panel = panel
        self.index = None
        self._proc = None
        self._ident = None

    @property
    def identity(self):
        if self._ident is None:
            self._ident = self.process._ident  # pylint: disable=protected-access
        return self._ident

    @property
    def process(self):
        return self._proc

    @process.setter
    def process(self, process):
        if isinstance(process, Snapshot):
            process = process.real
        self._proc = process
        self._ident = None

    @property
    def pid(self):
        try:
            return self.identity[0]
        except TypeError:
            return None

    def move(self, direction=0):
        if direction == 0:
            return

        processes = self.panel.snapshots
        if len(processes) > 0:
            if not self.is_set():
                if direction > 0:
                    self.index = 0
                else:
                    self.index = len(processes) - 1
            else:
                self.index = min(max(0, self.index + direction), len(processes) - 1)
            self.process = processes[self.index]
        else:
            self.clear()

    def owned(self):
        return self.is_set() and (IS_SUPERUSER or self.process.username() == CURRENT_USER)

    def send_signal(self, sig):
        if self.owned():
            try:
                self.process.send_signal(sig)
            except psutil.Error:
                pass
            else:
                time.sleep(0.5)
                if not self.process.is_running():
                    self.clear()

    def terminate(self):
        if self.owned():
            try:
                self.process.terminate()
            except psutil.Error:
                pass
            else:
                time.sleep(0.5)
                self.clear()

    def kill(self):
        if self.owned():
            try:
                self.process.kill()
            except psutil.Error:
                pass
            else:
                time.sleep(0.5)
                self.clear()

    def interrupt(self):
        return self.send_signal(signal.SIGINT)

    def clear(self):
        self.index = None
        self._proc = None
        self._ident = None

    reset = clear

    def is_set(self):
        if self.index is not None and self.process is not None:
            return True
        self.clear()
        return False

    __bool__ = is_set

    def is_same(self, process):
        try:
            return self.identity == process.identity
        except AttributeError:
            try:
                return self.identity == process._ident  # pylint: disable=protected-access
            except AttributeError:
                pass
        except TypeError:
            pass

        return False

    __eq__ = is_same

    def is_same_on_host(self, process):
        try:
            return self.identity[:2] == process.identity[:2]
        except AttributeError:
            try:
                return self.identity[:2] == process._ident[:2]  # pylint: disable=protected-access
            except AttributeError:
                pass
        except TypeError:
            pass

        return False


class ProcessPanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7

    def __init__(self, devices, win=None, root=None):
        super().__init__(win, root)
        self.width = 79
        self.height = 6

        self.devices = devices

        self.host_headers = ['%CPU', '%MEM', 'TIME', 'COMMAND']

        self.selected = Selected(panel=self)
        self.host_offset = -1

        self._snapshot_buffer = []
        self._snapshots = []
        self.snapshot_lock = threading.RLock()
        self.snapshots = self.take_snapshots()
        self._snapshot_daemon = threading.Thread(name='process-snapshot-daemon',
                                                 target=self._snapshot_target, daemon=True)
        self._daemon_started = threading.Event()

    @property
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots):
        time_length = max(4, max([len(p.running_time_human) for p in snapshots], default=4))
        time_header = ' ' * (time_length - 4) + 'TIME'
        info_length = max([len(p.host_info) for p in snapshots], default=0)
        height = max(6, 5 + len(snapshots) + (len(set([p.device.index for p in snapshots])) - 1))

        with self.snapshot_lock:
            self._snapshots = snapshots
            self.need_redraw = (self.need_redraw or self.height > height or self.host_headers[-2] != time_header)
            self.host_headers[-2] = time_header
            self.height = height
            self.host_offset = max(-1, min(self.host_offset, info_length - 45))

            if self.selected.is_set():
                identity = self.selected.identity
                self.selected.clear()
                for i, process in enumerate(snapshots):
                    if process.identity == identity:
                        self.selected.index = i
                        self.selected.process = process
                        break

    def take_snapshots(self):
        snapshots = list(filter(None, map(lambda process: process.snapshot(), self.processes.values())))

        time_length = max(4, max([len(p.running_time_human) for p in snapshots], default=4))
        for snapshot in snapshots:
            snapshot.host_info = '{:>5} {:>5}  {}  {}'.format(
                snapshot.cpu_percent_string,
                snapshot.memory_percent_string,
                ' ' * (time_length - len(snapshot.running_time_human)) + snapshot.running_time_human,
                snapshot.command
            )

        with self.snapshot_lock:
            self._snapshot_buffer = snapshots

        return snapshots

    def _snapshot_target(self):
        self._daemon_started.wait()
        while self._daemon_started.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def header_lines(self):
        header = [
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ Processes:                                                                  │',
            '│ GPU    PID      USER  GPU MEM  {:<44} │'.format('  '.join(self.host_headers)),
            '╞═════════════════════════════════════════════════════════════════════════════╡',
        ]
        if len(self.snapshots) == 0:
            header.extend([
                '│  No running compute processes found                                         │',
                '╘═════════════════════════════════════════════════════════════════════════════╛',
            ])
        return header

    @property
    @ttl_cache(ttl=1.0)
    def processes(self):
        processes = {}
        for device in self.devices:
            for p in device.processes.values():
                try:
                    username = p.username()
                    processes[(p.device.index, username != 'N/A', username, p.pid)] = p
                except psutil.Error:
                    pass
        return OrderedDict([((key[-1], key[0]), processes[key]) for key in sorted(processes.keys())])

    def poke(self):
        if not self._daemon_started.is_set():
            self._daemon_started.set()
            self._snapshot_daemon.start()

        with self.snapshot_lock:
            self.snapshots = self._snapshot_buffer

        super().poke()

    def draw(self):
        self.color_reset()

        if self.need_redraw:
            for y, line in enumerate(self.header_lines(), start=self.y):
                self.addstr(y, self.x, line)
        host_offset = max(self.host_offset, 0)
        command_offset = max(14 + len(self.host_headers[-2]) - host_offset, 0)
        if command_offset > 0:
            host_headers = '  '.join(self.host_headers)
            self.addstr(self.y + 2, self.x + 33, '{:<44}'.format(host_headers[host_offset:]))
        else:
            self.addstr(self.y + 2, self.x + 33, '{:<44}'.format('COMMAND'))

        if len(self.snapshots) > 0:
            y = self.y + 4
            prev_device_index = None
            color = -1
            for process in self.snapshots:
                device_index = process.device.index
                if prev_device_index != device_index:
                    color = process.device.display_color
                    if prev_device_index is not None:
                        self.addstr(y, self.x,
                                    '├─────────────────────────────────────────────────────────────────────────────┤')
                        y += 1
                    prev_device_index = device_index

                host_info = process.host_info
                if self.host_offset < 0:
                    host_info = cut_string(host_info, padstr='..', maxlen=45)
                else:
                    host_info = host_info[self.host_offset:self.host_offset + 45]
                self.addstr(y, self.x,
                            '│ {:>3} {:>6} {} {:>7} {:>8} {:<45} │'.format(
                                device_index, cut_string(process.pid, maxlen=6, padstr='.'),
                                process.type, cut_string(process.username, maxlen=7, padstr='+'),
                                process.gpu_memory_human, host_info
                            ))
                if not process.is_running and process.command == 'No Such Process':
                    if command_offset == 0:
                        self.addstr(y, self.x + 33 + command_offset, process.command)
                    self.color_at(y, self.x + 33 + command_offset, width=15, fg='red')
                if self.host_offset > 0:
                    self.addstr(y, self.x + 32, ' ')

                if self.selected.is_same(process):
                    self.color_at(y, self.x + 1, width=77, fg='cyan', attr='bold | reverse')
                else:
                    if self.selected.is_same_on_host(process):
                        self.addstr(y, self.x + 1, '=')
                        self.color_at(y, self.x + 1, width=1, attr='bold | blink')
                    self.color_at(y, self.x + 2, width=3, fg=color)
                y += 1
            self.addstr(y, self.x, '╘═════════════════════════════════════════════════════════════════════════════╛')
        else:
            self.addstr(self.y + 4, self.x,
                        '│  No running compute processes found                                         │')

        if self.selected.owned():
            if IS_SUPERUSER:
                self.addstr(self.y - 1, self.x + 1, '!CAUTION: SUPERUSER LOGGED-IN.')
                self.color_at(self.y - 1, self.x + 1, width=1, fg='red', attr='blink')
                self.color_at(self.y - 1, self.x + 2, width=29, fg='yellow', attr='italic')
            self.addstr(self.y - 1, self.x + 32, '(Press T(TERM)/K(KILL)/^c(INT) to send signals)')
            self.color_at(self.y - 1, self.x + 39, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y - 1, self.x + 41, width=4, fg='red', attr='bold')
            self.color_at(self.y - 1, self.x + 47, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y - 1, self.x + 49, width=4, fg='red', attr='bold')
            self.color_at(self.y - 1, self.x + 55, width=2, fg='magenta', attr='bold | italic')
            self.color_at(self.y - 1, self.x + 58, width=3, fg='red', attr='bold')
        else:
            self.addstr(self.y - 1, self.x, ' ' * self.width)

    def finalize(self):
        self.need_redraw = False

    def destroy(self):
        super().destroy()
        self._daemon_started.clear()

    def print(self):
        lines = self.header_lines()

        if len(self.snapshots) > 0:
            prev_device_index = None
            color = None
            for process in self.snapshots:
                device_index = process.device.index
                if prev_device_index != device_index:
                    color = process.device.display_color
                    if prev_device_index is not None:
                        lines.append('├─────────────────────────────────────────────────────────────────────────────┤')
                    prev_device_index = device_index

                line = '│ {} {:>6} {} {:>7} {:>8} {:<45} │'.format(
                    colored('{:>3}'.format(device_index), color),
                    cut_string(process.pid, maxlen=6, padstr='.'), process.type,
                    cut_string(process.username, maxlen=7, padstr='+'),
                    process.gpu_memory_human,
                    cut_string(process.host_info, padstr='..', maxlen=45)
                )
                if not process.is_running and process.command == 'No Such Process':
                    line = line.replace(process.command, colored(process.command, color='red'))
                lines.append(line)

            lines.append('╘═════════════════════════════════════════════════════════════════════════════╛')

        print('\n'.join(lines))

    def press(self, key):
        self.root.keymaps.use_keymap('process')
        self.root.press(key)

    def click(self, event):
        direction = event.wheel_direction()
        if event.shift():
            self.host_offset += direction
        else:
            self.selected.move(direction=direction)
        return True
