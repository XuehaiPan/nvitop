# This file is part of nvitop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import signal
import threading
import time
from collections import OrderedDict

import psutil
from cachetools.func import ttl_cache

from ..displayable import Displayable
from ..utils import colored, cut_string


class Selected(object):
    def __init__(self, panel):
        self.panel = panel
        self.current_user = self.panel.current_user
        self.index = None
        self._proc = None
        self._ident = None

    @property
    def identity(self):
        if self._ident is None:
            try:
                self._ident = self.process.identity
            except AttributeError:
                try:
                    self._ident = self.process._ident  # pylint: disable=protected-access
                except AttributeError:
                    pass
        return self._ident

    @property
    def process(self):
        return self._proc

    @process.setter
    def process(self, process):
        self._proc = process
        self._ident = None

    def move(self, direction=0):
        if direction == 0:
            return

        with self.panel.snapshot_lock:
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
        return self.is_set() and self.process.username == self.current_user

    def send_signal(self, sig):
        if self.owned():
            try:
                psutil.Process(self.process.pid).send_signal(sig)
            except psutil.Error:
                pass
            else:
                if sig != signal.SIGINT:
                    self.clear()
                time.sleep(1.0)

    def kill(self):
        return self.send_signal(signal.SIGKILL)

    def terminate(self):
        return self.send_signal(signal.SIGTERM)

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


class ProcessPanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7

    def __init__(self, devices, win=None, root=None):
        super(ProcessPanel, self).__init__(win, root)
        self.width = 79
        self.height = 6

        self.devices = devices

        self.snapshots = []
        self.snapshot_lock = threading.RLock()
        self.take_snapshot()
        self.snapshot_daemon = threading.Thread(name='process-snapshot-daemon',
                                                target=self._snapshot_target, daemon=True)
        self.daemon_started = threading.Event()

        self.current_user = psutil.Process().username()
        self.selected = Selected(panel=self)
        self.offset = -1

    def header_lines(self):
        header = [
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ Processes:                                                                  │',
            '│ GPU    PID    USER  GPU MEM  %CPU  %MEM      TIME  COMMAND                  │',
            '╞═════════════════════════════════════════════════════════════════════════════╡'
        ]
        if len(self.snapshots) == 0:
            header.extend([
                '│  No running compute processes found                                         │',
                '╘═════════════════════════════════════════════════════════════════════════════╛'
            ])
        return header

    def take_snapshot(self):
        snapshots = list(filter(None, map(lambda process: process.snapshot(), self.processes.values())))

        with self.snapshot_lock:
            self.snapshots = snapshots

        return snapshots

    def _snapshot_target(self):
        self.daemon_started.wait()
        while self.daemon_started.is_set():
            self.take_snapshot()
            time.sleep(self.SNAPSHOT_INTERVAL)

    @property
    @ttl_cache(ttl=1.0)
    def processes(self):
        processes = {}
        for device in self.devices:
            for p in device.processes.values():
                try:
                    processes[(p.device.index, p.username(), p.pid)] = p
                except psutil.Error:
                    pass
        return OrderedDict([((key[-1], key[0]), processes[key]) for key in sorted(processes.keys())])

    def poke(self):
        if not self.daemon_started.is_set():
            self.daemon_started.set()
            self.snapshot_daemon.start()

        with self.snapshot_lock:
            snapshots = self.snapshots

        n_processes = len(snapshots)
        n_used_devices = len(set([p.device.index for p in snapshots]))
        if n_processes > 0:
            height = 5 + n_processes + n_used_devices - 1
        else:
            height = 6

        max_info_len = 0
        for process in snapshots:
            max_info_len = max(max_info_len, len(process.host_info))
        self.offset = max(-1, min(self.offset, max_info_len - 47))

        self.need_redraw = (self.need_redraw or self.height > height)
        self.height = height

        super(ProcessPanel, self).poke()

    def draw(self):
        self.color_reset()

        if self.need_redraw:
            for y, line in enumerate(self.header_lines(), start=self.y):
                self.addstr(y, self.x, line)

        if self.offset < 22:
            self.addstr(self.y + 2, self.x + 31,
                        '{:<46}'.format('%CPU  %MEM      TIME  COMMAND'[max(self.offset, 0):]))
        else:
            self.addstr(self.y + 2, self.x + 31, '{:<46}'.format('COMMAND'))
        self.addstr(self.y, self.x + 23, '═' * 54)

        with self.snapshot_lock:
            snapshots = self.snapshots

        self.selected.index = None
        if len(snapshots) > 0:
            y = self.y + 4
            prev_device_index = None
            color = -1
            for i, process in enumerate(snapshots):
                device_index = process.device.index
                if prev_device_index is None or prev_device_index != device_index:
                    color = process.device.display_color
                host_info = process.host_info
                if self.offset < 0:
                    host_info = cut_string(host_info, padstr='..', maxlen=47)
                else:
                    host_info = host_info[self.offset:self.offset + 47]

                if prev_device_index is not None and prev_device_index != device_index:
                    self.addstr(y, self.x,
                                '├─────────────────────────────────────────────────────────────────────────────┤')
                    y += 1
                prev_device_index = device_index
                self.addstr(y, self.x,
                            '│ {:>3} {:>6} {:>7} {:>8} {:<47} │'.format(
                                device_index, process.pid, cut_string(process.username, maxlen=7, padstr='+'),
                                process.gpu_memory_human, host_info
                            ))
                if self.offset > 0:
                    self.addstr(y, self.x + 30, ' ')
                if self.selected.index is None and process.identity == self.selected.identity:
                    self.color_at(y, self.x + 1, width=77, fg='cyan', attr='bold | reverse')
                    self.selected.index = i
                    self.selected.process = process

                    memory_usage = ' Memory-Usage: {} '.format(process.device.memory_usage)
                    gpu_utilization = ' GPU-Util: {} '.format(process.device.gpu_utilization)
                    mem_len, gpu_len = len(memory_usage), len(gpu_utilization)
                    x = self.x + self.width - 2 - mem_len - 2 - gpu_len
                    self.addstr(self.y, x, memory_usage)
                    self.addstr(self.y, x + 2 + mem_len, gpu_utilization)
                    self.color_at(self.y, x, width=mem_len, fg=color)
                    self.color_at(self.y, x + 2 + mem_len, width=gpu_len, fg=color)
                else:
                    self.color_at(y, self.x + 2, width=3, fg=color)
                y += 1
            self.addstr(y, self.x, '╘═════════════════════════════════════════════════════════════════════════════╛')
        else:
            self.addstr(self.y + 4, self.x,
                        '│  No running compute processes found                                         │')
            self.offset = -1

        if self.selected.owned():
            self.addstr(self.y - 1, self.x + 12,
                        '(Press k(KILL)/t(TERM)/^c(INT) to send signals to selected process)')
            self.color_at(self.y - 1, self.x + 19, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y - 1, self.x + 21, width=4, fg='red', attr='bold')
            self.color_at(self.y - 1, self.x + 27, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y - 1, self.x + 29, width=4, fg='red', attr='bold')
            self.color_at(self.y - 1, self.x + 35, width=2, fg='magenta', attr='bold | italic')
            self.color_at(self.y - 1, self.x + 38, width=3, fg='red', attr='bold')
        else:
            self.addstr(self.y - 1, self.x, ' ' * self.width)

    def finalize(self):
        self.need_redraw = False

    def destroy(self):
        super(ProcessPanel, self).destroy()
        self.daemon_started.clear()

    def print(self):
        snapshots = self.take_snapshot()

        lines = self.header_lines()

        if len(snapshots) > 0:
            prev_device_index = None
            color = None
            for process in snapshots:
                device_index = process.device.index
                if prev_device_index is None or prev_device_index != device_index:
                    color = process.device.display_color
                if prev_device_index is not None and prev_device_index != device_index:
                    lines.append('├─────────────────────────────────────────────────────────────────────────────┤')
                prev_device_index = device_index

                lines.append('│ {} {:>6} {:>7} {:>8} {:<47} │'.format(
                    colored('{:>3}'.format(device_index), color), process.pid,
                    cut_string(process.username, maxlen=7, padstr='+'),
                    process.gpu_memory_human, cut_string(process.host_info, padstr='..', maxlen=47)
                ))

            lines.append('╘═════════════════════════════════════════════════════════════════════════════╛')

        print('\n'.join(lines))

    def press(self, key):
        self.root.keymaps.use_keymap('process')
        self.root.press(key)

    def click(self, event):
        direction = event.wheel_direction()
        if event.shift():
            self.offset += direction
        else:
            self.selected.move(direction=direction)
        return True
