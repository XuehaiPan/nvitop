# This file is part of nvitop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name,line-too-long

import threading
import time
from collections import OrderedDict

import psutil
from cachetools.func import ttl_cache

from .displayable import Displayable
from .utils import colored, cut_string, nvml_check_return, nvml_query


class DevicePanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7

    def __init__(self, devices, compact, win, root=None):
        super(DevicePanel, self).__init__(win, root)

        self.devices = devices
        self.device_count = len(self.devices)
        self._compact = compact
        self.width = 79
        self.height = 4 + (3 - int(compact)) * (self.device_count + 1)
        self.full_height = 4 + 3 * (self.device_count + 1)
        if self.device_count == 0:
            self.height = self.full_height = 6

        self.driver_version = str(nvml_query('nvmlSystemGetDriverVersion'))
        cuda_version = nvml_query('nvmlSystemGetCudaDriverVersion')
        if nvml_check_return(cuda_version, int):
            self.cuda_version = str(cuda_version // 1000 + (cuda_version % 1000) / 100)
        else:
            self.cuda_version = 'N/A'

        self.formats_compact = [
            '│ {index:>3} {fan_speed:>3} {temperature:>4} {performance_state:>3} {power_state:>12} '
            '│ {memory_usage:>20} │ {gpu_utilization:>7}  {compute_mode:>11} │'
        ]
        self.formats_full = [
            '│ {index:>3}  {name:>18}  {persistence_mode:<4} '
            '│ {bus_id:<16} {display_active:>3} │ {ecc_errors:>20} │',
            '│ {fan_speed:>3}  {temperature:>4}  {performance_state:>4}  {power_state:>12} '
            '│ {memory_usage:>20} │ {gpu_utilization:>7}  {compute_mode:>11} │'
        ]

        self.snapshots = []
        self.snapshot_lock = threading.RLock()
        self.take_snapshot()
        self.snapshot_daemon = threading.Thread(name='device-snapshot-daemon',
                                                target=self._snapshot_target, daemon=True)
        self.daemon_started = threading.Event()

    @property
    def compact(self):
        return self._compact

    @compact.setter
    def compact(self, value):
        if self._compact != value:
            self.need_redraw = True
            self._compact = value
            self.height = 4 + (3 - int(self.compact)) * (self.device_count + 1)

    def header_lines(self):
        header = [
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ NVIDIA-SMI {0:<6}       Driver Version: {0:<6}       CUDA Version: {1:<5}    │'.format(self.driver_version,
                                                                                                      self.cuda_version),
        ]
        if self.device_count > 0:
            header.append('├───────────────────────────────┬──────────────────────┬──────────────────────┤')
            if self.compact:
                header.append('│ GPU Fan Temp Perf Pwr:Usg/Cap │         Memory-Usage │ GPU-Util  Compute M. │')
            else:
                header.extend([
                    '│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │',
                    '│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │'
                ])
            header.append('╞═══════════════════════════════╪══════════════════════╪══════════════════════╡')
        else:
            header.extend([
                '╞═════════════════════════════════════════════════════════════════════════════╡',
                '│  No visible CUDA devices found                                              │',
                '╘═════════════════════════════════════════════════════════════════════════════╛'
            ])
        return header

    def frame_lines(self):
        frame = self.header_lines()
        if self.device_count > 0:
            if self.compact:
                frame.extend(self.device_count * [
                    '│                               │                      │                      │',
                    '├───────────────────────────────┼──────────────────────┼──────────────────────┤'
                ])
            else:
                frame.extend(self.device_count * [
                    '│                               │                      │                      │',
                    '│                               │                      │                      │',
                    '├───────────────────────────────┼──────────────────────┼──────────────────────┤'
                ])
            frame.pop()
            frame.append('╘═══════════════════════════════╧══════════════════════╧══════════════════════╛')
        return frame

    def take_snapshot(self):
        snapshots = list(map(lambda device: device.snapshot(), self.devices))

        with self.snapshot_lock:
            self.snapshots = snapshots

        return snapshots

    def _snapshot_target(self):
        self.daemon_started.wait()
        while self.daemon_started.is_set():
            self.take_snapshot()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def poke(self):
        if not self.daemon_started.is_set():
            self.daemon_started.set()
            self.snapshot_daemon.start()

        super(DevicePanel, self).poke()

    def draw(self):
        self.color_reset()

        if self.need_redraw:
            for y, line in enumerate(self.frame_lines(), start=self.y + 1):
                self.addstr(y, self.x, line)
            self.addstr(self.y, self.x + 62, '(Press q to quit)')
            self.color_at(self.y, self.x + 69, width=1, fg='magenta', attr='bold | italic')

        self.addstr(self.y, self.x, '{:<62}'.format(time.strftime('%a %b %d %H:%M:%S %Y')))

        if self.compact:
            formats = self.formats_compact
        else:
            formats = self.formats_full

        with self.snapshot_lock:
            snapshots = self.snapshots
        for index, device in enumerate(snapshots):
            device.name = cut_string(device.name, maxlen=18)
            for y, fmt in enumerate(formats, start=self.y + 4 + (len(formats) + 1) * (index + 1)):
                self.addstr(y, self.x, fmt.format(**device.__dict__))
                self.color_at(y, 2, width=29, fg=device.display_color)
                self.color_at(y, 34, width=20, fg=device.display_color)
                self.color_at(y, 57, width=20, fg=device.display_color)

    def finalize(self):
        self.need_redraw = False

    def destroy(self):
        super(DevicePanel, self).destroy()
        self.daemon_started.clear()

    def print(self):
        snapshots = self.take_snapshot()

        lines = [
            '{:<79}'.format(time.strftime('%a %b %d %H:%M:%S %Y')),
            *self.header_lines()
        ]

        if self.device_count > 0:
            for device in snapshots:
                device.name = cut_string(device.name, maxlen=18)

                def colorize(s):
                    return colored(s, device.display_color)  # pylint: disable=cell-var-from-loop

                for fmt in self.formats_full:
                    line = fmt.format(**device.__dict__)
                    lines.append('│'.join(map(colorize, line.split('│'))))

                lines.append('├───────────────────────────────┼──────────────────────┼──────────────────────┤')
            lines.pop()
            lines.append('╘═══════════════════════════════╧══════════════════════╧══════════════════════╛')

        print('\n'.join(lines))

    def press(self, key):
        self.root.keymaps.use_keymap('device')
        self.root.press(key)


class ProcessPanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7

    class Selected(object):
        def __init__(self, index=None, process=None):
            self.index = index
            self._proc = None
            self._ident = None
            self.process = process

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

        self.selected = self.Selected()
        self.offset = -1
        self.current_user = psutil.Process().username()

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
                else:
                    self.color_at(y, self.x + 2, width=3, fg=color)
                y += 1
            self.addstr(y, self.x, '╘═════════════════════════════════════════════════════════════════════════════╛')
        else:
            self.addstr(self.y + 4, self.x,
                        '│  No running compute processes found                                         │')
            self.offset = -1
        if self.selected.index is None:
            self.selected.clear()

        if self.selected.is_set() and self.selected.process.username == self.current_user:
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
