# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name,line-too-long

import threading
import time

from ...core import BufferedHistoryGraph, Device, host
from ...core.utils import colored, make_bar
from ..displayable import Displayable


class HostPanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7

    def __init__(self, devices, compact, win, root=None):
        super().__init__(win, root)

        self.devices = devices
        self.device_count = len(self.devices)

        prefix = ('AVG ' if self.device_count > 1 else '')
        self.average_memory_utilization = BufferedHistoryGraph(
            baseline=0.0,
            upperbound=100.0,
            width=32,
            height=5,
            dynamic_bound=False,
            format=(prefix + 'GPU MEM: {:.1f}%').format
        )
        self.average_gpu_utilization = BufferedHistoryGraph(
            baseline=0.0,
            upperbound=100.0,
            width=32,
            height=5,
            dynamic_bound=False,
            upsidedown=True,
            format=(prefix + 'GPU UTL: {:.1f}%').format
        )

        self._compact = compact
        self.width = max(79, root.width)
        self.full_height = 12
        self.compact_height = 2
        self.height = (self.compact_height if compact else self.full_height)

        self.cpu_percent = None
        self.virtual_memory = None
        self.swap_memory = None
        self.load_average = None
        self.snapshot_lock = root.lock
        self.take_snapshots()
        self._snapshot_daemon = threading.Thread(name='device-snapshot-daemon',
                                                 target=self._snapshot_target, daemon=True)
        self._daemon_started = threading.Event()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        width = max(79, value)
        if self._width != width and self.visible:
            self.need_redraw = True
            graph_width = max(width - 80, 20)
            self.average_memory_utilization.width = graph_width
            self.average_gpu_utilization.width = graph_width
            for device in self.devices:
                device.memory_utilization.history.width = graph_width
                device.gpu_utilization.history.width = graph_width
        self._width = width

    @property
    def compact(self):
        return self._compact

    @compact.setter
    def compact(self, value):
        if self._compact != value:
            self.need_redraw = True
            self._compact = value
            self.height = (self.compact_height if self.compact else self.full_height)

    def take_snapshots(self):
        with self.snapshot_lock:
            host.cpu_percent()
            self.cpu_percent = host.cpu_percent.history.last_value
            self.virtual_memory = host.virtual_memory()
            self.swap_memory = host.swap_memory()
            self.load_average = host.load_average()

            memory_utilizations = []
            gpu_utilizations = []
            for device in self.devices:
                memory_utilization = device.snapshot.memory_utilization
                gpu_utilization = device.snapshot.gpu_utilization
                if memory_utilization != 'N/A':
                    memory_utilizations.append(float(memory_utilization[:-1]))
                if gpu_utilization != 'N/A':
                    gpu_utilizations.append(float(gpu_utilization[:-1]))
            if len(memory_utilizations) > 0:
                self.average_memory_utilization.add(sum(memory_utilizations) / len(memory_utilizations))
            if len(gpu_utilizations) > 0:
                self.average_gpu_utilization.add(sum(gpu_utilizations) / len(gpu_utilizations))

    def _snapshot_target(self):
        self._daemon_started.wait()
        while self._daemon_started.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def frame_lines(self, compact=None):
        if compact is None:
            compact = self.compact
        if compact or self.ascii:
            return []

        remaining_width = self.width - 79
        data_line = '│                                                                             │'
        separator_line = '├────────────╴120s├─────────────────────────╴60s├──────────╴30s├──────────────┤'
        if remaining_width >= 22:
            data_line += ' ' * (remaining_width - 1) + '│'
            separator_line = separator_line[:-1] + '┼' + '─' * (remaining_width - 1) + '┤'

        frame = [
            '╞═══════════════════════════════╧══════════════════════╧══════════════════════╡',
            data_line, data_line, data_line, data_line, data_line,
            separator_line,
            data_line, data_line, data_line, data_line, data_line,
            '╘═════════════════════════════════════════════════════════════════════════════╛'
        ]
        if remaining_width >= 22:
            frame[0] = frame[0][:-1] + '╪' + '═' * (remaining_width - 1) + '╡'
            frame[-1] = frame[-1][:-1] + '╧' + '═' * (remaining_width - 1) + '╛'

        return frame

    def poke(self):
        if not self._daemon_started.is_set():
            self._daemon_started.set()
            self._snapshot_daemon.start()

        super().poke()

    def draw(self):
        self.color_reset()

        if self.load_average is not None:
            load_average = tuple('{:5.2f}'.format(value) if value < 100.0 else '100.0'
                                 for value in self.load_average)
        else:
            load_average = ('N/A',) * 3
        load_average = 'Load Average: {} {} {}'.format(*load_average)

        if self.compact or self.ascii:
            width_right = len(load_average) + 4
            width_left = self.width - 2 - width_right
            cpu_bar = '[ {} ]'.format(make_bar('CPU', self.cpu_percent, width_left - 4))
            memory_bar = '[ {} ]'.format(make_bar('MEM', self.virtual_memory.percent, width_left - 4))
            swap_bar = '[ {} ]'.format(make_bar('SWP', self.swap_memory.percent, width_right - 4))
            self.addstr(self.y, self.x, '{}  ( {} )'.format(cpu_bar, load_average))
            self.addstr(self.y + 1, self.x, '{}  {}'.format(memory_bar, swap_bar))
            self.color_at(self.y, self.x, width=len(cpu_bar), fg='cyan', attr='bold')
            self.color_at(self.y + 1, self.x, width=width_left, fg='magenta', attr='bold')
            self.color_at(self.y, self.x + width_left + 2, width=width_right, attr='bold')
            self.color_at(self.y + 1, self.x + width_left + 2, width=width_right, fg='blue', attr='bold')
            return

        remaining_width = self.width - 79

        if self.need_redraw:
            for y, line in enumerate(self.frame_lines(), start=self.y - 1):
                self.addstr(y, self.x, line)
            self.color_at(self.y + 5, self.x + 14, width=4, attr='dim')
            self.color_at(self.y + 5, self.x + 45, width=3, attr='dim')
            self.color_at(self.y + 5, self.x + 60, width=3, attr='dim')
            for offset, string in [(20, '╴30s├'), (35, '╴60s├'), (66, '╴120s├'), (126, '╴240s├')]:
                if offset <= remaining_width:
                    self.addstr(self.y + 5, self.x + self.width - offset, string)
                    self.color_at(self.y + 5, self.x + self.width - offset + 1, width=len(string) - 2, attr='dim')

        for y, line in enumerate(host.cpu_percent.history.graph, start=self.y):
            self.addstr(y, self.x + 1, line)
            self.color_at(y, self.x + 1, width=77, fg='cyan')
        self.addstr(self.y, self.x + 1, ' {} '.format(host.cpu_percent.history.last_value_string()))
        self.addstr(self.y + 1, self.x + 1, ' {} '.format(load_average))

        for y, line in enumerate(host.virtual_memory.history.graph, start=self.y + 6):
            self.addstr(y, self.x + 1, line)
            self.color_at(y, self.x + 1, width=77, fg='magenta')
        self.addstr(self.y + 9, self.x + 1, ' {} '.format(host.virtual_memory.history.last_value_string()))
        for y, line in enumerate(host.swap_memory.history.graph, start=self.y + 10):
            self.addstr(y, self.x + 1, line)
            self.color_at(y, self.x + 1, width=77, fg='blue')
        self.addstr(self.y + 10, self.x + 1, ' {} '.format(host.swap_memory.history.last_value_string()))

        if remaining_width >= 22:
            if self.device_count > 1 and self.root.selected.is_set():
                device = self.root.selected.process.device
                memory_utilization = device.memory_utilization.history
                gpu_utilization = device.gpu_utilization.history
                memory_display_color = device.snapshot.memory_display_color
                gpu_display_color = device.snapshot.gpu_display_color
            else:
                memory_utilization = self.average_memory_utilization
                gpu_utilization = self.average_gpu_utilization
                memory_display_color = Device.color_of(memory_utilization.last_value, type='memory')
                gpu_display_color = Device.color_of(gpu_utilization.last_value, type='gpu')

            for y, line in enumerate(memory_utilization.graph, start=self.y):
                self.addstr(y, self.x + 79, line)
                self.color_at(y, self.x + 79, width=remaining_width - 1, fg=memory_display_color)
            self.addstr(self.y, self.x + 79, ' {} '.format(memory_utilization.last_value_string()))

            for y, line in enumerate(gpu_utilization.graph, start=self.y + 6):
                self.addstr(y, self.x + 79, line)
                self.color_at(y, self.x + 79, width=remaining_width - 1, fg=gpu_display_color)
            self.addstr(self.y + 10, self.x + 79, ' {} '.format(gpu_utilization.last_value_string()))

    def finalize(self):
        self.need_redraw = False

    def print_width(self):
        if self.device_count > 0 and self.width >= 101:
            return self.width
        return 79

    def print(self):
        self.cpu_percent = host.cpu_percent()

        if self.load_average is not None:
            load_average = tuple('{:5.2f}'.format(value) if value < 100.0 else '100.0'
                                 for value in self.load_average)
        else:
            load_average = ('N/A',) * 3
        load_average = 'Load Average: {} {} {}'.format(*load_average)

        width_right = len(load_average) + 4
        width_left = self.width - 2 - width_right
        cpu_bar = '[ {} ]'.format(make_bar('CPU', self.cpu_percent, width_left - 4))
        memory_bar = '[ {} ]'.format(make_bar('MEM', self.virtual_memory.percent, width_left - 4))
        swap_bar = '[ {} ]'.format(make_bar('SWP', self.swap_memory.percent, width_right - 4))

        lines = [
            '{}  {}'.format(colored(cpu_bar, color='cyan', attrs=('bold',)),
                            colored('( {} )'.format(load_average), attrs=('bold',))),
            '{}  {}'.format(colored(memory_bar, color='magenta', attrs=('bold',)),
                            colored(swap_bar, color='blue', attrs=('bold',))),
        ]

        lines = '\n'.join(lines)
        if self.ascii:
            lines = lines.translate(self.ASCII_TRANSTABLE)

        try:
            print(lines)
        except UnicodeError:
            print(lines.translate(self.ASCII_TRANSTABLE))

    def press(self, key):
        self.root.keymaps.use_keymap('host')
        self.root.press(key)
