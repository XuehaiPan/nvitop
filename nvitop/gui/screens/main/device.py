# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=disallowed-name,invalid-name

import threading
import time

from cachetools.func import ttl_cache

from nvitop.core import Device
from nvitop.gui.library import Displayable, colored, cut_string, make_bar


class DevicePanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7

    def __init__(self, devices, compact, win, root):
        super().__init__(win, root)

        self.devices = devices
        self.device_count = len(self.devices)

        self._compact = compact
        self.width = max(79, root.width)
        self.full_height = 4 + 3 * (self.device_count + 1)
        self.compact_height = 4 + 2 * (self.device_count + 1)
        self.height = (self.compact_height if compact else self.full_height)
        if self.device_count == 0:
            self.height = self.full_height = self.compact_height = 6

        self.driver_version = Device.driver_version()
        self.cuda_version = Device.cuda_version()

        self.formats_compact = [
            '│ {index:>3} {fan_speed_string:>3} {temperature_string:>4} {performance_state:>3} {power_status:>12} '
            '│ {memory_usage:>20} │ {gpu_utilization_string:>7}  {compute_mode:>11} │',
        ]
        self.formats_full = [
            '│ {index:>3}  {name:<18}  {persistence_mode:<4} '
            '│ {bus_id:<16} {display_active:>3} │ {ecc_errors:>20} │',
            '│ {fan_speed_string:>3}  {temperature_string:>4}  {performance_state:>4}  {power_status:>12} '
            '│ {memory_usage:>20} │ {gpu_utilization_string:>7}  {compute_mode:>11} │',
        ]

        Device.SNAPSHOT_KEYS.extend(['memory_loading_intensity', 'memory_display_color',
                                     'gpu_loading_intensity', 'gpu_display_color',
                                     'loading_intensity', 'display_color'])

        self._snapshot_buffer = []
        self._snapshots = []
        self.snapshot_lock = root.lock
        self.snapshots = self.take_snapshots()
        self._snapshot_daemon = threading.Thread(name='device-snapshot-daemon',
                                                 target=self._snapshot_target, daemon=True)
        self._daemon_running = threading.Event()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        width = max(79, value)
        if self._width != width and self.visible:
            self.need_redraw = True
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

    @property
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots):
        with self.snapshot_lock:
            self._snapshots = snapshots

    @ttl_cache(ttl=1.0)
    def take_snapshots(self):
        snapshots = list(map(lambda device: device.as_snapshot(), self.devices))

        for device in snapshots:
            if device.name.startswith('NVIDIA '):
                device.name = device.name.replace('NVIDIA ', '', 1)
            device.name = cut_string(device.name, maxlen=18)
            if device.fan_speed >= 100:
                device.fan_speed_string = 'MAX'

        with self.snapshot_lock:
            self._snapshot_buffer = snapshots

        return snapshots

    def _snapshot_target(self):
        self._daemon_running.wait()
        while self._daemon_running.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def header_lines(self, compact=None):
        if compact is None:
            compact = self.compact

        header = [
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ NVIDIA-SMI {0:<12} Driver Version: {0:<12} CUDA Version: {1:<8} │'.format(self.driver_version,
                                                                                         self.cuda_version),
        ]
        if self.device_count > 0:
            header.append('├───────────────────────────────┬──────────────────────┬──────────────────────┤')
            if compact:
                header.append('│ GPU Fan Temp Perf Pwr:Usg/Cap │         Memory-Usage │ GPU-Util  Compute M. │')
            else:
                header.extend([
                    '│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │',
                    '│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │',
                ])
            header.append('╞═══════════════════════════════╪══════════════════════╪══════════════════════╡')
        else:
            header.extend([
                '╞═════════════════════════════════════════════════════════════════════════════╡',
                '│  No visible CUDA devices found                                              │',
                '╘═════════════════════════════════════════════════════════════════════════════╛',
            ])
        return header

    def frame_lines(self, compact=None):
        if compact is None:
            compact = self.compact

        frame = self.header_lines(compact)

        remaining_width = self.width - 79
        data_line = '│                               │                      │                      │'
        separator_line = '├───────────────────────────────┼──────────────────────┼──────────────────────┤'
        if self.width >= 100:
            data_line += ' ' * (remaining_width - 1) + '│'
            separator_line = separator_line[:-1] + '┼' + '─' * (remaining_width - 1) + '┤'

        if self.device_count > 0:
            if compact:
                frame.extend(self.device_count * [data_line, separator_line])
            else:
                frame.extend(self.device_count * [data_line, data_line, separator_line])
            frame.pop()
            frame.append('╘═══════════════════════════════╧══════════════════════╧══════════════════════╛')
            if self.width >= 100:
                frame[5 - int(compact)] = frame[5 - int(compact)][:-1] + '╪' + '═' * (remaining_width - 1) + '╕'
                frame[-1] = frame[-1][:-1] + '╧' + '═' * (remaining_width - 1) + '╛'

        return frame

    def poke(self):
        if not self._daemon_running.is_set():
            self._daemon_running.set()
            self._snapshot_daemon.start()

        with self.snapshot_lock:
            self.snapshots = self._snapshot_buffer

        super().poke()

    def draw(self):
        self.color_reset()

        if self.need_redraw:
            self.addstr(self.y, self.x, '(Press h for help or q to quit)'.rjust(79))
            self.color_at(self.y, self.x + 55, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y, self.x + 69, width=1, fg='magenta', attr='bold | italic')
            for y, line in enumerate(self.frame_lines(), start=self.y + 1):
                self.addstr(y, self.x, line)

        time_string = time.strftime('%a %b %d %H:%M:%S %Y')
        self.addstr(self.y, self.x, '{:<32}'.format(time_string))

        if self.compact:
            formats = self.formats_compact
        else:
            formats = self.formats_full

        remaining_width = self.width - 79
        draw_bars = (self.width >= 100)
        try:
            selected_device = self.parent.selected.process.device
        except AttributeError:
            selected_device = None
        for index, device in enumerate(self.snapshots):
            y_start = self.y + 4 + (len(formats) + 1) * (index + 1)
            if selected_device is not None:
                attr = ('bold' if device.real == selected_device else 'dim')
            else:
                attr = 0

            for y, fmt in enumerate(formats, start=y_start):
                self.addstr(y, self.x, fmt.format(**device.__dict__))
                self.color_at(y, self.x + 1, width=31, fg=device.display_color, attr=attr)
                self.color_at(y, self.x + 33, width=22, fg=device.display_color, attr=attr)
                self.color_at(y, self.x + 56, width=22, fg=device.display_color, attr=attr)

            if draw_bars:
                matrix = [
                    ('MEM', device.memory_utilization, device.memory_display_color),
                    ('UTL', device.gpu_utilization, device.gpu_display_color),
                ]
                if self.compact:
                    matrix.pop()
                for y, (prefix, utilization, color) in enumerate(matrix, start=y_start):
                    bar = make_bar(prefix, utilization, width=remaining_width - 3)
                    self.addstr(y, self.x + 80, bar + ' │')
                    self.color_at(y, self.x + 80, width=remaining_width - 3, fg=color, attr=attr)

    def destroy(self):
        super().destroy()
        self._daemon_running.clear()

    def print_width(self):
        if self.device_count > 0 and self.width >= 100:
            return self.width
        return 79

    def print(self):
        lines = [time.strftime('%a %b %d %H:%M:%S %Y'), *self.header_lines(compact=False)]

        if self.device_count > 0:
            for device in self.snapshots:
                def colorize(s):
                    if len(s) > 0:
                        return colored(s, device.display_color)  # pylint: disable=cell-var-from-loop
                    return ''

                for fmt in self.formats_full:
                    line = fmt.format(**device.__dict__)
                    lines.append('│'.join(map(colorize, line.split('│'))))

                lines.append('├───────────────────────────────┼──────────────────────┼──────────────────────┤')
            lines.pop()
            lines.append('╘═══════════════════════════════╧══════════════════════╧══════════════════════╛')

            if self.width >= 100:
                remaining_width = self.width - 79
                for index, device in enumerate(self.snapshots):
                    y_start = 4 + 3 * (index + 1)
                    lines[y_start - 1] = lines[y_start - 1][:-1]

                    if index == 0:
                        lines[y_start - 1] += '╪' + '═' * (remaining_width - 1) + '╕'
                    else:
                        lines[y_start - 1] += '┼' + '─' * (remaining_width - 1) + '┤'

                    matrix = [
                        ('MEM', device.memory_utilization,
                         Device.INTENSITY2COLOR[device.memory_loading_intensity]),
                        ('UTL', device.gpu_utilization,
                         Device.INTENSITY2COLOR[device.gpu_loading_intensity]),
                    ]
                    for y, (prefix, utilization, color) in enumerate(matrix, start=y_start):
                        bar = make_bar(prefix, utilization, width=remaining_width - 3)
                        lines[y] += ' {} │'.format(colored(bar, color))

                    if index == len(self.snapshots) - 1:
                        lines[y_start + 2] = lines[y_start + 2][:-1] + '╧' + '═' * (remaining_width - 1) + '╛'

        lines = '\n'.join(lines)
        if self.ascii:
            lines = lines.translate(self.ASCII_TRANSTABLE)

        try:
            print(lines)
        except UnicodeError:
            print(lines.translate(self.ASCII_TRANSTABLE))

    def press(self, key):
        self.root.keymaps.use_keymap('device')
        self.root.press(key)
