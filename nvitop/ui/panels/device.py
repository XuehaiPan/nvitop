# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name,line-too-long

import threading
import time

from ...device import Device
from ...utils import colored, cut_string, nvml_check_return, nvml_query
from ..displayable import Displayable


class DevicePanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7

    def __init__(self, devices, compact, win, root=None):
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

        self.driver_version = nvml_query('nvmlSystemGetDriverVersion')
        cuda_version = nvml_query('nvmlSystemGetCudaDriverVersion')
        if nvml_check_return(cuda_version, int):
            self.cuda_version = str(cuda_version // 1000 + (cuda_version % 1000) / 100)
        else:
            self.cuda_version = 'N/A'

        self.formats_compact = [
            '│ {index:>3} {fan_speed:>3} {temperature:>4} {performance_state:>3} {power_state:>12} '
            '│ {memory_usage:>20} │ {gpu_utilization:>7}  {compute_mode:>11} │',
        ]
        self.formats_full = [
            '│ {index:>3}  {name:>18}  {persistence_mode:<4} '
            '│ {bus_id:<16} {display_active:>3} │ {ecc_errors:>20} │',
            '│ {fan_speed:>3}  {temperature:>4}  {performance_state:>4}  {power_state:>12} '
            '│ {memory_usage:>20} │ {gpu_utilization:>7}  {compute_mode:>11} │',
        ]

        self._snapshot_buffer = []
        self._snapshots = []
        self.snapshot_lock = root.lock
        self.snapshots = self.take_snapshots()
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

    def take_snapshots(self):
        snapshots = list(map(lambda device: device.take_snapshot(), self.devices))

        with self.snapshot_lock:
            self._snapshot_buffer = snapshots

        return snapshots

    def _snapshot_target(self):
        self._daemon_started.wait()
        while self._daemon_started.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def header_lines(self, compact=None):
        if compact is None:
            compact = self.compact

        header = [
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ NVIDIA-SMI {0:<6}       Driver Version: {0:<6}       CUDA Version: {1:<5}    │'.format(self.driver_version,
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
        if self.device_count > 0:
            if compact:
                frame.extend(self.device_count * [
                    '│                               │                      │                      │',
                    '├───────────────────────────────┼──────────────────────┼──────────────────────┤',
                ])
            else:
                frame.extend(self.device_count * [
                    '│                               │                      │                      │',
                    '│                               │                      │                      │',
                    '├───────────────────────────────┼──────────────────────┼──────────────────────┤',
                ])
            frame.pop()
            frame.append('╘═══════════════════════════════╧══════════════════════╧══════════════════════╛')
        return frame

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
            self.addstr(self.y, self.x + 62, '(Press q to quit)')
            self.color_at(self.y, self.x + 69, width=1, fg='magenta', attr='bold | italic')
            for y, line in enumerate(self.frame_lines(), start=self.y + 1):
                self.addstr(y, self.x, line)

        time_string = time.strftime('%a %b %d %H:%M:%S %Y')
        self.addstr(self.y, self.x, '{:<62}'.format(time_string))
        self.color_at(self.y, self.x + len(time_string) - 11, width=1, attr='blink')
        self.color_at(self.y, self.x + len(time_string) - 8, width=1, attr='blink')

        if self.compact:
            formats = self.formats_compact
        else:
            formats = self.formats_full

        for index, device in enumerate(self.snapshots):
            y_start = self.y + 4 + (len(formats) + 1) * (index + 1)

            device.name = cut_string(device.name, maxlen=18)
            for y, fmt in enumerate(formats, start=y_start):
                self.addstr(y, self.x, fmt.format(**device.__dict__))
                self.color_at(y, 1, width=31, fg=device.display_color)
                self.color_at(y, 33, width=22, fg=device.display_color)
                self.color_at(y, 56, width=22, fg=device.display_color)

        remaining_width = self.width - 79
        if remaining_width >= 22:
            block_chars = ' ▏▎▍▌▋▊▉'
            for index, device in enumerate(self.snapshots):
                y_start = self.y + 4 + (len(formats) + 1) * (index + 1)

                if index == 0:
                    self.addstr(y_start - 1, self.x + 78, '╪' + '═' * (remaining_width - 1) + '╕')
                else:
                    self.addstr(y_start - 1, self.x + 78, '┼' + '─' * (remaining_width - 1) + '┤')

                matrix = [
                    ('MEM', device.memory_utilization,
                     Device.INTENSITY2COLOR[device.memory_loading_intensity]),
                    ('GPU', device.gpu_utilization,
                     Device.INTENSITY2COLOR[device.gpu_loading_intensity]),
                ]
                if self.compact:
                    matrix.pop()
                for y, (prefix, utilization, color) in enumerate(matrix, start=y_start):
                    bar = ' {}: '.format(prefix)
                    if utilization != 'N/A':
                        percentage = float(utilization[:-1]) / 100.0
                        quotient, remainder = divmod(max(1, int(8 * (remaining_width - 12) * percentage)), 8)
                        bar += '█' * quotient
                        if remainder > 0:
                            bar += block_chars[remainder]
                        bar += ' ' + utilization.replace('100%', 'MAX')
                    else:
                        bar += '░' * (remaining_width - 6) + ' N/A'
                    self.addstr(y, self.x + 79, bar.ljust(remaining_width - 1) + '│')
                    self.color_at(y, self.x + 80, width=remaining_width - 3, fg=color)

                if index == len(self.snapshots) - 1:
                    self.addstr(y_start + len(formats), self.x + 78,
                                '╧' + '═' * (remaining_width - 1) + '╛')

    def finalize(self):
        self.need_redraw = False

    def destroy(self):
        super().destroy()
        self._daemon_started.clear()

    def print_width(self):
        if len(self.devices) > 0 and self.width >= 101:
            return self.width
        else:
            return 79

    def print(self):
        lines = [time.strftime('%a %b %d %H:%M:%S %Y'), *self.header_lines(compact=False)]

        if self.device_count > 0:
            for device in self.snapshots:
                device.name = cut_string(device.name, maxlen=18)

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

            remaining_width = self.width - 79
            if remaining_width >= 22:
                block_chars = ' ▏▎▍▌▋▊▉'
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
                        ('GPU', device.gpu_utilization,
                         Device.INTENSITY2COLOR[device.gpu_loading_intensity]),
                    ]
                    for y, (prefix, utilization, color) in enumerate(matrix, start=y_start):
                        bar = ' {}: '.format(prefix)
                        if utilization != 'N/A':
                            percentage = float(utilization[:-1]) / 100.0
                            quotient, remainder = divmod(max(1, int(8 * (remaining_width - 12) * percentage)), 8)
                            bar += '█' * quotient
                            if remainder > 0:
                                bar += block_chars[remainder]
                            bar += ' ' + utilization.replace('100%', 'MAX')
                        else:
                            bar += '░' * (remaining_width - 6) + ' N/A'
                        lines[y] += colored(bar.ljust(remaining_width - 1), color) + '│'

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
