# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name,line-too-long

import threading
import time

from ..displayable import Displayable
from ...utils import colored, cut_string, nvml_check_return, nvml_query


class DevicePanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7

    def __init__(self, devices, compact, win, root=None):
        super().__init__(win, root)

        self.devices = devices
        self.device_count = len(self.devices)
        self._compact = compact
        self.width = 79
        self.height = 3 + (3 - int(compact)) * (self.device_count + 1)
        self.full_height = 3 + 3 * (self.device_count + 1)
        if self.device_count == 0:
            self.height = self.full_height = 5

        self.driver_version = str(nvml_query('nvmlSystemGetDriverVersion'))
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
        self.snapshot_lock = threading.RLock()
        self.snapshots = self.take_snapshots()
        self._snapshot_daemon = threading.Thread(name='device-snapshot-daemon',
                                                 target=self._snapshot_target, daemon=True)
        self._daemon_started = threading.Event()

    @property
    def compact(self):
        return self._compact

    @compact.setter
    def compact(self, value):
        if self._compact != value:
            self.need_redraw = True
            self._compact = value
            self.height = 3 + (3 - int(self.compact)) * (self.device_count + 1)

    @property
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots):
        with self.snapshot_lock:
            self._snapshots = snapshots

    def take_snapshots(self):
        snapshots = list(map(lambda device: device.snapshot(), self.devices))

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

    def frame_lines(self):
        frame = self.header_lines()
        if self.device_count > 0:
            if self.compact:
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
            for y, line in enumerate(self.frame_lines(), start=self.y):
                self.addstr(y, self.x, line)

        if self.compact:
            formats = self.formats_compact
        else:
            formats = self.formats_full

        for index, device in enumerate(self.snapshots):
            device.name = cut_string(device.name, maxlen=18)
            for y, fmt in enumerate(formats, start=self.y + 3 + (len(formats) + 1) * (index + 1)):
                self.addstr(y, self.x, fmt.format(**device.__dict__))
                self.color_at(y, 1, width=31, fg=device.display_color)
                self.color_at(y, 33, width=22, fg=device.display_color)
                self.color_at(y, 56, width=22, fg=device.display_color)

    def finalize(self):
        self.need_redraw = False

    def destroy(self):
        super().destroy()
        self._daemon_started.clear()

    def print(self):
        lines = self.header_lines()

        if self.device_count > 0:
            for device in self.snapshots:
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
