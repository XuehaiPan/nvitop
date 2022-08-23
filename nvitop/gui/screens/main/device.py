# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import threading
import time

from cachetools.func import ttl_cache

from nvitop.gui.library import NA, Device, Displayable, colored, cut_string, host, make_bar
from nvitop.version import __version__


class DevicePanel(Displayable):  # pylint: disable=too-many-instance-attributes
    NAME = 'device'
    SNAPSHOT_INTERVAL = 0.67

    def __init__(self, devices, compact, win, root):
        super().__init__(win, root)

        self.devices = devices
        self.device_count = len(self.devices)

        self.all_devices = []
        self.leaf_devices = []
        self.mig_device_counts = [0] * self.device_count
        self.mig_enabled_device_count = 0
        for i, device in enumerate(self.devices):
            self.all_devices.append(device)

            mig_devices = device.mig_devices()
            self.mig_device_counts[i] = len(mig_devices)
            if self.mig_device_counts[i] > 0:
                self.all_devices.extend(mig_devices)
                self.leaf_devices.extend(mig_devices)
                self.mig_enabled_device_count += 1
            else:
                self.leaf_devices.append(device)

        self.mig_device_count = sum(self.mig_device_counts)
        self.all_device_count = len(self.all_devices)
        self.leaf_device_count = len(self.leaf_devices)

        self._compact = compact
        self.width = max(79, root.width)
        self.compact_height = (
            4 + 2 * (self.device_count + 1) + self.mig_device_count + self.mig_enabled_device_count
        )
        self.full_height = self.compact_height + self.device_count + 1
        self.height = self.compact_height if compact else self.full_height
        if self.device_count == 0:
            self.height = self.full_height = self.compact_height = 6

        self.driver_version = Device.driver_version()
        self.cuda_driver_version = Device.cuda_driver_version()

        self._snapshot_buffer = []
        self._snapshots = []
        self.snapshot_lock = threading.Lock()
        self.snapshots = self.take_snapshots()
        self._snapshot_daemon = threading.Thread(
            name='device-snapshot-daemon', target=self._snapshot_target, daemon=True
        )
        self._daemon_running = threading.Event()

        self.formats_compact = [
            '│ {physical_index:>3} {fan_speed_string:>3} {temperature_string:>4} '
            '{performance_state:>3} {power_status:>12} '
            '│ {memory_usage:>20} │ {gpu_utilization_string:>7}  {compute_mode:>11} │'
        ]
        self.formats_full = [
            '│ {physical_index:>3}  {name:<18}  {persistence_mode:<4} '
            '│ {bus_id:<16} {display_active:>3} │ {total_volatile_uncorrected_ecc_errors:>20} │',
            '│ {fan_speed_string:>3}  {temperature_string:>4}  {performance_state:>4}  {power_status:>12} '
            '│ {memory_usage:>20} │ {gpu_utilization_string:>7}  {compute_mode:>11} │',
        ]

        self.mig_formats = [
            '│{physical_index:>2}:{mig_index:<2}{name:>12} @ GI/CI:{gpu_instance_id:>2}/{compute_instance_id:<2}'
            '│ {memory_usage:>20} │ BAR1: {bar1_memory_used_human:>8} / {bar1_memory_percent_string:>3} │'
        ]

        if host.WINDOWS:
            self.formats_full[0] = self.formats_full[0].replace(
                'persistence_mode', 'current_driver_model'
            )

        self.support_mig = any('N/A' not in device.mig_mode for device in self.snapshots)
        if self.support_mig:
            self.formats_full[0] = self.formats_full[0].replace(
                '{total_volatile_uncorrected_ecc_errors:>20}',
                '{mig_mode:>8}  {total_volatile_uncorrected_ecc_errors:>10}',
            )

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
            self.height = self.compact_height if self.compact else self.full_height

    @property
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots):
        with self.snapshot_lock:
            self._snapshots = snapshots

    @ttl_cache(ttl=1.0)
    def take_snapshots(self):
        snapshots = [device.as_snapshot() for device in self.all_devices]

        for device in snapshots:
            if device.name.startswith('NVIDIA '):
                device.name = device.name.replace('NVIDIA ', '', 1)
            if device.is_mig_device:
                device.name = device.name.rpartition(' ')[-1]
                if device.bar1_memory_percent is not NA:
                    device.bar1_memory_percent = round(device.bar1_memory_percent)
                    if device.bar1_memory_percent >= 100:
                        device.bar1_memory_percent_string = 'MAX'
                    else:
                        device.bar1_memory_percent_string = '{}%'.format(
                            round(device.bar1_memory_percent)
                        )
            else:
                device.name = cut_string(device.name, maxlen=18, padstr='..', align='right')
                device.current_driver_model = device.current_driver_model.replace('WDM', 'TCC')
                device.display_active = device.display_active.replace('Enabled', 'On').replace(
                    'Disabled', 'Off'
                )
                device.persistence_mode = device.persistence_mode.replace('Enabled', 'On').replace(
                    'Disabled', 'Off'
                )
                device.mig_mode = device.mig_mode.replace('N/A', 'N/A ')
                device.compute_mode = device.compute_mode.replace('Exclusive', 'E.')
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
            '│ NVITOP {:<9} Driver Version: {:<12} CUDA Driver Version: {:<8} │'.format(
                __version__.partition('+')[0], self.driver_version, self.cuda_driver_version
            ),
        ]
        if self.device_count > 0:
            header.append(
                '├───────────────────────────────┬──────────────────────┬──────────────────────┤'
            )
            if compact:
                header.append(
                    '│ GPU Fan Temp Perf Pwr:Usg/Cap │         Memory-Usage │ GPU-Util  Compute M. │'
                )
            else:
                header.extend(
                    (
                        '│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │',
                        '│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │',
                    )
                )
                if host.WINDOWS:
                    header[-2] = header[-2].replace('Persistence-M', '    TCC/WDDM ')
                if self.support_mig:
                    header[-2] = header[-2].replace('Volatile Uncorr. ECC', 'MIG M.   Uncorr. ECC')
            header.append(
                '╞═══════════════════════════════╪══════════════════════╪══════════════════════╡'
            )
        else:
            header.extend(
                (
                    '╞═════════════════════════════════════════════════════════════════════════════╡',
                    '│  No visible devices found                                                   │',
                    '╘═════════════════════════════════════════════════════════════════════════════╛',
                )
            )
        return header

    def frame_lines(self, compact=None):
        if compact is None:
            compact = self.compact

        frame = self.header_lines(compact)

        remaining_width = self.width - 79
        data_line = (
            '│                               │                      │                      │'
        )
        separator_line = (
            '├───────────────────────────────┼──────────────────────┼──────────────────────┤'
        )
        if self.width >= 100:
            data_line += ' ' * (remaining_width - 1) + '│'
            separator_line = separator_line[:-1] + '┼' + '─' * (remaining_width - 1) + '┤'

        if self.device_count > 0:
            for mig_device_count in self.mig_device_counts:
                if compact:
                    frame.extend((data_line, separator_line))
                else:
                    frame.extend((data_line, data_line, separator_line))
                if mig_device_count > 0:
                    frame.extend((data_line,) * mig_device_count + (separator_line,))

            frame.pop()
            frame.append(
                '╘═══════════════════════════════╧══════════════════════╧══════════════════════╛'
            )
            if self.width >= 100:
                frame[5 - int(compact)] = (
                    frame[5 - int(compact)][:-1] + '╪' + '═' * (remaining_width - 1) + '╕'
                )
                frame[-1] = frame[-1][:-1] + '╧' + '═' * (remaining_width - 1) + '╛'

        return frame

    def poke(self):
        if not self._daemon_running.is_set():
            self._daemon_running.set()
            self._snapshot_daemon.start()

        self.snapshots = self._snapshot_buffer

        super().poke()

    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def draw(self):
        self.color_reset()

        if self.need_redraw:
            self.addstr(self.y, self.x, '(Press h for help or q to quit)'.rjust(79))
            self.color_at(self.y, self.x + 55, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y, self.x + 69, width=1, fg='magenta', attr='bold | italic')
            for y, line in enumerate(self.frame_lines(), start=self.y + 1):
                self.addstr(y, self.x, line)

        self.addstr(self.y, self.x, cut_string(time.strftime('%a %b %d %H:%M:%S %Y'), maxlen=32))

        if self.compact:
            formats = self.formats_compact
        else:
            formats = self.formats_full

        remaining_width = self.width - 79
        draw_bars = self.width >= 100
        try:
            selected_device = self.parent.selected.process.device
        except AttributeError:
            selected_device = None

        y_start = self.y + len(formats) + 5
        prev_device_index = self.snapshots[0].tuple_index
        for index, device in enumerate(self.snapshots):  # pylint: disable=too-many-nested-blocks
            if (
                len(prev_device_index) != len(device.tuple_index)
                or prev_device_index[0] != device.tuple_index[0]
            ):
                y_start += 1

            attr = 0
            if selected_device is not None:
                if device.real == selected_device:
                    attr = 'bold'
                elif (
                    device.is_mig_device or device.physical_index != selected_device.physical_index
                ):
                    attr = 'dim'

            fmts = self.mig_formats if device.is_mig_device else formats
            for y, fmt in enumerate(fmts, start=y_start):
                self.addstr(y, self.x, fmt.format(**device.__dict__))
                self.color_at(y, self.x + 1, width=31, fg=device.display_color, attr=attr)
                self.color_at(y, self.x + 33, width=22, fg=device.display_color, attr=attr)
                self.color_at(y, self.x + 56, width=22, fg=device.display_color, attr=attr)

            if draw_bars:
                matrix = [
                    (
                        self.x + 80,
                        y_start,
                        remaining_width - 3,
                        'MEM',
                        device.memory_percent,
                        device.memory_display_color,
                    ),
                    (
                        self.x + 80,
                        y_start + 1,
                        remaining_width - 3,
                        'UTL',
                        device.gpu_utilization,
                        device.gpu_display_color,
                    ),
                ]
                if self.compact:
                    if remaining_width >= 44 and not device.is_mig_device:
                        left_width = (remaining_width - 6 + 1) // 2 - 1
                        right_width = (remaining_width - 6) // 2 + 1
                        matrix = [
                            (
                                self.x + 80,
                                y_start,
                                left_width,
                                'MEM',
                                device.memory_percent,
                                device.memory_display_color,
                            ),
                            (
                                self.x + 80 + left_width + 3,
                                y_start,
                                right_width,
                                'UTL',
                                device.gpu_utilization,
                                device.gpu_display_color,
                            ),
                        ]
                        separator = '┼' if index > 0 else '╤'
                        if len(prev_device_index) == 2:
                            separator = '┬'
                        self.addstr(y_start - 1, self.x + 80 + left_width + 1, separator)
                        self.addstr(y_start, self.x + 80 + left_width + 1, '│')
                        if index == len(self.snapshots) - 1:
                            self.addstr(y_start + 1, self.x + 80 + left_width + 1, '╧')
                    else:
                        if remaining_width >= 44 and len(prev_device_index) == 1:
                            self.addstr(y_start - 1, self.x + 80 + left_width + 1, '┴')
                        matrix.pop()
                elif device.is_mig_device:
                    matrix.pop()
                for x_offset, y, width, prefix, utilization, color in matrix:
                    # pylint: disable-next=disallowed-name
                    bar = make_bar(prefix, utilization, width=width)
                    self.addstr(y, x_offset, bar)
                    if self.TERM_256COLOR:
                        parts = bar.rstrip().split(' ')
                        prefix_len = len(parts[0])
                        bar_len = len(parts[1])
                        full_bar_len = width - prefix_len - 5
                        self.color_at(y, x_offset, width=width, fg=float(bar_len / full_bar_len))
                        for i, x in enumerate(
                            range(x_offset + prefix_len + 1, x_offset + prefix_len + 1 + bar_len)
                        ):
                            self.color_at(y, x, width=1, fg=float(i / full_bar_len))
                    else:
                        self.color_at(y, x_offset, width=width, fg=color, attr=attr)

            y_start += len(fmts)
            prev_device_index = device.tuple_index

    def destroy(self):
        super().destroy()
        self._daemon_running.clear()

    def print_width(self):
        if self.device_count > 0 and self.width >= 100:
            return self.width
        return 79

    def print(self):  # pylint: disable=too-many-locals,too-many-branches
        lines = [time.strftime('%a %b %d %H:%M:%S %Y'), *self.header_lines(compact=False)]

        if self.device_count > 0:
            prev_device_index = self.snapshots[0].tuple_index
            for device in self.snapshots:
                if (
                    len(prev_device_index) != len(device.tuple_index)
                    or prev_device_index[0] != device.tuple_index[0]
                ):
                    lines.append(
                        '├───────────────────────────────┼──────────────────────┼──────────────────────┤'
                    )

                def colorize(s):
                    if len(s) > 0:
                        # pylint: disable-next=cell-var-from-loop
                        return colored(s, device.display_color)
                    return ''

                fmts = self.mig_formats if device.is_mig_device else self.formats_full
                for fmt in fmts:
                    line = fmt.format(**device.__dict__)
                    lines.append('│'.join(map(colorize, line.split('│'))))

                prev_device_index = device.tuple_index

            lines.append(
                '╘═══════════════════════════════╧══════════════════════╧══════════════════════╛'
            )

            if self.width >= 100:
                remaining_width = self.width - 79
                y_start = 7
                prev_device_index = self.snapshots[0].tuple_index
                for index, device in enumerate(self.snapshots):
                    if (
                        len(prev_device_index) != len(device.tuple_index)
                        or prev_device_index[0] != device.tuple_index[0]
                    ):
                        lines[y_start] = (
                            lines[y_start][:-1] + '┼' + '─' * (remaining_width - 1) + '┤'
                        )
                        y_start += 1
                    elif index == 0:
                        lines[y_start - 1] = (
                            lines[y_start - 1][:-1] + '╪' + '═' * (remaining_width - 1) + '╕'
                        )

                    matrix = [
                        (
                            'MEM',
                            device.memory_percent,
                            Device.INTENSITY2COLOR[device.memory_loading_intensity],
                        ),
                        (
                            'UTL',
                            device.gpu_utilization,
                            Device.INTENSITY2COLOR[device.gpu_loading_intensity],
                        ),
                    ]
                    if device.is_mig_device:
                        matrix.pop()
                    for y, (prefix, utilization, color) in enumerate(matrix, start=y_start):
                        bar = make_bar(  # pylint: disable=disallowed-name
                            prefix, utilization, width=remaining_width - 3
                        )
                        lines[y] += ' {} │'.format(colored(bar, color))

                    if index == len(self.snapshots) - 1:
                        lines[y_start + len(matrix)] = (
                            lines[y_start + len(matrix)][:-1]
                            + '╧'
                            + '═' * (remaining_width - 1)
                            + '╛'
                        )

                    y_start += len(matrix)
                    prev_device_index = device.tuple_index

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
