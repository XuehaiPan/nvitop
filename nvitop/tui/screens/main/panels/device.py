# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, ClassVar

from nvitop.tui.library import (
    IS_WINDOWS,
    LAYOUT_WIDTH,
    MIN_WIDTH,
    NA,
    Device,
    MigDevice,
    Snapshot,
    colored,
    cut_string,
    make_bar_chart,
    ttl_cache,
)
from nvitop.tui.screens.main.panels.base import BasePanel
from nvitop.version import __version__


if TYPE_CHECKING:
    import curses

    from nvitop.tui.tui import TUI


__all__ = ['DevicePanel']


class DevicePanel(BasePanel):  # pylint: disable=too-many-instance-attributes
    NAME: ClassVar[str] = 'device'
    SNAPSHOT_INTERVAL: ClassVar[float] = 0.5

    def __init__(
        self,
        devices: list[Device],
        compact: bool,
        win: curses.window | None,
        root: TUI,
    ) -> None:
        super().__init__(win, root)

        self.devices: list[Device] = devices
        self.device_count: int = len(self.devices)

        self.all_devices: list[Device | MigDevice] = []
        self.leaf_devices: list[Device | MigDevice] = []
        self.mig_device_counts: list[int] = [0] * self.device_count
        self.mig_enabled_device_count: int = 0
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

        self.mig_device_count: int = sum(self.mig_device_counts)
        self.all_device_count: int = len(self.all_devices)
        self.leaf_device_count: int = len(self.leaf_devices)

        self._compact: bool = compact
        self._compact_height: int = (
            4 + 2 * (self.device_count + 1) + self.mig_device_count + self.mig_enabled_device_count
        )
        self._full_height: int = self._compact_height + self.device_count + 1
        # The narrow layout has one line per device (MIG devices included) and no separators
        self.narrow_height: int = 7 + self.all_device_count
        if self.device_count == 0:
            self._full_height = self._compact_height = self.narrow_height = 6
        self._narrow_layout: tuple[int, bool, int, int, int] | None = None
        self.width: int = max(MIN_WIDTH, root.width)
        self.height: int = self.compact_height if compact else self.full_height

        self.driver_version: str = Device.driver_version()
        self.cuda_driver_version: str = Device.cuda_driver_version()

        self._snapshot_buffer: list[Snapshot] = []
        self._snapshots: list[Snapshot] = []
        self.snapshot_lock = threading.Lock()
        self.snapshots: list[Snapshot] = self.take_snapshots()
        self._snapshot_daemon = threading.Thread(
            name='device-snapshot-daemon',
            target=self._snapshot_target,
            daemon=True,
        )
        self._daemon_running = threading.Event()

        self.formats_compact: list[str] = [
            (
                '│ {physical_index:>3} {fan_speed_string:>3} {temperature_string:>4} '
                '{performance_state:<3}{power_status:>13} '
                '│ {memory_usage:>20} │ {gpu_utilization_string:>7}  {compute_mode:>11} │'
            ),
        ]
        self.formats_full: list[str] = [
            (
                '│ {physical_index:>3}  {name:<19} {persistence_mode:>4} '
                '│ {bus_id:<16} {display_active:>3} │ {total_volatile_uncorrected_ecc_errors:>20} │'
            ),
            (
                '│ {fan_speed_string:>3}  {temperature_string:>4}  {performance_state:^4} {power_status:>13} '
                '│ {memory_usage:>20} │ {gpu_utilization_string:>7}  {compute_mode:>11} │'
            ),
        ]

        self.mig_formats: list[str] = [
            (
                '│{physical_index:>2}:{mig_index:<2}{name:>12} @ GI/CI:{gpu_instance_id:>2}/{compute_instance_id:<2}'
                '│ {memory_usage:>20} │ BAR1: {bar1_memory_used_human:>8} / {bar1_memory_percent_string:>3} │'
            ),
        ]

        if IS_WINDOWS:
            self.formats_full[0] = self.formats_full[0].replace(
                'persistence_mode',
                'current_driver_model',
            )

        self.support_mig: bool = any('N/A' not in device.mig_mode for device in self.snapshots)
        if self.support_mig:
            self.formats_full[0] = self.formats_full[0].replace(
                '{total_volatile_uncorrected_ecc_errors:>20}',
                '{mig_mode:>8}  {total_volatile_uncorrected_ecc_errors:>10}',
            )
        if all(len(device.power_status.rpartition(' / ')[-1]) < 5 for device in self.snapshots):
            self.formats_compact[0] = self.formats_compact[0].replace(
                'performance_state:<3',
                'performance_state:>3',
            )

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int) -> None:
        width = max(MIN_WIDTH, value)
        if self._width != width and self.visible:
            self.need_redraw = True
        self._width = width
        self.height = self.compact_height if self.compact else self.full_height

    @property
    def narrow(self) -> bool:
        """Whether the terminal is too narrow for the standard layout (e.g. a phone)."""
        return self.width < LAYOUT_WIDTH

    @property
    def compact_height(self) -> int:
        return self.narrow_height if self.narrow else self._compact_height

    @property
    def full_height(self) -> int:
        return self.narrow_height if self.narrow else self._full_height

    @property
    def compact(self) -> bool:
        return self._compact

    @compact.setter
    def compact(self, value: bool) -> None:
        if self._compact != value:
            self.need_redraw = True
            self._compact = value
            self.height = self.compact_height if self.compact else self.full_height

    @property
    def snapshots(self) -> list[Snapshot]:
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots: list[Snapshot]) -> None:
        with self.snapshot_lock:
            self._snapshots = snapshots

    @classmethod
    def set_snapshot_interval(cls, interval: float) -> None:
        assert interval > 0.0
        interval = float(interval)

        cls.SNAPSHOT_INTERVAL = min(interval / 3.0, 1.0)
        cls.take_snapshots = ttl_cache(ttl=interval)(  # type: ignore[method-assign]
            cls.take_snapshots.__wrapped__,  # type: ignore[attr-defined] # pylint: disable=no-member
        )

    @ttl_cache(ttl=1.0)
    def take_snapshots(self) -> list[Snapshot]:
        snapshots = [device.as_snapshot() for device in self.all_devices]

        for device in snapshots:
            if device.name.startswith('NVIDIA '):
                device.name = device.name.replace('NVIDIA ', '', 1)
            if device.is_mig_device:
                device.name = device.name.rpartition(' ')[-1]
                if device.bar1_memory_percent is not NA:
                    device.bar1_memory_percent = round(device.bar1_memory_percent)
                    device.bar1_memory_percent_string = (
                        'MAX'
                        if device.bar1_memory_percent >= 100
                        else f'{round(device.bar1_memory_percent)}%'
                    )
            else:
                device.name = cut_string(device.name, maxlen=19, padstr='..', align='right')
                device.current_driver_model = device.current_driver_model.replace('WDM', 'TCC')
                device.display_active = device.display_active.replace('Enabled', 'On').replace(
                    'Disabled',
                    'Off',
                )
                device.persistence_mode = device.persistence_mode.replace('Enabled', 'On').replace(
                    'Disabled',
                    'Off',
                )
                device.mig_mode = device.mig_mode.replace('N/A', 'N/A ')
                device.compute_mode = device.compute_mode.replace('Exclusive', 'E.')
                if device.fan_speed is not NA and device.fan_speed >= 100:
                    device.fan_speed_string = 'MAX'

        with self.snapshot_lock:
            self._snapshot_buffer = snapshots

        return snapshots

    def _snapshot_target(self) -> None:
        self._daemon_running.wait()
        while self._daemon_running.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def header_lines(self, compact: bool | None = None) -> list[str]:
        if compact is None:
            compact = self.compact

        version_infos = [
            'NVITOP {}'.format(__version__.partition('+')[0]),
            f'Driver Version: {self.driver_version}',
            f'CUDA Driver Version: {self.cuda_driver_version}',
        ]
        if sum(len(v) for v in version_infos) % 2 == 0:
            version_infos[0] += ' '
        version_seps = ' ' * max(2, (75 - sum(len(v) for v in version_infos)) // 2)

        header = [
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ {} │'.format(version_seps.join(version_infos).ljust(75, ' ')),
        ]
        if self.device_count > 0:
            header.append(
                '├───────────────────────────────┬──────────────────────┬──────────────────────┤',
            )
            if compact:
                header.append(
                    '│ GPU Fan Temp Perf Pwr:Usg/Cap │         Memory-Usage │ GPU-Util  Compute M. │',
                )
            else:
                header.extend(
                    (
                        '│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │',
                        '│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │',
                    ),
                )
                if IS_WINDOWS:
                    header[-2] = header[-2].replace('Persistence-M', '    TCC/WDDM ')
                if self.support_mig:
                    header[-2] = header[-2].replace('Volatile Uncorr. ECC', 'MIG M.   Uncorr. ECC')
            header.append(
                '╞═══════════════════════════════╪══════════════════════╪══════════════════════╡',
            )
        else:
            header.extend(
                (
                    '╞═════════════════════════════════════════════════════════════════════════════╡',
                    '│  No visible devices found                                                   │',
                    '╘═════════════════════════════════════════════════════════════════════════════╛',
                ),
            )
        return header

    def frame_lines(self, compact: bool | None = None) -> list[str]:
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
                '╘═══════════════════════════════╧══════════════════════╧══════════════════════╛',
            )
            if self.width >= 100:
                frame[5 - int(compact)] = (
                    frame[5 - int(compact)][:-1] + '╪' + '═' * (remaining_width - 1) + '╕'
                )
                frame[-1] = frame[-1][:-1] + '╧' + '═' * (remaining_width - 1) + '╛'

        return frame

    def poke(self) -> None:
        if not self._daemon_running.is_set():
            self._daemon_running.set()
            self._snapshot_daemon.start()

        self.snapshots = self._snapshot_buffer

        super().poke()

    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def draw(self) -> None:
        self.color_reset()

        if self.narrow:
            self.draw_narrow()
            return

        if self.need_redraw:
            self.addstr(self.y, self.x, '(Press h for help or q to quit)'.rjust(79))
            self.color_at(self.y, self.x + 55, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y, self.x + 69, width=1, fg='magenta', attr='bold | italic')
            for y, line in enumerate(self.frame_lines(), start=self.y + 1):
                self.addstr(y, self.x, line)

        self.addstr(self.y, self.x, cut_string(time.strftime('%a %b %d %H:%M:%S %Y'), maxlen=32))

        formats = self.formats_compact if self.compact else self.formats_full

        remaining_width = self.width - 79
        draw_bars = self.width >= 100
        try:
            selected_device = self.parent.selection.process.device  # type: ignore[union-attr]
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

            attr: int | str = 0
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
                left_width = (remaining_width - 6 + 1) // 2 - 1
                right_width = (remaining_width - 6) // 2 + 1
                matrix: list[list[tuple[int, int, str, float, str, str]]] = []
                if device.is_mig_device:
                    matrix = [
                        [
                            (
                                self.x + 80,
                                remaining_width - 3,
                                'MEM',
                                device.memory_percent,
                                device.memory_display_color,
                                device.memory_used_human,
                            ),
                        ],
                    ]
                    if remaining_width >= 44 and len(prev_device_index) == 1:
                        self.addstr(y_start - 1, self.x + 80 + left_width + 1, '┴')
                elif self.compact:
                    if remaining_width >= 44:
                        matrix = [
                            [
                                (
                                    self.x + 80,
                                    left_width,
                                    'MEM',
                                    device.memory_percent,
                                    device.memory_display_color,
                                    device.memory_used_human,
                                ),
                                (
                                    self.x + 80 + left_width + 3,
                                    right_width,
                                    'UTL',
                                    device.gpu_utilization,
                                    device.gpu_display_color,
                                    f'@ {device.clock_infos.sm}MHz',
                                ),
                            ],
                        ]
                        separator = '┼' if index > 0 else '╤'
                        if len(prev_device_index) == 2:
                            separator = '┬'
                        self.addstr(y_start - 1, self.x + 80 + left_width + 1, separator)
                        if index == len(self.snapshots) - 1:
                            self.addstr(y_start + 1, self.x + 80 + left_width + 1, '╧')
                    else:
                        matrix = [
                            [
                                (
                                    self.x + 80,
                                    remaining_width - 3,
                                    'MEM',
                                    device.memory_percent,
                                    device.memory_display_color,
                                    device.memory_used_human,
                                ),
                            ],
                        ]
                else:
                    if remaining_width >= 44:
                        matrix = [
                            [
                                (
                                    self.x + 80,
                                    left_width,
                                    'MEM',
                                    device.memory_percent,
                                    device.memory_display_color,
                                    device.memory_used_human,
                                ),
                                (
                                    self.x + 80 + left_width + 3,
                                    right_width,
                                    'MBW',
                                    device.memory_utilization,
                                    device.bandwidth_display_color,
                                    f'@ {device.clock_infos.memory}MHz',
                                ),
                            ],
                            [
                                (
                                    self.x + 80,
                                    left_width,
                                    'UTL',
                                    device.gpu_utilization,
                                    device.gpu_display_color,
                                    f'@ {device.clock_infos.sm}MHz',
                                ),
                                (
                                    self.x + 80 + left_width + 3,
                                    right_width,
                                    'PWR',
                                    device.power_utilization,
                                    device.power_display_color,
                                    f'{device.power_status.partition(" / ")[0]}',
                                ),
                            ],
                        ]
                        separator = '┼' if index > 0 else '╤'
                        if len(prev_device_index) == 2:
                            separator = '┬'
                        self.addstr(y_start - 1, self.x + 80 + left_width + 1, separator)
                        if index == len(self.snapshots) - 1:
                            self.addstr(y_start + 2, self.x + 80 + left_width + 1, '╧')
                    else:
                        matrix = [
                            [
                                (
                                    self.x + 80,
                                    remaining_width - 3,
                                    'MEM',
                                    device.memory_percent,
                                    device.memory_display_color,
                                    device.memory_used_human,
                                ),
                            ],
                            [
                                (
                                    self.x + 80,
                                    remaining_width - 3,
                                    'UTL',
                                    device.gpu_utilization,
                                    device.gpu_display_color,
                                    f'@ {device.clock_infos.sm}MHz',
                                ),
                            ],
                        ]

                for y, row in enumerate(matrix, start=y_start):
                    for x_offset, width, prefix, utilization, color, extra_text in row:
                        self.draw_bar_chart(
                            y,
                            x_offset,
                            width,
                            prefix=prefix,
                            utilization=utilization,
                            color=color,
                            extra_text=extra_text,
                            attr=attr,
                        )

            y_start += len(fmts)
            prev_device_index = device.tuple_index

    # pylint: disable-next=too-many-arguments,too-many-locals
    def draw_bar_chart(
        self,
        y: int,
        x: int,
        width: int,
        *,
        prefix: str,
        utilization: float | str,
        color: str,
        extra_text: str,
        attr: int | str,
    ) -> None:
        bar_chart = make_bar_chart(
            prefix,
            utilization,
            width=width,
            extra_text=extra_text if 'N/A' not in extra_text else '',
            swap_text=not extra_text.endswith('MHz'),
            extra_blank=' ',
        )
        self.addstr(y, x, f'{bar_chart} │')
        if self.TERM_256COLOR:
            parts = bar_chart.rstrip().split(' ')
            prefix_len = len(parts[0])
            bar_len = len(parts[1])
            full_bar_len = width - prefix_len - 5
            self.color_at(y, x, width=width, fg=float(bar_len / full_bar_len))
            for i, x_bar in enumerate(range(x + prefix_len + 1, x + prefix_len + 1 + bar_len)):
                self.color_at(y, x_bar, width=1, fg=float(i / full_bar_len))
        else:
            self.color_at(y, x, width=width, fg=color, attr=attr)

    def narrow_layout(self) -> tuple[int, bool, int, int, int]:
        """Return the column widths of the narrow layout.

        The narrow layout fits terminals narrower than the standard one (e.g. a phone) with
        one line per device: index, fan, temperature and power, then memory and utilization
        bars sharing the remaining width. The fan column is left out when no device has a fan
        (e.g. passively cooled data center boards).
        """
        index_width = 5 if self.mig_device_count > 0 else 3
        has_fan = any(
            device.fan_speed is not NA for device in self.snapshots if not device.is_mig_device
        )
        info_width = index_width + 13 + (4 if has_fan else 0)
        bars_width = self.width - info_width - 8
        memory_width = (bars_width + 1) // 2
        return index_width, has_fan, info_width, memory_width, bars_width - memory_width

    def narrow_frame_lines(self) -> list[str]:
        index_width, has_fan, info_width, memory_width, utilization_width = self.narrow_layout()

        version_info = '  '.join(
            (
                'NVITOP {}'.format(__version__.partition('+')[0]),
                f'Driver: {self.driver_version}',
                f'CUDA: {self.cuda_driver_version}',
            ),
        )
        frame = [
            '╒' + '═' * (self.width - 2) + '╕',
            f'│ {cut_string(version_info, maxlen=self.width - 4).ljust(self.width - 4)} │',
        ]
        if self.device_count == 0:
            frame.extend(
                (
                    '╞' + '═' * (self.width - 2) + '╡',
                    '│ {} │'.format('No visible devices found'.ljust(self.width - 4)),
                    '╘' + '═' * (self.width - 2) + '╛',
                ),
            )
            return frame

        def row(left: str, middle: str, right: str, fill: str) -> str:
            return (
                left
                + fill * info_width
                + middle
                + fill * (memory_width + 2)
                + middle
                + fill * (utilization_width + 2)
                + right
            )

        titles = ' {:>{}} {}{:>4} {:>5} '.format(
            'GPU',
            index_width,
            'Fan ' if has_fan else '',
            'Temp',
            'Pwr',
        )
        frame.extend(
            (
                row('├', '┬', '┤', '─'),
                '│{}│{}│{}│'.format(
                    titles,
                    (' Memory-Usage' if memory_width >= 13 else ' Memory').ljust(memory_width + 2),
                    ' GPU-Util'.ljust(utilization_width + 2),
                ),
                row('╞', '╪', '╡', '═'),
            ),
        )
        frame.extend(row('│', '│', '│', ' ') for _ in range(self.all_device_count))
        frame.append(row('╘', '╧', '╛', '═'))
        return frame

    # pylint: disable-next=too-many-locals
    def draw_narrow(self) -> None:
        layout = self.narrow_layout()
        index_width, has_fan, info_width, memory_width, utilization_width = layout
        if layout != self._narrow_layout:  # e.g. the fan readings come and go
            self._narrow_layout = layout
            self.need_redraw = True

        if self.need_redraw:
            # The hint and the column of each highlighted key
            hint, key_offsets = (
                ('(Press h for help or q to quit)', (7, 21))
                if self.width >= 57
                else ('(h: help, q: quit)', (1, 10))
            )
            hint_x = self.x + self.width - len(hint)
            self.addstr(self.y, hint_x, hint)
            for offset in key_offsets:
                self.color_at(self.y, hint_x + offset, width=1, fg='magenta', attr='bold | italic')
            for y, line in enumerate(self.narrow_frame_lines(), start=self.y + 1):
                self.addstr(y, self.x, line)

        self.addstr(self.y, self.x, cut_string(time.strftime('%a %b %d %H:%M:%S %Y'), maxlen=24))
        if self.device_count == 0:
            return

        try:
            selected_device = self.parent.selection.process.device  # type: ignore[union-attr]
        except AttributeError:
            selected_device = None

        memory_x = self.x + info_width + 3
        utilization_x = memory_x + memory_width + 3
        for y, device in enumerate(self.snapshots, start=self.y + 6):
            attr: int | str = 0
            if selected_device is not None:
                if device.real == selected_device:
                    attr = 'bold'
                elif (
                    device.is_mig_device or device.physical_index != selected_device.physical_index
                ):
                    attr = 'dim'

            if device.is_mig_device:
                index = f'{device.physical_index}:{device.mig_index}'
                info = cut_string(device.name, maxlen=info_width - index_width - 3)
                info = ' {:>{}} {:>{}} '.format(
                    index,
                    index_width,
                    info,
                    info_width - index_width - 3,
                )
            else:
                power = device.power_status.partition(' / ')[0].replace(' ', '')
                info = ' {:>{}} {}{:>4} {:>5} '.format(
                    device.physical_index,
                    index_width,
                    f'{device.fan_speed_string:>3} ' if has_fan else '',
                    cut_string(device.temperature_string, maxlen=4),
                    cut_string(power, maxlen=5),
                )
            self.addstr(y, self.x + 1, info)
            self.color_at(y, self.x + 1, width=info_width, fg=device.display_color, attr=attr)

            self.draw_bar_chart(
                y,
                memory_x,
                memory_width,
                prefix='MEM',
                utilization=device.memory_percent,
                color=device.memory_display_color,
                extra_text=device.memory_used_human,
                attr=attr,
            )
            self.draw_bar_chart(
                y,
                utilization_x,
                utilization_width,
                prefix='UTL',
                utilization=device.gpu_utilization,
                color=device.gpu_display_color,
                extra_text='' if device.is_mig_device else f'@ {device.clock_infos.sm}MHz',
                attr=attr,
            )

    def destroy(self) -> None:
        super().destroy()
        self._daemon_running.clear()

    def print_width(self) -> int:
        if self.device_count > 0 and self.width >= 100:
            return self.width
        return 79

    def print(self) -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        lines = [time.strftime('%a %b %d %H:%M:%S %Y'), *self.header_lines(compact=False)]

        if self.device_count > 0:
            prev_device_index = self.snapshots[0].tuple_index
            for device in self.snapshots:
                if (
                    len(prev_device_index) != len(device.tuple_index)
                    or prev_device_index[0] != device.tuple_index[0]
                ):
                    lines.append(
                        '├───────────────────────────────┼──────────────────────┼──────────────────────┤',
                    )

                def colorize(s: str) -> str:
                    if s:
                        # pylint: disable-next=cell-var-from-loop
                        return colored(s, device.display_color)  # noqa: B023
                    return ''

                fmts = self.mig_formats if device.is_mig_device else self.formats_full
                for fmt in fmts:
                    line = fmt.format(**device.__dict__)
                    lines.append('│'.join(map(colorize, line.split('│'))))

                prev_device_index = device.tuple_index

            lines.append(
                '╘═══════════════════════════════╧══════════════════════╧══════════════════════╛',
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

                    left_width = (remaining_width - 6 + 1) // 2 - 1
                    right_width = (remaining_width - 6) // 2 + 1
                    matrix: list[list[tuple[str, float, str, int, str]]] = []
                    if device.is_mig_device:
                        matrix = [
                            [
                                (
                                    'MEM',
                                    device.memory_percent,
                                    device.memory_display_color,
                                    remaining_width - 3,
                                    device.memory_used_human,
                                ),
                            ],
                        ]
                        if remaining_width >= 44 and len(prev_device_index) == 1:
                            lines[y_start - 1] = (
                                lines[y_start - 1][: -right_width - 4]
                                + '┴'
                                + lines[y_start - 1][-right_width - 3 :]
                            )
                    else:
                        if remaining_width >= 44:
                            matrix = [
                                [
                                    (
                                        'MEM',
                                        device.memory_percent,
                                        device.memory_display_color,
                                        left_width,
                                        device.memory_used_human,
                                    ),
                                    (
                                        'MBW',
                                        device.memory_utilization,
                                        device.bandwidth_display_color,
                                        right_width,
                                        f'@ {device.clock_infos.memory}MHz',
                                    ),
                                ],
                                [
                                    (
                                        'UTL',
                                        device.gpu_utilization,
                                        device.gpu_display_color,
                                        left_width,
                                        f'@ {device.clock_infos.sm}MHz',
                                    ),
                                    (
                                        'PWR',
                                        device.power_utilization,
                                        device.power_display_color,
                                        right_width,
                                        f'{device.power_status.partition(" / ")[0]}',
                                    ),
                                ],
                            ]
                            separator = '┼' if index > 0 else '╤'
                            if len(prev_device_index) == 2:
                                separator = '┬'
                            lines[y_start - 1] = (
                                lines[y_start - 1][: -right_width - 4]
                                + separator
                                + lines[y_start - 1][-right_width - 3 :]
                            )
                        else:
                            matrix = [
                                [
                                    (
                                        'MEM',
                                        device.memory_percent,
                                        device.memory_display_color,
                                        remaining_width - 3,
                                        device.memory_used_human,
                                    ),
                                ],
                                [
                                    (
                                        'UTL',
                                        device.gpu_utilization,
                                        device.gpu_display_color,
                                        remaining_width - 3,
                                        f'@ {device.clock_infos.sm}MHz',
                                    ),
                                ],
                            ]
                    for y, row in enumerate(matrix, start=y_start):
                        for prefix, utilization, color, width, extra_text in row:
                            bar_chart = make_bar_chart(
                                prefix,
                                utilization,
                                width=width,
                                extra_text=extra_text if 'N/A' not in extra_text else '',
                                swap_text=not extra_text.endswith('MHz'),
                                extra_blank=' ',
                            )
                            lines[y] += f' {colored(bar_chart, color)} │'  # type: ignore[arg-type]

                    if index == len(self.snapshots) - 1:
                        lines[y_start + len(matrix)] = (
                            lines[y_start + len(matrix)][:-1]
                            + '╧'
                            + '═' * (remaining_width - 1)
                            + '╛'
                        )
                        if remaining_width >= 44 and len(matrix[0]) > 1:
                            lines[y_start + len(matrix)] = (
                                lines[y_start + len(matrix)][: -right_width - 4]
                                + '╧'
                                + lines[y_start + len(matrix)][-right_width - 3 :]
                            )

                    y_start += len(matrix)
                    prev_device_index = device.tuple_index

        content = '\n'.join(lines)
        if self.no_unicode:
            content = content.translate(self.ASCII_TRANSTABLE)

        try:
            print(content)
        except UnicodeError:
            print(content.translate(self.ASCII_TRANSTABLE))

    def press(self, key: int) -> bool:
        self.root.keymaps.use_keymap('device')
        return self.root.press(key)
