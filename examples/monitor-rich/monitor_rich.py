# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2026 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A responsive, read-only GPU dashboard using ``nvitop`` and Rich.

Run without cloning the repository using ``uv``::

    uv run https://github.com/XuehaiPan/nvitop/raw/HEAD/examples/monitor-rich/monitor_rich.py
"""

# /// script
# requires-python = ">= 3.8"
# dependencies = ["nvitop", "rich >= 14.0"]
# ///

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

# pylint: disable=import-error
from rich import box
from rich.console import Console, Group
from rich.control import Control
from rich.progress_bar import ProgressBar
from rich.segment import ControlType, Segment, Segments
from rich.table import Table
from rich.text import Text

# pylint: enable=import-error
from nvitop import NA, Device, GpuProcess, bytes2human, host, libnvml, utilization2string


if TYPE_CHECKING:
    from collections.abc import Generator

    from nvitop import Snapshot


MIN_BAR_WIDTH = 5
DEVICE_MIN_COLUMN_WIDTHS = {'TEMP': 4, 'PWR': 4}
DEVICE_MAX_COLUMN_WIDTHS = {'NAME': 18}
PROCESS_MIN_COLUMN_WIDTHS = {'SM%': 4, 'CPU%': 5, 'MEM%': 5}
PROCESS_MAX_COLUMN_WIDTHS = {'USER': 12}


class BreakLoop(Exception):  # noqa: N818
    """Exit the monitor and print its final snapshot."""


@dataclass
class SnapshotBatch:
    """Hold device and process snapshots, including their live process references."""

    devices: list[Snapshot]
    gpu_processes: list[Snapshot]


def collect_snapshot(devices: list[Device]) -> SnapshotBatch:
    """Refresh the cached device instances and collect processes from leaf devices only."""
    snapshots: list[Snapshot] = []
    processes: list[GpuProcess] = []
    for device in devices:
        try:
            snapshot = device.as_snapshot()
            device_processes = device.processes() if device.is_leaf_device() else {}
        except (libnvml.NVMLError_NotFound, libnvml.NVMLError_NotSupported):
            if not device.is_mig_device():
                raise
            continue  # instance removed, or MIG was disabled during collection
        snapshots.append(snapshot)
        processes.extend(device_processes.values())

    return SnapshotBatch(
        devices=snapshots,
        gpu_processes=GpuProcess.take_snapshots(processes, failsafe=True),
    )


def device_key(index: int | tuple[int, int]) -> tuple[int, int]:
    """Order physical devices before their MIG instances using numeric indices."""
    return index if isinstance(index, tuple) else (index, -1)


def device_label(index: int | tuple[int, int]) -> str:
    """Use the same physical or physical:MIG label in both tables."""
    return ':'.join(map(str, index)) if isinstance(index, tuple) else str(index)


def literal(value: object) -> Text:
    """Keep external strings on one row and escape terminal controls as visible text."""
    single_line = ' '.join(str(value).splitlines())
    escaped = []
    for char in single_line:
        code = ord(char)
        escaped.append(rf'\x{code:02x}' if code < 32 or 127 <= code < 160 else char)
    return Text(''.join(escaped), no_wrap=True, overflow='ellipsis')


def device_name(device: Snapshot, *, width: int | None = None) -> Text:
    """Omit the vendor, show MIG profiles, and let Rich truncate names to the available width."""
    name = device.name
    if name.startswith('NVIDIA '):
        name = name[len('NVIDIA ') :]
    if isinstance(device.index, tuple):
        name = name.rpartition(' ')[-1]
    text = literal(name)
    if width is None:
        width = DEVICE_MAX_COLUMN_WIDTHS.get('NAME')
    if width is not None:
        text.truncate(width)
    return text


def metric(value: object, unit: str = '') -> Text:
    """Format a finite metric, leaving unavailable values without a unit suffix."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return Text('N/A', style='dim')
    style = ''
    if unit == '%':
        style = 'green' if value < 50 else 'yellow' if value < 80 else 'red'
        return Text(utilization2string(value), style=style)
    return Text(f'{value:g}{unit}', style=style)


def percentage_bar(value: object, *, label: Text, width: int, label_width: int) -> Table | Text:
    """Show a percentage bar beside a label, using an empty bar for unavailable values."""
    if width < MIN_BAR_WIDTH:
        return label
    style = metric(value, '%').style
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = 0
    cell = Table.grid(padding=(0, 1))
    cell.add_column(width=width)
    cell.add_column(width=label_width, justify='right', no_wrap=True)
    cell.add_row(
        ProgressBar(
            total=100.0,
            completed=value,
            width=width,
            complete_style=style,
            finished_style=style,
        ),
        label,
    )
    return cell


def make_table(
    title: str,
    headers: list[str],
    *,
    min_column_widths: dict[str, int],
    max_column_widths: dict[str, int],
) -> Table:
    """Let Rich allocate space while keeping metric cells on one line."""
    table = Table(
        title=title,
        title_justify='left',
        box=box.SIMPLE_HEAD,
        padding=0,
        show_edge=False,
        pad_edge=False,
        expand=True,
        header_style='bold cyan',
    )
    for header in headers:
        flexible = header == 'COMMAND'
        table.add_column(
            header,
            justify='left' if header in {'NAME', 'USER', 'COMMAND'} else 'right',
            no_wrap=True,
            overflow='ellipsis',
            ratio=1 if flexible else None,
            width=4 if header == 'TEMP' else None,
            min_width=max(min_column_widths.get(header, 0), len(header)),
            max_width=max_column_widths.get(header),
        )
    return table


def measure_column_widths(
    rows: list[dict[str, Text]],
    *,
    headers: list[str],
    min_column_widths: dict[str, int],
    max_column_widths: dict[str, int],
) -> dict[str, int]:
    """Measure labels and headers within the table's configured column limits."""
    widths = {}
    for header in headers:
        width = max(
            len(header),
            min_column_widths.get(header, 0),
            *(row[header].cell_len for row in rows),
        )
        widths[header] = min(width, max_column_widths.get(header, width))
    return widths


def device_power(device: Snapshot, *, show_limit: bool = True) -> Text:
    """Color board power usage and optionally append its dimmed limit."""
    is_mig = isinstance(device.index, tuple)
    power = metric(NA if is_mig else round(float(device.power_usage) / 1000.0, 0), 'W')
    power.stylize(metric(power_percent(device), '%').style)
    if show_limit and not is_mig:
        power.append('/', style='dim')
        power.append(metric(round(float(device.power_limit) / 1000.0, 0), 'W').plain, style='dim')
    return power


def device_row(device: Snapshot) -> dict[str, Text]:
    """Format device cells before allocating column widths, with board metrics on physical rows."""
    is_mig = isinstance(device.index, tuple)
    memory = literal(device.memory_used_human)
    memory_percent = metric(device.memory_percent, '%')
    memory.stylize(memory_percent.style)
    memory.append('/', style='dim')
    memory_total = literal(device.memory_total_human)
    memory_total.stylize('dim')
    memory.append_text(memory_total)
    return {
        'GPU': literal(device_label(device.index)),
        'NAME': device_name(device),
        'GI/CI': literal(
            f'{device.gpu_instance_id}/{device.compute_instance_id}' if is_mig else '-',
        ),
        'TEMP': metric(NA if is_mig else device.temperature, 'C'),
        'FAN': metric(NA if is_mig else device.fan_speed, '%'),
        'PWR': device_power(device),
        'GPU-MEM': memory,
        'GMEM%': memory_percent,
        'GPU%': metric(device.gpu_utilization, '%'),
        'GMBW%': metric(device.memory_utilization, '%'),
    }


def device_bar_widths(*, available_width: int) -> dict[str, int]:
    """Add memory, GPU, bandwidth, then power bars, reserving a space before each label."""
    widths = {'GPU-MEM': 0, 'GPU%': 0, 'GMBW%': 0, 'PWR': 0}
    count = min(len(widths), max(0, available_width) // (MIN_BAR_WIDTH + 1))
    if count:
        bar_width, extra = divmod(available_width - count, count)
        for index, header in enumerate(widths):
            if index < count:
                widths[header] = bar_width + (index < extra)
    return widths


def power_percent(device: Snapshot) -> float:
    """Normalize board power to its limit, leaving MIG and unsupported limits unavailable."""
    if isinstance(device.index, tuple):
        return math.nan
    limit = float(device.power_limit)
    if not math.isfinite(limit) or limit <= 0:
        return math.nan
    return 100.0 * float(device.power_usage) / limit


# pylint: disable-next=too-many-locals
def device_table(devices: list[Snapshot], *, width: int) -> Table | Text:
    """Prioritize text columns, then memory, GPU, bandwidth, and power bars of at least five cells."""
    if not devices:
        return Text('No NVIDIA devices found', style='yellow')
    devices = sorted(devices, key=lambda device: device_key(device.index))
    rows = [device_row(device) for device in devices]
    headers = list(rows[0])
    if not any(isinstance(device.index, tuple) for device in devices):
        headers.remove('GI/CI')
    column_widths = measure_column_widths(
        rows,
        headers=headers,
        min_column_widths=DEVICE_MIN_COLUMN_WIDTHS,
        max_column_widths=DEVICE_MAX_COLUMN_WIDTHS,
    )
    text_width = sum(column_widths.values()) + len(headers) - 1
    if text_width > width:
        for device, row in zip(devices, rows):
            row['PWR'] = device_power(device, show_limit=False)
        power_width = measure_column_widths(
            rows,
            headers=['PWR'],
            min_column_widths=DEVICE_MIN_COLUMN_WIDTHS,
            max_column_widths=DEVICE_MAX_COLUMN_WIDTHS,
        )['PWR']
        text_width -= column_widths['PWR'] - power_width
        column_widths['PWR'] = power_width
    # Keep all columns at 80 or more; below that, prefer compact metrics over longer names.
    for header in ('NAME', 'FAN', 'GI/CI', 'GMBW%'):
        if width >= 80 or text_width <= width:
            break
        if header in headers:
            headers.remove(header)
            text_width -= column_widths[header] + 1
    if text_width > width and 'NAME' in headers:
        name_width = max(
            len('NAME'),
            DEVICE_MIN_COLUMN_WIDTHS.get('NAME', 0),
            column_widths['NAME'] - (text_width - width),
        )
        for row in rows:
            row['NAME'].truncate(name_width)
        text_width -= column_widths['NAME'] - name_width
        column_widths['NAME'] = name_width
    bar_widths = {
        header: bar_width
        for header, bar_width in device_bar_widths(available_width=width - text_width).items()
        if header in headers
    }
    has_bars = any(bar_widths.values())
    table = make_table(
        'GPUs',
        headers,
        min_column_widths=DEVICE_MIN_COLUMN_WIDTHS,
        max_column_widths=DEVICE_MAX_COLUMN_WIDTHS,
    )
    # Fill the panel through memory and GPU columns even without bars, keeping board metrics compact.
    for header, bar_width in bar_widths.items():
        column = table.columns[headers.index(header)]
        column.ratio = 1 if bar_width or (not has_bars and header in {'GPU-MEM', 'GPU%'}) else None
        column.width = column_widths[header] + (bar_width + 1 if bar_width else 0)
    for device, row in zip(devices, rows):
        cells: dict[str, Text | Table] = dict(row)
        percentages = {
            'GPU-MEM': device.memory_percent,
            'GPU%': device.gpu_utilization,
            'GMBW%': device.memory_utilization,
            'PWR': power_percent(device),
        }
        for header, bar_width in bar_widths.items():
            cells[header] = percentage_bar(
                percentages[header],
                label=row[header],
                width=bar_width,
                label_width=column_widths[header],
            )
        table.add_row(
            *(cells[header] for header in headers),
            style='cyan' if isinstance(device.index, tuple) else '',
        )
    return table


def process_row(process: Snapshot) -> dict[str, Text]:
    """Format process cells before measuring the space needed for optional columns."""
    # Match terminal interface precision and limit CPU labels to six cells.
    cpu_percent = process.cpu_percent
    if not isinstance(cpu_percent, (int, float)) or not math.isfinite(cpu_percent):
        cpu_label = metric(cpu_percent, '%')
    elif cpu_percent > 99999:
        cpu_label = Text('99999+', style='red')
    else:
        cpu_label = metric(int(cpu_percent) if cpu_percent >= 1000 else float(cpu_percent), '%')
    memory = literal(process.gpu_memory_human)
    memory.stylize(metric(process.gpu_memory_percent, '%').style)
    return {
        'GPU': literal(device_label(process.device.index)),
        'PID': literal(process.pid),
        'USER': literal(process.username),
        'GPU-MEM': memory,
        'SM%': metric(process.gpu_sm_utilization, '%'),
        'GMBW%': metric(process.gpu_memory_utilization, '%'),
        'CPU%': cpu_label,
        'MEM%': metric(process.memory_percent, '%'),
        'TIME': literal(process.running_time_human),
        'COMMAND': literal(process.command),
    }


def process_table(processes: list[Snapshot], *, width: int) -> Table | Text:
    """Render one row per device/PID association, including shared PIDs across instances."""
    if not processes:
        return Text('No running GPU processes', style='dim')
    processes = sorted(
        processes,
        key=lambda process: (device_key(process.device.index), process.pid),
    )
    rows = [process_row(process) for process in processes]
    headers = list(rows[0])
    column_widths = measure_column_widths(
        rows,
        headers=headers,
        min_column_widths=PROCESS_MIN_COLUMN_WIDTHS,
        max_column_widths=PROCESS_MAX_COLUMN_WIDTHS,
    )
    # Reserve a readable command prefix; it receives all remaining width in the rendered table.
    column_widths['COMMAND'] = min(column_widths['COMMAND'], 12)
    text_width = sum(column_widths.values()) + len(headers) - 1
    for header in ('TIME', 'USER', 'MEM%', 'CPU%', 'GMBW%'):
        if width >= 80 or text_width <= width:
            break
        headers.remove(header)
        text_width -= column_widths[header] + 1
    table = make_table(
        f'GPU processes ({len(processes)})',
        headers,
        min_column_widths=PROCESS_MIN_COLUMN_WIDTHS,
        max_column_widths=PROCESS_MAX_COLUMN_WIDTHS,
    )
    for row in rows:
        table.add_row(*(row[header] for header in headers))
    return table


def render_dashboard(devices: list[Snapshot], *, processes: list[Snapshot], width: int) -> Group:
    """Build a dashboard at the current terminal width without querying GPU state."""
    hostname = literal(f'{host.getuser()}@{host.hostname()}')
    hostname.stylize('bold green')
    header = Text.assemble(
        hostname,
        (' | ', 'dim'),
        (time.strftime('%Y-%m-%d %H:%M:%S'), 'bold white'),
        no_wrap=True,
        overflow='ellipsis',
    )
    virtual_memory = host.virtual_memory()
    summary = Text.assemble(
        (f'CPU {host.cpu_percent():.1f}%', 'bold cyan'),
        '  ',
        (f'MEM {virtual_memory.percent:.1f}% ({bytes2human(virtual_memory.used)})', 'bold magenta'),
        '  ',
        (f'SWAP {host.swap_memory().percent:.1f}%', 'bold blue'),
    )
    load_average: tuple[float, float, float] | None = host.load_average()
    load = 'N/A' if load_average is None else ' '.join(f'{value:.2f}' for value in load_average)
    load_summary = f'LOAD AVG (1/5/15m) {load}'
    summary.append('  ' if summary.cell_len + len(load_summary) + 2 <= width else '\n')
    summary.append(load_summary, style='bold yellow')
    return Group(
        header,
        summary,
        device_table(devices, width=width),
        process_table(processes, width=width),
    )


@contextmanager
def synchronized_output(console: Console) -> Generator[None]:
    """Present a complete frame together on terminals supporting synchronized output."""
    if not console.is_interactive or console.legacy_windows:
        with console:
            yield
        return

    # Finish rendering before starting the terminal's synchronized-update timeout.
    with console.capture() as capture:
        yield
    try:
        console.file.write(f'\x1b[?2026h{capture.get()}\x1b[?2026l')
        console.file.flush()
    except BaseException:
        # Release the terminal even if `Ctrl+C` interrupts a partially written frame.
        console.file.write('\x1b[?2026l')
        console.file.flush()
        raise


# pylint: disable-next=too-many-locals
def main(argv: list[str] | None = None) -> int:
    """Run the live dashboard, or print one complete snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        metavar='SECONDS',
        help='Polling interval (default: %(default).1f).',
    )
    parser.add_argument('--once', action='store_true', help='Print one snapshot and exit.')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('`--interval` must be a finite positive number of seconds.')

    console = Console()
    dashboard: Group | None = None
    try:
        devices: list[Device] = []
        for device in Device.all():
            devices.append(device)
            devices.extend(device.mig_devices())

        refresh_started = time.monotonic()
        snapshots = collect_snapshot(devices)
        if args.once or not console.is_interactive:
            dashboard = render_dashboard(
                snapshots.devices,
                processes=snapshots.gpu_processes,
                width=console.width,
            )
            raise BreakLoop  # noqa: TRY301

        quit_hint = Text('Ctrl+C to quit', style='dim')
        next_line = Control(ControlType.CARRIAGE_RETURN, (ControlType.CURSOR_DOWN, 1))
        # Rich has no control constructor for erasing from the cursor to the screen's end.
        clear_below = Segments([Segment('\x1b[J', control=[(ControlType.CLEAR, 0)])])
        with console.screen():
            while True:
                size = console.size
                dashboard = render_dashboard(
                    snapshots.devices,
                    processes=snapshots.gpu_processes,
                    width=size.width,
                )
                options = console.options.update(width=size.width, height=None)
                lines = console.render_lines(Group(dashboard, quit_hint), options, pad=True)
                if len(lines) > size.height:
                    overflow = Text.assemble(
                        ('... ', 'yellow'),
                        quit_hint,
                        no_wrap=True,
                        overflow='crop',
                    )
                    lines = lines[: size.height - 1] + console.render_lines(overflow, options)
                if console.legacy_windows:
                    # Legacy Windows needs blank rows to clear the remaining screen.
                    lines.extend([[Segment(' ' * size.width)]] * (size.height - len(lines)))

                with synchronized_output(console):
                    console.control(Control.home())
                    for row, line in enumerate(lines):
                        if row > 0:
                            console.control(next_line)
                        # No newline: writing the bottom row must not scroll the terminal.
                        console.print(Segments(line), end='', crop=False)
                    if len(lines) < size.height:
                        console.control(next_line)
                        console.print(clear_below, end='', crop=False)
                time.sleep(max(0.0, args.interval - (time.monotonic() - refresh_started)))
                refresh_started = time.monotonic()
                snapshots = collect_snapshot(devices)
    except (KeyboardInterrupt, BreakLoop):
        pass
    except libnvml.NVMLError as error:
        Console(stderr=True).print(Text(f'Cannot query NVIDIA devices: {error}', style='bold red'))
        return 1

    # Print the final dashboard snapshot after exiting the loop.
    if dashboard is not None:
        console.print(dashboard)
    return 0


if __name__ == '__main__':
    sys.exit(main())
