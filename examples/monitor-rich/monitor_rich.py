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
"""A responsive, read-only GPU dashboard using ``nvitop`` and Rich Live.

Run without cloning the repository using ``uv``::

    uv run https://github.com/XuehaiPan/nvitop/raw/HEAD/examples/monitor-rich/monitor_rich.py
"""

# /// script
# requires-python = ">= 3.8"
# dependencies = ["nvitop", "rich"]
# ///

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from nvitop import NA, Device, GpuProcess, bytes2human, host, libnvml, utilization2string


if TYPE_CHECKING:
    from nvitop import Snapshot


MIN_BAR_WIDTH = 5


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
    escaped = ''.join(
        f'\\x{ord(char):02x}' if ord(char) < 32 or 127 <= ord(char) < 160 else char
        for char in single_line
    )
    return Text(escaped, no_wrap=True, overflow='ellipsis')


def device_name(device: Snapshot, *, width: int = 19) -> Text:
    """Omit the vendor, show MIG profiles, and let Rich truncate names to the available width."""
    name = device.name
    if name.startswith('NVIDIA '):
        name = name[len('NVIDIA ') :]
    if isinstance(device.index, tuple):
        name = name.rpartition(' ')[-1]
    text = literal(name)
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


def percentage_bar(value: object, *, width: int, label_width: int) -> Table | Text:
    """Show bars of at least five cells, using empty bars with N/A labels for unavailable values."""
    if width < MIN_BAR_WIDTH:
        return metric(value, '%')
    label = metric(value, '%')
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = 0
    cell = Table.grid(padding=(0, 1))
    cell.add_column(width=width)
    cell.add_column(width=label_width, justify='right', no_wrap=True)
    cell.add_row(
        ProgressBar(
            total=100,
            completed=value,
            width=width,
            complete_style=label.style,
            finished_style=label.style,
        ),
        label,
    )
    return cell


def make_table(title: str, headers: list[str]) -> Table:
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
        max_width = None
        if header == 'NAME':
            max_width = 19
        elif header == 'USER':
            max_width = 12
        table.add_column(
            header,
            justify='left' if header in {'NAME', 'USER', 'COMMAND'} else 'right',
            no_wrap=True,
            overflow='ellipsis',
            ratio=1 if flexible else None,
            width=4 if header == 'TEMP' else None,
            max_width=max_width,
        )
    return table


def device_row(device: Snapshot) -> dict[str, Text]:
    """Format device cells before allocating column widths, with board metrics on physical rows."""
    is_mig = isinstance(device.index, tuple)
    memory = literal(device.memory_used_human)
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
        'PWR': metric(NA if is_mig else round(float(device.power_usage) / 1000.0, 0), 'W'),
        'GPU-MEM': memory,
        'GMEM%': metric(device.memory_percent, '%'),
        'GPU%': metric(device.gpu_utilization, '%'),
        'GMBW%': metric(device.memory_utilization, '%'),
    }


def device_bar_widths(*, available_width: int) -> dict[str, int]:
    """Allocate five-cell bars, preferring memory usage and allowing a space before each label."""
    widths = {'GMEM%': 0, 'GPU%': 0}
    if available_width >= 2 * (MIN_BAR_WIDTH + 1):
        widths['GPU%'] = available_width // 2 - 1
        widths['GMEM%'] = available_width - widths['GPU%'] - 2
    elif available_width >= MIN_BAR_WIDTH + 1:
        widths['GMEM%'] = available_width - 1
    return widths


def device_table(devices: list[Snapshot], *, width: int) -> Table | Text:
    """Prioritize text columns, then memory and GPU bars of at least five cells."""
    if not devices:
        return Text('No NVIDIA devices found', style='yellow')
    devices = sorted(devices, key=lambda device: device_key(device.index))
    rows = [device_row(device) for device in devices]
    headers = list(rows[0])
    if not any(isinstance(device.index, tuple) for device in devices):
        headers.remove('GI/CI')
    column_widths = {
        header: max(len(header), *(row[header].cell_len for row in rows)) for header in headers
    }
    text_width = sum(column_widths.values()) + len(headers) - 1
    # Keep all columns at 80 or more; below that, prefer compact metrics over longer names.
    for header in ('NAME', 'FAN', 'GI/CI', 'GMBW%'):
        if width >= 80 or text_width <= width:
            break
        if header in headers:
            headers.remove(header)
            text_width -= column_widths[header] + 1
    if text_width > width and 'NAME' in headers:
        name_width = max(len('NAME'), column_widths['NAME'] - (text_width - width))
        for device, row in zip(devices, rows):
            row['NAME'] = device_name(device, width=name_width)
        text_width -= column_widths['NAME'] - name_width
        column_widths['NAME'] = name_width
    bar_widths = device_bar_widths(available_width=width - text_width)
    table = make_table('GPUs', headers)
    # Fill the panel through percentage columns even without bars, keeping board metrics compact.
    for header, bar_width in bar_widths.items():
        column = table.columns[headers.index(header)]
        column.ratio = 1 if bar_width or not any(bar_widths.values()) else None
        column.width = column_widths[header] + (bar_width + 1 if bar_width else 0)
    for device, row in zip(devices, rows):
        cells: dict[str, Text | Table] = dict(row)
        cells['GPU%'] = percentage_bar(
            device.gpu_utilization,
            width=bar_widths['GPU%'],
            label_width=column_widths['GPU%'],
        )
        cells['GMEM%'] = percentage_bar(
            device.memory_percent,
            width=bar_widths['GMEM%'],
            label_width=column_widths['GMEM%'],
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
    return {
        'GPU': literal(device_label(process.device.index)),
        'PID': literal(process.pid),
        'USER': literal(process.username),
        'GPU-MEM': literal(process.gpu_memory_human),
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
    column_widths = {
        header: max(len(header), *(row[header].cell_len for row in rows)) for header in headers
    }
    column_widths['USER'] = min(column_widths['USER'], 12)
    # Reserve a readable command prefix; it receives all remaining width in the rendered table.
    column_widths['COMMAND'] = min(column_widths['COMMAND'], 12)
    text_width = sum(column_widths.values()) + len(headers) - 1
    for header in ('TIME', 'USER', 'MEM%', 'CPU%', 'GMBW%'):
        if width >= 80 or text_width <= width:
            break
        headers.remove(header)
        text_width -= column_widths[header] + 1
    table = make_table(f'GPU processes ({len(processes)})', headers)
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
        (
            f'MEM {virtual_memory.percent:.1f}% ({bytes2human(virtual_memory.used)})',
            'bold magenta',
        ),
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


def main(argv: list[str] | None = None) -> int:
    """Run the live dashboard, or print one complete snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        metavar='SECONDS',
        help='Polling interval (default: 1.0).',
    )
    parser.add_argument('--once', action='store_true', help='Print one snapshot and exit.')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('`--interval` must be a finite positive number of seconds.')

    console = Console()
    try:
        devices: list[Device] = []
        for device in Device.all():
            devices.append(device)
            devices.extend(device.mig_devices())
        snapshots = collect_snapshot(devices)
        if args.once or not console.is_interactive:
            console.print(
                render_dashboard(
                    snapshots.devices,
                    processes=snapshots.gpu_processes,
                    width=console.width,
                ),
            )
            return 0

        # Let `Console` own the alternate screen: `Live(screen=True)` crops before adding its
        # overflow marker, which would silently hide rows on short terminals.
        with console.screen(), Live(
            console=console,
            auto_refresh=False,
            vertical_overflow='ellipsis',
        ) as live:
            previous_size = console.size
            while True:
                size = console.size
                if size != previous_size:
                    console.clear()
                    previous_size = size
                live.update(
                    Group(
                        render_dashboard(
                            snapshots.devices,
                            processes=snapshots.gpu_processes,
                            width=size.width,
                        ),
                        Text('Ctrl+C to quit', style='dim'),
                    ),
                    refresh=True,
                )
                time.sleep(args.interval)
                snapshots = collect_snapshot(devices)
    except KeyboardInterrupt:
        return 0
    except libnvml.NVMLError as error:
        Console(stderr=True).print(Text(f'Cannot query NVIDIA devices: {error}', style='bold red'))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
