# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2026 Xuehai Pan. All Rights Reserved.
# Copyright 2026 YSP. All Rights Reserved.
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
"""Responsive text renderers for the Ascend NPU monitor."""

from __future__ import annotations

import re
import shutil
import time
import unicodedata
from typing import TYPE_CHECKING, Any, cast

from nvitop.api.npu_device import NpuDevice, NpuProcess
from nvitop.api.utils import NA, NaType, bytes2human, colored
from nvitop.version import __version__


if TYPE_CHECKING:
    from nvitop.api.termcolor import Color


BLOCK = '█'
EMPTY = '░'
BAR_WIDTH = 8

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')
_DEVICE_COLUMNS = [
    ('NPU', 4, 'index'),
    ('Health', 7, 'health'),
    ('AICore', 16, 'gpu'),
    ('HBM', 35, 'memory'),
    ('Temp', 8, 'temp'),
    ('Power', 9, 'power'),
    ('Name', 14, 'name'),
    ('Clock', 9, 'clock'),
    ('Bus-Id', 13, 'bus_id'),
]
_PROCESS_COLUMNS = [
    ('NPU', 8, 'npu'),
    ('PID', 9, 'pid'),
    ('HBM', 11, 'memory'),
    ('Command', 34, 'command'),
    ('User', 12, 'user'),
    ('CPU', 7, 'cpu'),
    ('Uptime', 9, 'uptime'),
]


def _plain(text: Any) -> str:
    """Return text without ANSI styling."""
    return _ANSI_ESCAPE.sub('', str(text))


def _display_width(text: str) -> int:
    """Measure terminal cells, including wide CJK characters."""
    return sum(
        0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {'F', 'W'} else 1
        for char in text
    )


def _truncate(text: str, width: int, marker: str) -> str:
    """Truncate plain text to a terminal-cell width."""
    if width <= 0:
        return ''
    marker_width = _display_width(marker)
    available = max(0, width - marker_width)
    used = 0
    chars = []
    for char in text:
        char_width = _display_width(char)
        if used + char_width > available:
            break
        chars.append(char)
        used += char_width
    return ''.join(chars) + marker


def _format_cell(text: Any, width: int, *, no_unicode: bool = False) -> str:
    """Pad or truncate a cell by visible width while preserving ANSI styling."""
    text = str(text)
    plain = _plain(text)
    display_width = _display_width(plain)
    if display_width > width:
        return _truncate(plain, width, '~' if no_unicode else '…')
    return text + ' ' * (width - display_width)


def _fit_line(text: str, width: int, *, no_unicode: bool = False) -> str:
    """Constrain a line to the terminal width."""
    if width <= 0:
        return ''
    plain = _plain(text)
    if _display_width(plain) <= width:
        return text
    return _truncate(plain, width, '~' if no_unicode else '…')


def _border(chars: str, widths: list[int], no_unicode: bool) -> str:
    if no_unicode:
        left, tee, right, hline = '+', '+', '+', '-'
    else:
        left, tee, right, hline = chars[0], chars[1], chars[2], '─'
    return left + tee.join(hline * width for width in widths) + right


def _vline(no_unicode: bool) -> str:
    return '|' if no_unicode else '│'


def _fit_columns(
    columns: list[tuple[str, int, str]],
    term_width: int,
    min_columns: int = 4,
) -> list[tuple[str, int, str]]:
    """Drop secondary columns, then shrink retained columns to the terminal."""
    if term_width <= 0:
        return columns

    def total_width(cols: list[tuple[str, int, str]]) -> int:
        return sum(width for _, width, _ in cols) + len(cols) + 1

    cols = list(columns)
    while len(cols) > min_columns and total_width(cols) > term_width:
        cols.pop()

    while total_width(cols) > term_width and any(width > 4 for _, width, _ in cols):
        index = max(range(len(cols)), key=lambda i: cols[i][1])
        name, width, key = cols[index]
        cols[index] = (name, width - 1, key)

    while len(cols) > 1 and total_width(cols) > term_width:
        cols.pop()
    while total_width(cols) > term_width:
        name, width, key = cols[-1]
        if width <= 1:
            break
        cols[-1] = (name, width - 1, key)
    return cols


def _bar(value: float | NaType, *, no_unicode: bool = False) -> str:
    block, empty = ('#', '.') if no_unicode else (BLOCK, EMPTY)
    if isinstance(value, NaType):
        return empty * BAR_WIDTH
    filled = round(max(0.0, min(100.0, float(value))) / 100.0 * BAR_WIDTH)
    return block * filled + empty * (BAR_WIDTH - filled)


def _metric_bar(
    value: float | NaType,
    color: str,
    colorful: bool,
    *,
    no_unicode: bool,
) -> str:
    bar = _bar(value, no_unicode=no_unicode)
    return colored(bar, cast('Color', color)) if colorful else bar


def _printable(value: Any) -> str:
    if isinstance(value, NaType):
        return 'N/A'
    return ''.join(char if char.isprintable() else ' ' for char in str(value))


def _numeric(value: Any) -> float | None:
    if isinstance(value, NaType):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent(value: float | NaType) -> str:
    return 'N/A' if isinstance(value, NaType) else f'{float(value):.0f}%'


def _format_npus(npus: tuple[int, ...]) -> str:
    if not npus:
        return 'N/A'
    if len(npus) == 1:
        return str(npus[0])
    consecutive = npus == tuple(range(npus[0], npus[-1] + 1))
    return f'{npus[0]}-{npus[-1]}' if consecutive else ','.join(map(str, npus))


def _format_duration(seconds: float | NaType) -> str:
    if isinstance(seconds, NaType) or seconds < 0:
        return 'N/A'
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    if days:
        return f'{days}d{hours:02d}h'
    if hours:
        return f'{hours}h{minutes:02d}m'
    return f'{minutes}m'


def render_header(
    devices: list[NpuDevice],
    *,
    monitor: bool = False,
    colorful: bool = False,
    driver: str | NaType | None = None,
    paused: bool = False,
    no_unicode: bool = False,
    width: int | None = None,
) -> str:
    """Render product, hardware and live-state information."""
    width = width or shutil.get_terminal_size().columns
    driver = NpuDevice.driver_version() if driver is None else driver
    names = sorted({_printable(device.name()) for device in devices})
    model = ', '.join(names) if names else 'Ascend NPU'
    separator = ' | ' if no_unicode else ' · '
    driver_text = f'{separator}Driver {_printable(driver)}' if not isinstance(driver, NaType) else ''
    state = ''
    if monitor:
        state = colored('PAUSED', 'yellow', attrs=('bold',)) if paused else colored('LIVE', 'green', attrs=('bold',))
        state = f'{separator}{state} {time.strftime("%H:%M:%S")}'
    line = f'nvitop-npu {__version__}{separator}{model} x{len(devices)}{driver_text}{state}'
    return _fit_line(
        colored(line, attrs=('bold',)) if colorful else line,
        width,
        no_unicode=no_unicode,
    )


def render_summary(
    devices: list[NpuDevice],
    *,
    process_count: int,
    colorful: bool = False,
    no_unicode: bool = False,
    width: int | None = None,
) -> str:
    """Render aggregate health and utilization for the selected devices."""
    width = width or shutil.get_terminal_size().columns
    health_values = [_printable(device.health()) for device in devices]
    healthy = sum(health == 'OK' for health in health_values)
    gpu_values = [value for device in devices if (value := _numeric(device.gpu_utilization())) is not None]
    used_values = [value for device in devices if (value := _numeric(device.memory_used())) is not None]
    total_values = [value for device in devices if (value := _numeric(device.memory_total())) is not None]
    power_values = [value for device in devices if (value := _numeric(device.power_usage())) is not None]
    temperatures = [value for device in devices if (value := _numeric(device.temperature())) is not None]

    gpu_average = sum(gpu_values) / len(gpu_values) if gpu_values else None
    memory_used = sum(used_values)
    memory_total = sum(total_values)
    memory_percent = memory_used / memory_total * 100 if memory_total else None
    power_total = sum(power_values) / 1000 if power_values else None
    peak_temperature = max(temperatures) if temperatures else None

    health_text = f'{healthy}/{len(devices)} OK'
    health_color = 'green' if healthy == len(devices) else 'red'
    health_text = (
        colored(health_text, cast('Color', health_color), attrs=('bold',))
        if colorful
        else health_text
    )
    separator = ' | ' if no_unicode else ' · '
    row1 = separator.join(
        (
            f'Health {health_text}',
            f'AICore {gpu_average:.0f}% avg' if gpu_average is not None else 'AICore N/A',
            (
                f'HBM {memory_percent:.0f}% '
                f'({bytes2human(memory_used)}/{bytes2human(memory_total)})'
                if memory_percent is not None
                else 'HBM N/A'
            ),
        ),
    )
    row2 = separator.join(
        (
            f'Power {power_total:.0f}W' if power_total is not None else 'Power N/A',
            f'Peak {peak_temperature:.0f}C' if peak_temperature is not None else 'Peak N/A',
            f'Processes {process_count}',
        ),
    )

    vline = _vline(no_unicode)
    hline = '-' if no_unicode else '─'
    top = ('+-' if no_unicode else '╭─') + ' Overview ' + hline * max(0, width - 13) + ('+' if no_unicode else '╮')
    bottom = ('+' if no_unicode else '╰') + hline * max(0, width - 2) + ('+' if no_unicode else '╯')
    body_width = max(0, width - 4)
    rows = [
        f'{vline} {_format_cell(_fit_line(row, body_width, no_unicode=no_unicode), body_width, no_unicode=no_unicode)} {vline}'
        for row in (row1, row2)
    ]
    return '\n'.join(
        (
            _fit_line(top, width, no_unicode=no_unicode),
            *rows,
            _fit_line(bottom, width, no_unicode=no_unicode),
        ),
    )


def render_devices_table(
    devices: list[NpuDevice],
    *,
    colorful: bool = False,
    no_unicode: bool = False,
) -> str:
    """Render the responsive device status table."""
    term_width = shutil.get_terminal_size().columns
    columns = _fit_columns(_DEVICE_COLUMNS, term_width)
    widths = [width for _, width, _ in columns]
    vline = _vline(no_unicode)
    lines = [
        _border('┌┬┐', widths, no_unicode),
        vline
        + vline.join(
            colored(_format_cell(name, width, no_unicode=no_unicode), attrs=('bold',))
            for name, width, _ in columns
        )
        + vline,
        _border('├┼┤', widths, no_unicode),
    ]

    for device in devices:
        gpu = device.gpu_utilization()
        memory = device.memory_percent()
        temperature = device.temperature()
        power = device.power_usage()
        health = device.health()
        cells = {
            'index': str(device.index),
            'health': (
                colored(_printable(health), cast('Color', device.health_color(health)))
                if colorful
                else _printable(health)
            ),
            'gpu': (
                f'{_metric_bar(gpu, device.utilization_color(gpu), colorful, no_unicode=no_unicode)} '
                f'{_percent(gpu)}'
            ),
            'memory': (
                f'{_metric_bar(memory, device.memory_color(memory), colorful, no_unicode=no_unicode)} '
                f'{_percent(memory)} {device.memory_usage()}'
            ),
            'name': _printable(device.name()),
            'power': (
                colored(device.power_status(), cast('Color', device.power_color(power)))
                if colorful
                else device.power_status()
            ),
            'temp': (
                colored(
                    f'{_printable(temperature)}C',
                    cast('Color', device.temperature_color(temperature)),
                )
                if colorful
                else f'{_printable(temperature)}C'
            ),
            'clock': f'{_printable(device.aicore_clock())}MHz',
            'bus_id': _printable(device.bus_id()),
        }
        lines.append(
            vline
            + vline.join(
                _format_cell(cells[key], width, no_unicode=no_unicode)
                for _, width, key in columns
            )
            + vline,
        )

    lines.append(_border('└┴┘', widths, no_unicode))
    return '\n'.join(lines)


def render_processes_table(
    processes: list[tuple[NpuProcess, tuple[int, ...]]],
    *,
    colorful: bool = False,
    no_unicode: bool = False,
) -> str:
    """Render the responsive NPU process table."""
    del colorful
    term_width = shutil.get_terminal_size().columns
    columns = _fit_columns(_PROCESS_COLUMNS, term_width)
    widths = [width for _, width, _ in columns]
    vline = _vline(no_unicode)
    lines = [
        _border('┌┬┐', widths, no_unicode),
        vline
        + vline.join(
            colored(_format_cell(name, width, no_unicode=no_unicode), attrs=('bold',))
            for name, width, _ in columns
        )
        + vline,
        _border('├┼┤', widths, no_unicode),
    ]
    now = time.time()
    for process, npus in processes:
        name = process.name()
        command = process.cmdline()
        if isinstance(command, NaType):
            command = name
        created = process.create_time()
        uptime = NA if isinstance(created, NaType) else max(0.0, now - created)
        cells = {
            'npu': _format_npus(npus),
            'pid': str(process.pid),
            'memory': _printable(process.used_memory_human()),
            'command': _printable(command),
            'user': _printable(process.username()),
            'cpu': f'{process.cpu_percent():.0f}%',
            'uptime': _format_duration(uptime),
        }
        lines.append(
            vline
            + vline.join(
                _format_cell(cells[key], width, no_unicode=no_unicode)
                for _, width, key in columns
            )
            + vline,
        )
    lines.append(_border('└┴┘', widths, no_unicode))
    return '\n'.join(lines)


def render_footer(
    *,
    compact: bool,
    paused: bool,
    sort_by: str,
    no_unicode: bool = False,
    width: int | None = None,
) -> str:
    """Render monitor key hints and current state."""
    width = width or shutil.get_terminal_size().columns
    view = 'compact' if compact else 'full'
    state = 'paused' if paused else 'running'
    separator = ' | ' if no_unicode else ' · '
    text = f'[q] quit  [space] pause  [r] refresh  [c] view:{view}  [s] sort:{sort_by}{separator}{state}'
    return _fit_line(text, width, no_unicode=no_unicode)


__all__ = [
    'render_devices_table',
    'render_footer',
    'render_header',
    'render_processes_table',
    'render_summary',
]
