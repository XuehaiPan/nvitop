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
"""One-shot GPU monitor with ANSI color, using the CUDA ordinal."""

from __future__ import annotations

import time

from nvitop import NA, CudaDevice, GpuProcess, bytes2human, colored, host


def label(text: str) -> str:
    """Format a bold-blue field label for the device summary."""
    return colored(text, color='blue', attrs=('bold',))


def field(name: str, value: object, unit: str = '', *, width: int = 20, pad: int = 0) -> str:
    """Render a colored ``- label: value`` field; an unavailable value drops its unit."""
    text = str(value) if value is NA else f'{value}{unit}'  # keep a bare `N/A`, never `N/A%`
    prefix = label(f'- {name}:'.ljust(width))
    return f'{prefix} {text.ljust(pad)}' if pad else f'{prefix} {text}'


def cpu_percent_text(value: object) -> str:
    """Format a CPU percentage that may exceed 100% (multi-threaded), matching how nvitop rounds it."""
    if not isinstance(value, (int, float)):
        return 'N/A'  # value is NA or otherwise unavailable
    if value < 1000.0:
        return f'{value:.1f}'  # e.g. 3.0, 282.7 — one decimal keeps 100%+ readable
    if value < 10000.0:
        return str(int(value))  # e.g. 1234 — drop the decimal to fit the column
    return '9999+'


def host_summary() -> str:
    """Build a one-line host summary: CPU, memory, swap, and load average."""
    cpu = host.cpu_percent(interval=0.1)  # interval forces a fresh sample; a bare call reads 0.0%
    virtual_memory = host.virtual_memory()
    swap_memory = host.swap_memory()
    # `load_average()` returns a 3-tuple of floats, or None on platforms without `getloadavg`.
    load_average: tuple[float, float, float] | None = host.load_average()
    load = 'N/A' if load_average is None else ' '.join(f'{value:.2f}' for value in load_average)
    stats = ' | '.join(
        (
            f'CPU {cpu:.1f}%',
            (
                f'Memory {bytes2human(virtual_memory.used)} / {bytes2human(virtual_memory.total)}'
                f' ({virtual_memory.percent:.1f}%)'
            ),
            f'Swap {bytes2human(swap_memory.used)} / {bytes2human(swap_memory.total)}',
            f'Load {load}',
        ),
    )
    host_label = colored('[Host]', color='yellow', attrs=('bold',))
    return f'{host_label} {stats}'


def device_header(device: CudaDevice) -> str:
    """Build the colored device header: green index tag, white device name, green total memory."""
    index_tag = colored(
        f'[CUDA {device.cuda_index} / NVML {device.physical_index}]',
        color='green',
        attrs=('bold',),
    )
    name_tag = colored(device.name(), color='white', attrs=('bold',))
    memory_tag = colored(f'({device.memory_total_human()})', color='green')
    return f'{index_tag} {name_tag} {memory_tag}'


def main() -> None:
    """Print a colored one-shot status summary for every CUDA-visible device."""
    print(
        colored(time.strftime('%a %b %d %H:%M:%S %Y'), color='cyan', attrs=('bold',))
        + '  '
        + colored(f'{host.getuser()}@{host.hostname()}', color='white', attrs=('bold',)),
    )
    print(host_summary())

    devices = CudaDevice.all()  # or `Device.all()` to use NVML ordinal instead
    for device in devices:
        # Batch all NVML queries for this device into a single round-trip
        with device.oneshot():
            processes = device.processes()

            print(device_header(device))
            memory_percent = device.memory_percent()
            memory_used = device.memory_used_human()
            memory_free = device.memory_free_human()
            if memory_percent is not NA:
                memory_used = f'{memory_used} ({memory_percent:.1f}%)'
                memory_free = f'{memory_free} ({100.0 - memory_percent:.1f}%)'
            for left, right in (
                (
                    field('Used Memory', memory_used, width=19, pad=18),
                    field('Free Memory', memory_free, width=15),
                ),
                (
                    field('GPU Utilization', device.gpu_utilization(), '%', width=19, pad=18),
                    field('SM Clock', device.sm_clock(), 'MHz', width=15),
                ),
                (
                    field('Memory Bandwidth', device.memory_utilization(), '%', width=19, pad=18),
                    field('Memory Clock', device.memory_clock(), 'MHz', width=15),
                ),
                (
                    field('Fan Speed', device.fan_speed(), '%', width=19, pad=18),
                    field('Temperature', device.temperature(), 'C', width=15),
                ),
            ):
                print(f'  {left}{right}'.rstrip())
        if len(processes) > 0:
            proc_snapshots = GpuProcess.take_snapshots(processes.values(), failsafe=True)
            proc_snapshots.sort(key=lambda process: (process.username, process.pid))

            print(label(f'  - Processes ({len(proc_snapshots)}):'))
            fmt = (
                '    {pid:<7}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}'
                '  {gpu_memory:>8}  {sm:>3}  {gmbw:>5}  {command:<}'
            ).format
            print(
                colored(
                    fmt(
                        pid='PID',
                        username='USERNAME',
                        cpu='CPU%',
                        host_memory='HOST-MEM',
                        time='TIME',
                        gpu_memory='GPU-MEM',
                        sm='SM%',
                        gmbw='GMBW%',
                        command='COMMAND',
                    ),
                    attrs=('bold',),
                ),
            )
            for snapshot in proc_snapshots:
                print(
                    fmt(
                        pid=snapshot.pid,
                        username=(
                            snapshot.username[:7]
                            + ('+' if len(snapshot.username) > 8 else snapshot.username[7:8])
                        ),
                        cpu=cpu_percent_text(snapshot.cpu_percent),
                        host_memory=snapshot.host_memory_human,
                        time=snapshot.running_time_human,
                        gpu_memory=(
                            snapshot.gpu_memory_human
                            if snapshot.gpu_memory_human is not NA
                            else 'WDDM:N/A'
                        ),
                        sm=snapshot.gpu_sm_utilization,
                        gmbw=snapshot.gpu_memory_utilization,
                        command=snapshot.command,
                    ),
                )
        else:
            print(colored('  - No Running Processes', attrs=('bold',)))


if __name__ == '__main__':
    main()
