# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
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

from nvitop import NA, CudaDevice, GpuProcess, colored


def label(text: str) -> str:
    """Format a bold-blue field label for the device summary."""
    return colored(text, color='blue', attrs=('bold',))


def main() -> None:
    """Print a colored one-shot status summary for every CUDA-visible device."""
    print(colored(time.strftime('%a %b %d %H:%M:%S %Y'), color='red', attrs=('bold',)))

    devices = CudaDevice.all()  # or `Device.all()` to use NVML ordinal instead
    separator = False
    for device in devices:
        # Batch all NVML queries for this device into a single round-trip
        with device.oneshot():
            processes = device.processes()

            print(colored(str(device), color='green', attrs=('bold',)))
            print(label('  - Fan speed:       ') + f'{device.fan_speed()}%')
            print(label('  - Temperature:     ') + f'{device.temperature()}C')
            print(label('  - GPU utilization: ') + f'{device.gpu_utilization()}%')
            print(label('  - Total memory:    ') + f'{device.memory_total_human()}')
            print(label('  - Used memory:     ') + f'{device.memory_used_human()}')
            print(label('  - Free memory:     ') + f'{device.memory_free_human()}')
        if len(processes) > 0:
            proc_snapshots = GpuProcess.take_snapshots(processes.values(), failsafe=True)
            proc_snapshots.sort(key=lambda process: (process.username, process.pid))

            print(label(f'  - Processes ({len(proc_snapshots)}):'))
            fmt = (
                '    {pid:<5}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}'
                '  {gpu_memory:>8}  {sm:>3}  {command:<}'
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
                        cpu=snapshot.cpu_percent,
                        host_memory=snapshot.host_memory_human,
                        time=snapshot.running_time_human,
                        gpu_memory=(
                            snapshot.gpu_memory_human
                            if snapshot.gpu_memory_human is not NA
                            else 'WDDM:N/A'
                        ),
                        sm=snapshot.gpu_sm_utilization,
                        command=snapshot.command,
                    ),
                )
        else:
            print(colored('  - No Running Processes', attrs=('bold',)))

        if separator:
            print('-' * 120)
        separator = True


if __name__ == '__main__':
    main()
