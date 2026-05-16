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
"""Minimal one-shot GPU monitor: prints status + processes for every device."""

from __future__ import annotations

from nvitop import Device


def main() -> None:
    """Print a one-shot status summary for every NVML-visible device."""
    devices = Device.all()  # or `Device.cuda.all()` to use CUDA ordinal instead
    for device in devices:
        # Batch all NVML queries for this device into a single round-trip
        with device.oneshot():
            processes = device.processes()
            sorted_pids = sorted(processes.keys())

            print(device)
            print(f'  - Fan speed:       {device.fan_speed()}%')
            print(f'  - Temperature:     {device.temperature()}C')
            print(f'  - GPU utilization: {device.gpu_utilization()}%')
            print(f'  - Total memory:    {device.memory_total_human()}')
            print(f'  - Used memory:     {device.memory_used_human()}')
            print(f'  - Free memory:     {device.memory_free_human()}')
            print(f'  - Processes ({len(processes)}): {sorted_pids}')
            for pid in sorted_pids:
                print(f'    - {processes[pid]}')
        print('-' * 120)


if __name__ == '__main__':
    main()
