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
"""Demonstrate `nvitop.take_snapshots` across NVML and CUDA enumerations."""

from __future__ import annotations

from nvitop import Device, take_snapshots


def main() -> None:
    """Exercise every form of :func:`nvitop.take_snapshots`."""
    print('# Snapshot of all NVML devices and the GPU processes on them')
    print(take_snapshots())  # equivalent to `take_snapshots(Device.all())`

    print()
    print('# Tuple unpacking (devices, gpu_processes)')
    device_snapshots, gpu_process_snapshots = take_snapshots(Device.all())
    print(f'devices: {len(device_snapshots)}, gpu_processes: {len(gpu_process_snapshots)}')

    print()
    print('# Ignore process snapshots')
    device_snapshots, _ = take_snapshots(gpu_processes=False)
    print(f'devices: {len(device_snapshots)}')

    print()
    print('# CUDA device enumeration (honors `CUDA_VISIBLE_DEVICES`)')
    print(take_snapshots(Device.cuda.all()))

    cuda_devices = Device.cuda.all()
    if cuda_devices:
        print()
        print('# Snapshot of just `CUDA 0`')
        print(take_snapshots(cuda_devices[:1]))


if __name__ == '__main__':
    main()
