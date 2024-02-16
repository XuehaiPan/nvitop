# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2024 Xuehai Pan. All Rights Reserved.
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

# pylint: disable=missing-module-docstring,missing-function-docstring

from __future__ import annotations

from nvitop.api import CudaDevice, Device, MiB


def get_devices_by_logical_ids(device_ids: list[int], unique: bool = True) -> list[CudaDevice]:
    cuda_devices = CudaDevice.from_indices(device_ids)

    devices = []
    presented = set()
    for device in cuda_devices:
        if device.cuda_index in presented and unique:
            continue
        devices.append(device)
        presented.add(device.cuda_index)

    return devices


def get_gpu_stats(
    devices: list[Device],
    memory_utilization: bool = True,
    gpu_utilization: bool = True,
    fan_speed: bool = False,
    temperature: bool = False,
) -> dict[str, float]:
    """Get the GPU status from NVML queries."""
    stats = {}
    for device in devices:
        prefix = f'gpu_id: {device.cuda_index}'
        if device.cuda_index != device.physical_index:
            prefix += f' (physical index: {device.physical_index})'
        with device.oneshot():
            if memory_utilization or gpu_utilization:
                utilization = device.utilization_rates()
                if memory_utilization:
                    stats[f'{prefix}/utilization.memory (%)'] = float(utilization.memory)
                if gpu_utilization:
                    stats[f'{prefix}/utilization.gpu (%)'] = float(utilization.gpu)
            if memory_utilization:
                stats[f'{prefix}/memory.used (MiB)'] = float(device.memory_used()) / MiB
                stats[f'{prefix}/memory.free (MiB)'] = float(device.memory_free()) / MiB
            if fan_speed:
                stats[f'{prefix}/fan.speed (%)'] = float(device.fan_speed())
            if temperature:
                stats[f'{prefix}/temperature.gpu (C)'] = float(device.fan_speed())

    return stats
