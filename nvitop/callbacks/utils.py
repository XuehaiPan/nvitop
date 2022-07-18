# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

from typing import Dict, List

from nvitop.core import CudaDevice, Device, MiB


def get_devices_by_logical_ids(device_ids: List[int], unique: bool = True) -> List[CudaDevice]:
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
    devices: List[Device],
    memory_utilization: bool = True,
    gpu_utilization: bool = True,
    fan_speed: bool = False,
    temperature: bool = False,
) -> Dict[str, float]:
    """Get the GPU status from NVML queries"""

    stats = {}
    for device in devices:
        prefix = 'gpu_id: {}'.format(device.cuda_index)
        if device.cuda_index != device.physical_index:
            prefix += ' (physical index: {})'.format(device.physical_index)
        with device.oneshot():
            if memory_utilization or gpu_utilization:
                utilization = device.utilization_rates()
                if memory_utilization:
                    stats['{}/utilization.memory (%)'.format(prefix)] = float(utilization.memory)
                if gpu_utilization:
                    stats['{}/utilization.gpu (%)'.format(prefix)] = float(utilization.gpu)
            if memory_utilization:
                stats['{}/memory.used (MiB)'.format(prefix)] = float(device.memory_used()) / MiB
                stats['{}/memory.free (MiB)'.format(prefix)] = float(device.memory_free()) / MiB
            if fan_speed:
                stats['{}/fan.speed (%)'.format(prefix)] = float(device.fan_speed())
            if temperature:
                stats['{}/temperature.gpu (C)'.format(prefix)] = float(device.fan_speed())

    return stats
