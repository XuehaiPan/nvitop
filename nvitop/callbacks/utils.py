# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

from typing import List

from nvitop.core import Device


def get_devices_by_logical_ids(device_ids: List[int], unique: bool = True) -> List[Device]:
    cuda_devices = Device.from_cuda_indices(device_ids)

    devices = []
    presented = set()
    for device in cuda_devices:
        if device.cuda_index in presented and unique:
            continue
        devices.append(device)
        presented.add(device.cuda_index)

    return devices
