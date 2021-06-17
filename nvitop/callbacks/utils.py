# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from typing import List

from ..core import Device


def get_devices_by_logical_ids(device_ids: List[int], unique: bool = True) -> List[Device]:
    visible_devices = Device.from_cuda_visible_devices()
    device_count = len(visible_devices)

    devices = []
    presented = set()
    for device_id in device_ids:
        if device_id in presented and unique:
            continue
        if 0 <= device_id < device_count:
            device = visible_devices[device_id]
            device.device_id = device_id
            devices.append(device)
            presented.add(device_id)
        else:
            raise RuntimeError('CUDA Error: invalid device ordinal')

    return devices
