# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring

import itertools
from collections import OrderedDict
from typing import List, NamedTuple, Iterable, Hashable, Union, Optional

from nvitop.core import host, utils
from nvitop.core.libnvml import nvml, nvmlCheckReturn
from nvitop.core.device import Device, PhysicalDevice, CudaDevice
from nvitop.core.process import HostProcess, GpuProcess, command_join
from nvitop.core.utils import *


__all__ = ['take_snapshots',
           'nvml', 'nvmlCheckReturn', 'NVMLError',
           'Device', 'PhysicalDevice', 'CudaDevice',
           'host', 'HostProcess', 'GpuProcess', 'command_join']
__all__.extend(utils.__all__)


NVMLError = nvml.NVMLError  # pylint: disable=no-member


SnapshotResult = NamedTuple('SnapshotResult',  # in bytes
                            [('devices', List[Snapshot]),
                             ('gpu_processes', List[Snapshot])])


def take_snapshots(
    devices: Optional[Union[Device, Iterable[Device]]] = None, *,
    gpu_processes: Optional[Union[GpuProcess, Iterable[GpuProcess]]] = None
) -> SnapshotResult:
    """Retrieve status of demanded devices and GPU processes.

    Args:
        devices (Optional[Union[Device, Iterable[Device]]]):
            Requested devices for snapshots. If not given, the devices will be
            determined from GPU processes:
                - All devices (no GPU processes are given)
                - Devices that used by given GPU processes
        gpu_processes (Optional[Union[GpuProcess, Iterable[GpuProcess]]]):
            Requested GPU processes snapshots. If not given, all GPU processes
            running on the requested device will be returned.

    Returns:
        SnapshotResult: a named tuple containing two lists of snapshots

    Note:
        If not arguments are specified, all devices and all GPU processes will
        be returned.

    Example::

        >>> from nvitop import take_snapshots, Device, CudaDevice
        >>> import os
        >>> os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

        >>> take_snapshots()  # equivalent to `take_snapshots(Device.all())`
        SnapshotResult(
            devices=[
                DeviceSnapshot(
                    real=Device(index=0, ...),
                    ...
                ),
                ...
            ],
            gpu_processes=[
                GpuProcessSnapshot(
                    real=GpuProcess(pid=xxxxxx, device=Device(index=0, ...), ...),
                    ...
                ),
                ...
            ]
        )

        >>> take_snapshots(CudaDevice.all())
        SnapshotResult(
            devices=[
                CudaDeviceSnapshot(
                    real=CudaDevice(cuda_index=0, physical_index=1, ...),
                    ...
                ),
                CudaDeviceSnapshot(
                    real=CudaDevice(cuda_index=1, physical_index=0, ...),
                    ...
                ),
            ],
            gpu_processes=[
                GpuProcessSnapshot(
                    real=GpuProcess(pid=xxxxxx, device=CudaDevice(cuda_index=0, ...), ...),
                    ...
                ),
                ...
            ]
        )

        >>> take_snapshots(CudaDevice(1))  # <CUDA 1> only
        SnapshotResult(
            devices=[
                CudaDeviceSnapshot(
                    real=CudaDevice(cuda_index=1, physical_index=0, ...),
                    ...
                )
            ],
            gpu_processes=[
                GpuProcessSnapshot(
                    real=GpuProcess(pid=xxxxxx, device=CudaDevice(cuda_index=1, ...), ...),
                    ...
                ),
                ...
            ]
        )
    """

    def unique(iterable: Iterable[Hashable]) -> List[Hashable]:
        return list(OrderedDict.fromkeys(iterable).keys())

    if isinstance(devices, Device):
        devices = [devices]
    if isinstance(gpu_processes, GpuProcess):
        gpu_processes = [gpu_processes]

    if gpu_processes is not None:
        gpu_processes = list(gpu_processes)
        process_devices = unique(process.device for process in gpu_processes)
        for device in process_devices:
            device.processes()  # update GPU status for requested GPU processes
        if devices is None:
            devices = process_devices
    else:
        if devices is None:
            devices = Device.all()
        else:
            devices = list(devices)
        gpu_processes = list(itertools.chain.from_iterable(device.processes().values()
                                                           for device in devices))

    devices = [device.as_snapshot() for device in devices]
    gpu_processes = GpuProcess.take_snapshots(gpu_processes, failsafe=True)

    return SnapshotResult(devices, gpu_processes)
