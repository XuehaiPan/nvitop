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
"""The live classes for GPU devices.

The core classes are :class:`Device` and :class:`CudaDevice` (also aliased as :attr:`Device.cuda`).
The type of returned instance created by ``Class(args)`` is depending on the given arguments.

``Device()`` returns:

.. code-block:: python

    - (index: int)        -> PhysicalDevice
    - (index: (int, int)) -> MigDevice
    - (uuid: str)         -> Union[PhysicalDevice, MigDevice]  # depending on the UUID value
    - (bus_id: str)       -> PhysicalDevice

``CudaDevice()`` returns:

.. code-block:: python

    - (cuda_index: int)        -> Union[CudaDevice, CudaMigDevice]  # depending on `CUDA_VISIBLE_DEVICES`
    - (uuid: str)              -> Union[CudaDevice, CudaMigDevice]  # depending on `CUDA_VISIBLE_DEVICES`
    - (nvml_index: int)        -> CudaDevice
    - (nvml_index: (int, int)) -> CudaMigDevice

Examples:
    >>> from nvitop import Device, CudaDevice
    >>> Device.driver_version()              # version of the installed NVIDIA display driver
    '470.129.06'

    >>> Device.count()                       # number of NVIDIA GPUs in the system
    10

    >>> Device.all()                         # all physical devices in the system
    [
        PhysicalDevice(index=0, ...),
        PhysicalDevice(index=1, ...),
        ...
    ]

    >>> nvidia0 = Device(index=0)            # -> PhysicalDevice
    >>> mig10   = Device(index=(1, 0))       # -> MigDevice
    >>> nvidia2 = Device(uuid='GPU-xxxxxx')  # -> PhysicalDevice
    >>> mig30   = Device(uuid='MIG-xxxxxx')  # -> MigDevice

    >>> nvidia0.memory_free()                # total free memory in bytes
    11550654464
    >>> nvidia0.memory_free_human()          # total free memory in human readable format
    '11016MiB'

    >>> nvidia2.as_snapshot()                # takes an onetime snapshot of the device
    PhysicalDeviceSnapshot(
        real=PhysicalDevice(index=2, ...),
        ...
    )

    >>> import os
    >>> os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    >>> os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'

    >>> CudaDevice.count()                     # number of NVIDIA GPUs visible to CUDA applications
    4
    >>> Device.cuda.count()                    # use alias in class `Device`
    4

    >>> CudaDevice.all()                       # all CUDA visible devices (or `Device.cuda.all()`)
    [
        CudaDevice(cuda_index=0, nvml_index=3, ...),
        CudaDevice(cuda_index=1, nvml_index=2, ...),
        ...
    ]

    >>> cuda0 = CudaDevice(cuda_index=0)       # use CUDA ordinal (or `Device.cuda(0)`)
    >>> cuda1 = CudaDevice(nvml_index=2)       # use NVML ordinal
    >>> cuda2 = CudaDevice(uuid='GPU-xxxxxx')  # use UUID string

    >>> cuda0.memory_free()                    # total free memory in bytes
    11550654464
    >>> cuda0.memory_free_human()              # total free memory in human readable format
    '11016MiB'

    >>> cuda1.as_snapshot()                    # takes an onetime snapshot of the device
    CudaDeviceSnapshot(
        real=CudaDevice(cuda_index=1, nvml_index=2, ...),
        ...
    )
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import contextlib
import functools
import multiprocessing as mp
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generator, Iterable, NamedTuple, overload

from nvitop.api import libcuda, libcudart, libnvml
from nvitop.api.process import GpuProcess
from nvitop.api.utils import (
    NA,
    UINT_MAX,
    NaType,
    Snapshot,
    boolify,
    bytes2human,
    memoize_when_activated,
)


if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing_extensions import (
        Literal,  # Python 3.8+
        Self,  # Python 3.11+
    )


__all__ = [
    'Device',
    'PhysicalDevice',
    'MigDevice',
    'CudaDevice',
    'CudaMigDevice',
    'parse_cuda_visible_devices',
    'normalize_cuda_visible_devices',
]

# Class definitions ################################################################################


class MemoryInfo(NamedTuple):  # in bytes # pylint: disable=missing-class-docstring
    total: int | NaType
    free: int | NaType
    used: int | NaType


class ClockInfos(NamedTuple):  # in MHz # pylint: disable=missing-class-docstring
    graphics: int | NaType
    sm: int | NaType
    memory: int | NaType
    video: int | NaType


class ClockSpeedInfos(NamedTuple):  # pylint: disable=missing-class-docstring
    current: ClockInfos
    max: ClockInfos


class UtilizationRates(NamedTuple):  # in percentage # pylint: disable=missing-class-docstring
    gpu: int | NaType
    memory: int | NaType
    encoder: int | NaType
    decoder: int | NaType


class ThroughputInfo(NamedTuple):  # in KiB/s # pylint: disable=missing-class-docstring
    tx: int | NaType
    rx: int | NaType

    @property
    def transmit(self) -> int | NaType:
        """Alias of :attr:`tx`."""
        return self.tx

    @property
    def receive(self) -> int | NaType:
        """Alias of :attr:`rx`."""
        return self.rx


# pylint: disable-next=missing-class-docstring,too-few-public-methods
class ValueOmitted:
    def __repr__(self) -> str:
        return '<VALUE OMITTED>'


_VALUE_OMITTED: str = ValueOmitted()  # type: ignore[assignment]
del ValueOmitted


class Device:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Live class of the GPU devices, different from the device snapshots.

    :meth:`Device.__new__()` returns different types depending on the given arguments.

    .. code-block:: python

        - (index: int)        -> PhysicalDevice
        - (index: (int, int)) -> MigDevice
        - (uuid: str)         -> Union[PhysicalDevice, MigDevice]  # depending on the UUID value
        - (bus_id: str)       -> PhysicalDevice

    Examples:
        >>> Device.driver_version()              # version of the installed NVIDIA display driver
        '470.129.06'

        >>> Device.count()                       # number of NVIDIA GPUs in the system
        10

        >>> Device.all()                         # all physical devices in the system
        [
            PhysicalDevice(index=0, ...),
            PhysicalDevice(index=1, ...),
            ...
        ]

        >>> nvidia0 = Device(index=0)            # -> PhysicalDevice
        >>> mig10   = Device(index=(1, 0))       # -> MigDevice
        >>> nvidia2 = Device(uuid='GPU-xxxxxx')  # -> PhysicalDevice
        >>> mig30   = Device(uuid='MIG-xxxxxx')  # -> MigDevice

        >>> nvidia0.memory_free()                # total free memory in bytes
        11550654464
        >>> nvidia0.memory_free_human()          # total free memory in human readable format
        '11016MiB'

        >>> nvidia2.as_snapshot()                # takes an onetime snapshot of the device
        PhysicalDeviceSnapshot(
            real=PhysicalDevice(index=2, ...),
            ...
        )

    Raises:
        libnvml.NVMLError_LibraryNotFound:
            If cannot find the NVML library, usually the NVIDIA driver is not installed.
        libnvml.NVMLError_DriverNotLoaded:
            If NVIDIA driver is not loaded.
        libnvml.NVMLError_LibRmVersionMismatch:
            If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
            driver without reloading the kernel module.
        libnvml.NVMLError_NotFound:
            If the device is not found for the given NVML identifier.
        libnvml.NVMLError_InvalidArgument:
            If the device index is out of range.
        TypeError:
            If the number of non-None arguments is not exactly 1.
        TypeError:
            If the given index is a tuple but is not consist of two integers.
    """

    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
    # https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices
    # GPU UUID        : `GPU-<GPU-UUID>`
    # MIG UUID        : `MIG-GPU-<GPU-UUID>/<GPU instance ID>/<compute instance ID>`
    # MIG UUID (R470+): `MIG-<MIG-UUID>`
    UUID_PATTERN: re.Pattern = re.compile(
        r"""^  # full match
        (?:(?P<MigMode>MIG)-)?                                 # prefix for MIG UUID
        (?:(?P<GpuUuid>GPU)-)?                                 # prefix for GPU UUID
        (?(MigMode)|(?(GpuUuid)|GPU-))                         # always have a prefix
        (?P<UUID>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})  # UUID for the GPU/MIG device in lower case
        # Suffix for MIG device while using GPU UUID with GPU instance (GI) ID and compute instance (CI) ID
        (?(MigMode)                                            # match only when the MIG prefix matches
            (?(GpuUuid)                                        # match only when provide with GPU UUID
                /(?P<GpuInstanceId>\d+)                        # GI ID of the MIG device
                /(?P<ComputeInstanceId>\d+)                    # CI ID of the MIG device
            |)
        |)
        $""",  # full match
        flags=re.VERBOSE,
    )

    GPU_PROCESS_CLASS: type[GpuProcess] = GpuProcess
    cuda: type[CudaDevice] = None  # type: ignore[assignment] # defined in below
    """Shortcut for class :class:`CudaDevice`."""

    _nvml_index: int | tuple[int, int]

    @classmethod
    def is_available(cls) -> bool:
        """Test whether there are any devices and the NVML library is successfully loaded."""
        try:
            return cls.count() > 0
        except libnvml.NVMLError:
            return False

    @staticmethod
    def driver_version() -> str | NaType:
        """The version of the installed NVIDIA display driver. This is an alphanumeric string.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=0 --format=csv,noheader,nounits --query-gpu=driver_version

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
        """
        return libnvml.nvmlQuery('nvmlSystemGetDriverVersion')

    @staticmethod
    def cuda_driver_version() -> str | NaType:
        """The maximum CUDA version supported by the NVIDIA display driver. This is an alphanumeric string.

        This can be different from the version of the CUDA Runtime. See also :meth:`cuda_runtime_version`.

        Returns: Union[str, NaType]
            The maximum CUDA version supported by the NVIDIA display driver.

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
        """
        cuda_driver_version = libnvml.nvmlQuery('nvmlSystemGetCudaDriverVersion')
        if libnvml.nvmlCheckReturn(cuda_driver_version, int):
            major = cuda_driver_version // 1000
            minor = (cuda_driver_version % 1000) // 10
            revision = cuda_driver_version % 10
            if revision == 0:
                return f'{major}.{minor}'
            return f'{major}.{minor}.{revision}'
        return NA

    max_cuda_version = cuda_driver_version

    @staticmethod
    def cuda_runtime_version() -> str | NaType:
        """The CUDA Runtime version. This is an alphanumeric string.

        This can be different from the CUDA driver version. See also :meth:`cuda_driver_version`.

        Returns: Union[str, NaType]
            The CUDA Runtime version, or :const:`nvitop.NA` when no CUDA Runtime is available or no
            CUDA-capable devices are present.
        """
        try:
            return libcudart.cudaRuntimeGetVersion()
        except libcudart.cudaError:
            return NA

    cudart_version = cuda_runtime_version

    @classmethod
    def count(cls) -> int:
        """The number of NVIDIA GPUs in the system.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=0 --format=csv,noheader,nounits --query-gpu=count

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
        """
        return libnvml.nvmlQuery('nvmlDeviceGetCount', default=0)

    @classmethod
    def all(cls) -> list[PhysicalDevice]:
        """Return a list of all physical devices in the system."""
        return cls.from_indices()  # type: ignore[return-value]

    @classmethod
    def from_indices(
        cls,
        indices: int | Iterable[int | tuple[int, int]] | None = None,
    ) -> list[PhysicalDevice | MigDevice]:
        """Return a list of devices of the given indices.

        Args:
            indices (Iterable[Union[int, Tuple[int, int]]]):
                Indices of the devices. For each index, get :class:`PhysicalDevice` for single int
                and :class:`MigDevice` for tuple (int, int). That is:
                - (int)        -> PhysicalDevice
                - ((int, int)) -> MigDevice

        Returns: List[Union[PhysicalDevice, MigDevice]]
            A list of :class:`PhysicalDevice` and/or :class:`MigDevice` instances of the given indices.

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            libnvml.NVMLError_NotFound:
                If the device is not found for the given NVML identifier.
            libnvml.NVMLError_InvalidArgument:
                If the device index is out of range.
        """
        if indices is None:
            try:
                indices = range(cls.count())
            except libnvml.NVMLError:
                return []

        if isinstance(indices, int):
            indices = [indices]

        return list(map(cls, indices))  # type: ignore[arg-type]

    @staticmethod
    def from_cuda_visible_devices() -> list[CudaDevice]:
        """Return a list of all CUDA visible devices.

        The CUDA ordinal will be enumerate from the ``CUDA_VISIBLE_DEVICES`` environment variable.

        Note:
            The result could be empty if the ``CUDA_VISIBLE_DEVICES`` environment variable is invalid.

        See also for CUDA Device Enumeration:
            - `CUDA Environment Variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_
            - `CUDA Device Enumeration for MIG Device <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices>`_

        Returns: List[CudaDevice]
            A list of :class:`CudaDevice` instances.
        """  # pylint: disable=line-too-long
        visible_device_indices = Device.parse_cuda_visible_devices()

        device_index: int | tuple[int, int]
        cuda_devices: list[CudaDevice] = []
        for cuda_index, device_index in enumerate(visible_device_indices):  # type: ignore[assignment]
            cuda_devices.append(CudaDevice(cuda_index, nvml_index=device_index))

        return cuda_devices

    @staticmethod
    def from_cuda_indices(
        cuda_indices: int | Iterable[int] | None = None,
    ) -> list[CudaDevice]:
        """Return a list of CUDA devices of the given CUDA indices.

        The CUDA ordinal will be enumerate from the ``CUDA_VISIBLE_DEVICES`` environment variable.

        See also for CUDA Device Enumeration:
            - `CUDA Environment Variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_
            - `CUDA Device Enumeration for MIG Device <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices>`_

        Args:
            cuda_indices (Iterable[int]):
                The indices of the GPU in CUDA ordinal, if not given, returns all visible CUDA devices.

        Returns: List[CudaDevice]
            A list of :class:`CudaDevice` of the given CUDA indices.

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            RuntimeError:
                If the index is out of range for the given ``CUDA_VISIBLE_DEVICES`` environment variable.
        """  # pylint: disable=line-too-long
        cuda_devices = Device.from_cuda_visible_devices()
        if cuda_indices is None:
            return cuda_devices

        if isinstance(cuda_indices, int):
            cuda_indices = [cuda_indices]

        cuda_indices = list(cuda_indices)
        cuda_device_count = len(cuda_devices)

        devices = []
        for cuda_index in cuda_indices:
            if not 0 <= cuda_index < cuda_device_count:
                raise RuntimeError(f'CUDA Error: invalid device ordinal: {cuda_index!r}.')
            device = cuda_devices[cuda_index]
            devices.append(device)

        return devices

    @staticmethod
    def parse_cuda_visible_devices(
        cuda_visible_devices: str | None = _VALUE_OMITTED,
    ) -> list[int] | list[tuple[int, int]]:
        """Parse the given ``CUDA_VISIBLE_DEVICES`` value into a list of NVML device indices.

        This is a alias of :func:`parse_cuda_visible_devices`.

        Note:
            The result could be empty if the ``CUDA_VISIBLE_DEVICES`` environment variable is invalid.

        See also for CUDA Device Enumeration:
            - `CUDA Environment Variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_
            - `CUDA Device Enumeration for MIG Device <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices>`_

        Args:
            cuda_visible_devices (Optional[str]):
                The value of the ``CUDA_VISIBLE_DEVICES`` variable. If not given, the value from the
                environment will be used. If explicitly given by :data:`None`, the ``CUDA_VISIBLE_DEVICES``
                environment variable will be unset before parsing.

        Returns: Union[List[int], List[Tuple[int, int]]]
            A list of int (physical device) or a list of tuple of two integers (MIG device) for the
            corresponding real device indices.
        """  # pylint: disable=line-too-long
        return parse_cuda_visible_devices(cuda_visible_devices)

    @staticmethod
    def normalize_cuda_visible_devices(cuda_visible_devices: str | None = _VALUE_OMITTED) -> str:
        """Parse the given ``CUDA_VISIBLE_DEVICES`` value and convert it into a comma-separated string of UUIDs.

        This is an alias of :func:`normalize_cuda_visible_devices`.

        Note:
            The result could be empty string if the ``CUDA_VISIBLE_DEVICES`` environment variable is invalid.

        See also for CUDA Device Enumeration:
            - `CUDA Environment Variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_
            - `CUDA Device Enumeration for MIG Device <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices>`_

        Args:
            cuda_visible_devices (Optional[str]):
                The value of the ``CUDA_VISIBLE_DEVICES`` variable. If not given, the value from the
                environment will be used. If explicitly given by :data:`None`, the ``CUDA_VISIBLE_DEVICES``
                environment variable will be unset before parsing.

        Returns: str
            The comma-separated string (GPU UUIDs) of the ``CUDA_VISIBLE_DEVICES`` environment variable.
        """  # pylint: disable=line-too-long
        return normalize_cuda_visible_devices(cuda_visible_devices)

    def __new__(
        cls,
        index: int | tuple[int, int] | str | None = None,
        *,
        uuid: str | None = None,
        bus_id: str | None = None,
    ) -> Self:
        """Create a new instance of Device.

        The type of the result is determined by the given argument.

        .. code-block:: python

            - (index: int)        -> PhysicalDevice
            - (index: (int, int)) -> MigDevice
            - (uuid: str)         -> Union[PhysicalDevice, MigDevice]  # depending on the UUID value
            - (bus_id: str)       -> PhysicalDevice

        Note: This method takes exact 1 non-None argument.

        Returns: Union[PhysicalDevice, MigDevice]
            A :class:`PhysicalDevice` instance or a :class:`MigDevice` instance.

        Raises:
            TypeError:
                If the number of non-None arguments is not exactly 1.
            TypeError:
                If the given index is a tuple but is not consist of two integers.
        """
        if (index, uuid, bus_id).count(None) != 2:
            raise TypeError(
                f'Device(index=None, uuid=None, bus_id=None) takes 1 non-None arguments '
                f'but (index, uuid, bus_id) = {(index, uuid, bus_id)!r} were given',
            )

        if cls is not Device:
            # Use the subclass type if the type is explicitly specified
            return super().__new__(cls)

        # Auto subclass type inference logic goes here when `cls` is `Device` (e.g., calls `Device(...)`)
        match: re.Match | None = None
        if isinstance(index, str):
            match = cls.UUID_PATTERN.match(index)
            if match is not None:  # passed by UUID
                index, uuid = None, index
        elif isinstance(uuid, str):
            match = cls.UUID_PATTERN.match(uuid)

        if index is not None:
            if not isinstance(index, int):
                if not isinstance(index, tuple):
                    raise TypeError(
                        f'index must be an integer, or a tuple of two integers, or a valid UUID string, '
                        f'but index = {index!r} was given',
                    )
                if not (
                    len(index) == 2 and isinstance(index[0], int) and isinstance(index[1], int)
                ):
                    raise TypeError(
                        f'index for MIG device must be a tuple of two integers '
                        f'but index = {index!r} was given',
                    )
                return super().__new__(MigDevice)  # type: ignore[return-value]
        elif uuid is not None and match is not None and match.group('MigMode') is not None:
            return super().__new__(MigDevice)  # type: ignore[return-value]
        return super().__new__(PhysicalDevice)  # type: ignore[return-value]

    def __init__(
        self,
        index: int | str | None = None,
        *,
        uuid: str | None = None,
        bus_id: str | None = None,
    ) -> None:
        """Initialize the instance created by :meth:`__new__()`.

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            libnvml.NVMLError_NotFound:
                If the device is not found for the given NVML identifier.
            libnvml.NVMLError_InvalidArgument:
                If the device index is out of range.
        """
        if isinstance(index, str) and self.UUID_PATTERN.match(index) is not None:  # passed by UUID
            index, uuid = None, index

        index, uuid, bus_id = (
            arg.encode() if isinstance(arg, str) else arg for arg in (index, uuid, bus_id)
        )

        self._name: str = NA
        self._uuid: str = NA
        self._bus_id: str = NA
        self._memory_total: int | NaType = NA
        self._memory_total_human: str = NA
        self._nvlink_link_count: int | None = None
        self._nvlink_throughput_counters: tuple[tuple[int | NaType, int]] | None = None
        self._is_mig_device: bool | None = None
        self._cuda_index: int | None = None
        self._cuda_compute_capability: tuple[int, int] | NaType | None = None

        if index is not None:
            self._nvml_index = index  # type: ignore[assignment]
            try:
                self._handle = libnvml.nvmlQuery(
                    'nvmlDeviceGetHandleByIndex',
                    index,
                    ignore_errors=False,
                )
            except libnvml.NVMLError_GpuIsLost:
                self._handle = None
                self._name = 'ERROR: GPU is Lost'
            except libnvml.NVMLError_Unknown:
                self._handle = None
                self._name = 'ERROR: Unknown'
        else:
            try:
                if uuid is not None:
                    self._handle = libnvml.nvmlQuery(
                        'nvmlDeviceGetHandleByUUID',
                        uuid,
                        ignore_errors=False,
                    )
                else:
                    self._handle = libnvml.nvmlQuery(
                        'nvmlDeviceGetHandleByPciBusId',
                        bus_id,
                        ignore_errors=False,
                    )
            except libnvml.NVMLError_GpuIsLost:
                self._handle = None
                self._nvml_index = NA  # type: ignore[assignment]
                self._name = 'ERROR: GPU is Lost'
            except libnvml.NVMLError_Unknown:
                self._handle = None
                self._nvml_index = NA  # type: ignore[assignment]
                self._name = 'ERROR: Unknown'
            else:
                self._nvml_index = libnvml.nvmlQuery('nvmlDeviceGetIndex', self._handle)

        self._max_clock_infos: ClockInfos = ClockInfos(graphics=NA, sm=NA, memory=NA, video=NA)
        self._lock: threading.RLock = threading.RLock()

        self._ident: tuple[Hashable, str] = (self.index, self.uuid())
        self._hash: int | None = None

    def __repr__(self) -> str:
        """Return a string representation of the device."""
        return '{}(index={}, name={!r}, total_memory={})'.format(  # noqa: UP032
            self.__class__.__name__,
            self.index,
            self.name(),
            self.memory_total_human(),
        )

    def __eq__(self, other: object) -> bool:
        """Test equality to other object."""
        if not isinstance(other, Device):
            return NotImplemented
        return self._ident == other._ident

    def __hash__(self) -> int:
        """Return a hash value of the device."""
        if self._hash is None:
            self._hash = hash(self._ident)
        return self._hash

    def __getattr__(self, name: str) -> Any | Callable[..., Any]:
        """Get the object attribute.

        If the attribute is not defined, make a method from ``pynvml.nvmlDeviceGet<AttributeName>(handle)``.
        The attribute name will be converted to PascalCase string.

        Raises:
            AttributeError:
                If the attribute is not defined in ``pynvml.py``.

        Examples:
            >>> device = Device(0)

            >>> # Method `cuda_compute_capability` is not implemented in the class definition
            >>> PhysicalDevice.cuda_compute_capability
            AttributeError: type object 'Device' has no attribute 'cuda_compute_capability'

            >>> # Dynamically create a new method from `pynvml.nvmlDeviceGetCudaComputeCapability(device.handle, *args, **kwargs)`
            >>> device.cuda_compute_capability
            <function PhysicalDevice.cuda_compute_capability at 0x7fbfddf5d9d0>

            >>> device.cuda_compute_capability()
            (8, 6)
        """  # pylint: disable=line-too-long
        try:
            return super().__getattr__(name)  # type: ignore[misc]
        except AttributeError:
            if name == '_cache':
                raise
            if self._handle is None:
                return lambda: NA

            match = libnvml.VERSIONED_PATTERN.match(name)
            if match is not None:
                name = match.group('name')
                suffix = match.group('suffix')
            else:
                suffix = ''

            try:
                pascal_case = name.title().replace('_', '')
                func = getattr(libnvml, 'nvmlDeviceGet' + pascal_case + suffix)
            except AttributeError:
                pascal_case = ''.join(
                    part[:1].upper() + part[1:] for part in filter(None, name.split('_'))
                )
                func = getattr(libnvml, 'nvmlDeviceGet' + pascal_case + suffix)

            def attribute(*args: Any, **kwargs: Any) -> Any:
                try:
                    return libnvml.nvmlQuery(
                        func,
                        self._handle,
                        *args,
                        **kwargs,
                        ignore_errors=False,
                    )
                except libnvml.NVMLError_NotSupported:
                    return NA

            attribute.__name__ = name
            attribute.__qualname__ = f'{self.__class__.__name__}.{name}'
            setattr(self, name, attribute)
            return attribute

    def __reduce__(self) -> tuple[type[Device], tuple[int | tuple[int, int]]]:
        """Return state information for pickling."""
        return self.__class__, (self._nvml_index,)

    @property
    def index(self) -> int | tuple[int, int]:
        """The NVML index of the device.

        Returns: Union[int, Tuple[int, int]]
            Returns an int for physical device and tuple of two integers for MIG device.
        """
        return self._nvml_index

    @property
    def nvml_index(self) -> int | tuple[int, int]:
        """The NVML index of the device.

        Returns: Union[int, Tuple[int, int]]
            Returns an int for physical device and tuple of two integers for MIG device.
        """
        return self._nvml_index

    @property
    def physical_index(self) -> int:
        """The index of the physical device.

        Returns: int
            An int for the physical device index. For MIG devices, returns the index of the parent
            physical device.
        """
        return self._nvml_index  # type: ignore[return-value] # will be overridden in MigDevice

    @property
    def handle(self) -> libnvml.c_nvmlDevice_t:
        """The NVML device handle."""
        return self._handle

    @property
    def cuda_index(self) -> int:
        """The CUDA device index.

        The value will be evaluated on the first call.

        Raises:
            RuntimeError:
                If the current device is not visible to CUDA applications (i.e. not listed in the
                ``CUDA_VISIBLE_DEVICES`` environment variable or the environment variable is invalid).
        """
        if self._cuda_index is None:
            visible_device_indices = self.parse_cuda_visible_devices()
            try:
                self._cuda_index = visible_device_indices.index(self.index)  # type: ignore[arg-type]
            except ValueError as ex:
                raise RuntimeError(
                    f'CUDA Error: Device(index={self.index}) is not visible to CUDA applications',
                ) from ex

        return self._cuda_index

    def name(self) -> str | NaType:
        """The official product name of the GPU. This is an alphanumeric string. For all products.

        Returns: Union[str, NaType]
            The official product name, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=name
        """
        if self._name is NA:
            self._name = libnvml.nvmlQuery('nvmlDeviceGetName', self.handle)
        return self._name

    def uuid(self) -> str | NaType:
        """This value is the globally unique immutable alphanumeric identifier of the GPU.

        It does not correspond to any physical label on the board.

        Returns: Union[str, NaType]
            The UUID of the device, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=name
        """
        if self._uuid is NA:
            self._uuid = libnvml.nvmlQuery('nvmlDeviceGetUUID', self.handle)
        return self._uuid

    def bus_id(self) -> str | NaType:
        """PCI bus ID as "domain:bus:device.function", in hex.

        Returns: Union[str, NaType]
            The PCI bus ID of the device, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=pci.bus_id
        """
        if self._bus_id is NA:
            self._bus_id = libnvml.nvmlQuery(
                lambda handle: libnvml.nvmlDeviceGetPciInfo(handle).busId,
                self.handle,
            )
        return self._bus_id

    def serial(self) -> str | NaType:
        """This number matches the serial number physically printed on each board.

        It is a globally unique immutable alphanumeric value.

        Returns: Union[str, NaType]
            The serial number of the device, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=serial
        """
        return libnvml.nvmlQuery('nvmlDeviceGetSerial', self.handle)

    @memoize_when_activated
    def memory_info(self) -> MemoryInfo:  # in bytes
        """Return a named tuple with memory information (in bytes) for the device.

        Returns: MemoryInfo(total, free, used)
            A named tuple with memory information, the item could be :const:`nvitop.NA` when not applicable.
        """
        memory_info = libnvml.nvmlQuery('nvmlDeviceGetMemoryInfo', self.handle)
        if libnvml.nvmlCheckReturn(memory_info):
            return MemoryInfo(total=memory_info.total, free=memory_info.free, used=memory_info.used)
        return MemoryInfo(total=NA, free=NA, used=NA)

    def memory_total(self) -> int | NaType:  # in bytes
        """Total installed GPU memory in bytes.

        Returns: Union[int, NaType]
            Total installed GPU memory in bytes, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=memory.total
        """
        if self._memory_total is NA:
            self._memory_total = self.memory_info().total
        return self._memory_total

    def memory_used(self) -> int | NaType:  # in bytes
        """Total memory allocated by active contexts in bytes.

        Returns: Union[int, NaType]
            Total memory allocated by active contexts in bytes, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=memory.used
        """
        return self.memory_info().used

    def memory_free(self) -> int | NaType:  # in bytes
        """Total free memory in bytes.

        Returns: Union[int, NaType]
            Total free memory in bytes, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=memory.free
        """
        return self.memory_info().free

    def memory_total_human(self) -> str | NaType:  # in human readable
        """Total installed GPU memory in human readable format.

        Returns: Union[str, NaType]
            Total installed GPU memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """
        if self._memory_total_human is NA:
            self._memory_total_human = bytes2human(self.memory_total())
        return self._memory_total_human

    def memory_used_human(self) -> str | NaType:  # in human readable
        """Total memory allocated by active contexts in human readable format.

        Returns: Union[int, NaType]
            Total memory allocated by active contexts in human readable format, or :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long
        return bytes2human(self.memory_used())

    def memory_free_human(self) -> str | NaType:  # in human readable
        """Total free memory in human readable format.

        Returns: Union[int, NaType]
            Total free memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """
        return bytes2human(self.memory_free())

    def memory_percent(self) -> float | NaType:  # in percentage
        """The percentage of used memory over total memory (``0 <= p <= 100``).

        Returns: Union[float, NaType]
            The percentage of used memory over total memory, or :const:`nvitop.NA` when not applicable.
        """
        memory_info = self.memory_info()
        if libnvml.nvmlCheckReturn(memory_info.used, int) and libnvml.nvmlCheckReturn(
            memory_info.total,
            int,
        ):
            return round(100.0 * memory_info.used / memory_info.total, 1)
        return NA

    def memory_usage(self) -> str:  # string of used memory over total memory (in human readable)
        """The used memory over total memory in human readable format.

        Returns: str
            The used memory over total memory in human readable format, or :const:`'N/A / N/A'` when not applicable.
        """  # pylint: disable=line-too-long
        return f'{self.memory_used_human()} / {self.memory_total_human()}'

    @memoize_when_activated
    def bar1_memory_info(self) -> MemoryInfo:  # in bytes
        """Return a named tuple with BAR1 memory information (in bytes) for the device.

        Returns: MemoryInfo(total, free, used)
            A named tuple with BAR1 memory information, the item could be :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long
        memory_info = libnvml.nvmlQuery('nvmlDeviceGetBAR1MemoryInfo', self.handle)
        if libnvml.nvmlCheckReturn(memory_info):
            return MemoryInfo(
                total=memory_info.bar1Total,
                free=memory_info.bar1Free,
                used=memory_info.bar1Used,
            )
        return MemoryInfo(total=NA, free=NA, used=NA)

    def bar1_memory_total(self) -> int | NaType:  # in bytes
        """Total BAR1 memory in bytes.

        Returns: Union[int, NaType]
            Total BAR1 memory in bytes, or :const:`nvitop.NA` when not applicable.
        """
        return self.bar1_memory_info().total

    def bar1_memory_used(self) -> int | NaType:  # in bytes
        """Total used BAR1 memory in bytes.

        Returns: Union[int, NaType]
            Total used BAR1 memory in bytes, or :const:`nvitop.NA` when not applicable.
        """
        return self.bar1_memory_info().used

    def bar1_memory_free(self) -> int | NaType:  # in bytes
        """Total free BAR1 memory in bytes.

        Returns: Union[int, NaType]
            Total free BAR1 memory in bytes, or :const:`nvitop.NA` when not applicable.
        """
        return self.bar1_memory_info().free

    def bar1_memory_total_human(self) -> str | NaType:  # in human readable
        """Total BAR1 memory in human readable format.

        Returns: Union[int, NaType]
            Total BAR1 memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """
        return bytes2human(self.bar1_memory_total())

    def bar1_memory_used_human(self) -> str | NaType:  # in human readable
        """Total used BAR1 memory in human readable format.

        Returns: Union[int, NaType]
            Total used BAR1 memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """
        return bytes2human(self.bar1_memory_used())

    def bar1_memory_free_human(self) -> str | NaType:  # in human readable
        """Total free BAR1 memory in human readable format.

        Returns: Union[int, NaType]
            Total free BAR1 memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """
        return bytes2human(self.bar1_memory_free())

    def bar1_memory_percent(self) -> float | NaType:  # in percentage
        """The percentage of used BAR1 memory over total BAR1 memory (0 <= p <= 100).

        Returns: Union[float, NaType]
            The percentage of used BAR1 memory over total BAR1 memory, or :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long
        memory_info = self.bar1_memory_info()
        if libnvml.nvmlCheckReturn(memory_info.used, int) and libnvml.nvmlCheckReturn(
            memory_info.total,
            int,
        ):
            return round(100.0 * memory_info.used / memory_info.total, 1)
        return NA

    def bar1_memory_usage(self) -> str:  # in human readable
        """The used BAR1 memory over total BAR1 memory in human readable format.

        Returns: str
            The used BAR1 memory over total BAR1 memory in human readable format, or :const:`'N/A / N/A'` when not applicable.
        """  # pylint: disable=line-too-long
        return f'{self.bar1_memory_used_human()} / {self.bar1_memory_total_human()}'

    @memoize_when_activated
    def utilization_rates(self) -> UtilizationRates:  # in percentage
        """Return a named tuple with GPU utilization rates (in percentage) for the device.

        Returns: UtilizationRates(gpu, memory, encoder, decoder)
            A named tuple with GPU utilization rates (in percentage) for the device, the item could be :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long
        gpu, memory, encoder, decoder = NA, NA, NA, NA

        utilization_rates = libnvml.nvmlQuery('nvmlDeviceGetUtilizationRates', self.handle)
        if libnvml.nvmlCheckReturn(utilization_rates):
            gpu, memory = utilization_rates.gpu, utilization_rates.memory

        encoder_utilization = libnvml.nvmlQuery('nvmlDeviceGetEncoderUtilization', self.handle)
        if libnvml.nvmlCheckReturn(encoder_utilization, list) and len(encoder_utilization) > 0:
            encoder = encoder_utilization[0]

        decoder_utilization = libnvml.nvmlQuery('nvmlDeviceGetDecoderUtilization', self.handle)
        if libnvml.nvmlCheckReturn(decoder_utilization, list) and len(decoder_utilization) > 0:
            decoder = decoder_utilization[0]

        return UtilizationRates(gpu=gpu, memory=memory, encoder=encoder, decoder=decoder)

    def gpu_utilization(self) -> int | NaType:  # in percentage
        """Percent of time over the past sample period during which one or more kernels was executing on the GPU.

        The sample period may be between 1 second and 1/6 second depending on the product.

        Returns: Union[int, NaType]
            The GPU utilization rate in percentage, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=utilization.gpu
        """
        return self.utilization_rates().gpu

    gpu_percent = gpu_utilization  # in percentage

    def memory_utilization(self) -> int | NaType:  # in percentage
        """Percent of time over the past sample period during which global (device) memory was being read or written.

        The sample period may be between 1 second and 1/6 second depending on the product.

        Returns: Union[int, NaType]
            The memory bandwidth utilization rate of the GPU in percentage, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=utilization.memory
        """  # pylint: disable=line-too-long
        return self.utilization_rates().memory

    def encoder_utilization(self) -> int | NaType:  # in percentage
        """The encoder utilization rate  in percentage.

        Returns: Union[int, NaType]
            The encoder utilization rate  in percentage, or :const:`nvitop.NA` when not applicable.
        """
        return self.utilization_rates().encoder

    def decoder_utilization(self) -> int | NaType:  # in percentage
        """The decoder utilization rate  in percentage.

        Returns: Union[int, NaType]
            The decoder utilization rate  in percentage, or :const:`nvitop.NA` when not applicable.
        """
        return self.utilization_rates().decoder

    @memoize_when_activated
    def clock_infos(self) -> ClockInfos:  # in MHz
        """Return a named tuple with current clock speeds (in MHz) for the device.

        Returns: ClockInfos(graphics, sm, memory, video)
            A named tuple with current clock speeds (in MHz) for the device, the item could be :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long
        return ClockInfos(
            graphics=libnvml.nvmlQuery(
                'nvmlDeviceGetClockInfo',
                self.handle,
                libnvml.NVML_CLOCK_GRAPHICS,
            ),
            sm=libnvml.nvmlQuery('nvmlDeviceGetClockInfo', self.handle, libnvml.NVML_CLOCK_SM),
            memory=libnvml.nvmlQuery('nvmlDeviceGetClockInfo', self.handle, libnvml.NVML_CLOCK_MEM),
            video=libnvml.nvmlQuery(
                'nvmlDeviceGetClockInfo',
                self.handle,
                libnvml.NVML_CLOCK_VIDEO,
            ),
        )

    clocks = clock_infos

    @memoize_when_activated
    def max_clock_infos(self) -> ClockInfos:  # in MHz
        """Return a named tuple with maximum clock speeds (in MHz) for the device.

        Returns: ClockInfos(graphics, sm, memory, video)
            A named tuple with maximum clock speeds (in MHz) for the device, the item could be :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long
        clock_infos = self._max_clock_infos._asdict()
        for name, clock in clock_infos.items():
            if clock is NA:
                clock_type = getattr(
                    libnvml,
                    'NVML_CLOCK_{}'.format(name.replace('memory', 'mem').upper()),
                )
                clock = libnvml.nvmlQuery('nvmlDeviceGetMaxClockInfo', self.handle, clock_type)
                clock_infos[name] = clock
        self._max_clock_infos = ClockInfos(**clock_infos)
        return self._max_clock_infos

    max_clocks = max_clock_infos

    def clock_speed_infos(self) -> ClockSpeedInfos:  # in MHz
        """Return a named tuple with the current and the maximum clock speeds (in MHz) for the device.

        Returns: ClockSpeedInfos(current, max)
            A named tuple with the current and the maximum clock speeds (in MHz) for the device.
        """
        return ClockSpeedInfos(current=self.clock_infos(), max=self.max_clock_infos())

    def graphics_clock(self) -> int | NaType:  # in MHz
        """Current frequency of graphics (shader) clock in MHz.

        Returns: Union[int, NaType]
            The current frequency of graphics (shader) clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.current.graphics
        """  # pylint: disable=line-too-long
        return self.clock_infos().graphics

    def sm_clock(self) -> int | NaType:  # in MHz
        """Current frequency of SM (Streaming Multiprocessor) clock in MHz.

        Returns: Union[int, NaType]
            The current frequency of SM (Streaming Multiprocessor) clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.current.sm
        """  # pylint: disable=line-too-long
        return self.clock_infos().sm

    def memory_clock(self) -> int | NaType:  # in MHz
        """Current frequency of memory clock in MHz.

        Returns: Union[int, NaType]
            The current frequency of memory clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.current.memory
        """
        return self.clock_infos().memory

    def video_clock(self) -> int | NaType:  # in MHz
        """Current frequency of video encoder/decoder clock in MHz.

        Returns: Union[int, NaType]
            The current frequency of video encoder/decoder clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.current.video
        """  # pylint: disable=line-too-long
        return self.clock_infos().video

    def max_graphics_clock(self) -> int | NaType:  # in MHz
        """Maximum frequency of graphics (shader) clock in MHz.

        Returns: Union[int, NaType]
            The maximum frequency of graphics (shader) clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.max.graphics
        """  # pylint: disable=line-too-long
        return self.max_clock_infos().graphics

    def max_sm_clock(self) -> int | NaType:  # in MHz
        """Maximum frequency of SM (Streaming Multiprocessor) clock in MHz.

        Returns: Union[int, NaType]
            The maximum frequency of SM (Streaming Multiprocessor) clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.max.sm
        """  # pylint: disable=line-too-long
        return self.max_clock_infos().sm

    def max_memory_clock(self) -> int | NaType:  # in MHz
        """Maximum frequency of memory clock in MHz.

        Returns: Union[int, NaType]
            The maximum frequency of memory clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.max.memory
        """
        return self.max_clock_infos().memory

    def max_video_clock(self) -> int | NaType:  # in MHz
        """Maximum frequency of video encoder/decoder clock in MHz.

        Returns: Union[int, NaType]
            The maximum frequency of video encoder/decoder clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.max.video
        """  # pylint: disable=line-too-long
        return self.max_clock_infos().video

    def fan_speed(self) -> int | NaType:  # in percentage
        """The fan speed value is the percent of the product's maximum noise tolerance fan speed that the device's fan is currently intended to run at.

        This value may exceed 100% in certain cases. Note: The reported speed is the intended fan
        speed. If the fan is physically blocked and unable to spin, this output will not match the
        actual fan speed. Many parts do not report fan speeds because they rely on cooling via fans
        in the surrounding enclosure.

        Returns: Union[int, NaType]
            The fan speed value in percentage, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=fan.speed
        """  # pylint: disable=line-too-long
        return libnvml.nvmlQuery('nvmlDeviceGetFanSpeed', self.handle)

    def temperature(self) -> int | NaType:  # in Celsius
        """Core GPU temperature in degrees C.

        Returns: Union[int, NaType]
            The core GPU temperature in Celsius degrees, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=temperature.gpu
        """
        return libnvml.nvmlQuery(
            'nvmlDeviceGetTemperature',
            self.handle,
            libnvml.NVML_TEMPERATURE_GPU,
        )

    @memoize_when_activated
    def power_usage(self) -> int | NaType:  # in milliwatts (mW)
        """The last measured power draw for the entire board in milliwatts.

        Returns: Union[int, NaType]
            The power draw for the entire board in milliwatts, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            $(( "$(nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=power.draw)" * 1000 ))
        """
        return libnvml.nvmlQuery('nvmlDeviceGetPowerUsage', self.handle)

    power_draw = power_usage  # in milliwatts (mW)

    @memoize_when_activated
    def power_limit(self) -> int | NaType:  # in milliwatts (mW)
        """The software power limit in milliwatts.

        Set by software like nvidia-smi.

        Returns: Union[int, NaType]
            The software power limit in milliwatts, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            $(( "$(nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=power.limit)" * 1000 ))
        """
        return libnvml.nvmlQuery('nvmlDeviceGetPowerManagementLimit', self.handle)

    def power_status(self) -> str:  # string of power usage over power limit in watts (W)
        """The string of power usage over power limit in watts.

        Returns: str
            The string of power usage over power limit in watts, or :const:`'N/A / N/A'` when not applicable.
        """  # pylint: disable=line-too-long
        power_usage = self.power_usage()
        power_limit = self.power_limit()
        if libnvml.nvmlCheckReturn(power_usage, int):
            power_usage = f'{round(power_usage / 1000)}W'  # type: ignore[assignment]
        if libnvml.nvmlCheckReturn(power_limit, int):
            power_limit = f'{round(power_limit / 1000)}W'  # type: ignore[assignment]
        return f'{power_usage} / {power_limit}'

    def pcie_throughput(self) -> ThroughputInfo:  # in KiB/s
        """The current PCIe throughput in KiB/s.

        This function is querying a byte counter over a 20ms interval and thus is the PCIe
        throughput over that interval.

        Returns: ThroughputInfo(tx, rx)
            A named tuple with current PCIe throughput in KiB/s, the item could be
            :const:`nvitop.NA` when not applicable.
        """
        return ThroughputInfo(tx=self.pcie_tx_throughput(), rx=self.pcie_rx_throughput())

    @memoize_when_activated
    def pcie_tx_throughput(self) -> int | NaType:  # in KiB/s
        """The current PCIe transmit throughput in KiB/s.

        This function is querying a byte counter over a 20ms interval and thus is the PCIe
        throughput over that interval.

        Returns: Union[int, NaType]
            The current PCIe transmit throughput in KiB/s, or :const:`nvitop.NA` when not applicable.
        """
        return libnvml.nvmlQuery(
            'nvmlDeviceGetPcieThroughput',
            self.handle,
            libnvml.NVML_PCIE_UTIL_RX_BYTES,
        )

    @memoize_when_activated
    def pcie_rx_throughput(self) -> int | NaType:  # in KiB/s
        """The current PCIe receive throughput in KiB/s.

        This function is querying a byte counter over a 20ms interval and thus is the PCIe
        throughput over that interval.

        Returns: Union[int, NaType]
            The current PCIe receive throughput in KiB/s, or :const:`nvitop.NA` when not applicable.
        """
        return libnvml.nvmlQuery(
            'nvmlDeviceGetPcieThroughput',
            self.handle,
            libnvml.NVML_PCIE_UTIL_RX_BYTES,
        )

    def pcie_tx_throughput_human(self) -> str | NaType:  # in human readable
        """The current PCIe transmit throughput in human readable format.

        This function is querying a byte counter over a 20ms interval and thus is the PCIe
        throughput over that interval.

        Returns: Union[str, NaType]
            The current PCIe transmit throughput in human readable format, or :const:`nvitop.NA`
            when not applicable.
        """
        tx = self.pcie_tx_throughput()
        if libnvml.nvmlCheckReturn(tx, int):
            return f'{bytes2human(tx * 1024)}/s'
        return NA

    def pcie_rx_throughput_human(self) -> str | NaType:  # in human readable
        """The current PCIe receive throughput in human readable format.

        This function is querying a byte counter over a 20ms interval and thus is the PCIe
        throughput over that interval.

        Returns: Union[str, NaType]
            The current PCIe receive throughput in human readable format, or :const:`nvitop.NA` when
            not applicable.
        """
        rx = self.pcie_rx_throughput()
        if libnvml.nvmlCheckReturn(rx, int):
            return f'{bytes2human(rx * 1024)}/s'
        return NA

    def nvlink_link_count(self) -> int:
        """The number of NVLinks that the GPU has.

        Returns: Union[int, NaType]
            The number of NVLinks that the GPU has.
        """
        if self._nvlink_link_count is None:
            ((nvlink_link_count, _),) = libnvml.nvmlQueryFieldValues(
                self.handle,
                [libnvml.NVML_FI_DEV_NVLINK_LINK_COUNT],
            )
            if libnvml.nvmlCheckReturn(nvlink_link_count, int):
                self._nvlink_link_count = nvlink_link_count  # type: ignore[assignment]
            else:
                self._nvlink_link_count = 0
        return self._nvlink_link_count  # type: ignore[return-value]

    @memoize_when_activated
    def nvlink_throughput(
        self,
        interval: float | None = None,
    ) -> list[ThroughputInfo]:  # in KiB/s
        """The current NVLink throughput for each NVLink in KiB/s.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: List[ThroughputInfo(tx, rx)]
            A list of named tuples with current NVLink throughput for each NVLink in KiB/s, the item
            could be :const:`nvitop.NA` when not applicable.
        """
        nvlink_link_count = self.nvlink_link_count()
        if nvlink_link_count == 0:
            return []

        def query_nvlink_throughput_counters() -> tuple[tuple[int | NaType, int]]:
            return tuple(  # type: ignore[return-value]
                libnvml.nvmlQueryFieldValues(
                    self.handle,
                    [  # type: ignore[arg-type]
                        (libnvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, i)
                        for i in range(nvlink_link_count)
                    ]
                    + [
                        (libnvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, i)
                        for i in range(nvlink_link_count)
                    ],
                ),
            )

        if interval is not None:
            if not interval >= 0.0:
                raise ValueError(f'`interval` must be a non-negative number, got {interval!r}.')
            if interval > 0.0:
                self._nvlink_throughput_counters = query_nvlink_throughput_counters()
                time.sleep(interval)

        if self._nvlink_throughput_counters is None:
            self._nvlink_throughput_counters = query_nvlink_throughput_counters()
            time.sleep(0.02)  # 20ms

        old_throughput_counters = self._nvlink_throughput_counters
        new_throughput_counters = query_nvlink_throughput_counters()

        throughputs: list[int | NaType] = []
        for (old_counter, old_timestamp), (new_counter, new_timestamp) in zip(
            old_throughput_counters,
            new_throughput_counters,
        ):
            if (
                libnvml.nvmlCheckReturn(old_counter, int)
                and libnvml.nvmlCheckReturn(new_counter, int)
                and new_timestamp > old_timestamp
            ):
                throughputs.append(
                    round(
                        1_000_000 * (new_counter - old_counter) / (new_timestamp - old_timestamp),
                    ),
                )
            else:
                throughputs.append(NA)

        self._nvlink_throughput_counters = new_throughput_counters
        return [
            ThroughputInfo(tx=tx, rx=rx)
            for tx, rx in zip(throughputs[:nvlink_link_count], throughputs[nvlink_link_count:])
        ]

    def nvlink_total_throughput(
        self,
        interval: float | None = None,
    ) -> ThroughputInfo:  # in KiB/s
        """The total NVLink throughput for all NVLinks in KiB/s.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: ThroughputInfo(tx, rx)
            A named tuple with the total NVLink throughput for all NVLinks in KiB/s, the item could
            be :const:`nvitop.NA` when not applicable.
        """
        tx_throughputs = []
        rx_throughputs = []
        for tx, rx in self.nvlink_throughput(interval=interval):
            if libnvml.nvmlCheckReturn(tx, int):
                tx_throughputs.append(tx)
            if libnvml.nvmlCheckReturn(rx, int):
                rx_throughputs.append(rx)
        return ThroughputInfo(
            tx=sum(tx_throughputs) if tx_throughputs else NA,
            rx=sum(rx_throughputs) if rx_throughputs else NA,
        )

    def nvlink_mean_throughput(
        self,
        interval: float | None = None,
    ) -> ThroughputInfo:  # in KiB/s
        """The mean NVLink throughput for all NVLinks in KiB/s.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: ThroughputInfo(tx, rx)
            A named tuple with the mean NVLink throughput for all NVLinks in KiB/s, the item could
            be :const:`nvitop.NA` when not applicable.
        """
        tx_throughputs = []
        rx_throughputs = []
        for tx, rx in self.nvlink_throughput(interval=interval):
            if libnvml.nvmlCheckReturn(tx, int):
                tx_throughputs.append(tx)
            if libnvml.nvmlCheckReturn(rx, int):
                rx_throughputs.append(rx)
        return ThroughputInfo(
            tx=round(sum(tx_throughputs) / len(tx_throughputs)) if tx_throughputs else NA,
            rx=round(sum(rx_throughputs) / len(rx_throughputs)) if rx_throughputs else NA,
        )

    def nvlink_tx_throughput(
        self,
        interval: float | None = None,
    ) -> list[int | NaType]:  # in KiB/s
        """The current NVLink transmit data throughput in KiB/s for each NVLink.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: List[Union[int, NaType]]
            The current NVLink transmit data throughput in KiB/s for each NVLink, or
            :const:`nvitop.NA` when not applicable.
        """
        return [tx for tx, _ in self.nvlink_throughput(interval=interval)]

    def nvlink_mean_tx_throughput(
        self,
        interval: float | None = None,
    ) -> int | NaType:  # in KiB/s
        """The mean NVLink transmit data throughput for all NVLinks in KiB/s.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[int, NaType]
            The mean NVLink transmit data throughput for all NVLinks in KiB/s, or
            :const:`nvitop.NA` when not applicable.
        """
        return self.nvlink_mean_throughput(interval=interval).tx

    def nvlink_total_tx_throughput(
        self,
        interval: float | None = None,
    ) -> int | NaType:  # in KiB/s
        """The total NVLink transmit data throughput for all NVLinks in KiB/s.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[int, NaType]
            The total NVLink transmit data throughput for all NVLinks in KiB/s, or
            :const:`nvitop.NA` when not applicable.
        """
        return self.nvlink_total_throughput(interval=interval).tx

    def nvlink_rx_throughput(
        self,
        interval: float | None = None,
    ) -> list[int | NaType]:  # in KiB/s
        """The current NVLink receive data throughput for each NVLink in KiB/s.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[int, NaType]
            The current NVLink receive data throughput for each NVLink in KiB/s, or
            :const:`nvitop.NA` when not applicable.
        """
        return [rx for _, rx in self.nvlink_throughput(interval=interval)]

    def nvlink_mean_rx_throughput(
        self,
        interval: float | None = None,
    ) -> int | NaType:  # in KiB/s
        """The mean NVLink receive data throughput for all NVLinks in KiB/s.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[int, NaType]
            The mean NVLink receive data throughput for all NVLinks in KiB/s, or
            :const:`nvitop.NA` when not applicable.
        """
        return self.nvlink_mean_throughput(interval=interval).rx

    def nvlink_total_rx_throughput(
        self,
        interval: float | None = None,
    ) -> int | NaType:  # in KiB/s
        """The total NVLink receive data throughput for all NVLinks in KiB/s.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[int, NaType]
            The total NVLink receive data throughput for all NVLinks in KiB/s, or
            :const:`nvitop.NA` when not applicable.
        """
        return self.nvlink_total_throughput(interval=interval).rx

    def nvlink_tx_throughput_human(
        self,
        interval: float | None = None,
    ) -> list[str | NaType]:  # in human readable
        """The current NVLink transmit data throughput for each NVLink in human readable format.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[str, NaType]
            The current NVLink transmit data throughput for each NVLink in human readable format, or
            :const:`nvitop.NA` when not applicable.
        """
        return [
            f'{bytes2human(tx * 1024)}/s' if libnvml.nvmlCheckReturn(tx, int) else NA
            for tx in self.nvlink_tx_throughput(interval=interval)
        ]

    def nvlink_mean_tx_throughput_human(
        self,
        interval: float | None = None,
    ) -> str | NaType:  # in human readable
        """The mean NVLink transmit data throughput for all NVLinks in human readable format.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[str, NaType]
            The mean NVLink transmit data throughput for all NVLinks in human readable format, or
            :const:`nvitop.NA` when not applicable.
        """
        mean_tx = self.nvlink_mean_tx_throughput(interval=interval)
        if libnvml.nvmlCheckReturn(mean_tx, int):
            return f'{bytes2human(mean_tx * 1024)}/s'
        return NA

    def nvlink_total_tx_throughput_human(
        self,
        interval: float | None = None,
    ) -> str | NaType:  # in human readable
        """The total NVLink transmit data throughput for all NVLinks in human readable format.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[str, NaType]
            The total NVLink transmit data throughput for all NVLinks in human readable format, or
            :const:`nvitop.NA` when not applicable.
        """
        total_tx = self.nvlink_total_tx_throughput(interval=interval)
        if libnvml.nvmlCheckReturn(total_tx, int):
            return f'{bytes2human(total_tx * 1024)}/s'
        return NA

    def nvlink_rx_throughput_human(
        self,
        interval: float | None = None,
    ) -> list[str | NaType]:  # in human readable
        """The current NVLink receive data throughput for each NVLink in human readable format.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[str, NaType]
            The current NVLink receive data throughput for each NVLink in human readable format, or
            :const:`nvitop.NA` when not applicable.
        """
        return [
            f'{bytes2human(rx * 1024)}/s' if libnvml.nvmlCheckReturn(rx, int) else NA
            for rx in self.nvlink_rx_throughput(interval=interval)
        ]

    def nvlink_mean_rx_throughput_human(
        self,
        interval: float | None = None,
    ) -> str | NaType:  # in human readable
        """The mean NVLink receive data throughput for all NVLinks in human readable format.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[str, NaType]
            The mean NVLink receive data throughput for all NVLinks in human readable format, or
            :const:`nvitop.NA` when not applicable.
        """
        mean_rx = self.nvlink_mean_rx_throughput(interval=interval)
        if libnvml.nvmlCheckReturn(mean_rx, int):
            return f'{bytes2human(mean_rx * 1024)}/s'
        return NA

    def nvlink_total_rx_throughput_human(
        self,
        interval: float | None = None,
    ) -> str | NaType:  # in human readable
        """The total NVLink receive data throughput for all NVLinks in human readable format.

        This function is querying data counters between methods calls and thus is the NVLink
        throughput over that interval. For the first call, the function is blocking for 20ms to get
        the first data counters.

        Args:
            interval (Optional[float]):
                The interval in seconds between two calls to get the NVLink throughput. If
                ``interval`` is a positive number, compares throughput counters before and after the
                interval (blocking). If ``interval`` is :const`0.0` or :data:`None`, compares
                throughput counters since the last call, returning immediately (non-blocking).

        Returns: Union[str, NaType]
            The total NVLink receive data throughput for all NVLinks in human readable format, or
            :const:`nvitop.NA` when not applicable.
        """
        total_rx = self.nvlink_total_rx_throughput(interval=interval)
        if libnvml.nvmlCheckReturn(total_rx, int):
            return f'{bytes2human(total_rx * 1024)}/s'
        return NA

    def display_active(self) -> str | NaType:
        """A flag that indicates whether a display is initialized on the GPU's (e.g. memory is allocated on the device for display).

        Display can be active even when no monitor is physically attached. "Enabled" indicates an
        active display. "Disabled" indicates otherwise.

        Returns: Union[str, NaType]
            - :const:`'Disabled'`: if not an active display device.
            - :const:`'Enabled'`: if an active display device.
            - :const:`nvitop.NA`: if not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=display_active
        """  # pylint: disable=line-too-long
        return {0: 'Disabled', 1: 'Enabled'}.get(
            libnvml.nvmlQuery('nvmlDeviceGetDisplayActive', self.handle),
            NA,
        )

    def display_mode(self) -> str | NaType:
        """A flag that indicates whether a physical display (e.g. monitor) is currently connected to any of the GPU's connectors.

        "Enabled" indicates an attached display. "Disabled" indicates otherwise.

        Returns: Union[str, NaType]
            - :const:`'Disabled'`: if the display mode is disabled.
            - :const:`'Enabled'`: if the display mode is enabled.
            - :const:`nvitop.NA`: if not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=display_mode
        """  # pylint: disable=line-too-long
        return {0: 'Disabled', 1: 'Enabled'}.get(
            libnvml.nvmlQuery('nvmlDeviceGetDisplayMode', self.handle),
            NA,
        )

    def current_driver_model(self) -> str | NaType:
        """The driver model currently in use.

        Always "N/A" on Linux. On Windows, the TCC (WDM) and WDDM driver models are supported. The
        TCC driver model is optimized for compute applications. I.E. kernel launch times will be
        quicker with TCC. The WDDM driver model is designed for graphics applications and is not
        recommended for compute applications. Linux does not support multiple driver models, and
        will always have the value of "N/A".

        Returns: Union[str, NaType]
            - :const:`'WDDM'`: for WDDM driver model on Windows.
            - :const:`'WDM'`: for TTC (WDM) driver model on Windows.
            - :const:`nvitop.NA`: if not applicable, e.g. on Linux.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=driver_model.current
        """
        return {libnvml.NVML_DRIVER_WDDM: 'WDDM', libnvml.NVML_DRIVER_WDM: 'WDM'}.get(
            libnvml.nvmlQuery('nvmlDeviceGetCurrentDriverModel', self.handle),
            NA,
        )

    driver_model = current_driver_model

    def persistence_mode(self) -> str | NaType:
        """A flag that indicates whether persistence mode is enabled for the GPU. Value is either "Enabled" or "Disabled".

        When persistence mode is enabled the NVIDIA driver remains loaded even when no active
        clients, such as X11 or nvidia-smi, exist. This minimizes the driver load latency associated
        with running dependent apps, such as CUDA programs. Linux only.

        Returns: Union[str, NaType]
            - :const:`'Disabled'`: if the persistence mode is disabled.
            - :const:`'Enabled'`: if the persistence mode is enabled.
            - :const:`nvitop.NA`: if not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=persistence_mode
        """  # pylint: disable=line-too-long
        return {0: 'Disabled', 1: 'Enabled'}.get(
            libnvml.nvmlQuery('nvmlDeviceGetPersistenceMode', self.handle),
            NA,
        )

    def performance_state(self) -> str | NaType:
        """The current performance state for the GPU. States range from P0 (maximum performance) to P12 (minimum performance).

        Returns: Union[str, NaType]
            The current performance state in format ``P<int>``, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=pstate
        """  # pylint: disable=line-too-long
        performance_state = libnvml.nvmlQuery('nvmlDeviceGetPerformanceState', self.handle)
        if libnvml.nvmlCheckReturn(performance_state, int):
            performance_state = 'P' + str(performance_state)
        return performance_state

    def total_volatile_uncorrected_ecc_errors(self) -> int | NaType:
        """Total errors detected across entire chip.

        Returns: Union[int, NaType]
            The total number of uncorrected errors in volatile ECC memory, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=ecc.errors.uncorrected.volatile.total
        """  # pylint: disable=line-too-long
        return libnvml.nvmlQuery(
            'nvmlDeviceGetTotalEccErrors',
            self.handle,
            libnvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
            libnvml.NVML_VOLATILE_ECC,
        )

    def compute_mode(self) -> str | NaType:
        """The compute mode flag indicates whether individual or multiple compute applications may run on the GPU.

        Returns: Union[str, NaType]
            - :const:`'Default'`: means multiple contexts are allowed per device.
            - :const:`'Exclusive Thread'`: deprecated, use Exclusive Process instead
            - :const:`'Prohibited'`: means no contexts are allowed per device (no compute apps).
            - :const:`'Exclusive Process'`: means only one context is allowed per device, usable from multiple threads at a time.
            - :const:`nvitop.NA`: if not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=compute_mode
        """  # pylint: disable=line-too-long
        return {
            libnvml.NVML_COMPUTEMODE_DEFAULT: 'Default',
            libnvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: 'Exclusive Thread',
            libnvml.NVML_COMPUTEMODE_PROHIBITED: 'Prohibited',
            libnvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: 'Exclusive Process',
        }.get(libnvml.nvmlQuery('nvmlDeviceGetComputeMode', self.handle), NA)

    def cuda_compute_capability(self) -> tuple[int, int] | NaType:
        """The CUDA compute capability for the device.

        Returns: Union[Tuple[int, int], NaType]
            The CUDA compute capability version in format ``(major, minor)``, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=compute_cap
        """
        if self._cuda_compute_capability is None:
            self._cuda_compute_capability = libnvml.nvmlQuery(
                'nvmlDeviceGetCudaComputeCapability',
                self.handle,
            )
        return self._cuda_compute_capability

    def is_mig_device(self) -> bool:
        """Return whether or not the device is a MIG device."""
        if self._is_mig_device is None:
            is_mig_device = libnvml.nvmlQuery(
                'nvmlDeviceIsMigDeviceHandle',
                self.handle,
                default=False,
                ignore_function_not_found=True,
            )
            self._is_mig_device = bool(is_mig_device)  # nvmlDeviceIsMigDeviceHandle returns c_uint
        return self._is_mig_device

    def mig_mode(self) -> str | NaType:
        """The MIG mode that the GPU is currently operating under.

        Returns: Union[str, NaType]
            - :const:`'Disabled'`: if the MIG mode is disabled.
            - :const:`'Enabled'`: if the MIG mode is enabled.
            - :const:`nvitop.NA`: if not applicable, e.g. the GPU does not support MIG mode.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=mig.mode.current
        """
        if self.is_mig_device():
            return NA

        mig_mode, *_ = libnvml.nvmlQuery(
            'nvmlDeviceGetMigMode',
            self.handle,
            default=(NA, NA),
            ignore_function_not_found=True,
        )
        return {0: 'Disabled', 1: 'Enabled'}.get(mig_mode, NA)

    def is_mig_mode_enabled(self) -> bool:
        """Test whether the MIG mode is enabled on the device.

        Return :data:`False` if MIG mode is disabled or the device does not support MIG mode.
        """
        return boolify(self.mig_mode())

    def max_mig_device_count(self) -> int:
        """Return the maximum number of MIG instances the device supports.

        This method will return 0 if the device does not support MIG mode.
        """
        return 0  # implemented in PhysicalDevice

    def mig_devices(self) -> list[MigDevice]:
        """Return a list of children MIG devices of the current device.

        This method will return an empty list if the MIG mode is disabled or the device does not
        support MIG mode.
        """
        return []  # implemented in PhysicalDevice

    def is_leaf_device(self) -> bool:
        """Test whether the device is a physical device with MIG mode disabled or a MIG device.

        Return :data:`True` if the device is a physical device with MIG mode disabled or a MIG device.
        Otherwise, return :data:`False` if the device is a physical device with MIG mode enabled.
        """
        return self.is_mig_device() or not self.is_mig_mode_enabled()

    def to_leaf_devices(
        self,
    ) -> list[PhysicalDevice] | list[MigDevice] | list[CudaDevice] | list[CudaMigDevice]:
        """Return a list of leaf devices.

        Note that a CUDA device is always a leaf device.
        """
        if isinstance(self, CudaDevice) or self.is_leaf_device():
            return [self]  # type: ignore[return-value]
        return self.mig_devices()

    def processes(self) -> dict[int, GpuProcess]:
        """Return a dictionary of processes running on the GPU.

        Returns: Dict[int, GpuProcess]
            A dictionary mapping PID to GPU process instance.
        """
        processes = {}

        found_na = False
        for type, func in (  # pylint: disable=redefined-builtin
            ('C', 'nvmlDeviceGetComputeRunningProcesses'),
            ('G', 'nvmlDeviceGetGraphicsRunningProcesses'),
        ):
            for p in libnvml.nvmlQuery(func, self.handle, default=()):
                if isinstance(p.usedGpuMemory, int):
                    gpu_memory = p.usedGpuMemory
                else:
                    # Used GPU memory is `N/A` on Windows Display Driver Model (WDDM)
                    # or on MIG-enabled GPUs
                    gpu_memory = NA  # type: ignore[assignment]
                    found_na = True
                proc = processes[p.pid] = self.GPU_PROCESS_CLASS(
                    pid=p.pid,
                    device=self,
                    gpu_memory=gpu_memory,
                    gpu_instance_id=getattr(p, 'gpuInstanceId', UINT_MAX),
                    compute_instance_id=getattr(p, 'computeInstanceId', UINT_MAX),
                )
                proc.type = proc.type + type

        if len(processes) > 0:
            samples = libnvml.nvmlQuery(
                'nvmlDeviceGetProcessUtilization',
                self.handle,
                # Only utilization samples that were recorded after this timestamp will be returned.
                # The CPU timestamp, i.e. absolute Unix epoch timestamp (in microseconds), is used.
                # Here we use the timestamp 1 second ago to ensure the record buffer is not empty.
                time.time_ns() // 1000 - 1000_000,
                default=(),
            )
            for s in sorted(samples, key=lambda s: s.timeStamp):
                try:
                    processes[s.pid].set_gpu_utilization(s.smUtil, s.memUtil, s.encUtil, s.decUtil)
                except KeyError:  # noqa: PERF203
                    pass
            if not found_na:
                for pid in set(processes).difference(s.pid for s in samples):
                    processes[pid].set_gpu_utilization(0, 0, 0, 0)

        return processes

    def as_snapshot(self) -> Snapshot:
        """Return a onetime snapshot of the device.

        The attributes are defined in :attr:`SNAPSHOT_KEYS`.
        """
        with self.oneshot():
            return Snapshot(
                real=self,
                index=self.index,
                physical_index=self.physical_index,
                **{key: getattr(self, key)() for key in self.SNAPSHOT_KEYS},
            )

    SNAPSHOT_KEYS: ClassVar[list[str]] = [
        'name',
        'uuid',
        'bus_id',
        'memory_info',
        'memory_used',
        'memory_free',
        'memory_total',
        'memory_used_human',
        'memory_free_human',
        'memory_total_human',
        'memory_percent',
        'memory_usage',
        'utilization_rates',
        'gpu_utilization',
        'memory_utilization',
        'encoder_utilization',
        'decoder_utilization',
        'clock_infos',
        'max_clock_infos',
        'clock_speed_infos',
        'sm_clock',
        'memory_clock',
        'fan_speed',
        'temperature',
        'power_usage',
        'power_limit',
        'power_status',
        'pcie_throughput',
        'pcie_tx_throughput',
        'pcie_rx_throughput',
        'pcie_tx_throughput_human',
        'pcie_rx_throughput_human',
        'display_active',
        'display_mode',
        'current_driver_model',
        'persistence_mode',
        'performance_state',
        'total_volatile_uncorrected_ecc_errors',
        'compute_mode',
        'cuda_compute_capability',
        'mig_mode',
    ]

    # Modified from psutil (https://github.com/giampaolo/psutil)
    @contextlib.contextmanager
    def oneshot(self) -> Generator[None]:
        """A utility context manager which considerably speeds up the retrieval of multiple device information at the same time.

        Internally different device info (e.g. memory_info, utilization_rates, ...) may be fetched
        by using the same routine, but only one information is returned and the others are discarded.
        When using this context manager the internal routine is executed once (in the example below
        on memory_info()) and the other info are cached.

        The cache is cleared when exiting the context manager block. The advice is to use this every
        time you retrieve more than one information about the device.

        Examples:
            >>> from nvitop import Device
            >>> device = Device(0)
            >>> with device.oneshot():
            ...     device.memory_info()        # collect multiple info
            ...     device.memory_used()        # return cached value
            ...     device.memory_free_human()  # return cached value
            ...     device.memory_percent()     # return cached value
        """  # pylint: disable=line-too-long
        with self._lock:
            # pylint: disable=no-member
            if hasattr(self, '_cache'):
                # NOOP: this covers the use case where the user enters the
                # context twice:
                #
                # >>> with device.oneshot():
                # ...    with device.oneshot():
                # ...
                #
                # Also, since as_snapshot() internally uses oneshot()
                # I expect that the code below will be a pretty common
                # "mistake" that the user will make, so let's guard
                # against that:
                #
                # >>> with device.oneshot():
                # ...    device.as_snapshot()
                # ...
                yield
            else:
                try:
                    self.memory_info.cache_activate(self)  # type: ignore[attr-defined]
                    self.bar1_memory_info.cache_activate(self)  # type: ignore[attr-defined]
                    self.utilization_rates.cache_activate(self)  # type: ignore[attr-defined]
                    self.clock_infos.cache_activate(self)  # type: ignore[attr-defined]
                    self.max_clock_infos.cache_activate(self)  # type: ignore[attr-defined]
                    self.power_usage.cache_activate(self)  # type: ignore[attr-defined]
                    self.power_limit.cache_activate(self)  # type: ignore[attr-defined]
                    yield
                finally:
                    self.memory_info.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.bar1_memory_info.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.utilization_rates.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.clock_infos.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.max_clock_infos.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.power_usage.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.power_limit.cache_deactivate(self)  # type: ignore[attr-defined]


class PhysicalDevice(Device):
    """Class for physical devices.

    This is the real GPU installed in the system.
    """

    _nvml_index: int
    index: int
    nvml_index: int

    @property
    def physical_index(self) -> int:
        """Zero based index of the GPU. Can change at each boot.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=index
        """
        return self._nvml_index

    def max_mig_device_count(self) -> int:
        """Return the maximum number of MIG instances the device supports.

        This method will return 0 if the device does not support MIG mode.
        """
        return libnvml.nvmlQuery(
            'nvmlDeviceGetMaxMigDeviceCount',
            self.handle,
            default=0,
            ignore_function_not_found=True,
        )

    def mig_device(self, mig_index: int) -> MigDevice:
        """Return a child MIG device of the given index.

        Raises:
            libnvml.NVMLError:
                If the device does not support MIG mode or the given MIG device does not exist.
        """
        with _global_physical_device(self):
            return MigDevice(index=(self.index, mig_index))

    def mig_devices(self) -> list[MigDevice]:
        """Return a list of children MIG devices of the current device.

        This method will return an empty list if the MIG mode is disabled or the device does not
        support MIG mode.
        """
        mig_devices = []

        if self.is_mig_mode_enabled():
            max_mig_device_count = self.max_mig_device_count()
            with _global_physical_device(self):
                for mig_index in range(max_mig_device_count):
                    try:
                        mig_device = MigDevice(index=(self.index, mig_index))
                    except libnvml.NVMLError:  # noqa: PERF203
                        break
                    else:
                        mig_devices.append(mig_device)

        return mig_devices


class MigDevice(Device):  # pylint: disable=too-many-instance-attributes
    """Class for MIG devices."""

    _nvml_index: tuple[int, int]
    nvml_index: tuple[int, int]

    @classmethod
    def count(cls) -> int:
        """The number of total MIG devices aggregated over all physical devices."""
        return len(cls.all())

    @classmethod
    def all(cls) -> list[MigDevice]:  # type: ignore[override]
        """Return a list of MIG devices aggregated over all physical devices."""
        mig_devices = []
        for device in PhysicalDevice.all():
            mig_devices.extend(device.mig_devices())
        return mig_devices

    @classmethod
    def from_indices(  # type: ignore[override] # pylint: disable=signature-differs
        cls,
        indices: Iterable[tuple[int, int]],
    ) -> list[MigDevice]:
        """Return a list of MIG devices of the given indices.

        Args:
            indices (Iterable[Tuple[int, int]]):
                Indices of the MIG devices. Each index is a tuple of two integers.

        Returns: List[MigDevice]
            A list of :class:`MigDevice` instances of the given indices.

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            libnvml.NVMLError_NotFound:
                If the device is not found for the given NVML identifier.
        """
        return list(map(cls, indices))

    # pylint: disable-next=super-init-not-called
    def __init__(
        self,
        index: tuple[int, int] | str | None = None,
        *,
        uuid: str | None = None,
    ) -> None:
        """Initialize the instance created by :meth:`__new__()`.

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            libnvml.NVMLError_NotFound:
                If the device is not found for the given NVML identifier.
        """
        if isinstance(index, str) and self.UUID_PATTERN.match(index) is not None:  # passed by UUID
            index, uuid = None, index

        index, uuid = (arg.encode() if isinstance(arg, str) else arg for arg in (index, uuid))

        self._name: str = NA
        self._uuid: str = NA
        self._bus_id: str = NA
        self._memory_total: int | NaType = NA
        self._memory_total_human: str = NA
        self._gpu_instance_id: int | NaType = NA
        self._compute_instance_id: int | NaType = NA
        self._nvlink_link_count: int | None = None
        self._nvlink_throughput_counters: tuple[tuple[int | NaType, int]] | None = None
        self._is_mig_device: bool = True
        self._cuda_index: int | None = None
        self._cuda_compute_capability: tuple[int, int] | NaType | None = None

        if index is not None:
            self._nvml_index = index  # type: ignore[assignment]
            self._handle = None

            parent = _get_global_physical_device()
            if (
                parent is None
                or parent.handle is None
                or parent.physical_index != self.physical_index
            ):
                parent = PhysicalDevice(index=self.physical_index)
            self._parent = parent
            if self.parent.handle is not None:
                try:
                    self._handle = libnvml.nvmlQuery(
                        'nvmlDeviceGetMigDeviceHandleByIndex',
                        self.parent.handle,
                        self.mig_index,
                        ignore_errors=False,
                    )
                except libnvml.NVMLError_GpuIsLost:
                    pass
        else:
            self._handle = libnvml.nvmlQuery('nvmlDeviceGetHandleByUUID', uuid, ignore_errors=False)
            parent_handle = libnvml.nvmlQuery(
                'nvmlDeviceGetDeviceHandleFromMigDeviceHandle',
                self.handle,
                ignore_errors=False,
            )
            parent_index = libnvml.nvmlQuery(
                'nvmlDeviceGetIndex',
                parent_handle,
                ignore_errors=False,
            )
            self._parent = PhysicalDevice(index=parent_index)
            for mig_device in self.parent.mig_devices():
                if self.uuid() == mig_device.uuid():
                    self._nvml_index = mig_device.index
                    break
            else:
                raise libnvml.NVMLError_NotFound

        self._max_clock_infos = ClockInfos(graphics=NA, sm=NA, memory=NA, video=NA)
        self._lock = threading.RLock()

        self._ident = (self.index, self.uuid())
        self._hash = None

    @property
    def index(self) -> tuple[int, int]:
        """The index of the MIG device. This is a tuple of two integers."""
        return self._nvml_index

    @property
    def physical_index(self) -> int:
        """The index of the parent physical device."""
        return self._nvml_index[0]

    @property
    def mig_index(self) -> int:
        """The index of the MIG device over the all MIG devices of the parent device."""
        return self._nvml_index[1]

    @property
    def parent(self) -> PhysicalDevice:
        """The parent physical device."""
        return self._parent

    def gpu_instance_id(self) -> int | NaType:
        """The gpu instance ID of the MIG device.

        Returns: Union[int, NaType]
            The gpu instance ID of the MIG device, or :const:`nvitop.NA` when not applicable.
        """
        if self._gpu_instance_id is NA:
            self._gpu_instance_id = libnvml.nvmlQuery(
                'nvmlDeviceGetGpuInstanceId',
                self.handle,
                default=UINT_MAX,
            )
            if self._gpu_instance_id == UINT_MAX:
                self._gpu_instance_id = NA
        return self._gpu_instance_id

    def compute_instance_id(self) -> int | NaType:
        """The compute instance ID of the MIG device.

        Returns: Union[int, NaType]
            The compute instance ID of the MIG device, or :const:`nvitop.NA` when not applicable.
        """
        if self._compute_instance_id is NA:
            self._compute_instance_id = libnvml.nvmlQuery(
                'nvmlDeviceGetComputeInstanceId',
                self.handle,
                default=UINT_MAX,
            )
            if self._compute_instance_id == UINT_MAX:
                self._compute_instance_id = NA
        return self._compute_instance_id

    def as_snapshot(self) -> Snapshot:
        """Return a onetime snapshot of the device.

        The attributes are defined in :attr:`SNAPSHOT_KEYS`.
        """
        snapshot = super().as_snapshot()
        snapshot.mig_index = self.mig_index  # type: ignore[attr-defined]

        return snapshot

    SNAPSHOT_KEYS: ClassVar[list[str]] = [
        *Device.SNAPSHOT_KEYS,
        'gpu_instance_id',
        'compute_instance_id',
    ]


class CudaDevice(Device):
    """Class for devices enumerated over the CUDA ordinal.

    The order can be vary for different ``CUDA_VISIBLE_DEVICES`` environment variable.

    See also for CUDA Device Enumeration:
        - `CUDA Environment Variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_
        - `CUDA Device Enumeration for MIG Device <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices>`_

    :meth:`CudaDevice.__new__()` returns different types depending on the given arguments.

    .. code-block:: python

        - (cuda_index: int)        -> Union[CudaDevice, CudaMigDevice]  # depending on `CUDA_VISIBLE_DEVICES`
        - (uuid: str)              -> Union[CudaDevice, CudaMigDevice]  # depending on `CUDA_VISIBLE_DEVICES`
        - (nvml_index: int)        -> CudaDevice
        - (nvml_index: (int, int)) -> CudaMigDevice

    Examples:
        >>> import os
        >>> os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        >>> os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'

        >>> CudaDevice.count()                     # number of NVIDIA GPUs visible to CUDA applications
        4
        >>> Device.cuda.count()                    # use alias in class `Device`
        4

        >>> CudaDevice.all()                       # all CUDA visible devices (or `Device.cuda.all()`)
        [
            CudaDevice(cuda_index=0, nvml_index=3, ...),
            CudaDevice(cuda_index=1, nvml_index=2, ...),
            ...
        ]

        >>> cuda0 = CudaDevice(cuda_index=0)       # use CUDA ordinal (or `Device.cuda(0)`)
        >>> cuda1 = CudaDevice(nvml_index=2)       # use NVML ordinal
        >>> cuda2 = CudaDevice(uuid='GPU-xxxxxx')  # use UUID string

        >>> cuda0.memory_free()                    # total free memory in bytes
        11550654464
        >>> cuda0.memory_free_human()              # total free memory in human readable format
        '11016MiB'

        >>> cuda1.as_snapshot()                    # takes an onetime snapshot of the device
        CudaDeviceSnapshot(
            real=CudaDevice(cuda_index=1, nvml_index=2, ...),
            ...
        )

    Raises:
        libnvml.NVMLError_LibraryNotFound:
            If cannot find the NVML library, usually the NVIDIA driver is not installed.
        libnvml.NVMLError_DriverNotLoaded:
            If NVIDIA driver is not loaded.
        libnvml.NVMLError_LibRmVersionMismatch:
            If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
            driver without reloading the kernel module.
        libnvml.NVMLError_NotFound:
            If the device is not found for the given NVML identifier.
        libnvml.NVMLError_InvalidArgument:
            If the NVML index is out of range.
        TypeError:
            If the number of non-None arguments is not exactly 1.
        TypeError:
            If the given NVML index is a tuple but is not consist of two integers.
        RuntimeError:
            If the index is out of range for the given ``CUDA_VISIBLE_DEVICES`` environment variable.
    """  # pylint: disable=line-too-long

    _nvml_index: int
    index: int
    nvml_index: int

    @classmethod
    def is_available(cls) -> bool:
        """Test whether there are any CUDA-capable devices available."""
        return cls.count() > 0

    @classmethod
    def count(cls) -> int:
        """The number of GPUs visible to CUDA applications."""
        try:
            return len(super().parse_cuda_visible_devices())
        except libnvml.NVMLError:
            return 0

    @classmethod
    def all(cls) -> list[CudaDevice]:  # type: ignore[override]
        """All CUDA visible devices.

        Note:
            The result could be empty if the ``CUDA_VISIBLE_DEVICES`` environment variable is invalid.
        """
        return cls.from_indices()

    @classmethod
    def from_indices(  # type: ignore[override]
        cls,
        indices: int | Iterable[int] | None = None,
    ) -> list[CudaDevice]:
        """Return a list of CUDA devices of the given CUDA indices.

        The CUDA ordinal will be enumerate from the ``CUDA_VISIBLE_DEVICES`` environment variable.

        See also for CUDA Device Enumeration:
            - `CUDA Environment Variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_
            - `CUDA Device Enumeration for MIG Device <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices>`_

        Args:
            indices (Iterable[int]):
                The indices of the GPU in CUDA ordinal, if not given, returns all visible CUDA devices.

        Returns: List[CudaDevice]
            A list of :class:`CudaDevice` of the given CUDA indices.

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            RuntimeError:
                If the index is out of range for the given ``CUDA_VISIBLE_DEVICES`` environment variable.
        """
        return super().from_cuda_indices(indices)

    def __new__(
        cls,
        cuda_index: int | None = None,
        *,
        nvml_index: int | tuple[int, int] | None = None,
        uuid: str | None = None,
    ) -> Self:
        """Create a new instance of CudaDevice.

        The type of the result is determined by the given argument.

        .. code-block:: python

            - (cuda_index: int)        -> Union[CudaDevice, CudaMigDevice]  # depending on `CUDA_VISIBLE_DEVICES`
            - (uuid: str)              -> Union[CudaDevice, CudaMigDevice]  # depending on `CUDA_VISIBLE_DEVICES`
            - (nvml_index: int)        -> CudaDevice
            - (nvml_index: (int, int)) -> CudaMigDevice

        Note: This method takes exact 1 non-None argument.

        Returns: Union[CudaDevice, CudaMigDevice]
            A :class:`CudaDevice` instance or a :class:`CudaMigDevice` instance.

        Raises:
            TypeError:
                If the number of non-None arguments is not exactly 1.
            TypeError:
                If the given NVML index is a tuple but is not consist of two integers.
            RuntimeError:
                If the index is out of range for the given ``CUDA_VISIBLE_DEVICES`` environment variable.
        """
        if nvml_index is not None and uuid is not None:
            raise TypeError(
                f'CudaDevice(cuda_index=None, nvml_index=None, uuid=None) takes 1 non-None arguments '
                f'but (cuda_index, nvml_index, uuid) = {(cuda_index, nvml_index, uuid)!r} were given',
            )

        if cuda_index is not None and nvml_index is None and uuid is None:
            cuda_visible_devices = cls.parse_cuda_visible_devices()
            if not isinstance(cuda_index, int) or not 0 <= cuda_index < len(cuda_visible_devices):
                raise RuntimeError(f'CUDA Error: invalid device ordinal: {cuda_index!r}.')
            nvml_index = cuda_visible_devices[cuda_index]

        if cls is not CudaDevice:
            # Use the subclass type if the type is explicitly specified
            return super().__new__(cls, index=nvml_index, uuid=uuid)

        # Auto subclass type inference logic goes here when `cls` is `CudaDevice` (e.g., calls `CudaDevice(...)`)
        if (nvml_index is not None and not isinstance(nvml_index, int)) or is_mig_device_uuid(uuid):
            return super().__new__(CudaMigDevice, index=nvml_index, uuid=uuid)  # type: ignore[return-value]
        return super().__new__(CudaDevice, index=nvml_index, uuid=uuid)  # type: ignore[return-value]

    def __init__(
        self,
        cuda_index: int | None = None,
        *,
        nvml_index: int | tuple[int, int] | None = None,
        uuid: str | None = None,
    ) -> None:
        """Initialize the instance created by :meth:`__new__()`.

        Raises:
            libnvml.NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            libnvml.NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            libnvml.NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            libnvml.NVMLError_NotFound:
                If the device is not found for the given NVML identifier.
            libnvml.NVMLError_InvalidArgument:
                If the NVML index is out of range.
            RuntimeError:
                If the given device is not visible to CUDA applications (i.e. not listed in the
                ``CUDA_VISIBLE_DEVICES`` environment variable or the environment variable is invalid).
        """
        if cuda_index is not None and nvml_index is None and uuid is None:
            cuda_visible_devices = self.parse_cuda_visible_devices()
            if not isinstance(cuda_index, int) or not 0 <= cuda_index < len(cuda_visible_devices):
                raise RuntimeError(f'CUDA Error: invalid device ordinal: {cuda_index!r}.')
            nvml_index = cuda_visible_devices[cuda_index]

        super().__init__(index=nvml_index, uuid=uuid)  # type: ignore[arg-type]

        if cuda_index is None:
            cuda_index = super().cuda_index
        self._cuda_index: int = cuda_index

        self._ident: tuple[Hashable, str] = ((self._cuda_index, self.index), self.uuid())

    def __repr__(self) -> str:
        """Return a string representation of the CUDA device."""
        return '{}(cuda_index={}, nvml_index={}, name="{}", total_memory={})'.format(  # noqa: UP032
            self.__class__.__name__,
            self.cuda_index,
            self.index,
            self.name(),
            self.memory_total_human(),
        )

    def __reduce__(self) -> tuple[type[CudaDevice], tuple[int]]:
        """Return state information for pickling."""
        return self.__class__, (self._cuda_index,)

    def as_snapshot(self) -> Snapshot:
        """Return a onetime snapshot of the device.

        The attributes are defined in :attr:`SNAPSHOT_KEYS`.
        """
        snapshot = super().as_snapshot()
        snapshot.cuda_index = self.cuda_index  # type: ignore[attr-defined]

        return snapshot


Device.cuda = CudaDevice
"""Shortcut for class :class:`CudaDevice`."""


class CudaMigDevice(CudaDevice, MigDevice):  # type: ignore[misc]
    """Class for CUDA devices that are MIG devices."""

    _nvml_index: tuple[int, int]  # type: ignore[assignment]
    index: tuple[int, int]  # type: ignore[assignment]
    nvml_index: tuple[int, int]  # type: ignore[assignment]


def is_mig_device_uuid(uuid: str | None) -> bool:
    """Return :data:`True` if the argument is a MIG device UUID, otherwise, return :data:`False`."""
    if isinstance(uuid, str):
        match = Device.UUID_PATTERN.match(uuid)
        if match is not None and match.group('MigMode') is not None:
            return True
    return False


def parse_cuda_visible_devices(
    cuda_visible_devices: str | None = _VALUE_OMITTED,
) -> list[int] | list[tuple[int, int]]:
    """Parse the given ``CUDA_VISIBLE_DEVICES`` value into a list of NVML device indices.

    This function is aliased by :meth:`Device.parse_cuda_visible_devices`.

    Note:
        The result could be empty if the ``CUDA_VISIBLE_DEVICES`` environment variable is invalid.

    See also for CUDA Device Enumeration:
        - `CUDA Environment Variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_
        - `CUDA Device Enumeration for MIG Device <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices>`_

    Args:
        cuda_visible_devices (Optional[str]):
            The value of the ``CUDA_VISIBLE_DEVICES`` variable. If not given, the value from the
            environment will be used. If explicitly given by :data:`None`, the ``CUDA_VISIBLE_DEVICES``
            environment variable will be unset before parsing.

    Returns: Union[List[int], List[Tuple[int, int]]]
        A list of int (physical device) or a list of tuple of two integers (MIG device) for the
        corresponding real device indices.

    Examples:
        >>> import os
        >>> os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        >>> os.environ['CUDA_VISIBLE_DEVICES'] = '6,5'
        >>> parse_cuda_visible_devices()       # parse the `CUDA_VISIBLE_DEVICES` environment variable to NVML indices
        [6, 5]

        >>> parse_cuda_visible_devices('0,4')  # pass the `CUDA_VISIBLE_DEVICES` value explicitly
        [0, 4]

        >>> parse_cuda_visible_devices('GPU-18ef14e9,GPU-849d5a8d')  # accept abbreviated UUIDs
        [5, 6]

        >>> parse_cuda_visible_devices(None)   # get all devices when the `CUDA_VISIBLE_DEVICES` environment variable unset
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> parse_cuda_visible_devices('MIG-d184f67c-c95f-5ef2-a935-195bd0094fbd')           # MIG device support (MIG UUID)
        [(0, 0)]
        >>> parse_cuda_visible_devices('MIG-GPU-3eb79704-1571-707c-aee8-f43ce747313d/13/0')  # MIG device support (GPU UUID)
        [(0, 1)]
        >>> parse_cuda_visible_devices('MIG-GPU-3eb79704/13/0')                              # MIG device support (abbreviated GPU UUID)
        [(0, 1)]

        >>> parse_cuda_visible_devices('')     # empty string
        []
        >>> parse_cuda_visible_devices('0,0')  # invalid `CUDA_VISIBLE_DEVICES` (duplicate device ordinal)
        []
        >>> parse_cuda_visible_devices('16')   # invalid `CUDA_VISIBLE_DEVICES` (device ordinal out of range)
        []
    """  # pylint: disable=line-too-long
    if cuda_visible_devices is _VALUE_OMITTED:
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', default=None)

    return _parse_cuda_visible_devices(cuda_visible_devices, format='index')


def normalize_cuda_visible_devices(cuda_visible_devices: str | None = _VALUE_OMITTED) -> str:
    """Parse the given ``CUDA_VISIBLE_DEVICES`` value and convert it into a comma-separated string of UUIDs.

    This function is aliased by :meth:`Device.normalize_cuda_visible_devices`.

    Note:
        The result could be empty string if the ``CUDA_VISIBLE_DEVICES`` environment variable is invalid.

    See also for CUDA Device Enumeration:
        - `CUDA Environment Variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_
        - `CUDA Device Enumeration for MIG Device <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices>`_

    Args:
        cuda_visible_devices (Optional[str]):
            The value of the ``CUDA_VISIBLE_DEVICES`` variable. If not given, the value from the
            environment will be used. If explicitly given by :data:`None`, the ``CUDA_VISIBLE_DEVICES``
            environment variable will be unset before parsing.

    Returns: str
        The comma-separated string (GPU UUIDs) of the ``CUDA_VISIBLE_DEVICES`` environment variable.

    Examples:
        >>> import os
        >>> os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        >>> os.environ['CUDA_VISIBLE_DEVICES'] = '6,5'
        >>> normalize_cuda_visible_devices()        # normalize the `CUDA_VISIBLE_DEVICES` environment variable to UUID strings
        'GPU-849d5a8d-610e-eeea-1fd4-81ff44a23794,GPU-18ef14e9-dec6-1d7e-1284-3010c6ce98b1'

        >>> normalize_cuda_visible_devices('4')     # pass the `CUDA_VISIBLE_DEVICES` value explicitly
        'GPU-96de99c9-d68f-84c8-424c-7c75e59cc0a0'

        >>> normalize_cuda_visible_devices('GPU-18ef14e9,GPU-849d5a8d')  # normalize abbreviated UUIDs
        'GPU-18ef14e9-dec6-1d7e-1284-3010c6ce98b1,GPU-849d5a8d-610e-eeea-1fd4-81ff44a23794'

        >>> normalize_cuda_visible_devices(None)    # get all devices when the `CUDA_VISIBLE_DEVICES` environment variable unset
        'GPU-<GPU0-UUID>,GPU-<GPU1-UUID>,...'  # all GPU UUIDs

        >>> normalize_cuda_visible_devices('MIG-d184f67c-c95f-5ef2-a935-195bd0094fbd')           # MIG device support (MIG UUID)
        'MIG-d184f67c-c95f-5ef2-a935-195bd0094fbd'
        >>> normalize_cuda_visible_devices('MIG-GPU-3eb79704-1571-707c-aee8-f43ce747313d/13/0')  # MIG device support (GPU UUID)
        'MIG-37b51284-1df4-5451-979d-3231ccb0822e'
        >>> normalize_cuda_visible_devices('MIG-GPU-3eb79704/13/0')                              # MIG device support (abbreviated GPU UUID)
        'MIG-37b51284-1df4-5451-979d-3231ccb0822e'

        >>> normalize_cuda_visible_devices('')      # empty string
        ''
        >>> normalize_cuda_visible_devices('0,0')   # invalid `CUDA_VISIBLE_DEVICES` (duplicate device ordinal)
        ''
        >>> normalize_cuda_visible_devices('16')    # invalid `CUDA_VISIBLE_DEVICES` (device ordinal out of range)
        ''
    """  # pylint: disable=line-too-long
    if cuda_visible_devices is _VALUE_OMITTED:
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', default=None)

    return ','.join(_parse_cuda_visible_devices(cuda_visible_devices, format='uuid'))


# Helper functions #################################################################################


class _PhysicalDeviceAttrs(NamedTuple):
    index: int  # type: ignore[assignment]
    name: str
    uuid: str
    support_mig_mode: bool


_PHYSICAL_DEVICE_ATTRS: OrderedDict[str, _PhysicalDeviceAttrs] | None = None
_GLOBAL_PHYSICAL_DEVICE: PhysicalDevice | None = None
_GLOBAL_PHYSICAL_DEVICE_LOCK: threading.RLock = threading.RLock()


def _get_all_physical_device_attrs() -> OrderedDict[str, _PhysicalDeviceAttrs]:
    global _PHYSICAL_DEVICE_ATTRS  # pylint: disable=global-statement

    if _PHYSICAL_DEVICE_ATTRS is not None:
        return _PHYSICAL_DEVICE_ATTRS

    with _GLOBAL_PHYSICAL_DEVICE_LOCK:
        if _PHYSICAL_DEVICE_ATTRS is None:
            _PHYSICAL_DEVICE_ATTRS = OrderedDict(
                [
                    (
                        device.uuid(),
                        _PhysicalDeviceAttrs(
                            device.index,
                            device.name(),
                            device.uuid(),
                            libnvml.nvmlCheckReturn(device.mig_mode()),
                        ),
                    )
                    for device in PhysicalDevice.all()
                ],
            )
        return _PHYSICAL_DEVICE_ATTRS


def _does_any_device_support_mig_mode(uuids: Iterable[str] | None = None) -> bool:
    physical_device_attrs = _get_all_physical_device_attrs()
    uuids = uuids or physical_device_attrs.keys()
    return any(physical_device_attrs[uuid].support_mig_mode for uuid in uuids)


@contextlib.contextmanager
def _global_physical_device(device: PhysicalDevice) -> Generator[PhysicalDevice]:
    global _GLOBAL_PHYSICAL_DEVICE  # pylint: disable=global-statement

    with _GLOBAL_PHYSICAL_DEVICE_LOCK:
        try:
            _GLOBAL_PHYSICAL_DEVICE = device
            yield _GLOBAL_PHYSICAL_DEVICE
        finally:
            _GLOBAL_PHYSICAL_DEVICE = None


def _get_global_physical_device() -> PhysicalDevice:
    with _GLOBAL_PHYSICAL_DEVICE_LOCK:
        return _GLOBAL_PHYSICAL_DEVICE  # type: ignore[return-value]


@overload
def _parse_cuda_visible_devices(
    cuda_visible_devices: str | None,
    format: Literal['index'],  # pylint: disable=redefined-builtin
) -> list[int] | list[tuple[int, int]]: ...


@overload
def _parse_cuda_visible_devices(
    cuda_visible_devices: str | None,
    format: Literal['uuid'],  # pylint: disable=redefined-builtin
) -> list[str]: ...


@functools.lru_cache()
def _parse_cuda_visible_devices(  # pylint: disable=too-many-branches,too-many-statements
    cuda_visible_devices: str | None = None,
    format: Literal['index', 'uuid'] = 'index',  # pylint: disable=redefined-builtin
) -> list[int] | list[tuple[int, int]] | list[str]:
    """The underlining implementation for :meth:`parse_cuda_visible_devices`. The result will be cached."""
    assert format in {'index', 'uuid'}

    try:
        physical_device_attrs = _get_all_physical_device_attrs()
    except libnvml.NVMLError:
        return []
    gpu_uuids = set(physical_device_attrs)

    try:
        raw_uuids = (
            subprocess.check_output(  # noqa: S603
                [
                    sys.executable,
                    '-c',
                    textwrap.dedent(
                        f"""
                        import nvitop.api.device

                        print(
                            ','.join(
                                nvitop.api.device._parse_cuda_visible_devices_to_uuids(
                                    {cuda_visible_devices!r},
                                    verbose=False,
                                ),
                            ),
                        )
                        """,
                    ),
                ],
            )
            .decode('utf-8', errors='replace')
            .strip()
            .split(',')
        )
    except subprocess.CalledProcessError:
        pass
    else:
        uuids = [
            uuid if uuid in gpu_uuids else uuid.replace('GPU', 'MIG', 1)
            for uuid in map('GPU-{}'.format, raw_uuids)
        ]
        if gpu_uuids.issuperset(uuids) and not _does_any_device_support_mig_mode(uuids):
            if format == 'uuid':
                return uuids
            return [physical_device_attrs[uuid].index for uuid in uuids]
        cuda_visible_devices = ','.join(uuids)

    if cuda_visible_devices is None:
        cuda_visible_devices = ','.join(physical_device_attrs.keys())

    devices: list[Device] = []
    presented: set[str] = set()
    use_integer_identifiers: bool | None = None

    def from_index_or_uuid(index_or_uuid: int | str) -> Device:
        nonlocal use_integer_identifiers

        if isinstance(index_or_uuid, str):
            if index_or_uuid.isdigit():
                index_or_uuid = int(index_or_uuid)
            elif Device.UUID_PATTERN.match(index_or_uuid) is None:
                raise libnvml.NVMLError_NotFound

        if use_integer_identifiers is None:
            use_integer_identifiers = isinstance(index_or_uuid, int)

        if isinstance(index_or_uuid, int) and use_integer_identifiers:
            return Device(index=index_or_uuid)
        if isinstance(index_or_uuid, str) and not use_integer_identifiers:
            return Device(uuid=index_or_uuid)
        raise ValueError('invalid identifier')

    def strip_identifier(identifier: str) -> str:
        identifier = identifier.strip()
        if len(identifier) > 0 and (
            identifier[0].isdigit()
            or (len(identifier) > 1 and identifier[0] in {'+', '-'} and identifier[1].isdigit())
        ):
            offset = 1 if identifier[0] in {'+', '-'} else 0
            while offset < len(identifier) and identifier[offset].isdigit():
                offset += 1
            identifier = identifier[:offset]
        return identifier

    for identifier in map(strip_identifier, cuda_visible_devices.split(',')):
        if identifier in presented:
            return []  # duplicate identifiers found

        try:
            device = from_index_or_uuid(identifier)
        except (ValueError, libnvml.NVMLError):
            break

        devices.append(device)
        presented.add(identifier)

    mig_devices = [device for device in devices if device.is_mig_device()]
    if len(mig_devices) > 0:
        # Got MIG devices enumerated, use the first one
        devices = mig_devices[:1]  # at most one MIG device is visible
    else:
        # All devices in `CUDA_VISIBLE_DEVICES` are physical devices
        # Check if any GPU that enables MIG mode
        devices_backup = devices.copy()
        devices = []
        for device in devices_backup:
            if device.is_mig_mode_enabled():
                # Got available MIG devices, use the first MIG device and ignore all non-MIG GPUs
                try:
                    devices = [device.mig_device(mig_index=0)]  # at most one MIG device is visible
                except libnvml.NVMLError:
                    continue  # no MIG device available on the GPU
                else:
                    break  # got one MIG device
            else:
                devices.append(device)  # non-MIG device

    if format == 'uuid':
        return [device.uuid() for device in devices]
    return [device.index for device in devices]  # type: ignore[return-value]


def _parse_cuda_visible_devices_to_uuids(
    cuda_visible_devices: str | None = _VALUE_OMITTED,
    verbose: bool = True,
) -> list[str]:
    """Parse the given ``CUDA_VISIBLE_DEVICES`` environment variable in a separate process and return a list of device UUIDs.

    The UUIDs do not have a prefix ``GPU-`` or ``MIG-``.

    Args:
        cuda_visible_devices (Optional[str]):
            The value of the ``CUDA_VISIBLE_DEVICES`` variable. If not given, the value from the
            environment will be used. If explicitly given by :data:`None`, the ``CUDA_VISIBLE_DEVICES``
            environment variable will be unset before parsing.
        verbose (bool):
            Whether to raise an exception in the subprocess if failed to parse the ``CUDA_VISIBLE_DEVICES``.

    Returns: List[str]
        A list of device UUIDs without ``GPU-`` or ``MIG-`` prefixes.

    Raises:
        libcuda.CUDAError_NotInitialized:
            If cannot found the CUDA driver libraries.
        libcuda.CUDAError:
            If failed to parse the ``CUDA_VISIBLE_DEVICES`` environment variable.
    """  # pylint: disable=line-too-long
    if cuda_visible_devices is _VALUE_OMITTED:
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', default=None)

    # Do not inherit file descriptors and handles from the parent process
    # The `fork` start method should be considered unsafe as it can lead to crashes of the subprocess
    ctx = mp.get_context('spawn')
    queue = ctx.SimpleQueue()
    try:
        parser = ctx.Process(
            target=_cuda_visible_devices_parser,
            args=(cuda_visible_devices, queue, verbose),
            name='`CUDA_VISIBLE_DEVICES` parser',
            daemon=True,
        )
        parser.start()
        parser.join()
    finally:
        parser.kill()
    result = queue.get()

    if isinstance(result, Exception):
        raise result
    return result


def _cuda_visible_devices_parser(
    cuda_visible_devices: str | None,
    queue: mp.SimpleQueue,
    verbose: bool = True,
) -> None:
    try:
        if cuda_visible_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)

        # pylint: disable=no-member
        try:
            libcuda.cuInit()
        except (
            libcuda.CUDAError_NoDevice,
            libcuda.CUDAError_InvalidDevice,
            libcuda.CUDAError_SystemDriverMismatch,
            libcuda.CUDAError_CompatNotSupportedOnDevice,
        ):
            queue.put([])
            raise

        count = libcuda.cuDeviceGetCount()
        uuids = [libcuda.cuDeviceGetUuid(libcuda.cuDeviceGet(i)) for i in range(count)]
    except Exception as ex:  # pylint: disable=broad-except
        queue.put(ex)
        if verbose:
            raise
    else:
        queue.put(uuids)
        return
    finally:
        # Ensure non-empty queue
        queue.put(libcuda.CUDAError_NotInitialized())  # pylint: disable=no-member
