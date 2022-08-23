# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

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

import contextlib
import multiprocessing as mp
import os
import re
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Type, Union

from cachetools.func import ttl_cache

from nvitop.core import libcuda, libnvml
from nvitop.core.process import GpuProcess
from nvitop.core.utils import NA, NaType, Snapshot, boolify, bytes2human, memoize_when_activated


__all__ = ['Device', 'PhysicalDevice', 'MigDevice', 'CudaDevice', 'CudaMigDevice']


MemoryInfo = NamedTuple(
    'MemoryInfo',  # in bytes
    [('total', Union[int, NaType]), ('free', Union[int, NaType]), ('used', Union[int, NaType])],
)
ClockInfos = NamedTuple(
    'ClockInfos',  # in MHz
    [
        ('graphics', Union[int, NaType]),
        ('sm', Union[int, NaType]),
        ('memory', Union[int, NaType]),
        ('video', Union[int, NaType]),
    ],
)
ClockSpeedInfos = NamedTuple('ClockSpeedInfos', [('current', ClockInfos), ('max', ClockInfos)])
UtilizationRates = NamedTuple(
    'UtilizationRates',  # in percentage
    [
        ('gpu', Union[int, NaType]),
        ('memory', Union[int, NaType]),
        ('encoder', Union[int, NaType]),
        ('decoder', Union[int, NaType]),
    ],
)

_VALUE_OMITTED = object()


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
    UUID_PATTERN = re.compile(
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

    GPU_PROCESS_CLASS = GpuProcess
    cuda = None  # defined in below
    """Shortcut for class :class:`CudaDevice`."""

    @classmethod
    def is_available(cls) -> bool:
        """Returns whether there are any devices and the NVML library is successfully loaded."""

        try:
            return cls.count() > 0
        except libnvml.NVMLError:
            return False

    @staticmethod
    def driver_version() -> Union[str, NaType]:
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
    def cuda_driver_version() -> Union[str, NaType]:
        """The maximum CUDA version supported by the NVIDIA display driver.
        This can be different from the version of the CUDA runtime.

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
                return '{}.{}'.format(major, minor)
            return '{}.{}.{}'.format(major, minor, revision)
        return NA

    max_cuda_version = cuda_version = cuda_driver_version  # alias for backward compatibility

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
    def all(cls) -> List['PhysicalDevice']:
        """Returns a list of all physical devices in the system."""

        return cls.from_indices()

    @classmethod
    def from_indices(
        cls, indices: Optional[Union[int, Iterable[Union[int, Tuple[int, int]]]]] = None
    ) -> List[Union['PhysicalDevice', 'MigDevice']]:
        """Returns a list of devices of the given indices.

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

        return list(map(cls, indices))

    @staticmethod
    def from_cuda_visible_devices() -> List['CudaDevice']:
        """Returns a list of all CUDA visible devices.
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

        cuda_devices = []
        for cuda_index, device_index in enumerate(visible_device_indices):
            cuda_devices.append(CudaDevice(cuda_index, nvml_index=device_index))

        return cuda_devices

    cuda_all = from_cuda_visible_devices

    @staticmethod
    def from_cuda_indices(
        cuda_indices: Optional[Union[int, Iterable[int]]] = None
    ) -> List['CudaDevice']:
        """Returns a list of CUDA devices of the given CUDA indices.
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
                raise RuntimeError('CUDA Error: invalid device ordinal: {!r}.'.format(cuda_index))
            device = cuda_devices[cuda_index]
            devices.append(device)

        return devices

    @staticmethod
    def parse_cuda_visible_devices(
        cuda_visible_devices: Optional[str] = _VALUE_OMITTED,
    ) -> Union[List[int], List[Tuple[int, int]]]:
        """Parses the given ``CUDA_VISIBLE_DEVICES`` value into NVML device indices.

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
            A list of int (physical device) or a list of tuple of two ints (MIG device) for the
            corresponding real device indices.
        """  # pylint: disable=line-too-long

        if cuda_visible_devices is _VALUE_OMITTED:
            cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', default=None)

        return Device._parse_cuda_visible_devices(cuda_visible_devices)

    @staticmethod
    @ttl_cache(ttl=300.0)
    def _parse_cuda_visible_devices(  # pylint: disable=too-many-branches,too-many-statements
        cuda_visible_devices: Optional[str] = None,
    ) -> Union[List['PhysicalDevice'], List['MigDevice']]:
        """The underlining implementation for :meth:`parse_cuda_visible_devices`. The result will be cached."""

        try:
            physical_device_attrs = _get_all_physical_device_attrs()
        except libnvml.NVMLError:
            return []
        gpu_uuids = set(physical_device_attrs)

        try:
            raw_uuids = parse_cuda_visible_devices_to_uuids(cuda_visible_devices, verbose=False)
        except libcuda.CUDAError:
            pass
        else:
            uuids = [
                uuid if uuid in gpu_uuids else uuid.replace('GPU', 'MIG', 1)
                for uuid in map('GPU-{}'.format, raw_uuids)
            ]
            if gpu_uuids.issuperset(uuids) and not _does_any_device_support_mig_mode(uuids):
                return [physical_device_attrs[uuid].index for uuid in uuids]
            cuda_visible_devices = ','.join(uuids)

        if cuda_visible_devices is None:
            cuda_visible_devices = ','.join(physical_device_attrs.keys())

        def from_index_or_uuid(index_or_uuid: Union[int, str]) -> 'Device':
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

        devices = []
        presented = set()
        use_integer_identifiers = None
        for identifier in map(str.strip, cuda_visible_devices.split(',')):
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
                        devices = [
                            device.mig_device(mig_index=0)  # at most one MIG device is visible
                        ]
                    except libnvml.NVMLError:
                        continue  # no MIG device available on the GPU
                    else:
                        break  # got one MIG device
                else:
                    devices.append(device)  # non-MIG device

        return [device.index for device in devices]

    def __new__(
        cls,
        index: Optional[Union[int, Tuple[int, int], str]] = None,
        *,
        uuid: Optional[str] = None,
        bus_id: Optional[str] = None
    ) -> 'Device':
        """Creates a new instance of Device. The type of the result is determined by the given argument.

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
                'Device(index=None, uuid=None, bus_id=None) takes 1 non-None arguments '
                'but (index, uuid, bus_id) = {!r} were given'.format((index, uuid, bus_id))
            )

        if cls is not Device:
            return super().__new__(cls)

        match = None
        if isinstance(index, str):
            match = cls.UUID_PATTERN.match(index)
            if match is not None:  # passed by UUID
                index, uuid = None, index
        elif isinstance(uuid, str):
            match = cls.UUID_PATTERN.match(uuid)

        if index is not None:
            if not isinstance(index, int):
                if not (
                    isinstance(index, tuple)
                    and len(index) == 2
                    and isinstance(index[0], int)
                    and isinstance(index[1], int)
                ):
                    raise TypeError(
                        'index for MIG device must be a tuple of 2 integers '
                        'but index = {!r} was given'.format((index))
                    )
                return super().__new__(MigDevice)
        elif uuid is not None:
            if match is not None and match.group('MigMode') is not None:
                return super().__new__(MigDevice)
        return super().__new__(PhysicalDevice)

    def __init__(
        self,
        index: Optional[Union[int, str]] = None,
        *,
        uuid: Optional[str] = None,
        bus_id: Optional[str] = None
    ) -> None:
        """Initializes the instance created by ``__new__()``.

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

        index, uuid, bus_id = [
            arg.encode() if isinstance(arg, str) else arg for arg in (index, uuid, bus_id)
        ]

        self._name = NA
        self._uuid = NA
        self._bus_id = NA
        self._memory_total = NA
        self._memory_total_human = NA
        self._is_mig_device = None
        self._cuda_index = None

        if index is not None:
            self._nvml_index = index
            try:
                self._handle = libnvml.nvmlQuery(
                    'nvmlDeviceGetHandleByIndex', index, ignore_errors=False
                )
            except libnvml.NVMLError_GpuIsLost:
                self._handle = None
        else:
            try:
                if uuid is not None:
                    self._handle = libnvml.nvmlQuery(
                        'nvmlDeviceGetHandleByUUID', uuid, ignore_errors=False
                    )
                else:
                    self._handle = libnvml.nvmlQuery(
                        'nvmlDeviceGetHandleByPciBusId', bus_id, ignore_errors=False
                    )
            except libnvml.NVMLError_GpuIsLost:
                self._handle = None
                self._nvml_index = NA
            else:
                self._nvml_index = libnvml.nvmlQuery('nvmlDeviceGetIndex', self._handle)

        self._max_clock_infos = ClockInfos(graphics=NA, sm=NA, memory=NA, video=NA)
        self._timestamp = 0
        self._lock = threading.RLock()

        self._ident = (self.index, self.uuid())
        self._hash = None

    def __str__(self) -> str:
        return '{}(index={}, name="{}", total_memory={})'.format(
            self.__class__.__name__, self.index, self.name(), self.memory_total_human()
        )

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Device):
            return NotImplemented
        return self._ident == other._ident

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self._ident)
        return self._hash

    def __getattr__(self, name: str) -> Union[Any, Callable[..., Any]]:
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
            return super().__getattr__(name)
        except AttributeError:
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

            @ttl_cache(ttl=1.0)
            def attribute(*args, **kwargs):
                try:
                    return libnvml.nvmlQuery(
                        func, self._handle, *args, **kwargs, ignore_errors=False
                    )
                except libnvml.NVMLError_NotSupported:
                    return NA

            attribute.__name__ = name
            attribute.__qualname__ = '{}.{}'.format(self.__class__.__name__, name)
            setattr(self, name, attribute)
            return attribute

    def __reduce__(self) -> Tuple[Type['Device'], Tuple[Union[int, Tuple[int, int]]]]:
        return self.__class__, (self._nvml_index,)

    @property
    def index(self) -> Union[int, Tuple[int, int]]:
        """The NVML index of the device.

        Returns: Union[int, Tuple[int, int]]
            Returns an int for physical device and tuple of two integers for MIG device.
        """

        return self._nvml_index

    @property
    def nvml_index(self) -> Union[int, Tuple[int, int]]:
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

        return self._nvml_index  # will be overridden in MigDevice

    @property
    def handle(self) -> libnvml.c_nvmlDevice_t:
        """The NVML device handle."""

        return self._handle

    @property
    def cuda_index(self) -> int:
        """The CUDA device index. The value will be evaluated on the first call.

        Raises:
            RuntimeError:
                If the current device is not visible to CUDA applications (i.e. not listed in the
                ``CUDA_VISIBLE_DEVICES`` environment variable or the environment variable is invalid).
        """

        if self._cuda_index is None:
            visible_device_indices = self.parse_cuda_visible_devices()
            try:
                cuda_index = visible_device_indices.index(self.index)
            except ValueError as ex:
                raise RuntimeError(
                    'CUDA Error: Device(index={}) is not visible to CUDA applications'.format(
                        self.index
                    )
                ) from ex
            else:
                self._cuda_index = cuda_index

        return self._cuda_index

    def name(self) -> Union[str, NaType]:
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

    def uuid(self) -> Union[str, NaType]:
        """This value is the globally unique immutable alphanumeric identifier of the GPU. It does
        not correspond to any physical label on the board.

        Returns: Union[str, NaType]
            The UUID of the device, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=name
        """

        if self._uuid is NA:
            self._uuid = libnvml.nvmlQuery('nvmlDeviceGetUUID', self.handle)
        return self._uuid

    def bus_id(self) -> Union[str, NaType]:
        """PCI bus ID as "domain:bus:device.function", in hex.

        Returns: Union[str, NaType]
            The PCI bus ID of the device, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=pci.bus_id
        """

        if self._bus_id is NA:
            self._bus_id = libnvml.nvmlQuery(
                lambda handle: libnvml.nvmlDeviceGetPciInfo(handle).busId, self.handle
            )
        return self._bus_id

    def serial(self) -> Union[str, NaType]:
        """This number matches the serial number physically printed on each board. It is a globally
        unique immutable alphanumeric value.

        Returns: Union[str, NaType]
            The serial number of the device, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=serial
        """

        return libnvml.nvmlQuery('nvmlDeviceGetSerial', self.handle)

    @memoize_when_activated
    @ttl_cache(ttl=1.0)
    def memory_info(self) -> MemoryInfo:  # in bytes
        """Returns a named tuple with memory information (in bytes) for the device.

        Returns: MemoryInfo(total, free, used)
            A named tuple with memory information, the item could be :const:`nvitop.NA` when not applicable.
        """

        memory_info = libnvml.nvmlQuery('nvmlDeviceGetMemoryInfo', self.handle)
        if libnvml.nvmlCheckReturn(memory_info):
            return MemoryInfo(total=memory_info.total, free=memory_info.free, used=memory_info.used)
        return MemoryInfo(total=NA, free=NA, used=NA)

    def memory_total(self) -> Union[int, NaType]:  # in bytes
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

    def memory_used(self) -> Union[int, NaType]:  # in bytes
        """Total memory allocated by active contexts in bytes.

        Returns: Union[int, NaType]
            Total memory allocated by active contexts in bytes, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=memory.used
        """

        return self.memory_info().used

    def memory_free(self) -> Union[int, NaType]:  # in bytes
        """Total free memory in bytes.

        Returns: Union[int, NaType]
            Total free memory in bytes, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=memory.free
        """

        return self.memory_info().free

    def memory_total_human(self) -> Union[str, NaType]:  # in human readable
        """Total installed GPU memory in human readable format.

        Returns: Union[str, NaType]
            Total installed GPU memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """

        if self._memory_total_human is NA:
            self._memory_total_human = bytes2human(self.memory_total())
        return self._memory_total_human

    def memory_used_human(self) -> Union[str, NaType]:  # in human readable
        """Total memory allocated by active contexts in human readable format.

        Returns: Union[int, NaType]
            Total memory allocated by active contexts in human readable format, or :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long

        return bytes2human(self.memory_used())

    def memory_free_human(self) -> Union[str, NaType]:  # in human readable
        """Total free memory in human readable format.

        Returns: Union[int, NaType]
            Total free memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """

        return bytes2human(self.memory_free())

    def memory_percent(self) -> Union[float, NaType]:  # in percentage
        """The percentage of used memory over total memory (0 <= p <= 100).

        Returns: Union[float, NaType]
            The percentage of used memory over total memory, or :const:`nvitop.NA` when not applicable.
        """

        memory_info = self.memory_info()
        if libnvml.nvmlCheckReturn(memory_info.used, int) and libnvml.nvmlCheckReturn(
            memory_info.total, int
        ):
            return round(100.0 * memory_info.used / memory_info.total, 1)
        return NA

    def memory_usage(self) -> str:  # string of used memory over total memory (in human readable)
        """The used memory over total memory in human readable format.

        Returns: str
            The used memory over total memory in human readable format, or :const:`'N/A / N/A'` when not applicable.
        """  # pylint: disable=line-too-long

        return '{} / {}'.format(self.memory_used_human(), self.memory_total_human())

    @memoize_when_activated
    @ttl_cache(ttl=1.0)
    def bar1_memory_info(self) -> MemoryInfo:  # in bytes
        """Returns a named tuple with BAR1 memory information (in bytes) for the device.

        Returns: MemoryInfo(total, free, used)
            A named tuple with BAR1 memory information, the item could be :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long

        memory_info = libnvml.nvmlQuery('nvmlDeviceGetBAR1MemoryInfo', self.handle)
        if libnvml.nvmlCheckReturn(memory_info):
            return MemoryInfo(
                total=memory_info.bar1Total, free=memory_info.bar1Free, used=memory_info.bar1Used
            )
        return MemoryInfo(total=NA, free=NA, used=NA)

    def bar1_memory_total(self) -> Union[int, NaType]:  # in bytes
        """Total BAR1 memory in bytes.

        Returns: Union[int, NaType]
            Total BAR1 memory in bytes, or :const:`nvitop.NA` when not applicable.
        """

        return self.bar1_memory_info().total

    def bar1_memory_used(self) -> Union[int, NaType]:  # in bytes
        """Total used BAR1 memory in bytes.

        Returns: Union[int, NaType]
            Total used BAR1 memory in bytes, or :const:`nvitop.NA` when not applicable.
        """

        return self.bar1_memory_info().used

    def bar1_memory_free(self) -> Union[int, NaType]:  # in bytes
        """Total free BAR1 memory in bytes.

        Returns: Union[int, NaType]
            Total free BAR1 memory in bytes, or :const:`nvitop.NA` when not applicable.
        """

        return self.bar1_memory_info().free

    def bar1_memory_total_human(self) -> Union[str, NaType]:  # in human readable
        """Total BAR1 memory in human readable format.

        Returns: Union[int, NaType]
            Total BAR1 memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """

        return bytes2human(self.bar1_memory_total())

    def bar1_memory_used_human(self) -> Union[str, NaType]:  # in human readable
        """Total used BAR1 memory in human readable format.

        Returns: Union[int, NaType]
            Total used BAR1 memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """

        return bytes2human(self.bar1_memory_used())

    def bar1_memory_free_human(self) -> Union[str, NaType]:  # in human readable
        """Total free BAR1 memory in human readable format.

        Returns: Union[int, NaType]
            Total free BAR1 memory in human readable format, or :const:`nvitop.NA` when not applicable.
        """

        return bytes2human(self.bar1_memory_free())

    def bar1_memory_percent(self) -> Union[float, NaType]:  # in percentage
        """The percentage of used BAR1 memory over total BAR1 memory (0 <= p <= 100).

        Returns: Union[float, NaType]
            The percentage of used BAR1 memory over total BAR1 memory, or :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long

        memory_info = self.bar1_memory_info()
        if libnvml.nvmlCheckReturn(memory_info.used, int) and libnvml.nvmlCheckReturn(
            memory_info.total, int
        ):
            return round(100.0 * memory_info.used / memory_info.total, 1)
        return NA

    def bar1_memory_usage(self) -> str:  # in human readable
        """The used BAR1 memory over total BAR1 memory in human readable format.

        Returns: str
            The used BAR1 memory over total BAR1 memory in human readable format, or :const:`'N/A / N/A'` when not applicable.
        """  # pylint: disable=line-too-long

        return '{} / {}'.format(self.bar1_memory_used_human(), self.bar1_memory_total_human())

    @memoize_when_activated
    @ttl_cache(ttl=1.0)
    def utilization_rates(self) -> UtilizationRates:  # in percentage
        """Returns a named tuple with GPU utilization rates (in percentage) for the device.

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

    def gpu_utilization(self) -> Union[int, NaType]:  # in percentage
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

    def memory_utilization(self) -> Union[float, NaType]:  # in percentage
        """Percent of time over the past sample period during which global (device) memory was being read or written.
        The sample period may be between 1 second and 1/6 second depending on the product.

        Returns: Union[int, NaType]
            The memory bandwidth utilization rate of the GPU in percentage, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=utilization.memory
        """  # pylint: disable=line-too-long

        return self.utilization_rates().memory

    def encoder_utilization(self) -> Union[float, NaType]:  # in percentage
        """The encoder utilization rate  in percentage.

        Returns: Union[int, NaType]
            The encoder utilization rate  in percentage, or :const:`nvitop.NA` when not applicable.
        """

        return self.utilization_rates().encoder

    def decoder_utilization(self) -> Union[float, NaType]:  # in percentage\
        """The decoder utilization rate  in percentage.

        Returns: Union[int, NaType]
            The decoder utilization rate  in percentage, or :const:`nvitop.NA` when not applicable.
        """

        return self.utilization_rates().decoder

    @memoize_when_activated
    @ttl_cache(ttl=5.0)
    def clock_infos(self) -> ClockInfos:  # in MHz
        """Returns a named tuple with current clock speeds (in MHz) for the device.

        Returns: ClockInfos(graphics, sm, memory, video)
            A named tuple with current clock speeds (in MHz) for the device, the item could be :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long

        return ClockInfos(
            graphics=libnvml.nvmlQuery(
                'nvmlDeviceGetClockInfo', self.handle, libnvml.NVML_CLOCK_GRAPHICS
            ),
            sm=libnvml.nvmlQuery('nvmlDeviceGetClockInfo', self.handle, libnvml.NVML_CLOCK_SM),
            memory=libnvml.nvmlQuery('nvmlDeviceGetClockInfo', self.handle, libnvml.NVML_CLOCK_MEM),
            video=libnvml.nvmlQuery(
                'nvmlDeviceGetClockInfo', self.handle, libnvml.NVML_CLOCK_VIDEO
            ),
        )

    clocks = clock_infos

    @memoize_when_activated
    @ttl_cache(ttl=5.0)
    def max_clock_infos(self) -> ClockInfos:  # in MHz
        """Returns a named tuple with maximum clock speeds (in MHz) for the device.

        Returns: ClockInfos(graphics, sm, memory, video)
            A named tuple with maximum clock speeds (in MHz) for the device, the item could be :const:`nvitop.NA` when not applicable.
        """  # pylint: disable=line-too-long

        clock_infos = self._max_clock_infos._asdict()
        for name, clock in clock_infos.items():
            if clock is NA:
                clock_type = getattr(
                    libnvml, 'NVML_CLOCK_{}'.format(name.replace('memory', 'mem').upper())
                )
                clock = libnvml.nvmlQuery('nvmlDeviceGetMaxClockInfo', self.handle, clock_type)
                clock_infos[name] = clock
        self._max_clock_infos = ClockInfos(**clock_infos)
        return self._max_clock_infos

    max_clocks = max_clock_infos

    def clock_speed_infos(self) -> ClockSpeedInfos:  # in MHz
        """Returns a named tuple with the current and the maximum clock speeds (in MHz) for the device.

        Returns: ClockSpeedInfos(current, max)
            A named tuple with the current and the maximum clock speeds (in MHz) for the device.
        """

        return ClockSpeedInfos(current=self.clock_infos(), max=self.max_clock_infos())

    def graphics_clock(self) -> Union[int, NaType]:  # in MHz
        """Current frequency of graphics (shader) clock in MHz.

        Returns: Union[int, NaType]
            The current frequency of graphics (shader) clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.current.graphics
        """  # pylint: disable=line-too-long

        return self.clock_infos().graphics

    def sm_clock(self) -> Union[int, NaType]:  # in MHz
        """Current frequency of SM (Streaming Multiprocessor) clock in MHz.

        Returns: Union[int, NaType]
            The current frequency of SM (Streaming Multiprocessor) clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.current.sm
        """  # pylint: disable=line-too-long

        return self.clock_infos().sm

    def memory_clock(self) -> Union[int, NaType]:  # in MHz
        """Current frequency of memory clock in MHz.

        Returns: Union[int, NaType]
            The current frequency of memory clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.current.memory
        """

        return self.clock_infos().memory

    def video_clock(self) -> Union[int, NaType]:  # in MHz
        """Current frequency of video encoder/decoder clock in MHz.

        Returns: Union[int, NaType]
            The current frequency of video encoder/decoder clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.current.video
        """  # pylint: disable=line-too-long

        return self.clock_infos().video

    def max_graphics_clock(self) -> Union[int, NaType]:  # in MHz
        """Maximum frequency of graphics (shader) clock in MHz.

        Returns: Union[int, NaType]
            The maximum frequency of graphics (shader) clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.max.graphics
        """  # pylint: disable=line-too-long

        return self.clock_infos().graphics

    def max_sm_clock(self) -> Union[int, NaType]:  # in MHz
        """Maximum frequency of SM (Streaming Multiprocessor) clock in MHz.

        Returns: Union[int, NaType]
            The maximum frequency of SM (Streaming Multiprocessor) clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.max.sm
        """  # pylint: disable=line-too-long

        return self.clock_infos().sm

    def max_memory_clock(self) -> Union[int, NaType]:  # in MHz
        """Maximum frequency of memory clock in MHz.

        Returns: Union[int, NaType]
            The maximum frequency of memory clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.max.memory
        """

        return self.clock_infos().memory

    def max_video_clock(self) -> Union[int, NaType]:  # in MHz
        """Maximum frequency of video encoder/decoder clock in MHz.

        Returns: Union[int, NaType]
            The maximum frequency of video encoder/decoder clock in MHz, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=clocks.max.video
        """  # pylint: disable=line-too-long

        return self.clock_infos().video

    @ttl_cache(ttl=5.0)
    def fan_speed(self) -> Union[int, NaType]:  # in percentage
        """The fan speed value is the percent of the product's maximum noise tolerance fan speed that
        the device's fan is currently intended to run at. This value may exceed 100% in certain cases.
        Note: The reported speed is the intended fan speed. If the fan is physically blocked and unable
        to spin, this output will not match the actual fan speed. Many parts do not report fan speeds
        because they rely on cooling via fans in the surrounding enclosure.

        Returns: Union[int, NaType]
            The fan speed value in percentage, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=fan.speed
        """

        return libnvml.nvmlQuery('nvmlDeviceGetFanSpeed', self.handle)

    @ttl_cache(ttl=5.0)
    def temperature(self) -> Union[int, NaType]:  # in Celsius
        """Core GPU temperature. in degrees C.

        Returns: Union[int, NaType]
            The core GPU temperature in Celsius degrees, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=temperature.gpu
        """

        return libnvml.nvmlQuery(
            'nvmlDeviceGetTemperature', self.handle, libnvml.NVML_TEMPERATURE_GPU
        )

    @memoize_when_activated
    @ttl_cache(ttl=5.0)
    def power_usage(self) -> Union[int, NaType]:  # in milliwatts (mW)
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
    @ttl_cache(ttl=60.0)
    def power_limit(self) -> Union[int, NaType]:  # in milliwatts (mW)
        """The software power limit in milliwatts. Set by software like nvidia-smi.

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
            power_usage = '{}W'.format(round(power_usage / 1000.0))
        if libnvml.nvmlCheckReturn(power_limit, int):
            power_limit = '{}W'.format(round(power_limit / 1000.0))
        return '{} / {}'.format(power_usage, power_limit)

    @ttl_cache(ttl=60.0)
    def display_active(self) -> Union[str, NaType]:
        """A flag that indicates whether a display is initialized on the GPU's (e.g. memory is
        allocated on the device for display). Display can be active even when no monitor is
        physically attached. "Enabled" indicates an active display. "Disabled" indicates otherwise.

        Returns: Union[str, NaType]
            - :const:`'Disabled'`: if not an active display device.
            - :const:`'Enabled'`: if an active display device.
            - :const:`nvitop.NA`: if not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=display_active
        """

        return {0: 'Disabled', 1: 'Enabled'}.get(
            libnvml.nvmlQuery('nvmlDeviceGetDisplayActive', self.handle), NA
        )

    @ttl_cache(ttl=60.0)
    def display_mode(self) -> Union[str, NaType]:
        """A flag that indicates whether a physical display (e.g. monitor) is currently connected to
        any of the GPU's connectors. "Enabled" indicates an attached display. "Disabled" indicates
        otherwise.

        Returns: Union[str, NaType]
            - :const:`'Disabled'`: if the display mode is disabled.
            - :const:`'Enabled'`: if the display mode is enabled.
            - :const:`nvitop.NA`: if not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=display_mode
        """

        return {0: 'Disabled', 1: 'Enabled'}.get(
            libnvml.nvmlQuery('nvmlDeviceGetDisplayMode', self.handle), NA
        )

    @ttl_cache(ttl=60.0)
    def current_driver_model(self) -> Union[str, NaType]:
        """The driver model currently in use. Always "N/A" on Linux. On Windows, the TCC (WDM)
        and WDDM driver models are supported. The TCC driver model is optimized for compute
        applications. I.E. kernel launch times will be quicker with TCC. The WDDM driver model
        is designed for graphics applications and is not recommended for compute applications.
        Linux does not support multiple driver models, and will always have the value of "N/A".

        Returns: Union[str, NaType]
            - :const:`'WDDM'`: for WDDM driver model on Windows.
            - :const:`'WDM'`: for TTC (WDM) driver model on Windows.
            - :const:`nvitop.NA`: if not applicable, e.g. on Linux.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=driver_model.current
        """

        return {libnvml.NVML_DRIVER_WDDM: 'WDDM', libnvml.NVML_DRIVER_WDM: 'WDM'}.get(
            libnvml.nvmlQuery('nvmlDeviceGetCurrentDriverModel', self.handle), NA
        )

    driver_model = current_driver_model

    @ttl_cache(ttl=60.0)
    def persistence_mode(self) -> Union[str, NaType]:
        """A flag that indicates whether persistence mode is enabled for the GPU. Value is either
        "Enabled" or "Disabled". When persistence mode is enabled the NVIDIA driver remains loaded
        even when no active clients, such as X11 or nvidia-smi, exist. This minimizes the driver
        load latency associated with running dependent apps, such as CUDA programs. Linux only.

        Returns: Union[str, NaType]
            - :const:`'Disabled'`: if the persistence mode is disabled.
            - :const:`'Enabled'`: if the persistence mode is enabled.
            - :const:`nvitop.NA`: if not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=persistence_mode
        """

        return {0: 'Disabled', 1: 'Enabled'}.get(
            libnvml.nvmlQuery('nvmlDeviceGetPersistenceMode', self.handle), NA
        )

    @ttl_cache(ttl=5.0)
    def performance_state(self) -> Union[str, NaType]:
        """The current performance state for the GPU. States range from P0 (maximum performance) to
        P12 (minimum performance).

        Returns: Union[str, NaType]
            The current performance state in format ``P<int>``, or :const:`nvitop.NA` when not applicable.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=pstate
        """

        performance_state = libnvml.nvmlQuery('nvmlDeviceGetPerformanceState', self.handle)
        if libnvml.nvmlCheckReturn(performance_state, int):
            performance_state = 'P' + str(performance_state)
        return performance_state

    @ttl_cache(ttl=5.0)
    def total_volatile_uncorrected_ecc_errors(self) -> Union[int, NaType]:
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

    @ttl_cache(ttl=60.0)
    def compute_mode(self) -> Union[str, NaType]:
        """The compute mode flag indicates whether individual or multiple compute applications may
        run on the GPU.

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

    def is_mig_device(self) -> bool:
        """Returns whether or not the device is a MIG device."""

        if self._is_mig_device is None:
            is_mig_device = libnvml.nvmlQuery(
                'nvmlDeviceIsMigDeviceHandle',
                self.handle,
                default=False,
                ignore_function_not_found=True,
            )
            self._is_mig_device = bool(is_mig_device)  # nvmlDeviceIsMigDeviceHandle returns c_uint
        return self._is_mig_device

    @ttl_cache(ttl=60.0)
    def mig_mode(self) -> Union[str, NaType]:
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

        mig_mode = libnvml.nvmlQuery(
            'nvmlDeviceGetMigMode', self.handle, default=(NA, NA), ignore_function_not_found=True
        )[0]
        return {0: 'Disabled', 1: 'Enabled'}.get(mig_mode, NA)

    def is_mig_mode_enabled(self) -> bool:
        """Returns whether the MIG mode is enabled on the device. Returns :data:`False` if MIG mode is
        disabled or the device does not support MIG mode.
        """

        return boolify(self.mig_mode())

    def max_mig_device_count(self) -> int:
        """Returns the maximum number of MIG instances the device supports. Returns 0 if the device
        does not support MIG mode.
        """

        return 0  # implemented in PhysicalDevice

    def mig_devices(self) -> List['MigDevice']:
        """Returns a list of children MIG devices of the current device. Returns an empty list if
        the MIG mode is disabled or the device does not support MIG mode.
        """

        return []  # implemented in PhysicalDevice

    def is_leaf_device(self) -> bool:
        """Returns :data:`True` if the device is a physical device with MIG mode disabled or a MIG device.
        Otherwise returns :data:`False` if the device is a physical device with MIG mode enabled.
        """

        return self.is_mig_device() or not self.is_mig_mode_enabled()

    def to_leaf_devices(self) -> List[Union['PhysicalDevice', 'MigDevice', 'CudaDevice']]:
        """Returns a list of leaf devices. Note that a CUDA device is always a leaf device."""

        if isinstance(self, CudaDevice) or self.is_leaf_device():
            return [self]
        return self.mig_devices()

    @ttl_cache(ttl=2.0)
    def processes(self) -> Dict[int, GpuProcess]:
        """Returns a dictionary of processes running on the GPU.

        Returns: Dict[int, GpuProcess]
            A dictionary mapping PID to GPU process instance.
        """

        processes = {}

        for type, func in (  # pylint: disable=redefined-builtin
            ('C', 'nvmlDeviceGetComputeRunningProcesses'),
            ('G', 'nvmlDeviceGetGraphicsRunningProcesses'),
        ):
            for p in libnvml.nvmlQuery(func, self.handle, default=()):
                proc = processes[p.pid] = self.GPU_PROCESS_CLASS(
                    pid=p.pid,
                    device=self,
                    gpu_memory=(
                        p.usedGpuMemory
                        if isinstance(p.usedGpuMemory, int)
                        else NA  # used GPU memory is `N/A` in Windows Display Driver Model (WDDM)
                    ),
                    gpu_instance_id=getattr(p, 'gpuInstanceId', 0xFFFFFFFF),
                    compute_instance_id=getattr(p, 'computeInstanceId', 0xFFFFFFFF),
                )
                proc.type = proc.type + type

        if len(processes) > 0:
            samples = libnvml.nvmlQuery(
                'nvmlDeviceGetProcessUtilization', self.handle, self._timestamp, default=()
            )
            self._timestamp = max(min((s.timeStamp for s in samples), default=0) - 500000, 0)
            for s in samples:
                try:
                    processes[s.pid].set_gpu_utilization(s.smUtil, s.memUtil, s.encUtil, s.decUtil)
                except KeyError:
                    pass

        return processes

    def as_snapshot(self) -> Snapshot:
        """Returns a onetime snapshot of the device. The attributes are defined in :attr:`SNAPSHOT_KEYS`."""

        with self.oneshot():
            return Snapshot(
                real=self,
                index=self.index,
                physical_index=self.physical_index,
                **{key: getattr(self, key)() for key in self.SNAPSHOT_KEYS},
            )

    SNAPSHOT_KEYS = [
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
        'display_active',
        'display_mode',
        'current_driver_model',
        'persistence_mode',
        'performance_state',
        'total_volatile_uncorrected_ecc_errors',
        'compute_mode',
        'mig_mode',
    ]

    # Modified from psutil (https://github.com/giampaolo/psutil)
    @contextlib.contextmanager
    def oneshot(self):
        """Utility context manager which considerably speeds up the retrieval of multiple device
        information at the same time.

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
        """

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
                    self.memory_info.cache_activate(self)
                    self.bar1_memory_info.cache_activate(self)
                    self.utilization_rates.cache_activate(self)
                    self.clock_infos.cache_activate(self)
                    self.max_clock_infos.cache_activate(self)
                    self.power_usage.cache_activate(self)
                    self.power_limit.cache_activate(self)
                    yield
                finally:
                    self.memory_info.cache_deactivate(self)
                    self.bar1_memory_info.cache_deactivate(self)
                    self.utilization_rates.cache_deactivate(self)
                    self.clock_infos.cache_deactivate(self)
                    self.max_clock_infos.cache_deactivate(self)
                    self.power_usage.cache_deactivate(self)
                    self.power_limit.cache_deactivate(self)


class PhysicalDevice(Device):
    """Class for physical devices. This is the real GPU installed in the system."""

    @property
    def physical_index(self) -> int:
        """Zero based index of the GPU. Can change at each boot.

        Command line equivalent:

        .. code:: bash

            nvidia-smi --id=<IDENTIFIER> --format=csv,noheader,nounits --query-gpu=index
        """

        return self._nvml_index

    @ttl_cache(ttl=60.0)
    def max_mig_device_count(self) -> int:
        """Returns the maximum number of MIG instances the device supports. Returns 0 if the device
        does not support MIG mode.
        """

        return libnvml.nvmlQuery(
            'nvmlDeviceGetMaxMigDeviceCount', self.handle, default=0, ignore_function_not_found=True
        )

    @ttl_cache(ttl=60.0)
    def mig_device(self, mig_index: int) -> 'MigDevice':
        """Returns a child MIG device of the given index.

        Raises:
            libnvml.NVMLError:
                If the device does not support MIG mode or the given MIG device does not exist.
        """

        with _global_physical_device(self):
            return MigDevice(index=(self.index, mig_index))

    @ttl_cache(ttl=60.0)
    def mig_devices(self) -> List['MigDevice']:
        """Returns a list of children MIG devices of the current device. Returns an empty list if
        the MIG mode is disabled or the device does not support MIG mode.
        """

        mig_devices = []

        if self.is_mig_mode_enabled():
            max_mig_device_count = self.max_mig_device_count()
            with _global_physical_device(self):
                for mig_index in range(max_mig_device_count):
                    try:
                        mig_device = MigDevice(index=(self.index, mig_index))
                    except libnvml.NVMLError:
                        break
                    else:
                        mig_devices.append(mig_device)

        return mig_devices


class MigDevice(Device):  # pylint: disable=too-many-instance-attributes
    """Class for MIG devices."""

    @classmethod
    def count(cls) -> int:
        """The number of total MIG devices. Aggregated over all physical devices."""

        return len(cls.all())

    @classmethod
    def all(cls) -> List['MigDevice']:
        """Returns a list of MIG devices. Aggregated over all physical devices."""

        mig_devices = []
        for device in PhysicalDevice.all():
            mig_devices.extend(device.mig_devices())
        return mig_devices

    @classmethod
    def from_indices(  # pylint: disable=signature-differs
        cls, indices: Iterable[Tuple[int, int]]
    ) -> List['MigDevice']:
        """Returns a list of MIG devices of the given indices.

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
        self, index: Optional[Union[Tuple[int, int], str]] = None, *, uuid: Optional[str] = None
    ) -> None:
        """Initializes the instance created by ``__new__()``.

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

        index, uuid = [arg.encode() if isinstance(arg, str) else arg for arg in (index, uuid)]

        self._name = NA
        self._uuid = NA
        self._bus_id = NA
        self._memory_total = NA
        self._memory_total_human = NA
        self._gpu_instance_id = NA
        self._compute_instance_id = NA
        self._is_mig_device = True
        self._cuda_index = None

        if index is not None:
            self._nvml_index = index
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
                'nvmlDeviceGetDeviceHandleFromMigDeviceHandle', self.handle, ignore_errors=False
            )
            parent_index = libnvml.nvmlQuery(
                'nvmlDeviceGetIndex', parent_handle, ignore_errors=False
            )
            self._parent = PhysicalDevice(index=parent_index)
            for mig_device in self.parent.mig_devices():
                if self.uuid() == mig_device.uuid():
                    self._nvml_index = mig_device.index
                    break
            else:
                raise libnvml.NVMLError_NotFound

        self._max_clock_infos = ClockInfos(graphics=NA, sm=NA, memory=NA, video=NA)
        self._timestamp = 0
        self._lock = threading.RLock()

        self._ident = (self.index, self.uuid())
        self._hash = None

    @property
    def index(self) -> Tuple[int, int]:
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

    def gpu_instance_id(self) -> Union[int, NaType]:
        """The gpu instance ID of the MIG device.

        Returns: Union[int, NaType]
            The gpu instance ID of the MIG device, or :const:`nvitop.NA` when not applicable.
        """

        if self._gpu_instance_id is NA:
            self._gpu_instance_id = libnvml.nvmlQuery(
                'nvmlDeviceGetGpuInstanceId', self.handle, default=0xFFFFFFFF
            )
            if self._gpu_instance_id == 0xFFFFFFFF:
                self._gpu_instance_id = NA
        return self._gpu_instance_id

    def compute_instance_id(self) -> Union[int, NaType]:
        """The compute instance ID of the MIG device.

        Returns: Union[int, NaType]
            The compute instance ID of the MIG device, or :const:`nvitop.NA` when not applicable.
        """

        if self._compute_instance_id is NA:
            self._compute_instance_id = libnvml.nvmlQuery(
                'nvmlDeviceGetComputeInstanceId', self.handle, default=0xFFFFFFFF
            )
            if self._compute_instance_id == 0xFFFFFFFF:
                self._compute_instance_id = NA
        return self._compute_instance_id

    def as_snapshot(self) -> Snapshot:
        """Returns a onetime snapshot of the device. The attributes are defined in :attr:`SNAPSHOT_KEYS`."""

        snapshot = super().as_snapshot()
        snapshot.mig_index = self.mig_index

        return snapshot

    SNAPSHOT_KEYS = Device.SNAPSHOT_KEYS + ['gpu_instance_id', 'compute_instance_id']


class CudaDevice(Device):
    """Class for devices enumerated over the CUDA ordinal. The order can be vary for different
    ``CUDA_VISIBLE_DEVICES`` environment variable.

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

    @classmethod
    def is_available(cls) -> bool:
        """Returns whether there are any CUDA-capable devices available."""

        return cls.count() > 0

    @classmethod
    def count(cls) -> int:
        """The number of GPUs visible to CUDA applications."""

        try:
            return len(super().parse_cuda_visible_devices())
        except libnvml.NVMLError:
            return 0

    @classmethod
    def all(cls) -> List['CudaDevice']:
        """All CUDA visible devices.

        Note:
            The result could be empty if the ``CUDA_VISIBLE_DEVICES`` environment variable is invalid.
        """

        return cls.from_indices()

    @classmethod
    def from_indices(
        cls, indices: Optional[Union[int, Iterable[int]]] = None
    ) -> List['CudaDevice']:
        """Returns a list of CUDA devices of the given CUDA indices.
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
        """

        return super().from_cuda_indices(indices)

    def __new__(
        cls,
        cuda_index: Optional[int] = None,
        *,
        nvml_index: Optional[Union[int, Tuple[int, int]]] = None,
        uuid: Optional[str] = None
    ) -> 'Device':
        """Creates a new instance of CudaDevice. The type of the result is determined by the given argument.

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

        if cuda_index is not None and nvml_index is None and uuid is None:
            cuda_visible_devices = cls.parse_cuda_visible_devices()
            if not isinstance(cuda_index, int) or not 0 <= cuda_index < len(cuda_visible_devices):
                raise RuntimeError('CUDA Error: invalid device ordinal: {!r}.'.format(cuda_index))
            nvml_index = cuda_visible_devices[cuda_index]

        if not isinstance(nvml_index, int) or is_mig_device_uuid(uuid):
            return super().__new__(CudaMigDevice, index=nvml_index, uuid=uuid)

        return super().__new__(cls, index=nvml_index, uuid=uuid)

    def __init__(
        self,
        cuda_index: Optional[int] = None,
        *,
        nvml_index: Optional[Union[int, Tuple[int, int]]] = None,
        uuid: Optional[str] = None
    ) -> None:
        """Initializes the instance created by ``__new__()``.

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
                raise RuntimeError('CUDA Error: invalid device ordinal: {!r}.'.format(cuda_index))
            nvml_index = cuda_visible_devices[cuda_index]

        super().__init__(index=nvml_index, uuid=uuid)

        if cuda_index is None:
            cuda_index = super().cuda_index
        self._cuda_index = cuda_index

        self._ident = ((self._cuda_index, self.index), self.uuid())

    def __str__(self) -> str:
        return '{}(cuda_index={}, nvml_index={}, name="{}", total_memory={})'.format(
            self.__class__.__name__,
            self.cuda_index,
            self.index,
            self.name(),
            self.memory_total_human(),
        )

    __repr__ = __str__

    def __reduce__(self) -> Tuple[Type['CudaDevice'], Tuple[int]]:
        return self.__class__, (self._cuda_index,)

    def as_snapshot(self) -> Snapshot:
        """Returns a onetime snapshot of the device. The attributes are defined in :attr:`SNAPSHOT_KEYS`."""

        snapshot = super().as_snapshot()
        snapshot.cuda_index = self.cuda_index

        return snapshot


Device.cuda = CudaDevice
"""Shortcut for class :class:`CudaDevice`."""


class CudaMigDevice(CudaDevice, MigDevice):
    """Class for CUDA devices that are MIG devices."""


## Helper functions ##
_PhysicalDeviceAttrs = NamedTuple(
    'PhysicalDeviceAttrs',
    [
        ('index', int),
        ('name', str),
        ('uuid', str),
        ('support_mig_mode', bool),
    ],
)
_PHYSICAL_DEVICE_ATTRS = None
_GLOBAL_PHYSICAL_DEVICE = None
_GLOBAL_PHYSICAL_DEVICE_LOCK = threading.RLock()


def _get_all_physical_device_attrs() -> Dict[str, _PhysicalDeviceAttrs]:
    global _PHYSICAL_DEVICE_ATTRS  # pylint: disable=global-statement

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
                ]
            )
        return _PHYSICAL_DEVICE_ATTRS


def _does_any_device_support_mig_mode(uuids: Optional[Iterable[str]] = None) -> bool:
    physical_device_attrs = _get_all_physical_device_attrs()
    uuids = uuids or physical_device_attrs.keys()
    return any(physical_device_attrs[uuid].support_mig_mode for uuid in uuids)


@contextlib.contextmanager
def _global_physical_device(device: 'PhysicalDevice') -> 'PhysicalDevice':
    global _GLOBAL_PHYSICAL_DEVICE  # pylint: disable=global-statement

    with _GLOBAL_PHYSICAL_DEVICE_LOCK:
        try:
            _GLOBAL_PHYSICAL_DEVICE = device
            yield _GLOBAL_PHYSICAL_DEVICE
        finally:
            _GLOBAL_PHYSICAL_DEVICE = None


def _get_global_physical_device() -> 'PhysicalDevice':
    with _GLOBAL_PHYSICAL_DEVICE_LOCK:
        return _GLOBAL_PHYSICAL_DEVICE


def is_mig_device_uuid(uuid: Optional[str]) -> bool:
    """Returns :data:`True` if the argument is a MIG device UUID, otherwise, returns :data:`False`."""

    if isinstance(uuid, str):
        match = Device.UUID_PATTERN.match(uuid)
        if match is not None and match.group('MigMode') is not None:
            return True
    return False


def _cuda_visible_devices_parser(
    cuda_visible_devices: str, queue: mp.SimpleQueue, verbose: bool = True
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
        uuids = list(map(libcuda.cuDeviceGetUuid, map(libcuda.cuDeviceGet, range(count))))
        queue.put(uuids)
        return
    except Exception as ex:  # pylint: disable=broad-except
        queue.put(ex)
        if verbose:
            raise ex
    finally:
        # Ensure non-empty queue
        queue.put(libcuda.CUDAError_NotInitialized())  # pylint: disable=no-member


def parse_cuda_visible_devices_to_uuids(
    cuda_visible_devices: Optional[str] = _VALUE_OMITTED,
    verbose=True,
) -> List[str]:
    """Parses the given ``CUDA_VISIBLE_DEVICES`` environment variable in a separate process and
    returns a list of device UUIDs. The UUIDs do not have a prefix ``GPU-`` or ``MIG-``.

    Args:
        cuda_visible_devices (Optional[str]):
            The value of the ``CUDA_VISIBLE_DEVICES`` variable. If not given, the value from the
            environment will be used. If explicitly given by :data:`None`, the ``CUDA_VISIBLE_DEVICES``
            environment variable will be unset before parsing.

    Returns: List[str]
        A list of device UUIDs without ``GPU-`` or ``MIG-`` prefixes.

    Raises:
        libcuda.CUDAError_NotInitialized:
            If cannot found the CUDA driver libraries.
        libcuda.CUDAError:
            If failed to parse the ``CUDA_VISIBLE_DEVICES`` environment variable.
    """

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
        try:
            parser.kill()  # requires Python 3.7+
        except AttributeError:
            pass
    result = queue.get()

    if isinstance(result, Exception):
        raise result
    return result
