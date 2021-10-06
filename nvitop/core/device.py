# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import contextlib
import os
import re
import threading
from collections import namedtuple
from typing import List, Dict, Iterable, Callable, Union, Optional, Any

from cachetools.func import ttl_cache

from nvitop.core.libnvml import nvml
from nvitop.core.process import GpuProcess
from nvitop.core.utils import NA, NaType, Snapshot, bytes2human, utilization2string, memoize_when_activated


__all__ = ['Device', 'PhysicalDevice', 'CudaDevice']


MemoryInfo = namedtuple('MemoryInfo', ['total', 'free', 'used'])
UtilizationRates = namedtuple('UtilizationRates', ['gpu', 'memory'])


class Device(object):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    UUID_PATTEN = re.compile(r'^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')
    GPU_PROCESS_CLASS = GpuProcess

    @staticmethod
    def driver_version() -> Union[str, NaType]:
        return nvml.nvmlQuery('nvmlSystemGetDriverVersion')

    @staticmethod
    def cuda_version() -> Union[str, NaType]:
        cuda_version = nvml.nvmlQuery('nvmlSystemGetCudaDriverVersion')
        if nvml.nvmlCheckReturn(cuda_version, int):
            return str(cuda_version // 1000 + (cuda_version % 1000) / 100)
        return NA

    @classmethod
    def count(cls) -> int:
        return nvml.nvmlQuery('nvmlDeviceGetCount', default=0)

    @classmethod
    def all(cls) -> List['Device']:
        return cls.from_indices()

    @classmethod
    def from_indices(cls, indices: Optional[Union[int, Iterable[int]]] = None) -> List['Device']:
        if indices is None:
            indices = range(cls.count())

        if isinstance(indices, int):
            indices = [indices]

        return list(map(cls, indices))

    @staticmethod
    def from_cuda_visible_devices(cuda_visible_devices: Optional[str] = None) -> List['CudaDevice']:
        visible_device_indices = Device.parse_cuda_visible_devices(cuda_visible_devices)
        cuda_devices = Device.from_indices(visible_device_indices)

        cuda_devices = []
        for cuda_index, device_index in enumerate(visible_device_indices):
            cuda_devices.append(CudaDevice(cuda_index, physical_index=device_index))

        return cuda_devices

    @staticmethod
    def from_cuda_indices(cuda_indices: Optional[Union[int, Iterable[int]]] = None) -> List['CudaDevice']:
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
                raise RuntimeError('CUDA Error: invalid device ordinal')
            device = cuda_devices[cuda_index]
            devices.append(device)

        return devices

    @staticmethod
    def parse_cuda_visible_devices(cuda_visible_devices: Optional[str] = None) -> List[int]:
        if cuda_visible_devices is None:
            cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', default=None)
            if cuda_visible_devices is None:
                return list(range(Device.count()))

        return Device._parse_cuda_visible_devices(cuda_visible_devices)

    @staticmethod
    @ttl_cache(ttl=300.0)
    def _parse_cuda_visible_devices(cuda_visible_devices: str) -> List[int]:
        def from_index_or_uuid(index_or_uuid: Union[int, str]) -> 'Device':
            nonlocal use_integer_identifiers

            if isinstance(index_or_uuid, str):
                if index_or_uuid.isdigit():
                    index_or_uuid = int(index_or_uuid)
                elif Device.UUID_PATTEN.match(index_or_uuid) is None:
                    raise nvml.NVMLError_NotFound()  # pylint: disable=no-member

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
                raise RuntimeError('CUDA Error: invalid device ordinal')
            try:
                device = from_index_or_uuid(identifier)
            except (ValueError, nvml.NVMLError):
                break
            else:
                devices.append(device)
                presented.add(identifier)

        return [device.index for device in devices]

    def __init__(self, index: Optional[Union[int, str]] = None, *,
                 uuid: Optional[str] = None,
                 bus_id: Optional[str] = None,
                 serial: Optional[str] = None) -> None:

        if (index, uuid, bus_id, serial).count(None) != 3:
            raise TypeError(
                'Device(index=None, uuid=None, serial=None, bus_id=None) takes 1 non-None arguments '
                'but (index, uuid, bus_id, serial) = {} were given'.format((index, uuid, bus_id, serial))
            )

        if index is not None:
            if isinstance(index, str) and self.UUID_PATTEN.match(index) is not None:  # passed by UUID
                index, uuid = None, index

        index, uuid, bus_id, serial = [arg.encode() if isinstance(arg, str) else arg
                                       for arg in (index, uuid, bus_id, serial)]

        if index is not None:
            self._index = index
            try:
                self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleByIndex', index, ignore_errors=False)
            except nvml.NVMLError_GpuIsLost:  # pylint: disable=no-member
                self.handle = None
        else:
            try:
                if uuid is not None:
                    self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleByUUID', uuid, ignore_errors=False)
                elif bus_id is not None:
                    self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleByPciBusId', bus_id, ignore_errors=False)
                elif serial is not None:
                    self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleBySerial', serial, ignore_errors=False)
            except nvml.NVMLError_GpuIsLost:  # pylint: disable=no-member
                self.handle = None
                self._index = NA
            else:
                self._index = nvml.nvmlQuery('nvmlDeviceGetIndex', self.handle)

        self._cuda_index = None
        self._name = NA
        self._serial = NA
        self._uuid = NA
        self._bus_id = NA
        self._memory_total = NA
        self._memory_total_human = NA
        self._timestamp = 0
        self._lock = threading.RLock()

        self._ident = (self.index, self.uuid())
        self._hash = None

    def __str__(self) -> str:
        return '{}(index={}, name="{}", total_memory={})'.format(
            self.__class__.__name__,
            self.index, self.name(), self.memory_total_human()
        )

    __repr__ = __str__

    def __eq__(self, other: 'Device') -> bool:
        if not isinstance(other, Device):
            return NotImplemented
        return self._ident == other._ident

    def __ne__(self, other: 'Device') -> bool:
        return not self == other

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self._ident)
        return self._hash

    def __getattr__(self, name: str) -> Union[Any, Callable[..., Any]]:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self.handle is None:
                return NA

            try:
                func = getattr(nvml, 'nvmlDeviceGet' + name.title().replace('_', ''))
            except AttributeError:
                pascal_case = ''.join(part[:1].upper() + part[1:] for part in filter(None, name.split('_')))
                func = getattr(nvml, 'nvmlDeviceGet' + pascal_case)

            @ttl_cache(ttl=1.0)
            def attribute(*args, **kwargs):
                try:
                    return nvml.nvmlQuery(func, self.handle, *args, **kwargs, ignore_errors=False)
                except nvml.NVMLError_NotSupported:  # pylint: disable=no-member
                    return NA

            setattr(self, name, attribute)
            return attribute

    @property
    def index(self) -> int:
        return self._index

    @property
    def physical_index(self) -> int:
        return self._index

    @property
    def cuda_index(self) -> int:
        if self._cuda_index is None:
            visible_device_indices = self.parse_cuda_visible_devices()
            try:
                cuda_index = visible_device_indices.index(self.index)
            except ValueError as e:
                raise RuntimeError(
                    'CUDA Error: Device(index={}) is not visible to CUDA applications'.format(self.index)
                ) from e
            else:
                self._cuda_index = cuda_index

        return self._cuda_index

    def name(self) -> Union[str, NaType]:
        if self._name is NA:
            self._name = nvml.nvmlQuery('nvmlDeviceGetName', self.handle)
        return self._name

    def uuid(self) -> Union[str, NaType]:
        if self._uuid is NA:
            self._uuid = nvml.nvmlQuery('nvmlDeviceGetUUID', self.handle)
        return self._uuid

    def bus_id(self) -> Union[str, NaType]:
        if self._bus_id is NA:
            self._bus_id = nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetPciInfo(handle).busId, self.handle)
        return self._bus_id

    def serial(self) -> Union[str, NaType]:
        if self._serial is NA:
            self._serial = nvml.nvmlQuery('nvmlDeviceGetSerial', self.handle)
        return self._serial

    @memoize_when_activated
    @ttl_cache(ttl=1.0)
    def memory_info(self) -> MemoryInfo:  # in bytes
        memory_info = nvml.nvmlQuery('nvmlDeviceGetMemoryInfo', self.handle)
        if nvml.nvmlCheckReturn(memory_info):
            return MemoryInfo(total=memory_info.total, free=memory_info.free, used=memory_info.used)
        return MemoryInfo(total=NA, free=NA, used=NA)

    def memory_total(self) -> Union[int, NaType]:  # in bytes
        if self._memory_total is NA:
            self._memory_total = self.memory_info().total
        return self._memory_total

    def memory_used(self) -> Union[int, NaType]:  # in bytes
        return self.memory_info().used

    def memory_free(self) -> Union[int, NaType]:  # in bytes
        return self.memory_info().free

    def memory_total_human(self) -> Union[str, NaType]:  # in human readable
        if self._memory_total_human is NA:
            self._memory_total_human = bytes2human(self.memory_total())
        return self._memory_total_human

    def memory_used_human(self) -> Union[str, NaType]:
        return bytes2human(self.memory_used())

    def memory_free_human(self) -> Union[str, NaType]:
        return bytes2human(self.memory_free())

    def memory_percent(self) -> Union[float, NaType]:  # used memory over total memory (in percentage)
        memory_total, memory_free, memory_used = self.memory_info()  # pylint: disable=unused-variable

        if nvml.nvmlCheckReturn(memory_used, int) and nvml.nvmlCheckReturn(memory_total, int):
            return round(100.0 * memory_used / memory_total, 1)
        return NA

    def memory_usage(self) -> str:  # string of used memory over total memory (in human readable)
        return '{} / {}'.format(self.memory_used_human(), self.memory_total_human())

    @memoize_when_activated
    @ttl_cache(ttl=1.0)
    def utilization_rates(self) -> UtilizationRates:  # in percentage
        utilization_rates = nvml.nvmlQuery('nvmlDeviceGetUtilizationRates', self.handle)
        if nvml.nvmlCheckReturn(utilization_rates):
            return UtilizationRates(gpu=utilization_rates.gpu, memory=utilization_rates.memory)
        return UtilizationRates(gpu=NA, memory=NA)

    def memory_utilization(self) -> Union[float, NaType]:  # in percentage
        return self.utilization_rates().memory

    def gpu_utilization(self) -> Union[int, NaType]:  # in percentage
        return self.utilization_rates().gpu

    gpu_percent = gpu_utilization  # in percentage

    @ttl_cache(ttl=5.0)
    def fan_speed(self) -> Union[int, NaType]:  # in percentage
        return nvml.nvmlQuery('nvmlDeviceGetFanSpeed', self.handle)

    @ttl_cache(ttl=5.0)
    def temperature(self) -> Union[int, NaType]:  # in Celsius
        return nvml.nvmlQuery('nvmlDeviceGetTemperature', self.handle, nvml.NVML_TEMPERATURE_GPU)

    @memoize_when_activated
    @ttl_cache(ttl=5.0)
    def power_usage(self) -> Union[int, NaType]:  # in milliwatts (mW)
        return nvml.nvmlQuery('nvmlDeviceGetPowerUsage', self.handle)

    power_draw = power_usage  # in milliwatts (mW)

    @memoize_when_activated
    @ttl_cache(ttl=60.0)
    def power_limit(self) -> Union[int, NaType]:  # in milliwatts (mW)
        return nvml.nvmlQuery('nvmlDeviceGetPowerManagementLimit', self.handle)

    def power_status(self) -> str:  # string of power usage over power limit in watts (W)
        power_usage = self.power_usage()
        power_limit = self.power_limit()
        if nvml.nvmlCheckReturn(power_usage, int):
            power_usage = '{}W'.format(round(power_usage / 1000.0))
        if nvml.nvmlCheckReturn(power_limit, int):
            power_limit = '{}W'.format(round(power_limit / 1000.0))
        return '{} / {}'.format(power_usage, power_limit)

    @ttl_cache(ttl=60.0)
    def display_active(self) -> Union[str, NaType]:
        return {0: 'Off', 1: 'On'}.get(nvml.nvmlQuery('nvmlDeviceGetDisplayActive', self.handle), NA)

    @ttl_cache(ttl=60.0)
    def persistence_mode(self) -> Union[str, NaType]:
        return {0: 'Off', 1: 'On'}.get(nvml.nvmlQuery('nvmlDeviceGetPersistenceMode', self.handle), NA)

    @ttl_cache(ttl=5.0)
    def performance_state(self) -> Union[str, NaType]:
        performance_state = nvml.nvmlQuery('nvmlDeviceGetPerformanceState', self.handle)
        if nvml.nvmlCheckReturn(performance_state, int):
            performance_state = 'P' + str(performance_state)
        return performance_state

    @ttl_cache(ttl=60.0)
    def current_driver_model(self) -> Union[str, NaType]:
        return {
            nvml.NVML_DRIVER_WDDM: 'WDDM',
            nvml.NVML_DRIVER_WDM: 'WDM',
        }.get(nvml.nvmlQuery('nvmlDeviceGetCurrentDriverModel', self.handle), NA)

    @ttl_cache(ttl=5.0)
    def ecc_errors(self) -> Union[int, NaType]:
        return nvml.nvmlQuery('nvmlDeviceGetTotalEccErrors', self.handle,
                              nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                              nvml.NVML_VOLATILE_ECC)

    @ttl_cache(ttl=60.0)
    def compute_mode(self) -> Union[str, NaType]:
        return {
            nvml.NVML_COMPUTEMODE_DEFAULT: 'Default',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: 'E. Thread',
            nvml.NVML_COMPUTEMODE_PROHIBITED: 'Prohibited',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: 'E. Process',
        }.get(nvml.nvmlQuery('nvmlDeviceGetComputeMode', self.handle), NA)

    @ttl_cache(ttl=2.0)
    def processes(self) -> Dict[int, GpuProcess]:
        processes = {}

        for type, func in [('C', 'nvmlDeviceGetComputeRunningProcesses'),  # pylint: disable=redefined-builtin
                           ('G', 'nvmlDeviceGetGraphicsRunningProcesses')]:
            for p in nvml.nvmlQuery(func, self.handle, default=()):
                proc = processes[p.pid] = self.GPU_PROCESS_CLASS(pid=p.pid, device=self)
                proc.set_gpu_memory(p.usedGpuMemory if isinstance(p.usedGpuMemory, int)
                                    else NA)  # used GPU memory is `N/A` in Windows Display Driver Model (WDDM)
                proc.set_gpu_utilization(0, 0, 0)
                proc.type = proc.type + type

        if len(processes) > 0:
            samples = nvml.nvmlQuery('nvmlDeviceGetProcessUtilization', self.handle, self._timestamp, default=())
            self._timestamp = max(min((s.timeStamp for s in samples), default=0) - 500000, 0)
            for s in samples:
                try:
                    processes[s.pid].set_gpu_utilization(s.smUtil, s.memUtil, s.encUtil, s.decUtil)
                except KeyError:
                    pass

        return processes

    def as_snapshot(self) -> Snapshot:
        with self.oneshot():
            return Snapshot(real=self, index=self.index,
                            **{key: getattr(self, key)() for key in self.SNAPSHOT_KEYS})

    SNAPSHOT_KEYS = [
        'name', 'bus_id',

        'memory_info',
        'memory_used', 'memory_free', 'memory_total',
        'memory_used_human', 'memory_free_human', 'memory_total_human',
        'memory_usage', 'memory_percent',

        'utilization_rates', 'gpu_utilization', 'memory_utilization',

        'fan_speed', 'temperature',

        'power_usage', 'power_limit', 'power_status',

        'display_active', 'persistence_mode', 'performance_state',
        'current_driver_model', 'ecc_errors', 'compute_mode',
    ]

    def memory_percent_string(self) -> Union[str, NaType]:  # in percentage
        return utilization2string(self.memory_percent())

    def memory_utilization_string(self) -> Union[str, NaType]:  # in percentage
        return utilization2string(self.memory_utilization())

    def gpu_utilization_string(self) -> Union[str, NaType]:  # in percentage
        return utilization2string(self.gpu_utilization())

    gpu_percent_string = gpu_utilization_string  # in percentage

    def fan_speed_string(self) -> Union[str, NaType]:  # in percentage
        return utilization2string(self.fan_speed())

    def temperature_string(self) -> Union[str, NaType]:  # in Celsius
        temperature = self.temperature()
        if nvml.nvmlCheckReturn(temperature, int):
            temperature = str(temperature) + 'C'
        return temperature

    # Modified from psutil (https://github.com/giampaolo/psutil)
    @contextlib.contextmanager
    def oneshot(self):
        """Utility context manager which considerably speeds up the
        retrieval of multiple process information at the same time.

        Internally different device info (e.g. memory_info,
        utilization_rates, ...) may be fetched by using the same
        routine, but only one information is returned and the others
        are discarded.
        When using this context manager the internal routine is
        executed once (in the example below on memory_info()) and the
        other info are cached.

        The cache is cleared when exiting the context manager block.
        The advice is to use this every time you retrieve more than
        one information about the device. If you're lucky, you'll
        get a hell of a speedup.

        >>> from nvitop import Device
        >>> device = Device(0)
        >>> with device.oneshot():
        ...     device.memory_info()        # collect multiple info
        ...     device.memory_used()        # return cached value
        ...     device.memory_free_human()  # return cached value
        ...     device.memory_percent()     # return cached value
        >>>
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
                    self.utilization_rates.cache_activate(self)
                    self.power_usage.cache_activate(self)
                    self.power_limit.cache_activate(self)
                    yield
                finally:
                    self.memory_info.cache_deactivate(self)
                    self.utilization_rates.cache_deactivate(self)
                    self.power_usage.cache_deactivate(self)
                    self.power_limit.cache_deactivate(self)


PhysicalDevice = Device


class CudaDevice(Device):

    @classmethod
    def count(cls) -> int:
        return len(super().parse_cuda_visible_devices())

    @classmethod
    def all(cls) -> List['CudaDevice']:
        return super().from_cuda_visible_devices()

    def __init__(self, cuda_index: Optional[int] = None, *,
                 physical_index: Optional[Union[int, str]] = None,
                 uuid: Optional[str] = None) -> None:

        if cuda_index is not None and physical_index is None and uuid is None:
            cuda_visible_devices = self.parse_cuda_visible_devices()
            if not 0 <= cuda_index < len(cuda_visible_devices):
                raise RuntimeError('CUDA Error: invalid device ordinal')
            physical_index = cuda_visible_devices[cuda_index]

        super().__init__(index=physical_index, uuid=uuid)

        self._cuda_index = cuda_index

    def __str__(self) -> str:
        return '{}(cuda_index={}, physical_index={}, name="{}", total_memory={})'.format(
            self.__class__.__name__,
            self.cuda_index, self.physical_index,
            self.name(), self.memory_total_human()
        )

    __repr__ = __str__

    def as_snapshot(self) -> Snapshot:
        snapshot = super().as_snapshot()
        snapshot.physical_index = self.physical_index
        snapshot.cuda_index = self.cuda_index

        return snapshot
