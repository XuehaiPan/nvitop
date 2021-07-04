# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import os
import re
from typing import List, Dict, Iterable, Callable, Union, Optional, Any

from cachetools.func import ttl_cache

from nvitop.core.libnvml import nvml
from nvitop.core.process import GpuProcess
from nvitop.core.utils import NA, NaType, Snapshot, bytes2human, utilization2string


__all__ = ['Device']


class Device(object):
    UUID_PATTEN = re.compile(r'^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')

    MEMORY_UTILIZATION_THRESHOLDS = (10, 80)
    GPU_UTILIZATION_THRESHOLDS = (10, 75)
    INTENSITY2COLOR = {'light': 'green', 'moderate': 'yellow', 'heavy': 'red'}

    @staticmethod
    def driver_version() -> Union[str, NaType]:
        return nvml.nvmlQuery('nvmlSystemGetDriverVersion')

    @staticmethod
    def cuda_version() -> Union[str, NaType]:
        cuda_version = nvml.nvmlQuery('nvmlSystemGetCudaDriverVersion')
        if nvml.nvmlCheckReturn(cuda_version, int):
            return str(cuda_version // 1000 + (cuda_version % 1000) / 100)
        return NA

    @staticmethod
    def count() -> int:
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

    @classmethod
    def from_cuda_visible_devices(cls, cuda_visible_devices: Optional[str] = None) -> List['Device']:
        if cuda_visible_devices is None:
            cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', default=None)
            if cuda_visible_devices is None:
                return cls.all()

        def from_index_or_uuid(index_or_uuid: Union[int, str]) -> 'Device':
            nonlocal use_indices

            if isinstance(index_or_uuid, str):
                if index_or_uuid.isdigit():
                    index_or_uuid = int(index_or_uuid)
                elif cls.UUID_PATTEN.match(index_or_uuid) is None:
                    raise nvml.NVMLError_NotFound()  # pylint: disable=no-member

            if use_indices is None:
                use_indices = isinstance(index_or_uuid, int)

            if isinstance(index_or_uuid, int) and use_indices:
                return cls(index=index_or_uuid)
            if isinstance(index_or_uuid, str) and not use_indices:
                return cls(uuid=index_or_uuid)
            raise ValueError('invalid identifier')

        devices = []
        presented = set()
        use_indices = None
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

        return devices

    @classmethod
    def from_cuda_indices(cls, cuda_indices: Optional[Union[int, Iterable[int]]] = None) -> List['Device']:
        cuda_devices = cls.from_cuda_visible_devices()
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
            device._cuda_index = cuda_index  # pylint: disable=protected-access
            devices.append(device)

        return devices

    def __init__(self, index: Optional[int] = None,
                 serial: Optional[str] = None,
                 uuid: Optional[str] = None,
                 bus_id: Optional[str] = None) -> None:
        args = (index, serial, uuid, bus_id)
        assert args.count(None) == 3
        index, serial, uuid, bus_id = [arg.encode() if isinstance(arg, str) else arg
                                       for arg in args]
        if index is not None:
            self.index = index
            try:
                self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleByIndex', index, ignore_errors=False)
            except nvml.NVMLError_GpuIsLost:  # pylint: disable=no-member
                self.handle = None
        else:
            try:
                if serial is not None:
                    self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleBySerial', serial, ignore_errors=False)
                elif uuid is not None:
                    self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleByUUID', uuid, ignore_errors=False)
                elif bus_id is not None:
                    self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleByPciBusId', bus_id, ignore_errors=False)
            except nvml.NVMLError_GpuIsLost:  # pylint: disable=no-member
                self.handle = None
                self.index = NA
            else:
                self.index = nvml.nvmlQuery('nvmlDeviceGetIndex', self.handle)

        self._cuda_index = None
        self._name = NA
        self._serial = NA
        self._uuid = NA
        self._bus_id = NA
        self._memory_total = NA
        self._memory_total_human = NA
        self._timestamp = 0

        self._ident = (self.index, self.uuid())
        self._hash = None
        self._snapshot = None

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
    def cuda_index(self):
        if self._cuda_index is None:
            return self.index
        return self._cuda_index

    def name(self) -> Union[str, NaType]:
        if self._name is NA:
            self._name = nvml.nvmlQuery('nvmlDeviceGetName', self.handle)
        return self._name

    def serial(self) -> Union[str, NaType]:
        if self._serial is NA:
            self._serial = nvml.nvmlQuery('nvmlDeviceGetSerial', self.handle)
        return self._serial

    def uuid(self) -> Union[str, NaType]:
        if self._uuid is NA:
            self._uuid = nvml.nvmlQuery('nvmlDeviceGetUUID', self.handle)
        return self._uuid

    def bus_id(self) -> Union[str, NaType]:
        if self._bus_id is NA:
            self._bus_id = nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetPciInfo(handle).busId, self.handle)
        return self._bus_id

    def memory_total(self) -> Union[int, NaType]:  # in bytes
        if self._memory_total is NA:
            self._memory_total = nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).total, self.handle)
        return self._memory_total

    def memory_total_human(self) -> Union[str, NaType]:  # in human readable
        if self._memory_total_human is NA:
            self._memory_total_human = bytes2human(self.memory_total())
        return self._memory_total_human

    @ttl_cache(ttl=60.0)
    def display_active(self) -> Union[str, NaType]:
        return {0: 'Off', 1: 'On'}.get(nvml.nvmlQuery('nvmlDeviceGetDisplayActive', self.handle), NA)

    @ttl_cache(ttl=60.0)
    def persistence_mode(self) -> Union[str, NaType]:
        return {0: 'Off', 1: 'On'}.get(nvml.nvmlQuery('nvmlDeviceGetPersistenceMode', self.handle), NA)

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

    @ttl_cache(ttl=5.0)
    def performance_state(self) -> Union[str, NaType]:
        performance_state = nvml.nvmlQuery('nvmlDeviceGetPerformanceState', self.handle)
        if nvml.nvmlCheckReturn(performance_state, int):
            performance_state = 'P' + str(performance_state)
        return performance_state

    @ttl_cache(ttl=5.0)
    def power_usage(self) -> Union[int, NaType]:  # in milliwatts (mW)
        return nvml.nvmlQuery('nvmlDeviceGetPowerUsage', self.handle)

    power_draw = power_usage  # in milliwatts (mW)

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

    @ttl_cache(ttl=5.0)
    def fan_speed(self) -> Union[str, NaType]:  # in percentage
        return nvml.nvmlQuery('nvmlDeviceGetFanSpeed', self.handle)

    def fan_speed_string(self) -> Union[str, NaType]:  # in percentage
        return utilization2string(self.fan_speed())

    @ttl_cache(ttl=5.0)
    def temperature(self) -> Union[str, NaType]:  # in Celsius
        return nvml.nvmlQuery('nvmlDeviceGetTemperature', self.handle, nvml.NVML_TEMPERATURE_GPU)

    def temperature_string(self) -> Union[str, NaType]:  # in Celsius
        temperature = self.temperature()
        if nvml.nvmlCheckReturn(temperature, int):
            temperature = str(temperature) + 'C'
        return temperature

    @ttl_cache(ttl=1.0)
    def memory_used(self) -> Union[int, NaType]:  # in bytes
        return nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).used, self.handle)

    def memory_used_human(self) -> Union[str, NaType]:
        return bytes2human(self.memory_used())

    @ttl_cache(ttl=1.0)
    def memory_free(self) -> Union[int, NaType]:  # in bytes
        return nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).free, self.handle)

    def memory_free_human(self) -> Union[str, NaType]:
        return bytes2human(self.memory_free())

    def memory_usage(self) -> str:  # string of used memory over total memory (in human readable)
        return '{} / {}'.format(self.memory_used_human(), self.memory_total_human())

    def memory_utilization(self) -> Union[float, NaType]:  # used memory over total memory (in percentage)
        memory_used = self.memory_used()
        memory_total = self.memory_total()
        if nvml.nvmlCheckReturn(memory_used, int) and nvml.nvmlCheckReturn(memory_total, int):
            return round(100.0 * memory_used / memory_total, 1)
        return NA

    def memory_utilization_string(self) -> Union[str, NaType]:  # in percentage
        return utilization2string(self.memory_utilization())

    @ttl_cache(ttl=1.0)
    def gpu_utilization(self) -> Union[int, NaType]:  # in percentage
        return nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetUtilizationRates(handle).gpu, self.handle)

    def gpu_utilization_string(self) -> Union[str, NaType]:  # in percentage
        return utilization2string(self.gpu_utilization())

    @ttl_cache(ttl=2.0)
    def processes(self) -> Dict[int, GpuProcess]:
        processes = {}

        for type, func in [('C', 'nvmlDeviceGetComputeRunningProcesses'),  # pylint: disable=redefined-builtin
                           ('G', 'nvmlDeviceGetGraphicsRunningProcesses')]:
            for p in nvml.nvmlQuery(func, self.handle, default=()):
                proc = processes[p.pid] = GpuProcess(pid=p.pid, device=self)
                proc.set_gpu_memory(p.usedGpuMemory if isinstance(p.usedGpuMemory, int)
                                    else NA)  # used GPU memory is `N/A` in Windows Display Driver Model (WDDM)
                proc.set_gpu_utilization(0, 0, 0)
                proc.type = proc.type + type

        if len(processes) > 0:
            samples = nvml.nvmlQuery('nvmlDeviceGetProcessUtilization', self.handle, self._timestamp, default=())
            self._timestamp = max(min((s.timeStamp for s in samples), default=0) - 500000, 0)
            for s in samples:
                try:
                    processes[s.pid].set_gpu_utilization(s.smUtil, s.encUtil, s.decUtil)
                except KeyError:
                    pass

        return processes

    def as_snapshot(self) -> Snapshot:
        self._snapshot = Snapshot(real=self, index=self.index,
                                  **{key: getattr(self, key)() for key in self.SNAPSHOT_KEYS})
        return self._snapshot

    @property
    def snapshot(self) -> Snapshot:
        if self._snapshot is None:
            self.as_snapshot()
        return self._snapshot

    SNAPSHOT_KEYS = [
        'name', 'bus_id',
        'persistence_mode', 'display_active', 'ecc_errors',
        'performance_state', 'compute_mode',
        'fan_speed', 'fan_speed_string', 'temperature', 'temperature_string',
        'power_usage', 'power_limit', 'power_status',
        'memory_used', 'memory_free', 'memory_total',
        'memory_used_human', 'memory_free_human', 'memory_total_human', 'memory_usage',
        'memory_utilization', 'gpu_utilization',
        'memory_utilization_string', 'gpu_utilization_string'
    ]

    def memory_loading_intensity(self) -> str:
        return self.loading_intensity_of(self.memory_utilization(), type='memory')

    def gpu_loading_intensity(self) -> str:
        return self.loading_intensity_of(self.gpu_utilization(), type='gpu')

    def loading_intensity(self) -> str:
        loading_intensity = (self.memory_loading_intensity(), self.gpu_loading_intensity())
        if 'heavy' in loading_intensity:
            return 'heavy'
        if 'moderate' in loading_intensity:
            return 'moderate'
        return 'light'

    def display_color(self) -> str:
        return self.INTENSITY2COLOR.get(self.loading_intensity())

    def memory_display_color(self) -> str:
        return self.INTENSITY2COLOR.get(self.memory_loading_intensity())

    def gpu_display_color(self) -> str:
        return self.INTENSITY2COLOR.get(self.gpu_loading_intensity())

    @staticmethod
    def loading_intensity_of(utilization: Union[int, float, str, NaType],
                             type: str = 'memory') -> str:  # pylint: disable=redefined-builtin
        thresholds = {'memory': Device.MEMORY_UTILIZATION_THRESHOLDS,
                      'gpu': Device.GPU_UTILIZATION_THRESHOLDS}.get(type)
        if utilization is NA:
            return 'heavy'
        if isinstance(utilization, str):
            utilization = utilization[:-1]
        utilization = float(utilization)
        if utilization >= thresholds[-1]:
            return 'heavy'
        if utilization >= thresholds[0]:
            return 'moderate'
        return 'light'

    @staticmethod
    def color_of(utilization: Union[int, float, str, NaType],
                 type: str = 'memory') -> str:  # pylint: disable=redefined-builtin
        return Device.INTENSITY2COLOR.get(Device.loading_intensity_of(utilization, type=type))
