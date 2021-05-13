# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

from collections import OrderedDict

from cachetools.func import ttl_cache

from .libnvml import nvml
from .process import GpuProcess
from .utils import Snapshot, bytes2human


__all__ = ['Device']


class Device(object):
    MEMORY_UTILIZATION_THRESHOLDS = (10, 80)
    GPU_UTILIZATION_THRESHOLDS = (10, 75)
    INTENSITY2COLOR = {'light': 'green', 'moderate': 'yellow', 'heavy': 'red'}

    @staticmethod
    def count():
        return nvml.nvmlQuery('nvmlDeviceGetCount')

    @classmethod
    def from_indices(cls, indices=None):
        if indices is None:
            indices = range(cls.count())
        devices = []
        for index in indices:
            devices.append(cls(index))
        return devices

    @classmethod
    def all(cls):
        return cls.from_indices()

    @staticmethod
    def driver_version():
        return nvml.nvmlQuery('nvmlSystemGetDriverVersion')

    @staticmethod
    def cuda_version():
        cuda_version = nvml.nvmlQuery('nvmlSystemGetCudaDriverVersion')
        if nvml.nvmlCheckReturn(cuda_version, int):
            return str(cuda_version // 1000 + (cuda_version % 1000) / 100)
        return 'N/A'

    def __init__(self, index=None, serial=None, uuid=None, bus_id=None):
        args = (index, serial, uuid, bus_id)
        assert args.count(None) == 3
        index, serial, uuid, bus_id = [arg.encode() if isinstance(arg, str) else arg
                                       for arg in args]
        if index is not None:
            self.index = index
            try:
                self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleByIndex', index, catch_error=False)
            except nvml.NVMLError_GpuIsLost:  # pylint: disable=no-member
                self.handle = None
        else:
            try:
                if serial is not None:
                    self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleBySerial', serial, catch_error=False)
                elif uuid is not None:
                    self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleByUUID', uuid, catch_error=False)
                elif bus_id is not None:
                    self.handle = nvml.nvmlQuery('nvmlDeviceGetHandleByPciBusId', bus_id, catch_error=False)
            except nvml.NVMLError_GpuIsLost:  # pylint: disable=no-member
                self.handle = None
                self.index = 'N/A'
            else:
                self.index = nvml.nvmlQuery('nvmlDeviceGetIndex', self.handle)

        self._name = nvml.nvmlQuery('nvmlDeviceGetName', self.handle)
        self._serial = nvml.nvmlQuery('nvmlDeviceGetSerial', self.handle)
        self._uuid = nvml.nvmlQuery('nvmlDeviceGetUUID', self.handle)
        self._bus_id = nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetPciInfo(handle).busId, self.handle)
        self._memory_total = nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).total, self.handle)
        self._memory_total_human = bytes2human(self.memory_total())
        self._power_limit = nvml.nvmlQuery('nvmlDeviceGetPowerManagementLimit', self.handle)

        self._ident = (self.index, self.bus_id())
        self._hash = None
        self._snapshot = None

    def __str__(self):
        return '{}(index={}, name="{}", total_memory={})'.format(
            self.__class__.__name__,
            self.index, self.name(), self.memory_total_human()
        )

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, Device):
            return NotImplemented
        return self._ident == other._ident

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._ident)
        return self._hash

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self.handle is None:
                return 'N/A'

            try:
                func = getattr(nvml, 'nvmlDeviceGet' + name.title().replace('_', ''))
            except AttributeError:
                pascal_case = ''.join(part[:1].upper() + part[1:] for part in filter(None, name.split('_')))
                func = getattr(nvml, 'nvmlDeviceGet' + pascal_case)

            @ttl_cache(ttl=1.0)
            def attribute(*args, **kwargs):
                try:
                    return nvml.nvmlQuery(func, self.handle, *args, **kwargs, catch_error=False)
                except nvml.NVMLError_NotSupported:  # pylint: disable=no-member
                    return 'N/A'

            setattr(self, name, attribute)
            return attribute

    def name(self):
        return self._name

    def serial(self):
        return self._serial

    def uuid(self):
        return self._uuid

    def bus_id(self):
        return self._bus_id

    def memory_total(self):
        return self._memory_total

    def memory_total_human(self):
        return self._memory_total_human

    def power_limit(self):
        return self._power_limit

    @ttl_cache(ttl=60.0)
    def display_active(self):
        return {0: 'Off', 1: 'On'}.get(nvml.nvmlQuery('nvmlDeviceGetDisplayActive', self.handle), 'N/A')

    @ttl_cache(ttl=60.0)
    def persistence_mode(self):
        return {0: 'Off', 1: 'On'}.get(nvml.nvmlQuery('nvmlDeviceGetPersistenceMode', self.handle), 'N/A')

    @ttl_cache(ttl=5.0)
    def ecc_errors(self):
        return nvml.nvmlQuery('nvmlDeviceGetTotalEccErrors', self.handle,
                              nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                              nvml.NVML_VOLATILE_ECC)

    @ttl_cache(ttl=60.0)
    def compute_mode(self):
        return {
            nvml.NVML_COMPUTEMODE_DEFAULT: 'Default',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: 'E. Thread',
            nvml.NVML_COMPUTEMODE_PROHIBITED: 'Prohibited',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: 'E. Process',
        }.get(nvml.nvmlQuery('nvmlDeviceGetComputeMode', self.handle), 'N/A')

    @ttl_cache(ttl=5.0)
    def performance_state(self):
        performance_state = nvml.nvmlQuery('nvmlDeviceGetPerformanceState', self.handle)
        if nvml.nvmlCheckReturn(performance_state, int):
            performance_state = 'P' + str(performance_state)
        return performance_state

    @ttl_cache(ttl=5.0)
    def power_usage(self):
        return nvml.nvmlQuery('nvmlDeviceGetPowerUsage', self.handle)

    @ttl_cache(ttl=5.0)
    def power_state(self):
        power_usage = self.power_usage()
        power_limit = self.power_limit()
        if nvml.nvmlCheckReturn(power_usage, int) and nvml.nvmlCheckReturn(power_limit, int):
            return '{}W / {}W'.format(power_usage // 1000, power_limit // 1000)
        return 'N/A'

    @ttl_cache(ttl=5.0)
    def fan_speed(self):
        fan_speed = nvml.nvmlQuery('nvmlDeviceGetFanSpeed', self.handle)
        if nvml.nvmlCheckReturn(fan_speed, int):
            fan_speed = str(fan_speed) + '%'
        return fan_speed

    @ttl_cache(ttl=5.0)
    def temperature(self):
        temperature = nvml.nvmlQuery('nvmlDeviceGetTemperature', self.handle, nvml.NVML_TEMPERATURE_GPU)
        if nvml.nvmlCheckReturn(temperature, int):
            temperature = str(temperature) + 'C'
        return temperature

    @ttl_cache(ttl=1.0)
    def memory_used(self):
        return nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).used, self.handle)

    @ttl_cache(ttl=1.0)
    def memory_used_human(self):
        return bytes2human(self.memory_used())

    @ttl_cache(ttl=1.0)
    def memory_free(self):
        return nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).free, self.handle)

    @ttl_cache(ttl=1.0)
    def memory_free_human(self):
        return bytes2human(self.memory_free())

    @ttl_cache(ttl=1.0)
    def memory_usage(self):
        memory_used = self.memory_used()
        memory_total = self.memory_total()
        if nvml.nvmlCheckReturn(memory_used, int) and nvml.nvmlCheckReturn(memory_total, int):
            return '{} / {}'.format(bytes2human(memory_used), bytes2human(memory_total))
        return 'N/A'

    @ttl_cache(ttl=1.0)
    def memory_utilization(self):
        memory_used = self.memory_used()
        memory_total = self.memory_total()
        if nvml.nvmlCheckReturn(memory_used, int) and nvml.nvmlCheckReturn(memory_total, int):
            return 100 * memory_used // memory_total
        return 'N/A'

    @ttl_cache(ttl=1.0)
    def memory_utilization_string(self):
        memory_utilization = self.memory_utilization()
        if nvml.nvmlCheckReturn(memory_utilization, int):
            return str(memory_utilization) + '%'
        return 'N/A'

    @ttl_cache(ttl=1.0)
    def gpu_utilization(self):
        return nvml.nvmlQuery(lambda handle: nvml.nvmlDeviceGetUtilizationRates(handle).gpu, self.handle)

    @ttl_cache(ttl=1.0)
    def gpu_utilization_string(self):
        gpu_utilization = self.gpu_utilization()
        if nvml.nvmlCheckReturn(gpu_utilization, int):
            return str(gpu_utilization) + '%'
        return 'N/A'

    @ttl_cache(ttl=2.0)
    def processes(self):
        processes = OrderedDict()

        for type, func in [('C', 'nvmlDeviceGetComputeRunningProcesses'),  # pylint: disable=redefined-builtin
                           ('G', 'nvmlDeviceGetGraphicsRunningProcesses')]:
            for p in nvml.nvmlQuery(func, self.handle, default=()):
                proc = processes[p.pid] = GpuProcess(pid=p.pid, device=self)
                proc.set_gpu_memory(p.usedGpuMemory if isinstance(p.usedGpuMemory, int) else 'N/A')
                proc.type = proc.type + type

        return processes

    @ttl_cache(ttl=1.0)
    def as_snapshot(self):
        self._snapshot = Snapshot(real=self, index=self.index,
                                  **{key: getattr(self, key)() for key in self._snapshot_keys})
        return self._snapshot

    @property
    def snapshot(self):
        if self._snapshot is None:
            self.as_snapshot()
        return self._snapshot

    @ttl_cache(ttl=1.0)
    def memory_loading_intensity(self):
        return self.loading_intensity_of(self.memory_utilization(), type='memory')

    @ttl_cache(ttl=1.0)
    def gpu_loading_intensity(self):
        return self.loading_intensity_of(self.gpu_utilization(), type='gpu')

    @ttl_cache(ttl=1.0)
    def loading_intensity(self):
        loading_intensity = (self.memory_loading_intensity(), self.gpu_loading_intensity())
        if 'heavy' in loading_intensity:
            return 'heavy'
        if 'moderate' in loading_intensity:
            return 'moderate'
        return 'light'

    @ttl_cache(ttl=1.0)
    def display_color(self):
        return self.INTENSITY2COLOR.get(self.loading_intensity())

    @ttl_cache(ttl=1.0)
    def memory_display_color(self):
        return self.INTENSITY2COLOR.get(self.memory_loading_intensity())

    @ttl_cache(ttl=1.0)
    def gpu_display_color(self):
        return self.INTENSITY2COLOR.get(self.gpu_loading_intensity())

    @staticmethod
    def loading_intensity_of(utilization, type='memory'):  # pylint: disable=redefined-builtin
        thresholds = {'memory': Device.MEMORY_UTILIZATION_THRESHOLDS,
                      'gpu': Device.GPU_UTILIZATION_THRESHOLDS}.get(type)
        if utilization == 'N/A':
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
    def color_of(utilization, type='memory'):  # pylint: disable=redefined-builtin
        return Device.INTENSITY2COLOR.get(Device.loading_intensity_of(utilization, type=type))

    _snapshot_keys = [
        'name',
        'persistence_mode', 'bus_id', 'display_active', 'ecc_errors',
        'fan_speed', 'temperature', 'performance_state',
        'power_usage', 'power_limit', 'power_state', 'compute_mode',
        'memory_used', 'memory_free', 'memory_total', 'memory_usage',
        'memory_utilization', 'memory_utilization_string',
        'gpu_utilization', 'gpu_utilization_string',
        'memory_loading_intensity', 'gpu_loading_intensity', 'loading_intensity',
        'memory_display_color', 'gpu_display_color', 'display_color'
    ]
