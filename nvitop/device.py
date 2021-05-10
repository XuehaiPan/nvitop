# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

from collections import OrderedDict

import psutil
import pynvml as nvml
from cachetools.func import ttl_cache

from .history import BufferedHistoryGraph
from .process import GpuProcess
from .utils import Snapshot, bytes2human, nvml_check_return, nvml_query


class Device(object):
    MEMORY_UTILIZATION_THRESHOLDS = (10, 80)
    GPU_UTILIZATION_THRESHOLDS = (10, 75)
    INTENSITY2COLOR = {'light': 'green', 'moderate': 'yellow', 'heavy': 'red'}

    def __init__(self, index):
        self.index = index
        try:
            self.handle = nvml.nvmlDeviceGetHandleByIndex(index)
        except nvml.NVMLError_GpuIsLost:  # pylint: disable=no-member
            self.handle = None

        self._name = nvml_query(nvml.nvmlDeviceGetName, self.handle)
        self._bus_id = nvml_query(lambda handle: nvml.nvmlDeviceGetPciInfo(handle).busId, self.handle)
        self._memory_total = nvml_query(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).total, self.handle)
        self._memory_total_human = bytes2human(self.memory_total())
        self._power_limit = nvml_query(nvml.nvmlDeviceGetPowerManagementLimit, self.handle)

        self._ident = (self.index, self.bus_id())
        self._hash = None
        self._last_snapshot = None

        def get_value(value):
            if value != 'N/A':
                value = float(value[:-1])
            return value

        memory_prefix = 'GPU ' + str(self.index) + ' MEM: '
        self.memory_utilization = BufferedHistoryGraph(
            baseline=0.0,
            upperbound=100.0,
            width=32,
            height=5,
            dynamic_bound=False,
            format=lambda x: memory_prefix + str(int(x)) + '%'
        )(
            self.memory_utilization,
            get_value=get_value
        )
        gpu_prefix = 'GPU ' + str(self.index) + ' UTL: '
        self.gpu_utilization = BufferedHistoryGraph(
            baseline=0.0,
            upperbound=100.0,
            width=32,
            height=5,
            dynamic_bound=False,
            upsidedown=True,
            format=lambda x: gpu_prefix + str(int(x)) + '%'
        )(
            self.gpu_utilization,
            get_value=get_value
        )

    def __str__(self):
        return 'GPU({}, {}, {})'.format(self.index, self.name(), self.memory_total_human())

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

    def name(self):
        return self._name

    def bus_id(self):
        return self._bus_id

    def memory_total(self):
        return self._memory_total

    def memory_total_human(self):
        return self._memory_total_human

    def power_limit(self):
        return self._power_limit

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
        elif 'moderate' in loading_intensity:
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

    @ttl_cache(ttl=60.0)
    def display_active(self):
        return {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetDisplayActive, self.handle), 'N/A')

    @ttl_cache(ttl=60.0)
    def persistence_mode(self):
        return {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetPersistenceMode, self.handle), 'N/A')

    @ttl_cache(ttl=5.0)
    def ecc_errors(self):
        return nvml_query(nvml.nvmlDeviceGetTotalEccErrors, self.handle,
                          nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                          nvml.NVML_VOLATILE_ECC)

    @ttl_cache(ttl=60.0)
    def compute_mode(self):
        return {
            nvml.NVML_COMPUTEMODE_DEFAULT: 'Default',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: 'E. Thread',
            nvml.NVML_COMPUTEMODE_PROHIBITED: 'Prohibited',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: 'E. Process',
        }.get(nvml_query(nvml.nvmlDeviceGetComputeMode, self.handle), 'N/A')

    @ttl_cache(ttl=5.0)
    def performance_state(self):
        performance_state = nvml_query(nvml.nvmlDeviceGetPerformanceState, self.handle)
        if nvml_check_return(performance_state, int):
            performance_state = 'P' + str(performance_state)
        return performance_state

    @ttl_cache(ttl=5.0)
    def power_usage(self):
        return nvml_query(nvml.nvmlDeviceGetPowerUsage, self.handle)

    @ttl_cache(ttl=5.0)
    def power_state(self):
        power_usage = self.power_usage()
        power_limit = self.power_limit()
        if nvml_check_return(power_usage, int) and nvml_check_return(power_limit, int):
            return '{}W / {}W'.format(power_usage // 1000, power_limit // 1000)
        return 'N/A'

    @ttl_cache(ttl=5.0)
    def fan_speed(self):
        fan_speed = nvml_query(nvml.nvmlDeviceGetFanSpeed, self.handle)
        if nvml_check_return(fan_speed, int):
            if fan_speed < 100:
                fan_speed = str(fan_speed) + '%'
            else:
                fan_speed = 'MAX'
        return fan_speed

    @ttl_cache(ttl=5.0)
    def temperature(self):
        temperature = nvml_query(nvml.nvmlDeviceGetTemperature, self.handle, nvml.NVML_TEMPERATURE_GPU)
        if nvml_check_return(temperature, int):
            temperature = str(temperature) + 'C'
        return temperature

    @ttl_cache(ttl=1.0)
    def memory_used(self):
        return nvml_query(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).used, self.handle)

    @ttl_cache(ttl=1.0)
    def memory_usage(self):
        memory_used = self.memory_used()
        memory_total = self.memory_total()
        if nvml_check_return(memory_used, int) and nvml_check_return(memory_total, int):
            return '{} / {}'.format(bytes2human(memory_used), bytes2human(memory_total))
        return 'N/A'

    @ttl_cache(ttl=1.0)
    def memory_utilization(self):
        memory_used = self.memory_used()
        memory_total = self.memory_total()
        if nvml_check_return(memory_used, int) and nvml_check_return(memory_total, int):
            return str(100 * memory_used // memory_total) + '%'
        return 'N/A'

    @ttl_cache(ttl=1.0)
    def gpu_utilization(self):
        gpu_utilization = nvml_query(lambda handle: nvml.nvmlDeviceGetUtilizationRates(handle).gpu, self.handle)
        if nvml_check_return(gpu_utilization, int):
            return str(gpu_utilization) + '%'
        return 'N/A'

    @ttl_cache(ttl=2.0)
    def processes(self):
        processes = OrderedDict()

        for type, func in [('C', nvml.nvmlDeviceGetComputeRunningProcesses),  # pylint: disable=redefined-builtin
                           ('G', nvml.nvmlDeviceGetGraphicsRunningProcesses)]:
            try:
                running_processes = func(self.handle)
            except nvml.NVMLError:
                pass
            else:
                for p in running_processes:
                    try:
                        proc = processes[p.pid] = GpuProcess(pid=p.pid, device=self)
                    except psutil.Error:
                        try:
                            del processes[p.pid]
                        except KeyError:
                            pass
                        continue
                    else:
                        proc.set_gpu_memory(p.usedGpuMemory if isinstance(p.usedGpuMemory, int) else 'N/A')
                        proc.type = proc.type + type

        return processes

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

    @ttl_cache(ttl=1.0)
    def take_snapshot(self):
        self._last_snapshot = Snapshot(real=self, index=self.index,
                                       **{key: getattr(self, key)() for key in self._snapshot_keys})
        return self._last_snapshot

    @property
    def last_snapshot(self):
        if self._last_snapshot is None:
            self.take_snapshot()
        return self._last_snapshot

    _snapshot_keys = [
        'name',
        'persistence_mode', 'bus_id', 'display_active', 'ecc_errors',
        'fan_speed', 'temperature', 'performance_state',
        'power_usage', 'power_limit', 'power_state',
        'memory_used', 'memory_total', 'memory_usage', 'memory_utilization',
        'gpu_utilization', 'compute_mode',
        'memory_loading_intensity', 'gpu_loading_intensity', 'loading_intensity',
        'memory_display_color', 'gpu_display_color', 'display_color'
    ]
