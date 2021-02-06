# This file is part of nvhtop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

import datetime
from collections import OrderedDict

import psutil
from cachetools.func import ttl_cache

import pynvml as nvml


def bytes2human(x):
    if x == 'N/A':
        return x

    if x < (1 << 10):
        return '{}B'.format(x)
    if x < (1 << 20):
        return '{}KiB'.format(x >> 10)
    else:
        return '{}MiB'.format(x >> 20)


def timedelta2human(dt):
    if dt.days > 1:
        return '{} days'.format(dt.days)
    else:
        hours, seconds = divmod(86400 * dt.days + dt.seconds, 3600)
        return '{:02d}:{:02d}:{:02d}'.format(hours, *divmod(seconds, 60))


def nvml_query(func, *args, **kwargs):
    if isinstance(func, str):
        func = getattr(nvml, func)

    try:
        retval = func(*args, **kwargs)
    except nvml.NVMLError:
        return 'N/A'
    else:
        if isinstance(retval, bytes):
            retval = retval.decode('UTF-8')
        return retval


def nvml_check_return(retval, types=None):
    if types is None:
        return (retval != 'N/A')
    else:
        return (retval != 'N/A' and isinstance(retval, types))


class Snapshot(object):
    def __init__(self, **items):
        for key, value in items.items():
            setattr(self, key, value)

    def __bool__(self):
        return bool(self.__dict__)


class GProcess(psutil.Process):
    def __init__(self, pid, device, gpu_memory, proc_type='C'):
        super(GProcess, self).__init__(pid)
        super(GProcess, self).cpu_percent()
        self.device = device
        self.gpu_memory = gpu_memory
        self.proc_type = proc_type

    @ttl_cache(ttl=5.0)
    def snapshot(self):
        try:
            snapshot = Snapshot(
                device=self.device,
                gpu_memory=self.gpu_memory,
                proc_type=self.proc_type,
                running_time=datetime.datetime.now() - datetime.datetime.fromtimestamp(self.create_time())
            )
            snapshot.__dict__.update(super(GProcess, self).as_dict())
        except psutil.Error:
            return None
        else:
            return snapshot


@ttl_cache(ttl=30.0)
def get_gpu_process(pid, device):
    return GProcess(pid, device, gpu_memory=0, proc_type='')


class Device(object):
    MEMORY_UTILIZATION_THRESHOLDS = (10, 80)
    GPU_UTILIZATION_THRESHOLDS = (10, 75)

    def __init__(self, index):
        self.index = index
        try:
            self.handle = nvml.nvmlDeviceGetHandleByIndex(index)
        except nvml.NVMLError_GpuIsLost:  # pylint: disable=no-member
            self.handle = None

        self.name = nvml_query(nvml.nvmlDeviceGetName, self.handle)
        self.bus_id = nvml_query(lambda handle: nvml.nvmlDeviceGetPciInfo(handle).busId, self.handle)
        self.memory_total = nvml_query(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).total, self.handle)
        self.power_limit = nvml_query(nvml.nvmlDeviceGetPowerManagementLimit, self.handle)

    def __str__(self):
        return 'GPU({}, {}, {})'.format(self.index, self.name, bytes2human(self.memory_total))

    __repr__ = __str__

    @property
    @ttl_cache(ttl=1.0)
    def loading_intensity(self):
        gpu_utilization = nvml_query(lambda handle: nvml.nvmlDeviceGetUtilizationRates(handle).gpu, self.handle)
        memory_used = self.memory_used
        memory_total = self.memory_total
        if nvml_check_return(memory_used, int) and nvml_check_return(memory_total, int):
            memory_utilization = 100 * memory_used // memory_total
        else:
            memory_utilization = 'N/A'

        if not (nvml_check_return(gpu_utilization, int) and nvml_check_return(memory_utilization, int)):
            return 'heavy'
        if gpu_utilization >= self.GPU_UTILIZATION_THRESHOLDS[-1] or memory_utilization >= self.MEMORY_UTILIZATION_THRESHOLDS[-1]:
            return 'heavy'
        if gpu_utilization >= self.GPU_UTILIZATION_THRESHOLDS[0] or memory_utilization >= self.MEMORY_UTILIZATION_THRESHOLDS[0]:
            return 'moderate'
        return 'light'

    @property
    @ttl_cache(ttl=1.0)
    def display_color(self):
        return {'light': 'green', 'moderate': 'yellow', 'heavy': 'red'}.get(self.loading_intensity)

    @property
    @ttl_cache(ttl=60.0)
    def display_active(self):
        return {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetDisplayActive, self.handle), 'N/A')

    @property
    @ttl_cache(ttl=60.0)
    def persistence_mode(self):
        return {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetPersistenceMode, self.handle), 'N/A')

    @property
    @ttl_cache(ttl=5.0)
    def ecc_errors(self):
        return nvml_query(nvml.nvmlDeviceGetTotalEccErrors, self.handle,
                          nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                          nvml.NVML_VOLATILE_ECC)

    @property
    @ttl_cache(ttl=60.0)
    def compute_mode(self):
        return {
            nvml.NVML_COMPUTEMODE_DEFAULT: 'Default',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: 'E. Thread',
            nvml.NVML_COMPUTEMODE_PROHIBITED: 'Prohibited',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: 'E. Process',
        }.get(nvml_query(nvml.nvmlDeviceGetComputeMode, self.handle), 'N/A')

    @property
    @ttl_cache(ttl=5.0)
    def performance_state(self):
        performance_state = nvml_query(nvml.nvmlDeviceGetPerformanceState, self.handle)
        if nvml_check_return(performance_state, int):
            performance_state = 'P' + str(performance_state)
        return performance_state

    @property
    @ttl_cache(ttl=5.0)
    def power_usage(self):
        return nvml_query(nvml.nvmlDeviceGetPowerUsage, self.handle)

    @property
    @ttl_cache(ttl=5.0)
    def power_state(self):
        power_usage = self.power_usage
        power_limit = self.power_limit
        if nvml_check_return(power_usage, int) and nvml_check_return(power_limit, int):
            return '{}W / {}W'.format(power_usage // 1000, power_limit // 1000)
        else:
            return 'N/A'

    @property
    @ttl_cache(ttl=5.0)
    def fan_speed(self):
        fan_speed = nvml_query(nvml.nvmlDeviceGetFanSpeed, self.handle)
        if nvml_check_return(fan_speed, int):
            fan_speed = str(fan_speed) + '%'
        return fan_speed

    @property
    @ttl_cache(ttl=5.0)
    def temperature(self):
        temperature = nvml_query(nvml.nvmlDeviceGetTemperature, self.handle, nvml.NVML_TEMPERATURE_GPU)
        if nvml_check_return(temperature, int):
            temperature = str(temperature) + 'C'
        return temperature

    @property
    @ttl_cache(ttl=1.0)
    def memory_used(self):
        return nvml_query(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).used, self.handle)

    @property
    @ttl_cache(ttl=1.0)
    def memory_usage(self):
        memory_used = self.memory_used
        memory_total = self.memory_total
        if nvml_check_return(memory_used, int) and nvml_check_return(memory_total, int):
            return '{} / {}'.format(bytes2human(memory_used), bytes2human(memory_total))
        else:
            return 'N/A'

    @property
    @ttl_cache(ttl=1.0)
    def memory_utilization(self):
        memory_used = self.memory_used
        memory_total = self.memory_total
        if nvml_check_return(memory_used, int) and nvml_check_return(memory_total, int):
            return str(100 * memory_used // memory_total) + '%'
        else:
            return 'N/A'

    @property
    @ttl_cache(ttl=1.0)
    def gpu_utilization(self):
        gpu_utilization = nvml_query(lambda handle: nvml.nvmlDeviceGetUtilizationRates(handle).gpu, self.handle)
        if nvml_check_return(gpu_utilization, int):
            gpu_utilization = str(gpu_utilization) + '%'
        return gpu_utilization

    @property
    @ttl_cache(ttl=2.0)
    def processes(self):
        processes = OrderedDict()

        for proc_type, func in [('C', nvml.nvmlDeviceGetComputeRunningProcesses),
                                ('G', nvml.nvmlDeviceGetComputeRunningProcesses)]:
            try:
                running_processes = func(self.handle)
            except nvml.NVMLError:
                pass
            else:
                for p in running_processes:
                    try:
                        proc = processes[p.pid] = get_gpu_process(pid=p.pid, device=self)
                    except psutil.Error:
                        try:
                            del processes[p.pid]
                        except KeyError:
                            pass
                        continue
                    proc.gpu_memory = p.usedGpuMemory
                    if proc.proc_type != proc_type:
                        proc.proc_type = 'C+G'
                    else:
                        proc.proc_type = proc_type

        return processes

    @ttl_cache(ttl=1.0)
    def snapshot(self):
        return Snapshot(**{key: getattr(self, key) for key in self._snapshot_keys})

    _snapshot_keys = ['index', 'name', 'loading_intensity', 'display_color',
                      'persistence_mode', 'bus_id', 'display_active', 'ecc_errors',
                      'fan_speed', 'temperature', 'performance_state',
                      'power_usage', 'power_limit', 'power_state',
                      'memory_used', 'memory_total', 'memory_usage',
                      'gpu_utilization', 'compute_mode']
