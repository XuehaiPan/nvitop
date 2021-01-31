#!/usr/bin/python3

# To Run:
# $ python3 nvhtop.py

import curses
import datetime
import sys
import time

import psutil
from cachetools import cached, TTLCache

import pynvml as nvml


def bytes2human(x):
    if x < (1 << 10):
        return '{}B'.format(x)
    if x < (1 << 20):
        return '{}KiB'.format(x >> 10)
    else:
        return '{}MiB'.format(x >> 20)


def nvml_query(func, *args, **kwargs):
    try:
        retval = func(*args, **kwargs)
    except nvml.NVMLError as error:
        if error.value == nvml.NVML_ERROR_NOT_SUPPORTED:
            return 'N/A'
        else:
            return str(error)
    else:
        if isinstance(retval, bytes):
            retval = retval.decode('UTF-8')
        return retval


class GProcess(psutil.Process):
    def __init__(self, pid, device, gpu_memory, type='C'):
        super(GProcess, self).__init__(pid)
        super(GProcess, self).cpu_percent()
        self.user = self.username()
        self.device = device
        self.gpu_memory = gpu_memory
        self.type = type

    @cached(cache=TTLCache(maxsize=128, ttl=5.0))
    def as_dict(self):
        return {
            'device': self.device,
            'pid': self.pid,
            'username': self.user,
            'gpu_memory': self.gpu_memory,
            'cpu_percent': self.cpu_percent(),
            'memory_percent': self.memory_percent(),
            'running_time': datetime.datetime.now() - datetime.datetime.fromtimestamp(self.create_time()),
            'cmdline': self.cmdline()
        }


@cached(cache=TTLCache(maxsize=128, ttl=30.0))
def get_gpu_process(pid, device):
    return GProcess(pid, device, gpu_memory=0, type='')


class Device(object):
    MEMORY_FREE_RATIO = 0.05
    MEMORY_MODERATE_RATIO = 0.9
    GPU_FREE_RATIO = 0.05
    GPU_MODERATE_RATIO = 0.75

    def __init__(self, index):
        self.index = index
        self.handle = nvml.nvmlDeviceGetHandleByIndex(index)
        self.name = nvml_query(nvml.nvmlDeviceGetName, self.handle)
        self.bus_id = nvml_query(lambda handle: nvml.nvmlDeviceGetPciInfo(handle).busId, self.handle)
        self.memory_total = nvml_query(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).total, self.handle)
        self.power_limit = nvml_query(nvml.nvmlDeviceGetPowerManagementLimit, self.handle)

    def __str__(self):
        return 'GPU({}, {}, {})'.format(self.index, self.name, bytes2human(self.memory_total))

    __repr__ = __str__

    @property
    def condition(self):
        try:
            memory_utilization = self.memory_used / self.memory_total
            gpu_utilization = float(self.utilization[:-1]) / 100
        except ValueError:
            return 'high'

        if gpu_utilization >= self.GPU_MODERATE_RATIO or memory_utilization >= self.MEMORY_MODERATE_RATIO:
            return 'high'
        if gpu_utilization >= self.GPU_FREE_RATIO or memory_utilization >= self.MEMORY_FREE_RATIO:
            return 'moderate'
        return 'free'

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=5.0))
    def display_active(self):
        return {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetDisplayActive, self.handle), 'N/A')

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=5.0))
    def persistence_mode(self):
        return {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetPersistenceMode, self.handle), 'N/A')

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=2.0))
    def ecc_errors(self):
        return nvml_query(nvml.nvmlDeviceGetTotalEccErrors, self.handle,
                          nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                          nvml.NVML_VOLATILE_ECC)

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=2.0))
    def fan_speed(self):
        fan_speed = nvml_query(nvml.nvmlDeviceGetFanSpeed, self.handle)
        if fan_speed != 'N/A':
            fan_speed = str(fan_speed) + '%'
        return fan_speed

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=2.0))
    def utilization(self):
        utilization = nvml_query(nvml.nvmlDeviceGetUtilizationRates, self.handle).gpu
        if utilization != 'N/A':
            utilization = str(utilization) + '%'
        return utilization

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=5.0))
    def compute_mode(self):
        return {
            nvml.NVML_COMPUTEMODE_DEFAULT: 'Default',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: 'E. Thread',
            nvml.NVML_COMPUTEMODE_PROHIBITED: 'Prohibited',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: 'E. Process',
        }.get(nvml_query(nvml.nvmlDeviceGetComputeMode, self.handle), 'N/A')

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=2.0))
    def temperature(self):
        temperature = nvml_query(nvml.nvmlDeviceGetTemperature, self.handle, nvml.NVML_TEMPERATURE_GPU)
        if temperature != 'N/A':
            temperature = str(temperature) + 'C'
        return temperature

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=5.0))
    def performance_state(self):
        performance_state = nvml_query(nvml.nvmlDeviceGetPerformanceState, self.handle)
        if performance_state != 'N/A':
            performance_state = 'P' + str(performance_state)
        return performance_state

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=1.0))
    def memory_used(self):
        return nvml_query(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).used, self.handle)

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=1.0))
    def power_usage(self):
        return nvml_query(nvml.nvmlDeviceGetPowerUsage, self.handle)

    @property
    @cached(cache=TTLCache(maxsize=128, ttl=2.0))
    def processes(self):
        processes = {}

        for p in nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
            try:
                proc = processes[p.pid] = get_gpu_process(pid=p.pid, device=self)
            except psutil.NoSuchProcess:
                try:
                    del processes[p.pid]
                except KeyError:
                    pass
            proc.gpu_memory = p.usedGpuMemory
            if proc.type == 'G':
                proc.type = 'C+G'
            else:
                proc.type = 'C'
        for p in nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle):
            try:
                proc = processes[p.pid] = get_gpu_process(pid=p.pid, device=self)
            except psutil.NoSuchProcess:
                try:
                    del processes[p.pid]
                except KeyError:
                    pass
            proc.gpu_memory = p.usedGpuMemory
            if proc.type == 'C':
                proc.type = 'C+G'
            else:
                proc.type = 'G'
        return processes

    @cached(cache=TTLCache(maxsize=128, ttl=1.0))
    def as_dict(self):
        return {key: getattr(self, key) for key in self._as_dict_keys}

    _as_dict_keys = ['index', 'name', 'condition',
                     'persistence_mode', 'bus_id', 'display_active', 'ecc_errors',
                     'fan_speed', 'temperature', 'performance_state',
                     'power_usage', 'power_limit',
                     'memory_used', 'memory_total',
                     'utilization', 'compute_mode']


class Top(object):
    def __init__(self):
        self.driver_version = str(nvml_query(nvml.nvmlSystemGetDriverVersion))
        self.cuda_version = str(nvml_query(nvml.nvmlSystemGetCudaDriverVersion))
        if self.cuda_version != 'N/A':
            self.cuda_version = self.cuda_version[:-3] + '.' + self.cuda_version[-2]

        self.device_count = nvml.nvmlDeviceGetCount()
        self.devices = list(map(Device, range(self.device_count)))

        self.rows = []

        self.win = None
        self.termsize = None
        self.n_rows = 0
        self.init_curses()

    def exit(self):
        curses.endwin()
        for row in self.rows:
            if not isinstance(row, str):
                row = row[0]
            print(row)

    def init_curses(self):
        COLOR_NONE = -1

        self.win = curses.initscr()
        curses.start_color()
        try:
            curses.use_default_colors()
        except curses.error:
            pass
        try:
            curses.init_pair(1, curses.COLOR_GREEN, COLOR_NONE)
        except curses.error:
            pass
        try:
            curses.init_pair(2, curses.COLOR_YELLOW, COLOR_NONE)
        except curses.error:
            pass
        try:
            curses.init_pair(3, curses.COLOR_RED, COLOR_NONE)
        except curses.error:
            pass
        curses.noecho()
        curses.cbreak()
        curses.curs_set(False)
        self.win.nodelay(True)

    def redraw(self):
        need_clear = False

        n_used_devices = 0
        processes = {}
        for device in self.devices:
            device_processes = device.processes
            if len(device_processes) > 0:
                processes.update(device.processes)
                n_used_devices += 1

        n_term_rows, n_term_cols = termsize = self.win.getmaxyx()
        if n_used_devices > 0:
            compact = (n_term_rows < 7 + 3 * self.device_count + 6 + len(processes) + n_used_devices - 1)
        else:
            compact = (n_term_rows < 7 + 3 * self.device_count + 7)

        self.rows.clear()
        self.rows.extend([
            '{:<79}'.format(time.strftime('%a %b %d %H:%M:%S %Y')),
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ NVIDIA-SMI {0:<6}       Driver Version: {0:<6}       CUDA Version: {1:<5}    │'.format(self.driver_version,
                                                                                                      self.cuda_version),
            '├───────────────────────────────┬──────────────────────┬──────────────────────┤'
        ])
        if compact:
            self.rows.append('│ GPU  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │')
        else:
            self.rows.extend([
                '│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │',
                '│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │'
            ])
        self.rows.append('╞═══════════════════════════════╪══════════════════════╪══════════════════════╡')

        for device in self.devices:
            device_info = device.as_dict()
            if len(device_info['name']) > 18:
                device_info['name'] = device_info['name'][:15] + '...'

            device_info['memory'] = '{} / {}'.format(bytes2human(device_info['memory_used']),
                                                     bytes2human(device_info['memory_total']))

            if device_info['power_usage'] != 'N/A' and device_info['power_limit'] != 'N / A':
                device_info['power'] = '{}W / {}W'.format(device_info['power_usage'] // 1000,
                                                          device_info['power_limit'] // 1000)
            else:
                device_info['power'] = 'N/A'

            attr = {
                'free': curses.color_pair(1),
                'moderate': curses.color_pair(2),
                'high': curses.color_pair(3)
            }.get(device_info['condition'])
            if compact:
                self.rows.append((
                    '│ {:>3}  {:>4}  {:>4}  {:>12} │ {:>20} │ {:>7}  {:>11} │'.format(
                        device_info['index'],
                        device_info['temperature'],
                        device_info['performance_state'],
                        device_info['power'],
                        device_info['memory'],
                        device_info['utilization'],
                        device_info['compute_mode']
                    ),
                    attr
                ))
            else:
                self.rows.extend([
                    (
                        '│ {:>3}  {:>18}  {:<4} │ {:<16} {:>3} │ {:>20} │'.format(
                            device_info['index'],
                            device_info['name'],
                            device_info['persistence_mode'],
                            device_info['bus_id'],
                            device_info['display_active'],
                            device_info['ecc_errors']
                        ),
                        attr
                    ),
                    (
                        '│ {:>3}  {:>4}  {:>4}  {:>12} │ {:>20} │ {:>7}  {:>11} │'.format(
                            device_info['fan_speed'],
                            device_info['temperature'],
                            device_info['performance_state'],
                            device_info['power'],
                            device_info['memory'],
                            device_info['utilization'],
                            device_info['compute_mode']
                        ),
                        attr
                    )
                ])
            self.rows.append('├───────────────────────────────┼──────────────────────┼──────────────────────┤')

            device_processes = device.processes
            if len(device_processes) > 0:
                processes.update(device.processes)
                n_used_devices += 1
        self.rows.pop()
        self.rows.append('╘═══════════════════════════════╧══════════════════════╧══════════════════════╛')

        self.rows.extend([
            '                                                                               ',
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ Processes:                                                                  │',
            '│ GPU    PID    USER  GPU MEM  %CPU  %MEM      TIME  COMMAND                  │',
            '╞═════════════════════════════════════════════════════════════════════════════╡'
        ])

        if len(processes) > 0:
            processes = sorted(processes.values(), key=lambda proc: (proc.device.index, proc.user, proc.pid))
            prev_device_index = None
            attr = 0
            for proc in processes:
                try:
                    proc_info = proc.as_dict()
                except psutil.NoSuchProcess:
                    need_clear = True
                    continue
                device_index = proc_info['device'].index
                if prev_device_index is None or prev_device_index != device_index:
                    attr = {
                        'free': curses.color_pair(1),
                        'moderate': curses.color_pair(2),
                        'high': curses.color_pair(3)
                    }.get(proc_info['device'].condition)
                try:
                    cmdline = proc.cmdline()
                    cmdline[0] = proc.name()
                except psutil.NoSuchProcess:
                    cmdline = proc_info['cmdline']
                except IndexError:
                    cmdline = []
                cmdline = ' '.join(cmdline).strip()
                if len(cmdline) > 24:
                    cmdline = cmdline[:21] + '...'
                if len(proc_info['username']) >= 8:
                    proc_info['username'] = proc_info['username'][:6] + '+'
                running_time = proc_info['running_time']
                if running_time.days > 1:
                    running_time = '{} days'.format(running_time.days)
                else:
                    hours, seconds = divmod(86400 * running_time.days + running_time.seconds, 3600)
                    running_time = '{:02d}:{:02d}:{:02d}'.format(hours, *divmod(seconds, 60))
                if prev_device_index is not None and prev_device_index != device_index:
                    self.rows.append('├─────────────────────────────────────────────────────────────────────────────┤')
                prev_device_index = device_index
                self.rows.append((
                    '│ {:>3} {:>6} {:>7} {:>8} {:>5.1f} {:>5.1f}  {:>8}  {:<24} │'.format(
                        device_index,
                        proc.pid,
                        proc_info['username'],
                        bytes2human(proc_info['gpu_memory']),
                        proc_info['cpu_percent'],
                        proc_info['memory_percent'],
                        running_time,
                        cmdline
                    ),
                    attr
                ))
        else:
            self.rows.append('│  No running compute processes found                                         │')

        self.rows.append('╘═════════════════════════════════════════════════════════════════════════════╛')

        if need_clear or len(self.rows) < self.n_rows or termsize != self.termsize:
            self.win.clear()
        self.n_rows = len(self.rows)
        self.termsize = termsize
        for y, row in enumerate(self.rows):
            try:
                if isinstance(row, str):
                    self.win.addstr(y, 0, row)
                else:
                    self.win.addstr(y, 0, *row)
            except curses.error:
                break
        self.win.refresh()

    def loop(self):
        key = -1
        while True:
            try:
                self.redraw()
                for i in range(10):
                    key = self.win.getch()
                    if key == -1 or key == ord('q'):
                        break
                curses.flushinp()
                if key == ord('q'):
                    break
            except KeyboardInterrupt:
                pass
            time.sleep(0.5)


def main():
    try:
        nvml.nvmlInit()
    except nvml.NVMLError as error:
        if error.value == nvml.NVML_ERROR_LIBRARY_NOT_FOUND:
            print(error, file=sys.stderr)
            exit(1)
        raise

    top = Top()

    top.loop()

    top.exit()

    nvml.nvmlShutdown()


if __name__ == '__main__':
    main()
