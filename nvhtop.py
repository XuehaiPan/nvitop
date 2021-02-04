#!/usr/bin/python3

# To Run:
# $ python3 nvhtop.py

import argparse
import datetime
import sys
import time
from contextlib import contextmanager

import psutil
from cachetools.func import ttl_cache
import pynvml as nvml


try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):
        return text

COLOR_GREEN = 0
COLOR_YELLOW = 0
COLOR_RED = 0


@contextmanager
def libcurses():
    try:
        import curses
    except ImportError:
        yield None, None
        return

    win = curses.initscr()
    win.nodelay(True)
    curses.noecho()
    curses.cbreak()
    curses.curs_set(False)

    default = -1
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    for i, color_name in enumerate(['COLOR_GREEN', 'COLOR_YELLOW', 'COLOR_RED'], start=1):
        try:
            curses.init_pair(i, getattr(curses, color_name), default)
            globals()[color_name] = curses.color_pair(i)
        except curses.error:
            pass

    yield curses, win
    curses.endwin()


def bytes2human(x):
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


def cut_string(s, maxlen, padstr='...', align='left'):
    assert align in ('left', 'right')

    if len(s) <= maxlen:
        return s
    if align == 'left':
        return s[:maxlen - len(padstr)] + padstr
    else:
        return padstr + s[-(maxlen - len(padstr)):]


def nvml_query(func, *args, **kwargs):
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


class GProcess(psutil.Process):
    def __init__(self, pid, device, gpu_memory, proc_type='C'):
        super(GProcess, self).__init__(pid)
        super(GProcess, self).cpu_percent()
        self.user = self.username()
        self.device = device
        self.gpu_memory = gpu_memory
        self.proc_type = proc_type

    @ttl_cache(ttl=5.0)
    def as_dict(self):
        proc_info = {
            'device': self.device,
            'gpu_memory': self.gpu_memory,
            'proc_type': self.proc_type,
            'running_time': datetime.datetime.now() - datetime.datetime.fromtimestamp(self.create_time())
        }
        proc_info.update(super(GProcess, self).as_dict())
        return proc_info


@ttl_cache(ttl=30.0)
def get_gpu_process(pid, device):
    return GProcess(pid, device, gpu_memory=0, proc_type='')


class Device(object):
    MEMORY_UTILIZATION_THRESHOLD_LIGHT = 5
    MEMORY_UTILIZATION_THRESHOLD_MODERATE = 90
    GPU_UTILIZATION_THRESHOLD_LIGHT = 5
    GPU_UTILIZATION_THRESHOLD_MODERATE = 75

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
    def load(self):
        gpu_utilization = nvml_query(lambda handle: nvml.nvmlDeviceGetUtilizationRates(handle).gpu, self.handle)
        memory_used = self.memory_used
        memory_total = self.memory_total
        if nvml_check_return(memory_used, int) and nvml_check_return(memory_total, int):
            memory_utilization = 100 * memory_used // memory_total
        else:
            memory_utilization = 'N/A'

        if not (nvml_check_return(gpu_utilization, int) and nvml_check_return(memory_utilization, int)):
            return 'heavy'
        if gpu_utilization >= self.GPU_UTILIZATION_THRESHOLD_MODERATE or memory_utilization >= self.MEMORY_UTILIZATION_THRESHOLD_MODERATE:
            return 'heavy'
        if gpu_utilization >= self.GPU_UTILIZATION_THRESHOLD_LIGHT or memory_utilization >= self.MEMORY_UTILIZATION_THRESHOLD_LIGHT:
            return 'moderate'
        return 'light'

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
        processes = {}

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
    def as_dict(self):
        return {key: getattr(self, key) for key in self._as_dict_keys}

    _as_dict_keys = ['index', 'name', 'load',
                     'persistence_mode', 'bus_id', 'display_active', 'ecc_errors',
                     'fan_speed', 'temperature', 'performance_state',
                     'power_usage', 'power_limit', 'power_state',
                     'memory_used', 'memory_total', 'memory_usage',
                     'gpu_utilization', 'compute_mode']


class Top(object):
    def __init__(self, mode='auto', curses=None, win=None):
        assert mode in ('auto', 'full', 'compact')
        self.mode = mode
        self.compact = (mode == 'compact')

        self.driver_version = str(nvml_query(nvml.nvmlSystemGetDriverVersion))
        cuda_version = nvml_query(nvml.nvmlSystemGetCudaDriverVersion)
        if nvml_check_return(cuda_version, int):
            self.cuda_version = str(cuda_version // 1000 + (cuda_version % 1000) / 100)
        else:
            self.cuda_version = 'N/A'

        self.device_count = nvml.nvmlDeviceGetCount()
        self.devices = list(map(Device, range(self.device_count)))

        self.curses = curses
        self.win = win
        self.termsize = None
        self.n_lines = 0

    def redraw(self):
        need_clear = False

        n_term_lines, _ = termsize = self.win.getmaxyx()
        if self.mode == 'auto':
            n_used_devices = 0
            processes = {}
            for device in self.devices:
                device_processes = device.processes
                if len(device_processes) > 0:
                    processes.update(device.processes)
                    n_used_devices += 1
            if n_used_devices > 0:
                self.compact = (n_term_lines < 7 + 3 * self.device_count + 6 + len(processes) + n_used_devices - 1)
            else:
                self.compact = (n_term_lines < 7 + 3 * self.device_count + 7)

        lines = [
            '{:<79}'.format(time.strftime('%a %b %d %H:%M:%S %Y')),
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ NVIDIA-SMI {0:<6}       Driver Version: {0:<6}       CUDA Version: {1:<5}    │'.format(self.driver_version,
                                                                                                      self.cuda_version),
            '├───────────────────────────────┬──────────────────────┬──────────────────────┤'
        ]
        if self.compact:
            lines.append('│ GPU Fan Temp Perf Pwr:Usg/Cap │         Memory-Usage │ GPU-Util  Compute M. │')
        else:
            lines.extend([
                '│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │',
                '│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │'
            ])
        lines.append('╞═══════════════════════════════╪══════════════════════╪══════════════════════╡')

        processes = {}
        for device in self.devices:
            device_info = device.as_dict()

            attr = {
                'light': COLOR_GREEN,
                'moderate': COLOR_YELLOW,
                'heavy': COLOR_RED
            }.get(device_info['load'])
            if self.compact:
                lines.append((
                    '│ {:>3} {:>3} {:>4} {:>3} {:>12} │ {:>20} │ {:>7}  {:>11} │'.format(
                        device_info['index'],
                        device_info['fan_speed'],
                        device_info['temperature'],
                        device_info['performance_state'],
                        device_info['power_state'],
                        device_info['memory_usage'],
                        device_info['gpu_utilization'],
                        device_info['compute_mode']
                    ),
                    attr
                ))
            else:
                lines.extend([
                    (
                        '│ {:>3}  {:>18}  {:<4} │ {:<16} {:>3} │ {:>20} │'.format(
                            device_info['index'],
                            cut_string(device_info['name'], maxlen=18),
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
                            device_info['power_state'],
                            device_info['memory_usage'],
                            device_info['gpu_utilization'],
                            device_info['compute_mode']
                        ),
                        attr
                    )
                ])
            lines.append('├───────────────────────────────┼──────────────────────┼──────────────────────┤')

            device_processes = device.processes
            if len(device_processes) > 0:
                processes.update(device.processes)
        lines.pop()
        lines.append('╘═══════════════════════════════╧══════════════════════╧══════════════════════╛')

        lines.extend([
            '                                                                               ',
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ Processes:                                                                  │',
            '│ GPU    PID    USER  GPU MEM  %CPU  %MEM      TIME  COMMAND                  │',
            '╞═════════════════════════════════════════════════════════════════════════════╡'
        ])

        if len(processes) > 0:
            processes = sorted(processes.values(), key=lambda p: (p.device.index, p.user, p.pid))
            prev_device_index = None
            attr = 0
            for proc in processes:
                try:
                    proc_info = proc.as_dict()
                except psutil.Error:
                    need_clear = True
                    continue
                device_index = proc_info['device'].index
                if prev_device_index is None or prev_device_index != device_index:
                    attr = {
                        'light': COLOR_GREEN,
                        'moderate': COLOR_YELLOW,
                        'heavy': COLOR_RED
                    }.get(proc_info['device'].load)
                try:
                    cmdline = proc_info['cmdline']
                    cmdline[0] = proc_info['name']
                except IndexError:
                    cmdline = ['Terminated']
                cmdline = cut_string(' '.join(cmdline).strip(), maxlen=24)

                if prev_device_index is not None and prev_device_index != device_index:
                    lines.append('├─────────────────────────────────────────────────────────────────────────────┤')
                prev_device_index = device_index
                lines.append((
                    '│ {:>3} {:>6} {:>7} {:>8} {:>5.1f} {:>5.1f}  {:>8}  {:<24} │'.format(
                        device_index, proc.pid,
                        cut_string(proc_info['username'], maxlen=7, padstr='+'),
                        bytes2human(proc_info['gpu_memory']), proc_info['cpu_percent'], proc_info['memory_percent'],
                        timedelta2human(proc_info['running_time']), cmdline
                    ),
                    attr
                ))
        else:
            lines.append('│  No running compute processes found                                         │')

        lines.append('╘═════════════════════════════════════════════════════════════════════════════╛')

        if need_clear or len(lines) < self.n_lines or termsize != self.termsize:
            self.win.clear()
        self.n_lines = len(lines)
        self.termsize = termsize
        for y, line in enumerate(lines):
            try:
                if isinstance(line, str):
                    self.win.addstr(y, 0, line)
                else:
                    line, attr = line
                    for x, ch in enumerate(line):
                        try:
                            if ch not in '│ ':
                                self.win.addstr(y, x, ch, attr)
                            else:
                                self.win.addstr(y, x, ch)
                        except self.curses.error:
                            if x == 0:
                                raise
                            else:
                                break
            except self.curses.error:
                break
        self.win.refresh()

    def loop(self):
        if self.win is None:
            return

        key = -1
        while True:
            try:
                self.redraw()
                for i in range(10):
                    key = self.win.getch()
                    if key == -1 or key == ord('q'):
                        break
                self.curses.flushinp()
                if key == ord('q'):
                    break
                time.sleep(0.5)
            except KeyboardInterrupt:
                pass

    def print(self):
        lines = [
            '{:<79}'.format(time.strftime('%a %b %d %H:%M:%S %Y')),
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ NVIDIA-SMI {0:<6}       Driver Version: {0:<6}       CUDA Version: {1:<5}    │'.format(self.driver_version,
                                                                                                      self.cuda_version),
            '├───────────────────────────────┬──────────────────────┬──────────────────────┤',
            '│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │',
            '│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │',
            '╞═══════════════════════════════╪══════════════════════╪══════════════════════╡'
        ]

        processes = {}
        for device in self.devices:
            color = {'light': 'green', 'moderate': 'yellow', 'heavy': 'red'}.get(device.load)
            line1 = '│ {:>3}  {:>18}  {:<4} │ {:<16} {:>3} │ {:>20} │'.format(device.index,
                                                                              cut_string(device.name, maxlen=18),
                                                                              device.persistence_mode,
                                                                              device.bus_id,
                                                                              device.display_active,
                                                                              device.ecc_errors)
            line2 = '│ {:>3}  {:>4}  {:>4}  {:>12} │ {:>20} │ {:>7}  {:>11} │'.format(device.fan_speed,
                                                                                      device.temperature,
                                                                                      device.performance_state,
                                                                                      device.power_state,
                                                                                      device.memory_usage,
                                                                                      device.gpu_utilization,
                                                                                      device.compute_mode)
            lines.extend([
                '│'.join(map(lambda s: colored(s, color), line1.split('│'))),
                '│'.join(map(lambda s: colored(s, color), line2.split('│')))
            ])
            lines.append('├───────────────────────────────┼──────────────────────┼──────────────────────┤')

            device_processes = device.processes
            if len(device_processes) > 0:
                processes.update(device.processes)
        lines.pop()
        lines.append('╘═══════════════════════════════╧══════════════════════╧══════════════════════╛')

        lines.extend([
            '                                                                               ',
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ Processes:                                                                  │',
            '│ GPU    PID    USER  GPU MEM  %CPU  %MEM      TIME  COMMAND                  │',
            '╞═════════════════════════════════════════════════════════════════════════════╡'
        ])

        if len(processes) > 0:
            processes = sorted(processes.values(), key=lambda p: (p.device.index, p.user, p.pid))
            prev_device_index = None
            color = None
            for proc in processes:
                try:
                    proc_info = proc.as_dict()
                except psutil.Error:
                    continue
                device_index = proc_info['device'].index
                if prev_device_index is None or prev_device_index != device_index:
                    color = {'light': 'green', 'moderate': 'yellow', 'heavy': 'red'}.get(proc_info['device'].load)
                try:
                    cmdline = proc_info['cmdline']
                    cmdline[0] = proc_info['name']
                except IndexError:
                    cmdline = ['Terminated']
                cmdline = cut_string(' '.join(cmdline).strip(), maxlen=24)

                if prev_device_index is not None and prev_device_index != device_index:
                    lines.append('├─────────────────────────────────────────────────────────────────────────────┤')
                prev_device_index = device_index
                lines.append('│ {} {:>6} {:>7} {:>8} {:>5.1f} {:>5.1f}  {:>8}  {:<24} │'.format(
                    colored('{:>3}'.format(device_index), color), proc.pid,
                    cut_string(proc_info['username'], maxlen=7, padstr='+'),
                    bytes2human(proc_info['gpu_memory']), proc_info['cpu_percent'], proc_info['memory_percent'],
                    timedelta2human(proc_info['running_time']), cmdline
                ))
        else:
            lines.append('│  No running compute processes found                                         │')

        lines.append('╘═════════════════════════════════════════════════════════════════════════════╛')

        print('\n'.join(lines))


def main():
    try:
        nvml.nvmlInit()
    except nvml.NVMLError_LibraryNotFound as error:  # pylint: disable=no-member
        print(error, file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(prog='nvhtop.py', description='A interactive Nvidia-GPU process viewer.')
    parser.add_argument('-m', '--monitor', type=str, default='notpresented',
                        nargs='?', choices=['auto', 'full', 'compact'],
                        help='Run as a resource monitor. '
                             'Continuously report query data, rather than the default of just once. '
                             'If no argument is specified, the default mode `auto` is used.')
    args = parser.parse_args()
    if args.monitor is None:
        args.monitor = 'auto'

    if args.monitor != 'notpresented':
        with libcurses() as (curses, win):
            top = Top(mode=args.monitor, curses=curses, win=win)
            top.loop()
    else:
        top = Top()
    top.print()

    nvml.nvmlShutdown()


if __name__ == '__main__':
    sys.exit(main())
