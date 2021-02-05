# This file is part of nvhtop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

import time
from collections import OrderedDict

import psutil
from cachetools.func import ttl_cache

from .displayable import Displayable
from .monitor import nvml_query, nvml_check_return, bytes2human, timedelta2human


try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):
        return text


def cut_string(s, maxlen, padstr='...', align='left'):
    assert align in ('left', 'right')

    if len(s) <= maxlen:
        return s
    if align == 'left':
        return s[:maxlen - len(padstr)] + padstr
    else:
        return padstr + s[-(maxlen - len(padstr)):]


class DevicePanel(Displayable):
    def __init__(self, devices, compact, win):
        super(DevicePanel, self).__init__(win)

        self.devices = devices
        self.device_count = len(self.devices)
        self._compact = compact
        self.width = 79
        self.height = 4 + (3 - int(compact)) * (self.device_count + 1)

        self.driver_version = str(nvml_query('nvmlSystemGetDriverVersion'))
        cuda_version = nvml_query('nvmlSystemGetCudaDriverVersion')
        if nvml_check_return(cuda_version, int):
            self.cuda_version = str(cuda_version // 1000 + (cuda_version % 1000) / 100)
        else:
            self.cuda_version = 'N/A'

    @property
    def compact(self):
        return self._compact

    @compact.setter
    def compact(self, value):
        if self._compact != value:
            self.need_redraw = True
            self._compact = value
            self.height = 4 + (3 - int(self.compact)) * (self.device_count + 1)

    def draw(self):
        self.color_reset()

        if self.need_redraw:
            frame = [
                '╒═════════════════════════════════════════════════════════════════════════════╕',
                '│ NVIDIA-SMI {0:<6}       Driver Version: {0:<6}       CUDA Version: {1:<5}    │'.format(self.driver_version,
                                                                                                          self.cuda_version),
                '├───────────────────────────────┬──────────────────────┬──────────────────────┤'
            ]
            if self.compact:
                frame.extend([
                    '│ GPU Fan Temp Perf Pwr:Usg/Cap │         Memory-Usage │ GPU-Util  Compute M. │',
                    '╞═══════════════════════════════╪══════════════════════╪══════════════════════╡'
                ])
                frame.extend(self.device_count * [
                    '│                               │                      │                      │',
                    '├───────────────────────────────┼──────────────────────┼──────────────────────┤'
                ])
            else:
                frame.extend([
                    '│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │',
                    '│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │',
                    '╞═══════════════════════════════╪══════════════════════╪══════════════════════╡'
                ])
                frame.extend(self.device_count * [
                    '│                               │                      │                      │',
                    '│                               │                      │                      │',
                    '├───────────────────────────────┼──────────────────────┼──────────────────────┤'
                ])
            frame.pop()
            frame.append('╘═══════════════════════════════╧══════════════════════╧══════════════════════╛')

            for y, line in enumerate(frame, start=self.y + 1):
                self.addstr(y, self.x, line)

        self.addstr(self.y, self.x, '{:<79}'.format(time.strftime('%a %b %d %H:%M:%S %Y')))

        for device in self.devices:
            device_info = device.as_dict()

            self.color(device_info['color'])
            if self.compact:
                y = self.y + 4 + 2 * (device.index + 1)
                self.addstr(y, self.x + 2,
                            '{:>3} {:>3} {:>4} {:>3} {:>12}'.format(device_info['index'],
                                                                    device_info['fan_speed'],
                                                                    device_info['temperature'],
                                                                    device_info['performance_state'],
                                                                    device_info['power_state']))
                self.addstr(y, self.x + 34, '{:>20}'.format(device_info['memory_usage']))
                self.addstr(y, self.x + 57, '{:>7}  {:>11}'.format(device_info['gpu_utilization'],
                                                                   device_info['compute_mode']))

            else:
                y = self.y + 4 + 3 * (device.index + 1)
                self.addstr(y, self.x + 2,
                            '{:>3}  {:>18}  {:<4}'.format(device_info['index'],
                                                          cut_string(device_info['name'], maxlen=18),
                                                          device_info['persistence_mode']))
                self.addstr(y, self.x + 34, '{:<16} {:>3}'.format(device_info['bus_id'],
                                                                  device_info['display_active']))
                self.addstr(y, self.x + 57, '{:>20}'.format(device_info['ecc_errors']))
                self.addstr(y + 1, self.x + 2,
                            '{:>3}  {:>4}  {:>4}  {:>12}'.format(device_info['fan_speed'],
                                                                 device_info['temperature'],
                                                                 device_info['performance_state'],
                                                                 device_info['power_state']))
                self.addstr(y + 1, self.x + 34, '{:>20}'.format(device_info['memory_usage']))
                self.addstr(y + 1, self.x + 57, '{:>7}  {:>11}'.format(device_info['gpu_utilization'],
                                                                       device_info['compute_mode']))

    def finalize(self):
        self.need_redraw = False
        self.color_reset()

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

        print('\n'.join(lines))


class ProcessPanel(Displayable):
    def __init__(self, devices, win=None):
        super(ProcessPanel, self).__init__(win)
        self.width = 79
        self.height = 6

        self.devices = devices

    def update_height(self):
        processes = self.processes
        n_processes = len(processes)
        n_used_devices = len(set([p.device.index for p in processes.values()]))
        if n_processes > 0:
            height = 5 + len(processes) + n_used_devices - 1
        else:
            height = 6

        self.need_redraw = (self.need_redraw or self.height > height)
        self.height = height

    @property
    @ttl_cache(ttl=1.0)
    def processes(self):
        processes = {}
        for device in self.devices:
            processes.update(device.processes)
        processes = sorted(processes.values(), key=lambda p: (p.device.index, p.user, p.pid))
        return OrderedDict([(p.pid, p) for p in processes])

    def draw(self):
        self.color_reset()

        if self.need_redraw:
            header = [
                '╒═════════════════════════════════════════════════════════════════════════════╕',
                '│ Processes:                                                                  │',
                '│ GPU    PID    USER  GPU MEM  %CPU  %MEM      TIME  COMMAND                  │',
                '╞═════════════════════════════════════════════════════════════════════════════╡'
            ]

            for y, line in enumerate(header, start=self.y):
                self.addstr(y, self.x, line)

        processes = self.processes
        y = self.y + 4
        if len(processes) > 0:
            prev_device_index = None
            color = None
            for proc in processes.values():
                try:
                    proc_info = proc.as_dict()
                except psutil.Error:
                    continue
                device_index = proc_info['device'].index
                if prev_device_index is None or prev_device_index != device_index:
                    color = proc_info['device'].color
                try:
                    cmdline = proc_info['cmdline']
                    cmdline[0] = proc_info['name']
                except IndexError:
                    cmdline = ['Terminated']
                cmdline = cut_string(' '.join(cmdline).strip(), maxlen=24)

                if prev_device_index is not None and prev_device_index != device_index:
                    self.addstr(
                        y, self.x, '├─────────────────────────────────────────────────────────────────────────────┤')
                    y += 1
                prev_device_index = device_index
                self.addstr(y, self.x, '│ ')
                self.color(color)
                self.addstr(y, self.x + 2, '{:>3}'.format(device_index))
                self.color_reset()
                self.addstr(y, self.x + 5,
                            ' {:>6} {:>7} {:>8} {:>5.1f} {:>5.1f}  {:>8}  {:<24} │'.format(
                                proc.pid, cut_string(proc_info['username'], maxlen=7, padstr='+'),
                                bytes2human(proc_info['gpu_memory']), proc_info['cpu_percent'],
                                proc_info['memory_percent'], timedelta2human(proc_info['running_time']),
                                cmdline
                            ))
                y += 1
            y -= 1
        else:
            self.addstr(y, self.x, '│  No running compute processes found                                         │')
        self.addstr(y + 1, self.x, '╘═════════════════════════════════════════════════════════════════════════════╛')

    def finalize(self):
        self.need_redraw = False
        self.color_reset()

    def print(self):
        lines = [
            '╒═════════════════════════════════════════════════════════════════════════════╕',
            '│ Processes:                                                                  │',
            '│ GPU    PID    USER  GPU MEM  %CPU  %MEM      TIME  COMMAND                  │',
            '╞═════════════════════════════════════════════════════════════════════════════╡'
        ]

        processes = self.processes
        if len(processes) > 0:
            prev_device_index = None
            color = None
            for proc in processes.values():
                try:
                    proc_info = proc.as_dict()
                except psutil.Error:
                    continue
                device_index = proc_info['device'].index
                if prev_device_index is None or prev_device_index != device_index:
                    color = proc_info['device'].color
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
