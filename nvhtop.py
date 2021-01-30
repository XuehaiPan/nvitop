#################################################################################
# Copyright (c) 2019, NVIDIA Corporation.  All rights reserved.                 #
#                                                                               #
# Redistribution and use in source and binary forms, with or without            #
# modification, are permitted provided that the following conditions are met:   #
#                                                                               #
#    * Redistributions of source code must retain the above copyright notice,   #
#      this list of conditions and the following disclaimer.                    #
#    * Redistributions in binary form must reproduce the above copyright        #
#      notice, this list of conditions and the following disclaimer in the      #
#      documentation and/or other materials provided with the distribution.     #
#    * Neither the name of the NVIDIA Corporation nor the names of its          #
#      contributors may be used to endorse or promote products derived from     #
#      this software without specific prior written permission.                 #
#                                                                               #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"   #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE     #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE    #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE     #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR           #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF          #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN       #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)       #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF        #
# THE POSSIBILITY OF SUCH DAMAGE.                                               #
#################################################################################

# To Run:
# $ python nvhtop.py

import datetime
import sys
import time
from collections import OrderedDict

import psutil

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
        self.device = device
        self.gpu_memory = gpu_memory
        self.type = type
        self.cpu_percent()


class Device(object):
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
    def display_active(self):
        return {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetDisplayActive, self.handle), 'N/A')

    @property
    def persistence_mode(self):
        return {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetPersistenceMode, self.handle), 'N/A')

    @property
    def ecc_errors(self):
        return nvml_query(nvml.nvmlDeviceGetTotalEccErrors, self.handle,
                          nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                          nvml.NVML_VOLATILE_ECC)

    @property
    def fan_speed(self):
        fan_speed = nvml_query(nvml.nvmlDeviceGetFanSpeed, self.handle)
        if fan_speed != 'N/A':
            fan_speed = str(fan_speed) + '%'
        return fan_speed

    @property
    def utilization(self):
        utilization = nvml_query(nvml.nvmlDeviceGetUtilizationRates, self.handle).gpu
        if utilization != 'N/A':
            utilization = str(utilization) + '%'
        return utilization

    @property
    def compute_mode(self):
        return {
            nvml.NVML_COMPUTEMODE_DEFAULT: 'Default',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: 'E. Thread',
            nvml.NVML_COMPUTEMODE_PROHIBITED: 'Prohibited',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: 'E. Process',
        }.get(nvml_query(nvml.nvmlDeviceGetComputeMode, self.handle), 'N/A')

    @property
    def temperature(self):
        temperature = nvml_query(nvml.nvmlDeviceGetTemperature, self.handle, nvml.NVML_TEMPERATURE_GPU)
        if temperature != 'N/A':
            temperature = str(temperature) + 'C'
        return temperature

    @property
    def performance_state(self):
        performance_state = nvml_query(nvml.nvmlDeviceGetPerformanceState, self.handle)
        if performance_state != 'N/A':
            performance_state = 'P' + str(performance_state)
        return performance_state

    @property
    def memory_used(self):
        return nvml_query(lambda handle: nvml.nvmlDeviceGetMemoryInfo(handle).used, self.handle)

    @property
    def power_usage(self):
        return nvml_query(nvml.nvmlDeviceGetPowerUsage, self.handle)

    @property
    def processes(self):
        process = {}

        for proc in nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
            process[proc.pid] = GProcess(pid=proc.pid, device=self, gpu_memory=proc.usedGpuMemory, type='C')
        for proc in nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle):
            if proc.pid in process:
                process[proc.pid].type = 'C+G'
            else:
                process[proc.pid] = GProcess(pid=proc.pid, device=self, gpu_memory=proc.usedGpuMemory, type='G')
        return OrderedDict(sorted(process.items()))


class Top(object):
    def __init__(self):
        self.driver_version = str(nvml_query(nvml.nvmlSystemGetDriverVersion))
        self.cuda_version = str(nvml_query(nvml.nvmlSystemGetCudaDriverVersion))
        if self.cuda_version != 'N/A':
            self.cuda_version = self.cuda_version[:-3] + '.' + self.cuda_version[-2]

        self.device_count = nvml.nvmlDeviceGetCount()
        self.devices = list(map(Device, range(self.device_count)))

    def redraw(self):
        lines = [
            time.strftime('%a %b %d %H:%M:%S %Y'),
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
            name = device.name
            if len(name) > 18:
                name = name[:15] + '...'

            memory = '{} / {}'.format(bytes2human(device.memory_used), bytes2human(device.memory_total))

            power_usage = device.power_usage
            if power_usage != 'N/A' and device.power_limit != 'N/A':
                power = '{}W / {}W'.format(power_usage // 1000, device.power_limit // 1000)
            else:
                power = 'N/A'

            lines.extend([
                '│ {:>3}  {:>18}  {:<4} │ {:<16} {:>3} │ {:>20} │'.format(device.index, name, device.persistence_mode,
                                                                          device.bus_id, device.display_active,
                                                                          device.ecc_errors),
                '│ {:>3}  {:>4}  {:>4}  {:>12} │ {:>20} │ {:>7}  {:>11} │'.format(device.fan_speed, device.temperature,
                                                                                  device.performance_state,
                                                                                  power, memory,
                                                                                  device.utilization, device.compute_mode),
                '├───────────────────────────────┼──────────────────────┼──────────────────────┤'
            ])

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
            processes = sorted(processes.values(), key=lambda proc: (proc.device.index, proc.username(), proc.pid))
            now_time = datetime.datetime.now()
            prev_device_index = None
            for proc in processes:
                device_index = proc.device.index
                cmdline = proc.cmdline()
                cmdline[0] = proc.name()
                cmdline = ' '.join(cmdline).strip()
                if len(cmdline) > 24:
                    cmdline = cmdline[:21] + '...'
                username = proc.username()
                if len(username) >= 8:
                    username = username[:6] + '+'
                running_time = now_time - datetime.datetime.fromtimestamp(proc.create_time())
                if running_time.days > 1:
                    running_time = '{} days'.format(running_time.days)
                else:
                    hours, seconds = divmod(86400 * running_time.days + running_time.seconds, 3600)
                    running_time = '{:02d}:{:02d}:{:02d}'.format(hours, *divmod(seconds, 60))
                if prev_device_index is not None and prev_device_index != device_index:
                    lines.append('├─────────────────────────────────────────────────────────────────────────────┤')
                prev_device_index = device_index
                lines.append(
                    '│ {:>3} {:>6} {:>7} {:>8} {:>5.1f} {:>5.1f}  {:>8}  {:<24} │'.format(
                        device_index,
                        proc.pid,
                        username,
                        bytes2human(proc.gpu_memory),
                        proc.cpu_percent(),
                        proc.memory_percent(),
                        running_time,
                        cmdline
                    )
                )
        else:
            lines.append('│  No running compute processes found                                         │')

        lines.append('╘═════════════════════════════════════════════════════════════════════════════╛')

        print('\n'.join(lines))


def main():
    try:
        nvml.nvmlInit()
    except nvml.NVMLError as error:
        if error.value == nvml.NVML_ERROR_LIBRARY_NOT_FOUND:
            print(error, file=sys.stderr)
            exit(1)
        raise

    top = Top()

    top.redraw()

    nvml.nvmlShutdown()


if __name__ == '__main__':
    main()
