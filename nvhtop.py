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

import time
from collections import OrderedDict

import pynvml as nvml


def to_string(s):
    if isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return str(s)


def to_human_readable(x):
    if x < (1 << 10):
        return '{}B'.format(x)
    if x < (1 << 20):
        return '{}KiB'.format(x >> 10)
    else:
        return '{}MiB'.format(x >> 20)


def nvml_query(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except nvml.NVMLError as error:
        if error.value == nvml.NVML_ERROR_NOT_SUPPORTED:
            return 'N/A'
        else:
            return str(error)


def device_query():
    nvml.nvmlInit()

    driver_version = to_string(nvml.nvmlSystemGetDriverVersion())
    cuda_version = to_string(nvml.nvmlSystemGetCudaDriverVersion())
    if cuda_version != 'N/A':
        cuda_version = cuda_version[:-3] + '.' + cuda_version[-2]

    lines = [
        time.strftime('%a %b %d %H:%M:%S %Y'),
        '+-----------------------------------------------------------------------------+',
        '| NVIDIA-SMI {0:<6}       Driver Version: {0:<6}       CUDA Version: {1:<5}    |'.format(driver_version, cuda_version),
        '|-------------------------------+----------------------+----------------------+',
        '| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |',
        '| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |',
        '|===============================+======================+======================|'
    ]

    processes = OrderedDict()

    count = nvml.nvmlDeviceGetCount()

    for i in range(0, count):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)

        name = to_string(nvml_query(nvml.nvmlDeviceGetName, handle))
        if len(name) > 18:
            name = name[:15] + '...'
        persistence_mode = {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetPersistenceMode, handle), 'N/A')
        bus_id = to_string(nvml_query(nvml.nvmlDeviceGetPciInfo, handle).busId)
        fan = to_string(nvml_query(nvml.nvmlDeviceGetFanSpeed, handle))
        if fan != 'N/A':
            fan += '%'
        display_active = {0: 'Off', 1: 'On'}.get(nvml_query(nvml.nvmlDeviceGetDisplayActive, handle), 'N/A')
        ecc_errors = to_string(nvml_query(nvml.nvmlDeviceGetTotalEccErrors, handle, nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nvml.NVML_VOLATILE_ECC))
        utilization = to_string(nvml_query(nvml.nvmlDeviceGetUtilizationRates, handle).gpu) + '%'
        compute_mode = {
            nvml.NVML_COMPUTEMODE_DEFAULT: 'Default',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: 'E. Thread',
            nvml.NVML_COMPUTEMODE_PROHIBITED: 'Prohibited',
            nvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: 'E. Process',
        }.get(nvml_query(nvml.nvmlDeviceGetComputeMode, handle), 'N/A')
        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        memory = '{} / {}'.format(to_human_readable(memory_info.used), to_human_readable(memory_info.total))
        temperature = to_string(nvml_query(nvml.nvmlDeviceGetTemperature, handle, nvml.NVML_TEMPERATURE_GPU))
        if temperature != 'N/A':
            temperature += 'C'
        performance_state = to_string(nvml_query(nvml.nvmlDeviceGetPerformanceState, handle))
        if performance_state != 'N/A':
            performance_state = 'P' + performance_state
        power_usage = nvml_query(nvml.nvmlDeviceGetPowerUsage, handle)
        power_limit = nvml_query(nvml.nvmlDeviceGetPowerManagementLimit, handle)
        if power_usage != 'N/A' and power_limit != 'N/A':
            power = '{}W / {}W'.format(power_usage // 1000, power_limit // 1000)
        else:
            power = 'N/A'

        lines.extend([
            '| {:>3}  {:>18}  {:<4} | {:<16} {:>3} | {:>20} |'.format(i, name, persistence_mode,
                                                                      bus_id, display_active,
                                                                      ecc_errors),
            '| {:>3}  {:>4}  {:>4}  {:>12} | {:>20} | {:>7}  {:>11} |'.format(fan, temperature, performance_state, power,
                                                                              memory,
                                                                              utilization, compute_mode),
            '+-------------------------------+----------------------+----------------------+',
        ])

        device_process = {}

        for proc in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
            device_process[proc.pid] = {
                'gpu': i, 'pid': proc.pid, 'type': 'C',
                'name': to_string(nvml.nvmlSystemGetProcessName(proc.pid)),
                'memory': to_human_readable(proc.usedGpuMemory)
            }
        for proc in nvml.nvmlDeviceGetGraphicsRunningProcesses(handle):
            if proc.pid in device_process:
                device_process[proc.pid]['type'] = 'C+G'
            else:
                device_process[proc.pid] = {
                    'gpu': i, 'pid': proc.pid, 'type': 'G',
                    'name': to_string(nvml.nvmlSystemGetProcessName(proc.pid)),
                    'memory': to_human_readable(proc.usedGpuMemory)
                }
        for pid in sorted(device_process):
            processes[pid] = device_process[pid]

    lines.extend([
        '                                                                               ',
        '+-----------------------------------------------------------------------------+',
        '| Processes:                                                       GPU Memory |',
        '|  GPU       PID   Type   Process name                             Usage      |',
        '|=============================================================================|'
    ])

    if len(processes) > 0:
        for proc in processes.values():
            if len(proc['name']) > 42:
                proc['name'] = '...' + proc['name'][-39:]
            lines.append('|  {gpu:>3}  {pid:>8}    {type:>3}   {name:<42} {memory:>8} |'.format(**proc))
    else:
        lines.append('|  No running compute processes found                                         |')

    lines.append('+-----------------------------------------------------------------------------+')

    nvml.nvmlShutdown()

    return '\n'.join(lines)


if __name__ == '__main__':
    print(device_query())
