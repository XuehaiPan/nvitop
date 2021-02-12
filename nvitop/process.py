# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import datetime

import psutil
from cachetools.func import ttl_cache

from .utils import bytes2human, timedelta2human, Snapshot


def add_quotes(s):
    if '$' not in s and '\\' not in s:
        if ' ' not in s:
            return s
        if '"' not in s:
            return '"{}"'.format(s)
    if "'" not in s:
        return "'{}'".format(s)
    return '"{}"'.format(s.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$'))


class GProcess(psutil.Process):
    def __init__(self, pid, device, gpu_memory, type='C'):  # pylint: disable=redefined-builtin
        super(GProcess, self).__init__(pid)
        self._ident = (self.pid, self._create_time, device.index)
        self._hash = None

        super(GProcess, self).cpu_percent()
        self.device = device
        self.gpu_memory = gpu_memory
        self._type = ''
        self.type = type

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = ''
        if 'C' in value:
            self._type += 'C'
        if 'G' in value:
            self._type += 'G'
        if self._type == 'CG':
            self._type = 'C+G'

    @ttl_cache(ttl=1.0)
    def running_time(self):
        return datetime.datetime.now() - datetime.datetime.fromtimestamp(self.create_time())

    cpu_percent = ttl_cache(ttl=1.0)(psutil.Process.cpu_percent)
    memory_percent = ttl_cache(ttl=1.0)(psutil.Process.memory_percent)

    @ttl_cache(ttl=2.0)
    def snapshot(self):
        try:
            with self.oneshot():
                cmdline = self.cmdline()
                if len(cmdline) == 0:
                    raise psutil.Error
                cpu_percent = self.cpu_percent()
                memory_percent = self.memory_percent()
                running_time = self.running_time()
                running_time_human = timedelta2human(running_time)
                command = ' '.join(map(add_quotes, filter(None, map(str.strip, cmdline))))
                host_info = '{:>5.1f} {:>5.1f}  {:>8}  {}'.format(cpu_percent, memory_percent,
                                                                  running_time_human, command)
                snapshot = Snapshot(
                    process=self,
                    identity=self._ident,
                    pid=self.pid,
                    device=self.device,
                    gpu_memory=self.gpu_memory,
                    gpu_memory_human=bytes2human(self.gpu_memory),
                    type=self.type,
                    username=self.username(),
                    name=self.name(),
                    cmdline=cmdline,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    running_time=running_time,
                    running_time_human=running_time_human,
                    host_info=host_info
                )
        except psutil.Error:
            return None
        else:
            return snapshot

    @staticmethod
    @ttl_cache(ttl=30.0)
    def get(pid, device):
        return GProcess(pid, device, gpu_memory=0, type='')
