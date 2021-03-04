# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import datetime
import functools
import sys
import threading

import psutil
from cachetools.func import ttl_cache

from .utils import bytes2human, timedelta2human, Snapshot


if sys.platform != 'windows':
    def add_quotes(s):
        if '$' not in s and '\\' not in s:
            if ' ' not in s:
                return s
            if '"' not in s:
                return '"{}"'.format(s)
        if "'" not in s:
            return "'{}'".format(s)
        return '"{}"'.format(s.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$'))
else:
    def add_quotes(s):
        if '%' not in s and '^' not in s:
            if ' ' not in s:
                return s
            if '"' not in s:
                return '"{}"'.format(s)
        return '"{}"'.format(s.replace('^', '^^').replace('"', '^"').replace('%', '^%'))


def auto_garbage_clean(default=None):
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except psutil.Error:
                try:
                    with GProcess.INSTANCE_LOCK:
                        del GProcess.INSTANCES[(self.pid, self.device)]
                except KeyError:
                    pass
                return default
        return wrapped

    return wrapper


class GProcess(psutil.Process):
    INSTANCE_LOCK = threading.RLock()
    INSTANCES = {}

    def __new__(cls, pid, device, gpu_memory=0, type=''):  # pylint: disable=redefined-builtin
        try:
            return cls.INSTANCES[(pid, device)]
        except KeyError:
            pass

        instance = super(GProcess, cls).__new__(cls)
        instance.__init__(pid, device, gpu_memory=gpu_memory, type=type)
        with cls.INSTANCE_LOCK:
            cls.INSTANCES[(pid, device)] = instance
        return instance

    def __init__(self, pid, device, gpu_memory=0, type=''):  # pylint: disable=redefined-builtin
        super(GProcess, self).__init__(pid)
        self._ident = (self.pid, self._create_time, device.index)
        self._hash = None

        super(GProcess, self).cpu_percent()
        self.device = device
        self.set_gpu_memory(gpu_memory)
        self.type = type

    def gpu_memory(self):
        return self._gpu_memory

    def set_gpu_memory(self, value):
        self._gpu_memory = value

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
    @auto_garbage_clean(default=datetime.timedelta())
    def running_time(self):
        return datetime.datetime.now() - datetime.datetime.fromtimestamp(self.create_time())

    username = auto_garbage_clean(default='')(psutil.Process.username)
    name = auto_garbage_clean(default='')(psutil.Process.name)
    cpu_percent = ttl_cache(ttl=1.0)(auto_garbage_clean(default=0.0)(psutil.Process.cpu_percent))
    memory_percent = ttl_cache(ttl=1.0)(auto_garbage_clean(default=0.0)(psutil.Process.memory_percent))

    @auto_garbage_clean(default=[])
    def cmdline(self):
        cmdline = super(GProcess, self).cmdline()
        if len(cmdline) == 0:
            raise psutil.NoSuchProcess(pid=self.pid)
        return cmdline

    @ttl_cache(ttl=2.0)
    @auto_garbage_clean(default=None)
    def snapshot(self):
        with self.oneshot():
            username = self.username()
            name = self.name()
            cmdline = self.cmdline()
            cpu_percent = self.cpu_percent()
            memory_percent = self.memory_percent()
            gpu_memory = self.gpu_memory()
            running_time = self.running_time()
            running_time_human = timedelta2human(running_time)
            command = ' '.join(map(add_quotes, filter(None, map(str.strip, cmdline))))
            host_info = '{:>5.1f} {:>5.1f}  {:>8}  {}'.format(cpu_percent, memory_percent,
                                                              running_time_human, command)

        if (self.pid, self.device) not in self.INSTANCES:
            raise psutil.NoSuchProcess(pid=self.pid, name=name)

        return Snapshot(
            real=self,
            identity=self._ident,
            pid=self.pid,
            device=self.device,
            gpu_memory=gpu_memory,
            gpu_memory_human=bytes2human(gpu_memory),
            type=self.type,
            username=username,
            name=name,
            cmdline=cmdline,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            running_time=running_time,
            running_time_human=running_time_human,
            host_info=host_info
        )
