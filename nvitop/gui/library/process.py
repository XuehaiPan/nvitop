# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import functools
import threading

from nvitop.core import (host, HostProcess, GpuProcess as GpuProcessBase,
                         NA, Snapshot, command_join, timedelta2human)


__all__ = ['HostProcess', 'GpuProcess']


def auto_garbage_clean(default=None):
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except host.PsutilError as e:
                try:
                    with GpuProcess.INSTANCE_LOCK:
                        del GpuProcess.INSTANCES[(self.pid, self.device)]
                except KeyError:
                    pass
                try:
                    with GpuProcess.SNAPSHOT_LOCK:
                        del GpuProcess.HOST_SNAPSHOTS[self.pid]
                except KeyError:
                    pass
                try:
                    with HostProcess.INSTANCE_LOCK:
                        del HostProcess.INSTANCES[self.pid]
                except KeyError:
                    pass
                if isinstance(default, tuple):
                    if isinstance(e, host.AccessDenied) and default == ('No Such Process',):
                        return ['No Permissions']
                    return list(default)
                return default

        return wrapped

    return wrapper


class GpuProcess(GpuProcessBase):
    SNAPSHOT_LOCK = threading.RLock()
    HOST_SNAPSHOTS = {}

    running_time = auto_garbage_clean(default=NA)(GpuProcessBase.running_time)

    create_time = auto_garbage_clean(default=NA)(GpuProcessBase.create_time)

    username = auto_garbage_clean(default=NA)(GpuProcessBase.username)

    name = auto_garbage_clean(default=NA)(GpuProcessBase.name)

    cpu_percent = auto_garbage_clean(default=NA)(GpuProcessBase.cpu_percent)

    memory_percent = auto_garbage_clean(default=NA)(GpuProcessBase.memory_percent)

    @auto_garbage_clean(default=('No Such Process',))
    def cmdline(self):
        cmdline = self.host.cmdline()
        if len(cmdline) == 0 and not self._gone:
            cmdline = ['Zombie Process']
        return cmdline

    @classmethod
    def clear_host_snapshots(cls) -> None:
        with cls.SNAPSHOT_LOCK:
            cls.HOST_SNAPSHOTS.clear()

    @auto_garbage_clean(default=None)
    def as_snapshot(self) -> Snapshot:
        with self.SNAPSHOT_LOCK:
            try:
                host_snapshot = self.HOST_SNAPSHOTS[self.pid]
            except KeyError:
                with self.host.oneshot():
                    host_snapshot = Snapshot(
                        real=self.host,
                        is_running=self.host.is_running(),
                        status=self.host.status(),
                        username=self.username(),
                        name=self.name(),
                        cmdline=self.cmdline(),
                        cpu_percent=self.cpu_percent(),
                        memory_percent=self.memory_percent(),
                        running_time=self.running_time()
                    )

                host_snapshot.command = command_join(host_snapshot.cmdline)
                if host_snapshot.cpu_percent < 1000.0:
                    host_snapshot.cpu_percent_string = '{:.1f}%'.format(host_snapshot.cpu_percent)
                elif host_snapshot.cpu_percent < 10000:
                    host_snapshot.cpu_percent_string = '{}%'.format(int(host_snapshot.cpu_percent))
                else:
                    host_snapshot.cpu_percent_string = '9999+%'
                host_snapshot.memory_percent_string = '{:.1f}%'.format(host_snapshot.memory_percent)
                host_snapshot.running_time_human = timedelta2human(host_snapshot.running_time)

                self.HOST_SNAPSHOTS[self.pid] = host_snapshot

        return Snapshot(
            real=self,
            pid=self.pid,
            device=self.device,
            gpu_memory=self.gpu_memory(),
            gpu_memory_human=self.gpu_memory_human(),
            gpu_memory_utilization=self.gpu_memory_utilization(),
            gpu_memory_utilization_string=self.gpu_memory_utilization_string(),
            gpu_sm_utilization=self.gpu_sm_utilization(),
            gpu_sm_utilization_string=self.gpu_sm_utilization_string(),
            gpu_encoder_utilization=self.gpu_encoder_utilization(),
            gpu_encoder_utilization_string=self.gpu_encoder_utilization_string(),
            gpu_decoder_utilization=self.gpu_decoder_utilization(),
            gpu_decoder_utilization_string=self.gpu_decoder_utilization_string(),
            type=self.type,
            username=host_snapshot.username,
            name=host_snapshot.name,
            cmdline=host_snapshot.cmdline,
            command=host_snapshot.command,
            cpu_percent=host_snapshot.cpu_percent,
            cpu_percent_string=host_snapshot.cpu_percent_string,
            memory_percent=host_snapshot.memory_percent,
            memory_percent_string=host_snapshot.memory_percent_string,
            is_running=host_snapshot.is_running,
            running_time=host_snapshot.running_time,
            running_time_human=host_snapshot.running_time_human
        )
