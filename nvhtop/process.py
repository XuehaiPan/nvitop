# This file is part of nvhtop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

import datetime

import psutil
from cachetools.func import ttl_cache

from .utils import Snapshot


class GProcess(psutil.Process):
    def __init__(self, pid, device, gpu_memory, proc_type='C'):
        super(GProcess, self).__init__(pid)
        super(GProcess, self).cpu_percent()
        self.device = device
        self.gpu_memory = gpu_memory
        self.proc_type = proc_type

    @ttl_cache(ttl=2.0)
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

    @staticmethod
    @ttl_cache(ttl=30.0)
    def get(pid, device):
        return GProcess(pid, device, gpu_memory=0, proc_type='')
