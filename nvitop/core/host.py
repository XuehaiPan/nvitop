# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

from collections import defaultdict as _defaultdict

import psutil as _psutil
from cachetools.func import ttl_cache as _ttl_cache
from psutil import *  # pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin


__all__ = _psutil.__all__.copy()
__all__.extend(['load_average', 'ppid_map', 'reverse_ppid_map'])
__all__[__all__.index('Error')] = 'PsutilError'


PsutilError = Error
del Error  # pylint: disable=undefined-variable


cpu_percent = _ttl_cache(ttl=0.25)(_psutil.cpu_percent)


try:
    load_average = _ttl_cache(ttl=2.0)(_psutil.getloadavg)
except AttributeError:
    def load_average(): return None  # pylint: disable=multiple-statements


virtual_memory = _ttl_cache(ttl=0.25)(_psutil.virtual_memory)


swap_memory = _ttl_cache(ttl=0.25)(_psutil.swap_memory)


ppid_map = _psutil._ppid_map  # pylint: disable=protected-access


def reverse_ppid_map():
    tree = _defaultdict(list)
    for pid, ppid in ppid_map().items():
        tree[ppid].append(pid)

    return tree
