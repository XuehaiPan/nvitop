# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

import psutil
from cachetools.func import ttl_cache
from psutil import *  # pylint: disable=wildcard-import


__all__ = psutil.__all__.copy()
__all__.append('load_average')
__all__[__all__.index('Error')] = 'PsutilError'

del Error  # pylint: disable=undefined-variable
PsutilError = psutil.Error

cpu_count = psutil.cpu_count

cpu_percent = ttl_cache(ttl=0.25)(psutil.cpu_percent)

try:
    load_average = ttl_cache(ttl=2.0)(psutil.getloadavg)
except AttributeError:
    def load_average(): return None  # pylint: disable=multiple-statements

virtual_memory = ttl_cache(ttl=0.25)(psutil.virtual_memory)

swap_memory = ttl_cache(ttl=0.25)(psutil.swap_memory)
