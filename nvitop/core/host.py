# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

import psutil as _psutil
from cachetools.func import ttl_cache as _ttl_cache
from psutil import *  # pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin


__all__ = _psutil.__all__.copy()
__all__.append('load_average')
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
