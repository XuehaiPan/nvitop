# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

import psutil
from cachetools.func import ttl_cache

from .history import BufferedHistoryGraph


cpu_count = psutil.cpu_count

cpu_percent = BufferedHistoryGraph(
    baseline=0.0,
    upperbound=100.0,
    width=77, height=5,
    dynamic_bound=True,
    format='CPU: {:.1f}%'.format
)(ttl_cache(ttl=0.25)(psutil.cpu_percent))

try:
    load_average = ttl_cache(ttl=2.0)(psutil.getloadavg)
except AttributeError:
    def load_average(): return None  # pylint: disable=multiple-statements

virtual_memory = BufferedHistoryGraph(
    baseline=0.0,
    upperbound=100.0,
    width=77, height=4,
    dynamic_bound=False,
    upsidedown=True,
    format='MEM: {:.1f}%'.format
)(
    ttl_cache(ttl=0.25)(psutil.virtual_memory),
    get_value=lambda ret: ret.percent
)

swap_memory = BufferedHistoryGraph(
    baseline=0.0,
    upperbound=100.0,
    width=77, height=1,
    dynamic_bound=False,
    upsidedown=False,
    format='SWP: {:.1f}%'.format
)(
    ttl_cache(ttl=0.25)(psutil.swap_memory),
    get_value=lambda ret: ret.percent
)
