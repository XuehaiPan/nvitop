# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2024 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An interactive NVIDIA-GPU process viewer and beyond, the one-stop solution for GPU process management."""

import sys

from nvitop import api
from nvitop.api import *  # noqa: F403
from nvitop.api import collector, device, host, libcuda, libcudart, libnvml, process, utils
from nvitop.select import select_devices
from nvitop.version import __version__


__all__ = [*api.__all__, 'select_devices']

# Add submodules to the top-level namespace
for submodule in (collector, device, host, libcuda, libcudart, libnvml, process, utils):
    sys.modules[f'{__name__}.{submodule.__name__.rpartition(".")[-1]}'] = submodule

# Remove the nvitop.select module from sys.modules
# Required for `python -m nvitop.select` to work properly
sys.modules.pop(f'{__name__}.select', None)

del sys
