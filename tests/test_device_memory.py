# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2026 Xuehai Pan. All Rights Reserved.
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
"""Hardware-independent regression tests for device memory accounting."""

from __future__ import annotations

import threading
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from nvitop.api import host, libnvml
from nvitop.api.device import Device, MemoryInfo
from nvitop.api.utils import NA


# pylint: disable=protected-access
class DeviceMemoryTest(unittest.TestCase):
    """Exercise device memory accounting without initializing NVML."""

    def setUp(self) -> None:
        """Prepare a device and deterministic memory responses without a driver."""
        self.device = object.__new__(Device)
        self.device._handle = libnvml.c_nvmlDevice_t()
        self.device._memory_total = NA
        self.device._memory_total_human = NA
        self.device._lock = threading.RLock()
        self.vm = SimpleNamespace(total=1024, available=640, free=128, used=320)
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.query = stack.enter_context(
            patch.object(libnvml, 'nvmlQuery', side_effect=libnvml.NVMLError_NotSupported),
        )
        self.virtual_memory = stack.enter_context(
            patch.object(host, 'virtual_memory', return_value=self.vm),
        )

    def test_unified_memory_uses_available_capacity(self) -> None:
        """Reclaimable cache is available to both CPU and GPU allocations."""
        memory = self.device.memory_info()
        assert memory == MemoryInfo(total=1024, free=640, used=384, reserved=NA)
        assert memory.used + memory.free == memory.total
        self.query.assert_called_once_with(
            'nvmlDeviceGetMemoryInfo',
            self.device._handle,
            ignore_errors=False,
        )
        self.virtual_memory.assert_called_once_with()

    def test_zero_total_uses_shared_memory(self) -> None:
        """A successful zero-total NVML response also uses available capacity."""
        self.query.side_effect = None
        self.query.return_value = SimpleNamespace(total=0, free=0, used=0)
        assert self.device.memory_info() == MemoryInfo(1024, 640, 384, NA)

    def test_discrete_memory_preserves_nvml_values(self) -> None:
        """Successful NVML data is used even when VRAM exceeds host RAM."""
        self.query.side_effect = None
        self.query.return_value = SimpleNamespace(total=2048, free=1024, used=960, reserved=64)
        assert self.device.memory_info() == MemoryInfo(2048, 1024, 960, 64)
        self.virtual_memory.assert_not_called()

    def test_discrete_memory_without_reserved_field(self) -> None:
        """Older NVML memory responses retain N/A for reserved memory."""
        self.query.side_effect = None
        self.query.return_value = SimpleNamespace(total=1024, free=768, used=256)
        assert self.device.memory_info() == MemoryInfo(1024, 768, 256, NA)
        self.virtual_memory.assert_not_called()

    def test_other_nvml_errors_do_not_use_host_memory(self) -> None:
        """Permission errors and lost GPUs must not masquerade as shared memory."""
        for error in (
            libnvml.NVMLError_NoPermission,
            libnvml.NVMLError_GpuIsLost,
            libnvml.NVMLError_Unknown,
        ):
            with self.subTest(error=error):
                self.query.side_effect = error
                assert self.device.memory_info() == MemoryInfo(NA, NA, NA, NA)
        self.virtual_memory.assert_not_called()

    def test_unavailable_nvml_result_does_not_use_host_memory(self) -> None:
        """An unavailable response is not evidence of unified memory."""
        self.query.side_effect = None
        self.query.return_value = NA
        assert self.device.memory_info() == MemoryInfo(NA, NA, NA, NA)
        self.virtual_memory.assert_not_called()

    def test_missing_handle_does_not_query_memory(self) -> None:
        """A device without an NVML handle retains unavailable memory values."""
        self.device._handle = None
        assert self.device.memory_info() == MemoryInfo(NA, NA, NA, NA)
        self.query.assert_not_called()
        self.virtual_memory.assert_not_called()

    def test_shared_memory_refreshes_available_capacity(self) -> None:
        """Successive samples reflect changes in available system memory."""
        assert self.device.memory_info() == MemoryInfo(1024, 640, 384, NA)
        self.vm.available = 256
        assert self.device.memory_info() == MemoryInfo(1024, 256, 768, NA)

    def test_shared_memory_at_capacity_boundaries(self) -> None:
        """Both completely available and fully used memory remain consistent."""
        for available in (0, self.vm.total):
            with self.subTest(available=available):
                self.vm.available = available
                expected = MemoryInfo(1024, available, 1024 - available, NA)
                assert self.device.memory_info() == expected

    def test_derived_memory_values_use_available_capacity(self) -> None:
        """Memory helpers and percentages use the same shared-memory accounting."""
        assert self.device.memory_total() == 1024
        assert self.device.memory_free() == 640
        assert self.device.memory_used() == 384
        assert self.device.memory_percent() == 37.5

    def test_oneshot_keeps_memory_sample_consistent(self) -> None:
        """A oneshot context reuses one sample until the context ends."""
        with self.device.oneshot():
            assert self.device.memory_free() == 640
            self.vm.available = 256
            assert self.device.memory_used() == 384
            assert self.device.memory_percent() == 37.5
        self.virtual_memory.assert_called_once_with()
        assert self.device.memory_info() == MemoryInfo(1024, 256, 768, NA)
