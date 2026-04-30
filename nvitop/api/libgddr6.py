# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
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
# Ported from olealgoritme/gddr6 (https://github.com/olealgoritme/gddr6)
"""Read GDDR6/GDDR6X VRAM temperatures from NVIDIA GPU BAR0 registers.

This module reads memory temperatures directly from GPU BAR0 memory-mapped registers
via ``/dev/mem``. It requires Linux, root access, and the ``iomem=relaxed`` kernel
parameter. Only specific NVIDIA GPUs with GDDR6/GDDR6X memory are supported.

When requirements are not met, :func:`read_vram_temperature` returns ``None`` gracefully.
"""

from __future__ import annotations

import atexit
import logging
import mmap
import os
import struct
import threading


__all__ = ['read_vram_temperature']

logger = logging.getLogger(__name__)

# PCI device ID -> (register_offset, vram_type, arch, name)
GDDR6_DEVICE_TABLE: dict[int, tuple[int, str, str, str]] = {
    0x2684: (0x0000E2A8, 'GDDR6X', 'AD102', 'RTX 4090'),
    0x2685: (0x0000E2A8, 'GDDR6X', 'AD102', 'RTX 4090 D'),
    0x2702: (0x0000E2A8, 'GDDR6X', 'AD103', 'RTX 4080 Super'),
    0x2704: (0x0000E2A8, 'GDDR6X', 'AD103', 'RTX 4080'),
    0x2705: (0x0000E2A8, 'GDDR6X', 'AD103', 'RTX 4070 Ti Super'),
    0x2782: (0x0000E2A8, 'GDDR6X', 'AD104', 'RTX 4070 Ti'),
    0x2783: (0x0000E2A8, 'GDDR6X', 'AD104', 'RTX 4070 Super'),
    0x2786: (0x0000E2A8, 'GDDR6X', 'AD104', 'RTX 4070'),
    0x2860: (0x0000E2A8, 'GDDR6', 'AD106', 'RTX 4070 Max-Q / Mobile'),
    0x2203: (0x0000E2A8, 'GDDR6X', 'GA102', 'RTX 3090 Ti'),
    0x2204: (0x0000E2A8, 'GDDR6X', 'GA102', 'RTX 3090'),
    0x2208: (0x0000E2A8, 'GDDR6X', 'GA102', 'RTX 3080 Ti'),
    0x2206: (0x0000E2A8, 'GDDR6X', 'GA102', 'RTX 3080'),
    0x2216: (0x0000E2A8, 'GDDR6X', 'GA102', 'RTX 3080 LHR'),
    0x2484: (0x0000EE50, 'GDDR6', 'GA104', 'RTX 3070'),
    0x2488: (0x0000EE50, 'GDDR6', 'GA104', 'RTX 3070 LHR'),
    0x2531: (0x0000E2A8, 'GDDR6', 'GA106', 'RTX A2000'),
    0x2571: (0x0000E2A8, 'GDDR6', 'GA106', 'RTX A2000'),
    0x2232: (0x0000E2A8, 'GDDR6', 'GA102', 'RTX A4500'),
    0x2231: (0x0000E2A8, 'GDDR6', 'GA102', 'RTX A5000'),
    0x26B1: (0x0000E2A8, 'GDDR6', 'AD102', 'RTX A6000'),
    0x27B8: (0x0000E2A8, 'GDDR6', 'AD104', 'L4'),
    0x26B9: (0x0000E2A8, 'GDDR6', 'AD102', 'L40S'),
    0x2236: (0x0000E2A8, 'GDDR6', 'GA102', 'A10'),
}


def _nvml_bus_id_to_sysfs_path(bus_id: str) -> str:
    """Convert NVML bus_id format to sysfs device path.

    NVML returns e.g. ``'00000000:0A:00.0'``, sysfs uses ``'0000:0a:00.0'``.
    """
    parts = bus_id.split(':')
    if len(parts) == 3:
        domain = f'{int(parts[0], 16):04x}'
        sysfs_id = f'{domain}:{parts[1].lower()}:{parts[2].lower()}'
    else:
        sysfs_id = bus_id.lower()
    return f'/sys/bus/pci/devices/{sysfs_id}'


class GDDR6Context:
    """Singleton context for reading GDDR6/GDDR6X VRAM temperatures."""

    def __init__(self) -> None:
        """Initialize the GDDR6 context."""
        self._fd: int | None = None
        self._available: bool | None = None
        self._mmap_cache: dict[int, mmap.mmap] = {}
        self._device_info_cache: dict[str, tuple[int, int] | None] = {}
        self._lock = threading.Lock()
        self._warned: set[str] = set()

    def _ensure_init(self) -> bool:
        """Open ``/dev/mem`` for reading, returning whether it succeeded."""
        if self._available is not None:
            return self._available

        with self._lock:
            if self._available is not None:
                return self._available

            try:
                self._fd = os.open('/dev/mem', os.O_RDONLY)
                self._available = True
                atexit.register(self.cleanup)
            except PermissionError:
                logger.debug('Cannot open /dev/mem: permission denied (need root)')
                self._available = False
            except FileNotFoundError:
                logger.debug('/dev/mem not found')
                self._available = False
            except OSError as e:
                logger.debug('Cannot open /dev/mem: %s', e)
                self._available = False

        return self._available

    def _get_device_info(self, bus_id: str) -> tuple[int, int] | None:
        """Get (register_offset, bar0_address) for a device, or ``None`` if unsupported."""
        if bus_id in self._device_info_cache:
            return self._device_info_cache[bus_id]

        sysfs_path = _nvml_bus_id_to_sysfs_path(bus_id)
        result = None

        try:
            with open(f'{sysfs_path}/device', encoding='ascii') as f:
                dev_id = int(f.read().strip(), 16)

            entry = GDDR6_DEVICE_TABLE.get(dev_id)
            if entry is not None:
                offset = entry[0]
                with open(f'{sysfs_path}/resource', encoding='ascii') as f:
                    first_line = f.readline().strip()
                bar0 = int(first_line.split()[0], 16)
                if bar0 != 0:
                    result = (offset, bar0)
        except (OSError, ValueError, IndexError):
            pass

        self._device_info_cache[bus_id] = result
        return result

    def read_vram_temperature(self, bus_id: str) -> int | None:
        """Read VRAM temperature for a specific GPU.

        Args:
            bus_id: PCI bus ID as returned by NVML (e.g. ``'00000000:0A:00.0'``).

        Returns:
            Temperature in Celsius, or ``None`` if unavailable.
        """
        if not self._ensure_init():
            return None

        device_info = self._get_device_info(bus_id)
        if device_info is None:
            return None

        offset, bar0 = device_info
        phys_addr = bar0 + offset
        page_size = os.sysconf('SC_PAGE_SIZE')
        base_addr = phys_addr & ~(page_size - 1)
        page_offset = phys_addr - base_addr

        try:
            mapped = self._get_mmap(base_addr, page_size)
            if mapped is None:
                return None

            raw_value = struct.unpack_from('<I', mapped, page_offset)[0]
        except Exception:  # noqa: BLE001
            if bus_id not in self._warned:
                self._warned.add(bus_id)
                logger.debug('Failed to read VRAM temperature for %s', bus_id)
            return None
        else:
            return (raw_value & 0x00000FFF) // 0x20

    def _get_mmap(self, base_addr: int, length: int) -> mmap.mmap | None:
        """Get or create a memory mapping for a physical address region."""
        if base_addr in self._mmap_cache:
            return self._mmap_cache[base_addr]

        with self._lock:
            if base_addr in self._mmap_cache:
                return self._mmap_cache[base_addr]

            try:
                mapped = mmap.mmap(
                    self._fd,  # type: ignore[arg-type]
                    length,
                    mmap.MAP_SHARED,
                    mmap.PROT_READ,
                    offset=base_addr,
                )
            except OSError:
                key = str(base_addr)
                if key not in self._warned:
                    self._warned.add(key)
                    logger.debug(
                        'mmap failed for address 0x%x (iomem=relaxed needed?)',
                        base_addr,
                    )
                return None
            else:
                self._mmap_cache[base_addr] = mapped
                return mapped

    def cleanup(self) -> None:
        """Close all memory mappings and the ``/dev/mem`` file descriptor."""
        for mapped in self._mmap_cache.values():
            try:
                mapped.close()
            except Exception:  # noqa: BLE001,S110,PERF203
                pass  # Best-effort cleanup; nothing to do on failure
        self._mmap_cache.clear()

        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

        self._available = None


_context = GDDR6Context()


def read_vram_temperature(bus_id: str) -> int | None:
    """Read VRAM temperature for the GPU at the given PCI bus ID.

    Args:
        bus_id: PCI bus ID as returned by NVML (e.g. ``'00000000:0A:00.0'``).

    Returns:
        Temperature in Celsius, or ``None`` if unavailable.
    """
    return _context.read_vram_temperature(bus_id)
