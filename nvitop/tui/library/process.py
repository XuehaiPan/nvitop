# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nvitop.api import NA, HostProcess, Snapshot, utilization2string
from nvitop.api import GpuProcess as GpuProcessBase
from nvitop.tui.library.utils import IS_WINDOWS, IS_WSL


if TYPE_CHECKING:
    from typing_extensions import Self  # Python 3.11+

    from nvitop.tui.library.device import Device


__all__ = ['GpuProcess', 'HostProcess']


class GpuProcess(GpuProcessBase):
    _snapshot: Snapshot | None
    device: Device

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        instance = super().__new__(cls, *args, **kwargs)
        instance._snapshot = None
        return instance

    @property
    def snapshot(self) -> Snapshot:
        if self._snapshot is None:
            self.as_snapshot()
        return self._snapshot  # type: ignore[return-value]

    def host_snapshot(self) -> Snapshot:
        host_snapshot = super().host_snapshot()

        if host_snapshot.cpu_percent is NA:
            host_snapshot.cpu_percent_string = NA
        elif host_snapshot.cpu_percent < 1000.0:
            host_snapshot.cpu_percent_string = f'{host_snapshot.cpu_percent:.1f}%'
        elif host_snapshot.cpu_percent < 10000:
            host_snapshot.cpu_percent_string = f'{int(host_snapshot.cpu_percent)}%'
        else:
            host_snapshot.cpu_percent_string = '9999+%'

        if host_snapshot.memory_percent is NA:
            host_snapshot.memory_percent_string = NA
        else:
            host_snapshot.memory_percent_string = f'{host_snapshot.memory_percent:.1f}%'

        return host_snapshot

    def as_snapshot(
        self,
        *,
        host_process_snapshot_cache: dict[int, Snapshot] | None = None,
    ) -> Snapshot:
        snapshot = super().as_snapshot(host_process_snapshot_cache=host_process_snapshot_cache)

        snapshot.type = snapshot.type.replace('C+G', 'X')
        if snapshot.gpu_memory_human is NA and (IS_WINDOWS or IS_WSL):
            snapshot.gpu_memory_human = 'WDDM:N/A'

        snapshot.cpu_percent_string = snapshot.host.cpu_percent_string
        snapshot.memory_percent_string = snapshot.host.memory_percent_string

        if snapshot.is_running:
            snapshot.is_zombie = snapshot.cmdline == ['Zombie Process']
            snapshot.no_permissions = snapshot.cmdline == ['No Permissions']
            snapshot.is_gone = False
        else:
            snapshot.is_zombie = False
            snapshot.no_permissions = False
            snapshot.is_gone = snapshot.cmdline == ['No Such Process']

        snapshot.gpu_memory_percent_string = self.gpu_memory_percent_string()
        snapshot.gpu_sm_utilization_string = self.gpu_sm_utilization_string()
        snapshot.gpu_memory_utilization_string = self.gpu_memory_utilization_string()
        snapshot.gpu_encoder_utilization_string = self.gpu_encoder_utilization_string()
        snapshot.gpu_decoder_utilization_string = self.gpu_decoder_utilization_string()

        self._snapshot = snapshot  # pylint: disable=attribute-defined-outside-init
        return snapshot

    def gpu_memory_percent_string(self) -> str:  # in percentage
        return utilization2string(self.gpu_memory_percent())

    def gpu_sm_utilization_string(self) -> str:  # in percentage
        return utilization2string(self.gpu_sm_utilization())

    def gpu_memory_utilization_string(self) -> str:  # in percentage
        return utilization2string(self.gpu_memory_utilization())

    def gpu_encoder_utilization_string(self) -> str:  # in percentage
        return utilization2string(self.gpu_encoder_utilization())

    def gpu_decoder_utilization_string(self) -> str:  # in percentage
        return utilization2string(self.gpu_decoder_utilization())
