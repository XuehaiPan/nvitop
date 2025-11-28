# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import enum
from typing import Any, ClassVar, Literal

from nvitop.api import NA, NaType, Snapshot, libnvml, ttl_cache, utilization2string
from nvitop.api import MigDevice as MigDeviceBase
from nvitop.api import PhysicalDevice as DeviceBase
from nvitop.tui.library.process import GpuProcess, GpuProcessBase


__all__ = ['Device', 'MigDevice']


class LoadingIntensity(enum.IntEnum):
    LIGHT = 0
    MODERATE = 1
    HEAVY = 2

    def color(self) -> str:
        if self == LoadingIntensity.LIGHT:
            return 'green'
        if self == LoadingIntensity.MODERATE:
            return 'yellow'
        return 'red'


class Device(DeviceBase):  # pylint: disable=too-many-public-methods
    GPU_PROCESS_CLASS: ClassVar[type[GpuProcessBase]] = GpuProcess

    MEMORY_UTILIZATION_THRESHOLDS: ClassVar[tuple[int, int]] = (10, 80)
    GPU_UTILIZATION_THRESHOLDS: ClassVar[tuple[int, int]] = (10, 75)

    SNAPSHOT_KEYS: ClassVar[list[str]] = [
        'name',
        'bus_id',
        'memory_used',
        'memory_free',
        'memory_total',
        'memory_used_human',
        'memory_free_human',
        'memory_total_human',
        'memory_percent',
        'memory_usage',
        'gpu_utilization',
        'memory_utilization',
        'fan_speed',
        'temperature',
        'power_usage',
        'power_limit',
        'power_status',
        'display_active',
        'current_driver_model',
        'persistence_mode',
        'performance_state',
        'total_volatile_uncorrected_ecc_errors',
        'compute_mode',
        'mig_mode',
        'is_mig_device',
        'power_utilization',
        'memory_percent_string',
        'memory_utilization_string',
        'gpu_utilization_string',
        'fan_speed_string',
        'temperature_string',
        'memory_loading_intensity',
        'memory_display_color',
        'bandwidth_loading_intensity',
        'bandwidth_display_color',
        'gpu_loading_intensity',
        'gpu_display_color',
        'power_loading_intensity',
        'power_display_color',
        'loading_intensity',
        'display_color',
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._snapshot: Snapshot | None = None
        self.tuple_index: tuple[int] | tuple[int, int] = (
            (self.index,) if isinstance(self.index, int) else self.index
        )
        self.display_index: str = ':'.join(map(str, self.tuple_index))

    def as_snapshot(self) -> Snapshot:
        self._snapshot = super().as_snapshot()
        self._snapshot.tuple_index = self.tuple_index
        self._snapshot.display_index = self.display_index
        return self._snapshot

    @property
    def snapshot(self) -> Snapshot:
        if self._snapshot is None:
            self._snapshot = self.as_snapshot()
        return self._snapshot

    def mig_devices(self) -> list[MigDevice]:  # type: ignore[override]
        mig_devices = []

        if self.is_mig_mode_enabled():
            for mig_index in range(self.max_mig_device_count()):
                try:
                    mig_device = MigDevice(index=(self.index, mig_index))
                except libnvml.NVMLError:  # noqa: PERF203
                    break
                else:
                    mig_devices.append(mig_device)

        return mig_devices

    fan_speed = ttl_cache(ttl=5.0)(DeviceBase.fan_speed)
    temperature = ttl_cache(ttl=5.0)(DeviceBase.temperature)
    display_active = ttl_cache(ttl=5.0)(DeviceBase.display_active)
    display_mode = ttl_cache(ttl=5.0)(DeviceBase.display_mode)
    current_driver_model = ttl_cache(ttl=5.0)(DeviceBase.current_driver_model)
    persistence_mode = ttl_cache(ttl=5.0)(DeviceBase.persistence_mode)
    performance_state = ttl_cache(ttl=5.0)(DeviceBase.performance_state)
    total_volatile_uncorrected_ecc_errors = ttl_cache(ttl=5.0)(
        DeviceBase.total_volatile_uncorrected_ecc_errors,
    )
    compute_mode = ttl_cache(ttl=5.0)(DeviceBase.compute_mode)
    mig_mode = ttl_cache(ttl=5.0)(DeviceBase.mig_mode)

    def power_utilization(self) -> float | NaType:  # in percentage
        power_limit = self.power_limit()
        if not libnvml.nvmlCheckReturn(power_limit, int) or power_limit == 0:
            return NA
        power_usage = self.power_usage()
        if not libnvml.nvmlCheckReturn(power_usage, int):
            return NA
        return round(100.0 * power_usage / power_limit, 1)

    def memory_percent_string(self) -> str:  # in percentage
        return utilization2string(self.memory_percent())

    def memory_utilization_string(self) -> str:  # in percentage
        return utilization2string(self.memory_utilization())

    def gpu_utilization_string(self) -> str:  # in percentage
        return utilization2string(self.gpu_utilization())

    def fan_speed_string(self) -> str:  # in percentage
        return utilization2string(self.fan_speed())

    def temperature_string(self) -> str:  # in Celsius
        temperature = self.temperature()
        return f'{temperature}C' if libnvml.nvmlCheckReturn(temperature, int) else NA

    def memory_loading_intensity(self) -> LoadingIntensity:
        return self.loading_intensity_of(self.memory_percent(), type='memory')

    def bandwidth_loading_intensity(self) -> LoadingIntensity:
        return self.loading_intensity_of(self.memory_utilization(), type='memory')

    def gpu_loading_intensity(self) -> LoadingIntensity:
        return self.loading_intensity_of(self.gpu_utilization(), type='gpu')

    def power_loading_intensity(self) -> LoadingIntensity:
        return self.loading_intensity_of(self.power_utilization(), type='gpu')

    def loading_intensity(self) -> LoadingIntensity:
        return max(self.memory_loading_intensity(), self.gpu_loading_intensity())

    def display_color(self) -> str:
        if self.name().startswith('ERROR:'):
            return 'red'
        return self.loading_intensity().color()

    def memory_display_color(self) -> str:
        if self.name().startswith('ERROR:'):
            return 'red'
        return self.memory_loading_intensity().color()

    def bandwidth_display_color(self) -> str:
        if self.name().startswith('ERROR:'):
            return 'red'
        return self.bandwidth_loading_intensity().color()

    def gpu_display_color(self) -> str:
        if self.name().startswith('ERROR:'):
            return 'red'
        return self.gpu_loading_intensity().color()

    def power_display_color(self) -> str:
        if self.name().startswith('ERROR:'):
            return 'red'
        return self.power_loading_intensity().color()

    @staticmethod
    def loading_intensity_of(
        utilization: float | str,
        type: Literal['memory', 'gpu'] = 'memory',  # pylint: disable=redefined-builtin
    ) -> LoadingIntensity:
        thresholds = {
            'memory': Device.MEMORY_UTILIZATION_THRESHOLDS,
            'gpu': Device.GPU_UTILIZATION_THRESHOLDS,
        }[type]
        if utilization is NA:
            return LoadingIntensity.MODERATE
        if isinstance(utilization, str):
            utilization = utilization.replace('%', '')
        utilization = float(utilization)
        if utilization >= thresholds[-1]:
            return LoadingIntensity.HEAVY
        if utilization >= thresholds[0]:
            return LoadingIntensity.MODERATE
        return LoadingIntensity.LIGHT

    @staticmethod
    def color_of(
        utilization: float | str,
        type: Literal['memory', 'gpu'] = 'memory',  # pylint: disable=redefined-builtin
    ) -> str:
        return Device.loading_intensity_of(utilization, type=type).color()


class MigDevice(MigDeviceBase, Device):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._snapshot: Snapshot | None = None
        self.tuple_index: tuple[int] | tuple[int, int] = (
            (self.index,) if isinstance(self.index, int) else self.index
        )
        self.display_index: str = ':'.join(map(str, self.tuple_index))

    def memory_usage(self) -> str:  # string of used memory over total memory (in human-readable)
        return f'{self.memory_used_human()} / {self.memory_total_human():>8s}'

    loading_intensity = Device.memory_loading_intensity

    SNAPSHOT_KEYS: ClassVar[list[str]] = [
        'name',
        'memory_used',
        'memory_free',
        'memory_total',
        'memory_used_human',
        'memory_free_human',
        'memory_total_human',
        'memory_percent',
        'memory_usage',
        'bar1_memory_used_human',
        'bar1_memory_percent',
        'gpu_utilization',
        'memory_utilization',
        'total_volatile_uncorrected_ecc_errors',
        'mig_mode',
        'is_mig_device',
        'gpu_instance_id',
        'compute_instance_id',
        'memory_percent_string',
        'memory_utilization_string',
        'gpu_utilization_string',
        'memory_loading_intensity',
        'memory_display_color',
        'gpu_loading_intensity',
        'gpu_display_color',
        'loading_intensity',
        'display_color',
    ]
