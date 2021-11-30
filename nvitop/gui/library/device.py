# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

from nvitop.core import Device as DeviceBase, NA, Snapshot
from nvitop.gui.library.process import GpuProcess


__all__ = ['Device', 'NA']


class Device(DeviceBase):
    GPU_PROCESS_CLASS = GpuProcess

    MEMORY_UTILIZATION_THRESHOLDS = (10, 80)
    GPU_UTILIZATION_THRESHOLDS = (10, 75)
    INTENSITY2COLOR = {'light': 'green', 'moderate': 'yellow', 'heavy': 'red'}

    SNAPSHOT_KEYS = [
        'name', 'bus_id',

        'memory_used', 'memory_free', 'memory_total',
        'memory_used_human', 'memory_free_human', 'memory_total_human',
        'memory_percent', 'memory_usage',

        'gpu_utilization', 'memory_utilization',

        'fan_speed', 'temperature',

        'power_usage', 'power_limit', 'power_status',

        'display_active', 'current_driver_model',
        'persistence_mode', 'performance_state',
        'total_volatile_uncorrected_ecc_errors', 'compute_mode',

        'memory_percent_string', 'memory_utilization_string', 'gpu_utilization_string',
        'fan_speed_string', 'temperature_string',

        'memory_loading_intensity', 'memory_display_color',
        'gpu_loading_intensity', 'gpu_display_color',
        'loading_intensity', 'display_color'
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._snapshot = None

    def as_snapshot(self):
        self._snapshot = super().as_snapshot()
        return self._snapshot

    @property
    def snapshot(self) -> Snapshot:
        if self._snapshot is None:
            self.as_snapshot()
        return self._snapshot

    def memory_loading_intensity(self):
        return self.loading_intensity_of(self.memory_percent(), type='memory')

    def gpu_loading_intensity(self):
        return self.loading_intensity_of(self.gpu_utilization(), type='gpu')

    def loading_intensity(self):
        loading_intensity = (self.memory_loading_intensity(), self.gpu_loading_intensity())
        if 'heavy' in loading_intensity:
            return 'heavy'
        if 'moderate' in loading_intensity:
            return 'moderate'
        return 'light'

    def display_color(self):
        return self.INTENSITY2COLOR.get(self.loading_intensity())

    def memory_display_color(self):
        return self.INTENSITY2COLOR.get(self.memory_loading_intensity())

    def gpu_display_color(self):
        return self.INTENSITY2COLOR.get(self.gpu_loading_intensity())

    @staticmethod
    def loading_intensity_of(utilization, type='memory'):  # pylint: disable=redefined-builtin
        thresholds = {'memory': Device.MEMORY_UTILIZATION_THRESHOLDS,
                      'gpu': Device.GPU_UTILIZATION_THRESHOLDS}.get(type)
        if utilization is NA:
            return 'moderate'
        if isinstance(utilization, str):
            utilization = utilization.replace('%', '')
        utilization = float(utilization)
        if utilization >= thresholds[-1]:
            return 'heavy'
        if utilization >= thresholds[0]:
            return 'moderate'
        return 'light'

    @staticmethod
    def color_of(utilization, type='memory'):  # pylint: disable=redefined-builtin
        return Device.INTENSITY2COLOR.get(Device.loading_intensity_of(utilization, type=type))
