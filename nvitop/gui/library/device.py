# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

from nvitop.core import NA, Device as DeviceBase
from nvitop.gui.library.process import GpuProcess


__all__ = ['Device', 'NA']


class Device(DeviceBase):
    GPU_PROCESS_CLASS = GpuProcess

    MEMORY_UTILIZATION_THRESHOLDS = (10, 80)
    GPU_UTILIZATION_THRESHOLDS = (10, 75)
    INTENSITY2COLOR = {'light': 'green', 'moderate': 'yellow', 'heavy': 'red'}

    SNAPSHOT_KEYS = [
        *DeviceBase.SNAPSHOT_KEYS,
        'memory_loading_intensity', 'memory_display_color',
        'gpu_loading_intensity', 'gpu_display_color',
        'loading_intensity', 'display_color'
    ]

    def memory_loading_intensity(self):
        return self.loading_intensity_of(self.memory_utilization(), type='memory')

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
            return 'heavy'
        if isinstance(utilization, str):
            utilization = utilization[:-1]
        utilization = float(utilization)
        if utilization >= thresholds[-1]:
            return 'heavy'
        if utilization >= thresholds[0]:
            return 'moderate'
        return 'light'

    @staticmethod
    def color_of(utilization, type='memory'):  # pylint: disable=redefined-builtin
        return Device.INTENSITY2COLOR.get(Device.loading_intensity_of(utilization, type=type))
