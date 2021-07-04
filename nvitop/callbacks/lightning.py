# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=unused-argument,attribute-defined-outside-init

import time
from typing import Any, Dict

from pytorch_lightning.callbacks import Callback   # pylint: disable=import-error
from pytorch_lightning.utilities import DeviceType, rank_zero_only  # pylint: disable=import-error
from pytorch_lightning.utilities.exceptions import MisconfigurationException  # pylint: disable=import-error

from nvitop.core import nvml, MiB, NA
from nvitop.callbacks.utils import get_devices_by_logical_ids


# Modified from pytorch_lightning.callbacks.GPUStatsMonitor
class GpuStatsLogger(Callback):
    r"""
    Automatically log GPU stats during training stage. ``GpuStatsLogger`` is a
    callback and in order to use it you need to assign a logger in the ``Trainer``.

    Args:
        memory_utilization: Set to ``True`` to log used, free and the percentage of memory
            utilization at the start and end of each step. Default: ``True``.
        gpu_utilization: Set to ``True`` to log the percentage of GPU utilization
            at the start and end of each step. Default: ``True``.
        intra_step_time: Set to ``True`` to log the time of each step. Default: ``False``.
        inter_step_time: Set to ``True`` to log the time between the end of one step
            and the start of the next step. Default: ``False``.
        fan_speed: Set to ``True`` to log percentage of fan speed. Default: ``False``.
        temperature: Set to ``True`` to log the gpu temperature in degree Celsius.
            Default: ``False``.

    Raises:
        MisconfigurationException:
            If NVIDIA driver is not installed, not running on GPUs, or ``Trainer`` has no logger.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from nvitop.callbacks.lightning import GpuStatsLogger
        >>> gpu_stats = GpuStatsLogger() # doctest: +SKIP
        >>> trainer = Trainer(gpus=[..], logger=True, callbacks=[gpu_stats]) # doctest: +SKIP

    GPU stats are mainly based on NVML queries. The description of the queries is as follows:

    - **fan.speed** – The fan speed value is the percent of maximum speed that the device's fan is currently
      intended to run at. It ranges from 0 to 100 %. Note: The reported speed is the intended fan speed.
      If the fan is physically blocked and unable to spin, this output will not match the actual fan speed.
      Many parts do not report fan speeds because they rely on cooling via fans in the surrounding enclosure.
    - **memory.used** – Total memory allocated by active contexts.
    - **memory.free** – Total free memory.
    - **utilization.gpu** – Percent of time over the past sample period during which one or more kernels was
      executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product.
    - **utilization.memory** – Percent of time over the past sample period during which global (device) memory was
      being read or written. The sample period may be between 1 second and 1/6 second depending on the product.
    - **temperature** – Core GPU temperature, in degrees C.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        intra_step_time: bool = False,
        inter_step_time: bool = False,
        fan_speed: bool = False,
        temperature: bool = False
    ):
        super().__init__()

        try:
            nvml.nvmlInit()
        except nvml.NVMLError as ex:
            raise MisconfigurationException(
                'Cannot use GpuStatsLogger callback because NVIDIA driver is not installed.'
            ) from ex

        self._memory_utilization = memory_utilization
        self._gpu_utilization = gpu_utilization
        self._intra_step_time = intra_step_time
        self._inter_step_time = inter_step_time
        self._fan_speed = fan_speed
        self._temperature = temperature

    def on_train_start(self, trainer, pl_module) -> None:
        if not trainer.logger:
            raise MisconfigurationException('Cannot use GpuStatsLogger callback with Trainer that has no logger.')

        if trainer._device_type != DeviceType.GPU:  # pylint: disable=protected-access
            raise MisconfigurationException(
                'You are using GpuStatsLogger but are not running on GPU '
                'since gpus attribute in Trainer is set to {}.'.format(trainer.gpus)
            )

        device_ids = trainer.data_parallel_device_ids
        try:
            self._devices = get_devices_by_logical_ids(device_ids, unique=True)
        except (nvml.NVMLError, RuntimeError) as ex:
            raise ValueError(
                'Cannot use GpuStatsLogger callback because devices unavailable. '
                'Received: `gpus={}`'.format(device_ids)
            ) from ex

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module,
                             batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self._intra_step_time:
            self._snap_intra_step_time = time.monotonic()

        logs = self._get_gpu_stats()

        if self._inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs['batch_time/inter_step (ms)'] = (time.monotonic() - self._snap_inter_step_time) * 1000.0

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module,
                           outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self._inter_step_time:
            self._snap_inter_step_time = time.monotonic()

        logs = self._get_gpu_stats()

        if self._intra_step_time and self._snap_intra_step_time:
            logs['batch_time/intra_step (ms)'] = (time.monotonic() - self._snap_intra_step_time) * 1000.0

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get the gpu stats from NVML queries"""

        stats = {}
        for device in self._devices:
            prefix = 'gpu_id: {}'.format(device.cuda_index)
            if device.cuda_index != device.index:
                prefix += ' (real index: {})'.format(device.index)
            if self._memory_utilization or self._gpu_utilization:
                utilization = device.utilization_rates()
                if self._memory_utilization:
                    memory_utilization = float(utilization.memory if utilization is not NA else NA)
                    stats['{}/utilization.memory (%)'.format(prefix)] = memory_utilization
                if self._gpu_utilization:
                    gpu_utilization = float(utilization.gpu if utilization is not NA else NA)
                    stats['{}/utilization.gpu (%)'.format(prefix)] = float(gpu_utilization)
            if self._memory_utilization:
                stats['{}/memory.used (MiB)'.format(prefix)] = float(device.memory_used()) / MiB
                stats['{}/memory.free (MiB)'.format(prefix)] = float(device.memory_free()) / MiB
            if self._fan_speed:
                stats['{}/fan.speed (%)'.format(prefix)] = float(device.fan_speed())
            if self._temperature:
                stats['{}/temperature.gpu (℃)'.format(prefix)] = float(device.fan_speed())

        return stats
