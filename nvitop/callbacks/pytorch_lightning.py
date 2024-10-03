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

# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=unused-argument,attribute-defined-outside-init

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pytorch_lightning.callbacks import Callback  # pylint: disable=import-error
from pytorch_lightning.utilities import rank_zero_only  # pylint: disable=import-error
from pytorch_lightning.utilities.exceptions import (  # pylint: disable=import-error
    MisconfigurationException,
)

from nvitop.api import libnvml
from nvitop.callbacks.utils import get_devices_by_logical_ids, get_gpu_stats


if TYPE_CHECKING:
    import pytorch_lightning as pl


# Modified from pytorch_lightning.callbacks.GPUStatsMonitor
class GpuStatsLogger(Callback):  # pylint: disable=too-many-instance-attributes
    """Automatically log GPU stats during training stage. :class:`GpuStatsLogger` is a callback and
    in order to use it you need to assign a logger in the ``Trainer``.

    Args:
        memory_utilization (bool):
            Set to :data:`True` to log used, free and the percentage of memory utilization at the
            start and end of each step. Default: :data:`True`.
        gpu_utilization (bool):
            Set to :data:`True` to log the percentage of GPU utilization at the start and end of
            each step. Default: :data:`True`.
        intra_step_time (bool):
            Set to :data:`True` to log the time of each step. Default: :data:`False`.
        inter_step_time (bool):
            Set to :data:`True` to log the time between the end of one step and the start of the
            next step. Default: :data:`False`.
        fan_speed (bool):
            Set to :data:`True` to log percentage of fan speed. Default: :data:`False`.
        temperature (bool):
            Set to :data:`True` to log the gpu temperature in degree Celsius. Default: :data:`False`.

    Raises:
        MisconfigurationException:
            If NVIDIA driver is not installed, not running on GPUs, or ``Trainer`` has no logger.

    Examples:
        >>> from pytorch_lightning import Trainer
        >>> from nvitop.callbacks.pytorch_lightning import GpuStatsLogger
        >>> gpu_stats = GpuStatsLogger()
        >>> trainer = Trainer(gpus=[..], logger=True, callbacks=[gpu_stats])

    GPU stats are mainly based on NVML queries. The description of the queries is as follows:

    - **fan.speed** - The fan speed value is the percent of maximum speed that the device's fan is
      currently intended to run at. It ranges from 0 to 100 %. Note: The reported speed is the
      intended fan speed. If the fan is physically blocked and unable to spin, this output will not
      match the actual fan speed. Many parts do not report fan speeds because they rely on cooling
      via fans in the surrounding enclosure.
    - **memory.used** - Total memory allocated by active contexts, in MiBs.
    - **memory.free** - Total free memory, in MiBs.
    - **utilization.gpu** - Percent of time over the past sample period during which one or more
      kernels was executing on the GPU. The sample period may be between 1 second and 1/6 second
      depending on the product.
    - **utilization.memory** - Percent of time over the past sample period during which global
      (device) memory was being read or written. The sample period may be between 1 second and 1/6
      second depending on the product.
    - **temperature** - Core GPU temperature, in degrees C.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        intra_step_time: bool = False,
        inter_step_time: bool = False,
        fan_speed: bool = False,
        temperature: bool = False,
    ) -> None:
        super().__init__()

        try:
            libnvml.nvmlInit()
        except libnvml.NVMLError as ex:
            raise MisconfigurationException(
                'Cannot use GpuStatsLogger callback because NVIDIA driver is not installed.',
            ) from ex

        self._memory_utilization = memory_utilization
        self._gpu_utilization = gpu_utilization
        self._intra_step_time = intra_step_time
        self._inter_step_time = inter_step_time
        self._fan_speed = fan_speed
        self._temperature = temperature

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use GpuStatsLogger callback with Trainer that has no logger.',
            )

        if trainer.strategy.root_device.type != 'cuda':
            raise MisconfigurationException(
                f'You are using GpuStatsLogger but are not running on GPU. '
                f'The root device type is {trainer.strategy.root_device.type}.',
            )

        try:
            device_ids = trainer.device_ids  # pytorch-lightning >= 1.6.0
        except AttributeError:
            device_ids = trainer.data_parallel_device_ids  # pytorch-lightning < 1.6.0

        try:
            self._devices = get_devices_by_logical_ids(device_ids, unique=True)
        except (libnvml.NVMLError, RuntimeError) as ex:
            raise ValueError(
                f'Cannot use GpuStatsLogger callback because devices unavailable. '
                f'Received: `gpus={device_ids}`',
            ) from ex

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(  # pylint: disable=arguments-differ
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **kwargs: Any,
    ) -> None:
        if self._intra_step_time:
            self._snap_intra_step_time = time.monotonic()

        logs = self._get_gpu_stats()

        if self._inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs['batch_time/inter_step (ms)'] = 1000.0 * (
                time.monotonic() - self._snap_inter_step_time
            )

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(  # pylint: disable=arguments-differ
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **kwargs: Any,
    ) -> None:
        if self._inter_step_time:
            self._snap_inter_step_time = time.monotonic()

        logs = self._get_gpu_stats()

        if self._intra_step_time and self._snap_intra_step_time:
            logs['batch_time/intra_step (ms)'] = 1000.0 * (
                time.monotonic() - self._snap_intra_step_time
            )

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    def _get_gpu_stats(self) -> dict[str, float]:
        """Get the gpu status from NVML queries."""
        return get_gpu_stats(
            devices=self._devices,
            memory_utilization=self._memory_utilization,
            gpu_utilization=self._gpu_utilization,
            fan_speed=self._fan_speed,
            temperature=self._temperature,
        )
