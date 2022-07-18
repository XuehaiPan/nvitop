# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=unused-argument,attribute-defined-outside-init

import re
import time
from typing import Dict, List, Tuple, Union

from tensorflow.python.keras.callbacks import (  # pylint: disable=import-error,no-name-in-module
    Callback,
)

from nvitop.callbacks.utils import get_devices_by_logical_ids, get_gpu_stats
from nvitop.core import libnvml


# Ported version of .pytorch_lightning.GpuStatsLogger for Keras
class GpuStatsLogger(Callback):  # pylint: disable=too-many-instance-attributes
    r"""
    Automatically log GPU stats during training stage. :class:`GpuStatsLogger` is a
    callback and in order to use it you need to assign a TensorBoard callback or
    a CSVLogger callback to the model.

    Args:
        memory_utilization (bool):
            Set to :data:`True` to log used, free and the percentage of memory
            utilization at the start and end of each step. Default: :data:`True`.
        gpu_utilization (bool):
            Set to :data:`True` to log the percentage of GPU utilization
            at the start and end of each step. Default: :data:`True`.
        intra_step_time (bool):
            Set to :data:`True` to log the time of each step. Default: :data:`False`.
        inter_step_time (bool):
            Set to :data:`True` to log the time between the end of one step
            and the start of the next step. Default: :data:`False`.
        fan_speed (bool):
            Set to :data:`True` to log percentage of fan speed. Default: :data:`False`.
        temperature (bool):
            Set to :data:`True` to log the gpu temperature in degree Celsius.
            Default: :data:`False`.

    Raises:
        ValueError:
            If NVIDIA driver is not installed, or the `gpus` argument does not match available devices.

    Example::

        >>> from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
        >>> from tensorflow.python.keras.callbacks import TensorBoard
        >>> from nvitop.callbacks.keras import GpuStatsLogger
        >>> gpus = ['/gpu:0', '/gpu:1']  # or gpus = [0, 1] or gpus = 2
        >>> model = Xception(weights=None, ..)
        >>> model = multi_gpu_model(model, gpus)
        >>> model.compile(..)
        >>> tb_callback = TensorBoard(log_dir='./logs')
        >>> gpu_stats = GpuStatsLogger(gpus)
        >>> model.fit(.., callbacks=[gpu_stats, tb_callback])

    Note::
        The GpuStatsLogger callback should be placed before the TensorBoard / CSVLogger callback.

    GPU stats are mainly based on NVML queries. The description of the queries is as follows:

    - **fan.speed** - The fan speed value is the percent of maximum speed that the device's fan is currently
      intended to run at. It ranges from 0 to 100 %. Note: The reported speed is the intended fan speed.
      If the fan is physically blocked and unable to spin, this output will not match the actual fan speed.
      Many parts do not report fan speeds because they rely on cooling via fans in the surrounding enclosure.
    - **memory.used** - Total memory allocated by active contexts, in MiBs.
    - **memory.free** - Total free memory, in MiBs.
    - **utilization.gpu** - Percent of time over the past sample period during which one or more kernels was
      executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product.
    - **utilization.memory** - Percent of time over the past sample period during which global (device) memory was
      being read or written. The sample period may be between 1 second and 1/6 second depending on the product.
    - **temperature** - Core GPU temperature, in degrees C.
    """

    GPU_NAME_PATTERN = re.compile(r'^/(\w*device:)?GPU:(?P<ID>\d+)$', flags=re.IGNORECASE)

    def __init__(  # pylint: disable=too-many-arguments
        self,
        gpus: Union[int, Union[List[Union[int, str]], Tuple[Union[int, str], ...]]],
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
            raise ValueError(
                'Cannot use the GpuStatsLogger callback because the NVIDIA driver is not installed.'
            ) from ex

        if isinstance(gpus, (list, tuple)):
            gpus = list(gpus)
            for i, gpu_id in enumerate(gpus):
                if isinstance(gpu_id, str) and self.GPU_NAME_PATTERN.match(gpu_id):
                    gpus[i] = self.GPU_NAME_PATTERN.match(gpu_id).group('ID')
            gpu_ids = sorted(set(map(int, gpus)))
        else:
            gpu_ids = list(range(gpus))

        try:
            self._devices = get_devices_by_logical_ids(gpu_ids, unique=True)
        except (libnvml.NVMLError, RuntimeError) as ex:
            raise ValueError(
                'Cannot use GpuStatsLogger callback because devices unavailable. '
                'Received: `gpus={}`'.format(gpu_ids)
            ) from ex

        self._memory_utilization = memory_utilization
        self._gpu_utilization = gpu_utilization
        self._intra_step_time = intra_step_time
        self._inter_step_time = inter_step_time
        self._fan_speed = fan_speed
        self._temperature = temperature

    def on_train_epoch_start(self, epoch, logs=None) -> None:
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None

    def on_train_batch_start(self, batch, logs=None) -> None:
        logs = logs or {}

        if self._intra_step_time:
            self._snap_intra_step_time = time.monotonic()

        logs.update(self._get_gpu_stats())

        if self._inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs['batch_time/inter_step (ms)'] = 1000.0 * (
                time.monotonic() - self._snap_inter_step_time
            )

    def on_train_batch_end(self, batch, logs=None) -> None:
        logs = logs or {}

        if self._inter_step_time:
            self._snap_inter_step_time = time.monotonic()

        logs.update(self._get_gpu_stats())

        if self._intra_step_time and self._snap_intra_step_time:
            logs['batch_time/intra_step (ms)'] = 1000.0 * (
                time.monotonic() - self._snap_intra_step_time
            )

    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get the gpu status from NVML queries"""

        return get_gpu_stats(
            devices=self._devices,
            memory_utilization=self._memory_utilization,
            gpu_utilization=self._gpu_utilization,
            fan_speed=self._fan_speed,
            temperature=self._temperature,
        )
