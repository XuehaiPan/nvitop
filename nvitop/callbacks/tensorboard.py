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

# pylint: disable=missing-module-docstring

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np

    try:
        from tensorboard.summary import Writer as SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            pass


def add_scalar_dict(
    writer: SummaryWriter,
    main_tag: str,
    tag_scalar_dict: dict[str, int | float | np.floating],
    global_step: int | np.integer | None = None,
    walltime: float | None = None,
) -> None:
    """Add a batch of scalars to the writer.

    Batched version of ``writer.add_scalar``.
    """
    for tag, scalar in tag_scalar_dict.items():
        writer.add_scalar(f'{main_tag}/{tag}', scalar, global_step=global_step, walltime=walltime)
