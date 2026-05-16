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
"""Log `ResourceMetricCollector` output to TensorBoard alongside a tiny training loop."""

from __future__ import annotations

import os

import torch  # pylint: disable=import-error
from torch import nn  # pylint: disable=import-error
from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-error

from nvitop import CudaDevice, ResourceMetricCollector


def add_scalar_dict(
    writer: SummaryWriter,
    main_tag: str,
    tag_scalar_dict: dict[str, float],
    global_step: int | None = None,
    walltime: float | None = None,
) -> None:
    """Write a flat dict of scalars under a shared main tag."""
    for tag, scalar in tag_scalar_dict.items():
        writer.add_scalar(f'{main_tag}/{tag}', scalar, global_step=global_step, walltime=walltime)


def main() -> None:
    """Run a tiny training loop with `ResourceMetricCollector` logging into TensorBoard."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = nn.Linear(16, 4).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    writer = SummaryWriter()
    collector = ResourceMetricCollector(
        devices=CudaDevice.all(),  # log all visible CUDA devices and use the CUDA ordinal
        root_pids={os.getpid()},  # only log descendant processes of the current process
        interval=1.0,  # snapshot interval for background daemon thread
    )

    global_step = 0
    num_epoch = 2
    steps_per_epoch = 5
    for epoch in range(num_epoch):
        with collector(tag='train'):
            for _ in range(steps_per_epoch):
                inputs = torch.randn(8, 16, device=device)
                targets = torch.randn(8, 4, device=device)

                with collector(tag='batch'):
                    optimizer.zero_grad()
                    loss = criterion(net(inputs), targets)
                    loss.backward()
                    optimizer.step()

                    global_step += 1
                    add_scalar_dict(
                        writer,
                        'train',
                        {'loss': loss.item()},
                        global_step=global_step,
                    )
                    add_scalar_dict(  # tag='resources/train/batch/...'
                        writer,
                        'resources',
                        collector.collect(),
                        global_step=global_step,
                    )

            add_scalar_dict(  # tag='resources/train/...'
                writer,
                'resources',
                collector.collect(),
                global_step=epoch,
            )

    writer.close()
    print(f'Logged {global_step} steps to {writer.log_dir}.')


if __name__ == '__main__':
    main()
