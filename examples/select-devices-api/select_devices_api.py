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
"""Use `nvitop.select_devices` to programmatically pick CUDA devices for a job."""

from __future__ import annotations

import os

from nvitop import select_devices


def main() -> None:
    """Pick CUDA devices with `nvitop.select_devices` and set ``CUDA_VISIBLE_DEVICES``."""
    # Equivalent to the `nvisel` CLI, but as a Python call
    indices = select_devices(format='index', min_count=1, min_free_memory='100MiB')
    print(f'Selected indices: {indices}')

    uuids = select_devices(format='uuid', min_count=1, min_free_memory='100MiB')
    print(f'Selected UUIDs:   {uuids}')

    # Typical use: set `CUDA_VISIBLE_DEVICES` before importing torch/tensorflow
    if uuids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(uuids)
        print(f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')


if __name__ == '__main__':
    main()
