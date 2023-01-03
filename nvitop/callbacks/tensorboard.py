# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2022 Xuehai Pan. All Rights Reserved.
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


def add_scalar_dict(writer, main_tag, tag_scalar_dict, global_step=None, walltime=None):
    """Add a batch of scalars to the writer.

    Batched version of ``writer.add_scalar``.
    """
    for tag, scalar in tag_scalar_dict.items():
        writer.add_scalar(f'{main_tag}/{tag}', scalar, global_step=global_step, walltime=walltime)
