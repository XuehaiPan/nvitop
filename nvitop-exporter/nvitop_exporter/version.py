# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
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
"""Prometheus exporter built on top of ``nvitop``."""

# pylint: disable=invalid-name

__version__ = '1.6.0'
__license__ = 'Apache-2.0'
__author__ = __maintainer__ = 'Xuehai Pan'
__email__ = 'XuehaiPan@pku.edu.cn'
__release__ = False

if not __release__:
    import os
    import subprocess

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        prefix, sep, suffix = (
            subprocess.check_output(  # noqa: S603
                [  # noqa: S607
                    'git',
                    f'--git-dir={os.path.join(root_dir, ".git")}',
                    'describe',
                    '--abbrev=7',
                ],
                cwd=root_dir,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .lstrip('v')
            .replace('-', '.dev', 1)
            .replace('-', '+', 1)
            .partition('.dev')
        )
        if sep:
            version_prefix, dot, version_tail = prefix.rpartition('.')
            prefix = f'{version_prefix}{dot}{int(version_tail) + 1}'
            __version__ = f'{prefix}{sep}{suffix}'
            del version_prefix, dot, version_tail
        else:
            __version__ = prefix
        del prefix, sep, suffix
    except (OSError, subprocess.CalledProcessError):
        pass

    del os, subprocess, root_dir
