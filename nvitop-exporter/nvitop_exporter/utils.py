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
"""Utility functions for ``nvitop-exporter``."""

import socket


__all__ = ['get_ip_address']


# Reference: https://stackoverflow.com/a/28950776
def get_ip_address() -> str:
    """Best-effort guess of a routable local IPv4 address; silently falls back to ``127.0.0.1``."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.0)
    try:
        # Doesn't even have to be reachable; the kernel picks the source IP for the route.
        s.connect(('10.254.254.254', 1))
        return str(s.getsockname()[0])
    except OSError:
        # Non-blocking UDP probe surfaces every failure (no route, gaierror, EHOSTUNREACH, ...)
        # as an OSError subclass; collapse them all into the loopback fallback.
        return '127.0.0.1'
    finally:
        s.close()
