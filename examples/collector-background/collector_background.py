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
"""Run `ResourceMetricCollector` on a background daemon thread via `collect_in_background`."""

from __future__ import annotations

import time

from nvitop import Device, ResourceMetricCollector, collect_in_background


class InMemoryLogger:
    """Minimal stand-in for whatever logger you would normally write metrics to.

    For long-running deployments, swap ``self.records`` for a bounded buffer
    (e.g., ``collections.deque(maxlen=...)``) or stream samples to disk.
    """

    def __init__(self) -> None:
        """Start with an empty record buffer and the logger open."""
        self._closed = False
        self.records: list[dict[str, float]] = []

    def log(self, metrics: dict[str, float]) -> None:
        """Append one collector sample to the in-memory buffer."""
        self.records.append(metrics)

    def close(self) -> None:
        """Mark the logger closed so the next ``on_collect`` call stops the daemon."""
        self._closed = True

    def is_closed(self) -> bool:
        """Return :data:`True` once :meth:`close` has been called."""
        return self._closed


def main() -> None:
    """Drive `collect_in_background` for a few seconds, then stop cleanly."""
    logger = InMemoryLogger()

    def on_collect(metrics: dict[str, float]) -> bool:
        """Forward a sample to the logger; return False to stop the daemon."""
        if logger.is_closed():
            return False
        logger.log(metrics)
        print(f'Collected {len(metrics)} metrics at t={metrics["resources/timestamp"]:.1f}')
        return True

    def on_stop(_collector: ResourceMetricCollector) -> None:
        """Close the logger once the daemon thread shuts down."""
        if not logger.is_closed():
            logger.close()

    # Record metrics in the background every 2 seconds (5-second intervals are nicer in real use)
    collect_in_background(
        on_collect,
        ResourceMetricCollector(Device.cuda.all() or Device.all()),
        interval=2.0,
        on_stop=on_stop,
        tag='resources',
    )

    # Let the daemon collect a handful of samples, then shut it down by closing the logger
    time.sleep(8)
    logger.close()
    time.sleep(2.5)  # give the daemon one more tick to notice and exit
    print(f'Total records collected: {len(logger.records)}')


if __name__ == '__main__':
    main()
