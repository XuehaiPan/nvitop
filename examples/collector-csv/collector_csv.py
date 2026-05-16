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
"""Sample resource metrics with `ResourceMetricCollector` and append to a CSV file."""

from __future__ import annotations

import datetime
import time

import pandas as pd  # pylint: disable=import-error

from nvitop import ResourceMetricCollector


# Keep the demo short; bump this for a longer log
SAMPLES = 5
SAMPLE_INTERVAL_SECONDS = 2.0


def main() -> None:
    """Poll `ResourceMetricCollector` on a fixed interval and append to ``results.csv``."""
    collector = ResourceMetricCollector(root_pids={1}, interval=SAMPLE_INTERVAL_SECONDS)
    rows: list[dict[str, float]] = []

    with collector(tag='resources'):
        for _ in range(SAMPLES):
            time.sleep(SAMPLE_INTERVAL_SECONDS)

            metrics = collector.collect()
            rows.append(metrics)

    df = pd.DataFrame.from_records(rows)
    df.insert(0, 'time', df['resources/timestamp'].map(datetime.datetime.fromtimestamp))
    df.to_csv('results.csv', index=False)
    print(f'Wrote {len(df)} rows to results.csv')


if __name__ == '__main__':
    main()
