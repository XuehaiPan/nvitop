# Collector → CSV

Polls [`ResourceMetricCollector.collect()`][collect] on a fixed interval and appends each sample to a [pandas] `DataFrame`, then writes the result to `results.csv` with a parsed `time` column. Designed to be the simplest possible offline logger.

## APIs Used

- [`nvitop.ResourceMetricCollector`][collector]

## Install + Run

```bash
pip install -r examples/collector-csv/requirements.txt
python3 examples/collector-csv/collector_csv.py
```

By default the script takes 5 samples at 2-second intervals (≈10s wall time). Bump the `SAMPLES` / `SAMPLE_INTERVAL_SECONDS` constants at the top of the file for a longer run.

See [`../README.md`](../README.md) for the full example index.

[collect]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.ResourceMetricCollector.collect
[collector]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.ResourceMetricCollector
[pandas]: https://pandas.pydata.org
