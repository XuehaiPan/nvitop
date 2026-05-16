# Background `ResourceMetricCollector`

Demonstrates [`nvitop.collect_in_background`][cib]: the collector ticks on a daemon thread, calls your `on_collect` callback with each sample, and shuts itself down cleanly when the callback returns `False`. A small `InMemoryLogger` stands in for whatever sink ([TensorBoard], file, [Slack] webhook, …) you would normally write to.

## APIs Used

- [`nvitop.collect_in_background`][cib]
- [`nvitop.ResourceMetricCollector`][collector]
- [`nvitop.Device.cuda.all()`][cuda-all]

The same pattern is available as a one-liner via [`ResourceMetricCollector.daemonize`][daemonize].

## Run

```bash
python3 examples/collector-background/collector_background.py
```

The script collects in the background for ~10 seconds and prints each tick.

See [`../README.md`](../README.md) for the full example index.

[Slack]: https://slack.com
[TensorBoard]: https://www.tensorflow.org/tensorboard
[cib]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.collect_in_background
[collector]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.ResourceMetricCollector
[cuda-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.CudaDevice.all
[daemonize]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.ResourceMetricCollector.daemonize
