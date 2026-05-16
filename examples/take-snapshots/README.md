# `take_snapshots` Demo

Exercises every form of [`nvitop.take_snapshots`][take-snapshots] — the helper that captures the live state of both devices and processes in a single pass and returns plain dataclasses safe to serialize, cache, or pass between threads.

## APIs Used

- [`nvitop.take_snapshots`][take-snapshots]
- [`nvitop.Device.all()`][device-all]
- [`nvitop.Device.cuda.all()`][cuda-all]

## Run

```bash
python3 examples/take-snapshots/take_snapshots_demo.py
```

Requires only `nvitop` itself; no other dependencies.

See [`../README.md`](../README.md) for the full example index.

[cuda-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.CudaDevice.all
[device-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.Device.all
[take-snapshots]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.take_snapshots
