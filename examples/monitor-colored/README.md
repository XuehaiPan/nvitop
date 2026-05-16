# Colored GPU Monitor

A richer version of the minimal monitor that uses [`nvitop.colored`][colored] to highlight device names, sections, and column headers, plus per-process snapshots taken via [`GpuProcess.take_snapshots`][take-snapshots].

## APIs Used

- [`nvitop.CudaDevice.all()`][cuda-all]
- [`nvitop.GpuProcess.take_snapshots`][take-snapshots]
- [`nvitop.colored`][colored], [`nvitop.NA`][na]

## Run

```bash
python3 examples/monitor-colored/monitor_colored.py
```

Requires only `nvitop` itself; no other dependencies.

See [`../README.md`](../README.md) for the full example index.

[colored]: https://nvitop.readthedocs.io/en/latest/api/utils.html#nvitop.colored
[cuda-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.CudaDevice.all
[na]: https://nvitop.readthedocs.io/en/latest/api/utils.html#nvitop.NA
[take-snapshots]: https://nvitop.readthedocs.io/en/latest/api/process.html#nvitop.GpuProcess.take_snapshots
