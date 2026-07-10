# Colored GPU Monitor

A richer version of the minimal monitor. Uses [`nvitop.colored`][colored] to highlight the device name, field labels, and column headers, and adds a host summary line (CPU, memory, swap, and load average), a two-column device panel, and per-process snapshots taken via [`GpuProcess.take_snapshots`][take-snapshots].

## APIs Used

- [`nvitop.CudaDevice.all()`][cuda-all] and [`nvitop.GpuProcess.take_snapshots`][take-snapshots]
- [`nvitop.host`][host] for host metrics (`cpu_percent`, `virtual_memory`, `swap_memory`, `load_average`, `hostname`, `getuser`)
- [`nvitop.bytes2human`][bytes2human] to format host memory sizes
- [`nvitop.colored`][colored] and [`nvitop.NA`][na]

## Run

```bash
python3 examples/monitor-colored/monitor_colored.py
```

Or run it without cloning the repository, using [uv](https://docs.astral.sh/uv):

```bash
uv run https://github.com/XuehaiPan/nvitop/raw/HEAD/examples/monitor-colored/monitor_colored.py
```

Requires only `nvitop` itself; no other dependencies.

See [`../README.md`](../README.md) for the full example index.

[bytes2human]: https://nvitop.readthedocs.io/en/latest/api/utils.html#nvitop.bytes2human
[colored]: https://nvitop.readthedocs.io/en/latest/api/utils.html#nvitop.colored
[cuda-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.CudaDevice.all
[host]: https://nvitop.readthedocs.io/en/latest/api/host.html
[na]: https://nvitop.readthedocs.io/en/latest/api/utils.html#nvitop.NA
[take-snapshots]: https://nvitop.readthedocs.io/en/latest/api/process.html#nvitop.GpuProcess.take_snapshots
