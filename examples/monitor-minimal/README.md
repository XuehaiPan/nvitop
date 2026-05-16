# Minimal GPU Monitor

A one-shot, no-extras script that prints the same information `nvitop` shows in its TUI, but as plain text. Useful as a starting point for custom monitoring scripts.

## APIs Used

- [`nvitop.Device.all()`][device-all]
- [`nvitop.Device.processes()`][device-processes]
- The various `Device.<metric>()` accessors (`fan_speed`, `temperature`, `gpu_utilization`, `memory_*`).

## Run

```bash
python3 examples/monitor-minimal/monitor_minimal.py
```

Requires only `nvitop` itself; no other dependencies.

See [`../README.md`](../README.md) for the full example index.

[device-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.Device.all
[device-processes]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.Device.processes
