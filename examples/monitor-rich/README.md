# Rich Live GPU Monitor

A standalone terminal dashboard built with the public `nvitop` API and [Rich Live]. It shows host CPU, memory, swap, load averages, physical GPUs, MIG instances, and GPU processes. Tables adapt to the terminal size.

## APIs Used

- [`nvitop.Device.all`][device-all] and [`nvitop.Device.mig_devices`][mig-devices]
- [`nvitop.Device.as_snapshot`][device-snapshot] and [`nvitop.GpuProcess.take_snapshots`][take-snapshots]
- [`nvitop.host`][host] for host metrics

## Screenshot

![Rich Live dashboard](https://github.com/user-attachments/assets/2dabdf79-a821-4a1a-91d0-a7fb6dedffba)

## Install + Run

```bash
python3 -m pip install -r examples/monitor-rich/requirements.txt
python3 examples/monitor-rich/monitor_rich.py
```

Or run it without cloning the repository, using [uv](https://docs.astral.sh/uv) (which installs the dependencies automatically):

```bash
uv run https://github.com/XuehaiPan/nvitop/raw/HEAD/examples/monitor-rich/monitor_rich.py
```

The dashboard refreshes once per second. Press `Ctrl+C` to exit. Change the interval or print one snapshot with:

```bash
python3 examples/monitor-rich/monitor_rich.py --interval 2
python3 examples/monitor-rich/monitor_rich.py --once
```

Redirected output automatically prints one snapshot and exits. Unsupported metrics display as `N/A`. Devices are discovered once at startup; restart the monitor after changing the GPU or MIG configuration.

See [`../README.md`](../README.md) for the full example index.

[Rich Live]: https://rich.readthedocs.io/en/latest/live.html
[device-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.Device.all
[device-snapshot]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.Device.as_snapshot
[host]: https://nvitop.readthedocs.io/en/latest/api/host.html
[mig-devices]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.Device.mig_devices
[take-snapshots]: https://nvitop.readthedocs.io/en/latest/api/process.html#nvitop.GpuProcess.take_snapshots
