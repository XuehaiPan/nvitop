# Programmatic CUDA Device Selection

Shows [`nvitop.select_devices`][select-devices] — the Python API behind the [`nvisel`][nvisel] CLI. Useful for scripts that need to pick GPUs based on free memory, utilization, or other criteria before launching a training job.

## APIs Used

- [`nvitop.select_devices`][select-devices] — supports `format='index' | 'uuid' | 'device'`, `min_count`, `min_free_memory`, `max_gpu_utilization`, and more.

## Run

```bash
python3 examples/select-devices-api/select_devices_api.py
```

Requires only `nvitop` itself; no other dependencies.

See [`../README.md`](../README.md) for the full example index.

[nvisel]: https://nvitop.readthedocs.io/en/latest/select.html
[select-devices]: https://nvitop.readthedocs.io/en/latest/api/utils.html#nvitop.select_devices
