# Web Monitor (stdlib HTTP(S))

A minimal browser dashboard for [`nvitop.collect_in_background`][cib]: the collector ticks on a daemon thread, samples are pushed into a rotating ring buffer (24h by default), and a tiny `http.server`-based router serves both a one-page HTML dashboard and JSON snapshots at `/metrics.json` and `/history.json`. Stdlib only — no Flask, no TensorBoard, no extra dependencies. Supports HTTPS and mutual TLS via the same flag names as [`nvitop-exporter`][exporter].

## APIs Used

- [`nvitop.collect_in_background`][cib]
- [`nvitop.ResourceMetricCollector`][collector]
- [`nvitop.Device.cuda.all()`][cuda-all]
- [`nvitop.colored`][colored] (for the startup banner)

## Run

```bash
python3 examples/monitor-web/monitor_web.py --port 5555
```

The startup banner (printed to `stderr`, mirroring [`nvitop-exporter`][exporter]) reports the device count, per-GPU UUIDs, the retention/interval summary, and the three URLs:

```text
INFO: Found 1 device(s).
INFO: GPU 0: NVIDIA RTX 6000 Ada Generation (UUID: GPU-...)
INFO: Retention 1d at 1.0s interval (max 86400 samples).
INFO: Serving the dashboard at http://127.0.0.1:5555/
INFO:   - JSON snapshot:         http://127.0.0.1:5555/metrics.json
INFO:   - JSON history:          http://127.0.0.1:5555/history.json
```

The browser dashboard polls `/metrics.json` every `--interval` seconds and renders a card per visible GPU (utilization, memory, temperature, fan, power) plus a host footer.

Inspect the raw JSON from the CLI:

```bash
curl -s http://127.0.0.1:5555/metrics.json | python3 -m json.tool | head -30
curl -s 'http://127.0.0.1:5555/history.json?limit=10' | python3 -m json.tool | head -30
```

`/history.json` accepts two optional query parameters:

- `?limit=N` — return only the most recent `N` samples.
- `?since=EPOCH` — return samples strictly newer than the Unix timestamp `EPOCH`.

## Retention

Use `--retention` to size the rotating buffer. The flag accepts `s`/`m`/`h`/`d` suffixes; a bare number is treated as seconds.

```bash
python3 examples/monitor-web/monitor_web.py --retention 12h
python3 examples/monitor-web/monitor_web.py --retention 30min --interval 5
python3 examples/monitor-web/monitor_web.py --retention 600    # 600 seconds
```

The buffer holds at most `int(retention / interval)` samples, so memory scales as `samples × keys_per_sample × 8 B`. Bump `--interval` (e.g. `--interval 5`) to keep the same retention at lower memory cost.

## HTTPS / mTLS

Generate a throw-away self-signed certificate for local testing:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -subj '/CN=localhost' \
    -keyout key.pem -out cert.pem
```

Then serve over HTTPS:

```bash
python3 examples/monitor-web/monitor_web.py --port 5555 \
    --certfile cert.pem --keyfile key.pem
```

For mutual TLS (require the client to present a trusted certificate):

```bash
python3 examples/monitor-web/monitor_web.py --port 5555 \
    --certfile cert.pem --keyfile key.pem \
    --client-cafile ca.pem --client-auth-required
```

The TLS / mTLS flag names match [`nvitop-exporter`][exporter] so the same cert/key combo works for both tools.

See [`../README.md`](../README.md) for the full example index.

[cib]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.collect_in_background
[collector]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.ResourceMetricCollector
[colored]: https://nvitop.readthedocs.io/en/latest/api/utils.html#nvitop.colored
[cuda-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.CudaDevice.all
[exporter]: https://github.com/XuehaiPan/nvitop/tree/main/nvitop-exporter
