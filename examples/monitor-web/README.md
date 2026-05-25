# Web Monitor (HTTP(S) Dashboard)

`monitor_web.py` serves a small browser dashboard for [`nvitop.collect_in_background`][cib]. The Python side uses the standard-library `http.server` stack, stores collector samples in a rotating in-memory buffer, and exposes the same data through JSON endpoints. The browser side loads [Plotly] from a CDN for the time-series charts.

## APIs Used

- [`nvitop.collect_in_background`][cib]
- [`nvitop.ResourceMetricCollector`][collector]
- [`nvitop.Device.all()`][device-all]
- [`nvitop.bytes2human()`][bytes2human]
- [`nvitop.colored()`][colored] for the startup banner

## What It Shows

- Host CPU, host memory, swap, and buffer status badges.
- A host history chart for CPU percent and host memory percent, labeled with memory usage.
- One card per GPU, using the raw NVIDIA/NVML GPU index.
- Per-GPU current bars for GPU utilization, memory bandwidth, GPU memory, and power.
- One history chart per GPU under the cards, plotting the same four metrics.
- History range buttons for `1m`, `5m`, `15m`, `30m`, `1h`, `3h`, `6h`, `12h`, and `24h`.

Cards and plot legends read each metric's `…/last` keyed value from the collector snapshot (`/last` is a key suffix produced by `nvitop.ResourceMetricCollector`, not an HTTP route). The full JSON payload still includes aggregate variants such as `…/mean`, `…/min`, `…/max`, and `…/last`.

Process snapshots are disabled with `root_pids={}` so the dashboard tracks host and device metrics without collecting per-process GPU rows.

## Screenshot

![nvitop web dashboard](https://github.com/user-attachments/assets/b07abc8a-d0f0-4d0f-a7a2-09514cd28832)

## Run

```bash
python3 examples/monitor-web/monitor_web.py --port 5555
```

Open <http://127.0.0.1:5555/> in a browser.

The backend collector samples every `--interval` seconds, defaulting to `1.0`. The frontend polls `/metrics.json` every second and marks the dashboard stale if the latest sample is too old.

The startup banner is printed to `stderr`:

```text
INFO: Found N device(s).
INFO: GPU 0: <name> (UUID: GPU-...)
INFO: Retention 1d at 1s interval (max 86400 samples).
INFO: Serving the dashboard at http://127.0.0.1:5555/
INFO:   - JSON snapshot:       http://127.0.0.1:5555/metrics.json
INFO:   - JSON history:        http://127.0.0.1:5555/history.json
```

## JSON Endpoints

`/metrics.json` returns the latest sample plus metadata:

- `interval`: collector interval in seconds.
- `hostname`: server hostname displayed in the browser header and tab title.
- `server_time`: current server timestamp.
- `sample_time`: timestamp for the latest collected sample.
- `stale_seconds`: age of the latest sample, or `null` if no sample has been collected yet.
- `status`: lifecycle marker — `warming_up`, `ready`, `stalled`, or `failed`.
- `collector_error`: most recent collector failure message (cleared on the next successful sample), or `null` when healthy.
- `buffer`: object with `count`, `max_count`, `retention_seconds`, `retention_human`, `oldest_epoch`, and `newest_epoch`.
- `devices`: list of objects with `index`, `name`, `memory_total_mib`, `memory_total_human`, and `uuid`.
- `metrics`: raw collector metric keys and numeric values.
- `metrics_human`: human-readable memory values for finite MiB/GiB metrics.

Inspect it from the shell:

```bash
curl -s http://127.0.0.1:5555/metrics.json | python3 -m json.tool | head -60
```

`/history.json` returns buffered samples:

```bash
curl -s 'http://127.0.0.1:5555/history.json?limit=10' | python3 -m json.tool
curl -s 'http://127.0.0.1:5555/history.json?since=1779270000' | python3 -m json.tool
```

Supported query parameters:

- `bucket_seconds=N`: average samples into epoch-aligned `N`-second buckets.
- `limit=N`: return only the most recent `N` samples.
- `max_samples=N`: return at most `N` samples after filtering and bucket averaging.
- `since=EPOCH`: return samples strictly newer than the Unix timestamp `EPOCH`.

Unrecognized parameters are ignored. Parameters that are present but unparsable or out of range (non-numeric, `NaN`/`Infinity`, zero or negative for the positive-only parameters) return `400 Bad Request` with a message identifying the offending parameter, so typos surface immediately rather than silently returning the unfiltered history.

JSON responses are strict JSON. Non-finite collector values such as `NaN` and `Infinity` are serialized as `null`.

## History And Retention

Use `--retention` to size the rotating buffer. The flag accepts `s`, `m`/`min`, `h`, and `d` suffixes; a bare number is treated as seconds.

```bash
python3 examples/monitor-web/monitor_web.py --retention 12h
python3 examples/monitor-web/monitor_web.py --retention 30min --interval 5
python3 examples/monitor-web/monitor_web.py --retention 600
```

The buffer holds at most `max(1, int(retention / interval))` samples. Each sample stores the collector's metric dictionary, including aggregate keys such as `mean`, `min`, `max`, and `last`, so memory use depends on the sample count, the exported metric-key count, and normal Python object overhead. Increase `--interval` to keep the same retention window with fewer stored samples.

## TLS And Mutual TLS

Serve plain HTTP by default, or pass a certificate and key to enable HTTPS:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -subj '/CN=localhost' \
    -keyout key.pem -out cert.pem

python3 examples/monitor-web/monitor_web.py --port 5555 \
    --certfile cert.pem --keyfile key.pem
```

To require client certificates, also provide a trusted client CA bundle or CA directory:

```bash
python3 examples/monitor-web/monitor_web.py --port 5555 \
    --certfile cert.pem --keyfile key.pem \
    --client-cafile ca.pem --client-auth-required
```

`--client-cafile` (or `--client-capath`) and `--client-auth-required` are a single mutual-TLS bundle — they must be passed together. Passing only some of the three flags is rejected at startup.

## Useful Flags

- `--bind-address ADDRESS`, `--bind ADDRESS`, `-B ADDRESS`: bind address, default `127.0.0.1`.
- `--hostname HOSTNAME`, `--host HOSTNAME`, `-H HOSTNAME`: hostname to display in the dashboard.
- `--port PORT`, `-p PORT`: listen port, default `5555`.
- `--interval SEC`: collector interval in seconds, minimum `0.25`, default `1.0`.
- `--retention DURATION`: history retention, default `1d`.
- `--certfile PATH` and `--keyfile PATH`: enable HTTPS.
- `--client-cafile PATH` or `--client-capath PATH`: trusted client CAs for mutual TLS.
- `--client-auth-required`: require a valid client certificate.

See [`../README.md`](../README.md) for the full example index.

[Plotly]: https://plotly.com/javascript/
[bytes2human]: https://nvitop.readthedocs.io/en/latest/api/utils.html#nvitop.bytes2human
[cib]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.collect_in_background
[collector]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.ResourceMetricCollector
[colored]: https://nvitop.readthedocs.io/en/latest/api/utils.html#nvitop.colored
[device-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.Device.all
