# nvitop-exporter

Prometheus exporter built on top of `nvitop`.

## Quickstart

Start the exporter with the following command:

```bash
uvx nvitop-exporter --bind-address 0.0.0.0 --port 5050
# or
pipx run nvitop-exporter --bind-address 0.0.0.0 --port 5050
```

Then you can access the metrics at [`http://localhost:5050/metrics`](http://localhost:5050/metrics).

You will need to configure Prometheus to scrape the metrics from the exporter.

```yaml
scrape_configs:
  - job_name: 'nvitop-exporter'
    static_configs:
      - targets: ['localhost:5050']
```

## TLS / mTLS

The exporter can serve metrics over HTTPS, optionally requiring client certificate authentication (mTLS). TLS support is provided by `prometheus_client` (>= 0.19.0) and configured entirely through CLI flags — no config file is involved.

### Plain HTTPS

Provide a server certificate and private key:

```bash
nvitop-exporter --bind-address 0.0.0.0 --port 5050 \
    --certfile /path/to/server.crt \
    --keyfile /path/to/server.key
```

The metrics endpoint is then served at [`https://localhost:5050/metrics`](https://localhost:5050/metrics). Update the Prometheus scrape config to use the `https` scheme, and point it at the CA that signed your server certificate:

```yaml
scrape_configs:
  - job_name: 'nvitop-exporter'
    scheme: https
    static_configs:
      - targets: ['localhost:5050']
    tls_config:
      ca_file: /path/to/server-ca.crt
```

### Mutual TLS (mTLS)

To require scrapers to present a valid client certificate, pass a CA bundle (`--client-cafile`) or CA directory (`--client-capath`) **and** `--client-auth-required`:

```bash
nvitop-exporter --bind-address 0.0.0.0 --port 5050 \
    --certfile /path/to/server.crt \
    --keyfile /path/to/server.key \
    --client-cafile /path/to/clients-ca.crt \
    --client-auth-required
```

`--client-cafile` / `--client-capath` and `--client-auth-required` must be specified together. Passing a CA without `--client-auth-required` is rejected by the CLI to avoid the silent "trust but don't verify" configuration that the underlying `prometheus_client` API would otherwise allow.

Configure Prometheus to present its client certificate when scraping:

```yaml
scrape_configs:
  - job_name: 'nvitop-exporter'
    scheme: https
    static_configs:
      - targets: ['localhost:5050']
    tls_config:
      ca_file: /path/to/server-ca.crt
      cert_file: /path/to/prometheus-client.crt
      key_file: /path/to/prometheus-client.key
```

### Authentication beyond mTLS

The exporter does not implement HTTP basic auth, OAuth, or IP allowlisting. Following the standard Prometheus exporter pattern, run the exporter behind a reverse proxy (`NGINX`, `Traefik`, `Caddy`, ...) if any of those are required.

## Grafana Dashboard

A Grafana dashboard is provided to visualize the metrics collected by the exporter.
The source of the dashboard is [`dashboard.json`](grafana/dashboard.json).
The Grafana dashboard can also be imported as by ID [22589](https://grafana.com/grafana/dashboards/22589-nvitop-dashboard).

If you are using [`docker-compose`](https://docs.docker.com/compose), you can start a dashboard at [`http://localhost:3000`](http://localhost:3000) with the following command:

```bash
cd nvitop-exporter/grafana
docker compose up --build --detach
```

<p align="center">
  <img width="100%" src="https://github.com/user-attachments/assets/e4867e64-2ca9-45bc-b524-929053f9673d" alt="Grafana Dashboard">
  <br/>
  The Grafana dashboard for the exporter.
</p>
