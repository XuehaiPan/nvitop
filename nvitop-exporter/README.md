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
