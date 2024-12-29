# nvitop-exporter

Prometheus exporter built on top of `nvitop`.

## Quickstart

Start the exporter with the following command:

```bash
pipx run nvitop-exporter --bind-address 0.0.0.0 --port 5050
# or
uvx nvitop-exporter --bind-address 0.0.0.0 --port 5050
```

Then you can access the metrics at `http://localhost:5050/metrics`.

You will need to configure Prometheus to scrape the metrics from the exporter.

```yaml
scrape_configs:
  - job_name: 'nvitop-exporter'
    static_configs:
      - targets: ['localhost:5050']
```

## Grafana Dashboard

A Grafana dashboard is provided to visualize the metrics collected by the exporter.
The source of the dashboard is [`dashboard.json`](./dashboard.json).
The Grafana dashboard can also be imported as by ID [22589](https://grafana.com/grafana/dashboards/22589-nvitop-dashboard).

<p align="center">
  <img width="100%" src="https://github.com/user-attachments/assets/c1769a8b-2d06-47c4-8f76-c91dace132e9" alt="Filter">
  <br/>
  The Grafana dashboard for the exporter.
</p>
