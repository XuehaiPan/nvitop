# nvitop-exporter

Prometheus exporter built on top of `nvitop`.

## Quickstart

Start the exporter with the following command:

```bash
pipx run nvitop-exporter --bind-address 0.0.0.0 --port 5050
# or
uvx nvitop-exporter --bind-address 0.0.0.0 --port 5050
```

Then you can access the metrics at [`http://localhost:5050/metrics`](http://localhost:5050/metrics).

You will need to configure Prometheus to scrape the metrics from the exporter.

```yaml
scrape_configs:
  - job_name: 'nvitop-exporter'
    static_configs:
      - targets: ['localhost:5050']
```

### With Docker
After building `nvitop:latest`
```bash
cd nvitop-exporter/
docker buildx build -t nvitop-exporter:latest .
docker run -it --name nvitop-exporter --rm --runtime=nvidia --gpus=all --pid=host -p 5050:5050 nvitop-exporter:latest
```

If you need the exporter to report the local IP, you can replace `-p 5050:5050` with `--network=host`. This gives the container access to the host's network which have security implications, especially in multi-tenant environments.

## Grafana Dashboard

A Grafana dashboard is provided to visualize the metrics collected by the exporter.
The source of the dashboard is [`dashboard.json`](./dashboard.json).
The Grafana dashboard can also be imported as by ID [22589](https://grafana.com/grafana/dashboards/22589-nvitop-dashboard).

<p align="center">
  <img width="100%" src="https://github.com/user-attachments/assets/e4867e64-2ca9-45bc-b524-929053f9673d" alt="Grafana Dashboard">
  <br/>
  The Grafana dashboard for the exporter.
</p>
