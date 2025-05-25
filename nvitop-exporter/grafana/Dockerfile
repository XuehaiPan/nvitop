FROM ubuntu:latest AS dashboard

COPY dashboard.json /nvitop-dashboard.json

RUN sed -i -e 's|"uid"\s*:\s*"${DS_PROMETHEUS}"|"uid": "nvitop-prometheus"|g' \
    /nvitop-dashboard.json

FROM grafana/grafana:latest

COPY --from=dashboard /nvitop-dashboard.json /grafana/dashboards/nvitop-exporter/nvitop-dashboard.json
