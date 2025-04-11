# nvitop-dashboard
Use to view GPU metrics in a dashboard, if you don't already have one.
Only run after building `nvitop-exporter`.
* Prometheus scrapes the exporter defined in prometheus-conf.yml
* Grafana shows the dashboard, by connecting to the Prometheus server.


## Get started

1. Start Grafana/Prometheus
   ```
   docker compose up -d
   ```

2. Navigate to http://localhost:3000/dashboards

3. Import new dashboard or select the provisioned one


## Sanity checks
* Verify Prometheus has connection with the desired endpoint: http://localhost:9090/targets
* Verify Prometheus is performing scraping actions: run `count by (__name__)({__name__!=""})` in http://localhost:9090/query. The result should include the metrics shown in the endpoint being scraped.

## Network considerations
* Connections from outside a VPN can be tricky to set up in docker. Using `network_mode: "host"` in the compose and conf files can be a great place to start.
