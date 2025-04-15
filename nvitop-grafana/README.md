# nvitop-dashboard
Use to view GPU metrics in a dashboard, if you don't already have one.
Only run after building `nvitop-exporter`.
* Prometheus scrapes the exporter defined in prometheus-conf.yml
* Grafana shows the dashboard, by connecting to the Prometheus server.


## Get started

1. Start Grafana/Prometheus
   ```
   cd nvitop-grafana/
   docker compose up -d
   ```

2. Navigate to http://localhost:3000/dashboards -> New (upper right) -> Import

3. Copy code from `dashboard.json` and paste it into the appropriate field

4. Select `nvitop-prometheus` as data source -> Import

5. You should now be looking at a the beautiful nvitop dashboard

## Sanity checks
* Verify Prometheus has connection with the desired endpoint: http://localhost:9090/targets
* Verify Prometheus is performing scraping actions: run `count by (__name__)({__name__!=""})` in http://localhost:9090/query. The result should include the metrics shown in the endpoint being scraped.

## Network considerations
* Connections from outside a VPN can be tricky to set up in docker. Using `network_mode: "host"` in the compose and conf files can be a great place to start.
