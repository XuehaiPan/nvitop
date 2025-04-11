# nvitop-dashboard
Only run after starting `nvitop-exporter`. Use to view GPU metrics in a dashboard, if you don't already have one.
* Prometheus scrapes the exporter defined in prometheus-conf.yml
* Grafana shows the dashboard, by connecting to the Prometheus server.


## Get started
0. [Start nvitop-exporter](../nvitop-exporter/README.md)
1. Start Grafana/Prometheus
   ```
   docker compose up -d
   ```

2. Navigate to http://localhost:3000/dashboards and select the provisioned dashboard

3. That's it.


## Sanity checks
To verify that Prometheus 
* Verify Prometheus has connection with the desired endpoint: http://localhost:9090/targets
* Verify Prometheus is performing scraping actions: run `count by (__name__)({__name__!=""})` in http://localhost:9090/query. The result should include the metrics shown in the endpoint being scraped.

## Network considerations
* Connections from outside a VPN can be tricky to set up in docker. Using `network_mode: "host"` in the compose and conf files can be a great place to start.
