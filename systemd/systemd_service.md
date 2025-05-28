# nvitop.service

sudo cp nvitop-exporter /usr/local/bin/nvitop-exporter

sudo useradd --system --no-create-home --shell /bin/false nvitop-exporter
sudo mkdir -p /var/lib/nvitop-exporter
sudo chown nvitop-exporter:nvitop-exporter /var/lib/nvitop-exporter

sudo cp nvitop-exporter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nvitop-exporter
sudo systemctl start nvitop-exporter
