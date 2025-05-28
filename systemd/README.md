# nvitop-exporter Systemd Service

This directory contains files to run nvitop-exporter as a Linux systemd service for automatic startup and monitoring.

## 📋 Prerequisites

- Linux system with systemd (Ubuntu 16.04+, CentOS 7+, etc.)
- NVIDIA GPU drivers installed
- Root/sudo access for installation
- Python 3.8+ (will be installed if missing)

## 🚀 Quick Installation

### Automated Installation
```bash
# Clone the repository
git clone https://github.com/ntheanh201/nvitop.git
cd nvitop/systemd

# Run the installation script
sudo ./install.sh
```

### Manual Installation

1. **Install dependencies:**
   ```bash
   # Install Python and required packages
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip python3-dev

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   ```

2. **Build and install nvitop-exporter:**
   ```bash
   # Install shiv for building binary
   pip install shiv

   # Build the binary
   cd /path/to/nvitop
   shiv -e nvitop_exporter.__main__:main -o nvitop-exporter --site-packages . nvitop prometheus-client
   chmod +x nvitop-exporter

   # Install binary
   sudo cp nvitop-exporter /usr/local/bin/nvitop-exporter
   ```

3. **Create service user:**
   ```bash
   sudo useradd --system --no-create-home --shell /bin/false nvitop-exporter
   sudo mkdir -p /var/lib/nvitop-exporter
   sudo chown nvitop-exporter:nvitop-exporter /var/lib/nvitop-exporter
   ```

4. **Install service:**
   ```bash
   sudo cp nvitop-exporter.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable nvitop-exporter
   sudo systemctl start nvitop-exporter
   ```

## 🔧 Configuration

### Service Configuration

The service is configured via the systemd unit file. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `--bind-address` | `0.0.0.0` | Address to bind to |
| `--port` | `8080` | Port to listen on |
| `--interval` | `10` | Metrics collection interval (seconds) |

### Customizing the Service

Edit `/etc/systemd/system/nvitop-exporter.service`:

```ini
[Service]
ExecStart=/usr/local/bin/nvitop-exporter --bind-address=0.0.0.0 --port=9090 --interval=5
Environment=NVITOP_EXPORTER_PORT=9090
```

After changes:
```bash
sudo systemctl daemon-reload
sudo systemctl restart nvitop-exporter
```

### Security Settings

The service includes several security hardening features:

- **Dedicated User**: Runs as `nvitop-exporter` user (not root)
- **Limited Filesystem Access**: Read-only system, private tmp
- **Resource Limits**: Memory limit of 256MB
- **No New Privileges**: Cannot escalate privileges

## 📊 Usage

### Service Management

```bash
# Check status
sudo systemctl status nvitop-exporter

# View logs
sudo journalctl -u nvitop-exporter -f

# Start/stop/restart
sudo systemctl start nvitop-exporter
sudo systemctl stop nvitop-exporter
sudo systemctl restart nvitop-exporter

# Enable/disable auto-start
sudo systemctl enable nvitop-exporter
sudo systemctl disable nvitop-exporter
```

### Accessing Metrics

```bash
# Test metrics endpoint
curl http://localhost:8080/metrics

# Check if service is responding
curl -I http://localhost:8080/metrics
```

### Integration with Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'nvitop-exporter'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 30s
    metrics_path: /metrics
```

## 🔧 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check detailed status
sudo systemctl status nvitop-exporter -l

# Check logs
sudo journalctl -u nvitop-exporter --no-pager

# Check if binary exists
ls -la /usr/local/bin/nvitop-exporter

# Test binary manually
sudo -u nvitop-exporter /usr/local/bin/nvitop-exporter --help
```

#### Permission Issues
```bash
# Check user exists
id nvitop-exporter

# Check data directory permissions
ls -la /var/lib/nvitop-exporter

# Fix permissions
sudo chown nvitop-exporter:nvitop-exporter /var/lib/nvitop-exporter
```

#### NVIDIA Driver Issues
```bash
# Check NVIDIA drivers
nvidia-smi

# Check if user can access GPU
sudo -u nvitop-exporter nvidia-smi
```

#### Port Already in Use
```bash
# Check what's using port 8080
sudo netstat -tlnp | grep 8080
sudo ss -tlnp | grep 8080

# Change port in service file
sudo systemctl edit nvitop-exporter
```

### Log Analysis

```bash
# Recent logs
sudo journalctl -u nvitop-exporter --since "1 hour ago"

# Follow logs in real-time
sudo journalctl -u nvitop-exporter -f

# Logs with specific priority
sudo journalctl -u nvitop-exporter -p err

# Export logs
sudo journalctl -u nvitop-exporter --since "1 day ago" > nvitop-logs.txt
```

## 🗑️ Uninstallation

### Automated Uninstallation
```bash
sudo ./uninstall.sh
```

### Manual Uninstallation
```bash
# Stop and disable service
sudo systemctl stop nvitop-exporter
sudo systemctl disable nvitop-exporter

# Remove service file
sudo rm /etc/systemd/system/nvitop-exporter.service
sudo systemctl daemon-reload

# Remove binary
sudo rm /usr/local/bin/nvitop-exporter

# Remove user and data (optional)
sudo userdel nvitop-exporter
sudo rm -rf /var/lib/nvitop-exporter

# Remove from uv tools
uv tool uninstall nvitop-exporter
```

## 📁 File Structure

```
systemd/
├── nvitop-exporter.service    # Systemd service unit file
├── install.sh                 # Automated installation script
├── uninstall.sh              # Automated uninstallation script
└── README.md                  # This documentation
```

## 🔒 Security Considerations

- Service runs as dedicated non-root user
- Limited filesystem access via systemd security features
- No network access beyond binding to specified port
- Resource limits prevent resource exhaustion
- Private temporary directory

## 📈 Monitoring the Service

### Health Checks

```bash
# Service health
systemctl is-active nvitop-exporter

# Metrics endpoint health
curl -f http://localhost:8080/metrics > /dev/null && echo "OK" || echo "FAIL"

# Resource usage
systemctl show nvitop-exporter --property=MemoryCurrent,CPUUsageNSec
```

### Performance Tuning

Adjust these settings in the service file based on your needs:

```ini
# Increase memory limit for large GPU clusters
MemoryMax=512M

# Adjust collection interval
ExecStart=/usr/local/bin/nvitop-exporter --interval=5

# Change log level
Environment=NVITOP_LOG_LEVEL=DEBUG
```

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review service logs: `sudo journalctl -u nvitop-exporter`
3. Test the binary manually: `nvitop-exporter --help`
4. Open an issue on GitHub with logs and system information
