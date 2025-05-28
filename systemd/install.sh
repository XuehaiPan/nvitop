#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="nvitop-exporter"
SERVICE_USER="nvitop-exporter"
SERVICE_GROUP="nvitop-exporter"
INSTALL_DIR="/usr/local/bin"
SERVICE_DIR="/etc/systemd/system"
DATA_DIR="/var/lib/nvitop-exporter"

echo -e "${BLUE}🚀 Installing nvitop-exporter as systemd service${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ This script must be run as root${NC}"
    exit 1
fi

# Check if NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed.${NC}"
fi

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
apt-get update
apt-get install -y python3 python3-pip python3-dev

# Install uv
echo -e "${BLUE}📦 Installing uv package manager...${NC}"
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Build nvitop-exporter binary
echo -e "${BLUE}📦 Building nvitop-exporter binary...${NC}"
cd "$(dirname "$0")/.."
pip install shiv
shiv -e nvitop_exporter.__main__:main -o nvitop-exporter --site-packages . nvitop prometheus-client
chmod +x nvitop-exporter

# Create service user and group
echo -e "${BLUE}👤 Creating service user and group...${NC}"
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --no-create-home --shell /bin/false "$SERVICE_USER"
    echo -e "${GREEN}✅ Created user: $SERVICE_USER${NC}"
else
    echo -e "${YELLOW}⚠️  User $SERVICE_USER already exists${NC}"
fi

# Create data directory
echo -e "${BLUE}📁 Creating data directory...${NC}"
mkdir -p "$DATA_DIR"
chown "$SERVICE_USER:$SERVICE_GROUP" "$DATA_DIR"
chmod 755 "$DATA_DIR"

# Install binary
echo -e "${BLUE}📦 Installing binary...${NC}"
cp nvitop-exporter "$INSTALL_DIR/nvitop-exporter"
chown "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR/nvitop-exporter"
chmod 755 "$INSTALL_DIR/nvitop-exporter"

# Copy service file
echo -e "${BLUE}📋 Installing systemd service...${NC}"
cp "$(dirname "$0")/nvitop-exporter.service" "$SERVICE_DIR/"
chmod 644 "$SERVICE_DIR/nvitop-exporter.service"

# Reload systemd and enable service
echo -e "${BLUE}🔄 Reloading systemd and enabling service...${NC}"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# Start the service
echo -e "${BLUE}▶️  Starting nvitop-exporter service...${NC}"
systemctl start "$SERVICE_NAME"

# Check service status
sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${GREEN}✅ nvitop-exporter service is running successfully!${NC}"
    echo -e "${GREEN}📊 Metrics available at: http://localhost:8080/metrics${NC}"
else
    echo -e "${RED}❌ Service failed to start. Check logs with: journalctl -u $SERVICE_NAME${NC}"
    exit 1
fi

# Display useful commands
echo -e "\n${BLUE}📋 Useful commands:${NC}"
echo -e "  Status:    ${YELLOW}systemctl status $SERVICE_NAME${NC}"
echo -e "  Logs:      ${YELLOW}journalctl -u $SERVICE_NAME -f${NC}"
echo -e "  Stop:      ${YELLOW}systemctl stop $SERVICE_NAME${NC}"
echo -e "  Start:     ${YELLOW}systemctl start $SERVICE_NAME${NC}"
echo -e "  Restart:   ${YELLOW}systemctl restart $SERVICE_NAME${NC}"
echo -e "  Disable:   ${YELLOW}systemctl disable $SERVICE_NAME${NC}"
echo -e "  Metrics:   ${YELLOW}curl http://localhost:8080/metrics${NC}"

echo -e "\n${GREEN}🎉 Installation completed successfully!${NC}" 