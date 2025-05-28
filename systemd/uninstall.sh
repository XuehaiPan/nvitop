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
INSTALL_DIR="/usr/local/bin"
SERVICE_DIR="/etc/systemd/system"
DATA_DIR="/var/lib/nvitop-exporter"

echo -e "${BLUE}🗑️  Uninstalling nvitop-exporter systemd service${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root${NC}"
   exit 1
fi

# Stop and disable service
echo -e "${BLUE}⏹️  Stopping and disabling service...${NC}"
if systemctl is-active --quiet "$SERVICE_NAME"; then
    systemctl stop "$SERVICE_NAME"
    echo -e "${GREEN}✅ Service stopped${NC}"
fi

if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    systemctl disable "$SERVICE_NAME"
    echo -e "${GREEN}✅ Service disabled${NC}"
fi

# Remove service file
echo -e "${BLUE}📋 Removing systemd service file...${NC}"
if [[ -f "$SERVICE_DIR/$SERVICE_NAME.service" ]]; then
    rm -f "$SERVICE_DIR/$SERVICE_NAME.service"
    echo -e "${GREEN}✅ Service file removed${NC}"
fi

# Reload systemd
echo -e "${BLUE}🔄 Reloading systemd...${NC}"
systemctl daemon-reload
systemctl reset-failed

# Remove binary symlink
echo -e "${BLUE}🔗 Removing binary symlink...${NC}"
if [[ -L "$INSTALL_DIR/nvitop-exporter" ]]; then
    rm -f "$INSTALL_DIR/nvitop-exporter"
    echo -e "${GREEN}✅ Binary symlink removed${NC}"
fi

# Ask about removing user and data
echo -e "${YELLOW}❓ Do you want to remove the service user and data directory? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    # Remove service user
    if id "$SERVICE_USER" &>/dev/null; then
        userdel "$SERVICE_USER" 2>/dev/null || true
        echo -e "${GREEN}✅ Service user removed${NC}"
    fi
    
    # Remove data directory
    if [[ -d "$DATA_DIR" ]]; then
        rm -rf "$DATA_DIR"
        echo -e "${GREEN}✅ Data directory removed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Service user and data directory preserved${NC}"
fi

# Ask about removing uv tool installation
echo -e "${YELLOW}❓ Do you want to remove nvitop-exporter from uv tools? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    if command -v uv &> /dev/null; then
        uv tool uninstall nvitop-exporter 2>/dev/null || true
        echo -e "${GREEN}✅ nvitop-exporter removed from uv tools${NC}"
    fi
fi

echo -e "\n${GREEN}🎉 Uninstallation completed successfully!${NC}"
