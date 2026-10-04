#!/usr/bin/env bash
# ==============================================================================
# build_deb.sh - Build an all-in-one standalone .deb package for nvitop
# Compatible with: Ubuntu (22.04+), Debian (12+), Kali Linux, and all Debian-based distros
# ==============================================================================

set -euo pipefail

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo -e "${CYAN}====================================================${NC}"
echo -e "${CYAN}    nvitop Debian/Ubuntu/Kali Package Builder       ${NC}"
echo -e "${CYAN}====================================================${NC}"

# Parse command line flags
FORCE_NATIVE=false
for arg in "$@"; do
    case "$arg" in
        --native)
            FORCE_NATIVE=true
            shift
            ;;
    esac
done

# Detect version from nvitop/version.py
VERSION=$(python3 -c "
import ast
with open('nvitop/version.py') as f:
    for line in f:
        if line.startswith('__version__'):
            print(ast.literal_eval(line.split('=')[1].strip()))
            break
" 2>/dev/null || echo "1.7.1")

PKG_REVISION="1"
ARCH=$(dpkg --print-architecture 2>/dev/null || uname -m)
case "${ARCH}" in
    x86_64) ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
esac

DEB_NAME="nvitop_${VERSION}-${PKG_REVISION}_${ARCH}.deb"
OUTPUT_DIR="${SCRIPT_DIR}/dist"
BUILD_DIR="${SCRIPT_DIR}/build"
PKG_DIR="${BUILD_DIR}/deb_pkg"

echo -e "${BLUE}Target Package:${NC} ${DEB_NAME}"
echo -e "${BLUE}Version:${NC}        ${VERSION}"
echo -e "${BLUE}Architecture:${NC}   ${ARCH}"

# 1. Clean previous build artifacts
echo -e "\n${YELLOW}[1/6] Cleaning build directories...${NC}"
rm -rf "${PKG_DIR}" "${BUILD_DIR}/pyinstaller_work" "${BUILD_DIR}/pyinstaller_dist" "${BUILD_DIR}/entrypoint.py"
mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

# 2. Create entrypoint dispatcher for nvitop and nvisel
echo -e "\n${YELLOW}[2/6] Generating compilation entrypoint...${NC}"
cat << 'EOF' > "${BUILD_DIR}/entrypoint.py"
import os
import sys

def main():
    basename = os.path.basename(sys.argv[0])
    if basename.startswith('nvisel'):
        from nvitop.select import main as sel_main
        sys.exit(sel_main())
    else:
        from nvitop.cli import main as cli_main
        sys.exit(cli_main())

if __name__ == '__main__':
    main()
EOF

# 3. Compile standalone binary bundle
echo -e "\n${YELLOW}[3/6] Compiling nvitop into standalone bundle...${NC}"
USE_DOCKER=false
if [ "${FORCE_NATIVE}" = false ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    USE_DOCKER=true
fi

if [ "${USE_DOCKER}" = true ]; then
    echo -e "${CYAN}Building in Ubuntu 22.04 container for universal GLIBC compatibility (Ubuntu 22.04+, Debian 12+, Kali)...${NC}"
    HOST_UID=$(id -u)
    HOST_GID=$(id -g)
    docker run --rm \
        -v "${SCRIPT_DIR}:/workspace" \
        ubuntu:22.04 \
        bash -c "
            set -euo pipefail
            apt-get update -qq
            apt-get install -y -qq python3 python3-pip python3-venv > /dev/null
            python3 -m venv /tmp/venv
            /tmp/venv/bin/pip install --upgrade pip setuptools wheel > /dev/null
            /tmp/venv/bin/pip install /workspace pyinstaller pillow > /dev/null
            /tmp/venv/bin/pyinstaller \
                --name nvitop \
                --clean \
                --noconfirm \
                --collect-all nvitop \
                --collect-all pynvml \
                --hidden-import curses \
                --hidden-import psutil \
                --hidden-import pynvml \
                --distpath /workspace/build/pyinstaller_dist \
                --workpath /tmp/work \
                /workspace/build/entrypoint.py > /dev/null
            ln -sf nvitop /workspace/build/pyinstaller_dist/nvitop/nvisel
            chown -R ${HOST_UID}:${HOST_GID} /workspace/build /workspace/dist 2>/dev/null || true
        "
else
    echo -e "${CYAN}Building natively on host...${NC}"
    VENV_DIR="${BUILD_DIR}/build_venv"
    if [ ! -f "${VENV_DIR}/bin/pyinstaller" ]; then
        python3 -m venv "${VENV_DIR}"
        "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
        "${VENV_DIR}/bin/pip" install . pyinstaller pillow
    else
        "${VENV_DIR}/bin/pip" install -q --no-deps .
    fi
    "${VENV_DIR}/bin/pyinstaller" \
        --name nvitop \
        --clean \
        --noconfirm \
        --collect-all nvitop \
        --collect-all pynvml \
        --hidden-import curses \
        --hidden-import psutil \
        --hidden-import pynvml \
        --distpath "${BUILD_DIR}/pyinstaller_dist" \
        --workpath "${BUILD_DIR}/pyinstaller_work" \
        "${BUILD_DIR}/entrypoint.py" > /dev/null
fi

# Symlink nvisel inside the bundle
[ -e "${BUILD_DIR}/pyinstaller_dist/nvitop/nvisel" ] || ln -sf nvitop "${BUILD_DIR}/pyinstaller_dist/nvitop/nvisel"

# 4. Generate high-resolution desktop icons (SVG & multi-size PNGs)
echo -e "\n${YELLOW}[4/6] Generating desktop icons and assets...${NC}"
ICON_DIR="${BUILD_DIR}/icons"
mkdir -p "${ICON_DIR}"

cat << 'EOF' > "${ICON_DIR}/nvitop.svg"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#141818" />
      <stop offset="50%" stop-color="#0f1313" />
      <stop offset="100%" stop-color="#080a0a" />
    </linearGradient>
    <linearGradient id="nvGreen" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#8ce600" />
      <stop offset="100%" stop-color="#629e00" />
    </linearGradient>
    <linearGradient id="borderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#76b900" />
      <stop offset="40%" stop-color="#3d5e03" />
      <stop offset="100%" stop-color="#1b2e00" />
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Base Badge -->
  <rect x="24" y="24" width="464" height="464" rx="104" fill="url(#bgGrad)" stroke="url(#borderGrad)" stroke-width="12" />

  <!-- Terminal Grid Accents -->
  <path d="M70 160 H442 M70 240 H442 M70 320 H442" stroke="#253818" stroke-width="2" stroke-dasharray="6,6" opacity="0.6" />
  <path d="M160 70 V442 M256 70 V442 M352 70 V442" stroke="#253818" stroke-width="2" stroke-dasharray="6,6" opacity="0.6" />

  <!-- NVIDIA Eye Curve -->
  <path d="M140 256 C140 176 200 136 280 136 C344 136 384 176 384 176 C384 176 352 216 280 216 C230 216 204 236 204 256 C204 276 230 296 280 296 C352 296 384 336 384 336 C384 336 344 376 280 376 C200 376 140 336 140 256 Z" 
        fill="none" stroke="url(#nvGreen)" stroke-width="24" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)" />

  <!-- Center Core -->
  <circle cx="280" cy="256" r="32" fill="url(#nvGreen)" filter="url(#glow)" />

  <!-- GPU Load Pulse Wave -->
  <path d="M80 400 L180 400 L210 350 L240 430 L270 380 L300 410 L330 360 L360 400 L432 400" 
        fill="none" stroke="#76b900" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" opacity="0.9" />

  <!-- Activity Bars -->
  <rect x="90" y="80" width="36" height="12" rx="6" fill="#76b900" />
  <rect x="136" y="80" width="60" height="12" rx="6" fill="#76b900" />
  <rect x="206" y="80" width="40" height="12" rx="6" fill="#76b900" />
  <rect x="256" y="80" width="80" height="12" rx="6" fill="#3a5c00" />
  <rect x="346" y="80" width="50" height="12" rx="6" fill="#253b00" />
</svg>
EOF

# Render PNG sizes
python3 - << EOF
from PIL import Image, ImageDraw

def render_icon(size):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size / 256.0
    
    margin = 8 * s
    r = 48 * s
    draw.rounded_rectangle([margin, margin, size - margin, size - margin], radius=r, fill=(20, 24, 24, 255), outline=(118, 185, 0, 255), width=int(max(2, 6 * s)))
    
    green = (118, 185, 0, 255)
    light_green = (140, 230, 0, 255)
    dark_green = (60, 100, 0, 255)
    
    bar_y = 35 * s
    bar_h = 10 * s
    draw.rounded_rectangle([35*s, bar_y, 70*s, bar_y + bar_h], radius=4*s, fill=green)
    draw.rounded_rectangle([80*s, bar_y, 130*s, bar_y + bar_h], radius=4*s, fill=green)
    draw.rounded_rectangle([140*s, bar_y, 175*s, bar_y + bar_h], radius=4*s, fill=green)
    draw.rounded_rectangle([185*s, bar_y, 220*s, bar_y + bar_h], radius=4*s, fill=dark_green)
    
    bbox = [60 * s, 75 * s, 210 * s, 185 * s]
    draw.arc(bbox, start=200, end=350, fill=green, width=int(max(2, 12 * s)))
    bbox_in = [90 * s, 95 * s, 190 * s, 165 * s]
    draw.arc(bbox_in, start=20, end=170, fill=green, width=int(max(2, 8 * s)))
    
    cx, cy = 145 * s, 130 * s
    cr = 18 * s
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=light_green)
    
    pts = [
        (35 * s, 205 * s),
        (85 * s, 205 * s),
        (105 * s, 180 * s),
        (125 * s, 225 * s),
        (145 * s, 195 * s),
        (165 * s, 210 * s),
        (185 * s, 185 * s),
        (205 * s, 205 * s),
        (225 * s, 205 * s),
    ]
    draw.line(pts, fill=green, width=int(max(2, 6 * s)), joint='round')
    return img

for sz in [16, 32, 48, 64, 128, 256]:
    render_icon(sz).save(f"${ICON_DIR}/nvitop-{sz}.png")
EOF

# 5. Assemble Debian Package Tree
echo -e "\n${YELLOW}[5/6] Assembling Debian package hierarchy...${NC}"
mkdir -p "${PKG_DIR}/DEBIAN"
mkdir -p "${PKG_DIR}/usr/bin"
mkdir -p "${PKG_DIR}/usr/lib/nvitop"
mkdir -p "${PKG_DIR}/usr/share/applications"
mkdir -p "${PKG_DIR}/usr/share/pixmaps"
mkdir -p "${PKG_DIR}/usr/share/doc/nvitop"

# Copy compiled files
cp -r "${BUILD_DIR}/pyinstaller_dist/nvitop/." "${PKG_DIR}/usr/lib/nvitop/"

# Create symlinks in /usr/bin
ln -sf /usr/lib/nvitop/nvitop "${PKG_DIR}/usr/bin/nvitop"
ln -sf /usr/lib/nvitop/nvisel "${PKG_DIR}/usr/bin/nvisel"

# Create Desktop launcher wrapper script
cat << 'EOF' > "${PKG_DIR}/usr/bin/nvitop-launcher"
#!/bin/bash
# Wrapper to launch nvitop in a terminal emulator when started from Application Menu

# If already in an interactive terminal, run directly
if [ -t 0 ] && [ -t 1 ]; then
    exec /usr/lib/nvitop/nvitop "$@"
fi

TERMINALS=(
    "x-terminal-emulator"
    "gnome-terminal"
    "xfce4-terminal"
    "konsole"
    "alacritty"
    "kitty"
    "foot"
    "tilix"
    "terminator"
    "mate-terminal"
    "lxterminal"
    "qterminal"
    "urxvt"
    "rxvt"
    "xterm"
)

TITLE="nvitop - NVIDIA GPU Process Viewer"

for term in "${TERMINALS[@]}"; do
    if command -v "$term" >/dev/null 2>&1; then
        case "$term" in
            gnome-terminal|xfce4-terminal|mate-terminal|lxterminal|terminator|tilix)
                exec "$term" --title="$TITLE" -- /usr/lib/nvitop/nvitop "$@"
                ;;
            konsole|qterminal)
                exec "$term" -p tabtitle="$TITLE" -e /usr/lib/nvitop/nvitop "$@"
                ;;
            alacritty)
                exec "$term" --title "$TITLE" -e /usr/lib/nvitop/nvitop "$@"
                ;;
            kitty)
                exec "$term" --title "$TITLE" /usr/lib/nvitop/nvitop "$@"
                ;;
            foot)
                exec "$term" --title="$TITLE" /usr/lib/nvitop/nvitop "$@"
                ;;
            *)
                exec "$term" -T "$TITLE" -e /usr/lib/nvitop/nvitop "$@"
                ;;
        esac
    fi
done

exec /usr/lib/nvitop/nvitop "$@"
EOF

# Create Desktop Entry for Application Menu
cat << 'EOF' > "${PKG_DIR}/usr/share/applications/nvitop.desktop"
[Desktop Entry]
Type=Application
Version=1.0
Name=nvitop
GenericName=NVIDIA GPU Process Viewer
Comment=Interactive NVIDIA-GPU process viewer and monitor
Icon=nvitop
Exec=nvitop-launcher
Terminal=false
Categories=System;Monitor;HardwareSettings;
Keywords=nvidia;gpu;cuda;process;top;htop;monitor;system;vram;
StartupNotify=true
StartupWMClass=nvitop
EOF

# Install Icons
install -d "${PKG_DIR}/usr/share/icons/hicolor/scalable/apps"
cp "${ICON_DIR}/nvitop.svg" "${PKG_DIR}/usr/share/icons/hicolor/scalable/apps/nvitop.svg"
cp "${ICON_DIR}/nvitop.svg" "${PKG_DIR}/usr/share/pixmaps/nvitop.svg"
cp "${ICON_DIR}/nvitop-256.png" "${PKG_DIR}/usr/share/pixmaps/nvitop.png"

for sz in 16 32 48 64 128 256; do
    install -d "${PKG_DIR}/usr/share/icons/hicolor/${sz}x${sz}/apps"
    cp "${ICON_DIR}/nvitop-${sz}.png" "${PKG_DIR}/usr/share/icons/hicolor/${sz}x${sz}/apps/nvitop.png"
done

# Licenses and documentation
cp COPYING "${PKG_DIR}/usr/share/doc/nvitop/copyright"
cp README.md "${PKG_DIR}/usr/share/doc/nvitop/"

# Calculate installed size in KB
INSTALLED_SIZE=$(du -sk "${PKG_DIR}" | cut -f1)

# Create DEBIAN/control
cat << EOF > "${PKG_DIR}/DEBIAN/control"
Package: nvitop
Version: ${VERSION}-${PKG_REVISION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Installed-Size: ${INSTALLED_SIZE}
Maintainer: Xuehai Pan <XuehaiPan@pku.edu.cn>
Depends: libc6 (>= 2.14)
Recommends: x-terminal-emulator | gnome-terminal | xfce4-terminal | konsole | alacritty | kitty | xterm
Homepage: https://github.com/XuehaiPan/nvitop
Description: An interactive NVIDIA-GPU process viewer and beyond
 nvitop is the one-stop solution for GPU process management, featuring:
  * An interactive resource monitor with process details
  * Real-time metrics for GPU utilization, memory, temperature, and power
  * Easy process termination and filtering by user or PID
  * The nvisel utility for CUDA device selection in scripts
 This package includes the standalone compiled binary with all required
 dependencies bundled and desktop application menu integration.
EOF

# Create DEBIAN/postinst
cat << 'EOF' > "${PKG_DIR}/DEBIAN/postinst"
#!/bin/sh
set -e

if [ "$1" = "configure" ]; then
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database -q /usr/share/applications || true
    fi
    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
    fi
fi

exit 0
EOF

# Create DEBIAN/postrm
cat << 'EOF' > "${PKG_DIR}/DEBIAN/postrm"
#!/bin/sh
set -e

if [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database -q /usr/share/applications || true
    fi
    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
    fi
fi

exit 0
EOF

# Ensure proper permissions
find "${PKG_DIR}" -type d -exec chmod 755 {} +
find "${PKG_DIR}/usr" -type f -exec chmod 644 {} +
chmod 755 "${PKG_DIR}/DEBIAN/postinst" "${PKG_DIR}/DEBIAN/postrm"
chmod 755 "${PKG_DIR}/usr/bin/nvitop-launcher" "${PKG_DIR}/usr/lib/nvitop/nvitop"
[ -f "${PKG_DIR}/usr/lib/nvitop/nvisel" ] && chmod 755 "${PKG_DIR}/usr/lib/nvitop/nvisel" || true

# 6. Build .deb package with dpkg-deb using xz compression and root ownership
echo -e "\n${YELLOW}[6/6] Packaging into final .deb file...${NC}"
dpkg-deb --root-owner-group --build -Zxz "${PKG_DIR}" "${OUTPUT_DIR}/${DEB_NAME}"

# Also place a copy directly in the current workspace root for immediate access
cp "${OUTPUT_DIR}/${DEB_NAME}" "${SCRIPT_DIR}/${DEB_NAME}"

DEB_SIZE=$(du -h "${OUTPUT_DIR}/${DEB_NAME}" | cut -f1)

echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN}    BUILD SUCCESSFUL!                               ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo -e "${BLUE}Created Debian package:${NC} ${DEB_NAME} (${DEB_SIZE})"
echo -e "${BLUE}Location:${NC}               ${SCRIPT_DIR}/${DEB_NAME}"
echo -e "${BLUE}Also available at:${NC}      ${OUTPUT_DIR}/${DEB_NAME}"
echo ""
echo -e "${CYAN}Installation instructions:${NC}"
echo -e "  sudo dpkg -i ${DEB_NAME}"
echo -e "  # Or:"
echo -e "  sudo apt install ./${DEB_NAME}"
echo ""
echo -e "${CYAN}Features included:${NC}"
echo -e "  * Standalone compiled binary (zero pip or system python setup needed)"
echo -e "  * Works across Ubuntu (22.04+), Debian (12+), and Kali Linux"
echo -e "  * CLI commands: /usr/bin/nvitop and /usr/bin/nvisel"
echo -e "  * Desktop App Menu entry: /usr/share/applications/nvitop.desktop"
echo -e "  * High-res icons in scalable SVG and 16..256px PNGs"
echo -e "  * Auto terminal wrapper launcher for GUI application menu"
