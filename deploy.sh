#!/bin/bash
#
# Ludwig ML Pipeline - Deployment Script for Linux
# Sets up the pipeline as a systemd service
#

set -e

INSTALL_DIR="/opt/ludwig-ml-pipeline"
SERVICE_USER="ludwig"
SERVICE_NAME="ludwig-pipeline"

echo "========================================"
echo "  Ludwig ML Pipeline - Deployment"
echo "========================================"
echo ""
echo "This will:"
echo "  1. Copy files to $INSTALL_DIR"
echo "  2. Create systemd service"
echo "  3. Enable auto-start on boot"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 1
fi

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root (use sudo)"
   exit 1
fi

# Create service user
echo ""
echo "Creating service user..."
id -u $SERVICE_USER &>/dev/null || useradd -r -s /bin/false $SERVICE_USER

# Create installation directory
echo "Creating installation directory..."
mkdir -p $INSTALL_DIR
cp -r ./* $INSTALL_DIR/
chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR

# Install Python dependencies
echo ""
echo "Installing dependencies..."
cd $INSTALL_DIR
sudo -u $SERVICE_USER bash -c "
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
"

# Create systemd service file
echo ""
echo "Creating systemd service..."
cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=Ludwig ML Pipeline Service
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
ExecStart=$INSTALL_DIR/venv/bin/python run_pipeline.py examples/du-cell-ludwig.yaml
Restart=on-failure
RestartSec=10

# Resource limits (optimized for 16GB RAM, 8 CPUs)
MemoryLimit=12G
CPUQuota=600%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ludwig-pipeline

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
echo ""
echo "Reloading systemd..."
systemctl daemon-reload

# Enable service
echo "Enabling service..."
systemctl enable ${SERVICE_NAME}.service

echo ""
echo "========================================"
echo "✅ Deployment Complete!"
echo "========================================"
echo ""
echo "Service Commands:"
echo "  Start:   sudo systemctl start ${SERVICE_NAME}"
echo "  Stop:    sudo systemctl stop ${SERVICE_NAME}"
echo "  Status:  sudo systemctl status ${SERVICE_NAME}"
echo "  Logs:    sudo journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "Files installed to: $INSTALL_DIR"
echo "========================================"
