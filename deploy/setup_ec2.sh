#!/bin/bash
# =============================================================
# EC2 Ubuntu Setup Script for SAR Flood Prediction App
# Run this ON the EC2 instance (the deploy script will send it)
# =============================================================
set -e

echo "=========================================="
echo "  SAR Flood Prediction - EC2 Setup"
echo "=========================================="

# --- System packages ---
echo "[1/7] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git nginx

# --- Clone or update repo ---
APP_DIR="/home/ubuntu/sar-flood-prediction"
echo "[2/7] Setting up application directory..."
if [ -d "$APP_DIR" ]; then
    echo "  Repo exists, pulling latest..."
    cd "$APP_DIR"
    git pull origin main
else
    echo "  Cloning repository..."
    git clone https://github.com/regret64/sar-flood-prediction.git "$APP_DIR"
    cd "$APP_DIR"
fi

# --- Python virtual environment ---
echo "[3/7] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# --- Install dependencies ---
echo "[4/7] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# --- Create uploads directory ---
mkdir -p uploads

# --- Systemd service ---
echo "[5/7] Setting up systemd service..."
sudo tee /etc/systemd/system/floodapp.service > /dev/null <<'EOF'
[Unit]
Description=SAR Flood Prediction App (Gunicorn)
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/sar-flood-prediction
Environment="PATH=/home/ubuntu/sar-flood-prediction/venv/bin"
Environment="PORT=8000"
ExecStart=/home/ubuntu/sar-flood-prediction/venv/bin/gunicorn --workers 2 --bind 0.0.0.0:8000 --timeout 120 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable floodapp
sudo systemctl restart floodapp

# --- Nginx reverse proxy ---
echo "[6/7] Configuring Nginx..."
sudo tee /etc/nginx/sites-available/floodapp > /dev/null <<'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    location /static/ {
        alias /home/ubuntu/sar-flood-prediction/static/;
    }
}
EOF

# Enable site and disable default
sudo ln -sf /etc/nginx/sites-available/floodapp /etc/nginx/sites-enabled/floodapp
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# --- Final status ---
echo "[7/7] Checking service status..."
sleep 2
sudo systemctl status floodapp --no-pager || true

echo ""
echo "=========================================="
echo "  DEPLOYMENT COMPLETE!"
echo "  App running at: http://$(curl -s ifconfig.me)"
echo "=========================================="
