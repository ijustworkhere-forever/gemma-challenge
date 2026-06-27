#!/bin/bash

set -e

echo "Installing Gemma Research System..."

# system deps
sudo apt update
sudo apt install -y python3 python3-pip git

# project directory
sudo mkdir -p /opt/gemma-challenge
sudo cp -r ../* /opt/gemma-challenge

cd /opt/gemma-challenge

# python deps
pip3 install requests streamlit pandas pyyaml

# systemd install
sudo cp deploy/*.service /etc/systemd/system/

sudo systemctl daemon-reload

sudo systemctl enable gemma-controller
sudo systemctl enable gemma-swarm
sudo systemctl enable gemma-leaderboard
sudo systemctl enable gemma-dashboard

echo "Starting services..."

sudo systemctl start gemma-controller
sudo systemctl start gemma-swarm
sudo systemctl start gemma-leaderboard
sudo systemctl start gemma-dashboard

echo "System ONLINE"
echo "Dashboard: http://localhost:8501"
