#!/bin/bash
# ==============================================================================
# Vast.ai On-Start Script
# ==============================================================================
# This script runs automatically when your Vast.ai instance starts
# Set this in the "On-Start Script" field when creating an instance
# ==============================================================================

# Configuration
REPO_URL="https://github.com/YOUR_USERNAME/Toguzkumalak.git"
PROJECT_DIR="/workspace/togyzkumalak"
ENGINE_DIR="$PROJECT_DIR/gym-togyzkumalak-master/togyzkumalak-engine"

# Clone or update repository
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    git pull origin master 2>/dev/null || true
else
    git clone "$REPO_URL" "$PROJECT_DIR" 2>/dev/null || true
fi

# Install dependencies
cd "$ENGINE_DIR"
pip install -q -r requirements.txt 2>/dev/null || true

# Create directories
mkdir -p models/alphazero logs/{alphazero,games,gemini_battles/{games,sessions,summaries},self_play,training,wandb_local,ab_tests/results} training_data 2>/dev/null || true

# Start server in background
nohup python run.py > /workspace/server.log 2>&1 &

echo "Togyzkumalak Engine started on port 8000"
echo "Logs: /workspace/server.log"
