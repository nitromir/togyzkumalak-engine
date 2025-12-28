#!/bin/bash
# ==============================================================================
# Vast.ai Setup Script for Togyzkumalak Engine
# ==============================================================================
# Run this script after connecting to your Vast.ai instance via SSH
# Template: PyTorch (Vast) - vastai/pytorch
# ==============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  Togyzkumalak Engine - Vast.ai Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/YOUR_USERNAME/Toguzkumalak.git"
PROJECT_DIR="/workspace/togyzkumalak"
ENGINE_DIR="$PROJECT_DIR/gym-togyzkumalak-master/togyzkumalak-engine"

# Step 1: Update system
echo -e "${YELLOW}[1/6] Updating system packages...${NC}"
apt-get update -qq

# Step 2: Clone repository
echo -e "${YELLOW}[2/6] Cloning repository...${NC}"
if [ -d "$PROJECT_DIR" ]; then
    echo "Project directory exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull origin master
else
    git clone "$REPO_URL" "$PROJECT_DIR"
fi

# Step 3: Navigate to engine directory
echo -e "${YELLOW}[3/6] Setting up engine directory...${NC}"
cd "$ENGINE_DIR"

# Step 4: Install Python dependencies
echo -e "${YELLOW}[4/6] Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Step 5: Create necessary directories
echo -e "${YELLOW}[5/6] Creating directories...${NC}"
mkdir -p models/alphazero
mkdir -p logs/{alphazero,games,gemini_battles/{games,sessions,summaries},self_play,training,wandb_local,ab_tests/results}
mkdir -p training_data

# Step 6: Set up environment variables
echo -e "${YELLOW}[6/6] Setting up environment...${NC}"
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Gemini API Key (get from https://aistudio.google.com/)
GEMINI_API_KEY=your_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF
    echo -e "${YELLOW}⚠️  Please edit .env file and add your GEMINI_API_KEY${NC}"
fi

echo ""
echo -e "${GREEN}=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To start the server:"
echo "  cd $ENGINE_DIR"
echo "  python run.py"
echo ""
echo "Or use screen for persistent sessions:"
echo "  screen -S togyzkumalak"
echo "  python run.py"
echo "  (Ctrl+A, D to detach)"
echo ""
echo "Access the UI at:"
echo "  http://<your-vast-ip>:8000"
echo ""
echo "For GPU training, check CUDA:"
echo "  nvidia-smi"
echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')\""
echo "=============================================="
echo -e "${NC}"
