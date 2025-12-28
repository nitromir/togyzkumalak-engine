#!/bin/bash
# ==============================================================================
# Quick Setup Script for Vast.ai Instance
# ==============================================================================
# Run this ON THE SERVER after SSH connection
# ==============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚀 Togyzkumalak Engine - Quick Setup                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Step 1: Check GPU
echo -e "${CYAN}[1/6] Checking GPUs...${NC}"
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      Memory: {props.total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  CUDA not available!")
EOF

echo ""

# Step 2: Clone repository
echo -e "${CYAN}[2/6] Cloning repository...${NC}"
if [ -d "/workspace/togyzkumalak" ]; then
    echo "Repository exists, updating..."
    cd /workspace/togyzkumalak
    git pull origin master || true
else
    cd /workspace
    git clone https://github.com/nitromir/togyzkumalak-engine.git togyzkumalak
    cd togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine
fi

cd /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine

echo -e "${GREEN}✓ Repository ready${NC}"
echo ""

# Step 3: Install dependencies
echo -e "${CYAN}[3/6] Installing Python dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 4: Create directories
echo -e "${CYAN}[4/6] Creating directories...${NC}"
mkdir -p models/alphazero
mkdir -p logs/{alphazero,games,gemini_battles/{games,sessions,summaries},self_play,training,wandb_local,ab_tests/results}
mkdir -p training_data
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 5: Check training data
echo -e "${CYAN}[5/6] Checking training data...${NC}"
if [ -f "training_data/transitions_compact.jsonl" ]; then
    LINES=$(wc -l < training_data/transitions_compact.jsonl)
    echo -e "${GREEN}✓ Training data found: $LINES examples${NC}"
else
    echo -e "${YELLOW}⚠ Training data not found (bootstrap will be skipped)${NC}"
fi
echo ""

# Step 6: Start server
echo -e "${CYAN}[6/6] Starting server...${NC}"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Setup Complete!                                         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "To start the server:"
echo -e "  ${CYAN}cd /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine${NC}"
echo -e "  ${CYAN}python run.py${NC}"
echo ""
echo "Or run in background:"
echo -e "  ${CYAN}nohup python run.py > server.log 2>&1 &${NC}"
echo ""
echo "Then access UI at:"
echo -e "  ${CYAN}http://localhost:8000${NC}"
echo ""
echo "Monitor training:"
echo -e "  ${CYAN}python deploy/monitor.py${NC}"
echo ""
