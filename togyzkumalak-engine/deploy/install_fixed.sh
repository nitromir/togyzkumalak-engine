#!/bin/bash
# ==============================================================================
# Fixed Installation Command for Vast.ai
# ==============================================================================
# Use this if the default path doesn't work
# ==============================================================================

set -e

echo "🔍 Checking repository structure..."

cd /workspace

# Clone if not exists
if [ ! -d "togyzkumalak" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/nitromir/togyzkumalak-engine.git togyzkumalak
fi

cd togyzkumalak

# Check structure
echo "📁 Repository structure:"
ls -la

# Find the engine directory
if [ -d "togyzkumalak-engine" ]; then
    ENGINE_DIR="togyzkumalak-engine"
elif [ -d "gym-togyzkumalak-master/togyzkumalak-engine" ]; then
    ENGINE_DIR="gym-togyzkumalak-master/togyzkumalak-engine"
else
    echo "❌ Cannot find engine directory!"
    echo "Available directories:"
    find . -type d -maxdepth 3 | head -20
    exit 1
fi

echo "✅ Found engine at: $ENGINE_DIR"
cd "$ENGINE_DIR"

echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "📁 Creating directories..."
mkdir -p models/alphazero
mkdir -p logs/{alphazero,games,gemini_battles/{games,sessions,summaries},self_play,training,wandb_local,ab_tests/results}
mkdir -p training_data

echo ""
echo "🚀 Starting server..."
nohup python run.py > server.log 2>&1 &

sleep 3

echo ""
echo "📋 Server status:"
tail -10 server.log

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Access UI at: http://151.237.25.234:8000"
echo "📊 Monitor: python deploy/monitor.py"
