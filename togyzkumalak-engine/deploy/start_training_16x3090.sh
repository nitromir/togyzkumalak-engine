#!/bin/bash
# ==============================================================================
# Optimized AlphaZero Training for 16x RTX 3090
# ==============================================================================
# Hardware: 16x RTX 3090, AMD EPYC 7543, 434GB RAM
# Expected: ~200-250 iterations per hour
# ==============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🦾 AlphaZero Training - 16x RTX 3090 Config                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check GPU availability
echo "🔍 Checking GPUs..."
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(min(torch.cuda.device_count(), 16)):
    props = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {torch.cuda.get_device_name(i)} - {props.total_memory/1024**3:.1f}GB")
EOF

echo ""
echo "📋 Training Configuration:"
echo "   • Iterations: 250"
echo "   • Games/iter: 200"
echo "   • MCTS sims: 200"
echo "   • Batch size: 4096 (256 × 16 GPUs)"
echo "   • Hidden size: 512"
echo "   • Parallel games: 64"
echo "   • Bootstrap: enabled"
echo ""

# Confirm
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting training via API..."

curl -X POST "http://localhost:8000/api/training/alphazero/start" \
    -H "Content-Type: application/json" \
    -d '{
        "numIters": 250,
        "numEps": 200,
        "numMCTSSims": 200,
        "cpuct": 1.0,
        "batch_size": 4096,
        "hidden_size": 512,
        "epochs": 10,
        "use_bootstrap": true,
        "use_multiprocessing": true,
        "num_parallel_games": 64,
        "save_every_n_iters": 10
    }'

echo ""
echo ""
echo "✅ Training started!"
echo ""
echo "📊 Monitor progress:"
echo "   python deploy/monitor.py"
echo ""
echo "💾 Auto-sync checkpoints (run on LOCAL machine):"
echo "   ./deploy/sync_checkpoints.sh 'root@<ip> -p <port>'"
echo ""
echo "🌐 Web UI: http://localhost:8000"
