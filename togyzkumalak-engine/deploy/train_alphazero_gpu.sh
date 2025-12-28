#!/bin/bash
# ==============================================================================
# AlphaZero GPU Training Script for Vast.ai
# ==============================================================================
# Optimized for multi-GPU training on powerful instances (12x RTX 4090)
# ==============================================================================

set -e

echo "=============================================="
echo "  AlphaZero GPU Training"
echo "=============================================="

# Check CUDA availability
python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
EOF

# Training parameters optimized for 12x RTX 4090
# Adjust these based on your hardware
NUM_ITERATIONS=100
GAMES_PER_ITERATION=100
MCTS_SIMULATIONS=200
BATCH_SIZE=256
LEARNING_RATE=0.001
EPOCHS=10

echo ""
echo "Training Configuration:"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Games/Iteration: $GAMES_PER_ITERATION"
echo "  MCTS Simulations: $MCTS_SIMULATIONS"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Start training via API
curl -X POST "http://localhost:8000/api/training/alphazero/start" \
    -H "Content-Type: application/json" \
    -d "{
        \"iterations\": $NUM_ITERATIONS,
        \"games_per_iteration\": $GAMES_PER_ITERATION,
        \"mcts_simulations\": $MCTS_SIMULATIONS,
        \"batch_size\": $BATCH_SIZE,
        \"lr\": $LEARNING_RATE,
        \"epochs\": $EPOCHS,
        \"use_bootstrap\": true
    }"

echo ""
echo "Training started! Monitor progress at:"
echo "  http://<your-vast-ip>:8000 -> Training tab"
echo ""
echo "To monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
