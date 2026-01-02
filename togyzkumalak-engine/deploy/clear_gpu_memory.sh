#!/bin/bash
# Скрипт для полной очистки памяти GPU

echo "🔍 Checking GPU memory usage..."
nvidia-smi

echo ""
echo "🛑 Stopping training processes..."
pkill -f run.py
pkill -f probs
sleep 2

echo ""
echo "🧹 Clearing Python GPU cache..."
python3 -c "
import torch
import gc

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    gc.collect()
    print('✅ GPU cache cleared for all devices')
else:
    print('❌ CUDA not available')
"

echo ""
echo "🔍 Final GPU status:"
nvidia-smi

echo ""
echo "✅ Done! GPU memory should be cleared now."
