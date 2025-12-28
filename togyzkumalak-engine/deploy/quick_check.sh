#!/bin/bash
# Quick server check and start script
# Run this ON THE SERVER in your SSH session

echo "🔍 Checking server status..."

# Check if running
if ps aux | grep -v grep | grep -q "python run.py"; then
    echo "✅ Server is RUNNING"
    ps aux | grep -v grep | grep "python run.py"
else
    echo "❌ Server is NOT running"
    echo ""
    echo "🚀 Starting server..."
    cd /workspace/togyzkumalak/togyzkumalak-engine
    nohup python run.py > server.log 2>&1 &
    sleep 3
    echo ""
    echo "📋 Server log:"
    tail -15 server.log
fi

echo ""
echo "🌐 Testing server..."
curl -s http://localhost:8000/api/health && echo "" || echo "⚠ Server not responding"

echo ""
echo "✅ Done! Open http://localhost:8000 in browser (via SSH tunnel)"
