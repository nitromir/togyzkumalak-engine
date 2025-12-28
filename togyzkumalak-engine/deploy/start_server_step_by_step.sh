#!/bin/bash
# Step-by-step server start

cd /workspace/togyzkumalak/togyzkumalak-engine

echo "Checking if server is running..."
if ps aux | grep -v grep | grep -q "python run.py"; then
    echo "✅ Server is already running"
    ps aux | grep -v grep | grep "python run.py"
else
    echo "🚀 Starting server..."
    nohup python run.py > server.log 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    
    echo "Waiting for server to start..."
    sleep 5
    
    echo "📋 Server log:"
    tail -25 server.log
    
    echo ""
    echo "Testing server..."
    curl -s http://localhost:8000/api/health && echo "" || echo "⚠ Server not responding yet"
fi

echo ""
echo "✅ Done! Server should be running on port 8000"
echo "Open http://localhost:8000 in browser (via SSH tunnel)"
