# Auto-start server via SSH
$ErrorActionPreference = "Continue"

Write-Host "Connecting and starting server..." -ForegroundColor Cyan

$cmd = 'cd /workspace/togyzkumalak/togyzkumalak-engine && source /venv/main/bin/activate 2>/dev/null || true && if ! ps aux | grep -v grep | grep -q "python run.py"; then echo "Starting server..." && nohup python run.py > server.log 2>&1 & sleep 5 && tail -20 server.log; else echo "Server already running" && ps aux | grep -v grep | grep "python run.py"; fi && curl -s http://localhost:8000/api/health || echo "Server starting..."'

ssh -p 23396 root@151.237.25.234 $cmd

Write-Host "`nDone! Now open: http://localhost:8000" -ForegroundColor Green
Write-Host "Keep SSH tunnel open: ssh -p 23396 root@151.237.25.234 -L 8000:localhost:8000" -ForegroundColor Yellow
