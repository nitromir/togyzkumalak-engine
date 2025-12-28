# ==============================================================================
# Auto Check and Start Server via SSH
# ==============================================================================

$SSH_HOST = "151.237.25.234"
$SSH_PORT = "23396"
$SSH_USER = "root"

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🔍 Checking Server Status                                 ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if server is running
Write-Host "Checking if server is running..." -ForegroundColor Yellow
$checkCmd = "ps aux | grep 'python run.py' | grep -v grep"
$result = ssh -p $SSH_PORT "$SSH_USER@$SSH_HOST" $checkCmd 2>&1

if ($result -match "python run.py") {
    Write-Host "✓ Server is running!" -ForegroundColor Green
    Write-Host $result
} else {
    Write-Host "✗ Server is NOT running" -ForegroundColor Red
    Write-Host ""
    Write-Host "Starting server..." -ForegroundColor Yellow
    
    $startCmd = @"
cd /workspace/togyzkumalak/togyzkumalak-engine && \
nohup python run.py > server.log 2>&1 & \
sleep 3 && \
tail -10 server.log
"@
    
    ssh -p $SSH_PORT "$SSH_USER@$SSH_HOST" $startCmd
}

Write-Host ""
Write-Host "Testing server health..." -ForegroundColor Yellow
$health = ssh -p $SSH_PORT "$SSH_USER@$SSH_HOST" "curl -s http://localhost:8000/api/health" 2>&1

if ($health -match "ok" -or $health -match "status") {
    Write-Host "✓ Server is healthy!" -ForegroundColor Green
    Write-Host $health
} else {
    Write-Host "⚠ Server health check failed" -ForegroundColor Yellow
    Write-Host $health
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Keep SSH tunnel open:" -ForegroundColor White
Write-Host "   ssh -p $SSH_PORT $SSH_USER@$SSH_HOST -L 8000:localhost:8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Open in browser: http://localhost:8000" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
