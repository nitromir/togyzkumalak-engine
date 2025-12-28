# ==============================================================================
# Auto-Connect and Setup Script for Windows
# ==============================================================================
# This script connects to Vast.ai and sets up the project automatically
# ==============================================================================

param(
    [string]$Host = "151.237.25.234",
    [int]$Port = 23396,
    [string]$User = "root"
)

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🚀 Vast.ai Auto-Setup Script                               ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "Connecting to: $User@$Host`:$Port" -ForegroundColor Yellow
Write-Host ""

# Test SSH connection
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
try {
    $test = ssh -p $Port "$User@$Host" "echo 'Connected!'" 2>&1
    if ($test -match "Connected") {
        Write-Host "✓ SSH connection successful" -ForegroundColor Green
    } else {
        Write-Host "⚠ Connection test returned: $test" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ SSH connection failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Make sure OpenSSH is installed (Windows 10+ has it built-in)"
    Write-Host "2. Try: ssh -p $Port $User@$Host"
    Write-Host "3. If it asks for password, you may need to set up SSH keys"
    exit 1
}

Write-Host ""
Write-Host "Setting up project on remote server..." -ForegroundColor Yellow

# Setup command
$setupCmd = @"
cd /workspace && \
git clone https://github.com/nitromir/togyzkumalak-engine.git togyzkumalak 2>/dev/null || \
(cd togyzkumalak && git pull) && \
cd togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine && \
pip install -q -r requirements.txt && \
mkdir -p models/alphazero logs/{alphazero,games,gemini_battles/{games,sessions,summaries},self_play,training,wandb_local,ab_tests/results} training_data && \
echo '✅ Setup complete!'
"@

try {
    $result = ssh -p $Port "$User@$Host" $setupCmd 2>&1
    Write-Host $result
    Write-Host ""
    Write-Host "✓ Setup completed!" -ForegroundColor Green
} catch {
    Write-Host "✗ Setup failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Connect with port forwarding:" -ForegroundColor White
Write-Host "   ssh -p $Port $User@$Host -L 8000:localhost:8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. On the server, start the server:" -ForegroundColor White
Write-Host "   cd /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine" -ForegroundColor Yellow
Write-Host "   python run.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Open browser: http://localhost:8000" -ForegroundColor White
Write-Host ""
