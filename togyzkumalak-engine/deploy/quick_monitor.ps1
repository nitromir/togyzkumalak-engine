# Quick Server Monitor
param(
    [string]$SshConnection = "root@151.237.25.234 -p 23396"
)

$ErrorActionPreference = "Continue"

Write-Host "=== SERVER MONITOR ===" -ForegroundColor Cyan
Write-Host ""

# SSH tunnel in background
Write-Host "Creating SSH tunnel..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "ssh -N -L 8000:localhost:8000 $SshConnection" -WindowStyle Minimized
Start-Sleep -Seconds 3

Write-Host "Monitoring (Ctrl+C to stop)..." -ForegroundColor Green
Write-Host ""

try {
    while ($true) {
        Clear-Host
        Write-Host "=== STATUS ===" -ForegroundColor Cyan
        Write-Host "Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
        Write-Host ""
        
        # Server status
        try {
            $health = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -TimeoutSec 3
            Write-Host "Server: OK" -ForegroundColor Green
        } catch {
            Write-Host "Server: ERROR" -ForegroundColor Red
        }
        Write-Host ""
        
        # Training status
        try {
            $sessions = Invoke-WebRequest -Uri "http://localhost:8000/api/training/alphazero/sessions" -UseBasicParsing -TimeoutSec 3
            $data = $sessions.Content | ConvertFrom-Json
            
            if ($data.sessions.PSObject.Properties.Count -gt 0) {
                $taskId = ($data.sessions.PSObject.Properties | Select-Object -First 1).Name
                $session = $data.sessions.$taskId
                
                Write-Host "Training:" -ForegroundColor Cyan
                Write-Host "  Status: $($session.status)" -ForegroundColor $(if ($session.status -eq "running") { "Green" } else { "Yellow" })
                Write-Host "  Iteration: $($session.current_iteration)/$($session.total_iterations)" -ForegroundColor White
                Write-Host "  Progress: $([math]::Round($session.progress, 1))%" -ForegroundColor White
            } else {
                Write-Host "Training: Not running" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "Training: Check failed" -ForegroundColor Yellow
        }
        Write-Host ""
        
        # GPU status
        $gpuCmd = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -3"
        $gpuInfo = ssh $SshConnection.Split(" ") $gpuCmd 2>$null
        if ($gpuInfo) {
            Write-Host "GPU (first 3):" -ForegroundColor Cyan
            $gpuInfo -split "`n" | ForEach-Object {
                if ($_ -match "(\d+),\s*(\d+)\s*%,\s*(\d+)") {
                    Write-Host "  GPU $($matches[1]): $($matches[2])% util, $($matches[3])MB mem" -ForegroundColor Gray
                }
            }
        }
        Write-Host ""
        Write-Host "Refreshing in 5 seconds..." -ForegroundColor DarkGray
        
        Start-Sleep -Seconds 5
    }
} catch {
    Write-Host "`nStopped" -ForegroundColor Yellow
}
