# ==============================================================================
# Auto-Sync Checkpoints from Vast.ai to Local Machine (Windows PowerShell)
# ==============================================================================
#
# Continuously syncs new checkpoints from remote server to local directory.
# Run this on YOUR LOCAL WINDOWS machine while training runs on Vast.ai.
#
# Usage:
#   .\sync_checkpoints.ps1 -SshConnection "root@123.45.67.89 -p 12345"
#   .\sync_checkpoints.ps1 -SshConnection "root@123.45.67.89 -p 12345" -LocalDir ".\checkpoints" -Interval 60
#
# Requirements:
#   - OpenSSH client installed (Windows 10+ has it built-in)
#   - SSH key authentication configured
#
# ==============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$SshConnection,
    
    [string]$LocalDir = ".\alphazero_checkpoints",
    
    [int]$Interval = 30,
    
    [string]$RemotePath = "/workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine/models/alphazero"
)

# Create local directory
New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🔄 AlphaZero Checkpoint Auto-Sync (Windows)                 ║" -ForegroundColor Cyan
Write-Host "╠══════════════════════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "║  Remote: $SshConnection" -ForegroundColor Cyan
Write-Host "║  Path:   $RemotePath" -ForegroundColor Cyan
Write-Host "║  Local:  $LocalDir" -ForegroundColor Cyan
Write-Host "║  Interval: ${Interval}s" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Test SSH connection
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
try {
    $testResult = ssh $SshConnection.Split(" ") "echo 'Connected!'" 2>&1
    if ($testResult -match "Connected") {
        Write-Host "✓ SSH connection successful" -ForegroundColor Green
    } else {
        throw "Connection test failed"
    }
} catch {
    Write-Host "✗ SSH connection failed: $_" -ForegroundColor Red
    Write-Host "Make sure you can connect with: ssh $SshConnection"
    exit 1
}

$syncCount = 0

function Sync-Checkpoints {
    $script:syncCount++
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    Write-Host ""
    Write-Host "[$timestamp] Sync #$syncCount" -ForegroundColor Cyan
    
    # Get list of remote files
    try {
        $remoteFiles = ssh $SshConnection.Split(" ") "ls $RemotePath/*.pth.tar 2>/dev/null" 2>&1
        
        if ($remoteFiles -match "No such file") {
            Write-Host "  No checkpoints found on remote server yet..." -ForegroundColor Yellow
            return
        }
        
        $fileList = $remoteFiles -split "`n" | Where-Object { $_ -match "\.pth\.tar$" }
        $fileCount = $fileList.Count
        
        Write-Host "  Found $fileCount checkpoint(s) on remote server" -ForegroundColor Gray
        
        # Download each file
        foreach ($remoteFile in $fileList) {
            $fileName = Split-Path $remoteFile -Leaf
            $localFile = Join-Path $LocalDir $fileName
            
            # Check if file exists and is same size
            if (Test-Path $localFile) {
                Write-Host "  ⏭ $fileName (already exists)" -ForegroundColor Gray
            } else {
                Write-Host "  ⬇ Downloading $fileName..." -ForegroundColor Yellow
                scp $SshConnection.Split(" ")[0..($SshConnection.Split(" ").Count-1)] + ":$remoteFile" $LocalDir 2>&1 | Out-Null
                
                if (Test-Path $localFile) {
                    $size = (Get-Item $localFile).Length / 1MB
                    Write-Host "    ✓ Downloaded ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
                }
            }
        }
        
        # Also sync metrics file
        try {
            $metricsContent = ssh $SshConnection.Split(" ") "cat $RemotePath/training_metrics.json 2>/dev/null"
            if ($metricsContent) {
                $metricsContent | Out-File -FilePath (Join-Path $LocalDir "training_metrics.json") -Encoding UTF8
            }
        } catch {}
        
        # Report
        $localFiles = Get-ChildItem -Path $LocalDir -Filter "*.pth.tar" -ErrorAction SilentlyContinue
        $totalSize = ($localFiles | Measure-Object -Property Length -Sum).Sum / 1MB
        
        Write-Host "  ✓ Local checkpoints: $($localFiles.Count) ($([math]::Round($totalSize, 1)) MB)" -ForegroundColor Green
        
        # Show latest
        $latest = $localFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
            Write-Host "  ⭐ Latest: $($latest.Name)" -ForegroundColor Green
        }
        
    } catch {
        Write-Host "  Error: $_" -ForegroundColor Red
    }
}

function Show-Summary {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  📊 Sync Summary" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    
    $localFiles = Get-ChildItem -Path $LocalDir -Filter "*.pth.tar" -ErrorAction SilentlyContinue | 
                  Sort-Object Length -Descending
    
    foreach ($file in $localFiles | Select-Object -First 10) {
        $size = [math]::Round($file.Length / 1MB, 2)
        Write-Host "    $($file.Name) - $size MB"
    }
    
    $total = ($localFiles | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ""
    Write-Host "  Total: $($localFiles.Count) files, $([math]::Round($total, 1)) MB"
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
}

# Handle Ctrl+C
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    Write-Host "`nStopping sync..." -ForegroundColor Yellow
    Show-Summary
}

Write-Host ""
Write-Host "Starting auto-sync (Ctrl+C to stop)..." -ForegroundColor Green

try {
    while ($true) {
        Sync-Checkpoints
        
        # Countdown
        for ($i = $Interval; $i -gt 0; $i--) {
            Write-Host "`r  Next sync in ${i}s...  " -NoNewline
            Start-Sleep -Seconds 1
        }
        Write-Host "`r                        `r" -NoNewline
    }
} finally {
    Show-Summary
}
