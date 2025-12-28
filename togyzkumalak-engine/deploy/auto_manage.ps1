# ==============================================================================
# Автоматическое управление сервером и обучением на Vast.ai
# ==============================================================================
# Запусти этот скрипт на своем компьютере - он сам все сделает
# ==============================================================================

param(
    [string]$SshConnection = "root@151.237.25.234 -p 23396",
    [string]$ServerUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Continue"

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🤖 АВТОМАТИЧЕСКОЕ УПРАВЛЕНИЕ СЕРВЕРОМ                      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Функция для выполнения SSH команд
function Invoke-SshCommand {
    param([string]$Command)
    $fullCommand = "ssh $SshConnection `"$Command`""
    Write-Host "▶ $Command" -ForegroundColor Gray
    $result = Invoke-Expression $fullCommand 2>&1
    return $result
}

# Функция для проверки сервера через SSH туннель
function Test-Server {
    try {
        $response = Invoke-WebRequest -Uri "$ServerUrl/api/health" -TimeoutSec 5 -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

# 1. Проверка SSH подключения
Write-Host "1️⃣ Проверяю SSH подключение..." -ForegroundColor Yellow
$testResult = Invoke-SshCommand "echo 'Connected!'"
if ($testResult -match "Connected") {
    Write-Host "   ✅ SSH подключение работает" -ForegroundColor Green
} else {
    Write-Host "   ❌ SSH подключение не работает" -ForegroundColor Red
    Write-Host "   Проверь: ssh $SshConnection"
    exit 1
}
Write-Host ""

# 2. Проверка сервера
Write-Host "2️⃣ Проверяю сервер..." -ForegroundColor Yellow

# Создаем SSH туннель в фоне
Write-Host "   Создаю SSH туннель..." -ForegroundColor Gray
$tunnelJob = Start-Job -ScriptBlock {
    param($conn)
    ssh -N -L 8000:localhost:8000 $conn.Split(" ")
} -ArgumentList $SshConnection

Start-Sleep -Seconds 3

if (Test-Server) {
    Write-Host "   ✅ Сервер работает" -ForegroundColor Green
} else {
    Write-Host "   ⚠️ Сервер не отвечает, перезапускаю..." -ForegroundColor Yellow
    
    # Останавливаем старые процессы
    Invoke-SshCommand "pkill -9 -f 'python.*run.py'"
    Start-Sleep -Seconds 2
    
    # Запускаем новый сервер
    Write-Host "   Запускаю сервер..." -ForegroundColor Gray
    Invoke-SshCommand "cd /workspace/togyzkumalak/togyzkumalak-engine && /venv/main/bin/python run.py > server.log 2>&1 &"
    Start-Sleep -Seconds 5
    
    if (Test-Server) {
        Write-Host "   ✅ Сервер запущен" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Не удалось запустить сервер" -ForegroundColor Red
    }
}
Write-Host ""

# 3. Проверка обучения
Write-Host "3️⃣ Проверяю обучение..." -ForegroundColor Yellow
try {
    $sessionsResponse = Invoke-WebRequest -Uri "$ServerUrl/api/training/alphazero/sessions" -UseBasicParsing
    $sessions = ($sessionsResponse.Content | ConvertFrom-Json).sessions
    
    if ($sessions.PSObject.Properties.Count -gt 0) {
        $taskId = ($sessions.PSObject.Properties | Select-Object -First 1).Name
        $session = $sessions.$taskId
        
        Write-Host "   ✅ Обучение запущено!" -ForegroundColor Green
        Write-Host "      Task ID: $taskId" -ForegroundColor Gray
        Write-Host "      Статус: $($session.status)" -ForegroundColor Gray
        Write-Host "      Итерация: $($session.current_iteration)/$($session.total_iterations)" -ForegroundColor Gray
        Write-Host "      Прогресс: $([math]::Round($session.progress, 1))%" -ForegroundColor Gray
    } else {
        Write-Host "   ℹ️ Обучение не запущено" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠️ Не удалось проверить обучение: $_" -ForegroundColor Yellow
}
Write-Host ""

# 4. Мониторинг в реальном времени
Write-Host "4️⃣ Мониторинг (Ctrl+C для остановки)..." -ForegroundColor Yellow
Write-Host ""

try {
    while ($true) {
        Clear-Host
        Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
        Write-Host "║  📊 СТАТУС СЕРВЕРА И ОБУЧЕНИЯ                                ║" -ForegroundColor Cyan
        Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
        Write-Host ""
        
        $timestamp = Get-Date -Format "HH:mm:ss"
        Write-Host "Время: $timestamp" -ForegroundColor Gray
        Write-Host ""
        
        # Статус сервера
        if (Test-Server) {
            Write-Host "✅ Сервер: Работает" -ForegroundColor Green
        } else {
            Write-Host "❌ Сервер: Не отвечает" -ForegroundColor Red
        }
        Write-Host ""
        
        # Статус обучения
        try {
            $sessionsResponse = Invoke-WebRequest -Uri "$ServerUrl/api/training/alphazero/sessions" -UseBasicParsing -TimeoutSec 3
            $sessions = ($sessionsResponse.Content | ConvertFrom-Json).sessions
            
            if ($sessions.PSObject.Properties.Count -gt 0) {
                $taskId = ($sessions.PSObject.Properties | Select-Object -First 1).Name
                $session = $sessions.$taskId
                
                Write-Host "📈 Обучение:" -ForegroundColor Cyan
                Write-Host "   Task ID: $taskId" -ForegroundColor Gray
                Write-Host "   Статус: $($session.status)" -ForegroundColor $(if ($session.status -eq "running") { "Green" } else { "Yellow" })
                Write-Host "   Итерация: $($session.current_iteration)/$($session.total_iterations)" -ForegroundColor White
                Write-Host "   Прогресс: $([math]::Round($session.progress, 1))%" -ForegroundColor White
                
                if ($session.current_iteration -gt 0) {
                    $elapsed = if ($session.elapsed_time) { [math]::Round($session.elapsed_time, 0) } else { 0 }
                    Write-Host "   Время: ${elapsed}с" -ForegroundColor Gray
                }
            } else {
                Write-Host "ℹ️ Обучение: Не запущено" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "⚠️ Не удалось получить статус обучения" -ForegroundColor Yellow
        }
        Write-Host ""
        
        # GPU статус
        $gpuInfo = Invoke-SshCommand "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -3"
        if ($gpuInfo) {
            Write-Host "🎮 GPU (первые 3):" -ForegroundColor Cyan
            $gpuInfo -split "`n" | ForEach-Object {
                if ($_ -match "(\d+),\s*(\d+)\s*%,\s*(\d+)\s*MiB,\s*(\d+)\s*MiB") {
                    $gpuId = $matches[1]
                    $util = $matches[2]
                    $memUsed = $matches[3]
                    $memTotal = $matches[4]
                    $color = if ([int]$util -gt 0) { "Green" } else { "Gray" }
                    Write-Host "   GPU $gpuId : Util $util% | Mem ${memUsed}MB/${memTotal}MB" -ForegroundColor $color
                }
            }
        }
        Write-Host ""
        
        Write-Host "Обновление через 5 секунд... (Ctrl+C для остановки)" -ForegroundColor DarkGray
        
        Start-Sleep -Seconds 5
    }
} catch {
    Write-Host "`nОстановлено" -ForegroundColor Yellow
} finally {
    # Останавливаем SSH туннель
    if ($tunnelJob) {
        Stop-Job $tunnelJob
        Remove-Job $tunnelJob
    }
}
