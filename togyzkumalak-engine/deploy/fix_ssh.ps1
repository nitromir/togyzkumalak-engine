# ==============================================================================
# Fix SSH Connection for Vast.ai
# ==============================================================================
# Если SSH закрывается сразу, попробуй эти варианты
# ==============================================================================

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🔧 SSH Connection Troubleshooting                          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "Проблема: Connection closed by server" -ForegroundColor Yellow
Write-Host ""
Write-Host "Варианты решения:" -ForegroundColor Cyan
Write-Host ""

Write-Host "1️⃣  ИСПОЛЬЗУЙ JUPYTER (САМЫЙ ПРОСТОЙ!)" -ForegroundColor Green
Write-Host "   • На Vast.ai нажми зелёную кнопку 'Open'" -ForegroundColor White
Write-Host "   • В Jupyter: New → Terminal" -ForegroundColor White
Write-Host "   • Готово! Не нужен SSH ключ!" -ForegroundColor White
Write-Host ""

Write-Host "2️⃣  Попробуй через Proxy SSH:" -ForegroundColor Yellow
Write-Host "   ssh -p 16593 root@ssh7.vast.ai" -ForegroundColor White
Write-Host "   Потом внутри: ssh root@151.237.25.234 -p 23396" -ForegroundColor White
Write-Host ""

Write-Host "3️⃣  Настрой SSH ключ:" -ForegroundColor Yellow
Write-Host ""

# Генерируем ключ если нет
$keyPath = "$env:USERPROFILE\.ssh\id_rsa"
if (-not (Test-Path $keyPath)) {
    Write-Host "Генерируем SSH ключ..." -ForegroundColor Yellow
    ssh-keygen -t rsa -b 4096 -f $keyPath -N '""' -C "vastai-key"
    Write-Host "✓ Ключ создан" -ForegroundColor Green
} else {
    Write-Host "✓ SSH ключ уже существует" -ForegroundColor Green
}

# Показываем публичный ключ
Write-Host ""
Write-Host "Твой публичный ключ (скопируй его):" -ForegroundColor Cyan
Write-Host ""
$pubKey = Get-Content "$keyPath.pub"
Write-Host $pubKey -ForegroundColor Yellow
Write-Host ""

# Копируем в буфер
$pubKey | Set-Clipboard
Write-Host "✓ Ключ скопирован в буфер обмена!" -ForegroundColor Green
Write-Host ""

Write-Host "Теперь:" -ForegroundColor Cyan
Write-Host "1. Иди на Vast.ai → Settings → SSH Keys" -ForegroundColor White
Write-Host "2. Нажми 'Add Key'" -ForegroundColor White
Write-Host "3. Вставь ключ (Ctrl+V)" -ForegroundColor White
Write-Host "4. Сохрани" -ForegroundColor White
Write-Host "5. Попробуй подключиться снова:" -ForegroundColor White
Write-Host "   ssh -p 23396 root@151.237.25.234 -L 8000:localhost:8000" -ForegroundColor Yellow
Write-Host ""

Write-Host "Или просто используй Jupyter Terminal - это проще! 😊" -ForegroundColor Green
