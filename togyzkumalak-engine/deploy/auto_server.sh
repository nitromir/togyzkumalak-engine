#!/bin/bash
# Автоматическое управление сервером и мониторинг обучения
# Запусти этот скрипт на сервере один раз - он будет работать автоматически

cd /workspace/togyzkumalak/togyzkumalak-engine

# Функция проверки сервера
check_server() {
    curl -s http://localhost:8000/api/health > /dev/null 2>&1
    return $?
}

# Функция перезапуска сервера
restart_server() {
    echo "🔄 Перезапускаю сервер..."
    pkill -9 -f "python run.py"
    sleep 2
    nohup /venv/main/bin/python run.py > server.log 2>&1 &
    sleep 5
}

# Функция показа статуса
show_status() {
    clear
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  📊 СТАТУС СЕРВЕРА И ОБУЧЕНИЯ                                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Время: $(date '+%H:%M:%S')"
    echo ""
    
    # Сервер
    if check_server; then
        echo "✅ Сервер: Работает"
    else
        echo "❌ Сервер: Не отвечает"
        restart_server
        if check_server; then
            echo "✅ Сервер: Перезапущен"
        else
            echo "❌ Сервер: Не удалось запустить"
        fi
    fi
    echo ""
    
    # Обучение
    sessions=$(curl -s http://localhost:8000/api/training/alphazero/sessions 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$sessions" ]; then
        # Парсим JSON простым способом
        if echo "$sessions" | grep -q '"sessions"'; then
            status=$(echo "$sessions" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
            current_iter=$(echo "$sessions" | grep -o '"current_iteration":[0-9]*' | head -1 | cut -d':' -f2)
            total_iter=$(echo "$sessions" | grep -o '"total_iterations":[0-9]*' | head -1 | cut -d':' -f2)
            progress=$(echo "$sessions" | grep -o '"progress":[0-9.]*' | head -1 | cut -d':' -f2)
            
            echo "📈 Обучение:"
            echo "   Статус: $status"
            echo "   Итерация: $current_iter/$total_iter"
            if [ -n "$progress" ]; then
                printf "   Прогресс: %.1f%%\n" "$progress"
            fi
        else
            echo "ℹ️  Обучение: Не запущено"
        fi
    else
        echo "⚠️  Обучение: Не удалось проверить"
    fi
    echo ""
    
    # GPU
    echo "🎮 GPU (первые 3):"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -3 | while IFS=',' read -r idx util mem_used mem_total; do
        echo "   GPU $idx: Util ${util} | Mem ${mem_used}/${mem_total}"
    done
    echo ""
}

# Основной цикл
echo "🚀 Запуск автоматического мониторинга..."
echo "Нажми Ctrl+C для остановки"
echo ""

# Проверяем и запускаем сервер если нужно
if ! check_server; then
    restart_server
fi

# Мониторинг
while true; do
    show_status
    echo "Обновление через 5 секунд..."
    sleep 5
done
