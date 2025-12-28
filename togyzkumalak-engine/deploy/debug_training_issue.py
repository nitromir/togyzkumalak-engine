#!/usr/bin/env python3
"""
Детальная диагностика проблемы с обучением
"""

import requests
import os
import glob
import json
from datetime import datetime

print("=" * 70)
print("  ДИАГНОСТИКА ПРОБЛЕМЫ С ОБУЧЕНИЕМ")
print("=" * 70)
print()

# 1. Проверка активных сессий
print("1️⃣ Проверяю активные сессии...")
try:
    response = requests.get('http://localhost:8000/api/training/alphazero/sessions', timeout=5)
    sessions = response.json().get('sessions', {})
    
    if sessions:
        print(f"   ✅ Найдено активных сессий: {len(sessions)}")
        for task_id, session in sessions.items():
            print(f"      Task ID: {task_id[:8]}...")
            print(f"      Статус: {session.get('status', 'unknown')}")
            print(f"      Итерация: {session.get('current_iteration', 0)}/{session.get('total_iterations', 0)}")
    else:
        print("   ⚠️ Нет активных сессий")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

print()

# 2. Проверка чекпойнтов детально
print("2️⃣ Анализ чекпойнтов...")
checkpoints_dir = '/workspace/togyzkumalak/togyzkumalak-engine/models/alphazero'

if os.path.exists(checkpoints_dir):
    pattern = os.path.join(checkpoints_dir, "*.pth.tar")
    checkpoints = glob.glob(pattern)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    print(f"   Всего файлов: {len(checkpoints)}")
    print()
    
    # Ищем checkpoint_* файлы
    numbered_checkpoints = [cp for cp in checkpoints if 'checkpoint_' in os.path.basename(cp)]
    if numbered_checkpoints:
        # Извлекаем номера
        numbers = []
        for cp in numbered_checkpoints:
            name = os.path.basename(cp)
            try:
                num = int(name.replace('checkpoint_', '').replace('.pth.tar', ''))
                numbers.append(num)
            except:
                pass
        
        if numbers:
            max_iter = max(numbers)
            print(f"   📊 Максимальная итерация в чекпойнтах: {max_iter}")
            print(f"   Всего пронумерованных чекпойнтов: {len(numbers)}")
            print()
            print("   Последние 10 чекпойнтов:")
            for cp in numbered_checkpoints[:10]:
                name = os.path.basename(cp)
                mtime = datetime.fromtimestamp(os.path.getmtime(cp))
                size = os.path.getsize(cp) / (1024 * 1024)
                print(f"      {name} - {size:.2f} MB - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

print()

# 3. Проверка метрик файла
print("3️⃣ Проверяю файл метрик...")
metrics_file = os.path.join(checkpoints_dir, 'metrics.json')
if os.path.exists(metrics_file):
    try:
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        if 'iterations' in metrics_data and metrics_data['iterations']:
            iterations = metrics_data['iterations']
            print(f"   ✅ Найдено записей в metrics.json: {len(iterations)}")
            print()
            
            if iterations:
                latest = iterations[-1]
                print(f"   Последняя запись:")
                print(f"      Итерация: {latest.get('iteration', 'N/A')}")
                print(f"      Policy Loss: {latest.get('policy_loss', 0):.4f}")
                print(f"      Value Loss: {latest.get('value_loss', 0):.4f}")
                print(f"      Win Rate: {latest.get('win_rate', 0)*100:.1f}%")
        else:
            print("   ⚠️ Файл метрик пуст или не содержит данных")
    except Exception as e:
        print(f"   ❌ Ошибка чтения метрик: {e}")
else:
    print("   ⚠️ Файл metrics.json не найден")

print()

# 4. Проверка логов сервера
print("4️⃣ Проверяю логи сервера (последние 30 строк)...")
log_file = '/workspace/togyzkumalak/togyzkumalak-engine/server_error.log'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if lines:
            print("   Последние строки с ошибками:")
            # Ищем строки с ERROR, Exception, Traceback
            error_lines = [l for l in lines if any(x in l.upper() for x in ['ERROR', 'EXCEPTION', 'TRACEBACK', 'FAILED'])]
            if error_lines:
                for line in error_lines[-10:]:
                    print(f"      {line.rstrip()}")
            else:
                print("   Ошибок в логах не найдено")
                print("   Последние 10 строк:")
                for line in lines[-10:]:
                    print(f"      {line.rstrip()}")
        else:
            print("   Лог пуст")
else:
    print("   Файл логов не найден")

print()

# 5. Проверка логов обучения (если есть)
print("5️⃣ Проверяю логи обучения...")
training_log = os.path.join(checkpoints_dir, 'training.log')
if os.path.exists(training_log):
    with open(training_log, 'r') as f:
        lines = f.readlines()
        if lines:
            print("   Последние 15 строк:")
            for line in lines[-15:]:
                print(f"      {line.rstrip()}")
else:
    print("   Файл training.log не найден")

print()
print("=" * 70)
print()
print("💡 ВЫВОДЫ:")
print()
print("Если обучение остановилось на 8 итерации:")
print("  1. Проверь логи выше на наличие ошибок")
print("  2. Возможно обучение упало из-за ошибки")
print("  3. Или было остановлено вручную")
print()
print("Если метрики не обновляются:")
print("  1. Проверь что файл metrics.json существует и обновляется")
print("  2. Возможно проблема с сохранением метрик")
print()
print("РЕКОМЕНДАЦИИ:")
print("  1. Перезапусти обучение через UI")
print("  2. Следи за логами в реальном времени")
print("  3. Проверь что есть достаточно места на диске")
print()
print("=" * 70)
