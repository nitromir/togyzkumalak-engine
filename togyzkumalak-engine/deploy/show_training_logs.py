#!/usr/bin/env python3
"""
Показать логи обучения AlphaZero
"""

import os
import subprocess
import requests
import json
from datetime import datetime

print("=" * 60)
print("  ЛОГИ ОБУЧЕНИЯ ALPHAZERO")
print("=" * 60)
print()

# 1. Проверка статуса через API
print("1️⃣ Статус через API:")
print("-" * 60)
try:
    response = requests.get('http://localhost:8000/api/training/alphazero/sessions')
    sessions = response.json()
    
    for task_id, status in sessions.get('sessions', {}).items():
        print(f"   Task ID: {task_id}")
        print(f"   Статус: {status.get('status', 'unknown')}")
        print(f"   Итерация: {status.get('current_iteration', 0)} / {status.get('total_iterations', 0)}")
        print(f"   Прогресс: {status.get('progress', 0):.1f}%")
        
        # Детальная информация
        detail_response = requests.get(f'http://localhost:8000/api/training/alphazero/sessions/{task_id}')
        if detail_response.status_code == 200:
            detail = detail_response.json()
            print(f"   Игр завершено: {detail.get('games_completed', 0)} / {detail.get('total_games', 0)}")
            print(f"   Примеров собрано: {detail.get('examples_collected', 0)}")
            print(f"   Loss: {detail.get('current_loss', 'N/A')}")
            print(f"   Время: {detail.get('elapsed_time', 0):.0f} сек")
except Exception as e:
    print(f"   ⚠️ Ошибка получения статуса: {e}")

print()

# 2. Логи сервера
print("2️⃣ Логи сервера (последние 30 строк):")
print("-" * 60)
server_log = '/workspace/togyzkumalak/togyzkumalak-engine/server.log'
server_error_log = '/workspace/togyzkumalak/togyzkumalak-engine/server_error.log'

if os.path.exists(server_error_log):
    print("   ОШИБКИ (server_error.log):")
    try:
        with open(server_error_log, 'r') as f:
            lines = f.readlines()
            for line in lines[-30:]:
                if 'alphazero' in line.lower() or 'training' in line.lower() or 'error' in line.lower() or 'iteration' in line.lower():
                    print(f"   {line.rstrip()}")
    except Exception as e:
        print(f"   Ошибка чтения: {e}")
else:
    print("   ⚠️ Файл server_error.log не найден")

print()

if os.path.exists(server_log):
    print("   ВЫВОД (server.log - последние 30 строк с AlphaZero):")
    try:
        with open(server_log, 'r') as f:
            lines = f.readlines()
            # Фильтруем только строки связанные с AlphaZero
            relevant_lines = [l for l in lines if 'alphazero' in l.lower() or 'iteration' in l.lower() or 'self-play' in l.lower() or 'training' in l.lower() or 'mcts' in l.lower()]
            for line in relevant_lines[-30:]:
                print(f"   {line.rstrip()}")
    except Exception as e:
        print(f"   Ошибка чтения: {e}")
else:
    print("   ⚠️ Файл server.log не найден")

print()

# 3. Метрики обучения
print("3️⃣ Метрики обучения:")
print("-" * 60)
engine_dir = '/workspace/togyzkumalak/togyzkumalak-engine'
metrics_file = os.path.join(engine_dir, 'models', 'alphazero', 'metrics.json')
training_metrics_file = os.path.join(engine_dir, 'models', 'alphazero', 'training_metrics.json')

if os.path.exists(metrics_file):
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            print("   📊 metrics.json:")
            if 'history' in metrics and len(metrics['history']) > 0:
                last = metrics['history'][-1]
                print(f"      Последняя итерация: {last.get('iteration', 'N/A')}")
                print(f"      Policy Loss: {last.get('policy_loss', 'N/A')}")
                print(f"      Value Loss: {last.get('value_loss', 'N/A')}")
                print(f"      Win Rate: {last.get('win_rate', 'N/A')}")
                print(f"      Время итерации: {last.get('iteration_time_sec', 'N/A'):.1f} сек")
            else:
                print("      Метрики еще не собраны")
    except Exception as e:
        print(f"   Ошибка чтения metrics.json: {e}")

if os.path.exists(training_metrics_file):
    try:
        with open(training_metrics_file, 'r') as f:
            training_metrics = json.load(f)
            print("   📈 training_metrics.json:")
            if 'metrics' in training_metrics and len(training_metrics['metrics']) > 0:
                last = training_metrics['metrics'][-1]
                print(f"      Итерация: {last.get('iteration', 'N/A')}")
                print(f"      Примеров: {last.get('total_examples', 'N/A')}")
                print(f"      Время: {last.get('iteration_time_sec', 'N/A'):.1f} сек")
    except Exception as e:
        print(f"   Ошибка чтения training_metrics.json: {e}")

print()

# 4. Проверка процессов
print("4️⃣ Проверка процессов:")
print("-" * 60)
try:
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    processes = [p for p in result.stdout.split('\n') if 'run.py' in p or 'python' in p and 'alphazero' in p.lower()]
    if processes:
        print("   Найденные процессы:")
        for p in processes[:5]:
            print(f"   {p[:100]}")
    else:
        print("   ⚠️ Процессы не найдены")
except Exception as e:
    print(f"   Ошибка: {e}")

print()

# 5. Чекпоинты
print("5️⃣ Чекпоинты:")
print("-" * 60)
checkpoints_dir = os.path.join(engine_dir, 'models', 'alphazero')
if os.path.exists(checkpoints_dir):
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth.tar')]
    print(f"   Найдено чекпоинтов: {len(checkpoints)}")
    if checkpoints:
        print("   Последние 5:")
        for cp in sorted(checkpoints)[-5:]:
            cp_path = os.path.join(checkpoints_dir, cp)
            size = os.path.getsize(cp_path) / (1024 * 1024)  # MB
            mtime = datetime.fromtimestamp(os.path.getmtime(cp_path))
            print(f"      {cp} ({size:.1f} MB, {mtime.strftime('%H:%M:%S')})")
else:
    print("   ⚠️ Директория чекпоинтов не найдена")

print()
print("=" * 60)
print("  ВЫВОД:")
print("=" * 60)
print()
print("✅ Если видишь:")
print("   - Статус: 'running'")
print("   - Итерация увеличивается")
print("   - Игр завершено увеличивается")
print("   - Процесс Python запущен")
print()
print("   → Обучение идет нормально!")
print()
print("❌ Если видишь:")
print("   - Статус: 'error' или 'stopped'")
print("   - Процесс не найден")
print("   - Ошибки в логах")
print()
print("   → Обучение упало, нужно перезапустить")
print()
