#!/usr/bin/env python3
"""
Получить все логи обучения для диагностики
"""

import os
import requests
import subprocess
import json
from datetime import datetime

print("=" * 80)
print("  ПОЛНЫЕ ЛОГИ ОБУЧЕНИЯ ALPHAZERO")
print("=" * 80)
print()

# 1. Статус через API
print("1️⃣ СТАТУС ЧЕРЕЗ API:")
print("-" * 80)
try:
    response = requests.get('http://localhost:8000/api/training/alphazero/sessions')
    sessions = response.json()
    
    for task_id, status in sessions.get('sessions', {}).items():
        print(f"Task ID: {task_id}")
        print(f"Статус: {status.get('status', 'unknown')}")
        print(f"Итерация: {status.get('current_iteration', 0)} / {status.get('total_iterations', 0)}")
        print(f"Прогресс: {status.get('progress', 0):.1f}%")
        
        detail_response = requests.get(f'http://localhost:8000/api/training/alphazero/sessions/{task_id}')
        if detail_response.status_code == 200:
            detail = detail_response.json()
            print(f"Игр: {detail.get('games_completed', 0)} / {detail.get('total_games', 0)}")
            print(f"Примеров: {detail.get('examples_collected', 0)}")
            print(f"Loss: {detail.get('current_loss', 'N/A')}")
            print(f"Время: {detail.get('elapsed_time', 0):.0f} сек")
            print(f"Этап: {detail.get('current_phase', 'unknown')}")
except Exception as e:
    print(f"Ошибка: {e}")

print()

# 2. Все логи сервера с AlphaZero
print("2️⃣ ЛОГИ СЕРВЕРА (ВСЕ С ALPHAZERO):")
print("-" * 80)
server_error_log = '/workspace/togyzkumalak/togyzkumalak-engine/server_error.log'
server_log = '/workspace/togyzkumalak/togyzkumalak-engine/server.log'

if os.path.exists(server_error_log):
    print("ОШИБКИ (server_error.log - последние 100 строк):")
    with open(server_error_log, 'r') as f:
        lines = f.readlines()
        for line in lines[-100:]:
            if 'alphazero' in line.lower() or 'training' in line.lower() or 'error' in line.lower() or 'iteration' in line.lower() or 'episode' in line.lower() or 'mcts' in line.lower():
                print(f"  {line.rstrip()}")

print()

if os.path.exists(server_log):
    print("ВЫВОД (server.log - последние 100 строк с AlphaZero):")
    with open(server_log, 'r') as f:
        lines = f.readlines()
        relevant_lines = [l for l in lines if any(keyword in l.lower() for keyword in ['alphazero', 'iteration', 'self-play', 'training', 'mcts', 'episode', 'error', 'warning'])]
        for line in relevant_lines[-100:]:
            print(f"  {line.rstrip()}")

print()

# 3. Процессы Python
print("3️⃣ ПРОЦЕССЫ PYTHON:")
print("-" * 80)
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
processes = [p for p in result.stdout.split('\n') if 'python' in p.lower() and ('run.py' in p or 'alphazero' in p.lower())]
if processes:
    for p in processes:
        print(f"  {p}")
else:
    print("  Процессы не найдены")

print()

# 4. Метрики и чекпоинты
print("4️⃣ МЕТРИКИ И ЧЕКПОИНТЫ:")
print("-" * 80)
engine_dir = '/workspace/togyzkumalak/togyzkumalak-engine'
metrics_file = os.path.join(engine_dir, 'models', 'alphazero', 'metrics.json')
training_metrics_file = os.path.join(engine_dir, 'models', 'alphazero', 'training_metrics.json')

if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
        print("metrics.json:")
        if 'history' in metrics and len(metrics['history']) > 0:
            print(f"  История: {len(metrics['history'])} записей")
            last = metrics['history'][-1]
            print(f"  Последняя: iter {last.get('iteration', 'N/A')}, loss {last.get('policy_loss', 'N/A')}")
        else:
            print("  История пуста")

if os.path.exists(training_metrics_file):
    with open(training_metrics_file, 'r') as f:
        training_metrics = json.load(f)
        print("training_metrics.json:")
        if 'metrics' in training_metrics and len(training_metrics['metrics']) > 0:
            print(f"  Метрики: {len(training_metrics['metrics'])} записей")
        else:
            print("  Метрики пусты")

checkpoints_dir = os.path.join(engine_dir, 'models', 'alphazero')
if os.path.exists(checkpoints_dir):
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth.tar')]
    print(f"Чекпоинтов: {len(checkpoints)}")

print()

# 5. Проверка GPU
print("5️⃣ GPU СТАТУС:")
print("-" * 80)
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("GPU использование:")
        for line in result.stdout.strip().split('\n')[:5]:
            print(f"  {line}")
    else:
        print("  nvidia-smi не доступен")
except:
    print("  Не удалось проверить GPU")

print()
print("=" * 80)
