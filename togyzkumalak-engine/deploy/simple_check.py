#!/usr/bin/env python3
"""
Простая проверка статуса обучения
"""

import requests
import subprocess

print("Начинаю проверку...")

# 1. Проверка сервера
try:
    r = requests.get('http://localhost:8000/api/health', timeout=2)
    print(f"✅ Сервер работает: {r.json()}")
except Exception as e:
    print(f"❌ Сервер не работает: {e}")
    exit(1)

# 2. Проверка сессий
try:
    r = requests.get('http://localhost:8000/api/training/alphazero/sessions', timeout=5)
    sessions = r.json().get('sessions', {})
    print(f"Сессий: {len(sessions)}")
    
    if sessions:
        task_id = list(sessions.keys())[0]
        status_r = requests.get(f'http://localhost:8000/api/training/alphazero/sessions/{task_id}', timeout=5)
        status = status_r.json()
        print(f"Статус: {status.get('status')}")
        print(f"Итерация: {status.get('current_iteration', 0)}/{status.get('total_iterations', 0)}")
except Exception as e:
    print(f"Ошибка проверки сессий: {e}")

# 3. Процессы
try:
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
    processes = [p for p in result.stdout.split('\n') if 'run.py' in p or 'alphazero' in p.lower()]
    print(f"Процессов: {len(processes)}")
    if processes:
        for p in processes[:2]:
            print(f"  {p[:80]}")
except Exception as e:
    print(f"Ошибка проверки процессов: {e}")

# 4. GPU
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        utils = [x.strip() for x in result.stdout.strip().split('\n') if x.strip()]
        active_gpus = [u for u in utils if int(u) > 0]
        print(f"GPU активных: {len(active_gpus)}/{len(utils)}")
        if active_gpus:
            print(f"  Загрузка: {', '.join(active_gpus[:5])}%")
except Exception as e:
    print(f"Ошибка проверки GPU: {e}")

print("Проверка завершена")
