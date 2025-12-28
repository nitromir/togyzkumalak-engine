#!/usr/bin/env python3
"""
Тест запуска обучения через API
"""

import requests
import json
import time

print("=" * 70)
print("  ТЕСТ ЗАПУСКА ОБУЧЕНИЯ")
print("=" * 70)
print()

# Конфигурация обучения
config = {
    "numIters": 10,  # Маленькое число для теста
    "numEps": 20,
    "numMCTSSims": 50,
    "cpuct": 1.0,
    "batch_size": 1024,
    "hidden_size": 256,
    "epochs": 5,
    "use_bootstrap": False,  # Отключаем для быстрого теста
    "use_multiprocessing": True,
    "save_every_n_iters": 2
}

print("1️⃣ Отправляю запрос на запуск обучения...")
print(f"   Конфигурация: {json.dumps(config, indent=2)}")
print()

try:
    response = requests.post(
        'http://localhost:8000/api/training/alphazero/start',
        json=config,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        task_id = data.get('task_id')
        print(f"   ✅ Обучение запущено!")
        print(f"   Task ID: {task_id}")
        print()
        
        # Ждем немного и проверяем статус
        print("2️⃣ Проверяю статус через 5 секунд...")
        time.sleep(5)
        
        status_response = requests.get(
            f'http://localhost:8000/api/training/alphazero/sessions/{task_id}',
            timeout=5
        )
        
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"   Статус: {status.get('status', 'unknown')}")
            print(f"   Итерация: {status.get('current_iteration', 0)}/{status.get('total_iterations', 0)}")
            print(f"   Прогресс: {status.get('progress', 0):.1f}%")
        else:
            print(f"   ⚠️ Не удалось получить статус: {status_response.status_code}")
    else:
        print(f"   ❌ Ошибка запуска: {response.status_code}")
        print(f"   Ответ: {response.text}")
        
except requests.exceptions.Timeout:
    print("   ⚠️ Таймаут при запуске (возможно обучение запускается)")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
