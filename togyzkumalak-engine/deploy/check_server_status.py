#!/usr/bin/env python3
"""
Проверка статуса сервера и логов
"""

import subprocess
import os
import requests
import time

print("=" * 60)
print("  ПРОВЕРКА СТАТУСА СЕРВЕРА")
print("=" * 60)
print()

# 1. Проверка процессов
print("1️⃣ Проверяю процессы...")
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
processes = [line for line in result.stdout.split('\n') if 'run.py' in line or 'python' in line and 'run' in line]
print(f"   Найдено процессов: {len(processes)}")
for p in processes:
    if 'run.py' in p:
        print(f"   {p}")

print()

# 2. Проверка порта
print("2️⃣ Проверяю порт 8000...")
result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True)
port_lines = [line for line in result.stdout.split('\n') if ':8000' in line]
if port_lines:
    print("   Порт 8000 занят:")
    for line in port_lines:
        print(f"   {line}")
else:
    print("   ⚠️ Порт 8000 не слушается")

print()

# 3. Попытка подключения
print("3️⃣ Пробую подключиться к серверу...")
for i in range(5):
    try:
        response = requests.get('http://localhost:8000/api/health', timeout=3)
        print(f"   ✅ Сервер отвечает! Статус: {response.status_code}")
        print(f"   Ответ: {response.json()}")
        break
    except requests.exceptions.ConnectionError:
        print(f"   ⏳ Ожидание... ({i+1}/5)")
        time.sleep(2)
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        break

print()

# 4. Проверка логов (если есть)
print("4️⃣ Проверяю логи...")
log_files = ['server.log', 'nohup.out', 'run.log']
for log_file in log_files:
    if os.path.exists(log_file):
        print(f"   Найден файл: {log_file}")
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"   Последние 10 строк:")
                for line in lines[-10:]:
                    print(f"   {line.rstrip()}")
        except:
            pass

print()

# 5. Попытка запустить вручную и увидеть ошибки
print("5️⃣ Проверяю можно ли запустить сервер...")
engine_dir = '/workspace/togyzkumalak/togyzkumalak-engine'
if os.path.exists(engine_dir):
    os.chdir(engine_dir)
    print(f"   Директория: {os.getcwd()}")
    print(f"   run.py существует: {os.path.exists('run.py')}")
    
    # Попробуем импортировать модуль чтобы увидеть ошибки
    print("   Пробую импортировать модули...")
    import sys
    sys.path.insert(0, engine_dir)
    try:
        # Просто проверяем что файлы есть
        backend_dir = os.path.join(engine_dir, 'backend')
        print(f"   backend/ существует: {os.path.exists(backend_dir)}")
        main_py = os.path.join(backend_dir, 'main.py')
        print(f"   backend/main.py существует: {os.path.exists(main_py)}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

print()
print("=" * 60)
print("  РЕКОМЕНДАЦИИ:")
print("=" * 60)
print()
print("Если сервер не запускается, попробуй:")
print("1. Убить все процессы: pkill -f 'python.*run.py'")
print("2. Запустить вручную и увидеть ошибки:")
print("   cd /workspace/togyzkumalak/togyzkumalak-engine")
print("   /venv/main/bin/python run.py")
print()
print("Или запустить в фоне с выводом:")
print("   cd /workspace/togyzkumalak/togyzkumalak-engine")
print("   nohup /venv/main/bin/python run.py > server.log 2>&1 &")
print("   tail -f server.log")
print()
