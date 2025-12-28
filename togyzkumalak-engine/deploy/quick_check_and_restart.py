#!/usr/bin/env python3
"""
Быстрая проверка и перезапуск сервера
"""

import os
import subprocess
import sys
import time
import requests
import signal

print("=" * 60)
print("  ПРОВЕРКА И ПЕРЕЗАПУСК СЕРВЕРА")
print("=" * 60)
print()

# 1. Проверка сервера
print("1️⃣ Проверяю сервер...")
try:
    r = requests.get('http://localhost:8000/api/health', timeout=2)
    print("   ✅ Сервер работает!")
    print(f"   Ответ: {r.json()}")
    exit(0)
except:
    print("   ❌ Сервер не отвечает")
    print()

# 2. Проверка процессов
print("2️⃣ Проверяю процессы...")
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
processes = [p for p in result.stdout.split('\n') if 'run.py' in p]

if processes:
    print(f"   Найдено процессов: {len(processes)}")
    for p in processes:
        parts = p.split()
        if len(parts) > 1:
            print(f"   PID: {parts[1]} - {p[:80]}")
else:
    print("   Процессы не найдены")

print()

# 3. Остановка старых процессов
print("3️⃣ Останавливаю старые процессы...")
try:
    for p in processes:
        parts = p.split()
        if len(parts) > 1:
            try:
                pid = int(parts[1])
                os.kill(pid, signal.SIGTERM)
                print(f"   Остановлен PID: {pid}")
            except:
                pass
    subprocess.run(['pkill', '-9', '-f', 'run.py'], capture_output=True)
    time.sleep(2)
    print("   ✅ Процессы остановлены")
except Exception as e:
    print(f"   ⚠️ {e}")

print()

# 4. Запуск нового сервера
print("4️⃣ Запускаю новый сервер...")
engine_dir = '/workspace/togyzkumalak/togyzkumalak-engine'
os.chdir(engine_dir)
python_exe = sys.executable

process = subprocess.Popen(
    [python_exe, 'run.py'],
    stdout=open('server.log', 'w'),
    stderr=open('server_error.log', 'w'),
    cwd=engine_dir
)

print(f"   ✅ Процесс запущен! PID: {process.pid}")

print()

# 5. Проверка запуска
print("5️⃣ Проверяю запуск...")
for i in range(15):
    time.sleep(1)
    try:
        r = requests.get('http://localhost:8000/api/health', timeout=3)
        print(f"   ✅ Сервер работает! {r.json()}")
        print()
        print("=" * 60)
        print("  ✅ ГОТОВО!")
        print("=" * 60)
        break
    except:
        if i < 14:
            print(f"   ⏳ Ожидание... ({i+1}/15)")
        else:
            print("   ⚠️ Сервер не отвечает")
            print("   Проверь логи: tail -20 server_error.log")

print()
print("=" * 60)
