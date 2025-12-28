#!/usr/bin/env python3
"""
Обновление кода и перезапуск сервера
"""

import os
import subprocess
import sys
import time
import requests
import signal

print("=" * 60)
print("  ОБНОВЛЕНИЕ И ПЕРЕЗАПУСК")
print("=" * 60)
print()

# 1. Остановка старого сервера
print("1️⃣ Останавливаю старый сервер...")
try:
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    processes = [line for line in result.stdout.split('\n') if 'run.py' in line]
    for p in processes:
        parts = p.split()
        if len(parts) > 1:
            try:
                pid = int(parts[1])
                os.kill(pid, signal.SIGTERM)
                print(f"   Остановлен процесс {pid}")
            except:
                pass
    subprocess.run(['pkill', '-9', '-f', 'run.py'], capture_output=True)
    time.sleep(3)
    print("   ✅ Процессы остановлены")
except Exception as e:
    print(f"   ⚠️ {e}")

print()

# 2. Обновление кода
print("2️⃣ Обновляю код с GitHub...")
repo_path = '/workspace/togyzkumalak'
os.chdir(repo_path)

result = subprocess.run(
    ['git', 'pull', 'origin', 'master'],
    capture_output=True,
    text=True,
    timeout=60
)

print(f"   Код возврата: {result.returncode}")
if result.stdout:
    print(f"   Вывод: {result.stdout[:300]}")
if result.stderr and result.returncode != 0:
    print(f"   Ошибки: {result.stderr[:300]}")

if result.returncode == 0:
    print("   ✅ Код обновлен")
else:
    print("   ⚠️ Проблемы с обновлением, но продолжаю...")

print()

# 3. Запуск нового сервера
print("3️⃣ Запускаю новый сервер...")
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

# 4. Проверка
print("4️⃣ Проверяю сервер...")
for i in range(15):
    time.sleep(1)
    try:
        r = requests.get('http://localhost:8000/api/health', timeout=3)
        print(f"   ✅ Сервер работает! {r.json()}")
        print()
        print("=" * 60)
        print("  ✅ ГОТОВО!")
        print("=" * 60)
        print()
        print("  Обнови страницу в браузере (F5)")
        print()
        break
    except:
        if i < 14:
            print(f"   ⏳ Ожидание... ({i+1}/15)")
        else:
            print("   ⚠️ Сервер не отвечает")
            print("   Проверь логи: tail -20 server_error.log")

print()
print("=" * 60)
