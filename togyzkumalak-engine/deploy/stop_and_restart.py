#!/usr/bin/env python3
"""
Остановить обучение и перезапустить сервер с новым кодом
"""

import os
import subprocess
import sys
import time
import requests
import signal

print("=" * 60)
print("  ОСТАНОВКА И ПЕРЕЗАПУСК")
print("=" * 60)
print()

# 1. Остановить обучение
print("1️⃣ Останавливаю обучение...")
try:
    response = requests.get('http://localhost:8000/api/training/alphazero/sessions')
    sessions = response.json()
    
    for task_id in sessions.get('sessions', {}).keys():
        try:
            requests.post('http://localhost:8000/api/training/alphazero/stop', 
                         json={'task_id': task_id})
            print(f"   ✅ Остановлен: {task_id}")
        except:
            print(f"   ⚠️ Не удалось остановить: {task_id}")
except Exception as e:
    print(f"   ⚠️ Ошибка: {e}")

print()

# 2. Остановить сервер
print("2️⃣ Останавливаю сервер...")
subprocess.run(['pkill', '-9', '-f', 'python.*run.py'], capture_output=True)
time.sleep(3)

print()

# 3. Обновить код
print("3️⃣ Обновляю код...")
os.chdir('/workspace/togyzkumalak')
result = subprocess.run(['git', 'pull', 'origin', 'master'], capture_output=True, text=True)
print(f"   {result.stdout[:200]}")

print()

# 4. Запустить сервер
print("4️⃣ Запускаю сервер...")
os.chdir('/workspace/togyzkumalak/togyzkumalak-engine')
python_exe = sys.executable

process = subprocess.Popen(
    [python_exe, 'run.py'],
    stdout=open('server.log', 'w'),
    stderr=open('server_error.log', 'w'),
    cwd='/workspace/togyzkumalak/togyzkumalak-engine'
)

print(f"   ✅ Сервер запущен! PID: {process.pid}")

print()

# 5. Проверка
print("5️⃣ Проверяю сервер...")
for i in range(10):
    time.sleep(1)
    try:
        r = requests.get('http://localhost:8000/api/health', timeout=2)
        print(f"   ✅ Сервер работает! {r.json()}")
        print()
        print("=" * 60)
        print("  ✅ ГОТОВО!")
        print("=" * 60)
        print()
        print("  Теперь запусти обучение заново через UI")
        print()
        break
    except:
        if i < 9:
            print(f"   ⏳ Ожидание... ({i+1}/10)")
        else:
            print("   ⚠️ Сервер не отвечает")
