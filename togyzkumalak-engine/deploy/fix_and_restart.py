#!/usr/bin/env python3
"""
Исправление проблемы с портом и перезапуск
"""

import os
import subprocess
import sys
import time
import requests
import signal

print("=" * 60)
print("  ИСПРАВЛЕНИЕ И ПЕРЕЗАПУСК")
print("=" * 60)
print()

# Шаг 1: Убить ВСЕ процессы
print("1️⃣ Останавливаю ВСЕ процессы сервера...")
try:
    # Убиваем конкретные PID
    for pid in [10100, 19861]:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"   Отправлен SIGTERM процессу {pid}")
        except:
            pass
    
    # Убиваем все процессы с run.py
    subprocess.run(['pkill', '-9', '-f', 'python.*run.py'], capture_output=True, timeout=5)
    subprocess.run(['pkill', '-9', '-f', 'run.py'], capture_output=True, timeout=5)
    
    time.sleep(3)
    print("   ✅ Все процессы остановлены")
except Exception as e:
    print(f"   ⚠️ Ошибка: {e}")

print()

# Шаг 2: Проверка что порт свободен
print("2️⃣ Проверяю что порт 8000 свободен...")
for i in range(5):
    try:
        result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True)
        port_lines = [line for line in result.stdout.split('\n') if ':8000' in line and 'LISTEN' in line]
        if not port_lines:
            print("   ✅ Порт 8000 свободен")
            break
        else:
            print(f"   ⏳ Порт еще занят, жду... ({i+1}/5)")
            time.sleep(2)
    except:
        pass
else:
    print("   ⚠️ Порт все еще занят, но продолжаю...")

print()

# Шаг 3: Переход в директорию
print("3️⃣ Перехожу в директорию проекта...")
engine_dir = '/workspace/togyzkumalak/togyzkumalak-engine'
os.chdir(engine_dir)
print(f"   Директория: {os.getcwd()}")
print(f"   run.py существует: {os.path.exists('run.py')}")

print()

# Шаг 4: Обновление кода
print("4️⃣ Обновляю код...")
try:
    repo_root = os.path.dirname(engine_dir)
    os.chdir(repo_root)
    result = subprocess.run(
        ['git', 'pull', 'origin', 'master'],
        capture_output=True,
        text=True,
        timeout=60
    )
    if result.returncode == 0:
        print("   ✅ Код обновлен")
    else:
        print(f"   ⚠️ Проблемы с git pull: {result.stderr[:200]}")
except Exception as e:
    print(f"   ⚠️ Ошибка обновления: {e}")

print()

# Шаг 5: Запуск сервера
print("5️⃣ Запускаю новый сервер...")
os.chdir(engine_dir)
python_exe = sys.executable

try:
    # Запускаем с перенаправлением вывода
    process = subprocess.Popen(
        [python_exe, 'run.py'],
        stdout=open('server.log', 'w'),
        stderr=open('server_error.log', 'w'),
        cwd=engine_dir,
        env=os.environ.copy()
    )
    
    print(f"   ✅ Процесс запущен! PID: {process.pid}")
    print(f"   Логи: server.log и server_error.log")
    
except Exception as e:
    print(f"   ❌ Ошибка запуска: {e}")
    import traceback
    traceback.print_exc()

print()

# Шаг 6: Ожидание и проверка
print("6️⃣ Ожидаю запуска сервера...")
for i in range(15):
    time.sleep(1)
    try:
        response = requests.get('http://localhost:8000/api/health', timeout=3)
        if response.status_code == 200:
            print(f"   ✅ Сервер работает! Ответ: {response.json()}")
            print()
            print("=" * 60)
            print("  ✅ СЕРВЕР УСПЕШНО ЗАПУЩЕН!")
            print("=" * 60)
            print()
            print("  URL: http://localhost:8000")
            print("  PID:", process.pid)
            print()
            print("  💡 Обнови страницу в браузере (F5)")
            print()
            break
    except requests.exceptions.ConnectionError:
        if i < 14:
            print(f"   ⏳ Ожидание... ({i+1}/15)")
        else:
            print("   ⚠️ Сервер не отвечает")
            print()
            print("   Проверь логи:")
            print("   tail -20 /workspace/togyzkumalak/togyzkumalak-engine/server_error.log")
    except Exception as e:
        if i < 14:
            print(f"   ⏳ Ожидание... ({i+1}/15) - {type(e).__name__}")
        else:
            print(f"   ❌ Ошибка: {e}")

print()
print("=" * 60)
