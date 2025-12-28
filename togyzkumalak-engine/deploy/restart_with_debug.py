#!/usr/bin/env python3
"""
Перезапуск сервера с полной диагностикой
"""

import os
import subprocess
import sys
import time
import requests

print("=" * 60)
print("  ДИАГНОСТИКА И ПЕРЕЗАПУСК СЕРВЕРА")
print("=" * 60)
print()

# Шаг 1: Проверка текущего состояния
print("1️⃣ Проверяю текущий сервер...")
try:
    response = requests.get('http://localhost:8000/api/health', timeout=2)
    print(f"   ✅ Сервер работает! Статус: {response.status_code}")
    print(f"   Ответ: {response.json()}")
except Exception as e:
    print(f"   ⚠️ Сервер не отвечает: {e}")

print()

# Шаг 2: Поиск процессов
print("2️⃣ Ищу запущенные процессы сервера...")
try:
    result = subprocess.run(
        ['ps', 'aux'],
        capture_output=True,
        text=True
    )
    processes = [line for line in result.stdout.split('\n') if 'run.py' in line or 'python.*run' in line]
    if processes:
        print(f"   Найдено процессов: {len(processes)}")
        for p in processes[:3]:
            print(f"   {p[:80]}")
    else:
        print("   Процессы не найдены")
except Exception as e:
    print(f"   Ошибка поиска: {e}")

print()

# Шаг 3: Остановка старых процессов
print("3️⃣ Останавливаю старые процессы...")
try:
    # Попробуем разные способы
    subprocess.run(['pkill', '-f', 'python.*run.py'], capture_output=True, timeout=5)
    subprocess.run(['pkill', '-f', 'run.py'], capture_output=True, timeout=5)
    time.sleep(2)
    print("   ✅ Команды остановки выполнены")
except Exception as e:
    print(f"   ⚠️ Ошибка остановки: {e}")

print()

# Шаг 4: Проверка директорий
print("4️⃣ Проверяю директории...")
base_dir = '/workspace/togyzkumalak'
engine_dir = '/workspace/togyzkumalak/togyzkumalak-engine'

print(f"   Базовая директория: {base_dir}")
print(f"   Существует: {os.path.exists(base_dir)}")

print(f"   Директория движка: {engine_dir}")
print(f"   Существует: {os.path.exists(engine_dir)}")

if os.path.exists(engine_dir):
    os.chdir(engine_dir)
    print(f"   Текущая директория: {os.getcwd()}")
    print(f"   Файл run.py существует: {os.path.exists('run.py')}")
else:
    # Попробуем найти
    print("   ⚠️ Стандартный путь не найден, ищу...")
    for possible in ['/workspace/togyzkumalak', '/root/togyzkumalak', os.getcwd()]:
        if os.path.exists(possible):
            test_path = os.path.join(possible, 'togyzkumalak-engine', 'run.py')
            if os.path.exists(test_path):
                engine_dir = os.path.dirname(test_path)
                print(f"   ✅ Найден путь: {engine_dir}")
                os.chdir(engine_dir)
                break

print()

# Шаг 5: Обновление кода
print("5️⃣ Обновляю код с GitHub...")
try:
    # Переходим в корень репозитория
    repo_root = os.path.dirname(engine_dir) if os.path.exists(engine_dir) else base_dir
    os.chdir(repo_root)
    print(f"   Репозиторий: {os.getcwd()}")
    print(f"   .git существует: {os.path.exists('.git')}")
    
    if os.path.exists('.git'):
        result = subprocess.run(
            ['git', 'pull', 'origin', 'master'],
            capture_output=True,
            text=True,
            timeout=60
        )
        print(f"   Код возврата: {result.returncode}")
        if result.stdout:
            print(f"   Вывод: {result.stdout[:200]}")
        if result.stderr:
            print(f"   Ошибки: {result.stderr[:200]}")
    else:
        print("   ⚠️ .git не найден, пропускаю обновление")
except Exception as e:
    print(f"   ❌ Ошибка обновления: {e}")

print()

# Шаг 6: Запуск сервера
print("6️⃣ Запускаю новый сервер...")
try:
    os.chdir(engine_dir)
    python_exe = sys.executable
    print(f"   Python: {python_exe}")
    print(f"   Директория: {os.getcwd()}")
    
    # Запускаем в фоне
    process = subprocess.Popen(
        [python_exe, 'run.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=engine_dir,
        env=os.environ.copy()
    )
    
    print(f"   ✅ Процесс запущен! PID: {process.pid}")
    
except Exception as e:
    print(f"   ❌ Ошибка запуска: {e}")
    import traceback
    traceback.print_exc()

print()

# Шаг 7: Проверка запуска
print("7️⃣ Проверяю что сервер запустился...")
for i in range(10):
    time.sleep(1)
    try:
        response = requests.get('http://localhost:8000/api/health', timeout=2)
        if response.status_code == 200:
            print(f"   ✅ Сервер работает! Ответ: {response.json()}")
            print()
            print("=" * 60)
            print("  ✅ СЕРВЕР УСПЕШНО ЗАПУЩЕН!")
            print("=" * 60)
            print()
            print("  URL: http://localhost:8000")
            print("  PID:", process.pid if 'process' in locals() else "неизвестен")
            print()
            print("  💡 Обнови страницу в браузере (F5)")
            print()
            break
    except Exception as e:
        if i < 9:
            print(f"   ⏳ Ожидание... ({i+1}/10)")
        else:
            print(f"   ⚠️ Сервер не отвечает после 10 секунд")
            print(f"   Ошибка: {e}")
            print()
            print("   Проверь вручную:")
            print(f"   cd {engine_dir}")
            print(f"   {python_exe} run.py")

print()
print("=" * 60)
