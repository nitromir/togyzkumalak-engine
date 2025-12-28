#!/usr/bin/env python3
"""
Проверка логов сервера
"""

import os

log_dir = '/workspace/togyzkumalak/togyzkumalak-engine'
error_log = os.path.join(log_dir, 'server_error.log')
server_log = os.path.join(log_dir, 'server.log')

print("=" * 60)
print("  ЛОГИ СЕРВЕРА")
print("=" * 60)
print()

# Проверка error log
if os.path.exists(error_log):
    print("📋 ОШИБКИ (server_error.log):")
    print("-" * 60)
    with open(error_log, 'r') as f:
        content = f.read()
        if content:
            # Показываем последние 50 строк
            lines = content.split('\n')
            for line in lines[-50:]:
                print(line)
        else:
            print("   (файл пуст)")
    print()
else:
    print("⚠️ Файл server_error.log не найден")
    print()

# Проверка server log
if os.path.exists(server_log):
    print("📋 ВЫВОД (server.log):")
    print("-" * 60)
    with open(server_log, 'r') as f:
        content = f.read()
        if content:
            lines = content.split('\n')
            for line in lines[-50:]:
                print(line)
        else:
            print("   (файл пуст)")
    print()
else:
    print("⚠️ Файл server.log не найден")
    print()

# Проверка процесса
print("=" * 60)
print("  ПРОВЕРКА ПРОЦЕССА")
print("=" * 60)
print()

import subprocess
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
processes = [p for p in result.stdout.split('\n') if 'run.py' in p or '22159' in p]
if processes:
    print("Найденные процессы:")
    for p in processes:
        print(f"  {p}")
else:
    print("⚠️ Процесс 22159 не найден (возможно упал)")

print()
print("=" * 60)
