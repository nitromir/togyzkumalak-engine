#!/usr/bin/env python3
"""
Проверить последние логи с детальными ошибками
"""

import os
import subprocess

print("=" * 80)
print("  ПОСЛЕДНИЕ ЛОГИ С ОШИБКАМИ")
print("=" * 80)
print()

# Проверяем server_error.log
error_log = '/workspace/togyzkumalak/togyzkumalak-engine/server_error.log'
if os.path.exists(error_log):
    print("ОШИБКИ (последние 50 строк):")
    print("-" * 80)
    result = subprocess.run(['tail', '-50', error_log], capture_output=True, text=True)
    print(result.stdout)
    print()

# Проверяем server.log на наличие PREDICT, boardToObservation, FORWARD логов
server_log = '/workspace/togyzkumalak/togyzkumalak-engine/server.log'
if os.path.exists(server_log):
    print("ЛОГИ С КЛЮЧЕВЫМИ СЛОВАМИ (последние 100 строк):")
    print("-" * 80)
    result = subprocess.run(['grep', '-i', 'predict\|board\|forward\|error\|warning', server_log], 
                          capture_output=True, text=True)
    lines = result.stdout.split('\n')
    for line in lines[-50:]:
        if line.strip():
            print(line)
    print()

# Проверяем процессы
print("ПРОЦЕССЫ:")
print("-" * 80)
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
processes = [p for p in result.stdout.split('\n') if 'run.py' in p]
for p in processes[:3]:
    print(p[:100])
