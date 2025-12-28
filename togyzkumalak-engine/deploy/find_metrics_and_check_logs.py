#!/usr/bin/env python3
"""
Поиск файлов метрик и детальная проверка логов
"""

import os
import glob
import json
import subprocess

print("=" * 70)
print("  ПОИСК МЕТРИК И ПРОВЕРКА ЛОГОВ")
print("=" * 70)
print()

# 1. Поиск всех файлов метрик
print("1️⃣ Ищу файлы метрик...")
checkpoints_dir = '/workspace/togyzkumalak/togyzkumalak-engine/models/alphazero'
workspace_dir = '/workspace/togyzkumalak'

# Ищем все JSON файлы связанные с метриками
search_patterns = [
    os.path.join(checkpoints_dir, '*metrics*.json'),
    os.path.join(checkpoints_dir, '*training*.json'),
    os.path.join(workspace_dir, '**/*metrics*.json'),
    os.path.join(workspace_dir, '**/*training*.json'),
]

found_files = []
for pattern in search_patterns:
    files = glob.glob(pattern, recursive=True)
    found_files.extend(files)

if found_files:
    print(f"   ✅ Найдено файлов: {len(found_files)}")
    for f in found_files:
        size = os.path.getsize(f) / 1024  # KB
        print(f"      {f} ({size:.2f} KB)")
else:
    print("   ⚠️ Файлы метрик не найдены")

print()

# 2. Проверка training_metrics.json
print("2️⃣ Проверяю training_metrics.json...")
metrics_file = os.path.join(checkpoints_dir, 'training_metrics.json')
if os.path.exists(metrics_file):
    print(f"   ✅ Файл существует: {metrics_file}")
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        metrics = data.get('metrics', [])
        print(f"   Записей метрик: {len(metrics)}")
        
        if metrics:
            latest = metrics[-1]
            print(f"   Последняя запись:")
            print(f"      Итерация: {latest.get('iteration', 'N/A')}")
            print(f"      Policy Loss: {latest.get('policy_loss', 0):.4f}")
    except Exception as e:
        print(f"   ❌ Ошибка чтения: {e}")
else:
    print(f"   ⚠️ Файл не найден: {metrics_file}")

print()

# 3. Детальная проверка логов сервера
print("3️⃣ Детальная проверка логов сервера...")
log_file = '/workspace/togyzkumalak/togyzkumalak-engine/server_error.log'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Ищем строки связанные с обучением
    training_lines = []
    error_lines = []
    
    for i, line in enumerate(lines):
        line_upper = line.upper()
        if any(x in line_upper for x in ['ALPHAZERO', 'TRAINING', 'ITERATION', 'CHECKPOINT']):
            training_lines.append((i+1, line.rstrip()))
        if any(x in line_upper for x in ['ERROR', 'EXCEPTION', 'TRACEBACK', 'FAILED', 'CRASHED']):
            error_lines.append((i+1, line.rstrip()))
    
    if training_lines:
        print(f"   Найдено строк связанных с обучением: {len(training_lines)}")
        print("   Последние 15:")
        for num, line in training_lines[-15:]:
            print(f"      {num}: {line[:100]}")
    
    if error_lines:
        print()
        print(f"   ⚠️ Найдено ошибок: {len(error_lines)}")
        print("   Последние ошибки:")
        for num, line in error_lines[-10:]:
            print(f"      {num}: {line[:100]}")
    else:
        print("   Ошибок не найдено")
else:
    print("   Файл логов не найден")

print()

# 4. Проверка процессов обучения
print("4️⃣ Проверяю процессы обучения...")
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
training_processes = [p for p in result.stdout.split('\n') if any(x in p.lower() for x in ['alphazero', 'mcts', 'self-play', 'training'])]

if training_processes:
    print(f"   Найдено процессов: {len(training_processes)}")
    for p in training_processes[:5]:
        print(f"      {p[:100]}")
else:
    print("   Процессы обучения не найдены")

print()

# 5. Проверка последних чекпойнтов по времени
print("5️⃣ Последние чекпойнты по времени создания...")
if os.path.exists(checkpoints_dir):
    pattern = os.path.join(checkpoints_dir, "*.pth.tar")
    checkpoints = glob.glob(pattern)
    
    # Сортируем по времени модификации
    checkpoints_with_time = []
    for cp in checkpoints:
        try:
            mtime = os.path.getmtime(cp)
            checkpoints_with_time.append((cp, mtime))
        except:
            pass
    
    checkpoints_with_time.sort(key=lambda x: x[1], reverse=True)
    
    print("   Последние 5 чекпойнтов:")
    for cp, mtime in checkpoints_with_time[:5]:
        name = os.path.basename(cp)
        from datetime import datetime
        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"      {name} - {time_str}")

print()
print("=" * 70)
print()
print("💡 ВЫВОДЫ:")
print()
print("Если training_metrics.json не найден:")
print("  1. Обучение могло упасть до сохранения метрик")
print("  2. Или метрики сохраняются в другом месте")
print()
print("Если обучение остановилось на 39 итерации:")
print("  1. Проверь ошибки в логах выше")
print("  2. Возможно обучение было остановлено вручную")
print("  3. Или произошла ошибка которая остановила обучение")
print()
print("РЕКОМЕНДАЦИИ:")
print("  1. Перезапусти обучение через UI")
print("  2. Следи за логами в реальном времени")
print("  3. Проверь что все параметры корректны")
print()
print("=" * 70)
