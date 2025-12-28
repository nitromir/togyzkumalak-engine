#!/usr/bin/env python3
"""
Детальный мониторинг обучения с проверкой процессов
"""

import requests
import subprocess
import time
import os

print("=" * 70)
print("  ДЕТАЛЬНЫЙ МОНИТОРИНГ ОБУЧЕНИЯ")
print("=" * 70)
print()

# 1. Проверка активных сессий
print("1️⃣ Проверяю активные сессии...")
try:
    response = requests.get('http://localhost:8000/api/training/alphazero/sessions', timeout=5)
    sessions = response.json().get('sessions', {})
    
    if sessions:
        task_id = list(sessions.keys())[0]
        print(f"   ✅ Найдена сессия: {task_id[:8]}...")
        
        # Получаем детальный статус
        status_response = requests.get(
            f'http://localhost:8000/api/training/alphazero/sessions/{task_id}',
            timeout=5
        )
        
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"   Статус: {status.get('status', 'unknown')}")
            print(f"   Итерация: {status.get('current_iteration', 0)}/{status.get('total_iterations', 0)}")
            print(f"   Прогресс: {status.get('progress', 0):.1f}%")
            
            # Дополнительная информация если есть
            if 'elapsed_time' in status:
                print(f"   Время: {status.get('elapsed_time', 0):.1f} сек")
    else:
        print("   ⚠️ Нет активных сессий")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

print()

# 2. Проверка процессов Python
print("2️⃣ Проверяю процессы Python...")
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
python_processes = [p for p in result.stdout.split('\n') if 'python' in p.lower()]

# Ищем процессы связанные с обучением
training_processes = []
for p in python_processes:
    if any(x in p.lower() for x in ['alphazero', 'mcts', 'self-play', 'training', 'coach']):
        training_processes.append(p)

if training_processes:
    print(f"   Найдено процессов обучения: {len(training_processes)}")
    for p in training_processes:
        parts = p.split()
        if len(parts) > 1:
            pid = parts[1]
            cpu = parts[2] if len(parts) > 2 else '?'
            mem = parts[3] if len(parts) > 3 else '?'
            print(f"      PID: {pid} | CPU: {cpu}% | MEM: {mem}%")
            print(f"      Команда: {p[:100]}")
else:
    print("   ⚠️ Процессы обучения не найдены")

print()

# 3. Проверка использования GPU
print("3️⃣ Проверяю использование GPU...")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        print(f"   GPU статус:")
        for i, line in enumerate(lines[:5]):  # Первые 5 GPU
            parts = line.split(',')
            if len(parts) >= 5:
                gpu_id = parts[0].strip()
                name = parts[1].strip()
                util = parts[2].strip()
                mem_used = parts[3].strip()
                mem_total = parts[4].strip()
                print(f"      GPU {gpu_id}: {name} | Util: {util} | Mem: {mem_used}/{mem_total}")
    else:
        print("   ⚠️ Не удалось получить статус GPU")
except Exception as e:
    print(f"   ⚠️ Ошибка проверки GPU: {e}")

print()

# 4. Проверка логов в реальном времени
print("4️⃣ Проверяю последние строки логов...")
log_file = '/workspace/togyzkumalak/togyzkumalak-engine/server_error.log'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Последние 30 строк
    recent_lines = lines[-30:] if len(lines) > 30 else lines
    
    # Ищем строки связанные с обучением
    training_related = []
    for line in recent_lines:
        line_lower = line.lower()
        if any(x in line_lower for x in ['alphazero', 'training', 'iteration', 'checkpoint', 'mcts', 'self-play', 'episode', 'bootstrap']):
            training_related.append(line.rstrip())
    
    if training_related:
        print(f"   Найдено строк связанных с обучением: {len(training_related)}")
        print("   Последние:")
        for line in training_related[-15:]:
            print(f"      {line[:120]}")
    else:
        print("   ⚠️ В последних строках нет информации об обучении")
        print("   Последние 10 строк лога:")
        for line in recent_lines[-10:]:
            print(f"      {line.rstrip()[:120]}")

print()

# 5. Проверка файлов чекпойнтов (новые)
print("5️⃣ Проверяю новые чекпойнты...")
checkpoints_dir = '/workspace/togyzkumalak/togyzkumalak-engine/models/alphazero'
if os.path.exists(checkpoints_dir):
    import glob
    from datetime import datetime, timedelta
    
    pattern = os.path.join(checkpoints_dir, "*.pth.tar")
    checkpoints = glob.glob(pattern)
    
    # Чекпойнты созданные за последние 5 минут
    now = time.time()
    recent_checkpoints = []
    for cp in checkpoints:
        try:
            mtime = os.path.getmtime(cp)
            if now - mtime < 300:  # Последние 5 минут
                recent_checkpoints.append((cp, mtime))
        except:
            pass
    
    if recent_checkpoints:
        print(f"   ✅ Найдено новых чекпойнтов за последние 5 минут: {len(recent_checkpoints)}")
        for cp, mtime in sorted(recent_checkpoints, key=lambda x: x[1], reverse=True)[:5]:
            name = os.path.basename(cp)
            time_str = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
            print(f"      {name} - {time_str}")
    else:
        print("   ⚠️ Новых чекпойнтов не найдено")

print()
print("=" * 70)
print()
print("💡 ВЫВОДЫ:")
print()
print("Если итерация остается на 0:")
print("  1. Обучение может быть на стадии bootstrap (если включен)")
print("  2. Или на стадии первого self-play (может занять время)")
print("  3. Или есть проблема с параллельным выполнением")
print()
print("РЕКОМЕНДАЦИИ:")
print("  1. Подожди еще 2-3 минуты - первая итерация самая долгая")
print("  2. Проверь использование GPU - должны быть загружены")
print("  3. Если через 5 минут ничего не изменилось - есть проблема")
print()
print("=" * 70)
