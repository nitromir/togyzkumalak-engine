#!/usr/bin/env python3
"""
Проверка дискового пространства и очистка старых чекпойнтов
"""

import os
import shutil
import glob
from datetime import datetime

print("=" * 70)
print("  ПРОВЕРКА ДИСКОВОГО ПРОСТРАНСТВА")
print("=" * 70)
print()

# 1. Проверка свободного места
print("1️⃣ Проверяю свободное место...")
try:
    stat = shutil.disk_usage('/workspace')
    total_gb = stat.total / (1024**3)
    used_gb = stat.used / (1024**3)
    free_gb = stat.free / (1024**3)
    percent_used = (used_gb / total_gb) * 100
    
    print(f"   Всего: {total_gb:.2f} GB")
    print(f"   Использовано: {used_gb:.2f} GB ({percent_used:.1f}%)")
    print(f"   Свободно: {free_gb:.2f} GB")
    
    if free_gb < 1:
        print("   ⚠️ КРИТИЧЕСКИ МАЛО МЕСТА!")
    elif free_gb < 5:
        print("   ⚠️ Мало свободного места")
    else:
        print("   ✅ Места достаточно")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

print()

# 2. Размер директории с чекпойнтами
print("2️⃣ Размер директории с чекпойнтами...")
checkpoints_dir = '/workspace/togyzkumalak/togyzkumalak-engine/models/alphazero'

if os.path.exists(checkpoints_dir):
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(checkpoints_dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                total_size += os.path.getsize(filepath)
                file_count += 1
            except:
                pass
    
    size_gb = total_size / (1024**3)
    size_mb = total_size / (1024**2)
    
    print(f"   Файлов: {file_count}")
    print(f"   Размер: {size_gb:.2f} GB ({size_mb:.2f} MB)")
else:
    print("   Директория не найдена")

print()

# 3. Анализ чекпойнтов
print("3️⃣ Анализ чекпойнтов...")
if os.path.exists(checkpoints_dir):
    pattern = os.path.join(checkpoints_dir, "*.pth.tar")
    checkpoints = glob.glob(pattern)
    
    if checkpoints:
        # Группируем по размеру
        checkpoints_with_size = []
        for cp in checkpoints:
            try:
                size = os.path.getsize(cp)
                mtime = os.path.getmtime(cp)
                checkpoints_with_size.append((cp, size, mtime))
            except:
                pass
        
        checkpoints_with_size.sort(key=lambda x: x[2], reverse=True)  # По времени
        
        print(f"   Всего чекпойнтов: {len(checkpoints_with_size)}")
        print()
        print("   Последние 10 (по времени):")
        for cp, size, mtime in checkpoints_with_size[:10]:
            name = os.path.basename(cp)
            size_mb = size / (1024**2)
            time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"      {name} - {size_mb:.2f} MB - {time_str}")

print()

# 4. Рекомендации по очистке
print("4️⃣ Рекомендации по очистке...")
if os.path.exists(checkpoints_dir):
    pattern = os.path.join(checkpoints_dir, "*.pth.tar")
    checkpoints = glob.glob(pattern)
    
    if len(checkpoints) > 20:
        print(f"   ⚠️ Много чекпойнтов ({len(checkpoints)})")
        print("   Рекомендуется оставить только:")
        print("      - Последние 10 чекпойнтов")
        print("      - Лучшие 5 чекпойнтов (по loss)")
        print()
        print("   Можно удалить старые через:")
        print("      python deploy/cleanup_old_checkpoints.py")

print()
print("=" * 70)
