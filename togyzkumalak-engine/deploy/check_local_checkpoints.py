#!/usr/bin/env python3
"""
Проверка чекпойнтов на локальной машине
"""

import os
import glob
from datetime import datetime

# Путь к локальным чекпойнтам (измени на свой)
local_checkpoints_dir = os.path.expanduser("~/Documents/Toguzkumalak/gym-togyzkumalak-master/togyzkumalak-engine/models/alphazero")

# Альтернативные пути
possible_paths = [
    local_checkpoints_dir,
    "./models/alphazero",
    "../models/alphazero",
    os.path.join(os.getcwd(), "models", "alphazero"),
]

print("=" * 70)
print("  ПРОВЕРКА ЛОКАЛЬНЫХ ЧЕКПОЙНТОВ")
print("=" * 70)
print()

checkpoints_found = False

for checkpoints_dir in possible_paths:
    if os.path.exists(checkpoints_dir):
        print(f"📁 Директория: {checkpoints_dir}")
        print()
        
        # Ищем все .pth.tar файлы
        pattern = os.path.join(checkpoints_dir, "*.pth.tar")
        checkpoints = glob.glob(pattern)
        
        if checkpoints:
            checkpoints_found = True
            # Сортируем по времени модификации (новые первыми)
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            
            print(f"✅ Найдено чекпойнтов: {len(checkpoints)}")
            print()
            print("Последние чекпойнты:")
            print("-" * 70)
            
            for i, cp in enumerate(checkpoints[:10], 1):
                filename = os.path.basename(cp)
                size = os.path.getsize(cp) / (1024 * 1024)  # MB
                mtime = datetime.fromtimestamp(os.path.getmtime(cp))
                age = datetime.now() - mtime
                
                print(f"{i}. {filename}")
                print(f"   Размер: {size:.2f} MB")
                print(f"   Время: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age})")
                print()
            
            break

if not checkpoints_found:
    print("⚠️  Чекпойнты не найдены в стандартных местах")
    print()
    print("Проверенные пути:")
    for path in possible_paths:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"   {exists} {path}")
    print()
    print("💡 Укажи правильный путь к директории с чекпойнтами")

print("=" * 70)
