#!/usr/bin/env python3
"""
Очистка старых чекпойнтов, оставляя только лучшие и последние
"""

import os
import glob
import json
from datetime import datetime

print("=" * 70)
print("  ОЧИСТКА СТАРЫХ ЧЕКПОЙНТОВ")
print("=" * 70)
print()

checkpoints_dir = '/workspace/togyzkumalak/togyzkumalak-engine/models/alphazero'

if not os.path.exists(checkpoints_dir):
    print("❌ Директория не найдена")
    exit(1)

# Находим все чекпойнты
pattern = os.path.join(checkpoints_dir, "*.pth.tar")
all_checkpoints = glob.glob(pattern)

print(f"Найдено чекпойнтов: {len(all_checkpoints)}")
print()

# Группируем по номеру итерации
checkpoints_by_iter = {}
for cp in all_checkpoints:
    name = os.path.basename(cp)
    if 'checkpoint_' in name:
        try:
            iter_num = int(name.replace('checkpoint_', '').replace('.pth.tar', ''))
            checkpoints_by_iter[iter_num] = cp
        except:
            pass

if not checkpoints_by_iter:
    print("⚠️ Не найдено пронумерованных чекпойнтов")
    exit(0)

# Сортируем по итерации
sorted_iters = sorted(checkpoints_by_iter.keys(), reverse=True)

# Оставляем последние 10
keep_last = sorted_iters[:10]
print(f"Оставляем последние 10 итераций: {keep_last}")

# Также оставляем лучшие (если есть метрики)
keep_best = []
metrics_file = os.path.join(checkpoints_dir, 'metrics.json')
if os.path.exists(metrics_file):
    try:
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        if 'iterations' in metrics_data:
            # Находим лучшие по policy_loss
            iterations = metrics_data['iterations']
            best_by_loss = sorted(iterations, key=lambda x: x.get('policy_loss', float('inf')))[:5]
            keep_best = [x.get('iteration') for x in best_by_loss if 'iteration' in x]
            print(f"Оставляем лучшие 5 по loss: {keep_best}")
    except:
        pass

# Объединяем списки
keep_set = set(keep_last + keep_best)
print(f"Всего оставляем: {len(keep_set)} чекпойнтов")
print()

# Удаляем остальные
to_delete = []
for iter_num, cp_path in checkpoints_by_iter.items():
    if iter_num not in keep_set:
        to_delete.append((iter_num, cp_path))

if to_delete:
    print(f"Будет удалено: {len(to_delete)} чекпойнтов")
    print()
    
    # Показываем что будет удалено
    print("Чекпойнты для удаления:")
    for iter_num, cp_path in sorted(to_delete, key=lambda x: x[0]):
        name = os.path.basename(cp_path)
        size = os.path.getsize(cp_path) / (1024**2)
        print(f"   iter {iter_num}: {name} ({size:.2f} MB)")
    
    print()
    confirm = input("Удалить? (yes/no): ")
    
    if confirm.lower() == 'yes':
        freed_space = 0
        deleted_count = 0
        
        for iter_num, cp_path in to_delete:
            try:
                size = os.path.getsize(cp_path)
                os.remove(cp_path)
                freed_space += size
                deleted_count += 1
                print(f"   ✅ Удален: {os.path.basename(cp_path)}")
            except Exception as e:
                print(f"   ❌ Ошибка удаления {os.path.basename(cp_path)}: {e}")
        
        print()
        print(f"✅ Удалено: {deleted_count} файлов")
        print(f"   Освобождено: {freed_space / (1024**2):.2f} MB")
    else:
        print("Отменено")
else:
    print("✅ Нет чекпойнтов для удаления")

print()
print("=" * 70)
