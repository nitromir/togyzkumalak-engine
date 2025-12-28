#!/usr/bin/env python3
"""
Проверка завершенного обучения и чекпойнтов
"""

import requests
import os
import glob

print("=" * 70)
print("  ПРОВЕРКА ОБУЧЕНИЯ И ЧЕКПОЙНТОВ")
print("=" * 70)
print()

# 1. Проверка метрик
print("1️⃣ Проверяю метрики последней тренировки...")
try:
    response = requests.get('http://localhost:8000/api/training/alphazero/metrics', timeout=5)
    if response.status_code == 200:
        data = response.json()
        summary = data.get('summary', {})
        checkpoints = data.get('checkpoints', [])
        
        if summary:
            print("   ✅ Найдены метрики!")
            print()
            print(f"   Последняя итерация: {summary.get('latest_iteration', 0)}")
            print(f"   Policy Loss: {summary.get('latest_policy_loss', 0):.4f}")
            print(f"   Value Loss: {summary.get('latest_value_loss', 0):.4f}")
            print(f"   Win Rate: {summary.get('latest_win_rate', 0)*100:.1f}%")
            print(f"   Всего примеров: {summary.get('total_examples', 0):,}")
            
            best = summary.get('best_checkpoint')
            if best:
                print()
                print(f"   🏆 Лучший чекпойнт:")
                print(f"      Итерация: {best.get('iteration', 0)}")
                print(f"      Policy Loss: {best.get('policy_loss', 0):.4f}")
                print(f"      Файл: {best.get('filename', 'N/A')}")
        else:
            print("   ⚠️ Метрики не найдены")
        
        if checkpoints:
            print()
            print(f"   📦 Найдено чекпойнтов: {len(checkpoints)}")
            print("   Топ-5 чекпойнтов:")
            for i, cp in enumerate(checkpoints[:5], 1):
                print(f"      {i}. iter {cp.get('iteration', 0)} - loss: {cp.get('policy_loss', 0):.4f}")
    else:
        print(f"   ❌ Ошибка получения метрик: {response.status_code}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

print()

# 2. Проверка файлов чекпойнтов на сервере
print("2️⃣ Проверяю чекпойнты на сервере...")
checkpoints_dir = '/workspace/togyzkumalak/togyzkumalak-engine/models/alphazero'

if os.path.exists(checkpoints_dir):
    pattern = os.path.join(checkpoints_dir, "*.pth.tar")
    checkpoints = glob.glob(pattern)
    
    if checkpoints:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        print(f"   ✅ Найдено файлов: {len(checkpoints)}")
        print()
        print("   Последние чекпойнты:")
        for i, cp in enumerate(checkpoints[:5], 1):
            filename = os.path.basename(cp)
            size = os.path.getsize(cp) / (1024 * 1024)  # MB
            print(f"      {i}. {filename} ({size:.2f} MB)")
    else:
        print("   ⚠️ Файлы чекпойнтов не найдены")
else:
    print(f"   ⚠️ Директория не существует: {checkpoints_dir}")

print()

# 3. Проверка логов
print("3️⃣ Проверяю логи сервера...")
log_file = '/workspace/togyzkumalak/togyzkumalak-engine/server_error.log'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if lines:
            print("   Последние 10 строк логов:")
            for line in lines[-10:]:
                print(f"      {line.rstrip()}")
        else:
            print("   Лог пуст")
else:
    print("   Файл логов не найден")

print()
print("=" * 70)
print()
print("💡 РЕКОМЕНДАЦИИ:")
print()
print("Если обучение не запущено:")
print("  1. Открой UI в браузере")
print("  2. Перейди на вкладку '🧠 Тренировка'")
print("  3. Нажми '🚀 Запустить AlphaZero'")
print()
print("Если обучение завершилось:")
print("  - Проверь чекпойнты выше")
print("  - Загрузи лучший чекпойнт через UI")
print()
print("=" * 70)
