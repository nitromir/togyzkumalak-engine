#!/usr/bin/env python3
"""
Проверка статуса обучения AlphaZero
"""

import requests
import json
import os

print("=" * 60)
print("  СТАТУС ОБУЧЕНИЯ ALPHAZERO")
print("=" * 60)
print()

# Получаем статус задачи
task_id = "az_1766964165"  # Замени на свой task_id если нужно

try:
    # Получаем список всех сессий
    response = requests.get('http://localhost:8000/api/training/alphazero/sessions')
    sessions = response.json()
    
    print("📊 Активные сессии:")
    for sid, status in sessions.get('sessions', {}).items():
        print(f"   Task ID: {sid}")
        print(f"   Статус: {status.get('status', 'unknown')}")
        print(f"   Итерация: {status.get('current_iteration', 0)} / {status.get('total_iterations', 0)}")
        print(f"   Прогресс: {status.get('progress', 0):.1f}%")
        print(f"   Этап: {status.get('current_phase', 'unknown')}")
        print()
        
        # Детальная информация
        detail_response = requests.get(f'http://localhost:8000/api/training/alphazero/sessions/{sid}')
        if detail_response.status_code == 200:
            detail = detail_response.json()
            print("   Детали:")
            print(f"      Игр завершено: {detail.get('games_completed', 0)} / {detail.get('total_games', 0)}")
            print(f"      Эпизодов: {detail.get('episodes_completed', 0)}")
            print(f"      Примеров собрано: {detail.get('examples_collected', 0)}")
            print(f"      Loss: {detail.get('current_loss', 'N/A')}")
            print()
    
    # Проверяем логи обучения
    print("=" * 60)
    print("  ПРОВЕРКА ЛОГОВ")
    print("=" * 60)
    print()
    
    engine_dir = '/workspace/togyzkumalak/togyzkumalak-engine'
    metrics_file = os.path.join(engine_dir, 'models', 'alphazero', 'metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            print("📈 Последние метрики:")
            if 'history' in metrics and len(metrics['history']) > 0:
                last = metrics['history'][-1]
                print(f"   Итерация: {last.get('iteration', 'N/A')}")
                print(f"   Policy Loss: {last.get('policy_loss', 'N/A')}")
                print(f"   Value Loss: {last.get('value_loss', 'N/A')}")
                print(f"   Win Rate: {last.get('win_rate', 'N/A')}")
            else:
                print("   Метрики еще не собраны")
    else:
        print("⚠️ Файл метрик не найден (обучение еще не началось или не сохранило метрики)")
    
    print()
    
    # Проверяем чекпоинты
    checkpoints_dir = os.path.join(engine_dir, 'models', 'alphazero')
    if os.path.exists(checkpoints_dir):
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth.tar')]
        print(f"📦 Чекпоинтов найдено: {len(checkpoints)}")
        if checkpoints:
            print("   Последние:")
            for cp in sorted(checkpoints)[-5:]:
                print(f"      {cp}")
    
    print()
    print("=" * 60)
    print("  РЕКОМЕНДАЦИИ")
    print("=" * 60)
    print()
    print("Если GPU не используются:")
    print("1. AlphaZero сначала собирает данные через self-play (может занять время)")
    print("2. GPU используются только на этапе обучения нейросети")
    print("3. Проверь логи сервера для деталей")
    print()
    print("Для проверки GPU в реальном времени:")
    print("   watch -n 1 nvidia-smi")
    print()
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
