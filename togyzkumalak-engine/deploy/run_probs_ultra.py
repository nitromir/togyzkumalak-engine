#!/usr/bin/env python3
"""
Запуск PROBS Ultra training напрямую из терминала (без UI).
200 итераций, все чекпойнты сохраняются в models/probs/checkpoints/
"""

import os
import sys
import json
import time
import signal

# Добавляем путь к backend
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from probs_task_manager import PROBSTaskManager

# Глобальная переменная для остановки
stop_requested = False

def signal_handler(sig, frame):
    """Обработчик сигнала для корректной остановки."""
    global stop_requested
    print("\n\n⚠️  Получен сигнал остановки. Завершаю обучение после текущей итерации...")
    stop_requested = True

# Регистрируем обработчик сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    print("=" * 70)
    print("  PROBS ULTRA TRAINING - ПРЯМОЙ ЗАПУСК")
    print("=" * 70)
    print()
    
    # Конфигурация для 200 итераций - МОНСТР-КОНФИГ (128 ядер, 4x GPU)
    # Основано на Monster Config из UI (training.js:1444-1461)
    config = {
        'n_high_level_iterations': 200,
        'v_train_episodes': 8000,      # 8K партий для Self-play (GPU inference)
        'q_train_episodes': 4000,      # 4K эпизодов Q-train (CPU + GPU)
        'mem_max_episodes': 80000,     # Буфер памяти
        'train_batch_size': 2048,      # Большой батч для GPU
        'num_q_s_a_calls': 50,         # Глубина поиска Q
        'max_depth': 100,              # Макс глубина дерева
        'self_play_threads': 16,       # 16 потоков Self-play
        'sub_processes_cnt': 64,       # 64 воркера Q-train
        'evaluate_n_games': 100,        # 100 игр для стабильной оценки
        'device': 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
        'use_boost': True,             # Включаем Boosting
        'initial_checkpoint': None,
        'ultra_mode': True,
        'vs_alphazero_ratio': 0.3      # 30% игр против AlphaZero
    }
    
    print("📋 Конфигурация:")
    print(f"   Итераций: {config['n_high_level_iterations']}")
    print(f"   V-train эпизодов: {config['v_train_episodes']}")
    print(f"   Q-train эпизодов: {config['q_train_episodes']}")
    print(f"   Устройство: {config['device']}")
    print(f"   Ultra режим: {config['ultra_mode']} (30% игр vs AlphaZero)")
    print()
    
    # Проверяем путь сохранения чекпойнтов
    task_manager = PROBSTaskManager()
    checkpoints_dir = os.path.join(task_manager.models_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    print(f"💾 Чекпойнты будут сохраняться в: {checkpoints_dir}")
    print(f"📝 Логи будут сохраняться в: {os.path.join(task_manager.engine_dir, 'probs_training.log')}")
    print()
    
    # Проверяем существование директории
    if not os.path.exists(checkpoints_dir):
        print(f"❌ ОШИБКА: Директория для чекпойнтов не существует: {checkpoints_dir}")
        sys.exit(1)
    
    print("✅ Директория для чекпойнтов существует")
    print()
    
    try:
        # Запускаем обучение
        print("🚀 Запуск PROBS Ultra training...")
        task_id = task_manager.start_ultra_training(config)
        print(f"✅ Обучение запущено! Task ID: {task_id}")
        print()
        print("=" * 70)
        print("  ОБУЧЕНИЕ ЗАПУЩЕНО")
        print("=" * 70)
        print()
        print("📊 Мониторинг прогресса:")
        print(f"   - Логи: tail -f {os.path.join(task_manager.engine_dir, 'probs_training.log')}")
        print(f"   - Чекпойнты: ls -lh {checkpoints_dir}")
        print()
        print("⏹️  Для остановки нажмите Ctrl+C (остановится после текущей итерации)")
        print()
        
        # Мониторим прогресс
        last_iteration = 0
        while True:
            if stop_requested:
                print("\n🛑 Запрос на остановку получен...")
                task_manager.stop_requested = True
                break
            
            task_info = task_manager.tasks.get(task_id)
            if not task_info:
                print("❌ Задача не найдена!")
                break
            
            status = task_info.get("status")
            current_iter = task_info.get("current_iteration", 0)
            total_iter = task_info.get("total_iterations", 200)
            progress = task_info.get("progress", 0)
            
            if current_iter != last_iteration:
                print(f"📈 Итерация {current_iter}/{total_iter} ({progress:.1f}%) - Статус: {status}")
                last_iteration = current_iter
                
                # Показываем последние чекпойнты (формат: prefix_YYYYMMDD-HHMMSS.ckpt)
                if os.path.exists(checkpoints_dir):
                    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
                    if checkpoints:
                        latest = max(checkpoints, key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
                        size_mb = os.path.getsize(os.path.join(checkpoints_dir, latest)) / (1024 * 1024)
                        print(f"   💾 Последний чекпойнт: {latest} ({size_mb:.2f} MB)")
                        
                        # Показываем статистику по типам
                        iter_ckpts = [f for f in checkpoints if f.startswith('iter_')]
                        best_ckpts = [f for f in checkpoints if f.startswith('best_iter_')]
                        final_ckpts = [f for f in checkpoints if f.startswith('final')]
                        temp_ckpts = [f for f in checkpoints if f.startswith('temp_iter_')]
                        if iter_ckpts or best_ckpts or final_ckpts or temp_ckpts:
                            print(f"      (iter: {len(iter_ckpts)}, best: {len(best_ckpts)}, final: {len(final_ckpts)}, temp: {len(temp_ckpts)})")
            
            if status == "completed":
                print()
                print("=" * 70)
                print("  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
                print("=" * 70)
                print()
                print(f"💾 Все чекпойнты сохранены в: {checkpoints_dir}")
                
                # Показываем финальные чекпойнты (формат: prefix_YYYYMMDD-HHMMSS.ckpt)
                if os.path.exists(checkpoints_dir):
                    checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')])
                    print(f"\n📦 Всего чекпойнтов (.ckpt): {len(checkpoints)}")
                    
                    # Разделяем по типам (формат: prefix_timestamp.ckpt)
                    iter_ckpts = [f for f in checkpoints if f.startswith('iter_')]
                    best_ckpts = [f for f in checkpoints if f.startswith('best_iter_')]
                    final_ckpts = [f for f in checkpoints if f.startswith('final')]
                    temp_ckpts = [f for f in checkpoints if f.startswith('temp_iter_')]
                    other_ckpts = [f for f in checkpoints if not any([f.startswith(p) for p in ['iter_', 'best_iter_', 'final', 'temp_iter_']])]
                    
                    print(f"\n   Типы чекпойнтов:")
                    print(f"   - iter_*_*.ckpt (периодические): {len(iter_ckpts)}")
                    print(f"   - best_iter_*_*.ckpt (лучшие): {len(best_ckpts)}")
                    print(f"   - final_*.ckpt (финальный): {len(final_ckpts)}")
                    print(f"   - temp_iter_*_*.ckpt (временные): {len(temp_ckpts)}")
                    if other_ckpts:
                        print(f"   - другие: {len(other_ckpts)}")
                    
                    if checkpoints:
                        print("\n   Последние 5 (по времени создания):")
                        # Сортируем по времени модификации
                        checkpoints_by_time = sorted(checkpoints, key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
                        for ckpt in checkpoints_by_time[-5:]:
                            size_mb = os.path.getsize(os.path.join(checkpoints_dir, ckpt)) / (1024 * 1024)
                            mtime = os.path.getmtime(os.path.join(checkpoints_dir, ckpt))
                            from datetime import datetime
                            time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                            print(f"   - {ckpt} ({size_mb:.2f} MB, {time_str})")
                break
            
            if status == "error":
                error = task_info.get("error", "Unknown error")
                print()
                print("=" * 70)
                print("  ❌ ОШИБКА ОБУЧЕНИЯ")
                print("=" * 70)
                print(f"Ошибка: {error}")
                break
            
            time.sleep(5)  # Проверяем каждые 5 секунд
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        task_manager.stop_requested = True
    except Exception as e:
        print(f"\n\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n👋 Завершение работы...")

if __name__ == "__main__":
    main()
