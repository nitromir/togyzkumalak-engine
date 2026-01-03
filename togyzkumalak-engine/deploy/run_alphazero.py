#!/usr/bin/env python3
"""
Запуск AlphaZero training напрямую из терминала (без UI).
Оптимизировано для multi-GPU системы (4x GPU, 128 ядер).
"""

import os
import sys
import json
import time
import signal
import subprocess

# Добавляем путь к backend
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from task_manager import AlphaZeroTaskManager

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

def get_gpu_count():
    """Определяет количество доступных GPU."""
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
    except:
        pass
    return 0

def get_cpu_count():
    """Определяет количество CPU ядер."""
    return os.cpu_count() or 1

def main():
    print("=" * 70)
    print("  ALPHAZERO TRAINING - ПРЯМОЙ ЗАПУСК")
    print("=" * 70)
    print()
    
    # Определяем железо
    num_gpus = get_gpu_count()
    num_cpus = get_cpu_count()
    
    print(f"🔧 Обнаружено железа:")
    print(f"   GPU: {num_gpus}")
    print(f"   CPU ядер: {num_cpus}")
    print()
    
    # Конфигурация для multi-GPU системы (4x GPU, 128 ядер)
    # Основано на get_optimal_config из alphazero_trainer.py (строки 2316-2398)
    if num_gpus >= 4:
        # Medium setup (4-7 GPUs) - оптимизировано для 4x RTX 3090/4090
        config = {
            'iterations': 200,
            'games_per_iteration': 128,      # 32 игры на GPU
            'mcts_simulations': 60,          # Хорошая глубина для 4 GPU
            'batch_size': 1024,              # Большой батч для GPU
            'epochs': 8,
            'lr': 0.001,
            'hidden_size': 256,
            'arena_compare': 20,              # Быстрая оценка
            'use_bootstrap': True,
            'use_multiprocessing': True,     # КРИТИЧНО: включить multiprocessing
            'num_parallel_games': min(64, num_gpus * 4),  # 16 для 4 GPU
            'num_workers': min(num_cpus - 2, num_gpus * 10),  # 40 для 4 GPU, 128 ядер
            'save_every_n_iters': 2,
            'update_threshold': 0.55
        }
    elif num_gpus >= 2:
        # Small setup (2-3 GPUs)
        config = {
            'iterations': 200,
            'games_per_iteration': 24,
            'mcts_simulations': 20,
            'batch_size': 512,
            'epochs': 8,
            'lr': 0.001,
            'hidden_size': 256,
            'arena_compare': 10,
            'use_bootstrap': True,
            'use_multiprocessing': True,
            'num_parallel_games': min(64, num_gpus * 4),
            'num_workers': min(num_cpus - 2, num_gpus * 10),
            'save_every_n_iters': 5,
            'update_threshold': 0.55
        }
    else:
        # Single GPU
        config = {
            'iterations': 200,
            'games_per_iteration': 16,
            'mcts_simulations': 15,
            'batch_size': 256,
            'epochs': 5,
            'lr': 0.001,
            'hidden_size': 256,
            'arena_compare': 8,
            'use_bootstrap': True,
            'use_multiprocessing': True,
            'num_parallel_games': 8,
            'num_workers': min(num_cpus - 2, 10),
            'save_every_n_iters': 5,
            'update_threshold': 0.55
        }
    
    print("📋 Конфигурация:")
    print(f"   Итераций: {config['iterations']}")
    print(f"   Игр на итерацию: {config['games_per_iteration']}")
    print(f"   MCTS симуляций: {config['mcts_simulations']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Параллельных игр: {config['num_parallel_games']}")
    print(f"   Workers: {config['num_workers']}")
    print(f"   Multiprocessing: {config['use_multiprocessing']}")
    print()
    
    # Проверяем путь сохранения чекпойнтов
    task_manager = AlphaZeroTaskManager()
    checkpoints_dir = os.path.join(task_manager.engine_dir, "models", "alphazero")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    print(f"💾 Чекпойнты будут сохраняться в: {checkpoints_dir}")
    print()
    
    try:
        # Запускаем обучение
        print("🚀 Запуск AlphaZero training...")
        task_id = task_manager.start_training(config)
        print(f"✅ Обучение запущено! Task ID: {task_id}")
        print()
        print("=" * 70)
        print("  ОБУЧЕНИЕ ЗАПУЩЕНО")
        print("=" * 70)
        print()
        print("📊 Мониторинг прогресса:")
        print(f"   - Статус: curl http://localhost:8000/api/training/alphazero/sessions/{task_id}")
        print(f"   - Чекпойнты: ls -lh {checkpoints_dir}")
        print()
        print("⏹️  Для остановки нажмите Ctrl+C")
        print()
        
        # Мониторим прогресс
        last_iteration = 0
        while True:
            if stop_requested:
                print("\n🛑 Запрос на остановку получен...")
                task_manager.stop_task(task_id)
                break
            
            status = task_manager.get_status(task_id)
            if not status:
                print("❌ Задача не найдена!")
                break
            
            current_iter = status.get("current_iteration", 0)
            total_iter = status.get("total_iterations", config['iterations'])
            progress = status.get("progress", 0)
            task_status = status.get("status", "unknown")
            
            if current_iter != last_iteration:
                print(f"📈 Итерация {current_iter}/{total_iter} ({progress:.1f}%) - Статус: {task_status}")
                last_iteration = current_iter
                
                # Показываем последние чекпойнты
                if os.path.exists(checkpoints_dir):
                    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth.tar')]
                    if checkpoints:
                        latest = max(checkpoints, key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
                        size_mb = os.path.getsize(os.path.join(checkpoints_dir, latest)) / (1024 * 1024)
                        print(f"   💾 Последний чекпойнт: {latest} ({size_mb:.2f} MB)")
            
            if task_status == "completed":
                print()
                print("=" * 70)
                print("  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
                print("=" * 70)
                print()
                print(f"💾 Все чекпойнты сохранены в: {checkpoints_dir}")
                
                # Показываем финальные чекпойнты
                if os.path.exists(checkpoints_dir):
                    checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.pth.tar')])
                    print(f"\n📦 Всего чекпойнтов: {len(checkpoints)}")
                    if checkpoints:
                        print("   Последние 5:")
                        for ckpt in checkpoints[-5:]:
                            size_mb = os.path.getsize(os.path.join(checkpoints_dir, ckpt)) / (1024 * 1024)
                            print(f"   - {ckpt} ({size_mb:.2f} MB)")
                break
            
            if task_status == "error":
                error = status.get("error", "Unknown error")
                print()
                print("=" * 70)
                print("  ❌ ОШИБКА ОБУЧЕНИЯ")
                print("=" * 70)
                print(f"Ошибка: {error}")
                break
            
            time.sleep(5)  # Проверяем каждые 5 секунд
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        if 'task_id' in locals():
            task_manager.stop_task(task_id)
    except Exception as e:
        print(f"\n\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n👋 Завершение работы...")

if __name__ == "__main__":
    main()
