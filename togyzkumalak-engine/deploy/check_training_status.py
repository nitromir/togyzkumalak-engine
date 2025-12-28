#!/usr/bin/env python3
"""
Проверка статуса обучения AlphaZero в реальном времени
"""

import requests
import time
import json
from datetime import datetime

def format_time(seconds):
    """Форматирует секунды в читаемый формат"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}ч {minutes}м {secs}с"
    elif minutes > 0:
        return f"{minutes}м {secs}с"
    else:
        return f"{secs}с"

print("=" * 70)
print("  МОНИТОРИНГ ОБУЧЕНИЯ ALPHAZERO")
print("=" * 70)
print()

try:
    # Получаем список активных сессий
    response = requests.get('http://localhost:8000/api/training/alphazero/sessions', timeout=5)
    if response.status_code != 200:
        print("❌ Не удалось получить список сессий")
        print(f"   Статус: {response.status_code}")
        exit(1)
    
    sessions = response.json().get('sessions', {})
    
    if not sessions:
        print("ℹ️  Нет активных сессий обучения")
        print()
        print("Проверь метрики последней тренировки:")
        try:
            metrics_response = requests.get('http://localhost:8000/api/training/alphazero/metrics', timeout=5)
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                summary = metrics.get('summary', {})
                if summary:
                    print(f"   Последняя итерация: {summary.get('latest_iteration', 'N/A')}")
                    print(f"   Policy Loss: {summary.get('latest_policy_loss', 0):.4f}")
                    print(f"   Value Loss: {summary.get('latest_value_loss', 0):.4f}")
                    print(f"   Win Rate: {summary.get('latest_win_rate', 0)*100:.1f}%")
        except:
            pass
        exit(0)
    
    # Берем первую активную сессию
    task_id = list(sessions.keys())[0]
    session = sessions[task_id]
    
    print(f"📊 Активная сессия: {task_id[:8]}...")
    print()
    
    # Мониторинг в реальном времени
    print("🔄 Мониторинг в реальном времени (Ctrl+C для остановки):")
    print()
    
    start_time = time.time()
    last_iteration = 0
    
    try:
        while True:
            try:
                status_response = requests.get(
                    f'http://localhost:8000/api/training/alphazero/sessions/{task_id}',
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    
                    current_iter = status.get('current_iteration', 0)
                    total_iters = status.get('total_iterations', 0)
                    progress = status.get('progress', 0)
                    status_text = status.get('status', 'unknown')
                    
                    # Вычисляем скорость
                    elapsed = time.time() - start_time
                    if current_iter > last_iteration and elapsed > 0:
                        iter_per_sec = (current_iter - last_iteration) / elapsed
                        eta_seconds = (total_iters - current_iter) / iter_per_sec if iter_per_sec > 0 else 0
                        eta_str = format_time(eta_seconds)
                    else:
                        iter_per_sec = 0
                        eta_str = "вычисляется..."
                    
                    # Очищаем строку и выводим новую
                    print("\r" + " " * 70, end="")  # Очистка
                    print(f"\r📈 Итерация: {current_iter}/{total_iters} ({progress:.1f}%) | "
                          f"Статус: {status_text} | "
                          f"ETA: {eta_str}", end="", flush=True)
                    
                    last_iteration = current_iter
                    start_time = time.time()
                    
                    # Если завершено
                    if status_text in ['completed', 'error', 'stopped']:
                        print()
                        print()
                        print(f"✅ Обучение {status_text}")
                        break
                
                time.sleep(2)
                
            except KeyboardInterrupt:
                print()
                print()
                print("⏸️  Мониторинг остановлен")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                time.sleep(2)
    
    except KeyboardInterrupt:
        print()
        print()
        print("⏸️  Мониторинг остановлен")
    
    print()
    print("=" * 70)
    
    # Показываем финальные метрики
    print()
    print("📊 Финальные метрики:")
    try:
        metrics_response = requests.get('http://localhost:8000/api/training/alphazero/metrics', timeout=5)
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            summary = metrics.get('summary', {})
            if summary:
                print(f"   Итераций: {summary.get('latest_iteration', 0)}")
                print(f"   Policy Loss: {summary.get('latest_policy_loss', 0):.4f}")
                print(f"   Value Loss: {summary.get('latest_value_loss', 0):.4f}")
                print(f"   Win Rate: {summary.get('latest_win_rate', 0)*100:.1f}%")
                print(f"   Всего примеров: {summary.get('total_examples', 0):,}")
                
                # Показываем лучший чекпойнт
                best = summary.get('best_checkpoint')
                if best:
                    print()
                    print(f"   🏆 Лучший чекпойнт: iter {best.get('iteration', 0)}")
                    print(f"      Policy Loss: {best.get('policy_loss', 0):.4f}")
    except:
        pass

except requests.exceptions.ConnectionError:
    print("❌ Не удалось подключиться к серверу")
    print("   Убедись что сервер запущен на http://localhost:8000")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
