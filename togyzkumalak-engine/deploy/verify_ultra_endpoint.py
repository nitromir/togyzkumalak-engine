#!/usr/bin/env python3
"""
Быстрая проверка, что PROBS Ultra endpoint доступен
"""

import requests
import sys

print("=" * 60)
print("  ПРОВЕРКА PROBS ULTRA ENDPOINT")
print("=" * 60)
print()

# Проверка endpoint
try:
    print("1. Проверяю /api/training/probs/ultra/start...")
    response = requests.post(
        'http://localhost:8000/api/training/probs/ultra/start',
        json={},
        headers={'Content-Type': 'application/json'},
        timeout=5
    )
    
    print(f"   Статус: {response.status_code}")
    
    if response.status_code == 200:
        print("   ✅ Endpoint работает!")
        print(f"   Ответ: {response.json()}")
        sys.exit(0)
    elif response.status_code == 404:
        print("   ❌ 404 Not Found - endpoint не найден")
        print()
        print("   💡 Решение:")
        print("   1. Найдите правильную директорию проекта:")
        print("      cd /workspace")
        print("      find . -name 'run.py' -type f 2>/dev/null | head -1")
        print()
        print("   2. Обычно это одна из:")
        print("      cd /workspace/togyzkumalak-engine/togyzkumalak-engine")
        print("      # или")
        print("      cd /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine")
        print()
        print("   3. Обновите код:")
        print("      git pull origin master")
        print("      git log --oneline -3  # Должны быть коммиты с 'PROBS Ultra'")
        print()
        print("   4. Перезапустите сервер:")
        print("      pkill -f run.py")
        print("      sleep 3")
        print("      source /venv/main/bin/activate")
        print("      export PORT=8000")
        print("      python run.py")
        print()
        print("   3. Проверьте, что endpoint зарегистрирован:")
        print("      grep -n 'ultra/start' backend/main.py")
        sys.exit(1)
    else:
        print(f"   ⚠️ Неожиданный статус: {response.status_code}")
        print(f"   Ответ: {response.text[:200]}")
        sys.exit(1)
        
except requests.exceptions.ConnectionError:
    print("   ❌ Сервер не запущен или недоступен на порту 8000")
    print("   💡 Запустите сервер: python run.py")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    sys.exit(1)
