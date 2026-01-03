#!/usr/bin/env python3
"""
Проверка доступности PROBS Ultra endpoint и диагностика проблем
"""

import os
import sys
import requests
import subprocess
import json

print("=" * 80)
print("  ДИАГНОСТИКА PROBS ULTRA ENDPOINT")
print("=" * 80)
print()

# 1. Проверка, что сервер запущен
print("1️⃣ Проверка сервера...")
try:
    response = requests.get('http://localhost:8000/api/health', timeout=3)
    print(f"   ✅ Сервер отвечает! Статус: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("   ❌ Сервер не запущен или недоступен на порту 8000")
    print("   💡 Запустите: cd /workspace/togyzkumalak-engine/togyzkumalak-engine && python run.py")
    sys.exit(1)
except Exception as e:
    print(f"   ⚠️ Ошибка подключения: {e}")
    sys.exit(1)

print()

# 2. Проверка FastAPI документации (все endpoints)
print("2️⃣ Проверка зарегистрированных endpoints...")
try:
    response = requests.get('http://localhost:8000/docs', timeout=3)
    if response.status_code == 200:
        print("   ✅ FastAPI docs доступны: http://localhost:8000/docs")
        print("   💡 Откройте в браузере для просмотра всех endpoints")
    else:
        print(f"   ⚠️ Docs недоступны: {response.status_code}")
except Exception as e:
    print(f"   ⚠️ Не удалось проверить docs: {e}")

print()

# 3. Проверка OpenAPI схемы (список всех endpoints)
print("3️⃣ Проверка OpenAPI схемы (список endpoints)...")
try:
    response = requests.get('http://localhost:8000/openapi.json', timeout=3)
    if response.status_code == 200:
        openapi = response.json()
        paths = openapi.get('paths', {})
        
        # Ищем PROBS endpoints
        probs_endpoints = [path for path in paths.keys() if 'probs' in path.lower()]
        ultra_endpoints = [path for path in paths.keys() if 'ultra' in path.lower()]
        
        print(f"   ✅ Найдено PROBS endpoints: {len(probs_endpoints)}")
        for ep in probs_endpoints:
            print(f"      - {ep}")
        
        if ultra_endpoints:
            print(f"   ✅ Найдено ULTRA endpoints: {len(ultra_endpoints)}")
            for ep in ultra_endpoints:
                print(f"      - {ep}")
        else:
            print("   ❌ ULTRA endpoints НЕ НАЙДЕНЫ!")
            print("   💡 Возможно, код не обновлен или сервер не перезапущен")
    else:
        print(f"   ⚠️ OpenAPI недоступен: {response.status_code}")
except Exception as e:
    print(f"   ⚠️ Ошибка проверки OpenAPI: {e}")

print()

# 4. Прямая проверка endpoint
print("4️⃣ Прямая проверка /api/training/probs/ultra/start...")
try:
    response = requests.post(
        'http://localhost:8000/api/training/probs/ultra/start',
        json={},
        headers={'Content-Type': 'application/json'},
        timeout=5
    )
    print(f"   Статус: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ Endpoint работает! Ответ: {response.json()}")
    elif response.status_code == 404:
        print("   ❌ 404 Not Found - endpoint не зарегистрирован")
        print("   💡 Проверьте, что:")
        print("      1. Код обновлен: git pull origin master")
        print("      2. Сервер перезапущен после обновления")
        print("      3. Нет ошибок при импорте модулей (см. логи ниже)")
    elif response.status_code == 422:
        print(f"   ⚠️ 422 Validation Error - endpoint существует, но данные неверны")
        print(f"   Ответ: {response.json()}")
    else:
        print(f"   ⚠️ Неожиданный статус: {response.status_code}")
        print(f"   Ответ: {response.text[:200]}")
except requests.exceptions.RequestException as e:
    print(f"   ❌ Ошибка запроса: {e}")

print()

# 5. Проверка логов сервера на ошибки
print("5️⃣ Проверка логов сервера...")
project_dir = '/workspace/togyzkumalak-engine/togyzkumalak-engine'
if not os.path.exists(project_dir):
    project_dir = '/root/togyzkumalak-engine'

log_files = [
    os.path.join(project_dir, 'server_error.log'),
    os.path.join(project_dir, 'server.log'),
    os.path.join(project_dir, 'nohup.out'),
]

for log_file in log_files:
    if os.path.exists(log_file):
        print(f"   📋 {os.path.basename(log_file)} (последние 30 строк):")
        print("   " + "-" * 76)
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    # Показываем только важные строки
                    if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'import', 'module', 'ultra', 'probs']):
                        print(f"   {line.rstrip()}")
        except Exception as e:
            print(f"   ⚠️ Не удалось прочитать: {e}")
        print()

# 6. Проверка кода на удаленной машине
print("6️⃣ Проверка версии кода...")
try:
    result = subprocess.run(
        ['git', 'log', '--oneline', '-1'],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print(f"   Последний коммит: {result.stdout.strip()}")
        
        # Проверяем, есть ли метод start_ultra_training
        probs_file = os.path.join(project_dir, 'backend', 'probs_task_manager.py')
        if os.path.exists(probs_file):
            with open(probs_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'start_ultra_training' in content:
                    print("   ✅ Метод start_ultra_training найден в коде")
                else:
                    print("   ❌ Метод start_ultra_training НЕ найден в коде!")
                    print("   💡 Выполните: git pull origin master")
                
                if 'def start_ultra_training' in content:
                    print("   ✅ Функция start_ultra_training определена")
                else:
                    print("   ❌ Функция start_ultra_training НЕ определена!")
    else:
        print("   ⚠️ Не удалось проверить git")
except Exception as e:
    print(f"   ⚠️ Ошибка проверки кода: {e}")

print()
print("=" * 80)
print("  РЕКОМЕНДАЦИИ")
print("=" * 80)
print()
print("Если endpoint не найден (404):")
print("1. Обновите код: cd /workspace/togyzkumalak-engine/togyzkumalak-engine && git pull origin master")
print("2. Перезапустите сервер: pkill -f run.py && sleep 2 && python run.py")
print("3. Проверьте логи на ошибки импорта модулей")
print()
print("Если endpoint найден, но выдает ошибку:")
print("1. Проверьте логи выше на наличие ошибок")
print("2. Убедитесь, что AlphaZero чекпойнт доступен")
print("3. Проверьте, что метод start_ultra_training существует в probs_task_manager.py")
print()
