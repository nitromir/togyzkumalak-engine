#!/usr/bin/env python3
"""
Исправление конфликтов Git при обновлении
"""

import os
import subprocess
import sys

print("=" * 60)
print("  ИСПРАВЛЕНИЕ КОНФЛИКТОВ GIT")
print("=" * 60)
print()

# Переходим в корень репозитория
repo_root = '/workspace/togyzkumalak'
os.chdir(repo_root)

print("1️⃣ Проверяю статус Git...")
result = subprocess.run(['git', 'status'], capture_output=True, text=True)
print(result.stdout)

print()

# Проверяем есть ли конфликтующие файлы
print("2️⃣ Ищу конфликтующие файлы...")
conflict_file = 'togyzkumalak-engine/deploy/auto_server.sh'

if os.path.exists(conflict_file):
    print(f"   Найден файл: {conflict_file}")
    
    # Проверяем есть ли изменения
    result = subprocess.run(['git', 'diff', conflict_file], capture_output=True, text=True)
    if result.stdout:
        print("   ⚠️ Файл имеет локальные изменения")
        print("   Решение: откатываю изменения...")
        
        # Откатываем изменения
        subprocess.run(['git', 'checkout', '--', conflict_file], capture_output=True)
        print("   ✅ Изменения откачены")
    else:
        print("   ✅ Файл без изменений")
else:
    print("   ⚠️ Файл не найден (возможно уже удален)")

print()

# Проверяем все незакоммиченные изменения
print("3️⃣ Проверяю все незакоммиченные изменения...")
result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
if result.stdout.strip():
    print("   Найдены незакоммиченные изменения:")
    for line in result.stdout.strip().split('\n'):
        print(f"   {line}")
    
    print()
    print("   Откатываю все незакоммиченные изменения...")
    subprocess.run(['git', 'checkout', '.'], capture_output=True)
    print("   ✅ Все изменения откачены")
else:
    print("   ✅ Нет незакоммиченных изменений")

print()

# Теперь пробуем обновить
print("4️⃣ Обновляю код с GitHub...")
result = subprocess.run(
    ['git', 'pull', 'origin', 'master'],
    capture_output=True,
    text=True,
    timeout=60
)

if result.returncode == 0:
    print("   ✅ Код успешно обновлен!")
    if result.stdout:
        print(f"   {result.stdout[:300]}")
else:
    print(f"   ❌ Ошибка обновления: {result.stderr}")
    print()
    print("   Попробуй вручную:")
    print("   cd /workspace/togyzkumalak")
    print("   git reset --hard origin/master")
    print("   git pull origin master")

print()
print("=" * 60)
