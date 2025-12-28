#!/usr/bin/env python3
"""
Найти где находится git репозиторий
"""

import os

print("=" * 60)
print("  ПОИСК GIT РЕПОЗИТОРИЯ")
print("=" * 60)
print()

# Проверяем разные пути
paths_to_check = [
    '/workspace/togyzkumalak',
    '/workspace/togyzkumalak/togyzkumalak-engine',
    '/root/togyzkumalak',
    os.getcwd(),
    os.path.dirname(os.getcwd()),
    os.path.dirname(os.path.dirname(os.getcwd())),
]

print("Проверяю пути:")
for path in paths_to_check:
    if path and os.path.exists(path):
        git_path = os.path.join(path, '.git')
        exists = os.path.exists(git_path)
        print(f"  {path}")
        print(f"    Существует: ✅")
        print(f"    .git существует: {'✅' if exists else '❌'}")
        if exists:
            print(f"    ✅ НАЙДЕН GIT РЕПОЗИТОРИЙ!")
        print()

# Также проверим текущую директорию движка
engine_dir = '/workspace/togyzkumalak/togyzkumalak-engine'
if os.path.exists(engine_dir):
    print(f"Директория движка: {engine_dir}")
    print(f"  Существует: ✅")
    
    # Попробуем найти .git на уровень выше
    parent = os.path.dirname(engine_dir)
    parent_git = os.path.join(parent, '.git')
    print(f"  Родительская директория: {parent}")
    print(f"  .git в родительской: {'✅' if os.path.exists(parent_git) else '❌'}")
    
    # Попробуем найти .git на два уровня выше
    grandparent = os.path.dirname(parent)
    grandparent_git = os.path.join(grandparent, '.git')
    print(f"  Директория на 2 уровня выше: {grandparent}")
    print(f"  .git там: {'✅' if os.path.exists(grandparent_git) else '❌'}")

print()
print("=" * 60)
