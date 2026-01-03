# 🔄 Перезапуск сервера в Jupyter - Простая инструкция

## Шаг 1: Открой Jupyter Notebook на Vast.ai

Зайди в Jupyter Notebook где запущен сервер.

---

## Шаг 2: Создай новую ячейку

Нажми кнопку **"+"** или **"Code"** чтобы создать новую ячейку.

---

## Шаг 3: Скопируй и вставь этот код

```python
import os
import subprocess
import sys
import time
import requests

# 1. Останови старый сервер
subprocess.run(['pkill', '-f', 'python.*run.py'], capture_output=True)
time.sleep(2)

# 2. Обнови код с GitHub
os.chdir('/workspace/togyzkumalak')
subprocess.run(['git', 'pull', 'origin', 'master'], capture_output=True)

# 3. Запусти новый сервер
os.chdir('togyzkumalak-engine')
python_exe = sys.executable
subprocess.Popen([python_exe, 'run.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 4. Проверь что запустился
time.sleep(5)
try:
    r = requests.get('http://localhost:8000/api/health', timeout=2)
    print("✅ Сервер запущен!")
except:
    print("⏳ Сервер запускается...")
```

---

## Шаг 4: Нажми "Run" (▶️)

Нажми кнопку **"Run"** или **Shift+Enter** чтобы выполнить код.

---

## Готово! 

После этого обнови страницу в браузере на твоем компьютере (F5).

---

## Альтернатива: Используй готовый скрипт

Если скрипт `restart_server_jupyter.py` уже есть на сервере:

```python
%run deploy/restart_server_jupyter.py
```

И нажми **Run** (▶️).
