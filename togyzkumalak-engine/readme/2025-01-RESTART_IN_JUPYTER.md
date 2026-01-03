# 🔄 Как перезапустить сервер в Jupyter Notebook

## Быстрый способ

1. **Открой Jupyter Notebook на Vast.ai** (где запущен сервер)
2. **Создай новую ячейку** и вставь:

```python
%run deploy/restart_server_jupyter.py
```

Или скопируй весь код из `deploy/restart_server_jupyter.py` в ячейку и выполни.

---

## Альтернативный способ (вручную)

### Шаг 1: Останови текущий сервер

В новой ячейке Jupyter:

```python
import subprocess
import os

# Найди и останови процесс
result = subprocess.run(['pgrep', '-f', 'python.*run.py'], capture_output=True, text=True)
if result.returncode == 0:
    pids = result.stdout.strip().split('\n')
    for pid in pids:
        if pid:
            os.kill(int(pid), 15)  # SIGTERM
            print(f"Остановлен процесс {pid}")
```

### Шаг 2: Обнови код

```python
import subprocess
import os

# Перейди в корень репозитория
os.chdir('/workspace/togyzkumalak/togyzkumalak-engine')
os.chdir('..')  # В корень репозитория

# Обнови с GitHub
result = subprocess.run(['git', 'pull', 'origin', 'master'], capture_output=True, text=True)
print(result.stdout)
```

### Шаг 3: Запусти сервер заново

```python
import subprocess
import sys
import os

# Перейди в директорию сервера
os.chdir('/workspace/togyzkumalak/togyzkumalak-engine')

# Запусти сервер
python_exe = sys.executable
process = subprocess.Popen(
    [python_exe, 'run.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print(f"Сервер запущен! PID: {process.pid}")
print("Проверь http://localhost:8000/api/health")
```

---

## Еще проще: одна команда

Скопируй это в ячейку Jupyter:

```python
import os
import subprocess
import sys
import time
import requests

# 1. Останови старый сервер
subprocess.run(['pkill', '-f', 'python.*run.py'], capture_output=True)
time.sleep(2)

# 2. Обнови код
os.chdir('/workspace/togyzkumalak')
subprocess.run(['git', 'pull', 'origin', 'master'], capture_output=True)

# 3. Запусти новый
os.chdir('togyzkumalak-engine')
python_exe = sys.executable
subprocess.Popen([python_exe, 'run.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 4. Проверь
time.sleep(5)
try:
    r = requests.get('http://localhost:8000/api/health', timeout=2)
    print("✅ Сервер запущен!")
except:
    print("⏳ Сервер запускается...")
```

---

## Важно!

- **Обновление страницы в браузере на твоем компьютере НЕ перезапускает сервер**
- Сервер работает в Jupyter Notebook на удаленном сервере
- После перезапуска в Jupyter, **тогда** обнови страницу в браузере

---

## Проверка что сервер работает

В ячейке Jupyter:

```python
import requests
try:
    r = requests.get('http://localhost:8000/api/health')
    print("✅ Сервер работает:", r.json())
except:
    print("❌ Сервер не отвечает")
```
