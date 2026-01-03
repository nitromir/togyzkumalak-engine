# 🚀 Запуск сервера из Jupyter Notebook

## Способ 1: Через Python код (в ячейке)

Выполни этот код в ячейке Jupyter:

```python
import subprocess
import os
import time
import requests
from IPython.display import display, HTML

# Переходим в папку проекта
os.chdir('/workspace/togyzkumalak/togyzkumalak-engine')

# Проверяем запущен ли сервер
try:
    response = requests.get('http://localhost:8000/api/health', timeout=2)
    print("✅ Server is already running!")
except:
    print("🚀 Starting server...")
    # Запускаем сервер
    subprocess.Popen(['python', 'run.py'], 
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    # Ждём запуска
    for i in range(10):
        time.sleep(1)
        try:
            requests.get('http://localhost:8000/api/health', timeout=1)
            print("✅ Server started!")
            break
        except:
            continue

# Открываем UI
display(HTML('<iframe src="http://localhost:8000" width="100%" height="800" style="border:none"></iframe>'))
```

---

## Способ 2: Через Terminal в Jupyter

1. В Jupyter: **New → Terminal**
2. Выполни:
   ```bash
   cd /workspace/togyzkumalak/togyzkumalak-engine
   python run.py
   ```
3. Сервер запустится в этом терминале
4. В другом терминале или Notebook выполни:
   ```python
   from IPython.display import IFrame
   IFrame('http://localhost:8000', width='100%', height=800)
   ```

---

## Способ 3: Запуск в фоне через Terminal

В Jupyter Terminal:

```bash
cd /workspace/togyzkumalak/togyzkumalak-engine
nohup python run.py > server.log 2>&1 &
sleep 5
tail -20 server.log
```

Потом в Notebook:
```python
from IPython.display import IFrame
IFrame('http://localhost:8000', width='100%', height=800)
```
