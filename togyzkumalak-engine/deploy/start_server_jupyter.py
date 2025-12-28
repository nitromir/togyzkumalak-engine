# Запуск сервера из Jupyter Notebook
# Скопируй и выполни этот код в ячейке

import subprocess
import os
import time
import requests
from IPython.display import display, HTML
import sys

# Переходим в папку проекта
os.chdir('/workspace/togyzkumalak/togyzkumalak-engine')

# Находим python (может быть python3 или полный путь)
python_cmd = sys.executable  # Используем тот же python что и Jupyter

print(f"Using Python: {python_cmd}")
print(f"Working directory: {os.getcwd()}")

# Проверяем запущен ли сервер
try:
    response = requests.get('http://localhost:8000/api/health', timeout=2)
    if response.status_code == 200:
        print("✅ Server is already running!")
        display(HTML('<iframe src="http://localhost:8000" width="100%" height="800" style="border:none"></iframe>'))
    else:
        raise Exception("Server not responding")
except Exception as e:
    print(f"🚀 Starting server... (previous check: {e})")
    
    # Запускаем сервер в фоне
    try:
        process = subprocess.Popen(
            [python_cmd, 'run.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd='/workspace/togyzkumalak/togyzkumalak-engine'
        )
        print(f"✅ Server process started (PID: {process.pid})")
        
        # Ждём запуска
        print("Waiting for server to start...")
        for i in range(15):
            time.sleep(1)
            try:
                response = requests.get('http://localhost:8000/api/health', timeout=1)
                if response.status_code == 200:
                    print("✅ Server started successfully!")
                    display(HTML('<iframe src="http://localhost:8000" width="100%" height="800" style="border:none"></iframe>'))
                    break
            except:
                if i == 14:
                    print("⚠ Server is starting but not ready yet. Try again in a few seconds.")
                    print(f"Or check terminal output. Process PID: {process.pid}")
                continue
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("\nTry running in Terminal instead:")
        print("cd /workspace/togyzkumalak/togyzkumalak-engine")
        print("python run.py")
