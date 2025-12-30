# 🚀 Актуальные команды для Vast.ai (AlphaZero)

Это руководство содержит **только** работающие команды для запуска обучения на новой машине Vast.ai.

## 1. Подготовка (на новом сервере)

Если ты купил новую машину, сначала клонируй проект и установи зависимости:

```bash
# В терминале Vast.ai (не в Jupyter)
cd /workspace
git clone https://github.com/nitromir/togyzkumalak-engine
cd togyzkumalak-engine/togyzkumalak-engine
pip install -r requirements.txt
pip install gym==0.26.2
```

## 2. Магический скрипт запуска (в Jupyter)

Создай в Jupyter новую ячейку (Python) и запусти её. Этот скрипт сам всё почистит, обновит код, запустит сервер и стартанет обучение с нужной архитектурой (256-256-128).

```python
import os, requests, time, subprocess, sys

# 1. Настройка путей
project_dir = '/workspace/togyzkumalak-engine/togyzkumalak-engine'
if not os.path.exists(project_dir): project_dir = '/root/togyzkumalak-engine'
os.chdir(project_dir)

print("🛑 1. Чистим остатки (Юпитер НЕ трогаем)...")
# Убиваем только процессы обучения и сервера
os.system("pkill -9 -f run.py")
os.system("pkill -9 -f alphazero_trainer.py")
os.system("pkill -9 -f multiprocessing")
time.sleep(3)

print("📥 2. СИНХРОНИЗИРУЕМ ФИНАЛЬНЫЙ КОД С GITHUB...")
os.system("git fetch origin master")
os.system("git reset --hard origin/master")

print("\n🚀 3. ЗАПУСКАЕМ СЕРВЕР...")
# Запуск через sys.executable гарантирует использование правильного python
with open('server_error.log', 'w') as err_file:
    subprocess.Popen([sys.executable, 'run.py'], 
                     stdout=open('server.log', 'w'), 
                     stderr=err_file, 
                     start_new_session=True, 
                     cwd=project_dir)

print("⏳ Ожидание API (Health Check)...")
for i in range(20):
    try:
        if requests.get('http://localhost:8000/api/health', timeout=1).status_code == 200:
            print(f"✅ Сервер готов!")
            break
    except:
        time.sleep(2)

print("\n🔥 4. ЗАПУСКАЕМ ОБУЧЕНИЕ (4 GPU / BLITZ)...")
try:
    # Оптимальные параметры для 4x 4090:
    config = {
        "numIters": 100,
        "numEps": 440,        # 11 игр на каждый из 40 воркеров
        "numMCTSSims": 100,    # Глубокие раздумья для качества
        "batch_size": 4096,
        "epochs": 15,
        "num_workers": 44,     # Максимальная нагрузка на CPU/GPU
        "use_bootstrap": False, # Если уже есть чекпойнты, бутстрап не нужен
        "resume_from_checkpoint": True # ПРОДОЛЖИТЬ С ТОГО ЖЕ МЕСТА
    }
    r = requests.post('http://localhost:8000/api/training/alphazero/start', json=config)
    print(f"✅ СТАТУС ЗАПУСКА: {r.json()}")
    
    # Ждем немного и проверяем логи загрузки модели
    time.sleep(5)
    print("\n🔍 ПРОВЕРКА ЗАГРУЗКИ МОДЕЛИ:")
    if os.path.exists('server_error.log'):
        with open('server_error.log', 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                if "matched" in line:
                    print(f"🎯 ПОДТВЕРЖДЕНО: {line.strip()}")
                if "SUCCESSFULLY LOADED" in line:
                    print(f"🚀 {line.strip()}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
```

## 3. Как проверить, что всё работает?

Если ты видишь в логах (через скрипт выше или вручную) фразу:
`Checkpoint loaded: ... (27 layers matched)`
— это значит, что ИИ загрузился правильно и продолжает обучение.

## 4. SSH-тоннель (на твоём ПК в PowerShell)

Чтобы синхронизация чекпойнтов на твой компьютер работала, всегда держи этот тоннель открытым:

```powershell
# Замени ПОРТ и IP на данные новой машины из Vast.ai
ssh -p ПОРТ root@IP -L 8080:localhost:8000
```

---
*Обновлено: 30.12.2025. Архитектура: 256-256-128. Режим: 4 GPU Blitz.*
