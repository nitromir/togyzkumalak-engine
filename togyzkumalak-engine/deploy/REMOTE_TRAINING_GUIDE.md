# 🚀 Полное руководство по удалённому обучению AlphaZero на Vast.ai

## ⚡ Быстрый старт (Копипасти и запускай)

### 1. SSH-туннель (на своём ПК)
**Новый сервер (от 29.12.2025):**
```powershell
ssh -p 45511 root@171.226.152.139 -L 8000:localhost:8000
```

### 🔑 Справочная информация (SSH Keys)
**Instance Public Key:**
`ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCKbN+08Z0mz2xspbGhQK8Spc5XgGjQ3CCR9Qx2Z3xD9TSt4Cj/H+c9UFp5lPN6AkwqeoymHBR/qvD0lhtFCj/am+G5bCw6wBSQx7qjw8r5OtaxgwE+GU56rWW28u3DZA3cSuIG6YmJpTFFAOnPYTQVF4/9zkroRw984E3UIfaMi4+wqT8zUTmbx56J0ZzVR4xZsdvPTBO1cHO+zJ6feXJ4ckTApBswklnpUCVRkqF6Qk0RQcF4WGZaTGd4n0PvUvenceYwa7Jf7FPUwupgiYjOO3I4/GZrVRUdQCutlbkzXGgfizx8Cj6WW47ujsr4He4V04/lE2qrQdrHH3E65Ud1 rsa-key-20231218`

### 2. Обновить код на сервере (в Jupyter)
```python
import subprocess, os, time
os.chdir('/workspace/togyzkumalak/togyzkumalak-engine')
# Безопасная очистка: только сервер и обучение, не трогаем Jupyter
subprocess.run("pkill -9 -f run.py", shell=True)
subprocess.run("pkill -9 -f alphazero_trainer", shell=True)
subprocess.run("fuser -k 8000/tcp", shell=True)
time.sleep(2)
subprocess.run(['git', 'checkout', '.'], capture_output=True)
subprocess.run(['git', 'pull', 'origin', 'master'], capture_output=True)
print("✅ Код обновлён!")
```

### 3. Запустить сервер (в Jupyter)
```python
import subprocess, sys, os
os.chdir('/workspace/togyzkumalak/togyzkumalak-engine')
subprocess.Popen([sys.executable, 'run.py'], 
                 stdout=open('server.log', 'w'), 
                 stderr=open('server_error.log', 'w'),
                 start_new_session=True)
print("🚀 Сервер запущен!")
```

### 4. Запустить обучение (в Jupyter)
```python
import requests
blitz_config = {
    "numIters": 20,           # ~2 часа обучения
    "numEps": 128,            # Игр за итерацию (32 на каждую GPU)
    "numMCTSSims": 60,        # Глубина раздумий (баланс)
    "num_parallel_games": 4,  # РАСПРЕДЕЛЕНИЕ ПО 4 GPU
    "use_bootstrap": True     # Начать с человеческих данных
}
r = requests.post('http://localhost:8000/api/training/alphazero/start', json=blitz_config)
print(f"🚀 БЛИЦ ЗАПУЩЕН (4 GPU): {r.json()}")
```

---

## 📊 Мониторинг

### Проверить статус обучения
```python
import requests
r = requests.get('http://localhost:8000/api/training/alphazero/sessions')
print(r.json())
```

### Посмотреть логи ошибок
```bash
tail -n 50 /workspace/togyzkumalak/togyzkumalak-engine/server_error.log
```

### Посмотреть логи обучения
```bash
tail -n 50 /workspace/togyzkumalak/togyzkumalak-engine/server.log
```

### Проверить GPU
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

---

## 🛑 Экстренная остановка

### Убить всё (Сервер и Обучение)
```bash
pkill -9 -f run.py
pkill -9 -f alphazero_trainer
```

### Освободить порт
```bash
fuser -k 8000/tcp
```

---

## 📥 Скачать модель (на своём ПК)
```powershell
scp -P ПОРТ root@IP_АДРЕС:/workspace/togyzkumalak/togyzkumalak-engine/models/alphazero/best.pth.tar C:\Downloads\
```

---

## 🎯 Расчёт времени

| Параметр | Значение | Время 1 итерации |
|----------|----------|------------------|
| MCTS Sims: 30, Episodes: 64 | BLITZ | ~3-5 мин |
| MCTS Sims: 50, Episodes: 100 | Normal | ~10-15 мин |
| MCTS Sims: 100, Episodes: 200 | Quality | ~30-60 мин |

**Для 16 GPU:** множитель ~0.5x (параллельные игры ускоряют)

---

## ⚠️ Частые проблемы

### "Network masked all valid moves"
**Причина:** Сеть ещё не обучена, выдаёт нули.
**Решение:** Это нормально на старте. После 2-3 итераций пройдёт.

### Обучение идёт медленно
**Причина:** Старая версия кода с ProcessPoolExecutor.
**Решение:** Обнови код (`git pull`) и перезапусти сервер.

### GPU не используются
**Причина:** Сервер не видит CUDA.
**Решение:** Проверь `nvidia-smi` и перезапусти сервер.

---

*Обновлено: 29.12.2025 — Исправлен ProcessPoolExecutor, теперь batch self-play в 10-100x быстрее*
