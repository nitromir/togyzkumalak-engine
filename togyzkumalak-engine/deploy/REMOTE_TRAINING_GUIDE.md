# 🚀 Полное руководство по удалённому обучению AlphaZero на Vast.ai

## ⚡ Быстрый старт (Копипасти и запускай)

### 1. SSH-туннель (на своём ПК)
```powershell
ssh -p ПОРТ root@IP_АДРЕС -L 8000:localhost:8000
```

### 2. Обновить код на сервере (в Jupyter)
```python
import subprocess, os, time
os.chdir('/workspace/togyzkumalak/togyzkumalak-engine')
subprocess.run("pkill -9 -f python", shell=True)
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
    "numIters": 24,           # Итераций (24 = ~2 часа)
    "numEps": 64,             # Игр за итерацию
    "numMCTSSims": 30,        # MCTS симуляций (меньше = быстрее)
    "num_parallel_games": 8,  # Параллельных игр
    "use_bootstrap": True     # Начать с человеческих данных
}
r = requests.post('http://localhost:8000/api/training/alphazero/start', json=blitz_config)
print(f"🎮 Обучение запущено: {r.json()}")
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

### Убить всё
```bash
pkill -9 -f python
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
