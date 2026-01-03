# 🚀 Quick Start Guide - 16x RTX 3090

## Шаг 1: Подключение к серверу

### Вариант A: Прямое подключение (рекомендуется)
```bash
ssh -p 23396 root@151.237.25.234 -L 8000:localhost:8000
```

### Вариант B: Через прокси
```bash
ssh -p 16593 root@ssh7.vast.ai -L 8000:localhost:8000
```

**Важно:** Используй порт `8000` для туннеля (наш сервер работает на 8000, не 8080)

---

## Шаг 2: На сервере - быстрая установка

Скопируй и выполни **ОДНОЙ КОМАНДОЙ**:

```bash
cd /workspace && git clone https://github.com/nitromir/togyzkumalak-engine.git togyzkumalak && cd togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine && pip install -q -r requirements.txt && mkdir -p models/alphazero logs/{alphazero,games,gemini_battles/{games,sessions,summaries},self_play,training,wandb_local,ab_tests/results} training_data && echo "✅ Ready! Run: python run.py"
```

Или используй готовый скрипт:
```bash
curl -sSL https://raw.githubusercontent.com/nitromir/togyzkumalak-engine/master/gym-togyzkumalak-master/togyzkumalak-engine/deploy/vastai_quick_setup.sh | bash
```

---

## Шаг 3: Запуск сервера

```bash
cd /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine
python run.py
```

**Или в фоне:**
```bash
nohup python run.py > server.log 2>&1 &
```

---

## Шаг 4: Открой UI

На **своём компьютере** открой браузер:
```
http://localhost:8000
```

---

## Шаг 5: Запуск обучения

### Через UI:
1. Открой вкладку **🧠 Тренировка**
2. Нажми **"⚡ Авто-конфиг для GPU"**
3. Введи `1` (час)
4. Нажми **"🚀 Запустить AlphaZero"**

### Или через скрипт:
```bash
./deploy/start_training_16x3090.sh
```

---

## Шаг 6: Мониторинг

### На сервере (в новом терминале):
```bash
python deploy/monitor.py
```

### На своём компе (PowerShell):
```powershell
.\deploy\sync_checkpoints.ps1 -SshConnection "root@151.237.25.234 -p 23396"
```

---

## 🔧 Если SSH не работает

Если при подключении просит пароль или не подключается:

1. **Попробуй через Jupyter:**
   - Нажми зелёную кнопку **"Open"** на Vast.ai
   - В Jupyter: **New → Terminal**

2. **Или используй Proxy SSH:**
   ```bash
   ssh -p 16593 root@ssh7.vast.ai
   ```
   Потом внутри:
   ```bash
   ssh root@151.237.25.234 -p 23396
   ```

---

## ⚡ Быстрая команда для копипаста

**Всё в одной строке (установка + запуск):**

```bash
cd /workspace && git clone https://github.com/nitromir/togyzkumalak-engine.git togyzkumalak 2>/dev/null || (cd togyzkumalak && git pull) && cd togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine && pip install -q -r requirements.txt && mkdir -p models/alphazero logs/{alphazero,games,gemini_battles/{games,sessions,summaries},self_play,training,wandb_local,ab_tests/results} training_data && nohup python run.py > server.log 2>&1 & sleep 3 && echo "✅ Server started! Access: http://localhost:8000" && tail -f server.log
```
