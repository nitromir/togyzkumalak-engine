# 🚀 Установка через Jupyter Terminal (САМЫЙ ПРОСТОЙ СПОСОБ!)

## Шаг 1: Открой Jupyter

1. На Vast.ai нажми зелёную кнопку **"Open"**
2. Откроется Jupyter в браузере
3. Нажми **"New"** → **"Terminal"**

Готово! Ты в терминале сервера! 🎉

---

## Шаг 2: Установка (скопируй ВСЁ одной командой)

```bash
cd /workspace && git clone https://github.com/nitromir/togyzkumalak-engine.git togyzkumalak && cd togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine && pip install -q -r requirements.txt && mkdir -p models/alphazero logs/{alphazero,games,gemini_battles/{games,sessions,summaries},self_play,training,wandb_local,ab_tests/results} training_data && nohup python run.py > server.log 2>&1 & sleep 3 && tail -5 server.log
```

---

## Шаг 3: Проверь что сервер запустился

Должно появиться:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Шаг 4: Открой UI

В Jupyter нажми **"New"** → **"Notebook"** или просто открой в новом табе:
```
http://151.237.25.234:8000
```

Или используй туннель (если настроил):
```
http://localhost:8000
```

---

## Шаг 5: Запуск обучения

В Jupyter Terminal:

```bash
cd /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine
./deploy/start_training_16x3090.sh
```

Или через UI:
1. Открой http://151.237.25.234:8000
2. Вкладка **🧠 Тренировка**
3. **"⚡ Авто-конфиг для GPU"** → введи `1`
4. **"🚀 Запустить AlphaZero"**

---

## Мониторинг

В Jupyter Terminal (новый терминал):

```bash
cd /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine
python deploy/monitor.py
```

---

## 💡 Преимущества Jupyter Terminal:

✅ Не нужен SSH ключ
✅ Работает сразу
✅ Можно открыть несколько терминалов
✅ Можно загружать файлы через UI
✅ Можно скачивать файлы через UI
