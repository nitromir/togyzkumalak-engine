# Команды для Terminal в Jupyter

## Проверка статуса обучения

```bash
# Статус сессии
curl -s http://localhost:8000/api/training/alphazero/sessions | python3 -m json.tool

# Детальный статус конкретной сессии
curl -s http://localhost:8000/api/training/alphazero/sessions/az_1766960200 | python3 -m json.tool
```

## Проверка логов

```bash
# Последние 50 строк логов
tail -50 /workspace/togyzkumalak/togyzkumalak-engine/server_error.log

# Поиск ошибок
grep -i "error\|exception\|traceback\|failed" /workspace/togyzkumalak/togyzkumalak-engine/server_error.log | tail -20

# Поиск строк связанных с обучением
grep -i "alphazero\|training\|iteration\|mcts\|self-play" /workspace/togyzkumalak/togyzkumalak-engine/server_error.log | tail -30
```

## Проверка процессов

```bash
# Все процессы Python
ps aux | grep python

# Детальная информация о процессе сервера
ps -p 33020 -o pid,ppid,cmd,%mem,%cpu,etime

# Проверка потоков процесса
ps -T -p 33020 | head -20
```

## Проверка GPU

```bash
# Детальная информация о GPU
nvidia-smi

# Процессы использующие GPU
nvidia-smi pmon -c 1

# Использование памяти GPU
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

## Проверка чекпойнтов

```bash
# Список чекпойнтов
ls -lh /workspace/togyzkumalak/togyzkumalak-engine/models/alphazero/*.pth.tar | tail -10

# Новые чекпойнты (последние 5 минут)
find /workspace/togyzkumalak/togyzkumalak-engine/models/alphazero -name "*.pth.tar" -mmin -5

# Размер директории
du -sh /workspace/togyzkumalak/togyzkumalak-engine/models/alphazero
```

## Остановка и перезапуск

```bash
# Остановка сервера
kill 33020

# Или мягкая остановка
kill -TERM 33020

# Перезапуск сервера
cd /workspace/togyzkumalak/togyzkumalak-engine
/venv/main/bin/python run.py > server.log 2>&1 &
```
