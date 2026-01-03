# 🔐 SSH Connection Guide

## Твои данные для подключения:

```
Host: 151.237.25.234
Port: 23396
User: root
```

---

## 🪟 Windows PowerShell

### Вариант 1: Прямое подключение с туннелем

```powershell
ssh -p 23396 root@151.237.25.234 -L 8000:localhost:8000
```

**Что делает:**
- Подключается к серверу
- Создаёт туннель: твой `localhost:8000` → сервер `localhost:8000`
- После подключения можешь открыть http://localhost:8000 в браузере

### Вариант 2: Через прокси (если прямое не работает)

```powershell
ssh -p 16593 root@ssh7.vast.ai
```

Потом внутри:
```bash
ssh root@151.237.25.234 -p 23396
```

---

## 🐧 Linux/Mac

```bash
ssh -p 23396 root@151.237.25.234 -L 8000:localhost:8000
```

---

## ❓ Если SSH не работает

### Проблема: "Permission denied" или просит пароль

**Решение 1: Используй Jupyter (проще!)**
1. На Vast.ai нажми зелёную кнопку **"Open"**
2. Откроется Jupyter в браузере
3. **New → Terminal**
4. Готово! Ты в терминале сервера

**Решение 2: Настрой SSH ключ**

На Windows PowerShell:
```powershell
# Генерируем ключ (если нет)
ssh-keygen -t rsa -b 4096 -C "vastai-key"

# Копируем публичный ключ
Get-Content ~\.ssh\id_rsa.pub | Set-Clipboard
```

Потом на Vast.ai:
1. Settings → SSH Keys
2. Add Key
3. Вставь скопированный ключ

---

## 🚀 Быстрая установка после подключения

После того как подключился, выполни:

```bash
cd /workspace && \
git clone https://github.com/nitromir/togyzkumalak-engine.git togyzkumalak && \
cd togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine && \
pip install -q -r requirements.txt && \
mkdir -p models/alphazero logs/{alphazero,games,gemini_battles/{games,sessions,summaries},self_play,training,wandb_local,ab_tests/results} training_data && \
nohup python run.py > server.log 2>&1 & \
echo "✅ Server starting! Check: tail -f server.log"
```

---

## 📋 Чеклист подключения

- [ ] Открыл PowerShell
- [ ] Выполнил: `ssh -p 23396 root@151.237.25.234 -L 8000:localhost:8000`
- [ ] Увидел приглашение: `root@...:~#`
- [ ] Выполнил команду установки (выше)
- [ ] Открыл http://localhost:8000 в браузере
- [ ] Увидел интерфейс Togyzkumalak! 🎉

---

## 🔧 Полезные команды

**Проверить что сервер запущен:**
```bash
ps aux | grep python
```

**Посмотреть логи:**
```bash
tail -f server.log
```

**Остановить сервер:**
```bash
pkill -f "python run.py"
```

**Перезапустить:**
```bash
cd /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine
nohup python run.py > server.log 2>&1 &
```

---

## 💡 Совет

Если SSH не работает сразу, используй **Jupyter Terminal** - это самый простой способ попасть на сервер!
