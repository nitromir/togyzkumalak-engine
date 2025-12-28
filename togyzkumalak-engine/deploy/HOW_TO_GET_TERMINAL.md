# Как получить терминал для управления сервером

## Вариант 1: Jupyter Terminal (САМЫЙ ПРОСТОЙ) ✅

1. Открой Jupyter Notebook на Vast.ai (где запущен сервер)
2. Нажми кнопку **"New"** → **"Terminal"**
3. Готово! Терминал открыт

**Затем выполни:**
```bash
cd /workspace/togyzkumalak && git pull origin master && cd togyzkumalak-engine/deploy && chmod +x auto_server.sh && ./auto_server.sh
```

---

## Вариант 2: SSH с твоего компьютера

### Windows PowerShell:

```powershell
ssh -p 23396 root@151.237.25.234
```

Если просит ключ, используй:
```powershell
ssh -p 23396 -i $env:USERPROFILE\.ssh\id_rsa root@151.237.25.234
```

---

## Вариант 3: Через Vast.ai веб-интерфейс

1. Зайди на https://vast.ai
2. Открой свой инстанс (instance)
3. Найди кнопку **"SSH"** или **"Terminal"**
4. Откроется терминал прямо в браузере

---

## Рекомендация

**Используй Вариант 1 (Jupyter Terminal)** - это самый простой способ, и у тебя уже есть доступ к Jupyter.

После открытия терминала выполни команду выше - скрипт будет автоматически управлять сервером и показывать статус обучения.
