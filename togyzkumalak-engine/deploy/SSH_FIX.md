# 🔑 Правильная настройка SSH ключа для Vast.ai

## Важно: Два места для ключей!

На Vast.ai есть **ДВА** места где можно добавить SSH ключ:

### 1. Account SSH Keys (общие)
- Settings → SSH Keys
- Работают для всех инстансов

### 2. Instance SSH Keys (для конкретного инстанса) ⭐
- Открой свой инстанс
- В разделе "Config" или "SSH Keys"
- Добавь ключ ТУТ

**Попробуй добавить ключ в Instance SSH Keys!**

---

## Твой ключ (скопируй полностью):

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDeNZOqa4ERwscSVcHYSt+B/qZJiuIvmCXncrfOKbC20iDTe+EQql0cScTxLi8VJO5W2XEXetVVkkVwPsudl9F+Bt37RcthUPzxqPSuPK53G6LgiEmmR8z+egg5+/vaFD2GSoZrMsIpBecGlfIh9putJQnfPpzRE/srq1PzwlcXZ0DgbGG+yF2D/JN0Md9JFcGLa9p7mSQ0gEbXbZdLqRha893nsu/LbAfz8eIWO/IeRDIDkbviWTP+6h9SnvX96tOOAH0UZ3XT2btbIXj9AKYAbrKoIGbDTqygQ2h9eYNMdAAKAcu/nBXqmSH2iyxczX4Y0Ve1Hb5J3PXG/JfTRJwbCRJJfm40yUT5/JmUfGyNSPN/7A2xvVSwBNeeLn4rlpKuZN86S5FIjTUvvtosRugtNiIRgxcERY11xvYF8uZugOxhxt/m3pZEtNo6Vvtaxej3KG89iHDVXu5cPOokjr85U2or+VwFU0an6yr3QkzD2xQ9ehlKmDLaO0ULoPlRl18VKubvYpFgduxdyDP25FblJasY5GTfo5rtX6YxYi+XSE1I5DYPe7JpuX5XnOurEw5VhGDK1dljC5uz2WPd4+njwAofxKFJUPi/YAmZ1ZqV5VpuG2ynosPXvVcuCDedvGc9fb789kepJrySqT0EqBtm3hCEWa2JgoAGzCA4u4Mssw== vastai-key
```

**Важно:** Вставляй ВСЮ строку целиком, без переносов!

---

## После добавления:

1. **Перезагрузи инстанс** (Stop → Start)
2. Подожди 1-2 минуты
3. Попробуй подключиться:
   ```powershell
   ssh -p 23396 root@151.237.25.234 -L 8000:localhost:8000
   ```

---

## Или просто используй Jupyter! ⭐⭐⭐

**Не нужен SSH ключ вообще!**

1. На Vast.ai нажми **"Open"**
2. В Jupyter: **New → Terminal**
3. Готово!

Сервер уже запущен, можешь сразу работать!
