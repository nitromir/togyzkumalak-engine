import requests
import os
import time
from datetime import datetime

import json

# --- НАСТРОЙКИ ПО УМОЛЧАНИЮ ---
DEFAULT_CONFIG = {
    "remote_url": "http://localhost",
    "ports": [8000, 8080],
    "interval": 30,
    "enabled": True
}
CONFIG_FILE = "sync_config.json"
LOCAL_DIR_AZ = r"C:\Users\Admin\Documents\Toguzkumalak\gym-togyzkumalak-master\togyzkumalak-engine\models\alphazero"
LOCAL_DIR_PROBS = r"C:\Users\Admin\Documents\Toguzkumalak\gym-togyzkumalak-master\togyzkumalak-engine\models\probs"
LOG_FILE = "sync_log.txt"

# ... (предыдущие функции) ...

def sync_type(api_base, model_type):
    """Модель: alphazero или probs"""
    log_message(f"🔍 Проверка обновлений {model_type}...")
    endpoint = f"/api/training/{model_type}/checkpoints"
    local_dir = LOCAL_DIR_AZ if model_type == "alphazero" else LOCAL_DIR_PROBS
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    try:
        r = requests.get(f"{api_base}{endpoint}", timeout=5)
        if r.status_code != 200: return
        checkpoints = r.json().get('checkpoints', [])
        
        for cp in checkpoints:
            name = cp.get('name') or cp.get('filename')
            remote_mod = cp.get('modified') or cp.get('timestamp')
            local_path = os.path.join(local_dir, name)
            
            # Эндпоинт для скачивания отличается для PROBS
            download_url = f"{api_base}/api/training/{model_type}/checkpoints/{name}/download"
            
            should_download = False
            if not os.path.exists(local_path):
                should_download = True
            elif remote_mod:
                try:
                    remote_ts = datetime.fromisoformat(remote_mod.replace('Z', '+00:00')).timestamp()
                    local_ts = os.path.getmtime(local_path)
                    if remote_ts > local_ts + 10:
                        should_download = True
                except: pass
            
            if should_download:
                log_message(f"🆕 [Remote] Загрузка {model_type}: {name}...")
                with requests.get(download_url, stream=True, timeout=300) as dr:
                    dr.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in dr.iter_content(chunk_size=1024*1024):
                            if chunk: f.write(chunk)
                log_message(f"✅ Готово: {name}")
    except Exception as e:
        log_message(f"⚠️ Ошибка синхронизации {model_type}: {e}")

while True:
    config = load_config()
    if not config.get("enabled", True):
        time.sleep(10); continue

    remote_base_url = config.get("remote_url", "http://localhost")
    api_ports = config.get("ports", [8000, 8080])
    
    for port in api_ports:
        api_base = f"{remote_base_url}:{port}"
        try:
            # Проверяем связь
            r = requests.get(f"{api_base}/api/health", timeout=2)
            if r.status_code == 200:
                sync_type(api_base, "alphazero")
                sync_type(api_base, "probs")
                
                with open("sync_status.json", "w") as f:
                    json.dump({"last_sync": datetime.now().isoformat(), "server": api_base, "status": "active"}, f)
                break
        except: continue
    
    time.sleep(config.get("interval", 30))
