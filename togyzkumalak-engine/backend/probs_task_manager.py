"""
PROBS Task Manager for Togyzkumalak.
Manages PROBS training sessions, checkpoints, and model loading.
"""

import os
import sys
import json
import time
import threading
import datetime
import subprocess
from typing import Dict, Optional, List
from dataclasses import dataclass

probs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
    "../probs-main/python_impl_generic"))
if not os.path.exists(probs_path):
    probs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
        "../../probs-main/python_impl_generic"))

if probs_path not in sys.path:
    sys.path.insert(0, probs_path)

@dataclass
class PROBSTrainingConfig:
    """
    Конфигурация PROBS тренировки.
    Значения по умолчанию выбраны согласно оригинальному PROBS репозиторию.
    
    Для Тогызкумалака (менее сложная игра чем шахматы 6x6):
    - v_train_episodes: 500-1000 (оригинал для шахмат: 6000)
    - q_train_episodes: 250-500 (оригинал для шахмат: 3000)
    - num_q_s_a_calls: 20-50 (глубина beam search)
    """
    n_high_level_iterations: int = 100          # Было 10, нужно больше итераций для сходимости
    v_train_episodes: int = 500                  # Было 5! Критически мало для обучения
    q_train_episodes: int = 250                  # Было 5! Критически мало
    mem_max_episodes: int = 10000                # Было 500, увеличиваем буфер experience replay
    train_batch_size: int = 64                   # Было 32
    num_q_s_a_calls: int = 30                    # Было 10, глубина beam search
    max_depth: int = 50                          # Было 8, максимальная глубина дерева
    device: str = "cpu"
    use_boost: bool = False
    initial_checkpoint: Optional[str] = None
    dataset_drop_ratio: float = 0.5              # Новый параметр: предотвращение переобучения
    alphazero_move_num_sampling_moves: int = 5   # Новый: сколько первых ходов сэмплируем
    update_threshold: float = 0.50                # Минимальный win rate для принятия новой модели (как в AlphaZero)

class PROBSTaskManager:
    def __init__(self):
        self.tasks = {}
        self.current_task = None
        self.training_thread = None
        self.stop_requested = False
        self.engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.engine_dir, "models", "probs")
        self.probs_dir = probs_path
        self._loaded_model = None
        self.best_metric = -1.0
        self.best_checkpoint_name = None
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_best_info()

    def _load_best_info(self):
        info_path = os.path.join(self.models_dir, "best_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    data = json.load(f)
                    self.best_metric = data.get("metric", -1.0)
                    self.best_checkpoint_name = data.get("filename")
            except: pass

    def _save_best_info(self, filename, metric):
        self.best_metric = metric
        self.best_checkpoint_name = filename
        info_path = os.path.join(self.models_dir, "best_info.json")
        with open(info_path, "w") as f:
            json.dump({"filename": filename, "metric": metric, "date": datetime.datetime.now().isoformat()}, f)
    
    def get_best_info(self):
        """Get best checkpoint info."""
        if self.best_checkpoint_name and self.best_metric >= 0:
            return {
                "filename": self.best_checkpoint_name,
                "metric": self.best_metric,
                "date": datetime.datetime.now().isoformat()
            }
        return None
    
    def start_training(self, config):
        if self.current_task and self.tasks.get(self.current_task, {}).get("status") == "running":
            raise Exception("Training already running")
        task_id = "probs_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tasks[task_id] = {
            "task_id": task_id, "status": "starting", "config": config,
            "progress": 0, "current_iteration": 0,
            "total_iterations": config.get("n_high_level_iterations", 10),
            "start_time": time.time(), "elapsed_time": 0, "metrics": [], "error": None
        }
        self.current_task = task_id
        self.stop_requested = False
        self.training_thread = threading.Thread(target=self._run_training, args=(task_id, config), daemon=True)
        self.training_thread.start()
        return task_id
    
    def _run_training(self, task_id, config):
        log_path = os.path.join(self.engine_dir, "probs_training.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== PROBS Training Session {task_id} started at {datetime.datetime.now()} ===\n")
            
        def log_print(*args):
            msg = " ".join(map(str, args))
            try: print(f"[PROBS] {msg}")
            except: pass
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}\n")
            except: pass

        try:
            self.tasks[task_id]["status"] = "running"
            
            # Step 0: Initial Checkpoint
            initial_ckpt = config.get("initial_checkpoint")
            if initial_ckpt:
                log_print(f"Loading initial checkpoint: {initial_ckpt}")
                ckpt_p = os.path.join(self.models_dir, "checkpoints", initial_ckpt)
                if os.path.exists(ckpt_p):
                    if self.load_checkpoint(ckpt_p, config.get("device", "cpu")):
                        log_print("Successfully loaded initial weights.")
                    else: log_print("Warning: Failed to load initial checkpoint.")
                else: log_print(f"Warning: Checkpoint not found at {ckpt_p}")

            # Step 1: Boosting
            if config.get("use_boost", False):
                self.tasks[task_id]["status"] = "boosting"
                log_print("Starting Supervised Boosting...")
                boosted_model = self._run_boosting(task_id, config, log_print)
                self._loaded_model = boosted_model
                log_print("Boosting completed!")
            
            # Step 2: RL Process
            self.tasks[task_id]["status"] = "running"
            config_path = os.path.join(self.models_dir, task_id + "_config.yaml")
            self._create_probs_config(config_path, config)
            self._run_probs_process(task_id, config_path, config, log_print)
            
            self.tasks[task_id]["status"] = "stopped" if self.stop_requested else "completed"
            self.tasks[task_id]["progress"] = 100
            log_print(f"Training session {self.tasks[task_id]['status']}.")
        except Exception as e:
            self.tasks[task_id]["status"] = "error"
            self.tasks[task_id]["error"] = str(e)
            log_print(f"ERROR: {e}")
            import traceback
            with open(log_path, "a", encoding="utf-8") as f: traceback.print_exc(file=f)
            traceback.print_exc()

    def _run_boosting(self, task_id, config, log_print):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        import numpy as np
        
        data_file = os.path.join(self.engine_dir, "training_data", "transitions_compact.jsonl")
        if not os.path.exists(data_file):
            log_print(f"Boosting data not found: {data_file}")
            return None
            
        log_print("Loading transitions...")
        states, actions, rewards = [], [], []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    s = list(d["s"])
                    if len(s) < 128: s = s + [0.0] * (128 - len(s))
                    states.append(s[:128])
                    actions.append(d["a"])
                    rewards.append(d["r"])
                except: continue
        
        if len(states) < 100: return None
        
        device = "cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
        X = torch.FloatTensor(states).to(device)
        y_a = torch.LongTensor(actions).to(device)
        y_v = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        
        dataset = TensorDataset(X, y_a, y_v)
        loader = DataLoader(dataset, batch_size=config.get("train_batch_size", 32), shuffle=True)
        
        sys.path.insert(0, self.probs_dir)
        from probs_impl import probs_impl_common
        cfg_model = {"value": {"class": "ValueModelTK_v1", "learning_rate": 0.0005, "weight_decay": 0.0001}, "self_learner": {"class": "SelfLearningModelTK_v1", "learning_rate": 0.0005, "weight_decay": 0.0001}}
        mk = probs_impl_common.create_model_keeper(cfg_model, "togyzkumalak")
        
        # If we loaded initial checkpoint, use its weights
        if self._loaded_model:
            mk.models["value"].load_state_dict(self._loaded_model.models["value"].state_dict())
            mk.models["self_learner"].load_state_dict(self._loaded_model.models["self_learner"].state_dict())

        mk.to(device)
        v_model, q_model = mk.models["value"], mk.models["self_learner"]
        v_opt, q_opt = mk.optimizers["value"], mk.optimizers["self_learner"]
        criterion_q, criterion_v = nn.CrossEntropyLoss(), nn.MSELoss()
        
        log_print(f"Boosting on {device}...")
        for epoch in range(2):
            if self.stop_requested: break
            t_loss = 0
            for b_x, b_ya, b_yv in loader:
                if self.stop_requested: break
                q_opt.zero_grad(); out_q = q_model(b_x); loss_q = criterion_q(out_q, b_ya); loss_q.backward(); q_opt.step()
                v_opt.zero_grad(); out_v = v_model(b_x); loss_v = criterion_v(out_v, b_yv); loss_v.backward(); v_opt.step()
                t_loss += loss_q.item() + loss_v.item()
            log_print(f"Epoch {epoch+1}/2. Avg Loss: {t_loss/len(loader):.4f}")
        
        mk.save_checkpoint(os.path.join(self.models_dir, "checkpoints"), "boosted_start")
        return mk

    def _create_probs_config(self, config_path, config):
        """
        Создаёт YAML конфиг для PROBS тренировки.
        Параметры выровнены с оригинальным PROBS репозиторием.
        """
        ckpt = os.path.join(self.models_dir, "checkpoints").replace(os.sep, "/")
        
        # Получаем параметры с разумными значениями по умолчанию (как в оригинале)
        device = config.get("device", "cpu")
        sub_processes_cnt = config.get("sub_processes_cnt", 0)
        self_play_threads = config.get("self_play_threads", 1)  # Поддерживает мульти-GPU через кастомный Pool
        mem_max_episodes = config.get("mem_max_episodes", 10000)
        
        n_high_level_iterations = config.get("n_high_level_iterations", 100)
        v_train_episodes = config.get("v_train_episodes", 500)
        q_train_episodes = config.get("q_train_episodes", 250)
        train_batch_size = config.get("train_batch_size", 64)
        num_q_s_a_calls = config.get("num_q_s_a_calls", 30)
        max_depth = config.get("max_depth", 50)
        
        # Новые параметры из оригинального PROBS
        dataset_drop_ratio = config.get("dataset_drop_ratio", 0.5)
        alphazero_move_num_sampling_moves = config.get("alphazero_move_num_sampling_moves", 5)
        q_add_hardest_nodes_per_step = config.get("q_add_hardest_nodes_per_step", 5)
        
        evaluate_n_games = config.get("evaluate_n_games", 20)
        # ВАЖНО: Используем one_step_lookahead вместо random для более адекватной оценки
        evaluate_enemy = config.get("evaluate_enemy", "one_step_lookahead")
        # Порог для принятия новой модели (механизм отката)
        update_threshold = config.get("update_threshold", 0.50)
        
        yaml_str = f"""name: probs_togyzkumalak
env:
  name: togyzkumalak
  n_max_episode_steps: 200
cmd: train
infra:
  log: mem
  device: {device}
  sub_processes_cnt: {sub_processes_cnt}
  self_play_threads: {self_play_threads}
  mem_max_episodes: {mem_max_episodes}
train:
  n_high_level_iterations: {n_high_level_iterations}
  v_train_episodes: {v_train_episodes}
  q_train_episodes: {q_train_episodes}
  q_dataset_episodes_sub_iter: 1
  dataset_drop_ratio: {dataset_drop_ratio}
  checkpoints_dir: {ckpt}
  train_batch_size: {train_batch_size}
  self_learning_batch_size: {train_batch_size}
  get_q_dataset_batch_size: {train_batch_size}
  num_q_s_a_calls: {num_q_s_a_calls}
  max_depth: {max_depth}
  alphazero_move_num_sampling_moves: {alphazero_move_num_sampling_moves}
  q_add_hardest_nodes_per_step: {q_add_hardest_nodes_per_step}
  update_threshold: {update_threshold}
evaluate:
  evaluate_n_games: {evaluate_n_games}
  randomize_n_turns: 2
  enemy:
    kind: {evaluate_enemy}
model:
  value:
    class: ValueModelTK_v1
    learning_rate: 0.0003
    weight_decay: 0.0001
  self_learner:
    class: SelfLearningModelTK_v1
    learning_rate: 0.0003
    weight_decay: 0.0001
"""
        with open(config_path, "w", encoding="utf-8") as f: 
            f.write(yaml_str)
    
    def _run_probs_process(self, task_id, config_path, config, log_print):
        """
        Запускает PROBS тренировку согласно оригинальному пайплайну.
        
        КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Модель обучается НЕПРЕРЫВНО, без сброса весов.
        Логика "carousel" убрана - она мешала накоплению знаний.
        """
        import torch, yaml, multiprocessing, pickle
        original_cwd = os.getcwd()
        os.chdir(self.probs_dir)
        tasks_queues, results_queue, sub_processes = None, None, []

        try:
            if self.probs_dir not in sys.path: sys.path.insert(0, self.probs_dir)
            import environments, helpers
            from probs_impl import probs_impl_common, probs_impl_main
            
            with open(config_path, "r", encoding="utf-8") as f: probs_config = yaml.safe_load(f)
            dv = config.get("device", "cpu")
            device = "cuda" if dv == "auto" and torch.cuda.is_available() else ("cpu" if dv == "cuda" and not torch.cuda.is_available() else dv)
            log_print(f"Starting PROBS RL training on {device}...")
            log_print(f"Config: v_episodes={probs_config['train']['v_train_episodes']}, q_episodes={probs_config['train']['q_train_episodes']}")
            
            # Инициализация подпроцессов для параллельного сбора данных
            sub_processes_cnt = probs_config['infra']['sub_processes_cnt']
            if sub_processes_cnt > 0:
                log_print(f"Initializing {sub_processes_cnt} worker processes...")
                try: multiprocessing.set_start_method("spawn", force=True)
                except: pass
                
                # РАСПРЕДЕЛЕНИЕ ПО GPU ДЛЯ МАКСИМАЛЬНОЙ ЖАРЫ
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if gpu_count > 0:
                    log_print(f"Detected {gpu_count} GPUs. Distributing workers...")

                tasks_queues = [multiprocessing.Queue() for _ in range(sub_processes_cnt)]
                results_queue = multiprocessing.Queue()
                v_class = probs_config['model']['value']['class']
                q_class = probs_config['model']['self_learner']['class']
                for pi in range(sub_processes_cnt):
                    # Каждому воркеру своя GPU (по кругу)
                    worker_device = f"cuda:{pi % gpu_count}" if gpu_count > 0 else device
                    p = multiprocessing.Process(
                        target=probs_impl_main.worker, 
                        args=(tasks_queues[pi], results_queue, probs_config, worker_device, v_class, q_class)
                    )
                    p.daemon = True
                    p.start()
                    sub_processes.append(p)

            # --- ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ---
            # Если загружен чекпоинт - используем его, иначе создаём новую модель
            if self._loaded_model:
                mk = self._loaded_model
                log_print("Continuing training from loaded checkpoint...")
            else:
                mk = probs_impl_common.create_model_keeper(probs_config["model"], "togyzkumalak")
                log_print("Starting training from scratch...")
            
            mk.to(device)
            helpers.TENSORBOARD = helpers.MemorySummaryWriter()
            
            # Создаём LR schedulers для стабильного обучения
            total_iterations = probs_config["train"]["n_high_level_iterations"]
            for model_key in ['value', 'self_learner']:
                if model_key in mk.optimizers:
                    optimizer = mk.optimizers[model_key]
                    # CosineAnnealingLR: плавно снижает LR от начального до минимального
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, 
                        T_max=total_iterations,  # Период косинуса = количество итераций
                        eta_min=1e-5  # Минимальный LR (0.00001)
                    )
                    mk.schedulers[model_key] = scheduler
                    log_print(f"Created LR scheduler for {model_key}: start_lr={optimizer.param_groups[0]['lr']:.6f}, min_lr=1e-5")
            
            # Противник для оценки (теперь one_step_lookahead по умолчанию)
            metrics_enemy = probs_impl_common.create_agent(
                probs_config["evaluate"]["enemy"], "togyzkumalak", device
            )
            log_print(f"Evaluation enemy: {probs_config['evaluate']['enemy']['kind']}")
            
            # Experience replay буфер
            experience_replay = helpers.ExperienceReplay(
                max_episodes=probs_config['infra']['mem_max_episodes'], 
                create_env_func=environments.get_create_env_func("togyzkumalak")
            )

            total = probs_config["train"]["n_high_level_iterations"]
            
            # Загружаем информацию о лучшем чекпойнте ПЕРЕД началом обучения
            self._load_best_info()
            
            # Инициализируем best_win_rate для ТЕКУЩЕЙ сессии обучения
            # Используем лучший результат из предыдущих сессий как baseline,
            # но начинаем отслеживать лучший результат текущей сессии отдельно
            session_best_win_rate = -1.0  # Лучший результат текущей сессии
            previous_win_rate = -1.0      # Win rate предыдущей итерации (для отката)
            checkpoints_dir = os.path.join(self.models_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            # Порог для принятия новой модели (из конфига или по умолчанию)
            update_threshold = probs_config.get('train', {}).get('update_threshold', 0.50)
            
            for i in range(total):
                if self.stop_requested: 
                    log_print("Stop requested, finishing...")
                    break
                    
                # Обновляем статус
                self.tasks[task_id]["current_iteration"] = i + 1
                self.tasks[task_id]["progress"] = ((i + 1) / total) * 100
                self.tasks[task_id]["elapsed_time"] = time.time() - self.tasks[task_id]["start_time"]
                
                log_print(f"=== Iteration {i+1}/{total} ===")
                
                # МЕХАНИЗМ ОТКАТА: Сохраняем текущую модель перед обучением
                # (как в AlphaZero - сохраняем в temp.pth.tar перед каждой итерацией)
                temp_checkpoint_path = os.path.join(checkpoints_dir, f"temp_iter_{i+1}.ckpt")
                if i > 0 or previous_win_rate >= 0:  # Сохраняем temp только если не первая итерация
                    mk.save_checkpoint(checkpoints_dir, f"temp_iter_{i+1}")
                    log_print(f"[ROLLBACK] Saved backup checkpoint: temp_iter_{i+1}.ckpt")
                
                # ГЛАВНОЕ: Модель обучается непрерывно, БЕЗ сброса весов!
                # Это ключевое отличие от предыдущей "carousel" логики
                probs_impl_main.go_train_iteration(
                    probs_config, device, mk, metrics_enemy, i, 
                    tasks_queues=tasks_queues, 
                    results_queue=results_queue, 
                    experience_replay=experience_replay
                )
                
                # Обновляем LR schedulers после каждой итерации
                for model_key in ['value', 'self_learner']:
                    if model_key in mk.schedulers:
                        mk.schedulers[model_key].step()
                        current_lr = mk.optimizers[model_key].param_groups[0]['lr']
                        if i % 10 == 0:  # Логируем каждые 10 итераций
                            log_print(f"LR for {model_key}: {current_lr:.6f}")

                # Логируем и сохраняем метрики
                current_win_rate = None
                model_accepted = True  # Флаг принятия модели (для отката)
                
                if isinstance(helpers.TENSORBOARD, helpers.MemorySummaryWriter):
                    for key, vals in helpers.TENSORBOARD.points.items():
                        if key == 'wins' and len(vals) > 0:
                            current_win_rate = vals[-1]
                            log_print(f"Win rate vs {probs_config['evaluate']['enemy']['kind']}: {current_win_rate:.2%}")
                            
                            # МЕХАНИЗМ ОТКАТА: Проверяем, нужно ли откатываться к предыдущей модели
                            if previous_win_rate >= 0:
                                # Сравниваем с предыдущей итерацией
                                if current_win_rate < update_threshold:
                                    # Win rate ниже порога - откатываемся
                                    log_print(f"⚠️  REJECTING new model: win rate {current_win_rate:.2%} < threshold {update_threshold:.2%}")
                                    model_accepted = False
                                elif current_win_rate < (previous_win_rate - 0.02):
                                    # Win rate упал более чем на 2% - откатываемся
                                    log_print(f"⚠️  REJECTING new model: win rate dropped from {previous_win_rate:.2%} to {current_win_rate:.2%}")
                                    model_accepted = False
                                else:
                                    log_print(f"✅ ACCEPTING new model: win rate {current_win_rate:.2%} >= {previous_win_rate:.2%}")
                                    model_accepted = True
                            
                            # Если модель отклонена - откатываемся к предыдущей
                            if not model_accepted and i > 0:
                                temp_path = os.path.join(checkpoints_dir, f"temp_iter_{i+1}.ckpt")
                                if os.path.exists(temp_path):
                                    log_print(f"🔄 ROLLBACK: Loading previous model from {temp_path}")
                                    mk.load_from_checkpoint(temp_path, device)
                                    mk.to(device)
                                    mk.eval()
                                    # Используем предыдущий win rate
                                    current_win_rate = previous_win_rate
                                    log_print(f"✅ Rolled back to previous model (win rate: {previous_win_rate:.2%})")
                                else:
                                    log_print(f"⚠️  Rollback checkpoint not found: {temp_path}. Continuing with current model.")
                                    model_accepted = True  # Продолжаем, если нет backup
                            
                            # Сохраняем как лучшую если модель принята И:
                            # 1. Это первый результат в сессии (session_best_win_rate == -1.0), ИЛИ
                            # 2. Есть улучшение более чем на 1% относительно лучшего в текущей сессии, ИЛИ
                            # 3. Есть улучшение более чем на 1% относительно глобального лучшего
                            should_save = False
                            if model_accepted:
                                if session_best_win_rate < 0:
                                    # Первый результат - всегда сохраняем
                                    should_save = True
                                    log_print(f"[FIRST] Saving first checkpoint with win rate {current_win_rate:.2%}")
                                elif current_win_rate > (session_best_win_rate + 0.01):
                                    # Улучшение относительно текущей сессии
                                    should_save = True
                                    log_print(f"[SESSION BEST] Win rate improved from {session_best_win_rate:.2%} to {current_win_rate:.2%}")
                                elif self.best_metric >= 0 and current_win_rate > (self.best_metric + 0.01):
                                    # Улучшение относительно глобального лучшего
                                    should_save = True
                                    log_print(f"[GLOBAL BEST] Win rate {current_win_rate:.2%} beats global best {self.best_metric:.2%}")
                                
                                if should_save:
                                    session_best_win_rate = current_win_rate
                                    ckpt_name = f"best_iter_{i+1}.ckpt"
                                    mk.save_checkpoint(checkpoints_dir, f"best_iter_{i+1}")
                                    self._save_best_info(ckpt_name, current_win_rate)
                                    log_print(f"[NEW BEST] Win rate {current_win_rate:.2%} - saved as {ckpt_name}")
                            
                            # Обновляем previous_win_rate для следующей итерации
                            if model_accepted:
                                previous_win_rate = current_win_rate
                                # Удаляем временный чекпойнт после успешной итерации
                                temp_path = os.path.join(checkpoints_dir, f"temp_iter_{i+1}.ckpt")
                                if os.path.exists(temp_path):
                                    try:
                                        os.remove(temp_path)
                                        log_print(f"🧹 Cleaned up temp checkpoint: temp_iter_{i+1}.ckpt")
                                    except:
                                        pass
                            # Если откатились, previous_win_rate остается прежним, temp чекпойнт тоже остается
                
                # Периодическое сохранение чекпоинтов - реже (раз в 20 итераций)
                save_interval = max(5, total // 5) 
                if (i + 1) % save_interval == 0:
                    mk.save_checkpoint(checkpoints_dir, f"iter_{i+1}")
                    log_print(f"Checkpoint saved: iter_{i+1}.ckpt")
            
            # Финальное сохранение - ТОЛЬКО если финальная модель не хуже лучшей
            # Это предотвращает сохранение деградировавших моделей
            should_save_final = True
            if session_best_win_rate >= 0:
                # Проверяем финальный win rate перед сохранением
                final_win_rate = None
                if isinstance(helpers.TENSORBOARD, helpers.MemorySummaryWriter):
                    for key, vals in helpers.TENSORBOARD.points.items():
                        if key == 'wins' and len(vals) > 0:
                            final_win_rate = vals[-1]
                            break
                
                if final_win_rate is not None:
                    # Сохраняем финальную модель только если она не хуже лучшей более чем на 2%
                    if final_win_rate < (session_best_win_rate - 0.02):
                        should_save_final = False
                        log_print(f"⚠️ Final model win rate {final_win_rate:.2%} is worse than best {session_best_win_rate:.2%}. Skipping final checkpoint.")
                        log_print(f"💡 Use best_iter_* checkpoint instead: {self.best_checkpoint_name}")
            
            if should_save_final:
                mk.save_checkpoint(checkpoints_dir, "final")
                log_print(f"Training completed. Final checkpoint saved.")
            else:
                log_print(f"Training completed. Final checkpoint NOT saved (model degraded).")
            
            if session_best_win_rate >= 0:
                log_print(f"Best win rate in this session: {session_best_win_rate:.2%}")
            if self.best_metric >= 0:
                log_print(f"Global best win rate: {self.best_metric:.2%}")
                log_print(f"🏆 Best checkpoint: {self.best_checkpoint_name}")
            self._loaded_model = mk
            
        except Exception as e:
            raise e
        finally:
            # Корректное завершение подпроцессов
            if tasks_queues:
                for pi in range(len(tasks_queues)):
                    try: tasks_queues[pi].put_nowait((pi, "stop", None))
                    except: pass
            for p in sub_processes:
                p.join(timeout=2)
                if p.is_alive(): 
                    p.terminate()
            os.chdir(original_cwd)
    
    def stop_task(self, task_id):
        print(f"[PROBS] Stop requested for task {task_id}")
        if task_id not in self.tasks: return False
        if self.tasks[task_id]["status"] in ["running", "boosting"]:
            self.stop_requested = True
            self.tasks[task_id]["status"] = "stopping"
            return True
        return False
    
    def get_status(self, task_id): return self.tasks.get(task_id)
    def list_tasks(self): return {k: {"task_id": v["task_id"], "status": v["status"], "progress": v["progress"], "current_iteration": v.get("current_iteration", 0), "total_iterations": v.get("total_iterations", 0)} for k, v in self.tasks.items()}
    
    def get_checkpoints(self):
        cks, d = [], os.path.join(self.models_dir, "checkpoints")
        if not os.path.exists(d): return cks
        
        # Загружаем метрики из best_info.json для отображения
        best_metric_value = self.best_metric if self.best_metric > 0 else None
        
        for f in os.listdir(d):
            if f.endswith(".ckpt"):
                p = os.path.join(d, f); s = os.stat(p)
                is_best = f == self.best_checkpoint_name
                # Добавляем метрику только для лучшего чекпоинта
                metric = best_metric_value if is_best else None
                cks.append({
                    "filename": f, 
                    "path": p, 
                    "size_mb": round(s.st_size / 1048576, 2), 
                    "timestamp": datetime.datetime.fromtimestamp(s.st_mtime).isoformat(), 
                    "is_best": is_best,
                    "metric": metric  # Win rate для лучшего чекпоинта
                })
        cks.sort(key=lambda x: x["timestamp"], reverse=True)
        return cks
    
    def load_checkpoint(self, path, device="cpu"):
        try:
            import torch, yaml
            sys.path.insert(0, self.probs_dir)
            import helpers
            from probs_impl import probs_impl_common
            cfg = {"value": {"class": "ValueModelTK_v1", "learning_rate": 0.0005, "weight_decay": 0.0001}, "self_learner": {"class": "SelfLearningModelTK_v1", "learning_rate": 0.0005, "weight_decay": 0.0001}}
            orig = os.getcwd(); os.chdir(self.probs_dir)
            try:
                mk = probs_impl_common.create_model_keeper(cfg, "togyzkumalak")
                mk.load_from_checkpoint(path, device); mk.to(device); mk.eval()
                self._loaded_model = mk; print("[PROBS] Loaded:", path); return True
            finally: os.chdir(orig)
        except Exception as e: print("[PROBS] Error:", e); return False
    
    def get_loaded_model(self): return self._loaded_model
    def is_model_loaded(self): return self._loaded_model is not None
    
    def start_tournament(self, num_games=20):
        """Start PROBS tournament in background subprocess."""
        import subprocess
        import threading
        
        tournament_id = f"probs_tournament_{int(time.time())}"
        checkpoints_dir = os.path.join(self.models_dir, "checkpoints")
        probs_dir = self.probs_dir
        config_path = os.path.join(probs_dir, "configs", "train_togyzkumalak.yaml")
        
        # Запускаем турнир в отдельном процессе
        def run_tournament():
            try:
                log_path = os.path.join(self.models_dir, f"{tournament_id}.log")
                script_path = os.path.join(probs_dir, "probs_tournament.py")
                
                with open(log_path, "w") as log_file:
                    process = subprocess.Popen(
                        [sys.executable, script_path,
                         "--checkpoints-dir", checkpoints_dir,
                         "--config", config_path,
                         "--games", str(num_games),
                         "--device", "cuda"],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        cwd=probs_dir
                    )
                    process.wait()
                    
                    # Читаем результаты
                    results_path = os.path.join(checkpoints_dir, "tournament_results.json")
                    if os.path.exists(results_path):
                        with open(results_path, "r") as f:
                            results = json.load(f)
                            print(f"[PROBS Tournament] Completed: {results.get('leaderboard', [{}])[0]}")
            except Exception as e:
                print(f"[PROBS Tournament] Error: {e}")
        
        thread = threading.Thread(target=run_tournament, daemon=True)
        thread.start()
        
        return tournament_id
    
    def get_tournament_results(self):
        """Get latest tournament results."""
        results_path = os.path.join(self.models_dir, "checkpoints", "tournament_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    return json.load(f)
            except:
                return None
        return None

probs_task_manager = PROBSTaskManager()