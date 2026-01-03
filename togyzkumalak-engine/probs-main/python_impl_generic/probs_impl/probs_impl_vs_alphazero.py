"""
PROBS vs AlphaZero data collection for Ultra training.
Собирает данные игр PROBS против AlphaZero для смешанного обучения.
"""
import numpy as np
import torch
import time
from collections import Counter

import environments
import helpers
from probs_impl import probs_impl_common


VALUE_MODEL: helpers.BaseValueModel = None
SELF_LEARNING_MODEL: helpers.BaseSelfLearningModel = None
CONFIG: dict = None
ALPHAZERO_AGENT: helpers.BaseAgent = None
GET_DATASET_DEVICE: str = None


def init_worker_vs_az(value_model, self_learning_model, config, alphazero_agent, device, worker_id=0):
    """Инициализация воркера для игр против AlphaZero."""
    global VALUE_MODEL
    global SELF_LEARNING_MODEL
    global CONFIG
    global ALPHAZERO_AGENT
    global GET_DATASET_DEVICE
    
    # Распределяем по GPU
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count > 0:
        actual_device = f"cuda:{worker_id % gpu_count}"
    else:
        actual_device = device
        
    VALUE_MODEL = value_model.to(actual_device)
    SELF_LEARNING_MODEL = self_learning_model.to(actual_device)
    CONFIG = config
    ALPHAZERO_AGENT = alphazero_agent
    GET_DATASET_DEVICE = actual_device


def vs_alphazero_worker_task(q, gids, idx, value_model, self_learning_model, config, alphazero_agent, device):
    """Задача воркера для игр против AlphaZero."""
    try:
        init_worker_vs_az(value_model, self_learning_model, config, alphazero_agent, device, idx)
        result = multiprocessing_entry_vs_alphazero(gids)
        q.put(result)
    except Exception as e:
        import traceback
        q.put(("error", str(e), traceback.format_exc()))


def multiprocessing_entry_vs_alphazero(game_ids):
    """Точка входа для multiprocessing игр против AlphaZero."""
    torch.set_num_threads(1)  # Важно для multiprocessing

    if VALUE_MODEL is None:
        raise RuntimeError("VALUE_MODEL is None in worker process. Initialization failed.")

    VALUE_MODEL.eval()
    SELF_LEARNING_MODEL.eval()

    seed = int(time.time()) + game_ids[0]
    np.random.seed(seed)
    torch.manual_seed(seed + 5)

    stats = Counter()
    replay_episodes = []

    replay_episodes, stats = play_vs_alphazero(game_ids)

    return replay_episodes, stats


def play_vs_alphazero_iterative(env: helpers.BaseEnv, replay_episode: helpers.ExperienceReplayEpisode, 
                                 episode_stats: Counter, alphazero_agent: helpers.BaseAgent, probs_is_white: bool):
    """
    Итеративная игра PROBS против AlphaZero.
    probs_is_white: True если PROBS играет белыми, False если черными.
    """
    for stepi in range(CONFIG['env']['n_max_episode_steps']):
        current_player = env.player
        is_probs_turn = (current_player == 1 and probs_is_white) or (current_player == -1 and not probs_is_white)
        
        if is_probs_turn:
            # Ход PROBS - собираем данные для обучения
            action_mask = env.get_valid_actions_mask()
            to_eval_env = [env.get_rotated_encoded_state()]
            yield to_eval_env
            action_values = to_eval_env[1]

            action, greedy_action = probs_impl_common.sample_action(
                env, CONFIG, action_values, action_mask, is_v_not_q=True
            )

            if action == greedy_action:
                episode_stats['greedy_action_sum'] += 1
            episode_stats['greedy_action_cnt'] += 1

            reward_mul = 1 if env.is_white_to_move() else -1
            reward, done = env.step(action)
            replay_episode.on_action(action, reward * reward_mul, done)
        else:
            # Ход AlphaZero - просто делаем ход, не собираем данные
            if hasattr(alphazero_agent, 'reset_mcts'):
                if stepi == 0:  # Сбрасываем MCTS только в начале игры
                    alphazero_agent.reset_mcts()
            action = alphazero_agent.get_action(env, temp=0)
            reward, done = env.step(action)
            # Не добавляем в replay_episode, так как это ход противника
        
        if done:
            break

    yield None


@torch.no_grad()
def play_vs_alphazero(game_ids):
    """
    Играет PROBS против AlphaZero и собирает данные для обучения PROBS.
    Чередует цвета: PROBS играет белыми в четных играх, черными в нечетных.
    """
    create_env_func = environments.get_create_env_func(CONFIG['env']['name'])

    tasks_list = []  # [(iterator, to_eval_env, replay_episode, episode_stats)]
    replay_episodes = []
    stats = Counter()

    next_game_i = 0
    while next_game_i < len(game_ids) or len(tasks_list) > 0:
        # Инициализируем новые игры
        if next_game_i < len(game_ids) and len(tasks_list) < CONFIG['train']['self_learning_batch_size']:
            game_id = game_ids[next_game_i]
            env = create_env_func()
            env.reset()
            
            # Чередуем цвета: четные игры - PROBS белыми, нечетные - черными
            probs_is_white = (game_id % 2 == 0)
            
            replay_episode = helpers.ExperienceReplayEpisode()
            episode_stats = Counter()
            
            it = iter(play_vs_alphazero_iterative(env, replay_episode, episode_stats, ALPHAZERO_AGENT, probs_is_white))
            to_eval_q_a = next(it)
            
            if to_eval_q_a is not None:
                tasks_list.append((it, to_eval_q_a, replay_episode, episode_stats))
            
            next_game_i += 1
        else:
            # Батч-инференс для PROBS
            global SELF_LEARNING_MODEL
            # Убеждаемся, что модель на правильном устройстве
            if GET_DATASET_DEVICE and SELF_LEARNING_MODEL is not None:
                SELF_LEARNING_MODEL = SELF_LEARNING_MODEL.to(GET_DATASET_DEVICE)
            
            inputs_collection = [to_eval_env[0] for it, to_eval_env, replay_episode, episode_stats in tasks_list]
            action_values_batch = probs_impl_common.get_q_a_multi_inputs(SELF_LEARNING_MODEL, inputs_collection, GET_DATASET_DEVICE)

            new_tasks_list = []
            for (it, to_eval_env, replay_episode, episode_stats), action_values in zip(tasks_list, action_values_batch):
                to_eval_env.append(action_values)
                
                try:
                    to_eval_q_a = next(it)
                    if to_eval_q_a is not None:
                        new_tasks_list.append((it, to_eval_q_a, replay_episode, episode_stats))
                    else:
                        # Игра завершена
                        replay_episodes.append(replay_episode)
                        stats += episode_stats
                except StopIteration:
                    # Игра завершена
                    replay_episodes.append(replay_episode)
                    stats += episode_stats

            tasks_list = new_tasks_list

    return replay_episodes, stats


def go_vs_alphazero(value_model: helpers.BaseValueModel, self_learning_model: helpers.BaseSelfLearningModel, 
                    config: dict, experience_replay: helpers.ExperienceReplay, alphazero_agent: helpers.BaseAgent,
                    get_dataset_device: str, num_games: int):
    """
    Основная функция для сбора данных игр PROBS против AlphaZero.
    
    Args:
        value_model: V-модель PROBS
        self_learning_model: Q-модель PROBS
        config: Конфигурация обучения
        experience_replay: Буфер для сохранения данных
        alphazero_agent: Агент AlphaZero
        get_dataset_device: Устройство для инференса
        num_games: Количество игр для сбора данных
    """
    global VALUE_MODEL
    global SELF_LEARNING_MODEL
    global CONFIG
    global ALPHAZERO_AGENT
    global GET_DATASET_DEVICE

    GET_DATASET_DEVICE = get_dataset_device
    VALUE_MODEL = value_model
    SELF_LEARNING_MODEL = self_learning_model
    CONFIG = config
    ALPHAZERO_AGENT = alphazero_agent

    value_model.eval()
    self_learning_model.eval()

    stats = Counter()

    # Переводим модели на CPU перед пиклингом для подпроцессов
    value_model_cpu = value_model.cpu()
    self_learning_model_cpu = self_learning_model.cpu()

    with torch.no_grad():
        # ------------ No multiprocessing (для начала)
        if config['infra'].get('self_play_threads', 1) <= 1:
            game_ids = list(range(num_games))
            replay_episodes, episodes_stats = play_vs_alphazero(game_ids)
            for replay_episode in replay_episodes:
                experience_replay.append_replay_episode(replay_episode)
            stats += episodes_stats
        else:
            # ------------ Multiprocessing (если нужно)
            # Пока используем однопоточную версию для стабильности
            game_ids = list(range(num_games))
            replay_episodes, episodes_stats = play_vs_alphazero(game_ids)
            for replay_episode in replay_episodes:
                experience_replay.append_replay_episode(replay_episode)
            stats += episodes_stats

    if stats.get('greedy_action_cnt', 0) > 0:
        helpers.TENSORBOARD.append_scalar('vs_alphazero_greedy_action_freq', 
                                          stats['greedy_action_sum'] / stats['greedy_action_cnt'])

    print(f"Collected {len(replay_episodes)} episodes from PROBS vs AlphaZero games")
    return stats
