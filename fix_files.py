import os

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

write_file('probs-main/python_impl_generic/configs/train_togyzkumalak.yaml', """name: train_togyzkumalak
env:
  name: togyzkumalak
  n_max_episode_steps: 200

cmd: train
infra:
  log: tf
  device: cpu
  sub_processes_cnt: 0
  self_play_threads: 1
  mem_max_episodes: 100

train:
  n_high_level_iterations: 2
  v_train_episodes: 2
  q_train_episodes: 2
  q_dataset_episodes_sub_iter: 1
  dataset_drop_ratio: 0
  checkpoints_dir: checkpoints/togyzkumalak_test
  train_batch_size: 16
  self_learning_batch_size: 16
  get_q_dataset_batch_size: 16
  num_q_s_a_calls: 5
  max_depth: 5
  alphazero_move_num_sampling_moves: 2
  q_add_hardest_nodes_per_step: 0
  
evaluate:
  evaluate_n_games: 1
  randomize_n_turns: 0
  enemy:
    kind: random

model:
  value:
    class: ValueModelTK_v1
    learning_rate: 0.001
    weight_decay: 0.0001
  self_learner:
    class: SelfLearningModelTK_v1
    learning_rate: 0.001
    weight_decay: 0.0001

enemy:
  kind: random
""")

write_file('probs-main/python_impl_generic/environments/togyzkumalak_env.py', """import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.abspath(os.path.join(current_dir, "../../../gym-togyzkumalak-master/togyzkumalak-engine"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from backend.alphazero_trainer import TogyzkumalakGame
import helpers

def create_env_func():
    return TogyzkumalakEnv()

class TogyzkumalakEnv(helpers.BaseEnv):
    def __init__(self):
        self.game = TogyzkumalakGame()
        self.reset()

    def reset(self):
        self.board = self.game.getInitBoard()
        self.player = 1
        self.done = False
        self.move_counter = 0
        return self.board, self.get_valid_actions_mask()

    def copy(self):
        new_env = TogyzkumalakEnv.__new__(TogyzkumalakEnv)
        new_env.game = self.game
        new_env.board = self.board.copy()
        new_env.player = self.player
        new_env.done = self.done
        new_env.move_counter = self.move_counter
        return new_env

    def step(self, action):
        player_before_move = self.player
        self.board, next_player = self.game.getNextState(self.board, self.player, action)
        self.player = next_player
        self.move_counter += 1
        res = self.game.getGameEnded(self.board, self.player)
        if res != 0:
            self.done = True
            reward = -res if abs(res) > 0.5 else 0
        else:
            reward = 0
        return reward, self.done

    def get_valid_actions_mask(self):
        mask = self.game.getValidMoves(self.board, self.player)
        if np.sum(mask) == 0 and not self.done:
            self.done = True
        return mask

    def is_white_to_move(self):
        return self.player == 1

    def get_rotated_encoded_state(self):
        canonical = self.game.getCanonicalForm(self.board, self.player)
        obs = self.game.boardToObservation(canonical)
        return [obs.astype(np.float32)]

    def get_rotated_encoded_states_with_symmetry__value_model(self):
        return [self.get_rotated_encoded_state()]

    def get_rotated_encoded_states_with_symmetry__q_value_model(self, action_values):
        enc = self.get_rotated_encoded_state()[0]
        mask = self.get_valid_actions_mask().astype(np.float32)
        return [[enc, action_values.astype(np.float32), mask]]

    def render_ascii(self):
        player_str = "White (1)" if self.player == 1 else "Black (-1)"
        print("Move #", self.move_counter, ", Player:", player_str, ", Done:", self.done)
        print("White pits [0-8]:", self.board[:9].astype(int))
        print("Black pits [9-17]:", self.board[9:18].astype(int))
        print("Kazans - White:", int(self.board[20]), ", Black:", int(self.board[21]))
        print("Valid actions:", np.where(self.get_valid_actions_mask() == 1)[0])
""")

write_file('probs-main/python_impl_generic/environments/togyzkumalak_nn.py', """import torch
import torch.nn as nn
import helpers

N_ACTIONS = 9
INPUT_SIZE = 128
HIDDEN_SIZE = 256


class ResBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.fc2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = torch.relu(out + residual)
        return out


class ValueModelTK_v1(helpers.BaseValueModel):
    def __init__(self):
        super().__init__()
        self.fc_input = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.bn_input = nn.BatchNorm1d(HIDDEN_SIZE)
        self.res1 = ResBlock(HIDDEN_SIZE)
        self.res2 = ResBlock(HIDDEN_SIZE)
        self.fc_out1 = nn.Linear(HIDDEN_SIZE, 64)
        self.fc_out2 = nn.Linear(64, 1)
        
    def forward(self, obs):
        if isinstance(obs, list): 
            obs = obs[0]
        if isinstance(obs, torch.Tensor):
            if obs.dim() == 1: 
                obs = obs.unsqueeze(0)
        else:
            obs = torch.as_tensor(obs)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
        x = torch.relu(self.bn_input(self.fc_input(obs)))
        x = self.res1(x)
        x = self.res2(x)
        x = torch.relu(self.fc_out1(x))
        x = torch.tanh(self.fc_out2(x))
        return x


class SelfLearningModelTK_v1(helpers.BaseSelfLearningModel):
    def __init__(self):
        super().__init__()
        self.fc_input = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.bn_input = nn.BatchNorm1d(HIDDEN_SIZE)
        self.res1 = ResBlock(HIDDEN_SIZE)
        self.res2 = ResBlock(HIDDEN_SIZE)
        self.fc_out1 = nn.Linear(HIDDEN_SIZE, 128)
        self.fc_out2 = nn.Linear(128, N_ACTIONS)
        
    def forward(self, obs):
        if isinstance(obs, list): 
            obs = obs[0]
        if isinstance(obs, torch.Tensor):
            if obs.dim() == 1: 
                obs = obs.unsqueeze(0)
        else:
            obs = torch.as_tensor(obs)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
        x = torch.relu(self.bn_input(self.fc_input(obs)))
        x = self.res1(x)
        x = self.res2(x)
        x = torch.relu(self.fc_out1(x))
        x = self.fc_out2(x)
        return x
""")

write_file('probs-main/python_impl_generic/cross_arena.py', """import os
import sys
import numpy as np
import torch
import yaml
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import environments
import helpers
from probs_impl import probs_impl_common, alphazero_adapter


class Arena:
    def __init__(self, env_name):
        self.env_name = env_name
        self.create_env = environments.get_create_env_func(env_name)

    def play_game(self, white_agent, black_agent, max_moves=200, verbose=False):
        env = self.create_env()
        env.reset()
        agents = {1: white_agent, -1: black_agent}
        for agent in agents.values():
            if isinstance(agent, alphazero_adapter.AlphaZeroAgent):
                agent.reset_mcts()
        move_count = 0
        while not env.done and move_count < max_moves:
            agent = agents[env.player]
            action = agent.get_action(env)
            reward, done = env.step(action)
            move_count += 1
            if verbose:
                env.render_ascii()
        if move_count >= max_moves:
            return 0
        result = env.game.getGameEnded(env.board, 1)
        if result > 0.5:
            return 1
        elif result < -0.5:
            return -1
        return 0

    def run_match(self, agent1, agent2, n_games=10, verbose=False):
        results = {"a1w": 0, "a1b": 0, "a2w": 0, "a2b": 0, "draws": 0, "total": 0}
        games_per_side = n_games // 2
        print("Arena Match:", agent1.get_name(), "vs", agent2.get_name())
        for i in range(games_per_side):
            result = self.play_game(agent1, agent2, verbose=verbose)
            results["total"] += 1
            if result == 1:
                results["a1w"] += 1
            elif result == -1:
                results["a2b"] += 1
            else:
                results["draws"] += 1
            print("Game", i+1, "done")
        for i in range(games_per_side):
            result = self.play_game(agent2, agent1, verbose=verbose)
            results["total"] += 1
            if result == 1:
                results["a2w"] += 1
            elif result == -1:
                results["a1b"] += 1
            else:
                results["draws"] += 1
            print("Game", games_per_side + i + 1, "done")
        a1_total = results["a1w"] + results["a1b"]
        a2_total = results["a2w"] + results["a2b"]
        print("Results:", agent1.get_name(), a1_total, "wins,", agent2.get_name(), a2_total, "wins,", results["draws"], "draws")
        return results


class RandomAgent(helpers.BaseAgent):
    def get_action(self, env):
        return env.get_random_action()
    def get_name(self):
        return "Random"


def load_probs_agent(config_path, checkpoint_path=None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env_name = config["env"]["name"]
    device = config.get("infra", {}).get("device", "cpu")
    model_keeper = probs_impl_common.create_model_keeper(config["model"], env_name)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_keeper.load_checkpoint(checkpoint_path)
    model_keeper.to(device)
    model_keeper.eval()
    return probs_impl_common.SelfLearningAgent("PROBS", model_keeper, device)


def load_alphazero_agent(checkpoint_path, num_mcts_sims=50):
    return alphazero_adapter.AlphaZeroAgent(checkpoint_path, hidden_size=256, num_mcts_sims=num_mcts_sims)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_togyzkumalak.yaml")
    parser.add_argument("--probs-checkpoint", default=None)
    parser.add_argument("--az-checkpoint", default="../../gym-togyzkumalak-master/togyzkumalak-engine/models/alphazero/best.pth.tar")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vs-random", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print("Config not found:", args.config)
        return
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env_name = config["env"]["name"]
    arena = Arena(env_name)
    probs_agent = load_probs_agent(args.config, args.probs_checkpoint)
    if os.path.exists(args.az_checkpoint):
        az_agent = load_alphazero_agent(args.az_checkpoint, args.mcts_sims)
        arena.run_match(probs_agent, az_agent, n_games=args.games, verbose=args.verbose)
    else:
        print("AlphaZero checkpoint not found:", args.az_checkpoint)
        args.vs_random = True
    if args.vs_random:
        random_agent = RandomAgent()
        print("Testing against Random")
        arena.run_match(probs_agent, random_agent, n_games=args.games, verbose=args.verbose)


if __name__ == "__main__":
    main()
""")

write_file('probs-main/python_impl_generic/cross_train.py', """import os
import sys
import numpy as np
import torch
import yaml
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

az_backend_path = os.path.abspath(os.path.join(current_dir, "../../gym-togyzkumalak-master/togyzkumalak-engine"))
if az_backend_path not in sys.path:
    sys.path.insert(0, az_backend_path)

import environments
import helpers
from probs_impl import probs_impl_common, alphazero_adapter
from backend.alphazero_trainer import TogyzkumalakGame, NNetWrapper, AlphaZeroConfig


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_device(config_device):
    if config_device is None or config_device == 'cpu':
        return 'cpu'
    elif config_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return 'cuda' if torch.cuda.is_available() else 'cpu'


class CrossTrainer:
    def __init__(self, probs_config_path, az_checkpoint_path, output_dir="checkpoints/cross_train"):
        with open(probs_config_path, 'r', encoding='utf-8') as f:
            self.probs_config = yaml.safe_load(f)
        config_device = self.probs_config.get('infra', {}).get('device', 'cpu')
        self.device = get_device(config_device)
        print("[CrossTrainer] Using device:", self.device)
        self.env_name = self.probs_config['env']['name']
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self.create_env = environments.get_create_env_func(self.env_name)
        self.probs_model_keeper = probs_impl_common.create_model_keeper(self.probs_config['model'], self.env_name)
        self.probs_model_keeper.to(self.device)
        mem_max = self.probs_config.get('infra', {}).get('mem_max_episodes', 1000)
        self.probs_experience = helpers.ExperienceReplay(max_episodes=mem_max, create_env_func=self.create_env)
        self.az_game = TogyzkumalakGame()
        self.az_config = AlphaZeroConfig(hidden_size=256, num_mcts_sims=50)
        self.az_nnet = NNetWrapper(self.az_game, self.az_config)
        self.az_checkpoint_path = az_checkpoint_path
        if os.path.exists(az_checkpoint_path):
            folder = os.path.dirname(az_checkpoint_path)
            filename = os.path.basename(az_checkpoint_path)
            try:
                self.az_nnet.load_checkpoint(folder, filename)
                print("[CrossTrainer] Loaded AlphaZero:", az_checkpoint_path)
            except Exception as e:
                print("[CrossTrainer] Could not load AZ:", e)
        self.az_examples_history = deque(maxlen=20)
        self.stats = {'probs_wins': 0, 'az_wins': 0, 'draws': 0, 'total_games': 0}

    def create_probs_agent(self):
        return probs_impl_common.SelfLearningAgent("PROBS", self.probs_model_keeper, self.device)

    def create_az_agent(self):
        agent = alphazero_adapter.AlphaZeroAgent(self.az_checkpoint_path, hidden_size=256, num_mcts_sims=50)
        agent.nnet = self.az_nnet
        agent.reset_mcts()
        return agent

    def play_single_game(self, white_agent, black_agent, collect_probs=True, collect_az=True):
        env = self.create_env()
        env.reset()
        probs_episode = helpers.ExperienceReplayEpisode() if collect_probs else None
        az_examples = [] if collect_az else None
        move_history = []
        agents = {1: white_agent, -1: black_agent}
        while not env.done:
            current_player = env.player
            agent = agents[current_player]
            is_az = isinstance(agent, alphazero_adapter.AlphaZeroAgent)
            if is_az and collect_az:
                canonical = self.az_game.getCanonicalForm(env.board, env.player)
                pi = agent.get_action_probs(env, temp=1)
                action = np.random.choice(len(pi), p=pi)
                move_history.append((current_player, action, canonical.copy(), pi.copy()))
            else:
                action = agent.get_action(env)
                if is_az:
                    canonical = self.az_game.getCanonicalForm(env.board, env.player)
                    pi = np.zeros(9)
                    pi[action] = 1.0
                    move_history.append((current_player, action, canonical.copy(), pi.copy()))
            reward, done = env.step(action)
            if collect_probs and probs_episode:
                probs_episode.on_action(action, reward, done)
        final_result = self.az_game.getGameEnded(env.board, 1)
        result = 1 if final_result > 0.5 else (-1 if final_result < -0.5 else 0)
        if collect_az and az_examples is not None:
            for (player, action, canonical, pi) in move_history:
                v = result * player
                az_examples.append((canonical, pi, v))
        if collect_probs and probs_episode:
            self.probs_experience.append_replay_episode(probs_episode)
        return result, probs_episode, az_examples

    def run_cross_games(self, num_games=20):
        all_az_examples = []
        probs_agent = self.create_probs_agent()
        az_agent = self.create_az_agent()
        for i in range(num_games):
            if i % 2 == 0:
                white_agent, black_agent = probs_agent, az_agent
                probs_is_white = True
            else:
                white_agent, black_agent = az_agent, probs_agent
                probs_is_white = False
            az_agent.reset_mcts()
            result, _, az_examples = self.play_single_game(white_agent, black_agent)
            if az_examples:
                all_az_examples.extend(az_examples)
            self.stats['total_games'] += 1
            if result == 0:
                self.stats['draws'] += 1
            elif (result == 1 and probs_is_white) or (result == -1 and not probs_is_white):
                self.stats['probs_wins'] += 1
            else:
                self.stats['az_wins'] += 1
            print("Game", i+1, "/", num_games, "done")
        return all_az_examples

    def train_probs(self, batch_size=32):
        from probs_impl import probs_impl_train_v
        if len(self.probs_experience) < batch_size:
            print("[PROBS] Not enough data:", len(self.probs_experience))
            return
        print("[PROBS] Training on", len(self.probs_experience), "episodes")
        probs_impl_train_v.train_value_model(
            self.probs_model_keeper.models['value'], self.device,
            self.probs_model_keeper.optimizers['value'], self.probs_experience,
            batch_size=batch_size, drop_ratio=0)

    def train_alphazero(self, batch_size=64):
        all_examples = []
        for examples in self.az_examples_history:
            all_examples.extend(examples)
        if len(all_examples) < batch_size:
            print("[AZ] Not enough data:", len(all_examples))
            return
        print("[AZ] Training on", len(all_examples), "examples")
        self.az_nnet.train(all_examples)

    def run_iteration(self, num_games=20, train_probs=True, train_az=True):
        print("Cross-Training Iteration (", num_games, "games)")
        az_examples = self.run_cross_games(num_games)
        if az_examples:
            self.az_examples_history.append(az_examples)
        if train_probs:
            self.train_probs()
        if train_az:
            self.train_alphazero()
        self.print_stats()
        self.save_checkpoints()

    def print_stats(self):
        total = self.stats['total_games']
        if total == 0:
            return
        print("Stats after", total, "games:")
        print("PROBS:", self.stats['probs_wins'])
        print("AZ:", self.stats['az_wins'])
        print("Draws:", self.stats['draws'])

    def save_checkpoints(self):
        ensure_dir(os.path.join(self.output_dir, "probs"))
        ensure_dir(os.path.join(self.output_dir, "alphazero"))
        self.probs_model_keeper.save_checkpoint(os.path.join(self.output_dir, "probs"), "cross")
        self.az_nnet.save_checkpoint(os.path.join(self.output_dir, "alphazero"), "cross.pth.tar")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_togyzkumalak.yaml")
    parser.add_argument("--az-checkpoint", default="../../gym-togyzkumalak-master/togyzkumalak-engine/models/alphazero/best.pth.tar")
    parser.add_argument("--output", default="checkpoints/cross_train")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=20)
    args = parser.parse_args()
    trainer = CrossTrainer(args.config, args.az_checkpoint, args.output)
    for i in range(args.iterations):
        print("ITERATION", i+1, "/", args.iterations)
        trainer.run_iteration(num_games=args.games)


if __name__ == "__main__":
    main()
""")

write_file('probs-main/python_impl_generic/probs_impl/alphazero_adapter.py', """import numpy as np
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.abspath(os.path.join(current_dir, "../../../../gym-togyzkumalak-master/togyzkumalak-engine"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from backend.alphazero_trainer import TogyzkumalakGame, NNetWrapper, MCTS, AlphaZeroConfig
import helpers


class AlphaZeroAgent(helpers.BaseAgent):
    def __init__(self, checkpoint_path, hidden_size=256, num_mcts_sims=50, cpuct=1.0):
        self.game = TogyzkumalakGame()
        self.config = AlphaZeroConfig(hidden_size=hidden_size, num_mcts_sims=num_mcts_sims, cpuct=cpuct)
        self.nnet = NNetWrapper(self.game, self.config)
        self.checkpoint_path = checkpoint_path
        self._loaded = False
        self._try_load_checkpoint()
        self.mcts = None
        self.name = "AlphaZero_" + os.path.basename(checkpoint_path)

    def _try_load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            folder = os.path.dirname(self.checkpoint_path)
            filename = os.path.basename(self.checkpoint_path)
            try:
                self.nnet.load_checkpoint(folder, filename)
                self._loaded = True
                print("[AlphaZeroAgent] Loaded checkpoint:", self.checkpoint_path)
            except Exception as e:
                print("[AlphaZeroAgent] Could not load checkpoint:", e)
                self._loaded = False
        else:
            print("[AlphaZeroAgent] Checkpoint not found:", self.checkpoint_path)
            self._loaded = False

    def reset_mcts(self):
        self.mcts = MCTS(self.game, self.nnet, self.config)

    def get_action(self, env, temp=0):
        if self.mcts is None:
            self.reset_mcts()
        canonical_board = self.game.getCanonicalForm(env.board, env.player)
        probs = self.mcts.getActionProb(canonical_board, temp=temp)
        if temp == 0:
            action = np.argmax(probs)
        else:
            action = np.random.choice(len(probs), p=probs)
        return int(action)

    def get_action_probs(self, env, temp=1):
        if self.mcts is None:
            self.reset_mcts()
        canonical_board = self.game.getCanonicalForm(env.board, env.player)
        probs = self.mcts.getActionProb(canonical_board, temp=temp)
        return np.array(probs)

    def get_name(self):
        return self.name

    def is_loaded(self):
        return self._loaded
""")

write_file('probs-main/python_impl_generic/configs/train_togyzkumalak_gpu.yaml', """name: train_togyzkumalak_gpu
env:
  name: togyzkumalak
  n_max_episode_steps: 300

cmd: train
infra:
  log: tf
  device: cuda
  sub_processes_cnt: 4
  self_play_threads: 8
  mem_max_episodes: 10000

train:
  n_high_level_iterations: 500
  v_train_episodes: 50
  q_train_episodes: 50
  q_dataset_episodes_sub_iter: 5
  dataset_drop_ratio: 0.1
  checkpoints_dir: checkpoints/togyzkumalak_gpu
  train_batch_size: 256
  self_learning_batch_size: 256
  get_q_dataset_batch_size: 128
  num_q_s_a_calls: 20
  max_depth: 15
  alphazero_move_num_sampling_moves: 10
  q_add_hardest_nodes_per_step: 5
  
evaluate:
  evaluate_n_games: 20
  randomize_n_turns: 4
  enemy:
    kind: random

model:
  value:
    class: ValueModelTK_v1
    learning_rate: 0.001
    weight_decay: 0.0001
  self_learner:
    class: SelfLearningModelTK_v1
    learning_rate: 0.001
    weight_decay: 0.0001

enemy:
  kind: random
""")

print("Successfully updated 7 files.")
