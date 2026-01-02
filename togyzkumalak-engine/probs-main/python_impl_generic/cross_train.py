import os
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
