import os
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


def load_probs_agent(config_path, checkpoint_path=None, agent_name=None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env_name = config["env"]["name"]
    device = config.get("infra", {}).get("device", "cpu")
    model_keeper = probs_impl_common.create_model_keeper(config["model"], env_name)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_keeper.load_checkpoint(checkpoint_path)
        if agent_name is None:
            agent_name = os.path.basename(checkpoint_path)
    if agent_name is None:
        agent_name = "PROBS"
    model_keeper.to(device)
    model_keeper.eval()
    return probs_impl_common.SelfLearningAgent(agent_name, model_keeper, device)


def load_alphazero_agent(checkpoint_path, num_mcts_sims=50):
    return alphazero_adapter.AlphaZeroAgent(checkpoint_path, hidden_size=256, num_mcts_sims=num_mcts_sims)


def main():
    parser = argparse.ArgumentParser(description="Arena for comparing PROBS checkpoints")
    parser.add_argument("--config", default="configs/train_togyzkumalak.yaml")
    parser.add_argument("--probs-checkpoint", default=None, help="First PROBS checkpoint path")
    parser.add_argument("--probs-checkpoint2", default=None, help="Second PROBS checkpoint path (for PROBS vs PROBS)")
    parser.add_argument("--az-checkpoint", default="../../gym-togyzkumalak-master/togyzkumalak-engine/models/alphazero/best.pth.tar")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vs-random", action="store_true")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print("Config not found:", args.config)
        return
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env_name = config["env"]["name"]
    arena = Arena(env_name)
    
    # PROBS vs PROBS
    if args.probs_checkpoint and args.probs_checkpoint2:
        print(f"⚔️ PROBS vs PROBS: {args.probs_checkpoint} vs {args.probs_checkpoint2}")
        probs_agent1 = load_probs_agent(args.config, args.probs_checkpoint, agent_name=os.path.basename(args.probs_checkpoint))
        probs_agent1.model_keeper.to(args.device)
        probs_agent2 = load_probs_agent(args.config, args.probs_checkpoint2, agent_name=os.path.basename(args.probs_checkpoint2))
        probs_agent2.model_keeper.to(args.device)
        arena.run_match(probs_agent1, probs_agent2, n_games=args.games, verbose=args.verbose)
        return
    
    # PROBS vs AlphaZero or Random
    probs_agent = load_probs_agent(args.config, args.probs_checkpoint, agent_name=os.path.basename(args.probs_checkpoint) if args.probs_checkpoint else None)
    probs_agent.model_keeper.to(args.device)
    
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
