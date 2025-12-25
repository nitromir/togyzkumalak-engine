"""
Record a complete game with full board state at each step.
This script generates a replay.json file that can be viewed in the visualizer.
"""

import gym
import gym_togyzkumalak
import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
import io
import os

# Set encoding to UTF-8 for Kazakh characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class PolicyNetwork(nn.Module):
    """Simple MLP Policy Network for Togyzkumalak."""
    def __init__(self, input_size=128, hidden_size=64, output_size=9):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.network(x)


def get_board_state(env):
    """Extract readable board state from the environment."""
    board = env.board
    
    # Get kumalaks in each otau for both players
    white_holes = []
    black_holes = []
    white_tuzduk = -1
    black_tuzduk = -1
    
    for i in range(9):
        white_otau = board.gamers['white'].home[i]
        black_otau = board.gamers['black'].home[i]
        
        white_holes.append(white_otau.kumalaks if not white_otau.tuzduk else -1)
        black_holes.append(black_otau.kumalaks if not black_otau.tuzduk else -1)
        
        # Check if this hole is a tuzduk (captured by opponent)
        if white_otau.tuzduk:
            black_tuzduk = i  # Black captured white's hole i
        if black_otau.tuzduk:
            white_tuzduk = i  # White captured black's hole i
    
    return {
        "white": {
            "holes": white_holes,
            "kazan": board.gamers['white'].kazan.score,
            "tuzduk": white_tuzduk
        },
        "black": {
            "holes": black_holes,
            "kazan": board.gamers['black'].kazan.score,
            "tuzduk": black_tuzduk
        },
        "current_player": board.run.name
    }


def select_action(policy_net, observation, available_actions, epsilon=0.2):
    """Select action using policy network with epsilon-greedy exploration."""
    obs_flat = observation.flatten().astype(np.float32)
    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
    
    with torch.no_grad():
        action_probs = policy_net(obs_tensor).squeeze().numpy()
    
    masked_probs = action_probs * np.array(available_actions)
    
    if masked_probs.sum() == 0:
        masked_probs = np.array(available_actions, dtype=np.float32)
    
    masked_probs = masked_probs / masked_probs.sum()
    
    if np.random.random() < epsilon:
        valid_actions = [i for i, a in enumerate(available_actions) if a == 1]
        return np.random.choice(valid_actions) if valid_actions else 0
    
    action = np.random.choice(9, p=masked_probs)
    return action


def record_game(env, policy_net, game_id=1):
    """Record a complete game with all states."""
    print(f"\nRecording game {game_id}...")
    
    obs = env.reset()
    done = False
    
    # Initial state
    states = []
    initial_state = get_board_state(env)
    initial_state["step"] = 0
    initial_state["action"] = None
    initial_state["player"] = initial_state["current_player"]
    states.append(initial_state)
    
    step = 0
    while not done:
        available = env.available_action()
        current_player = env.board.run.name
        action = select_action(policy_net, obs, available)
        
        try:
            obs, reward, done, info = env.step(action)
            step += 1
            
            # Record state after move
            state = get_board_state(env)
            state["step"] = step
            state["action"] = int(action)
            state["player"] = current_player  # Player who made the move
            states.append(state)
            
        except Exception as e:
            print(f"  Error at step {step}: {e}")
            done = True
            break
    
    # Determine winner
    white_kazan = env.board.gamers['white'].kazan.score
    black_kazan = env.board.gamers['black'].kazan.score
    
    if white_kazan > 81:
        winner = "WHITE"
    elif black_kazan > 81:
        winner = "BLACK"
    elif white_kazan == 81 and black_kazan == 81:
        winner = "DRAW"
    else:
        winner = "WHITE" if white_kazan > black_kazan else "BLACK"
    
    game_record = {
        "game_id": game_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_steps": step,
        "winner": winner,
        "final_score": {
            "white": white_kazan,
            "black": black_kazan
        },
        "states": states
    }
    
    print(f"  Completed: {step} steps, Winner: {winner} (W:{white_kazan} B:{black_kazan})")
    return game_record


def main():
    print("=" * 60)
    print("TOGYZKUMALAK GAME RECORDER")
    print("=" * 60)
    
    # Create environment and network
    env = gym.make('Togyzkumalak-v0')
    policy_net = PolicyNetwork(input_size=128, hidden_size=64, output_size=9)
    
    # Record games
    num_games = 3
    all_games = []
    
    for i in range(num_games):
        game = record_game(env, policy_net, i + 1)
        all_games.append(game)
    
    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), "..", "visualizer", "replay.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_games, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(all_games)} games to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

