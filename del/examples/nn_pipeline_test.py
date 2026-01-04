"""
Neural Network Pipeline Test for Togyzkumalak Gym
This script tests the full pipeline: Environment -> Neural Network -> Training Loop
"""

import gym
import gym_togyzkumalak
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import sys
import io

# Set encoding to UTF-8 for Kazakh characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Log file path for debug instrumentation
LOG_PATH = r"c:\Users\Admin\Documents\Toguzkumalak\.cursor\debug.log"

def log_debug(location, message, data, hypothesis_id="PIPELINE"):
    """Write NDJSON log entry to debug file."""
    # #region agent log
    entry = {
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "sessionId": "nn-pipeline-test",
        "runId": "run1",
        "hypothesisId": hypothesis_id
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # #endregion


class PolicyNetwork(nn.Module):
    """
    Simple MLP Policy Network for Togyzkumalak.
    Input: 128 features (observation space)
    Output: 9 action probabilities (one for each otau/hole)
    """
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


def select_action(policy_net, observation, available_actions, epsilon=0.1):
    """
    Select action using policy network with epsilon-greedy exploration.
    Masks invalid actions.
    """
    # Flatten observation if needed
    obs_flat = observation.flatten().astype(np.float32)
    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
    
    # Get action probabilities from network
    with torch.no_grad():
        action_probs = policy_net(obs_tensor).squeeze().numpy()
    
    # Mask invalid actions (where available_actions is 0)
    masked_probs = action_probs * np.array(available_actions)
    
    # If all valid actions have zero probability, use uniform distribution
    if masked_probs.sum() == 0:
        masked_probs = np.array(available_actions, dtype=np.float32)
    
    # Normalize probabilities
    masked_probs = masked_probs / masked_probs.sum()
    
    # Epsilon-greedy: random action with probability epsilon
    if np.random.random() < epsilon:
        valid_actions = [i for i, a in enumerate(available_actions) if a == 1]
        return np.random.choice(valid_actions) if valid_actions else 0
    
    # Sample from probability distribution
    action = np.random.choice(9, p=masked_probs)
    return action


def run_single_game(env, policy_net, game_id):
    """Run a single game and return statistics."""
    # #region agent log
    log_debug(
        "nn_pipeline_test.py:run_single_game",
        f"Starting game {game_id}",
        {"game_id": game_id},
        "H1_ENV_WORKS"
    )
    # #endregion
    
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    actions_taken = []
    
    while not done:
        available = env.available_action()
        action = select_action(policy_net, obs, available, epsilon=0.3)
        actions_taken.append(action)
        
        try:
            obs, reward, done, info = env.step(action)
            total_reward = reward
            steps += 1
        except Exception as e:
            # #region agent log
            log_debug(
                "nn_pipeline_test.py:run_single_game:exception",
                f"Error during step in game {game_id}",
                {"game_id": game_id, "error": str(e), "step": steps},
                "H2_STEP_ERROR"
            )
            # #endregion
            done = True
            break
    
    # #region agent log
    log_debug(
        "nn_pipeline_test.py:run_single_game:complete",
        f"Game {game_id} completed",
        {
            "game_id": game_id,
            "total_steps": steps,
            "final_reward": float(total_reward),
            "actions_taken": [int(a) for a in actions_taken[:10]],  # First 10 actions
            "winner": "WHITE" if total_reward > 0 else ("BLACK" if total_reward < 0 else "DRAW")
        },
        "H1_ENV_WORKS"
    )
    # #endregion
    
    return {
        "game_id": game_id,
        "steps": steps,
        "reward": float(total_reward),
        "winner": "WHITE" if total_reward > 0 else ("BLACK" if total_reward < 0 else "DRAW")
    }


def main():
    print("=" * 60)
    print("TOGYZKUMALAK NEURAL NETWORK PIPELINE TEST")
    print("=" * 60)
    
    # #region agent log
    log_debug(
        "nn_pipeline_test.py:main:start",
        "Pipeline test started",
        {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "H1_ENV_WORKS"
    )
    # #endregion
    
    # Step 1: Create environment
    print("\n[1] Creating Togyzkumalak environment...")
    try:
        env = gym.make('Togyzkumalak-v0')
        obs = env.reset()
        obs_shape = obs.shape
        print(f"    Environment created successfully!")
        print(f"    Observation shape: {obs_shape}")
        print(f"    Action space: {env.action_space}")
        
        # #region agent log
        log_debug(
            "nn_pipeline_test.py:main:env_created",
            "Environment initialized",
            {
                "observation_shape": str(obs_shape),
                "action_space_n": 9,
                "initial_obs_sample": obs.flatten()[:10].tolist()
            },
            "H1_ENV_WORKS"
        )
        # #endregion
    except Exception as e:
        # #region agent log
        log_debug(
            "nn_pipeline_test.py:main:env_error",
            "Failed to create environment",
            {"error": str(e)},
            "H3_ENV_FAIL"
        )
        # #endregion
        print(f"    ERROR: {e}")
        return
    
    # Step 2: Create neural network
    print("\n[2] Creating Policy Neural Network...")
    try:
        policy_net = PolicyNetwork(input_size=128, hidden_size=64, output_size=9)
        
        # Test forward pass
        test_input = torch.randn(1, 128)
        test_output = policy_net(test_input)
        
        print(f"    Network architecture:")
        print(f"      Input:  128 features (observation)")
        print(f"      Hidden: 64 neurons x 2 layers (ReLU)")
        print(f"      Output: 9 actions (softmax probabilities)")
        print(f"    Test forward pass: {test_output.shape} -> sum={test_output.sum().item():.4f}")
        
        # Count parameters
        total_params = sum(p.numel() for p in policy_net.parameters())
        print(f"    Total parameters: {total_params}")
        
        # #region agent log
        log_debug(
            "nn_pipeline_test.py:main:network_created",
            "Neural network initialized",
            {
                "total_parameters": total_params,
                "test_output_shape": str(test_output.shape),
                "test_output_sum": float(test_output.sum().item()),
                "architecture": "MLP 128->64->64->9"
            },
            "H4_NN_WORKS"
        )
        # #endregion
    except Exception as e:
        # #region agent log
        log_debug(
            "nn_pipeline_test.py:main:network_error",
            "Failed to create network",
            {"error": str(e)},
            "H4_NN_WORKS"
        )
        # #endregion
        print(f"    ERROR: {e}")
        return
    
    # Step 3: Run test games
    print("\n[3] Running test games (Neural Network vs Random sampling)...")
    num_games = 10
    results = []
    
    for i in range(num_games):
        result = run_single_game(env, policy_net, i + 1)
        results.append(result)
        winner_symbol = "W" if result["winner"] == "WHITE" else ("B" if result["winner"] == "BLACK" else "D")
        print(f"    Game {i+1:2d}: {result['steps']:3d} steps, reward={result['reward']:+.1f} [{winner_symbol}]")
    
    # Step 4: Aggregate statistics
    print("\n[4] Test Results Summary:")
    print("-" * 40)
    
    white_wins = sum(1 for r in results if r["winner"] == "WHITE")
    black_wins = sum(1 for r in results if r["winner"] == "BLACK")
    draws = sum(1 for r in results if r["winner"] == "DRAW")
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])
    
    summary = {
        "total_games": num_games,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "avg_steps_per_game": float(avg_steps),
        "avg_reward": float(avg_reward),
        "win_rate_white": white_wins / num_games,
        "win_rate_black": black_wins / num_games
    }
    
    print(f"    Total games played:  {num_games}")
    print(f"    White wins:          {white_wins} ({white_wins/num_games*100:.1f}%)")
    print(f"    Black wins:          {black_wins} ({black_wins/num_games*100:.1f}%)")
    print(f"    Draws:               {draws} ({draws/num_games*100:.1f}%)")
    print(f"    Average steps/game:  {avg_steps:.1f}")
    print(f"    Average reward:      {avg_reward:+.3f}")
    
    # #region agent log
    log_debug(
        "nn_pipeline_test.py:main:summary",
        "Pipeline test completed - Final Summary",
        summary,
        "H5_PIPELINE_COMPLETE"
    )
    # #endregion
    
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nLogs written to: {LOG_PATH}")
    
    return summary


if __name__ == "__main__":
    main()

