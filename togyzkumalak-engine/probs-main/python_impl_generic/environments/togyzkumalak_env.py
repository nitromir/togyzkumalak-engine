import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Пробуем разные варианты путей к backend
backend_paths = [
    os.path.abspath(os.path.join(current_dir, "../../../gym-togyzkumalak-master/togyzkumalak-engine")),  # Локальная структура
    os.path.abspath(os.path.join(current_dir, "../../togyzkumalak-engine")),     # Структура на сервере
    os.path.abspath(os.path.join(current_dir, "../../..")),                        # Альтернативный вариант
]

for backend_path in backend_paths:
    if os.path.exists(os.path.join(backend_path, "backend", "alphazero_trainer.py")):
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        break

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
