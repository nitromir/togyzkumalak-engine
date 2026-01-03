import numpy as np
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
    def __init__(self, checkpoint_path, hidden_size=256, num_mcts_sims=200, cpuct=1.0):
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
