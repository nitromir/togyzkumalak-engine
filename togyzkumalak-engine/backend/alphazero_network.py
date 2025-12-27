import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np

# Add alpha-zero-general to path
ALPHAZERO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "alpha-zero-general-master"))
sys.path.append(ALPHAZERO_PATH)

try:
    from NeuralNet import NeuralNet
except ImportError:
    print(f"[WARNING] Could not find alpha-zero-general at {ALPHAZERO_PATH}")
    class NeuralNet:
        pass

from .game_manager import TogyzkumalakBoard

class AlphaZeroNet(nn.Module):
    """
    Dual-head Neural Network for AlphaZero.
    Policy head: move probabilities.
    Value head: position evaluation [-1, 1].
    """
    def __init__(self, input_size=128, hidden_size=512, output_size=9):
        super(AlphaZeroNet, self).__init__()
        
        # Backbone (Shared)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Policy Head
        self.policy_head = nn.Linear(hidden_size // 2, output_size)
        
        # Value Head
        self.value_head = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Shared features
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Policy output (logits)
        pi = self.policy_head(x)
        
        # Value output
        v = torch.tanh(self.value_head(x))
        
        return pi, v

class TogyzkumalakAlphaZeroNet(NeuralNet):
    """
    Wrapper for AlphaZeroNet to fit the NeuralNet interface.
    """
    def __init__(self, game):
        super().__init__(game)
        self.net = AlphaZeroNet(input_size=128, hidden_size=512, output_size=9)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.game = game

    def train(self, examples):
        """
        Trains the network with examples.
        examples: list of (board, pi, v)
        """
        import torch.optim as optim
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        
        self.net.train()
        
        for epoch in range(10):  # Simplified loop
            for board, pi, v in examples:
                # Convert board fields to observation
                obs = self.game.get_observation(board)
                
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                target_pi = torch.FloatTensor(pi).to(self.device)
                target_v = torch.FloatTensor([v]).to(self.device)
                
                # Forward
                pi_logits, v_pred = self.net(obs_tensor)
                
                # Loss
                loss_pi = -torch.sum(target_pi * F.log_softmax(pi_logits, dim=-1))
                loss_v = F.mse_loss(v_pred, target_v)
                total_loss = loss_pi + loss_v
                
                # Backward
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        Input: board (raw fields)
        Returns: pi (softmax), v [-1, 1]
        """
        obs = self.game.get_observation(board)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        self.net.eval()
        with torch.no_grad():
            pi_logits, v = self.net(obs_tensor)
            pi = torch.softmax(pi_logits, dim=-1)
            
        return pi.cpu().numpy()[0], float(v.cpu().numpy()[0])

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.net.state_dict(), filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception(f"No checkpoint found at {filepath}")
        self.net.load_state_dict(torch.load(filepath, map_location=self.device))
