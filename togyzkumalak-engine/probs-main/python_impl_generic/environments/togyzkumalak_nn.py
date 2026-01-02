import torch
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
