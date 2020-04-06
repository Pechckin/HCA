
from gym import make
import random
import numpy as np
from MountainCar import Continuous_MountainCarEnv
from collections import namedtuple, deque
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),

        )
        self.mu = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.sigma = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softplus()
        )

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.sigma(base_out)


class Credit(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Credit, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(state_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),

        )
        self.mu = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.sigma = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softplus()
        )

    def forward(self, x, r):
        base_out = self.base(torch.cat([x, r.unsqueeze(1)], dim=1))
        return self.mu(base_out), self.sigma(base_out)

