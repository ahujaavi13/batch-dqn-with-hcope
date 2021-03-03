import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_states, fc1_dim, fc2_dim, num_actions, seed):
        """Deep Q-Network"""
        super(DQN, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)

        self.num_states = num_states
        self.fc1_dims = fc1_dim
        self.fc2_dims = fc2_dim
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.num_actions)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, obs, is_training):
        """Forward"""

        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        action_values = self.fc3(x)

        if not is_training:
            action_values = self.softmax(action_values)

        return action_values
