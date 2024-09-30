from gymnasium import spaces
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete, learning_rate: float, name: str):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super(DQN, self).__init__()
        assert len(observation_space.shape) == 3, "observation space must have the form channels x width x height"

        self.name = name + ".pt"

        self.conv1 = nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(64 * 7 * 7, 512)
        self.linear2 = nn.Linear(512, action_space.n)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.loss = nn.MSELoss()

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: T.Tensor):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out).flatten(start_dim=1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out

    def save_checkpoint(self):
        T.save(self.state_dict(), self.name)

    def load_checkpoint(self):
        if os.path.exists(self.name):
            self.load_state_dict(T.load(self.name))
