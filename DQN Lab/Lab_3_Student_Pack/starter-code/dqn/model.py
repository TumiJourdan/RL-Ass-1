import numpy as np
from gym import spaces
import torch.nn as nn
import gym
import torch
from torchsummary import summary

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete,learning_rate=0.0001):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert (
            type(observation_space) == spaces.Box
        ), "observation_space must be of type Box"
        assert (
            len(observation_space.shape) == 3
        ), "observation space must have the form channels x width x height"
        assert (
            type(action_space) == spaces.Discrete
        ), "action_space must be of type Discrete"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Convolutional layers
        self.conv1 = self.conv_block(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = self.conv_block(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = self.conv_block(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Compute the size of the flattened features after the convolutional layers
        # Assuming input size is (4, 84, 84)

        # Fully connected layers
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, action_space.n)


    def forward(self, x):
        # Convert input to tensor and ensure proper type and shape

        x = np.array(x)
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if ( len(x.shape)==3):
            x= x.unsqueeze(0) 
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the tensor for fully connected layers
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x,start_dim=1)
        # Apply fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x.cpu()

    def conv_block(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.ReLU()
        )




# env = gym.make("PongNoFrameskip-v4")

# model=DQN(env.observation_space,env.action_space)
# model= model.to(device)

# summary(model, (4,86,86))