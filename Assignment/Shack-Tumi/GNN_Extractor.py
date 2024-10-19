import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
# debugging packages
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt


class CustomGNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(CustomGNN, self).__init__(observation_space, features_dim)
        print("obs space")
        print(type(observation_space["features"]))
        print(observation_space["features"].shape[0])
        self.num_node_features = observation_space["features"].shape[0]
        
        self.conv1 = GCNConv(self.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = nn.Linear(64, features_dim)

    def forward(self, observations) -> torch.Tensor:
        x = observations["features"]
        edge_index = observations["edge_index"]
        data = torch_geometric.Data(x=x, edge_index=edge_index)
        g = torch_geometric.utils.to_networkx(data, to_undirected=True)
        print(data)
        nx.draw(g)
        plt.show()

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        x = torch.mean(x, dim=0)  # Global mean pooling
        return F.relu(self.fc(x))