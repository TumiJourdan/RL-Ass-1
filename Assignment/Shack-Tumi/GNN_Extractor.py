import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
# debugging packages
import torch_geometric
from torch_geometric.data import Data,Batch
import networkx as nx
import matplotlib.pyplot as plt

# def create_node_gen_load_list(obs):

#     n_nodes = obs.n_sub

#     node_info = [[0.0, 0.0] for _ in range(n_nodes)]  # Initialize with [gen_p, load_p]
    
#     # Assign generator power to nodes
#     for gen_id, gen_sub_id in enumerate(obs.gen_to_subid):
#         node_info[gen_sub_id][0] = obs.gen_p[gen_id]
    
#     # Assign load power to nodes
#     for load_id, load_sub_id in enumerate(obs.load_to_subid):
#         node_info[load_sub_id][1] = obs.load_p[load_id]
    
#     return torch.tensor(node_info, dtype=torch.float32)

# def create_edge_info_list(obs):
#     n_edges = obs.n_line
#     edge_info = [[0.0, 0.0] for _ in range(n_edges)]  # Initialize with [rho, p_or]
    
#     for line_id in range(n_edges):
#         edge_info[line_id][0] = obs.rho[line_id]  # Line capacity usage (rho)
#         edge_info[line_id][1] = obs.p_or[line_id]  # Active power at origin side of the line
#     return torch.tensor(edge_info, dtype=torch.float32)

class CustomGNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 14*20*4):
        super(CustomGNN, self).__init__(observation_space, features_dim)
        n_node_features_number = observation_space["node_features"].shape[1]
        n_edge_features_number = observation_space["edge_features"].shape[1]
        
        self.conv1 = NNConv(n_node_features_number, 32, self.create_edge_nn(n_edge_features_number, 32 * n_node_features_number))
        self.conv2 = NNConv(32, 16, self.create_edge_nn(n_edge_features_number,16 * 32))

        number_of_nodes = observation_space["node_features"].shape[0]
        self.fc = nn.Linear(16, 14*20*4)

    def create_edge_nn(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )

    def forward(self, observations: dict) -> torch.Tensor:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        batch_size = observations["node_features"].shape[0]
        data_list = []
        
        for i in range(batch_size):
            nodes_feat = observations["node_features"][i].to(device)
            edges_feat = observations["edge_features"][i].to(device)
            edge_index = observations["edge_index"][i].to(device).to(torch.long)
            
            data = Data(x=nodes_feat, edge_index=edge_index, edge_attr=edges_feat)
            data_list.append(data)
        
        # Combine graphs into a batch
        batch = Batch.from_data_list(data_list).to(device)
        
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        # Global mean pooling
        x = global_mean_pool(x, batch.batch)
        # Final linear layer
        x = self.fc(x)
        return x