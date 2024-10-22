import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import grid2op
from grid2op import gym_compat
from grid2op.gym_compat import BoxGymObsSpace,GymEnv

# from grid2op
env = grid2op.make("l2rpn_case14_sandbox", test=True)
obs = env.get_obs()
# wraped as a gym environment
gym_env = GymEnv(env)

gym_obs = gym_env.observation_space

# we have edge connections amount of rhos
print("Capacity of powerlines")
print("Grid2OP : ",obs.rho)
print("Gym : ",gym_obs["rho"])

print("Power of powerlines")
print("Grid2OP : ",obs.p_or)
print("Gym : ",gym_obs["p_or"])

print("The active production value of each generator (expressed in MW)")
print("Grid2OP : ",obs.gen_p)
print("Gym : ",gym_obs["gen_p"])

print("The active load value of each consumption (expressed in MW).")
print("Grid2OP : ",obs.load_p)
print("Gym : ",gym_obs["load_p"])

print("Gennis ",obs.gen_to_subid)

print("///////////////////")
print(obs.get_energy_graph().nodes)
print(obs.get_energy_graph().edges)

print(obs.connectivity_matrix())

def create_node_gen_load_list(obs):

    n_nodes = obs.n_sub

    node_info = [[0.0, 0.0] for _ in range(n_nodes)]  # Initialize with [gen_p, load_p]
    
    # Assign generator power to nodes
    for gen_id, gen_sub_id in enumerate(obs.gen_to_subid):
        node_info[gen_sub_id][0] = obs.gen_p[gen_id]
    
    # Assign load power to nodes
    for load_id, load_sub_id in enumerate(obs.load_to_subid):
        node_info[load_sub_id][1] = obs.load_p[load_id]
    
    return torch.tensor(node_info, dtype=torch.float32)

def create_edge_info_list(obs):
    n_edges = obs.n_line
    edge_info = [[0.0, 0.0] for _ in range(n_edges)]  # Initialize with [rho, p_or]
    
    for line_id in range(n_edges):
        edge_info[line_id][0] = obs.rho[line_id]  # Line capacity usage (rho)
        edge_info[line_id][1] = obs.p_or[line_id]  # Active power at origin side of the line
    return torch.tensor(edge_info, dtype=torch.float32)


nodes_feat = create_node_gen_load_list(obs)
edges_feat = create_edge_info_list(obs)
print(nodes_feat)
print(edges_feat)

edges = list(obs.get_energy_graph().edges)

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(edge_index)
# Step 2: Create dummy node features
num_nodes = max(max(u, v) for u, v in edges) + 1  # Number of unique nodes

# Step 3: Create a Data object
data = Data(x=nodes_feat, edge_index=edge_index,edge_attr=edges_feat,num_nodes = obs.n_sub)
g = torch_geometric.utils.to_networkx(data, to_undirected=True)
print(data)
nx.draw(g)
plt.show()