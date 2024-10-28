import gymnasium as gym
from gymnasium.spaces import MultiBinary, Box, Dict, MultiDiscrete

from wrappers import *

import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np

import grid2op
from grid2op import gym_compat
from grid2op.gym_compat import ScalerAttrConverter, MultiDiscreteActSpace
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse

import torch

from GNN_Extractor import CustomGNN

class Gym2OpEnv(gym.Env):
    """
    A Gymnasium wrapper around the Grid2Op environment for power grid control.
    This environment allows the use of standard RL algorithms with Grid2Op.
    """
    
    def __init__(self, agent_type, use_gnn):
        """
        Initialize the environment with specified agent type and GNN usage.
        
        Args:
            agent_type (str): Type of agent ('base', 'improved', 'multi-agent', or 'hierarchy')
            use_gnn (bool): Whether to use Graph Neural Network for state representation
        """
        super().__init__()

        # Initialize backend and environment name
        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # Standard test case with 14 buses

        # Set up action and observation classes
        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward

        # Configure environment parameters
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Maximum number of substations that can be reconfigured per step
        p.MAX_LINE_STATUS_CHANGED = 4  # Maximum number of power lines that can be switched per step

        # Create the Grid2Op environment
        self._g2op_env = grid2op.make(
            self._env_name, 
            backend=self._backend, 
            test=False,
            action_class=action_class, 
            observation_class=observation_class,
            reward_class=reward_class, 
            param=p
        )

        # Setup reward function: combines N1 security criterion and L2RPN operational reward
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        # Create Gymnasium-compatible environment
        self._gym_env = gym_compat.GymEnv(self._g2op_env)
        self.use_gnn = use_gnn

        # Configure environment based on agent type
        if agent_type == "base":
            self.setup_base_actions()
        elif agent_type == "improved":
            self.setup_observations()
            self.setup_improved_actions()
        elif agent_type == 'multi-agent' or agent_type == 'hierarchy':
            self.setup_actions()
            if self.use_gnn:
                self.setup_observations_gnn()
            else:
                self.setup_observations()
        
        # Set observation and action spaces
        if not self.use_gnn:
            self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def setup_observations(self):
        """
        Configure the observation space for standard (non-GNN) agents.
        Normalizes generator and load powers, and selects relevant features.
        """
        obs_attr_to_keep = ["rho", "topo_vect", "gen_p", "load_p", "actual_dispatch", 'target_dispatch']
        obs_gym = self._gym_env.observation_space

        # Normalize generator power by maximum capacity
        self._gym_env.observation_space = self._gym_env.observation_space.reencode_space(
            "gen_p",
            ScalerAttrConverter(
                substract=0.,
                divide=self._g2op_env.gen_pmax
            )
        )

        # Normalize load power
        load_p_max = obs_gym["load_p"].high
        self._gym_env.observation_space = self._gym_env.observation_space.reencode_space(
            "load_p",
            ScalerAttrConverter(
                substract=0.0,
                divide=load_p_max
            )
        )

        # Keep only selected attributes
        self._gym_env.observation_space = self._gym_env.observation_space.keep_only_attr(obs_attr_to_keep)

    def setup_observations_gnn(self):
        """
        Configure the observation space for GNN-based agents.
        Creates a graph representation of the power grid with node and edge features.
        """
        obs_attr_to_keep = ["rho", "topo_vect", "gen_p", "load_p", "actual_dispatch", 'target_dispatch', 'p_or']
        self.obs_node_attr = ["gen_p", "load_p"]
        self.obs_edge_attr = ["rho", "p_or"]

        self._gym_env.observation_space.close()
        self._gym_env.observation_space = self._gym_env.observation_space.keep_only_attr(obs_attr_to_keep)

        # Define dimensions for the graph representation
        n_nodes = self._g2op_env.n_sub
        n_edges = self._g2op_env.n_line

        # Get edge connectivity information
        obs = self._g2op_env.get_obs()
        edges = list(obs.get_energy_graph().edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Define graph-structured observation space
        self.observation_space = Dict({
            "node_features": Box(low=-float('inf'), high=float('inf'), shape=(n_nodes, 2)),
            "edge_features": Box(low=-float('inf'), high=float('inf'), shape=(n_edges, 2)),
            "edge_index": Box(low=0, high=n_nodes-1, shape=(2, n_edges), dtype=int)
        })

    def _convert_observation(self, obs):
        """
        Convert Grid2Op observations to graph-structured format for GNN processing.
        
        Args:
            obs: Raw Grid2Op observation
            
        Returns:
            dict: Processed observation with node features, edge features, and connectivity
        """
        parsed_obs = obs
        n_nodes = self._g2op_env.n_sub
        n_edges = self._g2op_env.n_line

        # Build feature matrices
        node_features = self._construct_node_features(parsed_obs)
        edge_features = self._construct_edge_features(parsed_obs)

        # Convert to PyTorch tensors
        node_features = torch.tensor(node_features, dtype=torch.float32)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)

        # Get grid connectivity
        obs = self._g2op_env.get_obs()
        edges = list(obs.get_energy_graph().edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "edge_index": edge_index
        }

    def _parse_observation(self, obs):
        """
        Parse flat observation array into dictionary of named features.
        """
        parsed_obs = {}
        current_idx = 0
        for attr in self._gym_env.observation_space._attr_to_keep:
            prop = self._gym_env.observation_space._dict_properties[attr]
            shape = prop[2]
            size = np.prod(shape)
            parsed_obs[attr] = obs[current_idx:current_idx + size].reshape(shape)
            current_idx += size
        return parsed_obs

    def _construct_node_features(self, parsed_obs):
        """
        Construct node feature matrix by aggregating generator and load information.
        
        Args:
            parsed_obs (dict): Parsed observation dictionary
            
        Returns:
            numpy.ndarray: Node feature matrix (n_nodes × n_features)
        """
        node_features = np.zeros((self._g2op_env.n_sub, len(self.obs_node_attr)), dtype=np.float32)

        # Map generator powers to corresponding substations
        for gen_id, sub_id in enumerate(self._g2op_env.gen_to_subid):
            node_features[sub_id, 0] += parsed_obs["gen_p"][gen_id]

        # Map load powers to corresponding substations
        for load_id, sub_id in enumerate(self._g2op_env.load_to_subid):
            node_features[sub_id, 1] += parsed_obs["load_p"][load_id]

        return node_features

    def _construct_edge_features(self, parsed_obs):
        """
        Construct edge feature matrix from line properties.
        
        Args:
            parsed_obs (dict): Parsed observation dictionary
            
        Returns:
            numpy.ndarray: Edge feature matrix (n_edges × n_features)
        """
        edge_features = np.column_stack([parsed_obs[attr] for attr in self.obs_edge_attr])
        return edge_features.astype(np.float32)

    def setup_actions(self):
        """
        Configure action space for multi-agent and hierarchy agents.
        Includes bus changes, line status changes, generation curtailment and redispatch.
        """
        act_attr_to_keep = ["change_bus", "change_line_status", 'curtail', 'redispatch']
        self._gym_env.action_space = self._gym_env.action_space.keep_only_attr(act_attr_to_keep)
        
    def setup_base_actions(self):
        """Configure action space for base agent using discrete actions."""
        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space)
    
    def setup_improved_actions(self):
        """
        Configure action space for improved agent.
        Similar to multi-agent setup but using MultiDiscreteActSpace.
        """
        act_attr_to_keep = ["change_bus", "change_line_status", 'curtail', 'redispatch']
        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space, act_attr_to_keep)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            tuple: (observation, info)
        """
        obs, info = self._gym_env.reset(seed=seed)
        if self.use_gnn:
            obs = self._convert_observation(obs)
        return obs, info

    def step(self, action):
        """
        Take a step in the environment using the specified action.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        obs, reward, done, truncated, info = self._gym_env.step(action)
        if self.use_gnn:
            obs = self._convert_observation(obs)
        return obs, reward, done, truncated, info

    def render(self):
        """Render the environment."""
        return self._gym_env.render()


def main():

    agent = input("Which agent do you want to run:\nbase, improved, multi-agent, hierarchy, random \n")
    train_in = input('Do you want to train a model? \nyes or no \n')
    train = True if train_in[0]=='y' else False
    use_gnn_in = input('Do you want to use a GNN? \nOnly available for multi-agent and hierarchy \n')
    use_gnn = True if use_gnn_in[0] == 'y' else False

    max_steps = 10000

    env = Gym2OpEnv(agent_type=agent, use_gnn=use_gnn)

    print("#####################")
    print("# OBSERVATION SPACE #")
    print("#####################")
    print(env.observation_space)
    print("#####################\n")

    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    print(env.action_space)
    print("#####################\n\n")

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 500000,
        "env_name": "Grid2Op",
    }

    policy_kwargs = dict(
            features_extractor_class=CustomGNN,
            features_extractor_kwargs=dict(features_dim=1120),
        )

    if (agent == 'base' or agent == 'improved'):
        if agent == 'base':
            model = PPO('MultiInputPolicy', env=env, verbose=1, tensorboard_log="runs/base_agent")
            model_name = 'base_agent'
            if train is None:
                model.set_parameters('models/base_agent')
        else:
            model = PPO('MultiInputPolicy', env=env, verbose=1, tensorboard_log="runs/improved_agent")
            model_name = 'improved_agent'
            if train is None:
                model.set_parameters('models/improved_agent')

        if train == True:
            model.learn(100000, progress_bar=True)
            model.save(f'models/{model_name}')


    action_names = ['change_bus', 'change_line_status', 'curtail', 'redispatch']


    # # print(f"step = {curr_step} (reset):")
    # # print(f"\t obs = {obs}")
    # # print(f"\t info = {info}\n\n")

    if agent == 'multi-agent' or agent == 'hierarchy':

        # Create a separate environment for each agent
        env_change_bus_egent = SingleAgentWrapper(env,"change_bus")
        env_change_line_status_agent = SingleAgentWrapper(env, "change_line_status")
        env_curtail_agent = SingleAgentWrapper(env, "curtail")
        env_redispatch_agent = SingleAgentWrapper(env, "redispatch")

        # Create PPO models for each agent
        if use_gnn:
            ppo_change_bus = PPO(policy='MultiInputPolicy', env=env_change_bus_egent, verbose=1, tensorboard_log="runs/change_bus",policy_kwargs=policy_kwargs)
            ppo_change_line_status = PPO(policy='MultiInputPolicy', env=env_change_line_status_agent, verbose=1, tensorboard_log="runs/change_line",policy_kwargs=policy_kwargs)
            ppo_curtail = PPO(policy='MultiInputPolicy', env=env_curtail_agent, verbose=1, tensorboard_log="runs/curtail",policy_kwargs=policy_kwargs)
            ppo_redispatch = PPO(policy='MultiInputPolicy', env=env_redispatch_agent, verbose=1, tensorboard_log="runs/redispatch",policy_kwargs=policy_kwargs)    
        else:
            ppo_change_bus = PPO(policy='MultiInputPolicy', env=env_change_bus_egent, verbose=1, tensorboard_log="runs/change_bus",)
            ppo_change_line_status = PPO(policy='MultiInputPolicy', env=env_change_line_status_agent, verbose=1, tensorboard_log="runs/change_line",)
            ppo_curtail = PPO(policy='MultiInputPolicy', env=env_curtail_agent, verbose=1, tensorboard_log="runs/curtail",)
            ppo_redispatch = PPO(policy='MultiInputPolicy', env=env_redispatch_agent, verbose=1, tensorboard_log="runs/redispatch",)

        if train is None and use_gnn:
            ppo_change_bus.set_parameters('models/multi-agent models/change_bus_gnn_agent')
            ppo_change_line_status.set_parameters('models/multi-agent models/change_line_status_gnn_agent')
            ppo_curtail.set_parameters('models/multi-agent models/curtail_gnn_agent')
            ppo_redispatch.set_parameters('models/multi-agent models/redispatch_gnn_agent')
        elif train is None:
            ppo_change_bus.set_parameters('models/multi-agent models/change_bus_agent')
            ppo_change_line_status.set_parameters('models/multi-agent models/change_line_status_agent')
            ppo_curtail.set_parameters('models/multi-agent models/curtail_agent')
            ppo_redispatch.set_parameters('models/multi-agent models/redispatch_agent')


        ppo_agents = [ppo_change_bus, ppo_change_line_status, ppo_curtail, ppo_redispatch, ]
        multi_agent_env = MultiAgentWrapper(env, action_names) 



        if train and agent == 'multi-agent':

            # Train each agent separately

            ppo_change_bus.learn(total_timesteps=100000, progress_bar=True, )
            if use_gnn:
                ppo_change_bus.save('models/multi-agent models/change_bus_gnn_agent')
            else:
                ppo_change_bus.save('models/multi-agent models/change_bus_agent')

            ppo_change_line_status.learn(total_timesteps=100000, progress_bar=True, )
            if use_gnn:
                ppo_change_line_status.save('models/multi-agent models/change_line_status_gnn_agent')
            else:
                ppo_change_line_status.save('models/multi-agent models/change_line_status_agent')
            
            ppo_curtail.learn(total_timesteps=100000, progress_bar=True,)
            if use_gnn:
                ppo_curtail.save('models/multi-agent models/curtail_gnn_agent')
            else:
                ppo_curtail.save('models/multi-agent models/curtail_agent')
            
            ppo_redispatch.learn(total_timesteps=100000, progress_bar=True)
            if use_gnn:
                ppo_redispatch.save('models/multi-agent models/redispatch_gnn_agent')
            else:
                ppo_redispatch.save('models/multi-agent models/redispatch_agent')    



    episodes = 10
    avg_return = 0
    avg_step = 0

    if agent == "hierarchy":
        hierarchy_env = HierarchyWrapper(env,action_names,ppo_agents)

        if use_gnn:
            hierarchy_agent = PPO('MultiInputPolicy', env=hierarchy_env, policy_kwargs=policy_kwargs, tensorboard_log="runs/hierarchy_agent", verbose=1)
        else:
            hierarchy_agent = PPO('MultiInputPolicy', env=hierarchy_env, tensorboard_log="runs/hierarchy_agent", verbose=1)


        if train:
            name = f'{agent}'
            name += f'agent' if not use_gnn else f'_gnn_agent'

            hierarchy_agent.learn(total_timesteps=500000, progress_bar=True,)
            hierarchy_agent.save(f'models/{name}')

        if use_gnn:
            hierarchy_agent.set_parameters('models/hierarchy_gnn_agent')
        else:
            hierarchy_agent.set_parameters('models/hierarchy_agent')

    for e in range(episodes):
        curr_step = 0
        curr_return = 0

        is_done = False
        if agent == 'multi-agent':
            obs, info = multi_agent_env.reset()
        elif agent == 'hierarchy':
            obs, info = hierarchy_env.reset()
        else:
            obs, info = env.reset() 

        while not is_done and curr_step < max_steps:
            # action = env.action_space.sample()
            if agent == 'multi-agent':
                actions = []

                actions.append(ppo_change_bus.predict(obs)[0])
                actions.append(ppo_change_line_status.predict(obs)[0])
                actions.append(ppo_curtail.predict(obs)[0])
                actions.append(ppo_redispatch.predict(obs)[0])
                obs, reward, terminated, truncated, info = multi_agent_env.step(actions)
            elif agent == 'hierarchy':
                
                action = hierarchy_agent.predict(obs)
                obs, reward, terminated, truncated, info = hierarchy_env.step(action[0])
            elif agent =='random':
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                action = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action[0])             

            curr_step += 1
            curr_return += reward
            is_done = terminated or truncated

        avg_return += curr_return
        avg_step += curr_step

    avg_return /= episodes
    avg_step /= episodes

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {avg_return}")
    print(f"total steps = {avg_step}")
    print("###########")


if __name__ == "__main__":
    main()
