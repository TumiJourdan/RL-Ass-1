import gymnasium as gym
from gymnasium.spaces import MultiBinary, Box, Dict, MultiDiscrete

import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
# from typing import Dict, List, Tuple

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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*utcnow.*")

import torch

from GNN_Extractor import CustomGNN

class SingleAgentWrapper(gym.Env): #This is used for the individual agents in the multiagent
    def __init__(self, env, agent_action_key):
        super(SingleAgentWrapper, self).__init__()
        self.env = env
        self.agent_action_key = agent_action_key  # e.g., "change_line_status", "curtail", "redispatch"
        
        # Share the observation space among all agents
        self.observation_space = self.env.observation_space
        
        # Action space is limited to one key for this agent
        self.action_space = self.env.action_space[self.agent_action_key]

    def step(self, action):
        # Create a combined action dict where only the relevant part is filled by this agent
        combined_action = {self.agent_action_key: action}
        return self.env.step(combined_action)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

class MultiAgentWrapper(gym.Env):
    def __init__(self, env, agent_action_keys:list):
        super(MultiAgentWrapper, self).__init__()
        self.env = env
        self.agent_action_keys = agent_action_keys  # e.g., "change_line_status", "curtail", "redispatch"
        
        # Share the observation space among all agents
        self.observation_space = self.env.observation_space
        
        # Action space is limited to one key for this agent
        self.action_space = self.env.action_space

    def step(self, actions):
        # Create a combined action dict where only the relevant part is filled by this agent
        combined_action = {
            self.agent_action_keys[0]: actions[0],
            self.agent_action_keys[1]: actions[1],
            self.agent_action_keys[2]: actions[2],
            self.agent_action_keys[3]: actions[3]
        }
        return self.env.step(combined_action)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def render(self, mode='human'):
        return self.env.render(mode=mode)
    

class HierarchyWrapper(gym.Env):
    def __init__(self, env, agent_action_keys:list, agent_list) -> None:
        super(HierarchyWrapper, self).__init__()

        self.env = env

        self.agent_action_keys = agent_action_keys  # e.g., "change_line_status", "curtail", "redispatch"
        self.agent_list = agent_list

        # Share the observation space among all agents
        # self.observation_space = self.env.hierarchy_observation_space
        self.observation_space = self.env.observation_space

        self.action_space = MultiBinary(len(agent_list))

        self.obs = None

    def step(self, action):
        # Create a combined action dict where only the relevant part is filled by this agent

        combined_action = {}

        # print(action)

        for i, act in enumerate(action):
            if act:
                combined_action[self.agent_action_keys[i]] = self.agent_list[i].predict(self.obs)[0]

        # print(combined_action)

        step_return = self.env.step(combined_action)

        self.obs = step_return[0]

        return step_return

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.obs = obs
        return obs, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)


# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(
            self,
            agent_type,
            use_gnn
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.use_gnn = use_gnn

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
        
        if not self.use_gnn:
            self.observation_space = self._gym_env.observation_space
        
        self.action_space = self._gym_env.action_space

    def setup_observations(self):
        # TODO: Your code to specify & modify the observation space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        # print("WARNING: setup_observations is not doing anything. Implement your own code in this method.")
        obs_attr_to_keep = ["rho", "topo_vect", "gen_p", "load_p", "actual_dispatch", 'target_dispatch']

        obs_gym = self._gym_env.observation_space

        self._gym_env.observation_space = self._gym_env.observation_space.reencode_space("gen_p",
                                   ScalerAttrConverter(substract=0.,
                                                       divide=self._g2op_env.gen_pmax
                                                       )
                                   )

        # Use the scaled bounds to create the ScalerAttrConverter
        load_p_max = obs_gym["load_p"].high
        
        # # Normalize load_p between 0 and 1
        self._gym_env.observation_space = self._gym_env.observation_space.reencode_space(
            "load_p",
            ScalerAttrConverter(
                substract=0.0,  # Minimum value
                divide=load_p_max  # Maximum value
            )
        )

        self._gym_env.observation_space = self._gym_env.observation_space.keep_only_attr(obs_attr_to_keep)

    def setup_observations_gnn(self):
        """Define the observation space with node and edge features."""
        # TODO: Your code to specify & modify the observation space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        obs_attr_to_keep = ["rho", "topo_vect", "gen_p", "load_p", "actual_dispatch", 'target_dispatch', 'p_or']
        self.obs_node_attr = ["gen_p","load_p"]
        self.obs_edge_attr = ["rho","p_or"]

        self._gym_env.observation_space.close()

        # this is important as it is what is returned in our step and our reset functions (to us , we then can change it and have the function return somthing else)
        # we have to use their template code here as we cant control what the step function returns, we can only limit it
        self._gym_env.observation_space = self._gym_env.observation_space.keep_only_attr(obs_attr_to_keep)

        n_nodes = self._g2op_env.n_sub
        n_edges = self._g2op_env.n_line

        obs = self._g2op_env.get_obs()
        edges = list(obs.get_energy_graph().edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Define the observation space // this gets passed to the extractor initialization
        self.observation_space = Dict({
            "node_features": Box(low=-float('inf'), high=float('inf'), shape=(n_nodes, 2)),
            "edge_features": Box(low=-float('inf'), high=float('inf'), shape=(n_edges, 2)),
            "edge_index": Box(low=0, high=n_nodes-1, shape=(2, n_edges), dtype=int)
        })


    def _convert_observation(self, obs):
        """Convert flattened Box observations into tensors for nodes and edges."""
        # Parse the observation dictionary into separate arrays keyed by attribute name
        # parsed_obs = self._parse_observation(obs)
        parsed_obs = obs
        # Get the number of nodes and edges from the environment
        n_nodes = self._g2op_env.n_sub
        n_edges = self._g2op_env.n_line

        # Construct the node and edge features using the parsed observation
        node_features = self._construct_node_features(parsed_obs)  # Shape: (n_nodes, n_node_attr)
        edge_features = self._construct_edge_features(parsed_obs)  # Shape: (n_edges, n_edge_attr)

        # Convert the node and edge features into tensors
        node_features = torch.tensor(node_features, dtype=torch.float32)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)

        # Generate the edge index (connectivity graph) as a tensor
        obs = self._g2op_env.get_obs()
        edges = list(obs.get_energy_graph().edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "edge_index": edge_index
        }
    
    def _parse_observation(self, obs):
        parsed_obs = {}
        current_idx = 0
        for attr in self._gym_env.observation_space._attr_to_keep:
            prop = self._gym_env.observation_space._dict_properties[attr]
            shape = prop[2]
            size = np.prod(shape)
            parsed_obs[attr] = obs[current_idx:current_idx + size].reshape(shape)
            current_idx += size
        return parsed_obs
    
    # sadly we need to do this by hand
    def _construct_node_features(self, parsed_obs):
        """Construct node feature matrix (n_sub × n_node_attr)."""
        node_features = np.zeros((self._g2op_env.n_sub, len(self.obs_node_attr)), dtype=np.float32)

        # Assign generator power to nodes because of gen_to_subid and load_to_subid being case to case
        for gen_id, sub_id in enumerate(self._g2op_env.gen_to_subid):
            node_features[sub_id, 0] += parsed_obs["gen_p"][gen_id]  # Assign gen_p

        # Assign load power to nodes
        for load_id, sub_id in enumerate(self._g2op_env.load_to_subid):
            node_features[sub_id, 1] += parsed_obs["load_p"][load_id]  # Assign load_p

        return node_features
    

    def _construct_edge_features(self, parsed_obs):
        """Construct edge feature matrix (n_edges × n_edge_attr) programmatically."""
        edge_features = np.column_stack([parsed_obs[attr] for attr in self.obs_edge_attr])
        return edge_features.astype(np.float32)
        
    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        # print("WARNING: setup_actions is not doing anything. Implement your own code in this method.")

        # act_attr_to_keep = ["set_bus", "set_line_status", 'curtail', 'redispatch']
        act_attr_to_keep = ["change_bus", "change_line_status", 'curtail', 'redispatch']

        self._gym_env.action_space = self._gym_env.action_space.keep_only_attr(act_attr_to_keep)
        
    def setup_base_actions(self):
        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space)
        # self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)
    
    def setup_improved_actions(self):
        act_attr_to_keep = ["change_bus", "change_line_status", 'curtail', 'redispatch']

        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space, act_attr_to_keep)


    def reset(self, seed=None, options=None):
        obs, info = self._gym_env.reset(seed=seed)
        if self.use_gnn:
            obs = self._convert_observation(obs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self._gym_env.step(action)
        if self.use_gnn:
            obs = self._convert_observation(obs)
        return obs, reward, done, truncated, info

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


def main():
    # Random agent interacting in environment 

    agent = 'random'
    train = None
    use_gnn = False

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
            model.set_parameters('models/base_agent')
        else:
            model = PPO('MultiInputPolicy', env=env, verbose=1, tensorboard_log="runs/improved_agent")
            model_name = 'improved_agent'
            model.set_parameters('models/improved_agent')

        if train == 'base' or train == 'improved':
            wandb.init(
                    project="RL Assignment",
                    config=config,
                    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                    save_code=True,  # optional
                    name=model_name
            )

            wandb_callback = WandbCallback(
                gradient_save_freq=10,
                verbose=1,
            )

            model.learn(100000,callback=wandb_callback, progress_bar=True)
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

        if use_gnn:
            ppo_change_bus.set_parameters('models/multi-agent models/change_bus_gnn_agent')
            ppo_change_line_status.set_parameters('models/multi-agent models/change_line_status_gnn_agent')
            ppo_curtail.set_parameters('models/multi-agent models/curtail_gnn_agent')
            ppo_redispatch.set_parameters('models/multi-agent models/redispatch_gnn_agent')
        else:
            ppo_change_bus.set_parameters('models/multi-agent models/change_bus_agent')
            ppo_change_line_status.set_parameters('models/multi-agent models/change_line_status_agent')
            ppo_curtail.set_parameters('models/multi-agent models/curtail_agent')
            ppo_redispatch.set_parameters('models/multi-agent models/redispatch_agent')


        ppo_agents = [ppo_change_bus, ppo_change_line_status, ppo_curtail, ppo_redispatch, ]
        multi_agent_env = MultiAgentWrapper(env, action_names) 



        if train == 'multi-agent':

            # Train each agent separately

            wandb.init(
                    project="RL Assignment",
                    config=config,
                    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                    save_code=True,  # optional
                    name="redispatch_agent"
            )
            wandb_callback_bus = WandbCallback(
                gradient_save_freq=10,
                # model_save_freq=5000,
                # model_save_path=f"models/{'multi-agent models/change_bus_agent'}",
                verbose=2,
            )
            ppo_change_bus.learn(total_timesteps=100000, progress_bar=True, )
            if use_gnn:
                ppo_change_bus.save('models/multi-agent models/change_bus_gnn_agent')
            else:
                ppo_change_bus.save('models/multi-agent models/change_bus_agent')
            


            wandb_callback_line = WandbCallback(
                gradient_save_freq=10,
                # model_save_freq=5000,
                # model_save_path=f"models/{'multi-agent models/change_line_status_agent'}",
                verbose=2,
            )
            ppo_change_line_status.learn(total_timesteps=100000, progress_bar=True, )
            if use_gnn:
                ppo_change_line_status.save('models/multi-agent models/change_line_status_gnn_agent')
            else:
                ppo_change_line_status.save('models/multi-agent models/change_line_status_agent')
            


            wandb_callback_curtail = WandbCallback(
                gradient_save_freq=10,
                # model_save_freq=5000,
                # model_save_path=f"models/{'multi-agent models/curtail_agent'}",
                verbose=2,
            )
            ppo_curtail.learn(total_timesteps=100000, progress_bar=True, callback=wandb_callback_curtail)
            if use_gnn:
                ppo_curtail.save('models/multi-agent models/curtail_gnn_agent')
            else:
                ppo_curtail.save('models/multi-agent models/curtail_agent')
            

            wandb_callback_redispatch = WandbCallback(
                gradient_save_freq=10,
                # model_save_freq=5000,
                # model_save_path=f"models/{'multi-agent models/redispatch_agent'}",
                verbose=2,
            )
            ppo_redispatch.learn(total_timesteps=100000, progress_bar=True, callback=wandb_callback_redispatch)
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


        if train == 'hierarchy':
            name = f'models/{agent}'
            name += f'_agent' if not use_gnn else f'_gnn_agent'
            wandb.init(
                    project="RL Assignment",
                    config=config,
                    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                    save_code=True,  # optional
                    name=name
            )
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

            # print(f"step = {curr_step}: ")
            # print(f"\t obs = {obs}")
            # print(f"\t reward = {reward}")
            # print(f"\t terminated = {terminated}")
            # print(f"\t truncated = {truncated}")
            # print(f"\t info = {info}")

            # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
            # Invalid actions are replaced with 'do nothing' action
            # is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
            # print(f"\t is action valid = {is_action_valid}")
            # if not is_action_valid:
            #     print(f"\t\t reason = {info['exception']}")
            # print("\n")

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
