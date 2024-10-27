import gymnasium as gym
import torch
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from lightsim2grid import LightSimBackend
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
# personal imports
from stable_baselines3 import A2C
from grid2op.gym_compat import BoxGymObsSpace,MultiDiscreteActSpace,GymObservationSpace
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
# GNN
from GNN_Extractor import CustomGNN
import signal
import sys
# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(
            self
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
        
        self.observation_space = self._gym_env.observation_space
        self.obs_node_attr =[]
        self.obs_edge_attr =[]
        self.setup_observations()
        self.setup_actions()



    def setup_observations(self):
        """Define the observation space with node and edge features."""
        # TODO: Your code to specify & modify the observation space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        obs_attr_to_keep = ["rho", "p_or", "gen_p", "load_p"]
        self.obs_node_attr = ["gen_p","load_p"]
        self.obs_edge_attr = ["rho","p_or"]

        self._gym_env.observation_space.close()

        # this is important as it is what is returned in our step and our reset functions (to us , we then can change it and have the function return somthing else)
        # we have to use their template code here as we cant control what the step function returns, we can only limit it
        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space,
                                                         attr_to_keep=obs_attr_to_keep
                                                         )

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

    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        # user wants a multi-discrete action space
        act_attr_to_keep = ["one_line_set", "one_sub_set"]
        
        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space,
                                                            attr_to_keep=act_attr_to_keep)
        self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)

    def reset(self, seed=None, options=None):
        obs, info = self._gym_env.reset(seed=seed)
        parsed_obs = self._convert_observation(obs)
        return parsed_obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self._gym_env.step(action)
        parsed_obs = self._convert_observation(obs)
        return parsed_obs, reward, done, truncated, info
    

    

    def _convert_observation(self, obs):
        """Convert flattened Box observations into tensors for nodes and edges."""
        # Parse the observation dictionary into separate arrays keyed by attribute name
        parsed_obs = self._parse_observation(obs)
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
    
    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()

def main():
    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 500000,
        "env_name": "l2rpn_case14_sandbox",
    }
    run = wandb.init(
        project="Grid20p",
        config=config,
        name="custom_run_name",       # Set the run name here
        sync_tensorboard=True
    )

    print(torch.cuda.is_available())

    env = Gym2OpEnv()
    env = Monitor(env)

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

    def rmsprop_with_momentum(params, **kwargs):
        # Use kwargs to pass in the learning rate and other optimizer settings
        return torch.optim.RMSprop(params, momentum=0.9, alpha=0.99, eps=1e-5, **kwargs)
    
    policy_kwargs = dict(
        features_extractor_class=CustomGNN,
        features_extractor_kwargs=dict(features_dim=1120),
        optimizer_class=rmsprop_with_momentum,  # Add custom optimizer
    )

    model = A2C(
        config["policy_type"],
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )


    def save_and_exit(signum, frame):
        """Save the model and finish the run when interrupted."""
        print("\nInterrupted! Saving model and exiting...")
        model.save("A2C_GNN_NORMALIZE")
        run.finish()
        sys.exit(0)

    # Register the signal handler for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    try:
        model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback())
    finally:
        # Ensure the model is saved at the end, even if interrupted
        model.save("A2C_GNN_NORMALIZE")
        run.finish()

if __name__ == "__main__":
    main()