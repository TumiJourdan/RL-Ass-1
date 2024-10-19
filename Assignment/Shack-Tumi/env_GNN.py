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
from grid2op.gym_compat import BoxGymObsSpace,MultiDiscreteActSpace
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
# GNN
from GNN_Extractor import CustomGNN
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

        self.setup_observations()
        self.setup_actions()

        # self.observation_space = self._gym_env.observation_space
        # self.action_space = self._gym_env.action_space

    def setup_observations(self):
            obs_attr_to_keep = ["rho", "p_or", "gen_p", "load_p"]
            self._gym_env.observation_space.close()
            self._gym_env.observation_space = 
            
            # Get the graph structure
            obs = self._g2op_env.get_obs()
            
            print("Rho: {}".format(obs.rho.shape))
            print("p_or: {}".format(obs.p_or.shape))
            print("gen_p: {}".format(obs.gen_p.shape))
            print("load_p: {}".format(obs.load_p.shape))

            graph = obs.get_energy_graph()

            self.edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
            self.observation_space = Dict({
            "features": Box(
                low=self._gym_env.observation_space.low,
                high=self._gym_env.observation_space.high,
                dtype=np.float32
            ),
            "edge_index": Box(
                low=0,
                high=max(self.edge_index.max().item(), len(obs_attr_to_keep)),
                shape=self.edge_index.shape,
                dtype=np.int64
            )
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

    def reset(self, seed=None):
        obs, info = self._gym_env.reset(seed=seed)
        processed_obs = self._process_observation(obs)
        print(f"Reset observation: {processed_obs}",flush=True)  # Add this for debugging
        return processed_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._gym_env.step(action)
        processed_obs = self._process_observation(obs)
        print(f"Step observation: {processed_obs}",flush=True)  # Add this for debugging
        return processed_obs, reward, terminated, truncated, info
    
    def _process_observation(self, obs):
        # Example for extracting node-specific attributes
        # Assuming `obs` contains structured data for nodes
        node_data = []

        for node_id in range(number_of_nodes):
            node_features = {
                "rho": obs["rho"][node_id],
                "p_or": obs["p_or"][node_id],
                "gen_p": obs["gen_p"][node_id],
                "load_p": obs["load_p"][node_id]
            }
            node_data.append(node_features)

        return {
            "node_features": node_data,
            "edge_index": self.edge_index
        }

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()

def main():
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 100000,
        "env_name": "l2rpn_case14_sandbox",
    }
    run = wandb.init(
        project="Grid20p",
        config=config,
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

    policy_kwargs = dict(
        features_extractor_class=CustomGNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = A2C(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )

    model.learn(total_timesteps=100000, callback=WandbCallback())
    run.finish()

if __name__ == "__main__":
    main()