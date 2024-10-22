import gymnasium as gym
from gymnasium.spaces import MultiBinary
from gymnasium.spaces import Dict as GymDict

import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
from typing import Dict, List, Tuple

import grid2op
from grid2op import gym_compat
from grid2op.gym_compat import ContinuousToDiscreteConverter, MultiDiscreteActSpace, BoxGymObsSpace, ScalerAttrConverter
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*utcnow.*")



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


        
    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        # print("WARNING: setup_actions is not doing anything. Implement your own code in this method.")

        # act_attr_to_keep = ["set_bus", "set_line_status", 'curtail', 'redispatch']
        act_attr_to_keep = ["change_bus", "change_line_status", 'curtail', 'redispatch']

        self._gym_env.action_space = self._gym_env.action_space.keep_only_attr(act_attr_to_keep)
        
        # self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


def main():
    # Random agent interacting in environment 

    is_multi_agent = False
    is_hierarchy_agent = True

    max_steps = 10000

    env = Gym2OpEnv()

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
    
    # model = PPO("MultiInputPolicy",env, verbose=1, n_steps=2048)


    # # print(f"step = {curr_step} (reset):")
    # # print(f"\t obs = {obs}")
    # # print(f"\t info = {info}\n\n")

    # model.learn(total_timesteps=2048*5, progress_bar=True, log_interval=10, reset_num_timesteps=False)

    # Create a separate environment for each agent
    env_change_bus_egent = SingleAgentWrapper(env,"change_bus")
    env_change_line_status_agent = SingleAgentWrapper(env, "change_line_status")
    env_curtail_agent = SingleAgentWrapper(env, "curtail")
    env_redispatch_agent = SingleAgentWrapper(env, "redispatch")

    # Vectorize environments for each agent
    # vec_env_change_bus_egent = DummyVecEnv([lambda: env_change_bus_egent])
    # vec_env_change_line_status_agent = DummyVecEnv([lambda: env_change_line_status_agent])
    # vec_env_curtail_agent = DummyVecEnv([lambda: env_curtail_agent])
    # vec_env_redispatch_agent = DummyVecEnv([lambda: env_redispatch_agent])

    # Create PPO models for each agent
    ppo_change_bus = PPO(policy='MultiInputPolicy', env=env_change_bus_egent, verbose=1, tensorboard_log="runs/change_bus",)
    ppo_change_line_status = PPO(policy='MultiInputPolicy', env=env_change_line_status_agent, verbose=1, tensorboard_log="runs/change_line",)
    ppo_curtail = PPO(policy='MultiInputPolicy', env=env_curtail_agent, verbose=1, tensorboard_log="runs/curtail",)
    ppo_redispatch = PPO(policy='MultiInputPolicy', env=env_redispatch_agent, verbose=1, tensorboard_log="runs/redispatch",)


    # Train each agent separately

    # wandb.init(
    #         project="RL Assignment",
    #         config=config,
    #         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #         save_code=True,  # optional
    # )
    # wandb_callback_bus = WandbCallback(
    #     gradient_save_freq=10,
    #     # model_save_freq=5000,
    #     # model_save_path=f"models/{'multi-agent models/change_bus_agent'}",
    #     verbose=2,
    # )
    # ppo_change_bus.learn(total_timesteps=150000, progress_bar=True, callback=wandb_callback_bus)
    # ppo_change_bus.save('multi-agent models/change_bus_agent')

    # wandb.init(
    #         project="RL Assignment",
    #         config=config,
    #         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #         save_code=True,  # optional
    # )
    # wandb_callback_line = WandbCallback(
    #     gradient_save_freq=10,
    #     # model_save_freq=5000,
    #     # model_save_path=f"models/{'multi-agent models/change_line_status_agent'}",
    #     verbose=2,
    # )
    # ppo_change_line_status.learn(total_timesteps=150000, progress_bar=True, callback=wandb_callback_line)
    # ppo_change_line_status.save('multi-agent models/change_line_status_agent')

    # wandb.init(
    #         project="RL Assignment",
    #         config=config,
    #         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #         save_code=True,  # optional
    # )
    # wandb_callback_curtail = WandbCallback(
    #     gradient_save_freq=10,
    #     # model_save_freq=5000,
    #     # model_save_path=f"models/{'multi-agent models/curtail_agent'}",
    #     verbose=2,
    # )
    # ppo_curtail.learn(total_timesteps=200000, progress_bar=True, callback=wandb_callback_curtail)
    # ppo_curtail.save('multi-agent models/curtail_agent')

    # wandb.init(
    #         project="RL Assignment",
    #         config=config,
    #         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #         save_code=True,  # optional
    # )
    # wandb_callback_redispatch = WandbCallback(
    #     gradient_save_freq=10,
    #     # model_save_freq=5000,
    #     # model_save_path=f"models/{'multi-agent models/redispatch_agent'}",
    #     verbose=2,
    # )
    # ppo_redispatch.learn(total_timesteps=150000, progress_bar=True, callback=wandb_callback_redispatch)
    # ppo_redispatch.save('multi-agent models/redispatch_agent')

    ppo_change_bus.set_parameters('multi-agent models/change_bus_agent')
    ppo_change_line_status.set_parameters('multi-agent models/change_line_status_agent')
    ppo_curtail.set_parameters('multi-agent models/curtail_agent')
    ppo_redispatch.set_parameters('multi-agent models/redispatch_agent')

    ppo_agents = [ppo_change_bus, ppo_change_line_status, ppo_curtail, ppo_redispatch, ]

    episodes = 10
    avg_return = 0
    avg_step = 0

    action_names = ['change_bus', 'change_line_status', 'curtail', 'redispatch']

    multi_agent_env = MultiAgentWrapper(env, action_names) 

    hierarchy_env = HierarchyWrapper(env,action_names,ppo_agents)

    hierarchy_agent = PPO('MultiInputPolicy', env=hierarchy_env, tensorboard_log="runs/hierarchy_agent", verbose=1)

    hierarchy_agent.learn(total_timesteps=1000000, progress_bar=True,)
    hierarchy_agent.save('hierarchy_agent')

    for e in range(episodes):
        curr_step = 0
        curr_return = 0

        is_done = False
        if is_multi_agent:
            obs, info = multi_agent_env.reset()
        elif is_hierarchy_agent:
            obs, info = hierarchy_env.reset()
        else:
            obs, info = env.reset() 

        while not is_done and curr_step < max_steps:
            # action = env.action_space.sample()
            if is_multi_agent:
                actions = []

                actions.append(ppo_change_bus.predict(obs)[0])
                actions.append(ppo_change_line_status.predict(obs)[0])
                actions.append(ppo_curtail.predict(obs)[0])
                actions.append(ppo_redispatch.predict(obs)[0])
                obs, reward, terminated, truncated, info = multi_agent_env.step(actions)
            elif is_hierarchy_agent:
                
                actions = []

                actions.append(hierarchy_agent.predict(obs)[0])
                print(actions[0])
                actions.append(ppo_change_bus.predict(obs)[0])
                actions.append(ppo_change_line_status.predict(obs)[0])
                actions.append(ppo_curtail.predict(obs)[0])
                actions.append(ppo_redispatch.predict(obs)[0])
                obs, reward, terminated, truncated, info = hierarchy_env.step(actions)

            else:
                # action = model.predict(obs)
                # obs, reward, terminated, truncated, info = env.step(action)
                pass
                

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
