import gymnasium as gym

import torch

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from lightsim2grid import LightSimBackend

# personal imports
from stable_baselines3 import A2C
from grid2op.gym_compat import BoxGymObsSpace,MultiDiscreteActSpace
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
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
        obs_attr_to_keep = [
            "rho", "p_or", "a_ex", "v_or", "v_ex",
            "line_status", "topo_vect",
            "time_before_cooldown_line", "time_before_cooldown_sub",
            "gen_p", "gen_q", "target_dispatch", "actual_dispatch",
            "load_p", "load_q",
            "day", "hour_of_day"
        ]
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space,
                                                        attr_to_keep=obs_attr_to_keep
                                                        )
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                    low=self._gym_env.observation_space.low,
                                    high=self._gym_env.observation_space.high)
        

    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        # user wants a multi-discrete action space
        act_attr_to_keep = ["one_line_set", "one_sub_set", "set_bus", "redispatch", "curtail"]
        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space,
                                                            attr_to_keep=act_attr_to_keep)
        self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.losses = []
        self.value_losses = []
        self.rewards = []
        self.total_rewards = []

    def _on_step(self) -> bool:
        
        wandb.log({"train_loss": logs.get('loss', 0), "val_loss":logs.get('value_loss', 0),"reward": reward})
        return True

    def _on_training_end(self):
        print(f"Average Loss: {np.mean(self.losses)}")
        print(f"Average Value Loss: {np.mean(self.value_losses)}")
        print(f"Total Rewards: {np.sum(self.total_rewards)}")
def main():

    
    wandb.init(project="RL", name="Setup")


    # Random agent interacting in environment #
    
    max_steps = 100

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


    model = A2C("MlpPolicy", env, verbose=1)
    logging_callback = CustomLoggingCallback()

    #model = A2C.load("A2C", env=env)
    model.learn(total_timesteps=20000, callback=logging_callback)
    model.save("A2C_improved")


    # curr_step = 0
    # curr_return = 0

    # is_done = False
    # obs, info = env.reset()
    # print(f"step = {curr_step} (reset):")
    # print(f"\t obs = {obs}")
    # print(f"\t info = {info}\n\n")

    # while not is_done and curr_step < max_steps:
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)

    #     curr_step += 1
    #     curr_return += reward
    #     is_done = terminated or truncated

    #     print(f"step = {curr_step}: ")
    #     print(f"\t obs = {obs}")
    #     print(f"\t reward = {reward}")
    #     print(f"\t terminated = {terminated}")
    #     print(f"\t truncated = {truncated}")
    #     print(f"\t info = {info}")

    #     # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
    #     # Invalid actions are replaced with 'do nothing' action
    #     is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
    #     print(f"\t is action valid = {is_action_valid}")
    #     if not is_action_valid:
    #         print(f"\t\t reason = {info['exception']}")
    #     print("\n")

    # print("###########")
    # print("# SUMMARY #")
    # print("###########")
    # print(f"return = {curr_return}")
    # print(f"total steps = {curr_step}")
    # print("###########")


if __name__ == "__main__":
    main()
