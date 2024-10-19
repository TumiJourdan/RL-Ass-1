import gymnasium as gym
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from lightsim2grid import LightSimBackend
# personal imports
from stable_baselines3 import A2C
from grid2op.gym_compat import BoxGymObsSpace, MultiDiscreteActSpace, ScalerAttrConverter
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb


class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_stages = [
            ["one_line_set"],
            ["one_line_set", "one_sub_set"],
            ["one_line_set", "one_sub_set", "set_bus"],
            ["one_line_set", "one_sub_set", "set_bus", "set_line_status"],
            ["one_line_set", "one_sub_set", "set_bus", "set_line_status", "sub_set_bus"],
            ["one_line_set", "one_sub_set", "set_bus", "set_line_status", "sub_set_bus", "redispatch"]
        ]
        self.stage_timesteps = 1000  # Steps per stage
        self.current_stage = 0

    def _on_step(self) -> bool:
        # Check if we need to update the action space
        if (self.num_timesteps % self.stage_timesteps == 0 and
                self.current_stage < len(self.action_stages) - 1 and
                self.num_timesteps < 60000):
            self.current_stage = self.num_timesteps // self.stage_timesteps

            # Access the actual environment(s) inside DummyVecEnv
            for env in self.training_env.envs:
                env.unwrapped.update_action_space(self.action_stages[self.current_stage])

            # Reset the policy network's action space
            self.model.policy.action_space = self.training_env.action_space

        return True


class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.low = None
        self.high = None

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward

        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        self._gym_env = gym_compat.GymEnv(self._g2op_env)
        self.setup_observations()

        # Start with only the first action
        self.update_action_space(["one_line_set"])

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def update_action_space(self, available_actions):
        self._gym_env.action_space = MultiDiscreteActSpace(
            self._g2op_env.action_space,
            attr_to_keep=available_actions
        )
        self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)

    def setup_observations(self):
        # obs_attr_to_keep = ["rho", "p_or", "gen_p", "load_p"]
        # self._gym_env.observation_space.close()
        # self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space,
        #                                                  attr_to_keep=obs_attr_to_keep
        #                                                  )
        # # export observation space for the Grid2opEnv
        # self.observation_space = Box(shape=self._gym_env.observation_space.shape,
        #                              low=self._gym_env.observation_space.low,
        #                              high=self._gym_env.observation_space.high)
        obs_attr_to_keep = [
            "rho",
            "line_status",
            "topo_vect",
            "gen_p",
            "load_p",
            "p_or",
            "p_ex"
        ]

        obs_space = self._gym_env.observation_space
        converters = {}

        # Normalize generator power production (gen_p)
        converters["gen_p"] = ScalerAttrConverter(
            substract=0.0,
            divide=self._g2op_env.gen_pmax
        )

        load_p_max = obs_space["load_p"].high
        converters["load_p"] = ScalerAttrConverter(
            substract=0.0,
            divide=load_p_max
        )

        p_or_max = obs_space["p_or"].high
        converters["p_or"] = ScalerAttrConverter(
            substract=0.0,
            divide=p_or_max
        )

        p_ex_max = obs_space["p_ex"].high
        converters["p_ex"] = ScalerAttrConverter(
            substract=0.0,
            divide=p_ex_max
        )

        self._gym_env.observation_space = self._gym_env.observation_space.keep_only_attr(obs_attr_to_keep)

        # Apply converters
        for attr, converter in converters.items():
            self._gym_env.observation_space = self._gym_env.observation_space.reencode_space(attr, converter)



    def setup_actions(self):
        act_attr_to_keep = ["one_line_set", "one_sub_set", "set_bus", "set_line_status", "sub_set_bus", "redispatch"]#
        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space,
                                                           attr_to_keep=act_attr_to_keep)
        self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)

    def step(self, action):
        # Pad the action if necessary to match the full action space
        full_action = np.zeros(self._g2op_env.action_space.n, dtype=int)
        full_action[:len(action)] = action

        obs, reward, done, truncated, info = self._gym_env.step(full_action)

        return obs, reward, done, truncated, info
    # def step(self, action):
    #     obs, reward, done, truncated, info = self._gym_env.step(action)
    #     return obs, reward, done, truncated, info

    def reset(self, seed=None):
        obs, info = self._gym_env.reset(seed=seed, options=None)
        # Transform observations after resetting

        return obs, info
    # def reset(self, seed=None):
    #     return self._gym_env.reset(seed=seed)

    def render(self):
        return self._gym_env.render()


def main():
    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 500000,  # Increase for better results
        "env_name": "l2rpn_case14_sandbox",
    }

    run = wandb.init(
        project="Grid20p",
        config=config,
        sync_tensorboard=True
    )

    # Random agent interacting in environment #



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

    model = A2C("MultiInputPolicy", env=env, verbose=1,learning_rate=1e-4,
                tensorboard_log=f"runs/{run.id}",
                max_grad_norm=0.5)

    model = A2C.load("A2C_curriculum_trained 200k.zip", env=env, device="auto", print_system_info=True)
    callback = CurriculumCallback()


    model.learn(
        total_timesteps=500000,
        callback=[WandbCallback(), callback]#
    )
    model.save("A2C_curriculum_trained")


if __name__ == "__main__":
    main()