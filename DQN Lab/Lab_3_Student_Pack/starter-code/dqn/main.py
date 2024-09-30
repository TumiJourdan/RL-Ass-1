import numpy as np
import torch
from model import DQN
import gym
from wrappers import *

env = gym.make("PongNoFrameskip-v4")
env = MaxAndSkipEnv(env, skip=4)
env = EpisodicLifeEnv(env)
env = FireResetEnv(env)
env = WarpFrame(env)
env = PyTorchFrame(env)
env = FrameStack(env, k=4)

print("Observation space", env.observation_space.shape)
print("Action space", env.action_space.n)

model = DQN(env.observation_space, env.action_space)
image = env.reset()
image = torch.from_numpy(image).float()
print(image.shape)
out = model(image)
print(out)
