import random
import numpy as np
import gym
import torch
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from tqdm import tqdm


if __name__ == "__main__":

    hyper_params = {
    "seed": 42,  # which seed to use
    "env": "PongNoFrameskip-v4",  # name of the game
    "replay-buffer-size": int(5e3),  # replay buffer size
    "learning-rate": 1e-4,  # learning rate for Adam optimizer
    "discount-factor": 0.99,  # discount factor
    "num-steps": int(1e6),  # total number of steps to run the environment for
    "batch-size": 256,  # number of transitions to optimize at the same time
    "learning-starts": 10000,  # number of steps before learning starts
    "learning-freq": 5,  # number of iterations between every optimization step
    "use-double-dqn": False,  # use double deep Q-learning
    "target-update-freq": 1000,  # number of iterations between every target network update
    "eps-start": 1.0,  # e-greedy start threshold
    "eps-end": 0.01,  # e-greedy end threshold
    "eps-fraction": 0.1,  # fraction of num-steps
    "print-freq": 10,
}

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"],render_mode = "human")
    env.seed(hyper_params["seed"])

    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env, 4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    # TODO Create dqn agent
    # agent = DQNAgent( ... )
    agent = DQNAgent(env.observation_space,
                    env.action_space,replay_buffer,
                    hyper_params["use-double-dqn"],
                    hyper_params["learning-rate"],
                    hyper_params["batch-size"],
                    hyper_params["discount-factor"])


    agent.load_models()

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    agent.policy_network.eval()
    state = env.reset()
    for t in tqdm(range(hyper_params["num-steps"])):

        action = agent.act(state)
            
        next_state, reward, done, _ = env.step(action)
        
        episode_rewards[-1] += reward
        replay_buffer.add(state,action,reward,next_state,done)
        state = next_state
        
        if done:
            state = env.reset()
            episode_rewards.append(0.0)
