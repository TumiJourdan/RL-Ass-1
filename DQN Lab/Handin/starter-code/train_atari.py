# Group Members
# Shakeel Malagas: 2424161 
# Tumi Jourdan: 2180153
# Dean Solomon: 2347848
# Tao Yuan: 2332155


import random
import numpy as np
import gym
import torch
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from tqdm import tqdm
import wandb


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
    wandb.init(project="DQNET", name="Attempt 1")

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])
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

    state = env.reset()
    for t in tqdm(range(hyper_params["num-steps"])):
        # looks like a linear decay
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()
        # TODO
        #  select random action if sample is less equal than eps_threshold
        # take step in env
        # add state, action, reward, next_state, float(done) to reply memory - cast done to float
        # add reward to episode_reward
        if(sample <= eps_threshold):
            action = env.action_space.sample()
        else:
            action = agent.act(state)
            
        next_state, reward, done, _ = env.step(action)
        
        episode_rewards[-1] += reward
        replay_buffer.add(state,action,reward,next_state,done)
        state = next_state
        
        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            loss = agent.optimise_td_loss()
            wandb.log({"loss": loss, "step": t})

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.save_models()
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            wandb.log({
                "reward_100_eps": mean_100ep_reward,
            }, step=t)