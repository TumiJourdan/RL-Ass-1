import random
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *


def epsilon_greedy(num_actions: int, best_action: int, epsilon: float):
    if np.random.random() <= epsilon:
        return np.random.randint(num_actions)

    return best_action


def make_env(env_name: str, seed: int, k: int, mode: str = "rgb_array"):
    assert "NoFrameskip" in env_name, "Require environment with no frameskip"
    env = gym.make(env_name, render_mode=mode)
    env.seed(seed)

    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env, k)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, k)
    return env


def plot_curves(episode_rewards: list, episode_lengths: list, losses: list):
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)

    average_rewards = episode_rewards / episode_lengths

    x_values = np.arange(len(average_rewards))
    plt.plot(x_values, average_rewards)
    plt.title("Average reward per episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.savefig("rewards.png")

    x_values = np.arange(len(losses))
    plt.plot(x_values, losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig("loss.png")


def show_policy(env: gym.Env, agent: DQNAgent):
    done = False
    state, info = env.reset()

    while not done:
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state


def train_agent(hyper_params: dict) -> tuple:
    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    env = make_env(hyper_params["env"], hyper_params["seed"], hyper_params["num_frames_per_state"])

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        hyper_params["learning-rate"],
        hyper_params["batch-size"],
        hyper_params["discount-factor"],
    )

    agent.load_models()

    episode_rewards = [0.0]
    episode_lengths = [0]
    losses = []
    state, info = env.reset()

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])

    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (hyper_params["eps-end"] - hyper_params["eps-start"])

        action = epsilon_greedy(env.action_space.n, agent.act(state), eps_threshold)

        next_state, reward, done, truncated, info = env.step(action)

        replay_buffer.add(state, action, reward, next_state, done)

        episode_rewards[-1] += reward
        episode_lengths[-1] += 1

        state = next_state

        if done:
            state, info = env.reset()
            episode_rewards.append(0.0)
            episode_lengths.append(0)

        if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
            loss = agent.optimise_td_loss()
            losses.append(loss)

        if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
            agent.save_models()
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params["print-freq"] == 0:
            with open(hyper_params["stat_file"], "a") as file:
                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                file.write("********************************************************\n")
                file.write("steps: {}\n".format(t))
                file.write("episodes: {}\n".format(num_episodes))
                file.write("mean 100 episode reward: {}\n".format(mean_100ep_reward))
                file.write("% time spent exploring: {}\n".format(int(100 * eps_threshold)))
                file.write("********************************************************\n\n")

    return agent, episode_rewards, losses


def main():
    hyper_params = {
        "seed": 12,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5000),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
        "stat_file": "stats.txt",
        "num_frames_per_state": 4,
    }

    agent, episode_rewards, losses = train_agent(hyper_params)

    display_env = make_env(hyper_params["env"], hyper_params["seed"], hyper_params["num_frames_per_state"], "human")

    # plot_curves(episode_rewards, losses)
    show_policy(display_env, agent)


if __name__ == "__main__":
    main()
