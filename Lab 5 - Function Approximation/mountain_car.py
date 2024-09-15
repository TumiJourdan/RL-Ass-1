import numpy
import gym
import value_function as vf
import matplotlib.pyplot as plt

NUM_EPISODES = 50
ALPHA = 0.1
EPSILON = 0.1
GAMMA = 0.99

def main():
    env = gym.make('MountainCar-v0')
    val_func = vf.ValueFunction(0.001, env.action_space.n)

    Returns = []
    Steps = []

    for e in range(NUM_EPISODES):
        state, _ = env.reset()
        action = val_func.act(state, EPSILON)    
        done = None
        steps = 0
        total_reward = 0

        while not done:
            state_, reward, done, _, _ = env.step(action)
            steps += 1
            total_reward += reward

            if done:
                val_func.update(reward,state,action)
                break

            action_ = val_func.act(state_, EPSILON)

            val_func.update(target=reward + GAMMA*val_func(state_,action_),
                            state=state,
                            action=action)
            
            state = state_
            action = action_

        Returns.append(total_reward)
        Steps.append(steps)
    
    plt.plot([i for i in range(len(Returns))], Returns, label='Returns')
    plt.plot([i for i in range(len(Steps))], Steps, label='Steps')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
