###
# Group Members
# Shakeel Malagas: 2424161 
# Tumi Jourdan: 2180153
# Dean Solomon: 2347848
# Tao Yuan: 2332155
###

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def eps_greedy(Q, state, eps):
    if np.random.random() < eps:
        return np.random.choice(4)
    else:
        return np.argmax(Q[state])


def sarsa(num_episodes, env: gym.Env, gamma, alpha, decay_rate, eps, save_folder=None):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    return_array = np.array([])

    for episode in range(num_episodes):
        e = np.zeros((env.observation_space.n, env.action_space.n))
        state, _ = env.reset()
        action = eps_greedy(Q, state, eps)
        done = False
        total_reward = 0

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = eps_greedy(Q, next_state, eps)

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            e[state, action] += 1

            for s in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    Q[s, a] += alpha * delta * e[s, a]
                    e[s, a] *= gamma * decay_rate

            state = next_state
            action = next_action
            total_reward += reward
            
        if save_folder is not None:
            q_mean = np.mean(Q,axis=1)
            plt.imshow(np.reshape(q_mean,[4,12]))
            plt.colorbar() 
            plt.title( "2-D Heat Map" ) 
            plt.savefig(f"./{save_folder}/episode{episode}.png") 
            plt.close()
            
        return_array = np.append(return_array, total_reward)
        
    return return_array

env = gym.make('CliffWalking-v0',) #render_mode = 'human'

gamma = 0.99
alpha = 0.1
eps = 0.1
decay_rate = 0

# rewards = sarsa(200,env,gamma,alpha,decay_rate,eps, 'td0')
# decay_rate = 0.3
# rewards = sarsa(200,env,gamma,alpha,decay_rate,eps, 'td1')
# decay_rate = 0.5
# rewards = sarsa(200,env,gamma,alpha,decay_rate,eps, 'td2')

td0_return = np.array([])
td1_return = np.array([])
td2_return = np.array([])
runs = 4
episodes = 200

for i in range(runs):
    print(f"run {i}")
    decay_rate = 0
    td0_return = np.append(td0_return, sarsa(episodes,env,gamma,alpha,decay_rate,eps)) 
    
    decay_rate = 0.3
    td1_return = np.append(td1_return, sarsa(episodes,env,gamma,alpha,decay_rate,eps))
    
    decay_rate = 0.5
    td2_return = np.append(td2_return, sarsa(episodes,env,gamma,alpha,decay_rate,eps))

td0_return = np.reshape(td0_return, [runs,episodes])
td1_return = np.reshape(td1_return, [runs,episodes])
td2_return = np.reshape(td2_return, [runs,episodes])

td0_return_mean = np.mean(td0_return, axis=0)
td1_return_mean = np.mean(td1_return, axis=0)
td2_return_mean = np.mean(td2_return, axis=0)

td0_return_var = np.std(td0_return,axis=0)
td1_return_var = np.std(td1_return,axis=0)
td2_return_var = np.std(td2_return,axis=0)



t=[i for i in range(episodes)]

plt.plot(t, td0_return_mean, label='Average Return ofr lamba=0', linestyle='-')
plt.fill_between(t, td0_return_mean - td0_return_var, td0_return_mean + td0_return_var, alpha=0.2)
plt.plot(t, td1_return_mean, label='Average Return ofr lamba=0.3', linestyle='-.')
plt.fill_between(t, td1_return_mean - td1_return_var, td1_return_mean + td1_return_var, alpha=0.2)
plt.plot(t, td2_return_mean, label='Average Return ofr lamba=0.5')
plt.fill_between(t, td2_return_mean - td2_return_var, td2_return_mean + td2_return_var, alpha=0.2)
plt.legend()
plt.show()