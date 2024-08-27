import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
def one_step_lookahead(state, V,env):
    """
    Helper function to calculate the value for all action in a given state.

    Args:
        state: The state to consider (int)
        V: The value to use as an estimator, Vector of length env.observation_space.n

    Returns:
        A vector of length env.action_space.n containing the expected value of each action.
    """
    
    # looking at state, we need to look at actions around said state, careful boundaries
    MAX_Y = env.shape[0]
    MAX_X = env.shape[1]
    
    s = state
    
    y = state // MAX_X
    x = state % MAX_X
    
    ns_up = s if y == 0 else s - MAX_X
    ns_right = s if x == (MAX_X - 1) else s + 1
    ns_down = s if y == (MAX_Y - 1) else s + MAX_X
    ns_left = s if x == 0 else s - 1
    # up right down left
    return [V[ns_up],V[ns_right],V[ns_down],V[ns_left]]

def eps_greedy(V, state, eps,env):
    
    action_returns = one_step_lookahead(state,V,env)
    
    if np.random.random() < eps:
        return np.random.choice(4)
    else:
        best_move_index = np.argmax(action_returns)
        return best_move_index



def sarsa(num_episodes, env:gym.Env, gamma, alpha,decay_rate, eps, save_folder=None):
    return_array= np.array([])
    
    V = np.zeros(env.observation_space.n)

    
    for episode in range(num_episodes):
        state, _ = env.reset()  # Extract the integer state from the tuple
        done = False
        total_reward = 0
        visited_states = [state]
        e = np.zeros(env.observation_space.n)
        while not done:
            action = eps_greedy(V, state, eps,env)
            next_state, reward, done, _, _ = env.step(action)
            
            ### RENDER #######
            env.render()
            # time.sleep(0.1)
            ### RENDER #######
            
            delta = reward + gamma * V[next_state] - V[state]
            e[state] = e[state] +1 
            state = next_state
            total_reward += reward
            
            for t in range(env.observation_space.n):
                V[t] = V[t] + alpha*delta*e[t]
                e[t] = gamma*decay_rate*e[t]
                
            visited_states.append(state)
            state = next_state
            
            row = state // env.shape[1]
            col = state % env.shape[1]
            
            if (row == 3 and col == 11):
                break

        if save_folder is not None:
            plt.imshow(np.reshape(V,[4,12]))

            plt.colorbar() 

            plt.title( "2-D Heat Map" ) 
            plt.savefig(f"./{save_folder}/episode{episode}.png") 
            plt.close()
            
        if episode > 989:
            print("Episode:", episode, "Total Reward:", total_reward, "Path:", visited_states)
            
        return_array = np.append(return_array,total_reward)
        # print(V)
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
runs = 2
episodes = 20

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

td0_return_var = np.var(td0_return,axis=0)
td1_return_var = np.var(td1_return,axis=0)
td2_return_var = np.var(td2_return,axis=0)

print(f'{td0_return_var=}')

t=[i for i in range(episodes)]

# plt.plot(t, td0_return_mean, label='Average Return ofr lamba=0', linestyle='-')
# plt.fill_between(t, td0_return_mean - td0_return_var, td0_return_mean + td0_return_var, alpha=0.2)
# plt.plot(t, td1_return_mean, label='Average Return ofr lamba=0.3', linestyle='-.')
# plt.fill_between(t, td1_return_mean - td1_return_var, td1_return_mean + td1_return_var, alpha=0.2)
# plt.plot(t, td2_return_mean, label='Average Return ofr lamba=0.5')
# plt.fill_between(t, td2_return_mean - td2_return_var, td2_return_mean + td2_return_var, alpha=0.2)
# plt.legend()
# plt.show()