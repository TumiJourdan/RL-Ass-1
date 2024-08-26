###
# Group Members
# Shakeel Malagas: 2424161 
# Tumi Jourdan: 2180153
# Dean Solomon: 2347848
# Tao Yuan: 2332155
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt
import time


def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0
        for s in range(env.observation_space.n):  # Iterate over all states
            v = V[s]
            actions = policy[s]

            new_v = 0
            for i,a in enumerate(actions):
                for prob, next_state, reward, done in env.P[s][i]:
                    new_v += a * prob * (reward + discount_factor * V[next_state])

            V[s] = new_v
            delta = max([delta, abs(v - V[s])])


        if delta < theta:
            break

    return V


def policy_iteration(env:GridworldEnv, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    def one_step_lookahead(state, V):
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
    
    policy_stable = False
    policy = np.ones((env.observation_space.n,env.action_space.n))/4
    while(policy_stable == False):
        
        policy_eval_V = policy_evaluation_fn(env,policy,discount_factor = discount_factor)
        policy_stable = True
        
        for x in range(env.observation_space.n-1):
            # this line takes the action as an index from the policy [0.25,0.25,0.25,0.25]
            old_action = np.random.choice([0,1,2,3], size = None, replace= True, p=policy[x])
            
            expected_rewards = one_step_lookahead(x,policy_eval_V)
            max_reward_index = expected_rewards.index(max(expected_rewards))
            
            policy[x] = [0,0,0,0]
            policy[x][max_reward_index] = 1
            
            if(old_action != max_reward_index):
                policy_stable = False
        # Pick the action with the highest reward, and update the policy matrix, for example if right has the highest
        # reward the row for that state is [0,1,0,0]
        # Remember to compare the sampled action with the taken action to update the policy_stable
                 
    

    return (policy,policy_eval_V)


def value_iteration(env:GridworldEnv, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
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

    V = np.zeros(env.observation_space.n)
    count = 1
    while True:
        delta = 0
        for s in range(env.observation_space.n-1):  # Iterate over all states
            v = V[s]
            # possible actions
            actions = [0,1,2,3]
            new_v = []
            for i,a in enumerate(actions):
                result = 0
                for prob, next_state, reward, done in env.P[s][i]:
                    result += 1 * prob * (reward + discount_factor * V[next_state])
                new_v.append(result)           
            max_reward_index = new_v.index(max(new_v))
            V[s] = max(new_v)

            delta = max([delta, abs(v - V[s])])
        # debugging purposes    

        if delta < theta:
            break
        
    # determine greedy policy
    policy = np.zeros((env.observation_space.n,env.action_space.n))
    
    for s in range(env.observation_space.n):
        expected_rewards = one_step_lookahead(s,V)
        max_reward_index = expected_rewards.index(max(expected_rewards))
        
        policy[s] = [0,0,0,0]
        policy[s][max_reward_index] = 1

    return policy,V
        
def print_policy(policy):
    policy_print = np.zeros(policy.shape[0])

    for i in range(policy.shape[0]):
        policy_print[i] = np.argmax(policy[i])

    policy_print = np.reshape(policy_print, [5,5])
    policy_print[-1][-1] = -1
    print(policy_print)

def action2letter(num):
    if num == 0:
        return 'U'
    elif num == 1:
        return 'R'
    elif num == 2:
        return 'D'
    else:
        return "L"
    
def create_trajectory(env, num_moves, state):

    map = [[0 for j in range(env.shape[0])] for i in range(env.shape[1])]

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    availible_moves = [UP,RIGHT,DOWN,LEFT]


    for i in range(num_moves):

        action = np.random.choice(availible_moves)

        map[int(state//env.shape[0])][int(state%env.shape[1])] = action2letter(action)

        state_, reward, done, _ = env.step(action)

        state = state_

    map[int(state//env.shape[0])][int(state%env.shape[1])] = 'X'

    for i in range(env.shape[0]):
        print(map[i])


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

    # Queston 1

    print('Question 1 Trajectory')

    create_trajectory(env, 5, state)


    # TODO: generate random policy
    rand_policy = np.zeros([env.observation_space.n, env.action_space.n])

    for s in range(env.observation_space.n):
        actions = np.ones(4)

        actions /= sum(actions)

        rand_policy[s] = actions

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # TODO: evaluate random policy
    v = policy_evaluation(env,rand_policy, discount_factor=1)

    # TODO: print state value for each state, as grid shape
    
    print(np.reshape(v, [*env.shape]))    

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # TODO: use  policy improvement to compute optimal policy and state values
    policy, v = policy_iteration(env,policy_evaluation_fn=policy_evaluation)
    # call policy_iteration
    
    # TODO Print out best action for each state in grid shape

    print_policy(policy)
    print('')

    # TODO: print state value for each state, as grid shape

    print(np.reshape(v, [*env.shape]), '\n') 

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # TODO: use  value iteration to compute optimal policy and state values
    policy, v = value_iteration(env)
    # TODO Print out best action for each state in grid shape

    print_policy(policy)
    print('')

    # TODO: print state value for each state, as grid shape

    print(np.reshape(v, [*env.shape]), '\n') 

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)
    # QUestion 4.1.2
    values = np.logspace(-0.2, 0, num=30)
    results_policy = np.zeros(len(values))
    results_value = np.zeros(len(values))
    for discount_index in range(len(values)):
        policy_it_total_time = 0
        value_it_total_time = 0
        print(values[discount_index])
        for y in range(10):
            # Policy Iteration
            env.reset()
            start_time = time.time()
            policy_iteration(env,policy_evaluation,values[discount_index])
            end_time = time.time()
            policy_it_total_time += end_time - start_time
            # Value Iteration
            env.reset()
            start_time = time.time()
            value_iteration(env,theta=0.0001,discount_factor=values[discount_index])
            end_time = time.time()
            value_it_total_time += end_time - start_time
        policy_average_run = policy_it_total_time/10
        value_average_run = value_it_total_time/10
        results_policy[discount_index] = policy_average_run
        results_value[discount_index] = value_average_run
    print("Average policy :", results_policy)
    print("Average value :", results_value)
    # Plotting
    values = np.logspace(-0.2, 0, num=30)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(values, results_policy, label='Policy Iteration', marker='o')
    plt.plot(values, results_value, label='Value Iteration', marker='s')

    # Adding labels and title
    plt.xlabel('Discount Factor')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Comparison of Policy Iteration and Value Iteration Runtimes')
    plt.xscale('log')  # Logarithmic scale for the x-axis
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    
if __name__ == "__main__":
    main()