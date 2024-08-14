###
# Group Members
# Name:Student Number
# Name:Student Number
# Name:Student Number
# Name:Student Number
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt


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
                    new_v += 1/4 * prob * (reward + discount_factor * V[next_state])

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
    policy_stable = False
    while(policy_stable == False):
        policy_eval_V = policy_evaluation_fn()
        
        policy = np.ones((env.observation_space.n,env.action_space.n))/4
        print(policy)
        policy_stable = True
        
        for x in range(env.observation_space.n):
            # this line takes the action as an index from the policy [0.25,0.25,0.25,0.25]
            old_action = np.random.choice([0,1,2,3], size = None, replace= True, p=policy[x])
            
            expected_rewards = one_step_lookahead(x,policy_eval_V)
            max_reward_index = expected_rewards.index(max(expected_rewards))
            if(old_action != max_reward_index):
                policy_stable = False
            else:
                return (policy,policy_eval_V)
            
        
        # Pick the action with the highest reward, and update the policy matrix, for example if right has the highest
        # reward the row for that state is [0,1,0,0]
        # Remember to compare the sampled action with the taken action to update the policy_stable
        
            
            

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
        grid = np.arange(env.observation_space.n).reshape(env.shape)
        it = np.nditer(grid, flags=['multi_index'])
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

    raise NotImplementedError


def value_iteration(env, theta=0.0001, discount_factor=1.0):
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
        raise NotImplementedError

    raise NotImplementedError


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

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
    policy, v = [], []  # call policy_iteration

    # TODO Print out best action for each state in grid shape

    # TODO: print state value for each state, as grid shape

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
    policy, v = [], []  # call value_iteration

    # TODO Print out best action for each state in grid shape

    # TODO: print state value for each state, as grid shape

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)


if __name__ == "__main__":
    main()
