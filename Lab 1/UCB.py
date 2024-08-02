import numpy as np
import matplotlib.pyplot as plt

# Class representing a single arm of the multi-armed bandit
class Arm:
    def __init__(self):
        # Each arm has a mean reward drawn from a normal distribution
        self.mean = np.random.normal(0, np.sqrt(3))
        self.variance = 1

    # Method to simulate pulling the arm, returning a reward
    def pull(self):
        reward = np.random.normal(self.mean, np.sqrt(self.variance))
        return reward

# Class representing the multi-armed bandit with a given strategy
class MultiArmedBandit:
    def __init__(self, num_arms, strategy):
        self.arms = [Arm() for _ in range(num_arms)]
        self.strategy = strategy
        self.strategy.initialize(num_arms)

    # Method to pull a specific arm
    def pull_arm(self, arm_index):
        if 0 <= arm_index < len(self.arms):
            return self.arms[arm_index].pull()
        else:
            raise ValueError("Invalid arm index.")

    # Method to update strategy estimates based on the reward received
    def update_estimates(self, arm_index, reward):
        self.strategy.update(arm_index, reward)

    # Method to select the next arm to pull based on the strategy
    def select_arm(self):
        return self.strategy.select_arm()

# Base class for different strategies (e.g., UCB, epsilon-greedy)
class Strategy:
    def initialize(self, num_arms):
        self.counts = np.zeros(num_arms)  # Number of times each arm has been pulled
        self.values = np.zeros(num_arms)  # Estimated value of each arm
        self.total_pulls = 0  # Total number of pulls

    # Method to update the estimates for a given arm
    def update(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.total_pulls += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        # Incremental update of the mean value estimate for the arm
        self.values[arm_index] = value + (reward - value) / n

    # Method to select the next arm to pull (to be implemented by specific strategies)
    def select_arm(self):
        raise NotImplementedError

# Class implementing the UCB (Upper Confidence Bound) strategy
class UCBStrategy(Strategy):
    def select_arm(self):
        if self.total_pulls < len(self.counts):
            # Pull each arm once initially
            return self.total_pulls
        # Calculate UCB values for each arm
        ucb_values = self.values + 2 * np.sqrt(np.log(self.total_pulls) / self.counts)
        # Select the arm with the highest UCB value
        return np.argmax(ucb_values)

# Function to run the UCB algorithm
def UCB():
    num_arms = 10
    num_iterations = 1000
    num_runs = 100

    all_rewards = np.zeros((num_runs, num_iterations))

    for run in range(num_runs):
        ucb_strategy = UCBStrategy()
        bandit_ucb = MultiArmedBandit(num_arms, ucb_strategy)
        rewards = np.zeros(num_iterations)

        for i in range(num_iterations):
            arm_index = bandit_ucb.select_arm()
            reward = bandit_ucb.pull_arm(arm_index)
            bandit_ucb.update_estimates(arm_index, reward)
            rewards[i] = reward

        all_rewards[run] = rewards

    average_rewards = np.mean(all_rewards, axis=0)

    plt.plot(average_rewards)
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.title('Reward per Iteration (UCB)')
    plt.show()
    return average_rewards

# Run the UCB algorithm
UCB()
