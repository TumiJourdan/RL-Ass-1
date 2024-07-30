import numpy as np
import matplotlib.pyplot as plt


class Arm:
    def __init__(self):
        # Mean of each arm is drawn from a Gaussian with mean 0 and variance 3
        self.mean = np.random.normal(0, np.sqrt(3))
        # Variance of rewards from each arm is fixed at 1
        self.variance = 1

    def pull(self):
        # Return a reward sampled from a Gaussian with the arm's mean and variance
        reward = np.random.normal(self.mean, np.sqrt(self.variance))
        return reward

class DefaultStrategy:
    def initialize(self, num_arms):
        pass

    def update(self, arm_index, reward):
        pass

    def select_arm(self):
        return 0

class MultiArmedBandit:
    def __init__(self, num_arms, strategy=None):
        # Initialize the arms
        self.arms = [Arm() for _ in range(num_arms)]
        self.strategy = strategy if strategy is not None else DefaultStrategy()
        self.strategy.initialize(num_arms)


    def pull_arm(self, arm_index):
        # Pull the specified arm and get a reward
        if 0 <= arm_index < len(self.arms):
            return self.arms[arm_index].pull()
        else:
            raise ValueError("Invalid arm index.")

    def update_estimates(self, arm_index, reward):
        self.strategy.update(arm_index, reward)

    # Method to select the next arm to pull based on the strategy
    def select_arm(self):
        return self.strategy.select_arm()
    
class Results:
    def __init__(self,num_arms):
        self.arms = num_arms
        self.results = []
        
    def store_results(self,result):
        self.results.append(result)
    
    def displayGraphs(self):
        steps = self.results.count
        averages = []
        for x in self.results:
            averages.append(np.mean(x))
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.xticks(range(len(averages)))
        
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



def main():
    num_arms = 10  # Number of arms in the bandit
    mab = MultiArmedBandit(num_arms)

    # Example of pulling each arm and printing the reward
    for i in range(num_arms):
        reward = mab.pull_arm(i)
        print(f"Pulled arm {i}, received reward: {reward:.2f}")

    


def epsilon_greedy_optimistic(mab:MultiArmedBandit,iterations:int,alpha:float, optimistic_estimate:float):
    # need ten running sums
    running_sums = np.zeros(10)
    
    for x in running_sums:
        x = optimistic_estimate
    # Explore or exploit
    
    # exploit
    max_estimate = np.max(running_sums)
    candidates = np.where(running_sums == max_estimate)[0]
    arm_index =  np.random.choice(candidates)
    # Explore
    arm_index = np.random.choice(running_sums)
    
    # do action
    reward = mab.pull_arm(arm_index)
    
    Qn = running_sums[arm_index]
    Rn = reward
    running_sums[arm_index] +=Qn + alpha*(Rn - Qn)
    

if __name__ == "__main__":
    main()

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


    return average_rewards

# Run the UCB algorithm
ucb=UCB()
plt.plot(ucb)
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.title('Reward per Iteration (UCB)')
plt.show()