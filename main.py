import numpy as np
import matplotlib.pyplot as plt
import random

class Results:
    def __init__(self,num_arms):
        self.arms = num_arms
        self.results = []
        self.rewards = []
        
    def store_results(self,result):
        self.results.append(result)
    def store_rewards(self,reward):
        self.rewards.append(reward)
        
    def displayGraphs(self):
        nprewards = np.array(self.rewards)
        rewards_reshaped = nprewards.reshape(-1, 100)
        averages = rewards_reshaped.mean(axis=1)
        
        plt.plot(averages)
        plt.show()
        

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
# Class representing the multi-armed bandit with a given strategy
class MultiArmedBandit:
    def __init__(self, num_arms, strategy=None):
        # Initialize the arms
        self.arms = [Arm() for _ in range(num_arms)]
        self.strategy = strategy if strategy is not None else DefaultStrategy()
        self.strategy.initialize(num_arms)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def clear(self):
        self.strategy.initialize(10)
        self.counts = np.zeros(10)
        self.values = np.zeros(10)

    def pull_arm(self, arm_index):
        # Pull the specified arm and get a reward
        if 0 <= arm_index < len(self.arms):
            return self.arms[arm_index].pull()
        else:
            raise ValueError("Invalid arm index.")

    def update_estimates(self, arm_index, reward):
        # Update the count for the chosen arm
        self.counts[arm_index] += 1
        # Update the estimated value of the chosen arm using incremental formula
        self.values[arm_index] += (reward - self.values[arm_index]) / self.counts[arm_index]

    def update_estimatesUCB(self, arm_index, reward):
        self.strategy.update(arm_index, reward)

    # Method to select the next arm to pull based on the strategy
    def select_arm(self, epsilon):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            # Exploration: select a random arm
            return np.random.randint(0, len(self.arms))
        else:
            # Exploitation: select the arm with the highest estimated value
            return np.argmax(self.values)
        
    def select_armUCB(self):
        return self.strategy.select_arm()
    
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
    


class UCBStrategy(Strategy):
    def select_arm(self):
        if self.total_pulls < len(self.counts):
            # Pull each arm once initially
            return self.total_pulls
        # Calculate UCB values for each arm
        ucb_values = self.values + 2 * np.sqrt(np.log(self.total_pulls) / self.counts)
        # Select the arm with the highest UCB value
        return np.argmax(ucb_values)




class epsilon_greedy_optimistic():
    
    def __init__(self,learning_alpha,optimistic_estimate,epsilon) -> None:
        self.iterations = 1000
        self.learning_alpha = learning_alpha
        self.optimistic_estimate = optimistic_estimate
        self.epsilon = epsilon
        self.mab = MultiArmedBandit(10)
    
    def run_greedy_opt(self):
        
        debug = False
        
        num_arms = 10
        results = Results(num_arms)
        running_sums = np.zeros(num_arms)

        for x in range(len(running_sums)):
            running_sums[x] = self.optimistic_estimate
            
        counter = 0
        while counter<self.iterations:
            ##
            if(debug):print(running_sums)
            # Explore or exploit
            flip_result = random.random()
            if(flip_result > self.epsilon):
                ##
                if(debug):print("Exploit")
                # exploit
                max_estimate = np.max(running_sums)
                candidates = np.where(running_sums == max_estimate)[0]
                arm_index =  int(np.random.choice(candidates))
            else:
                ##
                if(debug):print("Explore")
                # Explore
                arm_index = np.random.randint(0, len(running_sums))
                
            if(debug):print("Selected = ",running_sums[arm_index])
            # do action
            reward = self.mab.pull_arm(arm_index)
            Qn = running_sums[arm_index]
            Rn = reward
            running_sums[arm_index] = Qn + self.learning_alpha*(Rn - Qn)
            if(np.isinf(running_sums[arm_index])):
                print("Is infinity")
                print("Rn = ", Rn)
                print("Qn = ", Qn)
                
            results.store_results(running_sums)
            results.store_rewards(reward)
            counter +=1
        return results


# Function to run the UCB algorithm
def UCB(bandit_ucb):
    
    num_iterations = 1000
    num_runs = 100

    all_rewards = np.zeros((num_runs, num_iterations))

    for run in range(num_runs):
        
        bandit_ucb.clear()
        rewards = np.zeros(num_iterations)

        for i in range(num_iterations):
            arm_index = bandit_ucb.select_armUCB()
            reward = bandit_ucb.pull_arm(arm_index)
            bandit_ucb.update_estimatesUCB(arm_index, reward)
            rewards[i] = reward

        all_rewards[run] = rewards

    average_rewards = np.mean(all_rewards, axis=0)


    return average_rewards


def Egreedy(mab):
    
    num_iterations = 1000  # Number of iterations
    epsilon = 0.1  # Exploration rate

    all_rewards = np.zeros((100, num_iterations))

    for run in range(100):
        mab.clear()
        rewards = np.zeros(num_iterations)

        for i in range(num_iterations):
            # Select an arm to pull using epsilon-greedy strategy
            arm_index = mab.select_arm(epsilon)
            # Pull the selected arm
            reward = mab.pull_arm(arm_index)
            
            # Update the value estimates for the selected arm
            mab.update_estimates(arm_index, reward)
            rewards[i] = reward

        all_rewards[run] = rewards

    average_rewards = np.mean(all_rewards, axis=0)

    return average_rewards
        
    # Print the final estimated values, true means, and deviations of each arm after all iterations
    #print("\nFinal Estimates and True Means:")
    #for i in range(num_arms):
    #    estimated_value = mab.values[i]
    #    true_mean = mab.arms[i].mean
    #    deviation = np.abs(estimated_value - true_mean)
    #    print(f"Arm {i}: Estimated Value = {estimated_value:.2f}, True Mean = {true_mean:.2f}, Deviation = {deviation:.2f}")

ucb_strategy = UCBStrategy()
mab = MultiArmedBandit(10, ucb_strategy)
# Run the UCB algorithm
egreedy=Egreedy(mab)
ucb=UCB(mab)
ego = epsilon_greedy_optimistic(0.1,20,0.01)
ego_results = ego.run_greedy_opt()

ego_results.displayGraphs()

plt.plot(egreedy, label="e greedy")
plt.plot(ucb, label="UCB")
plt.legend(loc="lower right")
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.title('Reward per Iteration')
plt.show()
