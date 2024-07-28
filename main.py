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

class MultiArmedBandit:
    def __init__(self, num_arms):
        # Initialize the arms
        self.arms = [Arm() for _ in range(num_arms)]

    def pull_arm(self, arm_index):
        # Pull the specified arm and get a reward
        if 0 <= arm_index < len(self.arms):
            return self.arms[arm_index].pull()
        else:
            raise ValueError("Invalid arm index.")
    
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