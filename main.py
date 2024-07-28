import numpy as np
import matplotlib.pyplot as plt
import random

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
        
        
class epsilon_greedy_optimistic():
    def __init__(self,mab,iterations,learning_alpha,optimistic_estimate,epsilon) -> None:
        self.iterations = iterations
        self.learning_alpha = learning_alpha
        self.optimistic_estimate = optimistic_estimate
        self.epsilon = epsilon
        self.mab = mab
    
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



def main():
    num_arms = 10  # Number of arms in the bandit
    mab = MultiArmedBandit(num_arms)

    # Example of pulling each arm and printing the reward
    for i in range(num_arms):
        reward = mab.pull_arm(i)
        print(f"Pulled arm {i}, received reward: {reward:.2f}")
    
    epochs = 10000
    ego = epsilon_greedy_optimistic(mab,epochs,0.1,20,0.01)
    ego_results = ego.run_greedy_opt()
    
    ego_results.displayGraphs()

if __name__ == "__main__":
    main()