import numpy as np

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
        # Initialize counts and value estimates for each arm
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

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

    def select_arm(self, epsilon):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            # Exploration: select a random arm
            return np.random.randint(0, len(self.arms))
        else:
            # Exploitation: select the arm with the highest estimated value
            return np.argmax(self.values)

def main():
    num_arms = 10  # Number of arms in the bandit
    mab = MultiArmedBandit(num_arms)
    num_iterations = 10000  # Number of iterations, increased to 10000 for more stable estimates
    epsilon = 0.1  # Exploration rate

    for iteration in range(num_iterations):
        # Select an arm to pull using epsilon-greedy strategy
        arm_index = mab.select_arm(epsilon)
        # Pull the selected arm
        reward = mab.pull_arm(arm_index)
        # Update the value estimates for the selected arm
        mab.update_estimates(arm_index, reward)
        
        # Print progress every 1000 iterations
        if (iteration + 1) % 1000 == 0:
            print(f"Iteration {iteration + 1}")
            for i in range(num_arms):
                print(f"  Arm {i}: Estimated Value = {mab.values[i]:.2f}, True Mean = {mab.arms[i].mean:.2f}, Deviation = {abs(mab.values[i] - mab.arms[i].mean):.2f}")

    # Print the final estimated values, true means, and deviations of each arm after all iterations
    print("\nFinal Estimates and True Means:")
    for i in range(num_arms):
        estimated_value = mab.values[i]
        true_mean = mab.arms[i].mean
        deviation = np.abs(estimated_value - true_mean)
        print(f"Arm {i}: Estimated Value = {estimated_value:.2f}, True Mean = {true_mean:.2f}, Deviation = {deviation:.2f}")

main()
