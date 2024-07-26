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

    def pull_arm(self, arm_index):
        # Pull the specified arm and get a reward
        if 0 <= arm_index < len(self.arms):
            return self.arms[arm_index].pull()
        else:
            raise ValueError("Invalid arm index.")

def main():
    num_arms = 10  # Number of arms in the bandit
    mab = MultiArmedBandit(num_arms)

    # Example of pulling each arm and printing the reward
    for i in range(num_arms):
        reward = mab.pull_arm(i)
        print(f"Pulled arm {i}, received reward: {reward:.2f}")


def epsilon_greedy_optimistic():
    #pass
    print("fuck u python")

if __name__ == "__main__":
    main()