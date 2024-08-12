import numpy as np
from World import World
import math

import matplotlib.pyplot as plt

UP = (-1,0)
DOWN = (1,0)
LEFT = (0,-1)
RIGHT = (0,1)


class valIteration:
    def __init__(self,world:World):
        self.world = world
    
    def doValIteration(self,gridIn,discount = 1):
        deltas = []
        for x in range(self.world.grid.shape[0]):
            for y in range(self.world.grid.shape[1]):
                #pass iteration if state x is the goal
                grid = gridIn
                
                if((x,y) == self.world.goal):
                    continue
                
                _, nextStates = self.world.returnAvailableMove((x,y))
                sum = 0
                for j in nextStates:
                    reward = -1
                    probability = 1/len(nextStates)
                    nextStateReward = self.world.grid[tuple(j)]
                    sum += (probability)*(reward + discount * nextStateReward)
                    
                difference = abs(self.world.grid[x,y] - sum)
                deltas.append(difference)
                grid[x,y] = sum
                
        self.world.grid = grid
        return deltas        
        
        
    def thetaLoop(self,theta:float,boolInPlace,discount=1):
        delta = math.inf
        count = 0
        while (delta>theta):
            count += 1
            if(boolInPlace):
                deltas = self.doValIteration(self.world.grid,discount)
            else:
                tempInst = np.copy(self.world.grid)
                deltas = self.doValIteration(tempInst,discount)
            delta = np.max(deltas)
            print(delta)
            print(self.world.grid)
        return count
        
obstacleList = []
START = (3,3)
GOAL  =(0,0)



x_space =  np.logspace(-0.2, 0, num=20)
x_1array = np.zeros_like(x_space)
x_2array = np.zeros_like(x_space)
for x in range(len(x_space)):
    world = World((obstacleList), START, GOAL)
    
    valueiterationInst = valIteration(world)
    x_1array[x] = valueiterationInst.thetaLoop(0.1,True,x_space[x])
    
    world = World((obstacleList), START, GOAL)
    valueiterationInst = valIteration(world)
    x_2array[x] = valueiterationInst.thetaLoop(0.1,False,x_space[x])
    
    
# Plotting
plt.figure(figsize=(10, 6))

plt.plot(x_space, x_1array, label='In-Place Evaluation', marker='o')
plt.plot(x_space, x_2array, label='Out-of-Place Evaluation', marker='x')

# Labeling the axes
plt.xlabel('Discount Rate')
plt.ylabel('Number of Iterations to Convergence')

# Adding a title
plt.title('Comparison of Policy Evaluation Methods')

# Adding a legend
plt.legend()

# Display the plot
plt.show()