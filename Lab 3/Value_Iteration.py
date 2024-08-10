import numpy as np
from World import World
import math

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
        while (delta>theta):
            if(boolInPlace):
                deltas = self.doValIteration(self.world.grid,discount)
            else:
                tempInst = np.copy(self.world.grid)
                deltas = self.doValIteration(tempInst,discount)
            delta = np.max(deltas)
            print(delta)
            print(self.world.grid)
            
        
obstacleList = []
START = (3,3)
GOAL  =(0,0)
world = world = World((obstacleList), START, GOAL)

valueiterationInst = valIteration(world)
valueiterationInst.thetaLoop(0.1,True,0.1)