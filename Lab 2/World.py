import numpy as np

#constants 
UP = (-1,0)
DOWN = (1,0)
LEFT = (0,-1)
RIGHT = (0,1)
movelist=[UP,DOWN,LEFT,RIGHT]
class World:
    
    def setObstacles(self, obstacles):
        for obstacle in obstacles:
            self.grid[obstacle[0]][obstacle[1]] = 0

    def __init__(self, obstacles, start, goal): #obstacles is a list of tuple coordinates,  start and goal are tuple coordinates
        self.grid = np.zeros((7,7))
        self.grid = self.grid - 1

        self.AgentCoordinates = np.array(start)
        self.PathTravelled = [start]
        self.obstacleList = obstacles
        self.goal = goal

        #adding goal and obstacle to grid for visual
        self.grid[goal[0]][goal[1]] = 20

        self.setObstacles(obstacles)
        

    def takeAction(self, move):
        #move is integer, 1 2 3 4 (up down left right respectively)
        projectedCoordinates = self.AgentCoordinates + move
        
        if(projectedCoordinates[0] < 0 or projectedCoordinates[1] < 0 or projectedCoordinates[0] > 6 or projectedCoordinates[0] > 6): #out of bounds check
            self.PathTravelled.append(tuple(self.AgentCoordinates))
            return self.AgentCoordinates, -1
        
        #hitting obstacle check
        if any(np.array_equal(projectedCoordinates, obstacle) for obstacle in self.obstacleList):
            self.PathTravelled.append(tuple(self.AgentCoordinates))
            return self.AgentCoordinates, -1
        
        #goal check
        if np.array_equal(projectedCoordinates, self.goal):
            self.PathTravelled.append(tuple(projectedCoordinates))
            self.AgentCoordinates = projectedCoordinates
            return self.AgentCoordinates, 20
        
        #normal move
        self.PathTravelled.append(tuple(projectedCoordinates))
        self.AgentCoordinates = projectedCoordinates
        return self.AgentCoordinates, -1
    
    def getBoard(self):
        return self.grid
    
    def getPath(self):
        return self.PathTravelled
    
# example usage
# start = (6,0)
# goal = (0,0)
# obstacleList = [(2,0), (2,1), (2,2), (2,3), (2,4), (2,5)]
# world = World((obstacleList), start, goal)

# board = world.getBoard()
# print(board)

# position, reward = world.takeAction(UP)

# print(position, reward) #position is returned as np array, if you want coordinates then cast with tuple(position)

# path = world.getPath()
# print(path)