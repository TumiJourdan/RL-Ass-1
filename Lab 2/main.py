from World import World,movelist
from agent import RandomAgent, SmartAgent
from policyGrid import GRID


start = (6,0)
goal = (0,0)
obstacleList = [(2,0), (2,1), (2,2), (2,3), (2,4), (2,5)]
world = World((obstacleList), start, goal)
grid = GRID()
policy=grid.gridWorld


board = world.getBoard()
print(board)

agent1 = RandomAgent(position=start)
agent2 = SmartAgent(position=start)


print(f"AgentOne initial position: {agent1.position}")
for _ in range(50):
    move1 = agent1.chooseMove(movelist)
    agent1.makeMove(move1, world)


print(f"AgentOne final position: {agent1.position}")

while agent2.currentReward!=20:
    move1 = agent2.chooseMove(movelist,policy)
    agent2.makeMove(move1, world)


print(agent1.getPath())
print(agent1.reward)

print(agent2.getPath())
print(agent2.reward)