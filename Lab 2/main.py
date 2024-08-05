from World import World,movelist
from agent import RandomAgent, SmartAgent
from policyGrid import GRID


START = (6,0)
GOAL = (0,0)
obstacleList = [(2,0), (2,1), (2,2), (2,3), (2,4), (2,5)]
world = World((obstacleList), START, GOAL)
grid = GRID()
policy=grid.gridWorld

board = world.getBoard()
print(board)

agent1 = RandomAgent(position=START)
agent2 = SmartAgent(position=START)

# ================== AGENT 1 ====================
print(f"AgentOne initial position: {agent1.position}")
for _ in range(50):
    move1 = agent1.chooseMove(movelist)
    agent1.makeMove(move1, world)
    
print(f"AgentOne final position: {agent1.position}")
print(f"AgentOne world final position: {world.getPath()[-1]}")

# ================== AGENT 2 ====================
world = World((obstacleList), START, GOAL)
print(f"Agent Two initial position: {agent2.position}")
while agent2.currentReward!=20:
    move1 = agent2.chooseMove(movelist,policy)
    agent2.makeMove(move1, world)
print(f"AgentTwo final position: {agent2.position}")

print(agent1.getPath())
print(agent1.reward)

print(agent2.getPath())
print(agent2.reward)