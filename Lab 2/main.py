from World import World,movelist
from agent import RandomAgent, SmartAgent
from policyGrid import GRID
import matplotlib.pyplot as plt



def debugBoards(agent):
    path = agent.getPath()
    
    y,x = zip(*agent.getPath())
    plt.scatter(x,y,color='blue', marker='o')
    for i in range(len(x) - 1):
        plt.annotate(
            '', 
            xy=(x[i+1], y[i+1]), 
            xytext=(x[i], y[i]), 
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')    
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.gca().invert_yaxis()
    plt.show()

START = (6,0)
GOAL = (0,0)

obstacleList = [(2,0), (2,1), (2,2), (2,3), (2,4), (2,5)]
world = World((obstacleList), START, GOAL)
grid = GRID()
policy=grid.gridWorld

board = world.getBoard()
print(board)

def run():
    agent1 = RandomAgent(position=START)
    agent2 = SmartAgent(position=START)
    world = World((obstacleList), START, GOAL)
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
    return agent1,agent2

def many_runs(runs:int):
    sum1 = 0
    sum2 = 0
    for i in range(runs):
        agent1,agent2 = run()
        sum1 +=agent1.reward
        sum2 += agent2.reward
    return sum1/runs,sum2/runs
    

agent1,agent2 = run()

print(agent1.getPath())
print(agent1.reward)

print(agent2.getPath())
print(agent2.reward)

debugBoards(agent1)
debugBoards(agent2)

average1,average2 = many_runs(20)

numbers = [average1, average2]
labels = ['Random Agent', 'Greedy Agent']

# Plot
plt.bar(labels, numbers, color=['blue', 'orange'])
plt.xlabel('Labels')
plt.ylabel('Values')
plt.title('Reward Bar Graph')
plt.show()