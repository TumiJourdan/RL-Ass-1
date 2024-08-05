import random
class Agent:
    def __init__(self, position=(0, 0)):
        self.position = position
        self.reward=0

    def chooseMove(self,movelist):
        raise NotImplementedError("This method should be overridden by subclasses")

    def makeMove(self, move, world):
        self.position, move_reward  = world.takeAction(move)
        self.reward+=move_reward
        


class RandomAgent(Agent):
    def chooseMove(self,movelist):
        return random.choice(movelist)


class SmartAgent(Agent):
    def chooseMove(self):
        # Your specific implementation for AgentTwo
        return "Move chosen by AgentTwo"