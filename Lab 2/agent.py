import random
import numpy as np
class Agent:
    def __init__(self, position=(0, 0)):
        self.position = position
        self.reward=0
        self.currentReward=0
        self.path=[]

    def chooseMove(self,movelist):
        raise NotImplementedError("This method should be overridden by subclasses")

    def makeMove(self, move, world):
        self.position, move_reward  = world.takeAction(move)
        self.currentReward=move_reward
        self.reward+=move_reward
        self.path.append(tuple(self.position))

    def getPath(self):
        return self.path
        


class RandomAgent(Agent):
    def chooseMove(self,movelist):
        return random.choice(movelist)


class SmartAgent(Agent):
    def chooseMove(self, movelist, policy):
        scores = []
        for move in movelist:
            possible_move = self.position + move
            if 0 <= possible_move[0] < policy.shape[0] and 0 <= possible_move[1] < policy.shape[1]:  # Check boundaries
                scores.append((policy[tuple(possible_move)], move))
            else:
                scores.append((float('-inf'), move))  # Out of bounds moves get the lowest score

        max_score = max(scores, key=lambda x: x[0])
        best_moves = [move for score, move in scores if score == max_score[0]]

        return best_moves[0]