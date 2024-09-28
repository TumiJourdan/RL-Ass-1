from gym import spaces
import numpy as np
import copy
import torch.nn as nn
import torch
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

device = "cuda"

STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
DONE = 4

class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
        update_target =1,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """
        # TODO: Initialise agent's networks, optimiser and replay buffer
        self.dqn_model = DQN(observation_space,action_space,lr)
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.target_network = copy.deepcopy(self.dqn_model)
        self.update_target = update_target
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters(),lr = lr)

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss

        # Sample batch from the replay buffer
        samples = self.replay_buffer.sample(self.batch_size) # an array of length 5 where each element is the state, action etc, each batchsize length
        # get the TD error over the minibatch and average it out
        self.optimizer.zero_grad()
        loss_func = nn.MSELoss()
        
        target_output = self.target_network.forward(samples[NEXT_STATE])
        max_action = torch.max(target_output).type(torch.float)
        # max_action *= 1-samples[DONE] 
        r = torch.tensor(samples[REWARD]) 
        policy_network_out = self.dqn_model.forward(samples[STATE])[:,samples[ACTION]].type(torch.float)
    
        loss = loss_func(policy_network_out,r+self.gamma*max_action)
        loss_returned = loss.type(torch.float)
        torch.autograd.set_detect_anomaly(True)
        loss_returned.backward()
        self.optimizer.step()
    
        return loss_returned

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters
        self.target_network = copy.deepcopy(self.dqn_model)

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # TODO Select action greedily from the Q-network given the state
        policy_network_out = np.argmax(self.dqn_model.forward(state))
        return policy_network_out
