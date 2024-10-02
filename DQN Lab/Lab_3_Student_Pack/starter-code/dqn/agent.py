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
        self.policy_network = DQN(observation_space, action_space, lr, "policy")
        self.target_network = DQN(observation_space, action_space, lr, "target")
        self.learning_rate = lr
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma

        self.update_target_network()
        self.target_network.eval()



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


        states = torch.tensor(samples[STATE], dtype=torch.float).to(self.policy_network.device)/255.0
        actions = torch.tensor(samples[ACTION], dtype=torch.int).to(self.policy_network.device)
        next_states = torch.tensor(samples[NEXT_STATE], dtype=torch.float).to(self.policy_network.device)/255.0
        rewards = torch.tensor(samples[REWARD], dtype=torch.float).to(self.policy_network.device)
        dones = torch.tensor(samples[DONE]).to(self.policy_network.device)

        self.policy_network.train()
        
        predictions = self.policy_network(states)

        with torch.no_grad():
            next_states_predictions = self.target_network(next_states).detach().max(dim=1)[0]
            indices = torch.arange(actions.size(0), dtype=torch.int)
            targets = predictions.detach().clone()
            targets[indices, actions] = rewards + self.gamma * next_states_predictions * ~dones

        loss = self.policy_network.loss(predictions, targets).to(self.policy_network.device)

        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_models(self):
        self.policy_network.save_checkpoint()
        self.target_network.save_checkpoint()

    def load_models(self):
        self.policy_network.load_checkpoint()
        self.target_network.load_checkpoint()

    def act(self, state: np.ndarray) -> int:
        reshaped_state = np.reshape(state,(1, *state.shape)) / 255.0
        tensor_state = torch.tensor(reshaped_state, dtype=torch.float).to(self.policy_network.device)

        with torch.no_grad():
            output = self.policy_network(tensor_state)

        action = torch.argmax(output).item()
        return action
