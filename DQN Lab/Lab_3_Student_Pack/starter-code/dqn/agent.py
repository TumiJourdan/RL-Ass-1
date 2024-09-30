from gym import spaces
import numpy as np
import torch as T
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        lr: float,
        batch_size: int,
        gamma: float,
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

    def _sample_replays(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.policy_network.device) / 255.0
        actions = T.tensor(actions, dtype=T.int).to(self.policy_network.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.policy_network.device) / 255.0
        rewards = T.tensor(rewards, dtype=T.float).to(self.policy_network.device)
        dones = T.tensor(dones).to(self.policy_network.device)

        return states, actions, next_states, rewards, dones

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """

        self.policy_network.train()

        states, actions, next_states, rewards, dones = self._sample_replays()
        predictions = self.policy_network(states)

        with T.no_grad():
            next_states_predictions = self.target_network(next_states).detach().max(dim=1)[0]
            indices = T.arange(actions.size(0), dtype=T.int)
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

        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_models(self):
        self.policy_network.save_checkpoint()
        self.target_network.save_checkpoint()

    def load_models(self):
        self.policy_network.load_checkpoint()
        self.target_network.load_checkpoint()

    def act(self, state: np.ndarray) -> int:
        reshaped_state = state.reshape((1, *state.shape)) / 255.0
        tensor_state = T.tensor(reshaped_state, dtype=T.float).to(self.policy_network.device)

        with T.no_grad():
            output = self.policy_network(tensor_state)

        action = T.argmax(output).item()
        return action
