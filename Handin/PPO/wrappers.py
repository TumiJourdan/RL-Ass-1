import gymnasium as gym
from gymnasium.spaces import MultiBinary

class SingleAgentWrapper(gym.Env): #This is used for the individual agents in the multiagent
    def __init__(self, env, agent_action_key):
        super(SingleAgentWrapper, self).__init__()
        self.env = env
        self.agent_action_key = agent_action_key  # e.g., "change_line_status", "curtail", "redispatch"
        
        # Share the observation space among all agents
        self.observation_space = self.env.observation_space
        
        # Action space is limited to one key for this agent
        self.action_space = self.env.action_space[self.agent_action_key]

    def step(self, action):
        # Create a combined action dict where only the relevant part is filled by this agent
        combined_action = {self.agent_action_key: action}
        return self.env.step(combined_action)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

class MultiAgentWrapper(gym.Env):
    def __init__(self, env, agent_action_keys:list):
        super(MultiAgentWrapper, self).__init__()
        self.env = env
        self.agent_action_keys = agent_action_keys  # e.g., "change_line_status", "curtail", "redispatch"
        
        # Share the observation space among all agents
        self.observation_space = self.env.observation_space
        
        # Action space is limited to one key for this agent
        self.action_space = self.env.action_space

    def step(self, actions):
        # Create a combined action dict where only the relevant part is filled by this agent
        combined_action = {
            self.agent_action_keys[0]: actions[0],
            self.agent_action_keys[1]: actions[1],
            self.agent_action_keys[2]: actions[2],
            self.agent_action_keys[3]: actions[3]
        }
        return self.env.step(combined_action)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def render(self, mode='human'):
        return self.env.render(mode=mode)
    

class HierarchyWrapper(gym.Env):
    def __init__(self, env, agent_action_keys:list, agent_list) -> None:
        super(HierarchyWrapper, self).__init__()

        self.env = env

        self.agent_action_keys = agent_action_keys  # e.g., "change_line_status", "curtail", "redispatch"
        self.agent_list = agent_list

        # Share the observation space among all agents
        self.observation_space = self.env.observation_space

        self.action_space = MultiBinary(len(agent_list))

        self.obs = None

    def step(self, action):
        # Create a combined action dict where only the relevant part is filled by this agent

        combined_action = {}

        # print(action)

        for i, act in enumerate(action):
            if act:
                combined_action[self.agent_action_keys[i]] = self.agent_list[i].predict(self.obs)[0]

        # print(combined_action)

        step_return = self.env.step(combined_action)

        self.obs = step_return[0]

        return step_return

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.obs = obs
        return obs, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)