import gym
import numpy as np

# More simple task for testing the Agent
class Task():

    def __init__(self):
        env = gym.make('MountainCarContinuous-v0')
        self.env = env

        self.action_repeat = 3
        self.state_size = env.observation_space.shape[0] * self.action_repeat
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

    def step(self, action):
        substates = []
        for _ in range(self.action_repeat):
            next_state, reward, done, _ = self.env.step(action)
            substates.append(next_state)
            if done:
                break
        while len(substates) < self.action_repeat:
            substates.append(next_state)
        state = np.concatenate(substates)
        return state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        next_state = self.env.reset()
        substates = []
        while len(substates) < self.action_repeat:
            substates.append(next_state)
        state = np.concatenate(substates)
        return state

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
