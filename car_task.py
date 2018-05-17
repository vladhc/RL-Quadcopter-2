import gym

# More simple task for testing the Agent
class Task():

    def __init__(self):
        env = gym.make('MountainCarContinuous-v0')
        self.env = env

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

    def step(self, action):
        self.env.step(action)
        self.env.step(action)
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.env.reset()
        return state

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
