import tensorflow as tf
import numpy as np

from agents.q_network import QNetwork
from agents.policy_gradient import Policy
from agents.replay_buffer import ReplayBuffer, Experience
from agents.ou_noise import OUNoise

class Agent():

    def __init__(self, task, sess, stats):
        self.sess = sess
        self.task = task
        self.stats = stats

        self.q_network = QNetwork(sess, task.state_size, task.action_size, stats, hidden_units=16)
        self.actor = Policy(task, sess, hidden_units=16)

        self.gamma = 0.99 # reward discount rate

        # Exploration noise process
        exploration_mu = 0
        exploration_theta = 0.15
        exploration_sigma = 0.05
        self.noise = OUNoise(task.action_size, exploration_mu, exploration_theta, exploration_sigma)

        # Replay memory
        buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(buffer_size)

        self.sess.run(tf.global_variables_initializer())

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        # Save experience
        self._save_experience(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            self._learn()

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state, explore=False):
        """Returns actions for given state(s) as per current policy."""
        action = self.actor.act([state])[0]
        assert not np.isnan(action)

        if explore:
            action = action + self.noise.sample()
            action = np.maximum(action, self.task.action_low)
            action = np.minimum(action, self.task.action_high)

        assert not np.isnan(action)
        assert np.all(action >= self.task.action_low), "expected less than {:7.3f}, but was {}".format(task.action_low, action)
        assert np.all(action <= self.task.action_high)

        return action

    def _learn(self):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        experiences, experience_indexes = self.memory.sample(self.batch_size)
        action_size = self.task.action_size
        states = np.vstack([e.state for e in experiences])
        actions = np.array([e.action for e in experiences]).astype(np.float32).reshape(-1, action_size)
        rewards = np.array([e.reward for e in experiences]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences])

        # Get predicted next-state actions and Q values
        actions_next = self.actor.act(next_states)
        Q_targets_next = self.q_network.get_q(next_states, actions_next)

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        td_errs = self.q_network.learn(states, actions, Q_targets)
        self.memory.update_td_err(experience_indexes, td_errs)

        self.memory.scrape_stats(self.stats)

        # Train actor model
        scores = self.q_network.get_q(next_states, actions_next)
        self.actor.learn(next_states, scores)

    def _save_experience(self, state, action, reward, next_state, done):
        """Adds experience into ReplayBuffer. As a side effect, also learns q network on this sample."""
        # Get predicted next-state actions and Q values
        actions_next = self.actor.act([next_state])
        Q_target_next = self.q_network.get_q([next_state], actions_next)[0]

        # Compute Q targets for current states and train critic model (local)
        Q_target = reward + self.gamma * Q_target_next * (1 - done)
        td_err = self.q_network.learn([state], [action], [Q_target])

        self.memory.add(Experience(state, action, reward, next_state, done), td_err)
