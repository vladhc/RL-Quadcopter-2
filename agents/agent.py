import tensorflow as tf
import numpy as np

from agents.q_network import QNetwork
from agents.policy_gradient import Policy
from agents.replay_buffer import ReplayBuffer, Experience

class Agent():

    def __init__(self, task, sess, stats):
        self.sess = sess
        self.task = task
        self.stats = stats

        self.q_network = QNetwork(sess, task.state_size, task.action_size, stats)
        self.actor = Policy(task, sess)

        self.gamma = 0.99 # reward discount rate

        # Replay memory
        buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(buffer_size)

        self.sess.run(tf.global_variables_initializer())

    def reset_episode(self):
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        # Save experience
        self._save_experience(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self._learn_q(experiences)
            self._learn_policy(self.last_state)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        actions = self.actor.act([state]) 
        return actions[0]

    def _learn_policy(self, state):
        # Train actor model
        actions = self.actor.act([state])
        score = self.q_network.get_q([state], actions)[0][0]
        self.stats.scalar('taken_action_score', score)
        self.actor.learn(state, score)

    def _learn_q(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        action_size = self.task.action_size
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values
        actions_next = self.actor.act(next_states)
        Q_targets_next = self.q_network.get_q(next_states, actions_next)

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        td_errs = self.q_network.learn(states, actions, Q_targets)

        self.memory.scrape_stats(self.stats)

    def _save_experience(self, state, action, reward, next_state, done):
        """Adds experience into ReplayBuffer. As a side effect, also learns q network on this sample."""
        # Get predicted next-state actions and Q values
        actions_next = self.actor.act([next_state])
        Q_target_next = self.q_network.get_q([next_state], actions_next)[0]

        # Compute Q targets for current states and train critic model (local)
        Q_target = reward + self.gamma * Q_target_next * (1 - done)
        td_err = self.q_network.learn([state], [action], [Q_target])[0]

        self.memory.add(Experience(state, action, reward, next_state, done), td_err)
