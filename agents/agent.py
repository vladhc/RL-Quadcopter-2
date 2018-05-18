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

        tau = 0.05

        self.critic_local = QNetwork(
                sess, task, stats, name='critic_local',
                hidden_units=64, dropout_rate=0.2)
        self.critic_target = QNetwork(
                sess, task, stats, name='critic_target',
                hidden_units=64, dropout_rate=0.2)
        self.actor_local = Policy(
                sess, task, stats, name='actor_local',
                hidden_units=64, dropout_rate=0.75)
        self.actor_target = Policy(
                sess, task, stats, name='actor_target',
                hidden_units=64, dropout_rate=0.75)
        soft_copy_critic_ops = self._create_soft_copy_op(
                'critic_local', 'critic_target',
                tau=tau)
        soft_copy_actor_ops = self._create_soft_copy_op(
                'actor_local', 'actor_target',
                tau=tau)
        self._soft_copy_ops = []
        self._soft_copy_ops.extend(soft_copy_critic_ops)
        self._soft_copy_ops.extend(soft_copy_actor_ops)

        self.gamma = 0.7 # reward discount rate

        # Exploration noise process
        exploration_mu = 0
        exploration_theta = 0.15
        exploration_sigma = 0.25
        self.noise = OUNoise(task.action_size, exploration_mu, exploration_theta, exploration_sigma)

        # Replay memory
        self.batch_size = 128
        self.memory = ReplayBuffer(buffer_size=10000, decay_steps=1000)

        self.sess.run(tf.global_variables_initializer())

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.memory.decay_a()
        return state

    def step(self, action, reward, next_state, done):
        # Save experience
        self._save_experience(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.learn()

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state, explore=False):
        """Returns actions for given state(s) as per current policy."""
        actor = self.actor_local if explore else self.actor_target
        action = actor.act([state], explore)[0]
        assert not np.any(np.isnan(action))

        if explore:
            action = action + self.noise.sample()
            action = np.maximum(action, self.task.action_low)
            action = np.minimum(action, self.task.action_high)

        assert not np.any(np.isnan(action))
        assert np.all(action >= self.task.action_low), "expected less than {:7.3f}, but was {}".format(task.action_low, action)
        assert np.all(action <= self.task.action_high)

        return action

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples."""

        if len(self.memory) < self.batch_size:
            return

        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        experiences, experience_indexes = self.memory.sample(self.batch_size)
        action_size = self.task.action_size
        states = np.vstack([e.state for e in experiences])
        actions = np.array([e.action for e in experiences]).astype(np.float32).reshape(-1, action_size)
        rewards = np.array([e.reward for e in experiences]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences])

        # Get predicted next-state actions, Q and V values
        actions_next = self.actor_target.act(next_states)
        Q_targets_next, V_targets_next = self.critic_target.get_q_and_v(next_states, actions_next)

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        V_targets = rewards + self.gamma * V_targets_next * (1 - dones)
        td_errs = self.critic_local.learn(states, actions, Q_targets, V_targets)

        self.memory.update_td_err(experience_indexes, td_errs)

        self.memory.scrape_stats(self.stats)

        # Train actor model
        actions = self.actor_target.act(states)
        action_gradients = self.critic_target.get_action_gradients(states, actions)
        self.actor_local.learn(states, action_gradients)

        self._soft_copy()

    def _save_experience(self, state, action, reward, next_state, done):
        """Adds experience into ReplayBuffer. As a side effect, also learns q network on this sample."""
        # Get predicted next-state actions and Q values
        actions_next = self.actor_local.act([next_state])
        Q_targets_next, _ = self.critic_local.get_q_and_v([next_state], actions_next)
        Q_target_next = Q_targets_next[0]

        Q_target = reward + self.gamma * Q_target_next * (1 - done)
        td_err = self.critic_local.get_td_err([state], [action], [Q_target])

        self.memory.add(Experience(state, action, reward, next_state, done), td_err)

    def _soft_copy(self):
        self.sess.run(self._soft_copy_ops)

    def _create_soft_copy_op(self, scope_src, scope_dst, tau=0.01):
        var_src = tf.trainable_variables(scope=scope_src)
        var_dst = tf.trainable_variables(scope=scope_dst)
        copy_ops = []
        for src, dst in zip(var_src, var_dst):
            mixed = tau * src + (1.0 - tau) * dst
            copy_op = tf.assign(dst, mixed)
            copy_ops.append(copy_op)
        return copy_ops
