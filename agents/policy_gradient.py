import tensorflow as tf
import numpy as np


class Policy():

    def __init__(self, task, sess, stats, name='actor', hidden_units=64):
        self.task = task
        self.sess = sess
        self.stats = stats
        self.name = name

        self.state_size = task.state_size
        self.action_size = task.action_size

        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # create policy network
        with tf.variable_scope(name):
            self.state, self.action_output = self._create_policy_network(hidden_units)
            self.action_gradients, self.loss, self.update_op = self._create_update_network()

    def act(self, states):
        actions = self.sess.run(self.action_output, feed_dict={
            self.state: states,
        })
        return actions

    def learn(self, states, action_gradients):
        action_gradients = np.array(action_gradients).reshape(-1, self.action_size)
        _, loss = self.sess.run([self.update_op, self.loss], feed_dict={
            self.state: states,
            self.action_gradients: action_gradients,
        })
        self.stats.scalar('actor_loss', loss)
        return

    def _create_policy_network(self, hidden_units):
        state = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')

        x = tf.layers.dense(state, hidden_units)
        x = tf.nn.leaky_relu(x)

        x = tf.layers.dense(x, hidden_units)
        x = tf.nn.leaky_relu(x)

        x = tf.layers.dense(x, hidden_units)
        x = tf.nn.leaky_relu(x)

        logits = tf.layers.dense(x, self.action_size)
        output = tf.nn.sigmoid(logits)
        action_output = output * self.action_range + self.action_low
        return state, action_output

    def _create_update_network(self):
        action_gradients = tf.placeholder(tf.float32, (None, self.action_size), name='action_gradients')
        dq_dtheta = action_gradients * self.action_output
        loss = -tf.reduce_mean(dq_dtheta)
        theta = tf.trainable_variables(scope=self.name)
        train_op = tf.train.AdamOptimizer().minimize(loss, var_list=theta)

        return action_gradients, loss, train_op
