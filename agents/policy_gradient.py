import tensorflow as tf
import numpy as np


class Policy():

    def __init__(self, sess, task, stats,
            name='actor',
            hidden_units=64,
            dropout_rate=0.6,
            learning_rate=1e-4):
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
        with tf.variable_scope(name, reuse=False):
            self._create_policy_network(hidden_units, dropout_rate)
            self._create_update_network(learning_rate)

    def act(self, states, explore=False):
        actions = self.sess.run(self.action_output, feed_dict={
            self.state: states,
            self.is_training: explore,
        })
        return actions

    def learn(self, states, action_gradients):
        action_gradients = np.array(action_gradients).reshape(-1, self.action_size)
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.state: states,
            self.action_gradients: action_gradients,
            self.is_training: True,
        })
        self.stats.scalar('actor_loss', loss)
        return

    def _create_policy_network(self, hidden_units, dropout_rate):
        self.state = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        x = tf.layers.dense(self.state, hidden_units)
        x = tf.nn.leaky_relu(x)

        x = tf.layers.dense(x, hidden_units)
        x = tf.nn.leaky_relu(x)

        x = tf.layers.dropout(x, training=self.is_training, rate=dropout_rate)

        x = tf.layers.dense(x, hidden_units)
        x = tf.nn.leaky_relu(x)

        x = tf.layers.dropout(x, training=self.is_training, rate=dropout_rate)

        logits = tf.layers.dense(x, self.action_size)
        output = tf.nn.sigmoid(logits)
        self.action_output = output * self.action_range + self.action_low

    def _create_update_network(self, learning_rate):
        self.action_gradients = tf.placeholder(tf.float32, (None, self.action_size), name='action_gradients')
        dAdvantage_dTheta = self.action_gradients * self.action_output
        self.loss = -tf.reduce_mean(dAdvantage_dTheta)
        theta = tf.trainable_variables(scope=self.name)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=theta)
