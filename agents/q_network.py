import tensorflow as tf
import logging

class QNetwork():

    def __init__(self, sess, state_size, action_size, stat_collector):
        self.sess = sess
        self.stat_collector = stat_collector

        # create network
        self.state = tf.placeholder(tf.float32, shape=(None, state_size), name='state')
        self.action = tf.placeholder(tf.float32, shape=(None, action_size), name='action')

        # state network
        hidden_units = 64
        x = tf.layers.dense(self.state, hidden_units)
        x = tf.nn.relu(x)

        x = tf.layers.dense(x, hidden_units)
        state_out = tf.nn.relu(x)
        logging.debug('state_out', state_out)

        # action network
        x = tf.layers.dense(self.state, hidden_units)
        x = tf.nn.relu(x)

        x = tf.layers.dense(x, hidden_units)
        action_out = tf.nn.relu(x)
        logging.debug('action_out', action_out)

        # state + action → Q
        x = tf.concat([state_out, action_out], 1)
        logging.debug('state_out + action_out', x)

        x = tf.layers.dense(x, hidden_units)
        self.q = tf.layers.dense(x, 1)

        self.q_target = tf.placeholder(tf.float32, shape=(None, 1), name='q_target')
        self.td_err = tf.squeeze(self.q_target - self.q)
        self.q_loss = tf.losses.mean_squared_error(self.q_target, self.q)

        self.train_op = tf.train.GradientDescentOptimizer(0.001).minimize(self.q_loss)

    def learn(self, states, actions, q_targets):
        _, q_loss, td_errs = self.sess.run([self.train_op, self.q_loss, self.td_err], {
            self.state: states,
            self.action: actions,
            self.q_target: q_targets,
        })
        self.stat_collector.scalar('q_loss', q_loss)
        return td_errs

    def get_q(self, states, actions):
        return self.sess.run(self.q, {
            self.state: states,
            self.action: actions,
        })
