import tensorflow as tf
import logging

class QNetwork():

    def __init__(self, sess, state_size, action_size, stat_collector, name='critic', hidden_units=64):
        self.sess = sess
        self.stat_collector = stat_collector
        self.name = name

        with tf.variable_scope(name):
            # create network
            self.state = tf.placeholder(tf.float32, shape=(None, state_size), name='state')
            self.action = tf.placeholder(tf.float32, shape=(None, action_size), name='action')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            # state network
            x = self.state
            x = tf.layers.dense(x, hidden_units)
            x = tf.nn.leaky_relu(x)

            # x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.layers.dense(x, hidden_units)
            state_out = tf.nn.leaky_relu(x)

            # action network
            x = self.action
            x = tf.layers.dense(x, hidden_units)
            x = tf.nn.leaky_relu(x)

            # x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.layers.dense(x, hidden_units)
            action_out = tf.nn.leaky_relu(x)

            # state + action → Q
            x = state_out + action_out
            x = tf.nn.leaky_relu(x)
            # x = tf.concat([state_out, action_out], 1)

            # x = tf.layers.batch_normalization(x, training=self.is_training)
            # x = tf.layers.dense(x, hidden_units)

            self.q = tf.layers.dense(x, 1)
            self.action_gradients = tf.gradients(self.q, self.action)

            self.q_target = tf.placeholder(tf.float32, shape=(None, 1), name='q_target')
            self.td_err = tf.squeeze(self.q_target - self.q)
            self.q_loss = tf.losses.mean_squared_error(self.q_target, self.q)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        thetas = tf.trainable_variables(scope=self.name)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer().minimize(self.q_loss, var_list=thetas)

    def learn(self, states, actions, q_targets):
        _, q_loss, td_errs = self.sess.run([self.train_op, self.q_loss, self.td_err], {
            self.state: states,
            self.action: actions,
            self.q_target: q_targets,
            self.is_training: True,
        })
        self.stat_collector.scalar('q_loss', q_loss)
        return td_errs

    def get_action_gradients(self, states, actions):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state: states,
            self.action: actions,
            self.is_training: False,
        })[0]

    def get_q(self, states, actions):
        return self.sess.run(self.q, feed_dict={
            self.state: states,
            self.action: actions,
            self.is_training: False,
        })
