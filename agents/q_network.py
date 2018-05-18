import tensorflow as tf
import logging

class QNetwork():

    def __init__(self, sess, task, stat_collector, name='critic', hidden_units=64, dropout_rate=0.2):
        self.sess = sess
        self.stat_collector = stat_collector
        self.name = name

        action_range = task.action_high - task.action_low
        action_med = (task.action_low + task.action_high) / 2

        # Create network
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, shape=(None, task.state_size), name='state')
            self.action = tf.placeholder(tf.float32, shape=(None, task.action_size), name='action')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            # State stream
            x = tf.layers.dense(self.state, hidden_units)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.dropout(x, training=self.is_training, rate=dropout_rate)

            x = tf.layers.dense(x, hidden_units)
            state_out = tf.nn.leaky_relu(x)

            # Action stream
            x = (self.action - action_med) / action_range # normalization
            x = tf.layers.dense(x, hidden_units)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.dropout(x, training=self.is_training, rate=dropout_rate)

            x = tf.layers.dense(x, hidden_units)
            action_out = tf.nn.leaky_relu(x)

            # State stream → V
            x = state_out
            x = tf.layers.dense(x, hidden_units)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.dropout(x, training=self.is_training, rate=dropout_rate)

            self.v = tf.layers.dense(x, 1)

            # State stream and Action stream → Advantage
            x = tf.concat([state_out, action_out], 1)

            x = tf.layers.dense(x, hidden_units)
            x = tf.nn.leaky_relu(x)

            self.advantage = tf.layers.dense(x, 1)

            # Q = V + advantage
            self.q = self.v + self.advantage

            self.action_gradients = tf.gradients(self.advantage, self.action)

            self.q_target = tf.placeholder(
                    tf.float32,
                    shape=(None, 1),
                    name='q_target')
            self.v_target = tf.placeholder(
                    tf.float32,
                    shape=(None, 1),
                    name='v_target')
            self.advantage_target = self.q_target - self.v_target

            self.td_err = tf.squeeze(self.q_target - self.q)
            self.v_loss = tf.losses.mean_squared_error(self.v_target, self.v)
            self.q_loss = tf.losses.mean_squared_error(self.q_target, self.q)
            self.advantage_loss = tf.losses.mean_squared_error(self.advantage_target, self.advantage)

        thetas = tf.trainable_variables(scope=self.name)
        self.train_op = tf.train.AdamOptimizer().minimize(
                    self.q_loss,
                    var_list=thetas)

    def learn(self, states, actions, q_targets, v_targets):
        _, advantage_loss, q_loss, v_loss, td_errs = self.sess.run(
                [self.train_op, self.advantage_loss, self.q_loss, self.v_loss, self.td_err],
                feed_dict={
                    self.state: states,
                    self.action: actions,
                    self.q_target: q_targets,
                    self.v_target: v_targets,
                    self.is_training: True,
                })
        self.stat_collector.scalar('advantage_loss', advantage_loss)
        self.stat_collector.scalar('v_loss', v_loss)
        self.stat_collector.scalar('q_loss', q_loss)
        return td_errs

    def get_td_err(self, states, actions, q_targets):
        td_errs = self.sess.run(
                self.td_err,
                feed_dict={
                    self.state: states,
                    self.action: actions,
                    self.q_target: q_targets,
                    self.is_training: False,
                    })
        return td_errs

    def get_action_gradients(self, states, actions):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state: states,
            self.action: actions,
            self.is_training: False,
        })[0]

    def get_q_and_v(self, states, actions):
        return self.sess.run([self.q, self.v], feed_dict={
            self.state: states,
            self.action: actions,
            self.is_training: False,
        })
