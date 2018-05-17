import tensorflow as tf
import numpy as np


class Policy():

    def __init__(self, task, sess):
        self.task = task
        self.sess = sess

        self.state_size = task.state_size
        self.action_size = task.action_size

        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.alpha = 1e-3 # learning rate
        self.tau = 1.0 # for soft update of target parameters

        # create policy network
        self.state, self.action_output = self._create_policy_network()
        self.reset_delta_theta_op, self.update_delta_theta_ops, self.update_theta_ops = self._create_update_network()

    def act(self, states):
        actions = self.sess.run(self.action_output, feed_dict={
            self.state: states,
        })
        return actions

    def learn(self, state, score):
        self.sess.run(self.reset_delta_theta_op)
        self.sess.run(self.update_delta_theta_ops, feed_dict={
            self.state: [state],
            self.score: score,
        })
        self.sess.run(self.update_theta_ops)
        return

    def _create_policy_network(self, hidden_units=64):
        state = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
        x = tf.layers.dense(state, hidden_units)
        x = tf.nn.relu(x)

        x = tf.layers.dense(x, hidden_units)
        x = tf.nn.relu(x)

        logits = tf.layers.dense(x, self.action_size)
        output = tf.nn.sigmoid(logits)
        action_output = output * self.action_range + self.action_low
        return state, action_output

    def _create_update_network(self):
        y = tf.log(self.action_output)
        thetas = tf.trainable_variables()
        grad_thetas = tf.gradients(y, thetas)

        self.score = tf.placeholder(tf.float32)

        delta_theta_vars = []
        update_delta_theta_ops = []
        update_theta_ops = []

        for theta, grad_theta in zip(thetas, grad_thetas):
            if grad_theta is not None:
                delta_theta_var = tf.get_variable(
                        "theta_{}_delta".format(len(delta_theta_vars)),
                        shape=theta.get_shape(),
                        initializer=tf.zeros_initializer(),
                        trainable=False)
                delta_theta_vars.append(delta_theta_var)

                delta_theta = grad_theta * self.score * self.alpha
                update_delta_theta_op = tf.assign_add(delta_theta_var, delta_theta)
                update_delta_theta_ops.append(update_delta_theta_op)

                update_theta_op = tf.assign_add(theta, delta_theta_var * self.tau)
                update_theta_ops.append(update_theta_op)

        reset_delta_theta_op = tf.variables_initializer(delta_theta_vars)
        return reset_delta_theta_op, update_delta_theta_ops, update_theta_ops
