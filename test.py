import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from agents.agent import Agent
from stat_collector import StatCollector
from car_task import Task


num_episodes = 1000

task = Task()
stat = StatCollector()

tf.reset_default_graph()


with tf.Session() as sess:
    agent = Agent(task, sess, stat)
    saver = tf.train.Saver()

    for i_episode in range(1, num_episodes+1):
        stat.tick()

        # Evaluate policy
        total_score_eval = 0
        state = agent.reset_episode() # start a new episode
        done = False
        step = 0

        while not done:
            step += 1
            action = agent.act(state)
            state, reward, done = task.step(action)
            total_score_eval += reward
        stat.scalar('episode_steps_eval', step)
        stat.scalar('episode_reward_eval', total_score_eval)

        # Train policy and Q-Network
        total_score_train = 0
        state = agent.reset_episode() # start a new episode
        done = False
        step = 0

        while not done:
            step += 1
            action = agent.act(state, explore=True)
            next_state, reward, done = task.step(action)

            assert not np.isnan(reward)

            agent.step(action, reward, next_state, done)
            state = next_state
            total_score_train += reward

        stat.scalar('episode_steps_train', step)
        stat.scalar('episode_reward_train', total_score_train)

        print('Episode = {:4d}, score_train = {:7.3f}, score_test = {:7.3f}'.format(i_episode, total_score_train, total_score_eval))
        sys.stdout.flush()

        plt.clf()

        plt.subplot(321)
        plt.title('Episode reward')
        plt.plot(*stat.get_history('episode_reward_train'), label='Train')
        plt.plot(*stat.get_history('episode_reward_eval'), label='Eval')
        plt.grid(True)
        plt.legend()

        plt.subplot(322)
        plt.title('Q Loss')
        plt.plot(*stat.get_history('q_loss'))
        plt.grid(True)

        plt.subplot(323)
        plt.title('Steps')
        plt.plot(*stat.get_history('episode_steps_train'), label='Train')
        plt.plot(*stat.get_history('episode_steps_eval'), label='Eval')
        plt.grid(True)
        plt.legend()

        plt.subplot(324)
        plt.title('ReplayBuffer TD error mean')
        plt.plot(*stat.get_history('td_err_mean'))
        plt.grid(True)

        plt.subplot(325)
        plt.title('ReplayBuffer TD error deviation')
        plt.plot(*stat.get_history('td_err_deviation'))
        plt.grid(True)

        plt.pause(0.05)

        saver.save(sess, "./model.ckpt")
        plt.savefig("./graphs.png")

env.close()
