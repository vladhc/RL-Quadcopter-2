import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from agents.agent import Agent
from stat_collector import StatCollector
from car_task import Task


num_episodes = 1000
save_model_every = 50

task = Task()
stat = StatCollector()

tf.reset_default_graph()


with tf.Session() as sess:
    agent = Agent(task, sess, stat)
    saver = tf.train.Saver()

    for i_episode in range(1, num_episodes+1):
        stat.tick()
        total_score = 0
        
        state = agent.reset_episode() # start a new episode
        done = False
        step = 0

        while not done:
            step += 1
            stat.scalar('episode_steps', step)

            action = agent.act(state)
            assert np.all(action >= task.action_low), "expected less than {:7.3f}, but was {}".format(task.action_low, action)
            assert np.all(action <= task.action_high)

            next_state, reward, done = task.step(action)

            assert not np.isnan(reward)

            agent.step(action, reward, next_state, done)
            state = next_state
            total_score += reward

        if i_episode % save_model_every == 0:
            saver.save(sess, "./model-{}.ckpt".format(i_episode))
            plt.savefig("./graphs.png")

        stat.scalar('episode_reward', total_score)

        best_score = np.max(list(stat.get_history('episode_reward')))
        print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(i_episode, total_score, best_score), end="")
        sys.stdout.flush()

        plt.clf()

        plt.subplot(321)
        plt.plot(stat.get_history('episode_reward'))
        plt.ylabel('episode_reward')

        plt.subplot(322)
        plt.plot(stat.get_history('q_loss'))
        plt.ylabel('q_loss')

        plt.subplot(323)
        plt.plot(stat.get_history('episode_steps'))
        plt.ylabel('episode_steps')

        plt.subplot(324)
        plt.plot(stat.get_history('td_err_mean'))
        plt.ylabel('td_err_mean')

        plt.subplot(325)
        plt.plot(stat.get_history('td_err_deviation'))
        plt.ylabel('td_err_deviation')

        plt.pause(0.05)

env.close()
