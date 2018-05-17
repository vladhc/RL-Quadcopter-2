import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from agents.agent import Agent
from stat_collector import StatCollector
from car_task import Task


num_episodes = 1000
save_model_every = 50
evaluate_every = 10

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
            stat.scalar('episode_steps_train', step)

            action = agent.act(state, explore=True)

            next_state, reward, done = task.step(action)

            assert not np.isnan(reward)

            agent.step(action, reward, next_state, done)
            state = next_state
            total_score += reward

        stat.scalar('episode_reward_train', total_score)

        # Evaluate policy
        if i_episode % evaluate_every == 0:
            step = 0
            total_score = 0
            state = agent.reset_episode() # start a new episode
            done = False
            while not done:
                step += 1
                stat.scalar('episode_steps_eval', step)
                action = agent.act(state)
                state, reward, done = task.step(action)
                total_score += reward
            stat.scalar('episode_reward_eval', total_score)


        print("\rEpisode = {:4d}, score = {:7.3f}".format(i_episode, total_score), end="")
        sys.stdout.flush()

        if i_episode % save_model_every == 0:
            saver.save(sess, "./model.ckpt")
            plt.savefig("./graphs.png")

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
        plt.title('TD error mean')
        plt.plot(*stat.get_history('td_err_mean'))
        plt.grid(True)

        plt.subplot(325)
        plt.title('TD error deviation')
        plt.plot(*stat.get_history('td_err_deviation'))
        plt.grid(True)

        plt.pause(0.05)

env.close()
