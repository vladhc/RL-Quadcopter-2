import gym
import numpy as np
from time import sleep
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from agents.agent import Agent
from stat_collector import StatCollector

# More simple task for testing the Agent

class Task():

    def __init__(self):
        env = gym.make('MountainCarContinuous-v0')
        self.env = env

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.env.reset()
        return state


num_episodes = 1000
max_steps_per_episode = 200
save_model_every = 50

task = Task()

tf.reset_default_graph()

stat = StatCollector()


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

            if step >= max_steps_per_episode:
                done = True

        if i_episode % save_model_every == 0:
            saver.save(sess, "./model-{}.ckpt".format(step))
            plt.savefig("./graphs.png")

        stat.scalar('episode_reward', total_score)

        best_score = np.max(list(stat.get_history('episode_reward')))
        print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(i_episode, total_score, best_score), end="")
        sys.stdout.flush()

        plt.clf()

        plt.subplot(311)
        plt.plot(stat.get_history('episode_reward'))
        plt.ylabel('episode_reward')

        plt.subplot(312)
        plt.plot(stat.get_history('q_loss'))
        plt.ylabel('q_loss')

        plt.subplot(313)
        plt.plot(stat.get_history('episode_steps'))
        plt.ylabel('episode_steps')

        plt.pause(0.05)

    # show what agent has learned
    state = env.reset()
    for _ in range(1000):
        env.render()
        sleep(0.01)
        action = agent.act(state)
        _, _, done, _ = env.step(action) 
        if done:
            break

env.close()
