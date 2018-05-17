import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from agents.agent import Agent
from stat_collector import StatCollector
from car_task import Task
from utils import run_episode, plot_training_graphs


# Params
num_episodes = 1000
evaluate_every = 10
model_file = './model.ckpt'

with tf.Session() as sess:

    # Setup
    task = Task()
    stat = StatCollector()
    agent = Agent(task, sess, stat)
    saver = tf.train.Saver()

    # Run Training
    for i_episode in range(num_episodes):
        stat.tick()

        # Train policy and Q-Network
        score, steps = run_episode(sess, agent, task, train=True)
        stat.scalar('episode_steps_train', steps)
        stat.scalar('episode_reward_train', score)
        print('Episode = {:4d}, score train = {:7.3f}'.format(i_episode, score))

        # Evaluate policy
        if i_episode % evaluate_every == 0:
            score, steps = run_episode(sess, agent, task, train=False)
            stat.scalar('episode_steps_eval', steps)
            stat.scalar('episode_reward_eval', score)
            print('Episode = {:4d},  score eval = {:7.3f}'.format(i_episode, score))
            saver.save(sess, model_file)

        plot_training_graphs(stat)
        plt.pause(0.05)
        plt.savefig("./graphs.png")

task.close()
