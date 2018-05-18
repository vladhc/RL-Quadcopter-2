import tensorflow as tf

from agents.agent import Agent
from stat_collector import StatCollector
from task import Task
from utils import run_episode


model_file = 'model.ckpt'

# show what agent has learned
with tf.Session() as sess:
    task = Task()
    agent = Agent(task, sess, StatCollector())
    tf.train.Saver().restore(sess, model_file)
    while True:
        task.reset()
        score, steps = run_episode(sess, agent, task, train=False, render=True)

        print("Total reward:", score)
        print("Steps:", steps)

task.close()
