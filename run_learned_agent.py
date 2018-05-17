import sys
import tensorflow as tf
from time import sleep

from agents.agent import Agent
from stat_collector import StatCollector
from car_task import Task

task = Task()
stat = StatCollector()

model_file = sys.argv[1]
# show what agent has learned

with tf.Session() as sess:
    agent = Agent(task, sess, stat)
    tf.train.Saver().restore(sess, model_file)

    done = False
    state = agent.reset_episode() # start a new episode

    while not done:
        task.render()
        sleep(0.01)
        action = agent.act(state)
        _, _, done = task.step(action) 
        if done:
            break


task.close()
