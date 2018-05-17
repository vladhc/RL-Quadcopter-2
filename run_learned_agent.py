import tensorflow as tf
from time import sleep

from agents.agent import Agent
from stat_collector import StatCollector
from car_task import Task

task = Task()
stat = StatCollector()

model_file = 'model.ckpt'
# show what agent has learned

with tf.Session() as sess:
    agent = Agent(task, sess, stat)
    tf.train.Saver().restore(sess, model_file)

    done = False
    state = agent.reset_episode() # start a new episode
    total_reward = 0
    steps = 0

    while not done:
        steps += 1
        task.render()
        sleep(0.04)
        action = agent.act(state)
        state, reward, done = task.step(action)
        total_reward += reward

task.close()

print("Total reward:", total_reward)
print("Steps:", steps)
