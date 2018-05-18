import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import sleep


def run_episode(sess, agent, task, train=False, render=False):
    total_score = 0
    state = agent.reset_episode() # start a new episode
    done = False
    steps = 0

    while not done:
        steps += 1
        action = agent.act(state, explore=train)
        next_state, reward, done = task.step(action)
        assert not np.isnan(reward)

        if train:
            agent.step(action, reward, next_state, done)

        if render:
            task.render()
            sleep(0.04)

        state = next_state
        total_score += reward

    return total_score, steps


def _plot_smoothed(stat, metric, color='C0', label=None):
    x, y = stat.get_history(metric)
    plt.plot(x, y, alpha=0.3, color=color)
    df = pd.DataFrame(data={'y': list(y)}, index=x)
    w = df.rolling(8).mean()
    plt.plot(df.index, w, label=label, color=color)


def plot_training_graphs(stat):

    plt.clf()

    plt.subplots_adjust(top=.9, bottom=0.1, wspace=0.2, hspace=0.2)

    plt.subplot(321)
    plt.title('Episode reward')
    _plot_smoothed(stat, 'episode_reward_train', label='Train')
    plt.plot(*stat.get_history('episode_reward_eval'), label='Eval', color='C1')
    plt.grid(True)
    plt.legend()

    plt.subplot(322)
    plt.title('Advantage Loss')
    _plot_smoothed(stat, 'advantage_loss')
    plt.grid(True)

    plt.subplot(323)
    plt.title('Episode Steps')
    _plot_smoothed(stat, 'episode_steps_train', label='Train')
    plt.plot(*stat.get_history('episode_steps_eval'), label='Eval', color='C1')
    plt.grid(True)
    plt.legend()

    plt.subplot(324)
    plt.title('V Loss')
    _plot_smoothed(stat, 'v_loss')
    plt.grid(True)

    plt.subplot(325)
    plt.title('ReplayBuffer TD error mean')
    _plot_smoothed(stat, 'td_err_mean')
    plt.grid(True)

    plt.subplot(326)
    plt.title('Actor loss')
    _plot_smoothed(stat, 'actor_loss')
    plt.grid(True)
