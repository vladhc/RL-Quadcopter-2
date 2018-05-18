# Deep RL Quadcopter Controller

In this project, I'm trying to design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm.
Current version has simplified task - learn `MountainCarContinuous-v0` task from ai-gym.

## Features

* Dueling Q-Network as an Actor Critic;
* Policy as an Actor. It's simple 3 layer network with some dropout and leaky relu;
* Replay Buffer with priority replay. TD Error determines the priority. During training the priority becomes less important and Buffer starts to sample more randomly. Also Q-Network doesn't take into account that some samples are fetched more often;
* Statistics visualization;
* Model is saved and can be later used to see how agent behaves in the environment;
* Pure TensorFlow implementation. Because we can.

## Metrics

On this graph it can be easily seen, that removing priority sampling from the Replay Buffer isn't a good idea. In the beginning of the training the samples were taken by priority. After 100 episode they should be taken 100% random. But event on the 80th episode it's clear, that the training process wouldn't converge soon.

![Metrics](./graphs.png)

## Project Instructions

To train agent:
```
python train.py
```

To have a look how the learned agent plays:
```
python run_agent.py
```
