# Deep RL Quadcopter Controller

In this project, I'm trying to design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm.

## Features

* Dueling Q-Network as an Actor Critic;
* Double Q-Learning;
* Actor is a 3 layer network with some dropout and leaky relu;
* Replay Buffer with priority replay. TD Error determines the priority. During training the priority becomes less important and Buffer starts to sample more randomly. Also Q-Network doesn't take into account that some samples are fetched more often;
* Statistics visualization;
* Model is saved and agent behaviour can be replayed in the environment;
* Pure TensorFlow implementation. Because we can.

## Metrics

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
