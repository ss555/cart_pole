import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from actor_critic import _actor_network,_critic_network, OrnsteinUhlenbeckActionNoise


from buffer import Replay_Buffer

import gym
from tensorflow.keras.optimizers import Adam
import tensorflow as tf



env = gym.make('Pendulum-v0')

state_dim = len(env.reset()) #dimension of the states
action_dim = 1  #dimension of the action vector
action_bound_range=1 #action is between -1 and 1
dflt_dtype='float32'
gamma=0.99 # discount factor

observ_min=env.observation_space.low
observ_max=env.observation_space.high



