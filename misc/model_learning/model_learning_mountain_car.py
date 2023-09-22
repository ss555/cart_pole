# Environment
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
# torch.set_default_tensor_type("torch.cuda.FloatTensor")

# Visualization
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import trange
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import Monitor
import base64
import os
import math
# IO
from pathlib import Path
from gym import spaces
from stable_baselines3 import SAC
from src.utils import read_hyperparameters

transition = namedtuple('transition',['state','action','next_state'])

def collect_transitions(env, size=1000, action_repeat=2):
    done = False
    data = []
    for _ in range(size):
        action = env.action_space.sample()
        obs = env.reset()
        for _ in range(action_repeat):
            prev_obs = env.reset() if done else obs
            obs, rew, done, _ = env.step(action)
#             print(action)
            data.append(transition(torch.Tensor(prev_obs),
                                   torch.Tensor([float(action)]),
                                   torch.Tensor(obs)))
    return data

env = gym.make("MountainCarContinuous-v0")
data = collect_transitions(env, size=10000)


class DynamicsModel(nn.Module):
    STATE_X = 0
    STATE_Y = 1

    def __init__(self, state_size, action_size, hidden_size, dt=0.05):
        super().__init__()
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size, B_size = state_size * state_size, state_size * action_size
        self.A1 = nn.Linear(state_size + action_size, hidden_size)
        self.A2 = nn.Linear(hidden_size, A_size)
        self.B1 = nn.Linear(state_size + action_size, hidden_size)
        self.B2 = nn.Linear(hidden_size, B_size)

    def forward(self, x, u):
        '''
            predict x(t+1)=f(x(t),u(t))
            x: a batch of states
            u: a batch of actions
        '''

        xu = torch.cat((x, u), -1)
        A = self.A2(F.relu(self.A1(xu)))
        A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
        B = self.B2(F.relu(self.B1(xu)))
        B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        return x + (A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)).squeeze() * self.dt


# print(f'{env.action_space,env.observation_space}')
dynamics = DynamicsModel(state_size=env.observation_space.shape[0],
                         action_size=env.action_space.shape[0], hidden_size=32, dt=0.05)

optimizer = torch.optim.Adam(dynamics.parameters(), lr=0.01)

train_ratio = 0.7
train_data, validation_data = data[:int(len(data)*train_ratio)], data[int(len(data)*train_ratio):]

def compute_loss(model,data_t,loss_func=torch.nn.MSELoss()):
    states, actions, next_states=data_t
    next_states_predicted = model(states,actions)
    return loss_func(next_states,next_states_predicted)

def transpose_batch(batch):
    return transition(*map(torch.stack,zip(*batch)))

def train(model, train_data, validation_data, epochs=1000):
    train_data_t = transpose_batch(train_data)
    validation_data_t = transpose_batch(validation_data)
    losses = np.full((epochs,2),np.nan)
    for epoch in range(epochs):
        loss = compute_loss(model, train_data_t)
        validation_loss = compute_loss(model,validation_data_t)
        losses[epoch] = [loss.detach().numpy(), validation_loss.detach().numpy()]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plt.plot(losses)
    plt.yscale('log')
    plt.legend(['traning','validation'])
    plt.show()
#     return model
# model =
train(dynamics, train_data, validation_data)
from gym.utils import seeding
class vir_env(gym.Env):
    def __init__(self, model):
        self.model = model#.detach()
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )
        self.viewer = None

        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        # self.observation_space = spaces.Box(-1.2000000476837158, 0.6000000238418579, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.seed()

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
    def step(self, action):
        next_state = self.model(torch.Tensor(self.state).unsqueeze(0), torch.Tensor(action).unsqueeze(0))#DoubleTensor
        # next_state = self.model(torch.Tensor(self.state).unsqueeze(0),torch.Tensor(action).unsqueeze(0))  # DoubleTensor
        self.state = next_state.detach().numpy().squeeze(0)
        reward = 0
        position, velocity = self.state[0], self.state[1]
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position and velocity >= 0.0)
        if done:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([np.random.uniform(low=-0.6, high=0.4), 0])
        return self.state

def predict_traj(state, actions, model, action_repeat=1):
    n_s = []
    for action in actions:
        for _ in action_repeat:
            state = model(state,action)
            n_s.append(state)
    return torch.stack(n_s, dim=0)

from src.custom_callbacks import ProgressBarManager, EvalCustomCallback
def train_agent(dynamics_model):
    virtual_env = vir_env(dynamics_model)
    eval_env = gym.make("MountainCarContinuous-v0")
    eval_callback = EvalCustomCallback(eval_env=eval_env, eval_freq = 5000)


    params = read_hyperparameters('MountainCarContinuous-v0', './hyperparams/sac.yml')
    agent = SAC(**params,env = eval_env)
    # agent = SAC(**params,env = virtual_env)
    STEPS_TO_TRAIN = 5e5
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        agent.learn(total_timesteps=50000, callback=[cus_callback, eval_callback])
train_agent(dynamics)