# Environment
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
# torch.set_default_tensor_type("torch.cuda.FloatTensor")
from tqdm import tqdm
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
import pickle
# IO
from pathlib import Path
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import SAC, DQN
from src.utils import read_hyperparameters
from src.env_custom import CartPoleButter, CartPoleNN
from src.custom_callbacks import ProgressBarManager, EvalCustomCallback
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

env = CartPoleButter(discreteActions=False)
from stable_baselines3.common.save_util import load_from_pkl

# 2 ways to collect transitions randomly or from exp replay before
LOAD_BUFFER_PATH = "./weights/sac50-sim/sac_swingup_simulation.pkl"
# LOAD_BUFFER_PATH = "./weights/dqn/dqn_pi_swingup_bufferN"
buffer = load_from_pkl(LOAD_BUFFER_PATH)
# data = collect_transitions(env, size=10000)
def transitions_from_replay_buffer(buffer):
    data = []
    for i in range(buffer.actions.shape[0]):
        if np.any(buffer.observations[i].squeeze(0)!=[0., 0., 0., 0., 0.]) or np.any(buffer.next_observations[-1].squeeze(0)!=[0., 0., 0., 0., 0.]):
            data.append(transition(torch.Tensor(buffer.observations[i].squeeze(0)),
                                   torch.Tensor(buffer.actions[i].squeeze(0)),
                                   torch.Tensor(buffer.next_observations[i].squeeze(0))))
    print(f'num of tranitions: {len(data)}')
    return data
data = transitions_from_replay_buffer(buffer)


class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dt=0.05):
        super().__init__()
        self.STATE_X = 0
        self.STATE_Y = 1

        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size, B_size = state_size * state_size, state_size * action_size
        self.A1 = nn.Linear(state_size + action_size, hidden_size)
        self.A2 = nn.Linear(hidden_size, hidden_size)
        self.A3 = nn.Linear(hidden_size, A_size)
        self.B1 = nn.Linear(state_size + action_size, hidden_size)
        self.B2 = nn.Linear(hidden_size, hidden_size)
        self.B3 = nn.Linear(hidden_size, B_size)

    def forward(self, x, u):
        '''
            predict x(t+1)=f(x(t),u(t))
            x: a batch of states
            u: a batch of actions
        '''

        xu = torch.cat((x, u), -1)
        A = self.A3(F.relu(self.A2(F.relu(self.A1(xu)))))
        A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
        B = self.B3(F.relu(self.B2(F.relu(self.B1(xu)))))
        B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        return x + (A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)).squeeze() * self.dt

# class DynamicsModel(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size, dt=0.05):
#         super().__init__()
#         self.STATE_X = 0
#         self.STATE_Y = 1
#
#         self.state_size, self.action_size, self.dt = state_size, action_size, dt
#         A_size, B_size = state_size * state_size, state_size * action_size
#         self.A1 = nn.Linear(state_size + action_size, hidden_size)
#         self.A2 = nn.Linear(hidden_size, A_size)
#         self.B1 = nn.Linear(state_size + action_size, hidden_size)
#         self.B2 = nn.Linear(hidden_size, B_size)
#
#     def forward(self, x, u):
#         '''
#             predict x(t+1)=f(x(t),u(t))
#             x: a batch of states
#             u: a batch of actions
#         '''
#
#         xu = torch.cat((x, u), -1)
#         A = self.A2(F.relu(self.A1(xu)))
#         A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
#         B = self.B2(F.relu(self.B1(xu)))
#         B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
#         return x + (A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)).squeeze() * self.dt


# print(f'{env.action_space,env.observation_space}')
dynamics = DynamicsModel(state_size=env.observation_space.shape[0],
                         action_size=env.action_space.shape[0],
                         hidden_size=512, dt=0.05)
from torch.optim.lr_scheduler import ReduceLROnPlateau
optimizer = torch.optim.Adam(dynamics.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, patience=20, min_lr = 5e-4)

train_ratio = 0.7
np.random.shuffle(data)
train_data, validation_data = data[:int(len(data)*train_ratio)], data[int(len(data)*train_ratio):]

def compute_loss(model,data_t,loss_func=torch.nn.MSELoss()):
    states, actions, next_states=data_t
    next_states_predicted = model(states,actions)
    return loss_func(next_states,next_states_predicted)

def transpose_batch(batch):
    return transition(*map(torch.stack,zip(*batch)))

def train(model, train_data, validation_data, epochs=2000):
    train_data_t = transpose_batch(train_data)
    validation_data_t = transpose_batch(validation_data)
    losses = np.full((epochs,2),np.nan)
    for epoch in tqdm(range(epochs),desc='epochs of learning dynamics'):
        loss = compute_loss(model, train_data_t)
        validation_loss = compute_loss(model,validation_data_t)
        losses[epoch] = [loss.detach().numpy(), validation_loss.detach().numpy()]
        if epoch%100==0:
            print(f'loss: {loss}, validation loss is {validation_loss}')
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(validation_loss)
    plt.plot(losses)
    plt.yscale('log')
    plt.legend(['traning','validation'])
    plt.savefig('./model_learning/loss.pdf')
    plt.show()
    #     return model
    # model =
TRAIN_MODEL = False#True
path = './model_learning/cartpole_nn_2layers'
if TRAIN_MODEL:
    train(dynamics, train_data, validation_data)
    # with open(path, 'wb') as f:
    #     pickle.dump(dynamics, f)
    # torch.save(dynamics, path)
    torch.save(dynamics.state_dict(), path)
else:
    dynamics = DynamicsModel(state_size=env.observation_space.shape[0],
                         action_size=env.action_space.shape[0],
                         hidden_size=512, dt=0.05)
    dynamics.load_state_dict(torch.load(path))
    dynamics.eval()
    # with open('./model_learning/cartpole_nn.pickle', 'rb') as f:
    #     dynamics = pickle.load(f)


def train_agent(dynamics_model, discreteActions = True):
    virtual_env = CartPoleNN(dynamics, discreteActions=discreteActions)
    eval_env = CartPoleButter(discreteActions=discreteActions)
    eval_callback = EvalCustomCallback(eval_env=eval_env, eval_freq=5000)
    if not discreteActions:
        params = read_hyperparameters('sac_cartpole_50')
        agent = SAC(**params, env = virtual_env)
        # agent = SAC(**params,env = virtual_env)
    else:
        params = read_hyperparameters('dqn_cartpole_50')
        agent = DQN(**params, env = virtual_env)


    STEPS_TO_TRAIN = 5e5
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        agent.learn(total_timesteps=150000, callback=[cus_callback, eval_callback])

train_agent(dynamics, discreteActions = True)