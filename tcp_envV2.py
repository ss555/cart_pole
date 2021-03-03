import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import socket
import logging
from logging import info, basicConfig
from collections import deque
logname='TCP_SAC_DEBUG'
basicConfig(filename=logname,
			filemode='w',#'a' for append
			format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
			datefmt='%H:%M:%S',
			level=logging.DEBUG)

# def reward_fn(x,theta,action=0.0):
#     cost=2+np.cos(theta)-abs(x)/50
#     return cost
def reward_fnCosSin(x,costheta,action=0.0):
    cost=1+costheta-abs(x)/25
    return cost
class CartPoleCosSinRPIv2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 pi_conn,
                 seed: int = 0,):
        self.MAX_STEPS_PER_EPISODE = 1000
        self.FORMAT = 'utf-8'
        self.counter=0
        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.37
        self.v_max = 100
        self.w_max = 100
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            1.0,
            1.0,
            self.w_max])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype = np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.conn = pi_conn
        print('connected')
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #send action receive data-old
        self.counter+=1
        assert self.observation_space.contains(self.state), 'obs_err{}'.format(self.state)
        self.conn.sendall(str(action[0]).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        done = bool(state[-1])
        self.state=state[:-1]

        self.state[2]=np.clip(self.state[2],-1,1)
        self.state[3]=np.clip(self.state[3],-1,1)
        x = self.state[0]
        costheta = self.state[2]
        cost = reward_fnCosSin(x, costheta)

        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 5
            print('out of bound')
            self.state[0] = np.clip(x, -self.x_threshold, self.x_threshold)
        if self.MAX_STEPS_PER_EPISODE==self.counter:
            done=True
        info('state: {}, cost{}, done:{}'.format(self.state,cost,done))

        return self.state, cost, done, {}
    def reset(self):

        self.conn.sendall('RESET'.encode(self.FORMAT))

        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        self.state = state[:-1]
        info('reset with nsteps: {}, state:{}'.format(self.counter, self.state))
        self.counter = 0
        print('state {}'.format(self.state))
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleCosSinRPIhistory(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 pi_conn,
                 k_history_len=2,
                 seed: int = 0,):
        self.MAX_STEPS_PER_EPISODE = 400
        self.FORMAT = 'utf-8'
        self.counter=0
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.37
        self.v_max = 100
        self.w_max = 100
        self.k_history_len = k_history_len
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            1.0,
            1.0,
            self.w_max])
        high = np.append(high, np.ones(shape=self.k_history_len))
        self.action_history_buffer = deque(np.zeros(self.k_history_len), maxlen=self.k_history_len)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype = np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.conn = pi_conn
        print('connected')
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #send action receive data-old
        self.counter+=1
        assert self.observation_space.contains(self.state), 'obs_err{}'.format(self.state)
        self.conn.sendall(str(action[0]).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        done = bool(state[-1])
        self.state=state[:-1]
        self.action_history_buffer.append(action[0])

        self.state=np.append(self.state,self.action_history_buffer)
        self.state[2]=np.clip(self.state[2],-1,1)
        self.state[3]=np.clip(self.state[3],-1,1)
        x = self.state[0]
        costheta = self.state[2]
        cost = reward_fnCosSin(x, costheta)

        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 5
            print('out of bound')
            self.state[0] = np.clip(x, -self.x_threshold, self.x_threshold)
        if self.MAX_STEPS_PER_EPISODE==self.counter:
            done=True
        info('state: {}, cost{}, done:{}'.format(self.state,cost,done))

        return self.state, cost, done, {}
    def reset(self):

        self.conn.sendall('RESET'.encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        self.state = state[:-1]
        info('reset with nsteps: {}, state:{}'.format(self.counter, self.state))
        self.counter = 0
        self.state=np.append(self.state,np.zeros(self.k_history_len))
        print('state {}'.format(self.state))
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
