import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import socket

def reward_fn(x,theta,action=0.0):
    cost=2+np.cos(theta)-abs(x)/50
    if theta<math.pi/10 or theta>math.pi*1.9:
        cost+=3
    return cost


class CartPoleRPI(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 pi_conn,
                 seed: int = 0,):
        self.MAX_STEPS_PER_EPISODE = 3000
        self.FORMAT = 'utf-8'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        # self.theta_threshold_radians = 20 * 2 * math.pi / 360
        self.x_threshold = 5.0
        self.v_max = 50
        self.w_max = 50
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 2,
            self.w_max,
            1])
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
        #send action receive data
        assert self.observation_space.contains(self.state), 'obs_err'
        self.conn.sendall(str(action[0]).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        done = bool(state[4])
        self.state=state[:-1]

        x = self.state[0]
        theta = self.state[2]
        cost = reward_fn(x, theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost-100
        return self.state, cost, done, {}
    def reset(self):
        self.conn.sendall('RESET'.encode(self.FORMAT))
        print('reset')
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        self.state = state[:-1]
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
