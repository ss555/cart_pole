import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import socket
from collections import deque
from env_custom import reward_fnCos
class CartPoleCosSinRpiDiscrete3(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 pi_conn,
                 seed: int = 0):
        self.MAX_STEPS_PER_EPISODE = 800
        self.FORMAT = 'utf-8'
        self.counter = 0
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        self.v_max = 15
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1.0,
            1.0,
            self.w_max])
        self.action_space = spaces.Discrete(3)
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
        if abs(self.state[4])>100:
            self.state[4]=np.clip(self.state[4],-99.9,99.9)
            print('theta_dot bound from noise')
        assert self.observation_space.contains(self.state), 'obs_err{}'.format(self.state)
        if action==0:
            actionSend=-1.0
        elif action==1:
            actionSend=0.0
        elif action==2:
            actionSend=1.0
        else:
            raise Exception
        self.conn.sendall(str(actionSend).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        done = bool(state[-1])
        self.state=state[:-1]
        self.state[2]=np.clip(self.state[2],-1,1)
        self.state[3]=np.clip(self.state[3],-1,1)
        x = self.state[0]
        costheta = self.state[2]
        cost = reward_fnCos(x, costheta)
        if x <= -self.x_threshold or x >= self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 5
            print('out of bound')
            self.state[0] = np.clip(x, -self.x_threshold, self.x_threshold)
        if abs(self.state[-1]>11):
            print('speed limit')
        if self.MAX_STEPS_PER_EPISODE==self.counter:
            done=True
        # info('state: {}, cost{}, done:{}'.format(self.state,cost,done))
        return self.state, cost, done, {}
    def reset(self):
        self.conn.sendall('RESET'.encode(self.FORMAT))

        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        self.state = state[:-1]
        self.counter = 0
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

###for SAC
class CartPoleCosSinRPIv2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 pi_conn,
                 seed: int = 0,):
        self.MAX_STEPS_PER_EPISODE = 800
        self.FORMAT = 'utf-8'
        self.counter=0
        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.36
        self.v_max = 15
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
        cost = reward_fnCos(x, costheta)

        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 5
            print('out of bound')
            self.state[0] = np.clip(x, -self.x_threshold, self.x_threshold)
        if self.MAX_STEPS_PER_EPISODE==self.counter:
            done=True
        # info('state: {}, cost{}, action:{}'.format(self.state,cost,action))
        return self.state, cost, done, {}
    def reset(self):

        self.conn.sendall('RESET'.encode(self.FORMAT))

        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        self.state = state[:-1]
        # info('reset with nsteps: {}, state:{}'.format(self.counter, self.state))
        self.counter = 0
        print('state {}'.format(self.state))
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

'''
class CartPoleCosSinRpiHistory(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 pi_conn,
                 seed: int = 0):
        self.MAX_STEPS_PER_EPISODE = 800
        self.FORMAT = 'utf-8'
        self.counter = 0
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        self.v_max = 15
        self.w_max = 100
        self.kS=3
        high = np.array([
            self.x_threshold,
            self.v_max,
            1.0,
            1.0,
            self.w_max])
        high = np.hstack((np.tile(high,self.kS), np.tile(1,self.kS)))
        self.actionHistory = deque(np.zeros(self.kS),maxlen=self.kS)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype = np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.conn = pi_conn
        self.prevState=None
        print('connected')
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #send action receive data-old
        self.prevState=np.copy(self.state)

        self.counter+=1
        if abs(self.state[4])>100:
            self.state[4]=np.clip(self.state[4],-99.9,99.9)
            print('theta_dot bound from noise')
        # assert self.observation_space.contains(self.state), 'obs_err{}'.format(self.state)
        if action==0:
            actionSend=-1.0
        elif action==1:
            actionSend=0.0
        elif action==2:
            actionSend=1.0
        else:
            raise Exception
        self.actionHistory.append(action)
        self.conn.sendall(str(actionSend).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        done = bool(state[-1])
        self.state=state[:-1]
        self.state[2]=np.clip(self.state[2],-1,1)
        self.state[3]=np.clip(self.state[3],-1,1)
        x = self.state[0]
        costheta = self.state[2]
        cost = reward_fnCos(x, costheta)

        if x <= -self.x_threshold or x >= self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 5
            print('out of bound')
            self.state[0] = np.clip(x, -self.x_threshold, self.x_threshold)
        if self.MAX_STEPS_PER_EPISODE==self.counter:
            done=True
        info('state: {}, cost{}, action{}'.format(self.state,cost,action))
        # self.state[-1]/=10
        return np.hstack((self.state,self.prevState,self.actionHistory)), cost, done, {}
    def reset(self):
        self.conn.sendall('RESET'.encode(self.FORMAT))
        self.actionHistory = deque(np.zeros(2), maxlen=2)
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        self.state = state[:-1]
        info('reset with nsteps: {}, state:{}'.format(self.counter, self.state))
        self.counter = 0
        self.prevState=self.state
        print('state {}'.format(self.state))
        return np.hstack((self.state,self.prevState,self.actionHistory))

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

##5 actions

class CartPoleCosSinRpiDiscrete(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 pi_conn,
                 seed: int = 0):
        self.MAX_STEPS_PER_EPISODE = 1000
        self.FORMAT = 'utf-8'
        self.counter = 0
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        self.v_max = 15
        self.w_max = 300
        high = np.array([
            self.x_threshold,
            self.v_max,
            1.0,
            1.0,
            self.w_max])
        self.action_space = spaces.Discrete(2)
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
        if action==0:
            actionSend=-1.0
        elif action==1:
            actionSend=1.0
        else:
            print(f'invalid action{action}')
            raise Exception
        self.conn.sendall(str(actionSend).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        done = bool(state[-1])
        self.state=state[:-1]
        self.state[2]=np.clip(self.state[2],-1,1)
        self.state[3]=np.clip(self.state[3],-1,1)
        x = self.state[0]
        costheta = self.state[2]
        cost = reward_fnCos(x, costheta)

        if x <= -self.x_threshold or x >= self.x_threshold:
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
'''
