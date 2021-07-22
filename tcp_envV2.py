import csv
import math
import time
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from env_custom import reward_fnCos
from pendule_pi import PendulePy

class CartPoleCosSinRpiDiscrete3(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 pi_conn,
                 x_threshold: float = 0.355,
                 speed_threshold : int = 14,
                 seed: int = 0):
        self.MAX_STEPS_PER_EPISODE = 800
        self.FORMAT = 'utf-8'
        self.counter = 0
        self.speed_threshold = speed_threshold
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi
        self.x_threshold = x_threshold
        self.v_max = 15
        self.w_max = 30
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
        # if abs(self.state[4])>self.w_max:
        #     self.state[4]=np.clip(self.state[4],-self.w_max,self.w_max)
        #     print('theta_dot bound from noise')
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
        self.state = state[:-1]
        self.state[2] = np.clip(self.state[2],-1,1)
        self.state[3] = np.clip(self.state[3],-1,1)
        x = self.state[0]
        costheta = self.state[2]
        cost = reward_fnCos(x, costheta)
        if x <= -self.x_threshold or x >= self.x_threshold:
            done = True
            cost = cost - self.MAX_STEPS_PER_EPISODE / 5
            print('out of bound')
            self.state[0] = np.clip(x, -self.x_threshold, self.x_threshold)
        if abs(self.state[-1])>self.speed_threshold:
            print('speed limit')
            done = True
        if self.MAX_STEPS_PER_EPISODE<=self.counter:
            done=True
            print('reset because of episode steps')
        # print(self.state)

        # info('state: {}, cost{}, done:{}'.format(self.state,cost,done))
        return self.state, cost, done, {}
    def reset(self):
        self.conn.sendall('RESET'.encode(self.FORMAT))

        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        self.state = state[:-1]
        self.counter = 0
        print(f'reset done{self.state}')
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# class CartPoleClientDiscrete3(gym.Env):
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': 50
#     }
#     def __init__(self,
#                  pi_conn,
#                  seed: int = 0):
#         self.MAX_STEPS_PER_EPISODE = 800
#         self.FORMAT = 'utf-8'
#         self.counter = 0
#         # Angle at which to fail the episode
#         self.theta_threshold_radians = math.pi
#         self.x_threshold = 0.36
#         self.v_max = 15
#         self.w_max = 100
#         high = np.array([
#             self.x_threshold,
#             self.v_max,
#             1.0,
#             1.0,
#             self.w_max])
#         self.action_space = spaces.Discrete(3)
#         self.observation_space = spaces.Box(-high, high, dtype = np.float32)
#         self.seed(seed)
#         self.viewer = None
#         self.state = None
#         self.steps_beyond_done = None
#         self.conn = pi_conn
#         print('connected')
#     def seed(self, seed=0):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#     def step(self, action):
#         #send action receive data-old
#         self.counter+=1
#         if abs(self.state[4])>100:
#             self.state[4]=np.clip(self.state[4],-99.9,99.9)
#             print('theta_dot bound from noise')
#         assert self.observation_space.contains(self.state), 'obs_err{}'.format(self.state)
#         if action==0:
#             actionSend=-1.0
#         elif action==1:
#             actionSend=0.0
#         elif action==2:
#             actionSend=1.0
#         else:
#             raise Exception
#         self.conn.sendall(str(actionSend).encode(self.FORMAT))
#         sData = self.conn.recv(124).decode(self.FORMAT)
#         state = np.array(sData.split(',')).astype(np.float32)
#         done = bool(state[-1])
#         self.state=state[:-1]
#         self.state[2]=np.clip(self.state[2],-1,1)
#         self.state[3]=np.clip(self.state[3],-1,1)
#         x = self.state[0]
#         costheta = self.state[2]
#         cost = reward_fnCos(x, costheta)
#         if x <= -self.x_threshold or x >= self.x_threshold:
#             cost = cost - self.MAX_STEPS_PER_EPISODE / 5
#             print('out of bound')
#             self.state[0] = np.clip(x, -self.x_threshold, self.x_threshold)
#         if abs(self.state[-1]>11):
#             print('speed limit')
#         if self.MAX_STEPS_PER_EPISODE==self.counter:
#             done=True
#         # info('state: {}, cost{}, done:{}'.format(self.state,cost,done))
#         return self.state, cost, done, {}
#     def reset(self):
#         self.conn.sendall('RESET'.encode(self.FORMAT))
#         sData = self.conn.recv(124).decode(self.FORMAT)
#         state = np.array(sData.split(',')).astype(np.float32)
#         self.state = state[:-1]
#         self.counter = 0
#         return self.state
#
#     def render(self, mode='human'):
#         pass
#
#     def close(self):
#         if self.viewer:
#             self.viewer.close()
#             self.viewer = None

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

class CartPoleZmq(gym.Env):
    def __init__(self,
                 pendulePy: PendulePy,
                 MAX_STEPS_PER_EPISODE: int=800,
                 max_pwm=130,
                 discreteActions=True,
                 x_threshold: float= 0.34,
                 Te=0.05,
                 theta_dot_threshold_init:float=13,
                 monitor_filename:str = 'monitor.csv',
                 seed: int = 0):
        self.MAX_STEPS_PER_EPISODE = MAX_STEPS_PER_EPISODE
        self.pendulum=pendulePy
        self.max_pwm=max_pwm
        self.theta_dot_threshold_init=theta_dot_threshold_init
        self.Te=Te
        self.counter = 0
        self.discreteActions=discreteActions
        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = x_threshold
        self.v_max = 15
        self.w_max = 100
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            1.0,
            1.0,
            self.w_max])
        if self.discreteActions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.state = None
        print('connected')
        self.start_time=time.time()
        self.file_handler = open(monitor_filename,'wt')
        self.writer = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t"))
        self.writer.writeheader()
        self.file_handler.flush()

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # send action receive data-old
        try:
            if not self.discreteActions:
                self.pendulum.sendCommand(action*self.max_pwm)
            else:
                if action==0:
                    self.pendulum.sendCommand(-self.max_pwm)
                elif action==1:
                    self.pendulum.sendCommand(0)
                elif action==2:
                    self.pendulum.sendCommand(self.max_pwm)
            self.counter += 1
            self.pendulum.readState(blocking=True)#wait 1 step 25ms
            self.pendulum.readState(blocking=True)#50ms control
            # print(time.time()-self.prev_time)
            # if time.time()-self.prev_time>0.06:
            #     print('alert')
            # self.prev_time = time.time()
            angle=self.pendulum.angle
            costheta = np.cos(angle)
            self.state = [self.pendulum.position, self.pendulum.linvel, costheta,
                          np.sin(angle), self.pendulum.angvel]

            cost = reward_fnCos(self.pendulum.position, costheta)

            done = False
            if self.state[0] < -self.x_threshold or self.state[0] > self.x_threshold:
                cost = cost - self.MAX_STEPS_PER_EPISODE / 5
                print('out of bound')
                self.state[0] = np.clip(self.state[0], -self.x_threshold, self.x_threshold)
                done = True
            elif self.MAX_STEPS_PER_EPISODE <= self.counter:
                done = True
            elif abs(self.state[-1])>self.theta_dot_threshold_init:
                done=True
                print(f'theta_dot_limit {self.state[-1]}')
            self.rewards.append(cost)
            if done:
                ep_rew=np.sum(self.rewards)
                ep_info={"r": round(ep_rew, 6), "l": self.counter, "t": round(time.time() - self.start_time,6)}
                self.writer.writerow(ep_info)
        except:
            self.pendulum.sendCommand(0)
            print('error in step')
            raise ValueError
        # info('state: {}, cost{}, action:{}'.format(self.state,cost,action))
        return self.state, cost, done, {}

    def reset(self):
        # self.prev_time = time.time()

        self.pendulum.readState(blocking=True)
        if self.pendulum.position>0:
            while self.pendulum.position>0:
                self.pendulum.sendCommand(-70)
                self.pendulum.readState(blocking=True)
        else:
            while self.pendulum.position<0:
                self.pendulum.sendCommand(70)
                self.pendulum.readState(blocking=True)

        self.pendulum.sendCommand(0)
        self.pendulum.readState(blocking=True)
        angle=self.pendulum.angle
        while np.cos(angle)<0.995 or abs(self.pendulum.angvel)>0.02:
            time.sleep(5)
            self.pendulum.readState(blocking=True)
            angle = self.pendulum.angle
        self.state = [self.pendulum.position,self.pendulum.linvel,np.cos(angle),np.sin(angle),self.pendulum.angvel]
        self.counter = 0
        self.rewards = []
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        self.file_handler.close()

