import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
from collections import deque
from scipy.integrate import odeint
#PWM 180
[A,B,C,D]=[-19.21325391606273, 0.9619973845421196, -0.6332291217311722, 0.0]
# A=-19.38469711774052#-19.9765`38106642725
# B= 0.9712177675306626#1.0287320880446733
# C= -0.6447664688338154#-0.9326363456754534
#PWM
# A=-19.976538106642725
# B= 1.0287320880446733
# C= -0.9326363456754534
#pwm_offset=2
wFiltered=4.606770906465107
wAngular=4.6124220495309425#from noisy data##4.606770906465107#filtered##
#all pwm (-19.976538106642725, 1.0287320880446733, -0.9326363456754534, 0.035395644087165744)
def reward_fnCos(x, costheta, theta_dot=0,Kx=5):
    reward = 1+costheta-x**2-1*(abs(theta_dot)>10)
    #-abs(theta_dot)*0.05

    #produit de 2: 1+costheta*(1-x**4)
    #(1+costhetha)-K*(1+costheta)*(x)**4
    '''
    if abs(theta)<0.1:
        reward = 1+costheta-x**4*Kx
    else:
        reward = 1+costheta
    '''
    return reward
def _action_static_friction(action, threshold=0.083):
    if abs(action)<threshold:
        return 0
    if action>0:
        return action-threshold
    else:
        return action+threshold
length=0.48#center of mass
N_STEPS=800
# #TEST#
K1 = 18.8 #30#Guil  #dynamic friction
K2 = 0.099#0.15 #2 #0.1  #friction on theta
Mpoletest = 0.2
McartTest = 0.6
Mcart=0.44
# Mpole=0.03#0.05#s
Mpole=0.06#6#0.05#s
#Applied_force=5.6#REAL for 180pwm
Applied_force=5.6#5.6 #6 or 2.2(a*m)

class CartPoleContinous(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 Te=0.05,
                 randomReset=False,
                 f_a=A,
                 f_b=B,
                 f_c=C,
                 f_d=D,
                 integrator="semi-euler",
                 tensionMax=8.4706,
                 seed=0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.g = 9.806
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.4611167818372032  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = integrator  # 'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1  # 05
        self.wAngularIni = wAngular  # 4.488 #T=1.4285, w=
        self.reward = None
        self.randomReset = randomReset
        self.fa=f_a
        self.fb=f_b
        self.fc=f_c
        self.fd=f_d
        self.tensionMax=tensionMax
        self.arr=[]
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def _calculate_force(self,action):
        f=self.masscart*(self.fa*self.state[1]+self.fb*self.tensionMax*action+self.fc*np.sign(self.state[1])+self.fd)  #PWM 180 : 7.437548494321268
        # return f
        return f#+np.random.normal(scale=0.3)#works
    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot]=self.state
        n=100
        # self.wAngularIni=np.random.normal(self.wAngularIni, self.wAngularIni/20,1)[0]
        if self.kinematics_integrator=='semi-euler':
            for i in range(n):
                force=self._calculate_force(action)
                # force=np.random.normal(force,)
                xacc = ( force + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) / (self.masscart + self.masspole * sintheta ** 2)
                # self.wAngularIni = np.random.normal(wAngular, 0.006, 1)[0]
                thetaacc = self.wAngularIni**2 * sintheta + xacc / self.length * costheta - theta_dot * K2
                # xacc=np.random.normal(xacc, 0.03, 1)[0]
                # thetaacc=np.random.normal(thetaacc, 0.03, 1)[0]
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta   +=theta_dot*self.tau/n
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
            # theta_dot=np.random.normal(theta_dot,0.1,1)[0]
            # x_dot = np.random.normal(x_dot, 0.03, 1)[0]
            # theta_dot=np.random.normal(theta_dot,0.1,1)[0]
            # x_dot = np.random.normal(x_dot, 0.03, 1)[0]
            self.state = np.array([x, x_dot, costheta, sintheta, theta_dot], dtype=np.float32)
        else:
            [x, x_dot, costheta, sintheta, theta_dot] = self.state
            state = odeint(self.pend, [x, x_dot, math.atan2(sintheta, costheta), theta_dot], [0, 0.05],args=(action, self.fa, self.fb, self.fc))
            self.state = [state[-1, 0], state[-1, 1], math.cos(state[-1, 2]), math.sin(state[-1, 2]), state[-1, 3]]
        done=False
        self.COUNTER+=1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        cost = reward_fnCos(x, costheta,theta_dot)
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}

    def pend(self, state, t, action, fa, fb, fc):
        [x, x_dot, theta, theta_dot] = state
        dqdt = np.zeros_like(state)
        costheta, sintheta = [np.cos(theta), np.sin(theta)]
        dqdt[0] = state[1]
        force = self._calculate_force(action)
        dqdt[1] = (force + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) / (self.masscart + self.masspole * sintheta ** 2)
        dqdt[3] = self.g / self.length * sintheta + dqdt[1] / self.length * costheta - theta_dot * K2
        dqdt[2] = state[3]
        # print(dqdt)
        return dqdt
    def reset(self, costheta=-1, sintheta=0, xIni=None,iniSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        if not self.randomReset:
            self.state = np.zeros(shape=(5,))
            if xIni!=None:
                self.state[0] = xIni
                print(f'x = {self.state[0]}')
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.3, high=0.3)
            self.state[1] = self.np_random.uniform(low=-0.3, high=0.3)
            self.state[4] = self.np_random.uniform(low=-5, high=5)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):

        if self.viewer:
            self.viewer.close()
            self.viewer = None
#CartPoleDiscrete Physical equations
class CartPoleDiscrete(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 Te=0.05,
                 randomReset=False,
                 f_a=A,
                 f_b=B,
                 f_c=C,
                 f_d=D,
                 integrator="semi-euler",
                 tensionMax=8.4706,
                 seed=0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.g = 9.806
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.4611167818372032  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = integrator  # 'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1  # 05
        self.wAngularIni = wAngular  # 4.488 #T=1.4285, w=
        self.reward = None
        self.randomReset = randomReset
        self.fa=f_a
        self.fb=f_b
        self.fc=f_c
        self.fd=f_d
        self.tensionMax=tensionMax
        self.arr=[]
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def _calculate_force(self,action):
        f=self.masscart*(self.fa*self.state[1]+self.fb*self.tensionMax*action[0]+self.fc*np.sign(self.state[1])+self.fd)  #PWM 180 : 7.437548494321268
        # return f
        return f#+np.random.normal(scale=0.3)#works
    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot]=self.state
        if action==0:
            action=[-1.0]
        elif action==1:
            action=[0.0]
        elif action==2:
            action=[1.0]
        else:
            raise Exception
        n=1
        # self.wAngularIni=np.random.normal(self.wAngularIni, self.wAngularIni/20,1)[0]
        if self.kinematics_integrator=='semi-euler':
            for i in range(n):
                force=self._calculate_force(action)
                # force=np.random.normal(force,)
                xacc = ( force + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) / (self.masscart + self.masspole * sintheta ** 2)
                # self.wAngularIni = np.random.normal(wAngular, 0.006, 1)[0]
                thetaacc = self.wAngularIni**2 * sintheta + xacc / self.length * costheta - theta_dot * K2
                xacc=np.random.normal(xacc, 0.03, 1)[0]
                thetaacc=np.random.normal(thetaacc, 0.03, 1)[0]
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta   +=theta_dot*self.tau/n
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
            theta_dot=np.random.normal(theta_dot,0.1,1)[0]
            x_dot = np.random.normal(x_dot, 0.03, 1)[0]
            # theta_dot=np.random.normal(theta_dot,0.1,1)[0]
            # x_dot = np.random.normal(x_dot, 0.03, 1)[0]
            self.state = np.array([x, x_dot, costheta, sintheta, theta_dot], dtype=np.float32)
        else:
            [x, x_dot, costheta, sintheta, theta_dot] = self.state
            state = odeint(self.pend, [x, x_dot, math.atan2(sintheta, costheta), theta_dot], [0, 0.05],args=(action, self.fa, self.fb, self.fc))
            self.state = [state[-1, 0], state[-1, 1], math.cos(state[-1, 2]), math.sin(state[-1, 2]), state[-1, 3]]
        done=False
        self.COUNTER+=1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        cost = reward_fnCos(x, costheta,theta_dot)
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}

    def pend(self, state, t, action, fa, fb, fc):
        [x, x_dot, theta, theta_dot] = state
        dqdt = np.zeros_like(state)
        costheta, sintheta = [np.cos(theta), np.sin(theta)]
        dqdt[0] = state[1]
        force = self._calculate_force(action)
        dqdt[1] = (force + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) / (self.masscart + self.masspole * sintheta ** 2)
        dqdt[3] = self.g / self.length * sintheta + dqdt[1] / self.length * costheta - theta_dot * K2
        dqdt[2] = state[3]
        # print(dqdt)
        return dqdt
    def reset(self, costheta=-1, sintheta=0, xIni=None,iniSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        if not self.randomReset:
            self.state = np.zeros(shape=(5,))
            if xIni!=None:
                self.state[0] = xIni
                print(f'x = {self.state[0]}')
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.3, high=0.3)
            self.state[1] = self.np_random.uniform(low=-0.3, high=0.3)
            self.state[4] = self.np_random.uniform(low=-5, high=5)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):

        if self.viewer:
            self.viewer.close()
            self.viewer = None
'''
class CartPoleDiscrete5(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 Te=0.05,
                 randomReset=False,
                 f_a=-18.03005925191054,
                 f_b=0.965036433340654,
                 f_c=-0.8992003750802359,
                 seed=0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.g = 9.806
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.47  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'  # 'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1  # 05
        self.wAngularIni = 4.488  # 4.488 #T=1.4285, w=
        self.reward = None
        self.randomReset = randomReset
        self.fa=f_a
        self.fb=f_b
        self.fc=f_c
        self.tensionMax=8.4706
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def _calculate_force(self,action):
        return self.masscart*(self.fa*self.state[1]+self.fb*self.tensionMax*action[0]+self.fc*np.sign(self.state[1]))
    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot]=self.state
        if action==0:
            action=[-1.0]
        elif action==1:
            action=[-0.5]
        elif action==2:
            action=[0.0]
        elif action==3:
            action=[0.5]
        elif action==4:
            action=[1.0]
        else:
            raise Exception
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                xacc = (self._calculate_force(action) + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) / (self.masscart + self.masspole * sintheta ** 2)
                thetaacc = self.g / self.length * sintheta + xacc / self.length * costheta - theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta   +=theta_dot*self.tau/n
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
            # theta_dot=np.random.normal(theta_dot,0.2,1)[0]
            # x_dot = np.random.normal(x_dot, 0.1, 1)[0]
        else:
            pass
        done=False
        self.COUNTER+=1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)
        # print(f'{self.state}')
        cost = reward_fnCos(x, costheta,theta_dot)
        # print('cost: {}'.format(cost))
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}
    def reset(self, costheta=-1, sintheta=0, iniSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        if not self.randomReset:
            self.state = np.zeros(shape=(5,))
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[1] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[4] = self.np_random.uniform(low=-0.2, high=0.2)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

DEBUG=False
#Linearised model
class CartPoleCosSinTensionD3(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 randomReset=False,
                 seed = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.806
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.46  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'rk'#
        self.theta_threshold_radians =  math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        # self.v_max = 15
        # self.w_max = 30
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1#05
        self.wAngularIni = 4.6124220495309425#4.488 #T=1.4285, w=
        self.reward = None
        self.randomReset=randomReset
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def gTension(self, u, x_dot=0, uMin=0.805, slope=0.0545): #u in volts, PWM 17.1/255 slope=1/19 m/s/V
        if abs(u)<uMin:
            return 0
        return (u-np.sign(u)*uMin)*slope #Fr opposes the tension
        # return u*slope
    def step(self, action):#180 = 8.47V
        assert self.observation_space.contains(self.state), 'obs_err'
        self.COUNTER+=1
        if action==0:
            action=[-1.0]
        elif action==1:
            action=[0.0]
        elif action==2:
            action=[1.0]
        else:
            raise Exception
        x, x_dot, costheta, sintheta, theta_dot = self.state
        # self.wAngular = np.random.normal(self.wAngularIni,abs(self.wAngularIni/10),1)[0]
        self.wAngular=self.wAngularIni
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                xacc = 1 / self.tauMec * (-x_dot + self.gTension(u=action[0] * 8.47,x_dot=x_dot))#- np.sign(x_dot)*0.0438725) # static friction 0.0438725
                # xacc=np.random.normal(xacc,abs(xacc/20),1)[0]
                thetaacc = self.wAngular ** 2 * sintheta + xacc/self.length * costheta #- theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta+=theta_dot*self.tau/n
                ## perturbations WORKING
                # x=np.random.normal(x,abs(x/20),1)[0]
                theta = np.random.normal(theta, abs(1*math.pi / 180), 1)[0]
                costheta=np.cos(theta)
                sintheta=np.sin(theta)


            # x_dot=np.random.normal(x_dot,abs(x_dot/10),1)[0]
            theta_dot=np.random.normal(theta_dot,0.2,1)[0]
            x_dot = np.random.normal(x_dot, 0.1, 1)[0]
            # theta_dot = np.random.normal(theta_dot, abs(theta_dot / 30), 1)[0]
        elif self.kinematics_integrator=='rk2':
            xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)

        cost = reward_fnCos(x, costheta,theta_dot)
        # print('cost: {}'.format(cost))
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0, iniSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        if not self.randomReset:
            self.state = np.zeros(shape=(5,))
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[1] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[4] = self.np_random.uniform(low=-0.2, high=0.2)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            #model with action history/fr

class CartPoleCosSinTensionD(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.48  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 300
        # self.v_max = 15
        # self.w_max = 30
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1
        self.wAngular = 4.4#4.488 #T=1.4285, w=
        self.reward = None
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def gTension(self, u, x_dot=0, uMin=0.805, slope=0.0545): #u in volts, PWM 17.1/255 slope=1/19 m/s/V
        if abs(u)<uMin:
            return 0
        return (u-np.sign(u)*uMin)*slope #Fr opposes the tension
        # return u*slope
    def step(self, action):#180 = 8.47V
        assert self.observation_space.contains(self.state), 'obs_err'
        self.COUNTER+=1
        if action==0:
            action=[-1.0]
        elif action==1:
        #     action=[0.0]
        # elif action==2:
            action=[1.0]
        else:
            raise Exception
        x, x_dot, costheta, sintheta, theta_dot = self.state
        # x_dot=
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                xacc = 1 / self.tauMec * (-x_dot + self.gTension(u=action[0] * 8.47,x_dot=x_dot))#- np.sign(x_dot)*0.0438725) # static friction 0.0438725
                thetaacc = self.wAngular ** 2 * sintheta + xacc/self.length * costheta - theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta+=theta_dot*self.tau/n
                theta = math.atan2(sintheta, costheta)
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
            ## perturbations WORKING
            x_dot=np.random.normal(x_dot,abs(x_dot/20),1)
            theta_dot=np.random.normal(theta_dot,abs(theta_dot/20),1)
            # x_dot = np.random.normal(x_dot, abs(x_dot / 30), 1)
            # theta_dot = np.random.normal(theta_dot, abs(theta_dot / 30), 1)
        elif self.kinematics_integrator=='rk2':
            pass
            #xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)

        cost = reward_fnCos(x, costheta,theta_dot)
        # print('cost: {}'.format(cost))
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0, iniSpeed=0.0, deterministic=True):
        self.COUNTER=0
        self.steps_beyond_done = None
        if deterministic:
            self.state = np.zeros(shape=(5,))
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
            self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[1] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[4] = self.np_random.uniform(low=-0.2, high=0.2)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            #model with action history/fr

class CartPoleCosSinTension(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.46  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'rk'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 3
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.05
        self.wAngular = 4.4#4.488 1.4349472355174064
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def gTension(self, u, x_dot=0, uMin=0.805, slope=0.0545): #u in volts, PWM 17.1/255 slope=1/19 m/s/V
        if abs(u)<uMin:
            return 0
        return (u-np.sign(u)*uMin)*slope #Fr opposes the tension
        # return u*slope

    def step(self, action):#180 = 8.47V
        if not self.observation_space.contains(self.state):
            print(f'obs_err{self.state}')
            raise Exception
        self.COUNTER+=1
        x, x_dot, costheta, sintheta, theta_dot = self.state
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                #xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725) # static friction 0.0438725
                xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47,x_dot=x_dot))# - np.sign(x_dot)*0.0438725) # static friction 0.0438725
                thetaacc = self.wAngular ** 2 * sintheta + xacc/self.length * costheta - theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta+=theta_dot*self.tau/n
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
        #perturbations
        x_dot=np.random.normal(x_dot,abs(x_dot/20),1)
        theta_dot=np.random.normal(theta_dot,abs(theta_dot/20),1)

        # elif self.kinematics_integrator=='rk2':
        #     xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)

        cost = reward_fnCos(x, costheta,theta_dot)
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0, iniCartSpeed=0.0,iniThetaSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = np.zeros(shape=(5,))
        self.state[1] = iniCartSpeed
        self.state[4] = iniThetaSpeed
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            #model with action history/fr


#with U/force
class CartPoleCosSinDev(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5  # center of mass
        self.length = 0.45  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 100
        self.w_max = 100
        self.thetas=[]

        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.n_obs=20
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if DEBUG:
            print(self.state)
            print(action[0])
        assert self.observation_space.contains(self.state), 'obs_err'

        action[0]=_action_static_friction(action[0])
        self.COUNTER += 1
        appliedMaxForce = self.force_mag * self.total_mass
        force = action[0] * appliedMaxForce
        n=2
        for i in range(n):
            x, x_dot, costheta, sintheta, theta_dot = self.state
            theta = math.atan2(sintheta, costheta)
            #TODO!temp = (force + self.polemass_length * theta_dot ** 2 * sintheta - np.sign(x_dot)*appliedMaxForce*0.08)/self.total_mass
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)) - K2 * theta_dot

            if self.kinematics_integrator == 'euler':
                xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            elif self.kinematics_integrator == 'friction':
                xacc = -K1 * x_dot + temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            theta = self.rescale_angle(theta)
            self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
            done = False
            if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
                done = True
                x = np.clip(x, -self.x_threshold, self.x_threshold)
        # self.thetas.append(theta)
        cost = reward_fnCos(x, costheta)
        # cost = reward_fnCos(x, costheta, x_dot)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 4
        if DEBUG:
            print(cost)
            print(done)
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = np.zeros(shape=(5,))#
        # self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[1] = 0
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[2] = 0#visualise
        # self.state[3] = 1#visualise
        if DEBUG:
            print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height/2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
'''
