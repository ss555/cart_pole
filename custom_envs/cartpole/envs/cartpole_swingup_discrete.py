import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
from scipy.integrate import odeint
from time import time
import iir_filter
import json
from matplotlib import pyplot as plt

# PWM 180
[A, B, C, D] = [-21.30359185798466, 1.1088617953891196, -0.902272006611719, -0.03935160774012411]  # 20ms#(-7.794018686563599, 0.37538450501353504, -0.4891760779740128, -0.002568958116514183)
wAngular = 4.85658326956131


def reward_fnCos(x, costheta, sintheta=0, theta_dot=0, sparse=False, Kx=5):
    if sparse:
        reward = 0.0
        if abs(np.arctan2(sintheta, costheta)) < np.pi * 30 / 180 and abs(x) < 0.2:
            reward += 1
    else:
        # reward = 1 - costheta - (1-costheta)* Kx * x ** 2
        reward = 1 - costheta - Kx * x ** 2
    return reward


length = 0.416  # center of mass
N_STEPS = 800
# #TEST#
K1 = 18.8  # 30#Guil  #dynamic friction
kPendViscous = 0.11963736650935591  # 0.15 #2 #0.1  #friction on theta

class CartPoleDiscrete(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 Te=0.05,
                 discreteActions=True,
                 resetMode='experimental',
                 Mcart=0.5,
                 Mpole=0.075,
                 length=0.416,
                 f_a=-21.30359185798466,
                 f_b=1.1088617953891196,
                 f_c=-0.902272006611719,
                 f_d=-0.0393516077401241,
                 wAngular=4.85658326956131,
                 kPendViscous=0.11963736650935591,  # 0.0,#
                 integrator="semi-euler",
                 tensionMax=12,  # 8.4706
                 FILTER=False,
                 n=1,  # 2,5
                 Kp=0,
                 sparseReward=False,
                 Km=0,  # 1.2,
                 seed=0,
                 N_STEPS=800,
                 wAngularStd=0.0,  # 0.1
                 masspoleStd=0.0,  # 0.01
                 forceStd=0.0,
                 x_threshold=0.36,
                 thetaDotReset=None,
                 thetaReset=None,
                 THETA_DOT_LIMIT=100):  # 0.1
        '''
        :param Te: sampling time
        :param discreteActions: to use discrete Actions("True" to use with DQN) or continous ("False" to use with SAC)
        :param resetMode: experimental, goal , random or random_theta_thetaDot
        :param length: longeur du pendule
        :param f_a: viscous friction of motor Force= f_a*Speed+f_b*Vmotor+f_c*np.sign(Speed)+f_d (look report for more details)
        :param f_b: coefficient of proportionality for the motor Tension
        :param f_c: static friction for the motor movement which comprises friction of motor,reductor and cart
        :param f_d: assymmetry coefficient in the motor movement
        :param wAngular: frequence propre du pendule
        :param kPendViscous: viscous friction of a pendulum
        :param integrator: semi-euler(fast) or ode(long but more precise simulation)
        :param tensionMax: maximum tension of the motor, bu default 12V
        :param FILTER: weather to apply butterworth lowpass filter on measurements
        :param n: when using semi-euler integrator for the simulation, to increase the precision
        :param Kp: Noise process standard deviation error for the pendulum, the more elevated Kp is, the more noisy will be the system
        :param sparseReward: weather to train with Sparse or Informative rewards
        :param Km: Noise measurement standard deviation error on encoder readings, the more elevated Km is, the more noisy are the measurements(doesn't affect process)
        :param seed: seed for random initialisation
        :param N_STEPS: How many steps in an episode
        :param wAngularStd: gaussian uncertainty on the natural angular frequency of a pendulum
        :param masspoleStd: gaussian uncertainty on the mass
        :param forceStd: gaussian uncertainty in % on the force applied by the motor
        '''
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.sparseReward = sparseReward
        self.Km = Km  # bruit du mesure
        self.Kp = Kp  # bruit du process
        self.g = 9.806
        self.masscart = Mcart
        self.masspoleIni = Mpole
        self.total_mass = (self.masspoleIni + self.masscart)
        self.length = length  # center of mass
        # self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = integrator  # 'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = x_threshold
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        self.discreteActions = discreteActions
        if self.discreteActions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None

        self.wAngularIni = wAngular  # 4.488 #T=1.4285, w=
        self.reward = None
        self.resetMode = resetMode
        self.fa = f_a
        self.fb = f_b
        self.fc = f_c
        self.fd = f_d
        self.tensionMax = tensionMax
        self.arr = []
        self.n = n
        self.masspoleStd = masspoleStd
        self.wAngularStd = wAngularStd
        self.forceStd = forceStd
        self.FILTER = FILTER
        self.kPendViscous = kPendViscous
        self.thetaDotReset = thetaDotReset
        self.thetaReset = thetaReset
        self.THETA_DOT_LIMIT = THETA_DOT_LIMIT
        if self.FILTER:
            self.iirTheta_dot = iir_filter.IIR_filter(signal.butter(4, 0.9, 'lowpass', output='sos'))  # 2nd param 0.3

    def seed(self, seed=5):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _calculate_force(self, action):
        f = self.masscart * (self.fa * self.state[1] + self.fb * self.tensionMax * action[0] + self.fc * np.sign(
            self.state[1]) + self.fd)  # PWM 180 : 7.437548494321268
        return f

    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot] = self.state
        if self.discreteActions:
            if action == 0:
                action = [-1.0]
            elif action == 1:
                action = [0.0]
            elif action == 2:
                action = [1.0]
            else:
                raise Exception
        else:  # continous
            pass
        self.wAngularUsed = np.random.normal(self.wAngularIni, self.wAngularStd, 1)[0]
        self.masspole = np.random.normal(self.masspoleIni, self.masspoleStd, 1)[0]
        self.polemass_length = (self.masspole * self.length)
        if self.kinematics_integrator == 'semi-euler':
            for i in range(self.n):
                force = self._calculate_force(action)
                if self.forceStd != 0:
                    force += np.random.normal(0, scale=self.forceStd * abs(force) / 100)
                xacc = (
                                   force + self.masspole * theta_dot ** 2 * self.length * sintheta + self.masspole * self.g * sintheta * costheta) / (
                                   self.masscart + self.masspole * sintheta ** 2)
                thetaacc = -self.wAngularUsed ** 2 * sintheta - xacc / self.length * costheta - theta_dot * self.kPendViscous
                x_dot += self.tau / self.n * xacc
                x += x_dot * self.tau / self.n
                theta_dot += thetaacc * self.tau / self.n
                theta = math.atan2(sintheta, costheta)
                theta += theta_dot * self.tau / self.n
                costheta = np.cos(theta)
                sintheta = np.sin(theta)
        else:
            [x, x_dot, theta, theta_dot] = odeint(self.pend, [x, x_dot, math.atan2(sintheta, costheta), theta_dot],
                                                  [0, 0.05],
                                                  args=(action, self.fa, self.fb, self.fc))[-1, :]
        # adding process noise
        if self.Kp != 0:
            theta_dot = np.random.normal(theta_dot, self.Kp / self.tau, 1)[0]
            x_dot = np.random.normal(x_dot, 6e-3 * self.Kp / self.tau, 1)[0]
            theta = np.random.normal(theta, self.Kp, 1)[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        self.state = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        done = False
        self.COUNTER += 1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE or abs(
                theta_dot) > self.THETA_DOT_LIMIT:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        cost = reward_fnCos(x, costheta, sintheta=sintheta, theta_dot=theta_dot, sparse=self.sparseReward)
        if x == -self.x_threshold or x == self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 2
        # adding noise on observed variables (on x is negligible)
        if self.Km != 0:
            theta_dot = np.random.normal(theta_dot, self.Km / self.tau, 1)[0]
            x_dot = np.random.normal(x_dot, 6e-3 * self.Km / self.tau, 1)[0]
            theta = np.random.normal(theta, self.Km, 1)[0]
        # # # filtering
        if self.FILTER:
            x_dot = self.iirX_dot.filter(x_dot)
            theta_dot = self.iirTheta_dot.filter(theta_dot)
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32), cost, done, {}

    def pend(self, state, t, action, fa, fb, fc):
        [x, x_dot, theta, theta_dot] = state
        dqdt = np.zeros_like(state)
        costheta, sintheta = [np.cos(theta), np.sin(theta)]
        dqdt[0] = state[1]
        force = self._calculate_force(action)  # 0.44*(fa*x_dot+fb*self.tensionMax*action[0]+fc*np.sign(x_dot))
        # TODO force?
        dqdt[1] = (force  # self.masscart*(fa*x_dot+fb*8.47*action[0]+fc*np.sign(x_dot))#force
                   + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) \
                  / (self.masscart + self.masspole * sintheta ** 2)
        dqdt[3] = self.g / self.length * sintheta + dqdt[1] / self.length * costheta - theta_dot * kPendViscous
        dqdt[2] = state[3]
        return dqdt

    def reset(self, costheta=1, sintheta=0, xIni=0.0, x_ini_speed=0.0, theta_ini_speed=0.0):
        if self.FILTER:
            # self.iirX_dot=iir_filter.IIR_filter(signal.butter(4, 0.5, 'lowpass', output='sos'))
            self.iirTheta_dot.reset()
        self.COUNTER = 0

        if self.resetMode == 'experimental':
            self.state = np.zeros(shape=(5,))
            self.state[0] = xIni
            self.state[1] = x_ini_speed
            self.state[2] = costheta
            self.state[3] = sintheta
            self.state[4] = theta_ini_speed
        elif self.resetMode == 'goal':
            self.state = np.zeros(shape=(5,))
            self.state[2] = -1
            self.state[0] = self.np_random.uniform(low=-0.1, high=0.1)
        elif self.resetMode == 'random':
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.35, high=0.35)
            self.state[1] = self.np_random.uniform(low=-0.5, high=0.5)
            self.state[4] = self.np_random.uniform(low=-5, high=5)
            theta = self.np_random.uniform(-math.pi, math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        elif self.resetMode == 'random_theta_thetaDot':
            theta = self.np_random.uniform(-math.pi / 18, math.pi / 18)
            self.state = [xIni, x_ini_speed, np.cos(theta), np.sin(theta), self.np_random.uniform(low=-0.05, high=0.05)]
        else:
            print('not defined, choose from experimental/goal/random')
        if self.thetaDotReset is not None:
            self.state[-1] = self.thetaDotReset
        if self.thetaReset is not None:
            self.state[3] = np.cos(self.thetaReset)
            self.state[4] = np.sin(self.thetaReset)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)

    def rescale_angle(self, theta):
        return math.atan2(math.sin(theta), math.cos(theta))

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
        self.poletrans.set_rotation(theta + np.pi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleDiscreteImage(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 Te=0.05,
                 discreteActions=True,
                 resetMode='experimental',
                 Mcart=0.5,
                 Mpole=0.075,
                 length=0.416,
                 f_a=-21.30359185798466,
                 f_b=1.1088617953891196,
                 f_c=-0.902272006611719,
                 f_d=-0.0393516077401241,
                 wAngular=4.85658326956131,
                 kPendViscous=0.11963736650935591,  # 0.0,#
                 integrator="semi-euler",
                 tensionMax=12,  # 8.4706
                 FILTER=False,
                 n=5,  # 2,5
                 Kp=0,
                 sparseReward=False,
                 Km=0,  # 1.2,
                 seed=0,
                 N_STEPS=800,
                 wAngularStd=0.0,  # 0.1
                 masspoleStd=0.0,  # 0.01
                 forceStd=0.0,
                 x_threshold=0.36,
                 thetaDotReset=None,
                 thetaReset=None,
                 THETA_DOT_LIMIT=100):  # 0.1
        '''
        :param Te: sampling time
        :param discreteActions: to use discrete Actions("True" to use with DQN) or continous ("False" to use with SAC)
        :param resetMode: experimental, goal , random or random_theta_thetaDot
        :param length: longeur du pendule
        :param f_a: viscous friction of motor Force= f_a*Speed+f_b*Vmotor+f_c*np.sign(Speed)+f_d (look report for more details)
        :param f_b: coefficient of proportionality for the motor Tension
        :param f_c: static friction for the motor movement which comprises friction of motor,reductor and cart
        :param f_d: assymmetry coefficient in the motor movement
        :param wAngular: frequence propre du pendule
        :param kPendViscous: viscous friction of a pendulum
        :param integrator: semi-euler(fast) or ode(long but more precise simulation)
        :param tensionMax: maximum tension of the motor, bu default 12V
        :param FILTER: weather to apply butterworth lowpass filter on measurements
        :param n: when using semi-euler integrator for the simulation, to increase the precision
        :param Kp: Noise process standard deviation error for the pendulum, the more elevated Kp is, the more noisy will be the system
        :param sparseReward: weather to train with Sparse or Informative rewards
        :param Km: Noise measurement standard deviation error on encoder readings, the more elevated Km is, the more noisy are the measurements(doesn't affect process)
        :param seed: seed for random initialisation
        :param N_STEPS: How many steps in an episode
        :param wAngularStd: gaussian uncertainty on the natural angular frequency of a pendulum
        :param masspoleStd: gaussian uncertainty on the mass
        :param forceStd: gaussian uncertainty in % on the force applied by the motor
        '''
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.sparseReward = sparseReward
        self.Km = Km  # bruit du mesure
        self.Kp = Kp  # bruit du process
        self.g = 9.806
        self.masscart = Mcart
        self.masspoleIni = Mpole
        self.total_mass = (self.masspoleIni + self.masscart)
        self.length = length  # center of mass
        # self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = integrator  # 'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = x_threshold
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        self.discreteActions = discreteActions
        if self.discreteActions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None

        self.wAngularIni = wAngular  # 4.488 #T=1.4285, w=
        self.reward = None
        self.resetMode = resetMode
        self.fa = f_a
        self.fb = f_b
        self.fc = f_c
        self.fd = f_d
        self.tensionMax = tensionMax
        self.arr = []
        self.n = n
        self.masspoleStd = masspoleStd
        self.wAngularStd = wAngularStd
        self.forceStd = forceStd
        self.FILTER = FILTER
        self.kPendViscous = kPendViscous
        self.thetaDotReset = thetaDotReset
        self.thetaReset = thetaReset
        self.THETA_DOT_LIMIT = THETA_DOT_LIMIT
        if self.FILTER:
            self.iirTheta_dot = iir_filter.IIR_filter(signal.butter(4, 0.9, 'lowpass', output='sos'))  # 2nd param 0.3

    def seed(self, seed=5):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _calculate_force(self, action):
        f = self.masscart * (self.fa * self.state[1] + self.fb * self.tensionMax * action[0] + self.fc * np.sign(
            self.state[1]) + self.fd)  # PWM 180 : 7.437548494321268
        return f

    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot] = self.state
        if self.discreteActions:
            if action == 0:
                action = [-1.0]
            elif action == 1:
                action = [0.0]
            elif action == 2:
                action = [1.0]
            else:
                raise Exception
        else:  # continous
            pass
        self.wAngularUsed = np.random.normal(self.wAngularIni, self.wAngularStd, 1)[0]
        self.masspole = np.random.normal(self.masspoleIni, self.masspoleStd, 1)[0]
        self.polemass_length = (self.masspole * self.length)
        if self.kinematics_integrator == 'semi-euler':
            for i in range(self.n):
                force = self._calculate_force(action)
                if self.forceStd != 0:
                    force += np.random.normal(0, scale=self.forceStd * abs(force) / 100)
                xacc = (
                                   force + self.masspole * theta_dot ** 2 * self.length * sintheta + self.masspole * self.g * sintheta * costheta) / (
                                   self.masscart + self.masspole * sintheta ** 2)
                thetaacc = -self.wAngularUsed ** 2 * sintheta - xacc / self.length * costheta - theta_dot * self.kPendViscous
                x_dot += self.tau / self.n * xacc
                x += x_dot * self.tau / self.n
                theta_dot += thetaacc * self.tau / self.n
                theta = math.atan2(sintheta, costheta)
                theta += theta_dot * self.tau / self.n
                costheta = np.cos(theta)
                sintheta = np.sin(theta)
        else:
            [x, x_dot, theta, theta_dot] = odeint(self.pend, [x, x_dot, math.atan2(sintheta, costheta), theta_dot],
                                                  [0, 0.05],
                                                  args=(action, self.fa, self.fb, self.fc))[-1, :]
        # adding process noise
        if self.Kp != 0:
            theta_dot = np.random.normal(theta_dot, self.Kp / self.tau, 1)[0]
            x_dot = np.random.normal(x_dot, 6e-3 * self.Kp / self.tau, 1)[0]
            theta = np.random.normal(theta, self.Kp, 1)[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        self.state = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        done = False
        self.COUNTER += 1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE or abs(
                theta_dot) > self.THETA_DOT_LIMIT:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        cost = reward_fnCos(x, costheta, sintheta=sintheta, theta_dot=theta_dot, sparse=self.sparseReward)
        if x == -self.x_threshold or x == self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 2
        # adding noise on observed variables (on x is negligible)
        if self.Km != 0:
            theta_dot = np.random.normal(theta_dot, self.Km / self.tau, 1)[0]
            x_dot = np.random.normal(x_dot, 6e-3 * self.Km / self.tau, 1)[0]
            theta = np.random.normal(theta, self.Km, 1)[0]
        # # # filtering
        if self.FILTER:
            x_dot = self.iirX_dot.filter(x_dot)
            theta_dot = self.iirTheta_dot.filter(theta_dot)
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32), cost, done, {}

    def pend(self, state, t, action, fa, fb, fc):
        [x, x_dot, theta, theta_dot] = state
        dqdt = np.zeros_like(state)
        costheta, sintheta = [np.cos(theta), np.sin(theta)]
        dqdt[0] = state[1]
        force = self._calculate_force(action)  # 0.44*(fa*x_dot+fb*self.tensionMax*action[0]+fc*np.sign(x_dot))
        # TODO force?
        dqdt[1] = (force  # self.masscart*(fa*x_dot+fb*8.47*action[0]+fc*np.sign(x_dot))#force
                   + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) \
                  / (self.masscart + self.masspole * sintheta ** 2)
        dqdt[3] = self.g / self.length * sintheta + dqdt[1] / self.length * costheta - theta_dot * kPendViscous
        dqdt[2] = state[3]
        return dqdt

    def reset(self, costheta=1, sintheta=0, xIni=0.0, x_ini_speed=0.0, theta_ini_speed=0.0):
        if self.FILTER:
            # self.iirX_dot=iir_filter.IIR_filter(signal.butter(4, 0.5, 'lowpass', output='sos'))
            self.iirTheta_dot.reset()
        self.COUNTER = 0

        if self.resetMode == 'experimental':
            self.state = np.zeros(shape=(5,))
            self.state[0] = xIni
            self.state[1] = x_ini_speed
            self.state[2] = costheta
            self.state[3] = sintheta
            self.state[4] = theta_ini_speed
        elif self.resetMode == 'goal':
            self.state = np.zeros(shape=(5,))
            self.state[2] = -1
            self.state[0] = self.np_random.uniform(low=-0.1, high=0.1)
        elif self.resetMode == 'random':
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.35, high=0.35)
            self.state[1] = self.np_random.uniform(low=-0.5, high=0.5)
            self.state[4] = self.np_random.uniform(low=-5, high=5)
            theta = self.np_random.uniform(-math.pi, math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        elif self.resetMode == 'random_theta_thetaDot':
            theta = self.np_random.uniform(-math.pi / 18, math.pi / 18)
            self.state = [xIni, x_ini_speed, np.cos(theta), np.sin(theta), self.np_random.uniform(low=-0.05, high=0.05)]
        else:
            print('not defined, choose from experimental/goal/random')
        if self.thetaDotReset is not None:
            self.state[-1] = self.thetaDotReset
        if self.thetaReset is not None:
            self.state[3] = np.cos(self.thetaReset)
            self.state[4] = np.sin(self.thetaReset)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)

    def rescale_angle(self, theta):
        return math.atan2(math.sin(theta), math.cos(theta))

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
        self.poletrans.set_rotation(theta + np.pi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

