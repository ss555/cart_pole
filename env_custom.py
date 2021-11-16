import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
from scipy import signal
from collections import deque
from time import time
import iir_filter
import json
import torch
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy import integrate
from utils import rungekutta4

def reward_fnCos(x, costheta, sintheta=0, theta_dot=0, sparse=False, Kx=5, x_threshold=0.5):
    if sparse:
        reward = 0.0
        if abs(np.arctan2(sintheta,costheta))<np.pi*30/180 and abs(x)<0.2:
            reward += 1
    else:
        reward = (1 - costheta - Kx * x ** 2)/2
    return reward

class CartPole(gym.Env):
    '''
    env used in American Journal of Physics article; cartpole simulated after identification of parameters (see. identification (scritps))
    dynamics derived according to Euler-Lagrange
    '''
    def __init__(self,
                 Te=0.05,
                 discreteActions=True,
                 resetMode='experimental',
                 Mcart=0.5,
                 Mpole=0.075,
                 length=0.411488843930847,
                 f_a=-20.75180095541654,  # -21.30359185798466,
                 f_b=1.059719258572224,  # 1.1088617953891196,
                 f_c=-1.166390864012042,  # -0.902272006611719,
                 f_d=-0.09727843708918459,  # 0.0393516077401241, #0.0,#
                 wAngular=4.881653071189049,
                 kPendViscous=0.07035332644615992,  # 0.0,#
                 integrator="semi-euler",
                 tensionMax=12,#7.0588235294117645,  # 150PWM
                 n=1,  # 2,5
                 sparseReward=False,
                 N_STEPS=800,
                 x_threshold=0.36,
                 thetaDotReset=None,
                 thetaReset=None,
                 THETA_DOT_LIMIT=13):  # 0.1
        '''
        :param Te: sampling time
        :param discreteActions: to use discrete Actions("True" to use with DQN) or continous ("False" to use with SAC)
        :param resetMode: experimental, goal , random or random_theta_thetaDot
        :param length: longeur du pendule
        :param f_a: viscous friction of motor Force=f_a*Speed+f_b*Vmotor+f_c*np.sign(Speed)+f_d (look report for more details)
        :param f_b: coefficient of proportionality for the motor Tension
        :param f_c: static friction for the motor movement which comprises friction of motor,reductor and cart
        :param f_d: assymmetry coefficient in the motor movement
        :param wAngular: frequence propre du pendule
        :param kPendViscous: viscous friction of a pendulum
        :param integrator: semi-euler(fast) or ode(long but more precise simulation)
        :param tensionMax: maximum tension of the motor, bu default 12V
        :param forceStd: gaussian uncertainty in % on the force applied by the motor
        '''
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.sparseReward = sparseReward
        self.g = 9.806
        self.masscart = Mcart
        self.masspoleIni = Mpole
        self.total_mass = (self.masspoleIni + self.masscart)
        self.length = length  # center of mass
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = integrator  # 'rk'# 'scipy'
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
        self.kPendViscous = kPendViscous
        self.thetaDotReset = thetaDotReset
        self.thetaReset = thetaReset
        self.THETA_DOT_LIMIT = THETA_DOT_LIMIT
        self.total_count = 0  # how many steps from the start of initialisation
        self.episodeNum = 0

    def _calculate_force(self, action):  # TODO fc a 0
        try:
            if action[0] == 0:
                f = self.masscart * (self.fa * self.state[1] + self.fc * np.sign(self.state[1]))
            else:
                f = self.masscart * (
                            self.fa * self.state[1] + self.fb * (self.tensionMax * action[0]) + self.fc * np.sign(
                        self.state[1]) + self.fd)
        except:
            print('error on force')
        return f
    def preprocess_action(self,action):
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
        return action
    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot] = self.state
        self.preprocess_action(action)

        if self.kinematics_integrator == 'semi-euler':
            for i in range(self.n):
                force = self._calculate_force(action)
                xacc = (force + self.masspole * theta_dot ** 2 * self.length * sintheta + self.masspole * self.g * sintheta * costheta) / (
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
                                                  [0, 0.05], args=(action, self.fa, self.fb, self.fc))[-1, :]

        x, x_dot, theta, theta_dot = self.update_noise_filter(x, x_dot, theta, theta_dot)
        done, cost, x = self.update_cost_done_x()
        # adding noise on observed variables (on x is negligible)
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32), cost, done, {}
    def update_cost_done_x(self):
        x, _, costheta, sintheta, theta_dot = self.state
        done = False
        self.COUNTER += 1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE or abs(
                theta_dot) > self.THETA_DOT_LIMIT:
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        cost = reward_fnCos(x, costheta, sintheta=sintheta, theta_dot=theta_dot, sparse=self.sparseReward)
        if x == -self.x_threshold or x == self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 2
        return done, cost, x

    def update_noise_filter(self, x, x_dot, theta, theta_dot):
        '''
        adds process noise, updates self.state, adds measurement noise
        :param x:
        :param x_dot:
        :param theta:
        :param theta_dot:
        :return:
        '''
        #adding process noise
        if self.Kp != 0:
            theta_dot = np.random.normal(theta_dot, self.Kp / self.tau, 1)[0]
            x_dot = np.random.normal(x_dot, 6e-3 * self.Kp / self.tau, 1)[0]
            theta = np.random.normal(theta, self.Kp, 1)[0]
        self.state = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        if self.Km != 0:
            theta_dot = np.random.normal(theta_dot, self.Km / self.tau, 1)[0]
            x_dot = np.random.normal(x_dot, 6e-3 * self.Km / self.tau, 1)[0]
            theta = np.random.normal(theta, self.Km, 1)[0]
        # # # filtering
        if self.FILTER:
            x_dot = self.iirX_dot.filter(x_dot)
            theta_dot = self.iirTheta_dot.filter(theta_dot)
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def reset(self, costheta=None, sintheta=None, xIni=None, x_ini_speed=None, theta_ini_speed=None):
        self.episodeNum += 1
        self.total_count += self.COUNTER
        if self.FILTER:
            self.iirX_dot.reset()
            self.iirTheta_dot.reset()

        self.COUNTER = 0

        if self.resetMode == 'experimental':
            self.state = np.zeros(shape=(5,))
            self.state[2] = 1
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
            self.state = np.zeros(shape=(5,))
            theta = self.np_random.uniform(-math.pi / 18, math.pi / 18)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        else:
            print('not defined, choose from experimental/goal/random')
        if self.thetaDotReset is not None:
            self.state[-1] = self.thetaDotReset
        if self.thetaReset is not None:
            self.state[3] = np.cos(self.thetaReset)
            self.state[4] = np.sin(self.thetaReset)
        if xIni is not None:
            self.state[0] = xIni
        if x_ini_speed is not None:
            self.state[1] = x_ini_speed
        if costheta is not None:
            self.state[2] = costheta
        if sintheta is not None:
            self.state[3] = sintheta
        if theta_ini_speed is not None:
            self.state[4] = theta_ini_speed
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
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
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

        text = f'physical time: {round((self.COUNTER + self.total_count) * self.tau, 1)} seconds, episode: {self.episodeNum}'

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta + np.pi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array', text_to_show=text)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleButter(gym.Env):
    '''
    env used in American Journal of Physics article; cartpole simulated after identification of parameters (see. identification (scritps))
    dynamics derived according to Euler-Lagrange
    '''
    def __init__(self,
                 Te=0.05,
                 discreteActions=True,
                 resetMode='experimental',
                 Mcart=0.5,
                 Mpole=0.075,
                 length=0.411488843930847,
                 f_a=-20.75180095541654,  # -21.30359185798466,
                 f_b=1.059719258572224,  # 1.1088617953891196,
                 f_c=-1.166390864012042,  # -0.902272006611719,
                 f_d=-0.09727843708918459,  # 0.0393516077401241, #0.0,#
                 wAngular=4.881653071189049,
                 kPendViscous=0.07035332644615992,  # 0.0,#
                 integrator="semi-euler",
                 tensionMax=12,#7.0588235294117645,  # 150PWM
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
        :param f_a: viscous friction of motor Force=f_a*Speed+f_b*Vmotor+f_c*np.sign(Speed)+f_d (look report for more details)
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
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = integrator  # 'rk'# 'scipy'
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
        self.total_count = 0  # how many steps from the start of initialisation
        self.episodeNum = 0
        if self.FILTER:
            self.iirTheta_dot = iir_filter.IIR_filter(signal.butter(4, 0.9, 'lowpass', output='sos'))  # 2nd param 0.3
            self.iirX_dot = iir_filter.IIR_filter(signal.butter(4, 0.5, 'lowpass', output='sos'))

    def seed(self, seed=5):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _calculate_force(self, action):  # TODO fc a 0
        try:
            if action[0] == 0:
                f = self.masscart * (self.fa * self.state[1] + self.fc * np.sign(self.state[1]))
            else:
                f = self.masscart * (
                            self.fa * self.state[1] + self.fb * (self.tensionMax * action[0]) + self.fc * np.sign(
                        self.state[1]) + self.fd)
        except:
            print('error on force')
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

        if self.kinematics_integrator == 'semi-euler':
            for i in range(self.n):
                force = self._calculate_force(action)
                if self.forceStd != 0:
                    force += np.random.normal(0, scale=self.forceStd * abs(force) / 100)
                xacc = (force + self.masspole * theta_dot ** 2 * self.length * sintheta + self.masspole * self.g * sintheta * costheta) / (
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
                                                  [0, 0.05], args=(action, self.fa, self.fb, self.fc))[-1, :]
            raise Exception
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

    def reset(self, costheta=None, sintheta=None, xIni=None, x_ini_speed=None, theta_ini_speed=None):
        self.episodeNum += 1
        self.total_count += self.COUNTER
        if self.FILTER:
            self.iirX_dot.reset()
            self.iirTheta_dot.reset()

        self.COUNTER = 0

        if self.resetMode == 'experimental':
            self.state = np.zeros(shape=(5,))
            self.state[2] = 1
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
            self.state = np.zeros(shape=(5,))
            theta = self.np_random.uniform(-math.pi / 18, math.pi / 18)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        else:
            print('not defined, choose from experimental/goal/random')
        if self.thetaDotReset is not None:
            self.state[-1] = self.thetaDotReset
        if self.thetaReset is not None:
            self.state[3] = np.cos(self.thetaReset)
            self.state[4] = np.sin(self.thetaReset)
        if xIni is not None:
            self.state[0] = xIni
        if x_ini_speed is not None:
            self.state[1] = x_ini_speed
        if costheta is not None:
            self.state[2] = costheta
        if sintheta is not None:
            self.state[3] = sintheta
        if theta_ini_speed is not None:
            self.state[4] = theta_ini_speed
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
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
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

        text = f'physical time: {round((self.COUNTER + self.total_count) * self.tau, 1)} seconds, episode: {self.episodeNum}'

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta + np.pi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array', text_to_show=text)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleNN(CartPole):
    '''
    class to simulate cartpole as neural network (requires torch NN model)
    '''
    def __init__(self,
                 model):  # 0.1
        super(CartPole, self).__init__()
        self.model = model

    def seed(self, seed=5):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        super(CartPole, self).preprocess_action()
        if self.discreteActions:
            if action == 0:
                action = [-1.0]
            elif action == 1:
                action = [0.0]
            elif action == 2:
                action = [1.0]
            else:
                raise Exception
        next_state = self.model(torch.Tensor(self.state).unsqueeze(0), torch.Tensor(action).unsqueeze(0))
        self.state = next_state.detach().numpy().squeeze(0)
        [x, x_dot, costheta, sintheta, theta_dot] = self.state
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
        return np.array([x, x_dot, costheta, sintheta, theta_dot], dtype=np.float32), cost, done, {}


    def reset(self, costheta=None, sintheta=None, xIni=None, x_ini_speed=None, theta_ini_speed=None):
        return super(CartPole, self).reset()

    def render(self, mode='human'):
        return super(CartPole, self).render()

    def close(self):
        super(CartPole, self).close()

class CartPoleNNs(CartPole):
    '''
    class to simulate cartpole as neural network (requires torch NN model)
    '''
    def __init__(self,
                 models):  # 0.1
        super(CartPole, self).__init__()
        self.models = models

    def step(self, action):
        super(CartPole, self).preprocess_action(action)
        x, x_dot, costheta, sintheta, theta_dot = self.state
        #dynamics
        model = self.models[np.random.choice(len(self.models))]
        next_state = self.model(torch.Tensor(self.state).unsqueeze(0), torch.Tensor(action).unsqueeze(0))
        self.state = next_state.detach().numpy().squeeze(0)

        # x, x_dot, theta, theta_dot = super(CartPole, self).update_noise_filter(x, x_dot, theta, theta_dot)
        done, cost, x = super(CartPole, self).update_cost_done_x()
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32), cost, done, {}



class CartPoleRK4(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 20}
    def __init__(self,
                 Te=0.05,
                 discreteActions=True,
                 resetMode='experimental',
                 Mcart=0.5,
                 Mpole=0.075,
                 length=0.411488843930847,
                 f_a=20.75180095541654,  # -21.30359185798466,
                 f_b=1.059719258572224,  # 1.1088617953891196,
                 f_c=1.166390864012042,  # -0.902272006611719,
                 f_d=0.09727843708918459,  # 0.0393516077401241, #0.0,#
                 wAngular=4.881653071189049,
                 kPendViscous=0.07035332644615992,  # 0.0,#
                 integrator='ode',
                 tensionMax=12,#7.0588235294117645,  # 150PWM
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
                 THETA_DOT_LIMIT=100,
                 title_to_show = ''):  # 0.1
        '''
        :param Te: sampling time
        :param discreteActions: to use discrete Actions("True" to use with DQN) or continous ("False" to use with SAC)
        :param resetMode: experimental, goal , random or random_theta_thetaDot
        :param length: longeur du pendule
        :param f_a: viscous friction of motor Force=f_a*Speed+f_b*Vmotor+f_c*np.sign(Speed)+f_d (look report for more details)
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
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = integrator  # 'rk'# 'scipy'
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
        self.total_count = 0  # how many steps from the start of initialisation
        self.episodeNum = 0
        self.title_to_show = title_to_show
        if self.FILTER:
            self.iirTheta_dot = iir_filter.IIR_filter(signal.butter(4, 0.9, 'lowpass', output='sos'))  # 2nd param 0.3
            self.iirX_dot = iir_filter.IIR_filter(signal.butter(4, 0.5, 'lowpass', output='sos'))

    def seed(self, seed=5):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        if self.kinematics_integrator=='ode':
            [x, x_dot, theta, theta_dot] = odeint(self.pendulum_dynamics, y0=[x, x_dot, math.atan2(sintheta, costheta), theta_dot], t=[0, 0.05], args=(action, self.fa, self.fb, self.fc))[-1, :]
        else:
            [x, x_dot, theta, theta_dot] = rungekutta4(self.pendulum_dynamics,
                                                  y0=[x, x_dot, math.atan2(sintheta, costheta), theta_dot], t=[0, 0.05],
                                                  args=(action, self.fa, self.fb, self.fc))[-1, :]

        # adding process noise
        if self.Kp != 0:
            theta_dot = np.random.normal(theta_dot, self.Kp / self.tau, 1)[0]
            x_dot = np.random.normal(x_dot, 6e-3 * self.Kp / self.tau, 1)[0]
            theta = np.random.normal(theta, self.Kp, 1)[0]
            x = np.random.normal(x, 6e-3 * self.Kp, 1)[0]
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
        # adding noise on observed variables (on x is negligible?)
        if self.Km != 0:
            theta_dot = np.random.normal(theta_dot, self.Km / self.tau, 1)[0]
            x_dot = np.random.normal(x_dot, 6e-3 * self.Km / self.tau, 1)[0]
            theta = np.random.normal(theta, self.Km, 1)[0]
            x = np.random.normal(x, 6e-3 * self.Km, 1)[0]
        # # # filtering
        if self.FILTER:
            x_dot = self.iirX_dot.filter(x_dot)
            theta_dot = self.iirTheta_dot.filter(theta_dot)
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32), cost, done, {}

    def pendulum_dynamics(self, state, t, action, fa, fb, fc):
        [x, x_dot, theta, theta_dot] = state
        costheta, sintheta = [np.cos(theta), np.sin(theta)]
        action = np.array(action).reshape(1)[0]#make in standard format
        fdry = -self.fc * np.tanh(1000*x_dot)
        force = self.masscart * (-self.fa * x_dot + fdry)

        if action != 0:
            force += self.masscart * (-self.fd + self.fb * (self.tensionMax * action))
        if self.forceStd != 0:
            force += np.random.normal(0, scale=self.forceStd * abs(force) / 100)
        xacc = (force + self.masspole * theta_dot ** 2 * self.length * sintheta + self.masspole * self.g * sintheta * costheta) / (
                       self.masscart + self.masspole * sintheta ** 2)
        thetaacc = -self.wAngularUsed ** 2 * sintheta - xacc / self.length * costheta - theta_dot * self.kPendViscous
        return np.array([x_dot, xacc, theta_dot, thetaacc])

    def reset(self, costheta=None, sintheta=None, xIni=None, x_ini_speed=None, theta_ini_speed=None):
        self.episodeNum += 1
        self.total_count += self.COUNTER
        if self.FILTER:
            self.iirX_dot.reset()
            self.iirTheta_dot.reset()

        self.COUNTER = 0
        if self.resetMode == 'experimental':
            self.state = np.zeros(shape=(5,))
            self.state[2] = 1
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
            self.state = np.zeros(shape=(5,))
            theta = self.np_random.uniform(-math.pi / 18, math.pi / 18)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        else:
            print('not defined, choose from experimental/goal/random')
        if self.thetaDotReset is not None:
            self.state[-1] = self.thetaDotReset
        if self.thetaReset is not None:
            self.state[3] = np.cos(self.thetaReset)
            self.state[4] = np.sin(self.thetaReset)
        if xIni is not None:
            self.state[0] = xIni
        if x_ini_speed is not None:
            self.state[1] = x_ini_speed
        if costheta is not None:
            self.state[2] = costheta
        if sintheta is not None:
            self.state[3] = sintheta
        if theta_ini_speed is not None:
            self.state[4] = theta_ini_speed
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
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
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

        text = f'physical time: {round((self.COUNTER + self.total_count) * self.tau, 1)} seconds, episode: {self.episodeNum}'

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta + np.pi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array', text_to_show=text, title_to_show=self.title_to_show)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
