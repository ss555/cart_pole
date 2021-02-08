"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import argparse

class CartPoleCusBottomNoisy(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,masscart=0.4,
                 masspole=0.07,
                 std_masscart=0.02,
                 std_masspole=0.02,
                 std_x_dot=0.01,
                 std_theta_dot=0.1,
                 force_mag=20.0,
                 tau=0.05,
                 seed=0):
        self.gravity = 9.8
        self.masscart = masscart
        self.masspole = masspole
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.std_masscart = std_masscart
        self.std_masspole = std_masspole
        self.std_x_dot = std_x_dot
        self.std_theta_dot = std_theta_dot
        self.force_mag = force_mag
        self.COUNTER = 0
        self.gravity = 10
        self.masscart = 0.4
        self.masspole = 0.07
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = force_mag
        self.tau = tau  # seconds between state updates
        self.kinematics_integrator = 'friction'  # 'euler'
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 500
        self.w_max = 500

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 2,
            self.w_max,
            1])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = _action_static_friction(action)
        self.COUNTER += 1
        masspole = np.random.normal(self.masspole, self.std_masspole)
        masscart = np.random.normal(self.masscart, self.std_masscart)
        self.total_mass = (masspole + masscart)
        state = self.state
        x, x_dot, theta, theta_dot, _ = state
        force = action * self.force_mag
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass, 0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)), 0)
        xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass, 0)
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'friction':
            xacc = -0.1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass  # adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot, self.COUNTER / self.MAX_STEPS_PER_EPISODE], dtype=np.float32)
        done = False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            done = True

        # cost = reward_fn(x,x_dot,theta,theta_dot,self.x_threshold,self.observation_space.high[1],self.observation_space.high[-1],action)
        cost = reward_fn(x, theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - 100
        return self.state, cost, done, {}
        # tensorboard --logdir ./sac_cartpole_tensorboard/
        # INFO
        # reward_fn 4096 takes 1h to train
        # sparse rewards doesnt learn

    def reset(self):
        # self.state = np.array([0, 0, math.pi, 0]).astype(np.float32)
        self.steps_beyond_done = None
        self.COUNTER = 0
        state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        state[2] = math.pi - state[2]
        state[0] = self.np_random.uniform(low=-2, high=2)
        state = np.append(state, 0)
        self.state = state
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def reward_fnCos(x,costheta):
    cost = 2+costheta-x**2/25
    return cost
def reward_fn(x,theta,action=0.0):
    # cost=0
    # if abs(theta)<math.pi/15 and abs(x)<0.5:
    #     cost+=1
    cost=2+np.cos(theta)-x**2/25
    return cost
def _action_static_friction(action, threshold=0.08):
    if abs(action)<threshold:
        return 0
    if action>0:
        return action-threshold
    else:
        return action+threshold
N_STEPS=1000
K1 = 0.3  #dynamic friction
K2 = 1  #friction on theta
Mpoletest = 0.2
McartTest = 0.6
Mcart=0.45#0.45
Mpole=0.2
Applied_force=2
class CartPoleCusBottom(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.observation_space.contains(self.state), 'obs_err'
        action=_action_static_friction(action)
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, theta, theta_dot = self.state
        force = action*self.force_mag
        # print(force)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        theta=self.rescale_angle(theta)
        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x,-self.x_threshold,self.x_threshold)

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = math.pi
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        # print('reset state:',self.state)
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleCosSin(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
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

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.observation_space.contains(self.state), 'obs_err'
        action=_action_static_friction(action)
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, costheta, sintheta, theta_dot = self.state
        force = action*self.force_mag
        theta=math.atan2(sintheta, costheta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        theta=self.rescale_angle(theta)
        self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x,-self.x_threshold,self.x_threshold)

        cost=reward_fnCos(x,costheta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[2] = -1
        self.state[3] = 0
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        # print('reset state:',self.state)
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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

class CartPoleFriction(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.observation_space.contains(self.state), 'obs_err'
        action=_action_static_friction(action)
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, theta, theta_dot = self.state
        force = action*self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)) - K2*theta_dot
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        theta=self.rescale_angle(theta)
        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x,-self.x_threshold,self.x_threshold)

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = math.pi/2
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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
        theta=math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleTime(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 2
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print(self.state)
        assert self.observation_space.contains(self.state), 'obs_err'
        #action=_action_static_friction(action)
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, theta, theta_dot = self.state
        force = action*self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)

        if self.kinematics_integrator == 'euler':
            xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass, 0)
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            if theta>math.pi:
                theta-=2*math.pi
            elif theta<=-math.pi:
                theta+=2*math.pi
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            if theta>math.pi:
                theta-=2*math.pi
            elif theta<=-math.pi:
                theta+=2*math.pi
            theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x,-self.x_threshold,self.x_threshold)

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = self.rescale_angle(math.pi - self.state[2])
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleBullet(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 2
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print(self.state)
        assert self.observation_space.contains(self.state), 'obs_err'
        #action=_action_static_friction(action)
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, theta, theta_dot = self.state
        force = action*self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)

        if self.kinematics_integrator == 'euler':
            xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass, 0)
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            if theta>math.pi:
                theta-=2*math.pi
            elif theta<=-math.pi:
                theta+=2*math.pi
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            if theta>math.pi:
                theta-=2*math.pi
            elif theta<=-math.pi:
                theta+=2*math.pi
            theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)

        cost=reward_fn(x, theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER = 0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[2] = -1
        self.state[3] = 0
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleCusBottomDiscrete(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE=N_STEPS
        self.gravity = 10
        self.masscart = 0.45
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 0.2
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'
        self.K1 = K1 / self.total_mass
        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        self.v_max = 500
        self.w_max = 500

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 2,
            self.w_max])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        state = self.state
        x, x_dot, theta, theta_dot = state
        if action == 0:
            force = 0
        elif action == 1:
            force = -self.force_mag
        elif action == 2:
            force = self.force_mag
        else:
            print('invalid action space asked')
            exit(-1)
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)
        xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass,0)
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max)
        elif self.kinematics_integrator == 'friction':
            xacc = self.K1 * x_dot  + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER>self.MAX_STEPS_PER_EPISODE:
            done = True

        #cost = reward_fn(x,x_dot,theta,theta_dot,self.x_threshold,self.observation_space.high[1],self.observation_space.high[-1],action)
        cost=reward_fn(x,theta)
        if x <= -self.x_threshold or x >= self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}
#tensorboard --logdir ./sac_cartpole_tensorboard/
    #INFO
    #reward_fn 4096 takes 1h to train
    #sparse rewards doesnt learn
    def reset(self):
        # self.state = np.array([0, 0, math.pi, 0]).astype(np.float32)
        self.steps_beyond_done = None
        self.COUNTER = 0
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = math.pi - self.state[2]
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleDiscrete2actions(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE=1000
        self.gravity = 10
        self.masscart = 0.45
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 0.05
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        self.v_max = 500
        self.w_max = 500

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 2,
            self.w_max,
            1])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.COUNTER += 1
        self.total_mass = (self.masspole + self.masscart)
        state = self.state
        x, x_dot, theta, theta_dot, _ = state
        if action == 0:
            force = self.force_mag
        elif action == 1:
            force = -self.force_mag
        else:
            print('invalid action space asked')
            exit(-1)
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)
        xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass,0)
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max)
        elif self.kinematics_integrator == 'friction':
            xacc = -0.1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot, self.COUNTER/self.MAX_STEPS_PER_EPISODE],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}
    def reset(self):
        self.steps_beyond_done = None
        self.COUNTER = 0
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = math.pi - self.state[2]
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        self.state=np.append(self.state,0)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleConTest(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 time_limit=400,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE = time_limit
        self.gravity = 10
        self.masscart = 0.45
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 2
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max,
            1])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.total_mass = (self.masspole + self.masscart)
        state = self.state
        x, x_dot, theta, theta_dot, _ = state
        force = action*self.force_mag
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)
        xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass,0)
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max)
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        done=False
        self.COUNTER+=1
        if x < -self.x_threshold or x > self.x_threshold or self.MAX_STEPS_PER_EPISODE == self.COUNTER:
            done = True

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        state[2] = math.pi - state[2]
        state[0] = self.np_random.uniform(low=-2, high=2)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Box(-1.0, 1.0,shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        action=_action_static_friction(action)
        x, x_dot, theta, theta_dot = self.state
        force = action# self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        print('force app', force)
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0
        print('state is',self.state)

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[1] = 0
        self.state[2]=math.pi
        self.state[3]=3
        self.steps_beyond_done = None
        print('reset')
        print(self.state)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
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
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

