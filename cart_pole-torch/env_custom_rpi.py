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
import pigpio
def COST_1000(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = np.sign(r2) * ((10 * r2) ** 2) - 4 * abs(x) ** 2
    return cost
def reward_fn(x,theta,action=0.0):
    cost=2+np.cos(theta)-x**2/25
    if theta<math.pi/12 and theta>-math.pi/12:
        cost+=3
    return cost

def reward_fn2(x,theta):
    cost=3
    if abs(theta)>math.pi/15:
        cost-=2
    cost-=abs(x)
    return cost
def reward_fn3(x,theta):
    cost=3+np.cos(theta)*3-abs(x)
    return cost
def reward_fn4(x,theta):
    cost=4-abs(theta)-abs(x)
    return cost

class CartPoleCusBottom(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts at the bottom and the goal is to swing up and balance
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -180°           180°
        3	Pole Velocity At Tip

    Actions:
        Type: box(2)
        Num	Action
        -Force_threshold 	Push cart to the left
        1*Force_threshold	Push cart to the right

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Cart Position is more than ±x_threshold (center of the cart reaches the edge of the display)
        Episode length is greater than N_counter
        Solved Requirements
        Considered solved when the average reward is greater than or equal to x over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE=3000
        self.gravity = 10
        # 1 0.1 0.5 original
        self.masscart = 1
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 20
        self.tau = 0.03  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        # self.theta_threshold_radians = 20 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # self.v_max=1.5
        # self.w_max=1
        # FOR DATA
        self.v_max = 500
        self.w_max = 500

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 2,
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
    def applyForce(self, force):
        pi.set_PWM_dutycycle(24, force)
    def getState(self):
        #vitChariot = getVitChariot(posChariot)
        #vitPendule = getVitPendule(posPendule)
        return posChariot, getVitChariot(posChariot), posPendule, getVitPendule(posPendule)
    def step(self, action):
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = action*self.force_mag
        # apply force = action
        applyForce(force)
        #receive new state
        x, x_dot, theta, theta_dot = getState()
        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER>self.MAX_STEPS_PER_EPISODE:
            done = True

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}
    def reset(self):
        # GOTO center slowly

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

