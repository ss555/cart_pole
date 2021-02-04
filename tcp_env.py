import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import socket
HOST = '169.254.161.71'#'c0:3e:ba:6c:9e:eb'#'10.42.0.1'#wifiHot #'127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432
def reward_fn(x,theta,action=0.0):
    cost=2+np.cos(theta)-x**2/25
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
        3	Pole Velocity At Tip      -Inf            Inf

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
                 seed: int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = 3000
        self.FORMAT = 'utf-8'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        # self.theta_threshold_radians = 20 * 2 * math.pi / 360
        self.x_threshold = 5.0
        self.v_max = 500
        self.w_max = 500

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 2,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype = np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            self.conn, addr = s.accept()
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.COUNTER += 1
        with self.conn:
            self.conn.sendall(action.encode(self.FORMAT))
            sData = self.conn.recv(124).decode(self.FORMAT)
            self.state = np.array(sData.split(',')).astype(np.float32)
        done = False
        x = self.state[0]
        theta = self.state[2]
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER>self.MAX_STEPS_PER_EPISODE:
            done = True

        cost = reward_fn(x, theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost-100
        return self.state, cost, done, {}
    def reset(self):
        x,_,_,_=self.state
        with self.conn:
            if x>0:
                action = 0.2
            else:
                action = -0.2
            self.conn.sendall(action.encode(self.FORMAT))
            while True:
                sData = self.conn.recv(1024).decode(self.FORMAT)
                self.state = np.array(sData.split(',')).astype(np.float32)
                x, _, _, _ = self.state
                if abs(x)<0.05:
                    break
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
