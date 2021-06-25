"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import gym
import math
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
#rpi
import time
from matplotlib import pyplot as plt
from rotary_encoder import decoder
import logging
from logging import info, basicConfig
import pigpio
from pynput import keyboard
import sys
PI_INPUT = pigpio.INPUT
PI_PUD_UP = pigpio.PUD_UP
pi = pigpio.pi()

if not pi.connected:
    exit()
    print('exit')
    
logname='DQN_DEBUG'
basicConfig(filename=logname,
			filemode='w',#'a' for append
			format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
			datefmt='%H:%M:%S',
			level=logging.DEBUG)
            
def reward_fn(x,theta,action=0.0):
    cost=2+np.cos(theta)-x**2/25
    if theta<math.pi/12 and theta>-math.pi/12:
        cost+=1
    return cost
def cbf(gpio, level, tick):	
	pi.set_PWM_dutycycle(24,  0)
	pi.stop()
	#raise Exception('capteur fin course') 	
pi.set_glitch_filter(17, 50)
pi.set_glitch_filter(18, 50)
#chariot
start_time = time.time()
lengthLimit=5500
homingRequest=False
posChariot = 0
oldPosChariot =0
VitPendule = 0
#pendule
posPendule = math.pi
oldPosPendule = math.pi
VitPendule = 0

def cbf(gpio, level, tick):	
	pi.set_PWM_dutycycle(24,  0)
	pi.stop()	
pi.set_glitch_filter(17, 50)
pi.set_glitch_filter(18, 50)

def callbackPendule(way):
    global posPendule
    posPendule+=way/300*math.pi
    posPendule=posPendule%(math.pi*2)
    #info("pos pendule={}".format(posPendule))#*180/600  
    

    
def getVitChariot():
    global posChariot, oldPosChariot, old_time_chariot
    vitChariot = ((posChariot-oldPosChariot)/(time.time()-old_time_chariot))
    old_time_chariot = time.time()
    oldPosChariot = posChariot
    return vitChariot
    
def getVitPendule():
    global posPendule, oldPosPendule, old_time_pendule
    vitPendule = ((posPendule-oldPosPendule)/(time.time()-old_time_pendule))
    old_time_pendule = time.time()
    oldPosPendule = posPendule
    return vitPendule
            
def callbackChariot(way):
	global posChariot, lengthLimit, homingRequest	 
	posChariot += way/1000
	#info("pos chariot={}".format(posChariot))	
	
        
def applyForce(force, direction):
    if direction==1:
        pi.write(16,1); #1-right 0-left
    elif direction==0:
        pi.write(16,0); #1-right 0-left
    else:
        print('unknow direction')
    pi.set_PWM_dutycycle(24, force)
    
    
def getState():
    global posChariot, posPendule
    vChariot=getVitChariot()
    vPendule=getVitPendule()
    return [posChariot, vChariot, posPendule, vPendule]
    
class CartPoleCusBottomDqnPi(gym.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                pwm_mag: int=5,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE=3000

        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.5
        # FOR DATA
        self.v_max = 50
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 3,
            self.w_max])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.pwm_mag = pwm_mag
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        global homingRequest
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        assert self.observation_space.contains(self.state), err_msg
        if action==0:
            applyForce(self.pwm_mag,0)
        elif action==1:
            applyForce(self.pwm_mag,1)
        elif action==2:
            applyForce(0,0)
        else:
            print('invalid action space asked')
            exit(-1)
        self.COUNTER += 1
        
        #receive new state
        x, x_dot, theta, theta_dot = getState()
        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        info('x {}, x_dot {}, theta {}, theta_dot {}'.format(x, x_dot, theta, theta_dot))
        if abs(x)>self.x_threshold or self.COUNTER>self.MAX_STEPS_PER_EPISODE:
            done = True
        else:
            done = False
        cost = reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost-100
        return self.state, cost, done, {}
    def reset(self):
        # GOTO center slowly
        self.COUNTER=0
        info('reset')
        x, x_dot, theta, theta_dot = getState()
        if x > 0:
            homingRequest=True
            pi.write(16,1); #1-right 0-left
            pi.set_PWM_dutycycle(24,  70) #max 255
            while x>0:                
                time.sleep(0.01)
                x, _, _, _ = getState()	
        elif x < 0:
            homingRequest=True
            pi.write(16,0); #1-right 0-left
            pi.set_PWM_dutycycle(24,  70) #max 255		
            while x<0:
                time.sleep(0.01)
                x, _, _, _ = getState()
        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        info('x {}, x_dot {}, theta {}, theta_dot {}'.format(x, x_dot, theta, theta_dot))
        pi.set_PWM_dutycycle(24,  0)
        return np.array(getState())

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

'''
    except:
        info('smth went wrong')
        pi.set_PWM_dutycycle(24,  0)
'''
if __name__ == "__main__":
    try:
    
        cb1 = pi.callback(17, pigpio.FALLING_EDGE, cbf)
        cb2 = pi.callback(18, pigpio.FALLING_EDGE, cbf)
        decoderPendule = decoder(pi, 19, 26, callbackPendule) #pendule
        decoderChariot = decoder(pi, 20, 21, callbackChariot) #chariot
        env=CartPoleCusBottomDqnPi(pwm_mag=140)
        eval_callback = EvalCallback(env, best_model_save_path='stable_baselines_dqn_best/',
                                 log_path='./logs/', eval_freq=5000,
                                 deterministic=True, render=False)  
        policy_kwargs = dict(net_arch=[256, 256])
        model = DQN(MlpPolicy, env, learning_starts=500, batch_size=256, verbose=0, policy_kwargs=policy_kwargs)
        start_time=time.time()
        model.load("./stable_baselines_dqn_best/best_model.zip")
        old_time_chariot = time.time()
        old_time_pendule = time.time()
        
        for i in range(3):
            state=env.reset()
            while not done:
                action, _states = model.predict(state)
                state, rewards, done, info = env.step(action)
            

        model.save("dqn_cartpole")
        cb1.cancel()
        cb2.cancel()		
        decoderPendule.cancel()
        decoderChariot.cancel()
        pi.stop()
    except:
        logging.exception("Fatal error in main loop")
        info('smth went wrong')
        pi.set_PWM_dutycycle(24,  0)
        