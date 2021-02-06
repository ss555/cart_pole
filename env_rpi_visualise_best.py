""""
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
    cost=2+np.cos(theta)-x**2/100
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
    if (posChariot-oldPosChariot)>6:#because of clipping posChariot values to 0;2pi
        oldPosChariot+=math.pi*2
    elif (oldPosChariot-posChariot)>6:
        posChariot+=math.pi*2
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
        self.x_threshold = 5.0
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
        time.sleep(0.03)#Te
        #receive new state
        x, x_dot, theta, theta_dot = getState()
        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        print('x {}, x_dot {}, theta {}, theta_dot {}'.format(x, x_dot, theta, theta_dot))
        info('x {}, x_dot {}, theta {}, theta_dot {}'.format(x, x_dot, theta, theta_dot))
        
        if abs(x)>self.x_threshold or self.COUNTER>self.MAX_STEPS_PER_EPISODE:
            done = True
        else:
            done = False
        info('action taken: {}, done: {}'.format(action, done))
        cost = reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost-self.MAX_STEPS_PER_EPISODE
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
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    try:
    
        cb1 = pi.callback(17, pigpio.FALLING_EDGE, cbf)
        cb2 = pi.callback(18, pigpio.FALLING_EDGE, cbf)
        decoderPendule = decoder(pi, 19, 26, callbackPendule) #pendule
        decoderChariot = decoder(pi, 20, 21, callbackChariot) #chariot
        env=CartPoleCusBottomDqnPi(pwm_mag=140)
        policy_kwargs = dict(net_arch=[256, 256])
        model = DQN.load("./stable_baselines_dqn_best/best_model")
        start_time=time.time()
        old_time_chariot = time.time()
        old_time_pendule = time.time()
        
        for i in range(1):
            obs=env.reset()
            dones=False
            while not dones:
                action, _states = model.predict(obs)
                obs, rew, dones, _ = env.step(action)
                print(dones)
        obs=env.reset()
        cb1.cancel()
        cb2.cancel()		
        decoderPendule.cancel()
        decoderChariot.cancel()
        pi.stop()
    except:
        pi.set_PWM_dutycycle(24,  0)
        pi.stop()
        logging.exception("Fatal error in main loop")
        info('smth went wrong')
        
        
