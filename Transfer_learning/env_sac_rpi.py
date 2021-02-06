import gym
import math
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import SAC
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
    cost=2+np.cos(theta)-abs(x)/100
    return cost
def cbf(gpio, level, tick):	
	pi.set_PWM_dutycycle(24,  0)
	pi.stop()
	#raise Exception('capteur fin course') 	
pi.set_glitch_filter(17, 50)
pi.set_glitch_filter(18, 50)
#chariot
start_time = time.time()
lengthLimit=5000
posChariot = 0
oldPosChariot =0
VitPendule = 0
#pendule
posPendule = math.pi
oldPosPendule = math.pi
VitPendule = 0

#global variables
done=0
rPWM=60
SCALE=40#Puissance
COMPENSATE_STATIC=42
ENCODER_GLITCH=200
pi.set_glitch_filter(17, 50)
pi.set_glitch_filter(18, 50)
pi.set_glitch_filter(19, ENCODER_GLITCH)
pi.set_glitch_filter(26, ENCODER_GLITCH)
pi.set_glitch_filter(20, 10)
pi.set_glitch_filter(21, 10)


def callbackPendule(way):
    global posPendule
    posPendule+=way/300*math.pi
    #posPendule=posPendule%(math.pi*2)
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
    posPendule = rescale_angle(posPendule)
    oldPosPendule = posPendule
    return vitPendule

	
def callbackChariot(way):
	global posChariot, done	 
	posChariot += way/1000	
	# sens encodeur +6250:left 0-middle -6250: right	
	if posChariot>5.1 and not done:		
		done=1
		pi.write(16,1); #1-right 0-left
		pi.set_PWM_dutycycle(24,  rPWM)
	elif posChariot<-5.1 and not done:
		done=1
		pi.write(16,0); #1-right 0-left
		pi.set_PWM_dutycycle(24,  rPWM)               
    
    
def getState():
    global posChariot, posPendule
    vChariot=getVitChariot()
    vPendule=getVitPendule()
    rPosPendule = rescale_angle(posPendule)
    info('got state: {}'.format([posChariot, vChariot, rPosPendule, vPendule]))
    return [posChariot, vChariot, rPosPendule, vPendule]

def applyForce(action):
	global SCALE, COMPENSATE_STATIC
	if action > 0:
		pi.write(16,1); #1-right 0-left
	else:
		pi.write(16,0); #1-right 0-left
	pi.set_PWM_dutycycle(24, abs(action)*SCALE + COMPENSATE_STATIC)
    
def rescale_angle(theta):
        return math.atan2(math.sin(theta),math.cos(theta))    
    
class CartPoleSacPi(gym.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE=3000

        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 50
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        global done
        err_msg = "action invalid"
        assert self.action_space.contains(action), err_msg
        assert self.observation_space.contains(self.state), err_msg
        applyForce(action)
        self.COUNTER += 1
        time.sleep(0.049)#Te
        #receive new state
        x, x_dot, theta, theta_dot = getState()
        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        info('x {}, x_dot {}, theta {}, theta_dot {}'.format(x, x_dot, theta, theta_dot))
        if abs(x)>self.x_threshold or self.COUNTER>self.MAX_STEPS_PER_EPISODE:
            done = 1
        cost = reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost-100
        return self.state, cost, done, {}
    def reset(self):
        global done
        # GOTO center slowly
        self.COUNTER=0
        info('reset')
        x, _, _, _ = getState()
        if x > 0:
            pi.write(16,1); #1-right 0-left
            pi.set_PWM_dutycycle(24,  100) #max 255
            while x>0:                
                time.sleep(0.001)
                x, _, _, _ = getState()	
        elif x < 0:
            pi.write(16,0); #1-right 0-left
            pi.set_PWM_dutycycle(24,  100) #max 255		
            while x<0:
                time.sleep(0.001)
                x, _, _, _ = getState()
        pi.set_PWM_dutycycle(24,  0)
        done=0
        time.sleep(10)
        x, x_dot, theta, theta_dot = getState() #
        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        info('x {}, x_dot {}, theta {}, theta_dot {}'.format(x, x_dot, theta, theta_dot))        
        return self.state
    
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
        
        
        
        
        
        env=CartPoleSacPi()
        # Load the saved statistics
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load('envNorm.pkl', env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False


        
        model = SAC.load("./best_model", env=env)
        start_time = time.time()
        old_time_chariot = time.time()
        old_time_pendule = time.time()        
        
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if dones:
                env.reset()
        
    finally:        
        pi.set_PWM_dutycycle(24,  0)
        cb1.cancel()
        cb2.cancel()		
        decoderPendule.cancel()
        decoderChariot.cancel()
        pi.stop()
        logging.exception("Fatal error in main loop")
        info('smth went wrong')
        
