#tensorboard --logdir ./sac_cartpole_tensorboard/
'''
script to visualise cartpole env with 3 different modes
'''
import math
import gym
import numpy as np
from env_custom import CartPoleRK4
import time
from utils import plot
from matplotlib import pyplot as plt
import cartpole
from scipy.signal import iirfilter

Te = 5e-2
N = 1
# env = gym.make('CartPoleImageC-v0')
# env = CartPoleRK4(discreteActions=False, n=100)#Te=Te,n=N,integrator='semi-euler',resetMode='experimental')
# env = CartPoleRK4(f_c=0, discreteActions=False)
env = CartPoleRK4(f_c=50, discreteActions=False) #limit on f_c is a bit larger than 12*1.059719258572224
# env = CartPoleRK4(integrator='ode',discreteActions=False)
actArr=[0.0]
timeArr=[0.0]
env.reset(xIni=0)
start_time=time.time()
#mode of initialisation
mode='startFromPi' #'iniSpeed' 'oscillate'
ACTION = [1.0]#[1.0]#right
DISCRETE=type(env.action_space)==gym.spaces.discrete.Discrete

if mode=='startFromPi':
    obsArr = [env.reset(costheta=0, sintheta=1, xIni=0)]
elif mode=='iniSpeed':
    obsArr = [env.reset(iniSpeed=0.5)]
else:
    obsArr = [env.reset()]

old_time = start_time

for i in range(2000):
    if DISCRETE:
        obs, rewards, dones, _ = env.step(1)#FOR DISCRETE
    else:
        obs, rewards, dones, _ = env.step(ACTION)#go right DEBUG
    obsArr.append(obs)
    actArr.append(0.0)
    timeArr.append(time.time() - start_time)
    old_time=time.time()
    env.render()
    time.sleep(Te)
    if dones:
        print('reset: '+str(time.time()-start_time))
        env.reset()
        break
obsArr=np.array(obsArr)
theta = np.arctan2(obsArr[:, 3], obsArr[:, 2])
plt.plot(timeArr, theta, 'r.')
plt.show()
plot(obsArr, timeArr, actArr, plotlyUse=True)