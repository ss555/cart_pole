#tensorboard --logdir ./sac_cartpole_tensorboard/
import gym
import numpy as np
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from env_custom import CartPoleButter
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import math
import time
from matplotlib import pyplot as plt
from utils import plot, plot_line
import numpy as np
Te = 5e-2
env = CartPoleButter(Te, resetMode='experimental',discreteActions=False)
env.MAX_STEPS_PER_EPISODE = 10000
ENV_NORMALISE=False
if ENV_NORMALISE:
    # Load the saved statistics
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load('envNorm.pkl', env) #'./Transfer_learning/backup'
    # env = VecNormalize.load('./Transfer_learning/backup/envRpiNorm.pkl', env)
    #  do not update them at test time
    # env=VecNormalize(env,norm_obs=True,norm_reward=False,clip_obs=10000,clip_reward=10000)
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False
# resumeDir='./Transfer_learning/backup'
# model = SAC.load("./logs/sac/best_model", env=env)
model = SAC.load("./weights/sac50/best_model", env=env)
# model = SAC.load("./logs/best_model_training.zip", env=env)
obs = env.reset()
env.render()
observations=[]
start_time=time.time()

if ENV_NORMALISE:
    obsArr = [env.get_original_obs()[0]]
else:
    obsArr = [obs]
actArr=[0.0]
timeArr=[0.0]

for i in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, _ = env.step(action)
    if ENV_NORMALISE:
        obsArr.append(env.get_original_obs()[0])
    else:
        obsArr.append(obs)
    # actArr.append(action[0,0])
    actArr.append(action[0])
    timeArr.append(time.time()-start_time)
    #time.sleep(Te)
    env.render()
    if dones:
        env.reset()
plot(obsArr,timeArr,actArr)
# f_a = 13.52
# f_b = -0.72
# f_c = -0.68
# dv = -a * vs[i] + b * u + c * np.sign(vs[i]), donc u=(dv+avs[i]-c*np.sign(vs[i]))/b
# staticV= c/b=1=21PWM(0,08 en action), dynamique K1: 18.8, F
#u=-a/b,donc fk