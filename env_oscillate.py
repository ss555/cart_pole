#tensorboard --logdir ./sac_cartpole_tensorboard/
import gym
import numpy as np
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC, DDPG, TD3, DQN
from env_custom import CartPoleCosSinFriction, CartPoleCosSinDev, CartPoleCosSinObsNDev
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import math
import time
from matplotlib import pyplot as plt
from utils import plot,plot_line
import numpy as np
Te = 5e-2
env = CartPoleCosSinDev()#CartPoleCusBottom()CartPoleCosSin() #
# env= CartPoleCosSinFriction()#CartPoleCusBottom()CartPoleCosSin() #
env.MAX_STEPS_PER_EPISODE = 10000
# Load the saved statistics
env = DummyVecEnv([lambda: env])
# env = VecNormalize.load('envNorm.pkl', env)
#  do not update them at test time
env=VecNormalize(env,norm_obs=True,norm_reward=False,clip_obs=10000,clip_reward=10000)
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

#model = SAC.load("./logs/best_model", env=env)
# model = SAC.load("./logs/best_model_training.zip", env=env)
obs = env.reset()
observations=[]
start=time.time()




# for i in range(300):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, _ = env.step(action)
#     obsArr.append(env.get_original_obs()[0])
#     actArr.append(action[0,0])
#     timeArr.append(time.time()-start_time)
#     time.sleep(Te)
#     if dones:
#         env.reset()
# plot(obsArr,timeArr)
# f_a = 13.52
# f_b = -0.72
# f_c = -0.68
# dv = -a * vs[i] + b * u + c * np.sign(vs[i]), donc u=(dv+avs[i]-c*np.sign(vs[i]))/b
# staticV= c/b=1=21PWM(0,08 en action), dynamique K1: 18.8, F
#u=-a/b,donc fk
env = CartPoleCosSinDev()
env.reset()
start_time = time.time()
for i in range(16):
    #action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ = env.step([0.392])#PWM - 100
    obsArr.append(obs)
    #actArr.append(action[0])
    time.sleep(Te)
    timeArr.append(time.time() - start_time)
    #env.render()
    if dones:
        break
        env.reset()
for i in range(16):
    #action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ = env.step([-0.392])
    time.sleep(Te)
    obsArr.append(obs)
    #actArr.append(action[0])
    timeArr.append(time.time()-start_time)
    if dones:
        break
        env.reset()
# plot(timeArr)
plot(obsArr, timeArr)

# 3000inc/s=0.225ms/s
# while 1:
#     action,_states=model.predict(obs,deterministic=False)
#     # obs, rewards, dones, _ = env.step([0.0])
#     obs, rewards, dones, _ = env.step([action])
#     obs=obs[0]
#     # obs, rewards, dones, _ = env.step([[0.0]])
#     # observations.append(math.atan2(obs[3], obs[2]))
#     # plt.plot(observations)
#     # plt.xlabel('Timesteps')
#     # plt.ylabel('en rad')
#     # plt.show()
#     env.render()
#     time.sleep(Te)
#     if dones:
#         print('reset: '+str(time.time()-start))
#         env.reset()
#         break