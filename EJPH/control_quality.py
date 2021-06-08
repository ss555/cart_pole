import sys
import os
import numpy as np
import seaborn as sns
import time
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule, plot
from custom_callbacks import plot_results
from utils import evaluate_policy_episodes
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from env_custom import CartPoleButter
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from custom_callbacks import EvalCustomCallback
import pandas as pd
from matplotlib import pyplot as plt
from time import time
start_time=time()
TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
Te=0.05
EP_STEPS=3000
scoreArr=np.zeros_like(TENSION_RANGE)
stdArr=np.zeros_like(TENSION_RANGE)
episodeArr=[]
for i, tension in enumerate(TENSION_RANGE):
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=tension,
                         resetMode='random_theta_thetaDot', sparseReward=False)
    model = DQN.load(f'./EJPH/tension-perf/tension-perf{tension}', env=env)
    # episode_rewards, episode_lengths = evaluate_policy_episodes(env=env,model=model,n_eval_episodes=100,episode_steps=EP_STEPS)
    THETA_DOT_THRESHOLD=10
    N_TRIALS=10
    theta = np.linspace(-np.pi, np.pi, N_TRIALS)
    theta_dot = np.linspace(-THETA_DOT_THRESHOLD, THETA_DOT_THRESHOLD, N_TRIALS)
    arrTest = np.transpose([np.tile(theta, len(theta_dot)), np.repeat(theta_dot, len(theta))])
    episode_rewards = np.zeros((arrTest.shape[0], env.MAX_STEPS_PER_EPISODE), dtype=np.float32)
    lengthArr = np.zeros(arrTest.shape[0], dtype=np.float32)
    for j, elem in enumerate(arrTest):
        done = False
        l = 0
        # episode_rewards,episode_lengths =0,0
        obs = env.reset(costheta=np.cos(elem[0]), sintheta=np.sin(elem[0]), theta_ini_speed=elem[1])
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_rewards[j, l] = reward
            l += 1
        lengthArr[j] = l
    scoreArr[i] = np.mean(episode_rewards[:,1000:])
    stdArr[i] = np.std(episode_rewards[:,1000:])
    episodeArr.append(np.mean(episode_rewards,axis=0)[1000:])#taking the mean of 10 episodes in a steady state
    # epArr = [np.mean(s, axis=0) for s in episodeArr]
    print(scoreArr[i])
c='red'
# sns.set_context("paper")
# sns.set_style("whitegrid")
plt.boxplot(episodeArr,positions=TENSION_RANGE,patch_artist=True)
plt.grid()
# sns.boxplot(x=TENSION_RANGE,data=episodeArr)
plt.ylabel('mean reward per step')
plt.xlabel('Applied DC motor Tension (V)')
plt.title('boxplot distribution of the "greedy" policy reward depending on the tension')
plt.savefig('boxplot-control.pdf')
plt.show()

tensionMax = np.array(TENSION_RANGE)
plt.plot(tensionMax, scoreArr, 'ro-')
plt.fill_between(tensionMax, scoreArr + stdArr, scoreArr - stdArr, facecolor='red', alpha=0.5)
plt.xlabel('Tension (V)')
plt.ylabel('Rewards')
plt.title('Effect of the applied tension on the "greedy policy" reward')
plt.savefig('./EJPH/plots/episode_rew_10000eps')
plt.show()