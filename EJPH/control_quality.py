import sys
import os
import numpy as np
import seaborn as sns
import time

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))

from stable_baselines3 import DQN, SAC
from src.env_custom import CartPoleButter
from matplotlib import pyplot as plt
from time import time

# sns.set_context("paper")
# sns.set_style("whitegrid")
#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 100

start_time=time()
TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
Te=0.05
EP_STEPS=800

scoreArr=np.zeros_like(TENSION_RANGE)
stdArr=np.zeros_like(TENSION_RANGE)
episodeArr=[]
for i, tension in enumerate(TENSION_RANGE):
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=tension, resetMode='experimental', sparseReward=False)
    model = DQN.load(f'./EJPH/tension-perf/tension_sim_{tension}_V__best', env=env)
    # episode_rewards, episode_lengths = evaluate_policy_episodes(env=env,model=model,n_eval_episodes=100,episode_steps=EP_STEPS)
    THETA_DOT_THRESHOLD=0
    N_TRIALS=10
    THETA_THRESHOLD = 0#np.pi/18

    if THETA_DOT_THRESHOLD!=0:
        theta_dot = np.linspace(-THETA_DOT_THRESHOLD, THETA_DOT_THRESHOLD, N_TRIALS)
    else:
        theta_dot = [0]

    if THETA_THRESHOLD != 0:
        theta = np.linspace(-THETA_THRESHOLD, THETA_THRESHOLD, N_TRIALS)
    else:
        theta = [0]

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
    #NOT USED
    scoreArr[i] = np.mean(episode_rewards[:,-200:])
    stdArr[i] = np.std(episode_rewards[:,-200:])
    ##boxplot
    episodeArr.append(episode_rewards[:,-200:].flatten())#taking the mean of 10 episodes in a steady state
    # epArr = [np.mean(s, axis=0) for s in episodeArr]
    print(scoreArr[i])
plt.plot(arrTest[:,0], arrTest[:,1], '.')
plt.show()
c='red'
scale=2
fig,ax=plt.subplots(figsize=(5*scale,5*scale))
ax.boxplot(episodeArr, positions=TENSION_RANGE, patch_artist=True)
ax.grid()
# sns.boxplot(x=TENSION_RANGE,data=episodeArr)

ax.set_ylabel('mean reward per step')
ax.set_xlabel('Applied DC motor Tension (V)')

#INSET
# ax_new = fig.add_axes([0.35, 0.2, 0.5, 0.5])
# ax_new.boxplot(episodeArr[2:], positions=TENSION_RANGE[2:], patch_artist=True)
# # ax_new.set_xticks('')
# plt.setp(ax_new.get_yticklabels(), visible=False)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset,inset_axes
# axins = inset_axes(ax, width="60%",height="80%", loc='lower right')#,bbox_to_anchor=())
axins = inset_axes(ax,width="60%",height="80%", loc='lower right',borderpad=2)#,bbox_to_anchor=())
axins.boxplot(episodeArr[2:], positions=TENSION_RANGE[2:], patch_artist=True)

x1, x2, y1, y2 = 4, 12.2, 0.99, 1.0
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.grid()
# plt.setp(axins.get_yticklabels(), visible=False)
# plt.setp(axins.get_xticklabels(), visible=False)
mark_inset(ax, axins, loc1=3, loc2=4,visible=True,edgecolor='red')

if THETA_THRESHOLD==0:
    #plt.title('Effect of varying tension on greedy policy reward (Θ,Θ_dot)=(0,0)',fontsize = 9)
    plt.savefig('./EJPH/plots/boxplot-control.pdf')
else:
    # plt.title('Effect of varying tension on greedy policy reward (Θ,Θ_dot)=(-10°:10°,0)',fontsize = 9)
    plt.savefig('./EJPH/plots/boxplot-control10.pdf')
plt.show()

np.savez('test.npz',episodeArr=episodeArr,TENSION_RANGE=TENSION_RANGE)

TENSION_STD = False
VIOLIN_PLOT =False
if VIOLIN_PLOT:
    sns.violinplot(x='Voltage',y='Reward per step', data = episodeArr, scale_hue=True, positions=TENSION_RANGE, linewidth=0.5, bw=10, trim=True, inner="quart")
    plt.savefig('./EJPH/v.pdf')

    sns.violinplot(x='Voltage',y='Reward per step', data = episodeArr[:3],positions=TENSION_RANGE[:3])
    plt.legend()
    plt.title('distribution of rewards depending on applied tension')
    plt.savefig('./EJPH/v.pdf')
    plt.show()

    sns.violinplot(x='Voltage',y='Reward per step', data = episodeArr[3:],positions=TENSION_RANGE[3:])
    plt.legend()
    plt.title('distribution of rewards depending on applied tension')
    plt.savefig('./EJPH/v2.pdf')
    plt.show()

if TENSION_STD:

    plt.plot(episodeArr,TENSION_RANGE)
    plt.show()

    tensionMax = np.array(TENSION_RANGE)
    plt.plot(tensionMax, scoreArr, 'ro-')
    plt.fill_between(tensionMax, scoreArr + stdArr, scoreArr - stdArr, facecolor='red', alpha=0.5)
    plt.xlabel('Tension (V)')
    plt.ylabel('Rewards')
    plt.title('Effect of the applied tension on the "greedy policy" reward')
    plt.savefig('./EJPH/plots/episode_rew_10000eps')
    plt.show()