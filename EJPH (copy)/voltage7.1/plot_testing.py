import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
import glob
import seaborn as sns
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN, SAC
from env_custom import CartPoleButter
from custom_callbacks import EvalCustomCallback, EvalThetaDotMetric, moving_average
from matplotlib import rcParams, pyplot as plt
from custom_callbacks import plot_results
import plotly.express as px
from bokeh.palettes import d3
PLOT_TRAINING_REWARD=False
PLOT_EVAL_REWARD=True
TENSION_PLOT = False
TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 100
colorPalette = d3['Category20'][20]
# logdir = ''
colorArr = ['red', 'blue', 'green', 'cyan', 'yellow', 'tan', 'navy', 'black']
scoreArr = np.zeros_like(TENSION_RANGE)
stdArr = np.zeros_like(TENSION_RANGE)
Te=0.05
EP_STEPS = 800

def calculate_angle(prev_value, cos, sin, count=0):
    '''
    :param prev_value:
    :param cos: cosinus
    :param sin: sinus
    :return:
    '''
    if prev_value - np.arctan2(sin, cos) > np.pi:
        count += 1
        return np.arctan2(sin, cos), count
    elif np.arctan2(sin, cos) - prev_value > np.pi:
        count -= 1
        return np.arctan2(sin, cos), count
    return np.arctan2(sin, cos), count


PLOT_EPISODE_REWARD = True
figm1, ax1 = plt.subplots()
figm2, ax2 = plt.subplots()
fig = px.scatter()
fig2 = px.scatter()
# fig = px.scatter(x=[0], y=[0])
for i, tension in enumerate(TENSION_RANGE):
    prev_angle_value = 0.0
    count_tours = 0
    done = False
    if PLOT_EPISODE_REWARD:
        # env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=tension, resetMode='experimental', sparseReward=False)
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, tensionMax=tension,
                             resetMode='experimental')  # CartPoleButter(tensionMax=tension,resetMode='experimental')
        model = DQN.load(f'./EJPH/tension-perf-seed/tension_sim_{tension}_V__best.zip', env=env)
        theta = 0
        cosThetaIni = np.cos(theta)
        sinThetaIni = np.sin(theta)
        rewArr = []
        obs = env.reset(costheta=cosThetaIni, sintheta=sinThetaIni)
        # env.reset()
        thetaArr, thetaDotArr, xArr, xDotArr = [], [], [], []
        for j in range(EP_STEPS):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, _ = env.step(act)
            rewArr.append(rew)
            # if tension==4.7:
            #     env.render()
            angle, count_tours = calculate_angle(prev_angle_value, obs[2], obs[3], count_tours)
            prev_angle_value = angle
            thetaArr.append(angle + count_tours * np.pi * 2)
            thetaDotArr.append(obs[4])
            xArr.append(obs[0])
            xDotArr.append(obs[1])
            if done:
                print(f'ended episode {tension} with {count_tours} tours and {np.sum(rewArr)} reward')
                ax1.plot(thetaArr, '.')
                fig.add_scatter(x=np.linspace(1, EP_STEPS, EP_STEPS), y=thetaArr, name=f'volt: {tension}')
                fig2.add_scatter(x=np.linspace(1, EP_STEPS, EP_STEPS), y=xArr, name=f'volt: {tension}')
                break
                # ax1.savefig(logdir+'/thetaA.pdf')
        ax2.plot(moving_average(rewArr, 20), color=colorPalette[i])
    else:
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=100,
            deterministic=True,
            return_episode_rewards=True,
        )
        scoreArr[i] = np.mean(episode_rewards)
        stdArr[i] = np.std(episode_rewards)
        print('done')
if PLOT_EPISODE_REWARD:
    fig.show()
    fig2.show()
    ax1.legend([str(t) + 'V' for t in TENSION_RANGE], loc='upper right')
    ax2.legend([str(t) + 'V' for t in TENSION_RANGE], loc='upper right')
    ax1.set_xlabel('Time step')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Rewards')
    # plt.title('Effect of the applied tension on the "greedy policy" reward')
    figm2.savefig('./EJPH/plots/episode_seed_rew_tension.pdf')
    figm2.show()
    figm1.savefig(f'./EJPH/plots/episode_seed_theta{theta / np.pi * 180}.pdf')
    figm1.show()
else:
    tensionMax = np.array(TENSION_RANGE)
    plt.plot(tensionMax, scoreArr, 'ro-')
    plt.fill_between(tensionMax, scoreArr + stdArr, scoreArr - stdArr, facecolor='red', alpha=0.5)
    plt.xlabel('Tension (V)')
    plt.ylabel('Rewards')
    plt.title('Effect of the applied tension on the "greedy policy" reward')
    plt.savefig('./EJPH/plots/episode_rew_seed_10000eps')
    plt.show()
    np.savez(
        './EJPH/plots/tension-perf-seed10000ep',
        tensionRange=tensionMax,
        results=scoreArr,
        resultsStd=stdArr
    )
print('done inference on voltages')
# RAINBOW
print('plotting in rainbow for different voltages applied')
EP_LENGTH = 800
scoreArr1 = np.zeros_like(TENSION_RANGE)
scoreArr2 = np.zeros_like(TENSION_RANGE)
scoreArr3 = np.zeros_like(TENSION_RANGE)
scoreArr4 = np.zeros_like(TENSION_RANGE)
scoreArr5 = np.zeros_like(TENSION_RANGE)
p1, p2, p3, p4, p5 = 0.2, 0.4, 0.6, 0.8, 1
for j, tension in enumerate(TENSION_RANGE):
    env = CartPoleButter(Te=Te, N_STEPS=EP_LENGTH, discreteActions=True, tensionMax=tension, resetMode='experimental')
    model = DQN.load(f'./EJPH/tension-perf-seed/tension_sim_{tension}_V__best', env=env)
    # model = DQN.load(logdir + f'/tension-perf-seed/thetaDot10/tension_sim_{tension}_V_.zip_2', env=env)
    episode_rewards = 0
    obs = env.reset(costheta=0.984807753012208, sintheta=-0.17364817766693033)
    for i in range(EP_LENGTH):
        action, _state = model.predict(obs)
        obs, cost, done, _ = env.step(action)
        episode_rewards += cost
        if i == int(EP_LENGTH * p1 - 1):
            scoreArr1[j] = episode_rewards  # np.mean(episode_rewards)
        elif i == int(EP_LENGTH * p2 - 1):
            scoreArr2[j] = episode_rewards
        elif i == int(EP_LENGTH * p3 - 1):
            scoreArr3[j] = episode_rewards
        elif i == int(EP_LENGTH * p4 - 1):
            scoreArr4[j] = episode_rewards
        elif i == int(EP_LENGTH * p5 - 1):
            scoreArr5[j] = episode_rewards
        if done:
            print(f'observations: {obs} and i: {i}')
            break

    print('done')

fillArr = np.zeros_like(scoreArr1)
plt.plot(TENSION_RANGE, scoreArr1 / EP_LENGTH, 'o-r')
plt.fill_between(TENSION_RANGE, scoreArr1 / EP_LENGTH, fillArr, facecolor=colorArr[0], alpha=0.5)
plt.plot(TENSION_RANGE, scoreArr2 / EP_LENGTH, 'o-b')
plt.fill_between(TENSION_RANGE, scoreArr2 / EP_LENGTH, scoreArr1 / EP_LENGTH, facecolor=colorArr[1], alpha=0.5)
plt.plot(TENSION_RANGE, scoreArr3 / EP_LENGTH, 'o-g')
plt.fill_between(TENSION_RANGE, scoreArr3 / EP_LENGTH, scoreArr2 / EP_LENGTH, facecolor=colorArr[2], alpha=0.5)
plt.plot(TENSION_RANGE, scoreArr4 / EP_LENGTH, 'o-c')
plt.fill_between(TENSION_RANGE, scoreArr4 / EP_LENGTH, scoreArr3 / EP_LENGTH, facecolor=colorArr[3], alpha=0.5)
plt.plot(TENSION_RANGE, scoreArr5 / EP_LENGTH, 'o-y')
plt.fill_between(TENSION_RANGE, scoreArr5 / EP_LENGTH, scoreArr4 / EP_LENGTH, facecolor=colorArr[4], alpha=0.5)
plt.hlines(y=1, xmin=min(TENSION_RANGE), xmax=max(TENSION_RANGE), linestyles='--')
plt.grid()
plt.xlabel('Tension (V)')
plt.ylabel('Rewards')
# plt.title('Effect of the applied tension on the "greedy policy" reward')

# for p
plt.legend([f'{int(p1 * 100)}% of episode', f'{int(p2 * 100)}% of episode', f'{int(p3 * 100)}% of episode',
            f'{int(p4 * 100)}% of episode', f'{int(p5 * 100)}% of episode'],
           loc='best')
plt.savefig('./EJPH/plots/episode_rainbow_seed.pdf')
plt.show()

import sys
import os
import numpy as np
import seaborn as sns
import time
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from stable_baselines3 import DQN, SAC
from env_custom import CartPoleButter
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
    model = DQN.load(f'./EJPH/tension-perf-seed/tension_sim_{tension}_V__best', env=env)
    # episode_rewards, episode_lengths = evaluate_policy_episodes(env=env,model=model,n_eval_episodes=100,episode_steps=EP_STEPS)
    THETA_DOT_THRESHOLD=0
    N_TRIALS=10
    THETA_THRESHOLD = np.pi/18

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

plt.boxplot(episodeArr, positions=TENSION_RANGE, patch_artist=True)
plt.grid()
# sns.boxplot(x=TENSION_RANGE,data=episodeArr)
plt.ylabel('mean reward per step')
plt.xlabel('Applied DC motor Tension (V)')
if THETA_THRESHOLD==0:
    #plt.title('Effect of varying tension on greedy policy reward (Θ,Θ_dot)=(0,0)',fontsize = 9)
    plt.savefig('./EJPH/plots/boxplot-control-seed.pdf')
else:
    # plt.title('Effect of varying tension on greedy policy reward (Θ,Θ_dot)=(-10°:10°,0)',fontsize = 9)
    plt.savefig('./EJPH/plots/boxplot-control10-seed.pdf')
plt.show()


TENSION_STD = False
VIOLIN_PLOT =True
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
