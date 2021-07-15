'''
2 modes: PLOT_TRAINING_REWARD: plots the training reward from the .csv files
PLOT_EVAL_REWARD: plots evaluation reward from .npz files
'''
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
PLOT_TRAINING_REWARD=True
PLOT_EVAL_REWARD=True
TENSION_PLOT = True
TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]

f_aAr = 20.75180095541654,  # -21.30359185798466,
f_bAr = 1.059719258572224,  # 1.1088617953891196,
f_cAr = 1.166390864012042*np.array([0, 0.1, 1, 10]),  # -0.902272006611719,
f_c=round(1.166390864012042,4)
legsStatic = [round(f_c,4) for f_c in f_cAr]
f_d = 0.09727843708918459,  # 0.0393516077401241, #0.0,#
wAngular = 4.881653071189049,
kPendViscousAr = 0.07035332644615992*np.array([0, 0.1, 1, 10]),  # 0.0,#
kPendViscous = round(0.07035332644615992,4)
legsVisc = [round(kPendViscous,4) for kPendViscous in kPendViscousAr]
#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 100
colorPalette = d3['Category20'][20]

NUM_TIMESTEPS = 150000
EVAL_NUM_STEPS = 5000
timesteps = np.linspace(EVAL_NUM_STEPS, NUM_TIMESTEPS, int(NUM_TIMESTEPS / EVAL_NUM_STEPS))

xl = 'Timesteps'
yl = 'Rewards'

logdir='./plots'
STEPS_TO_TRAIN=100000
EP_STEPS=800
Te=0.05
#FOLDER DIRS
dirTension = './EJPH/tension-perf'
dirStatic = './EJPH/static-friction'
dirDynamic = './EJPH/dynamic-friction'
dirNoise = './EJPH/encoder-noise'
dirAction = './EJPH/action-noise'
dirReset = './EJPH/experimental-vs-random'
#TITLES IF NEEDED
t1="Effect of applied tension on training reward"
t2='Effect of static friction on training reward'
t3='Effect of viscous friction of a pendulum on training reward'
t4='Effect of measurement noise on training reward'
t5='Effect of action noise on training reward (std in %)'
t6='Effect of initialisation on training reward'
#UNCOMMENT SNS for nicer visual plots, comment above plt
# sns.set_context("paper")
# sns.set_style("whitegrid")
# sns.set(style='ticks',rc={"font.size": 10, 'font.family': ['sans-serif'], 'axes.grid': True, 'font.sans-serif': 'Times New Roman'})

def save_show_fig(xArr,yArr,legs=None,title=None,saveName=None, ax=None, fig=None, true_value=None):
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(len(xArr)):
        ax.plot(xArr[i], yArr[i]/EP_STEPS, color=colorPalette[i])
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('timesteps',)
    ax.set_ylabel('Rewards')
    if legs is not None:
        ax.legend(legs, loc='best',bbox_to_anchor=(1.01, 1))
    if fig is not None:
        fig.tight_layout()
    ax.grid()
    try:
        if saveName is not None and fig is not None:
            fig.savefig(saveName)
            plt.show()
    except:
        print('provide saveName for this plot')


def generate_legends(legends):
    legends = np.array([legend.split('_') for legend in legends])
    return legends


#Tension
figT, a = plt.subplots(nrows=2,ncols=2,figsize=(10,10))


#helper fcs
def plot_from_npz(filenames, xlabel, ylabel, legends=None, title=None, plot_std=False,saveName=None, ax=None, fig=None, true_value=None):
    for i,filename in enumerate(filenames):
        data = np.load(filename)
        meanRew, stdRew = np.mean(data["results"], axis=1)/EP_STEPS, np.std(data["results"], axis=1, keepdims=False)/EP_STEPS
        if ax is None:
            fig, ax = plt.subplots()
        if true_value==legends[i]:
            ax.plot(timesteps, meanRew, 'o-', color=colorPalette[i])
        else:
            ax.plot(timesteps, meanRew, 'o-', color=colorPalette[i], facecolors='none')
        if plot_std:
            plt.fill_between(timesteps, meanRew + stdRew, meanRew - stdRew, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    if title is not None:
        ax.set_title(title)
    if legends is not None:
        ax.legend(legends,bbox_to_anchor=(1.01, 1))
    if fig is not None:
        fig.tight_layout()
    if saveName is not None and fig is not None:
        fig.savefig(saveName)
    plt.show()

# return filenames
def reaarange_arr_from_idx(xArr, yArr, idx):
    return [xArr[i] for i in idx], [yArr[i] for i in idx]


def sort_arr_from_legs(xArr, yArr, legs):
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    return [xArr[i] for i in idx], [yArr[i] for i in idx], legs



if __name__=='__main__':
    '''
    PLOT THE TRAINING reward from csv log, namely monitor files
    '''
    if PLOT_TRAINING_REWARD:

        xArr, yArr, legs = plot_results('./EJPH/tension-perf',only_return_data=True)  # ,title=t1) #'Effect of varying tension on the learning'
        legs = [float(leg) for leg in legs[:, -3]]
        xArrT, yArrT, legsT = sort_arr_from_legs(xArr, yArr, legs)
        save_show_fig(xArrT, yArrT, title='Training reward',ax=a[0][0],true_value=7.1)  # ,title=t1

        xArr, yArr, legs = plot_results('./EJPH/static-friction', title=t2, only_return_data=True)
        legs = [round(float(leg[1:]), 4) for leg in legs[:, -2]]
        xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
        save_show_fig(xArr, yArr, legsStatic, saveName='./EJPH/plots/static.pdf')  # ,title=t2
        # save_show_fig(xArr, yArr, legs, saveName='./EJPH/plots/static.pdf')  # ,title=t2

        xArr, yArr, legs = plot_results('./EJPH/dynamic-friction', title=t3, only_return_data=True)
        legs = [round(float(leg),4) for leg in legs[:, -2]]
        xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
        save_show_fig(xArr, yArr, legsVisc, saveName='./EJPH/plots/dynamic.pdf')  # ,title=t3

        xArr, yArr, legs = plot_results(dirNoise, title=t4, only_return_data=True)
        legs = [round(float(leg), 4) for leg in legs[:, -3]]
        xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
        save_show_fig(xArr, yArr, legs, saveName='./EJPH/plots/noise.pdf')  # ,title=t4

        xArr, yArr, legs = plot_results('./EJPH/action-noise', title=t5, only_return_data=True)
        legs = [float(leg) for leg in legs[:, -2]]
        xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
        save_show_fig(xArr, yArr, legs, saveName='./EJPH/plots/action_noise.pdf')  # ,title=t5

        xArr, yArr, legs = plot_results('./EJPH/experimental-vs-random', title=t6, only_return_data=True)
        legs = [leg for leg in legs[:, -2]]
        save_show_fig(xArr, yArr, legs, saveName='./EJPH/plots/exp-vs-rand.pdf')  # ,title=t6

        xArr, yArr, legs = plot_results('./EJPH/seeds', title=t6, only_return_data=True)
        legs = legs[:, -1]
        xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
        save_show_fig(xArr, yArr, [leg[0] for leg in legs], saveName='./EJPH/plots/seeds.pdf')  # ,title=t6
    '''
    PLOT THE inference reward from .npz, namely EvalCallback logs
    '''

    if PLOT_EVAL_REWARD:
        title = 'Effect of applied tension on the "greedy policy" reward'

        filenames = sorted(glob.glob(dirTension + '/*.npz'))
        legs = np.array([legend.split('_') for legend in filenames])
        legs=[float(leg) for leg in legs[:,-3]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        legs = [str(leg) + 'V' for leg in legs]
        plot_from_npz(filenames,xl,yl, ax=a[0][1], title='Inference reward', true_value=7.1)

        filenames = sorted(glob.glob(dirDynamic + '/*.npz'))
        legs = np.array([legend.split('_') for legend in filenames])
        legs=[round(float(leg),4) for leg in legs[:,-2]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        plot_from_npz(filenames, xl, yl,legends=legsVisc, saveName='./EJPH/plots/greedy_dynamic.pdf')
        # plot_from_npz(filenames, xl, yl,legends=legs, saveName='./EJPH/plots/greedy_dynamic.pdf')

        filenames = sorted(glob.glob(dirStatic + '/*.npz'))
        legs = np.array([legend.split('_') for legend in filenames])
        legs=[round(float(leg[1:]),4) for leg in legs[:,-2]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        plot_from_npz(filenames, xl, yl,legends=legsStatic, saveName='./EJPH/plots/greedy_static.pdf')
        # plot_from_npz(filenames, xl, yl,legends=legs, saveName='./EJPH/plots/greedy_static.pdf')



        filenames = sorted(glob.glob(dirNoise + '/*.npz'))
        legs = np.array([legend.split('_') for legend in filenames])
        legs=[round(float(leg),4) for leg in legs[:,-3]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        plot_from_npz(filenames, xl, yl, legends=legs, saveName='./EJPH/plots/greedy_noise.pdf')

        filenames = sorted(glob.glob(dirAction + '/*.npz'))
        legs = np.array([legend.split('_') for legend in filenames])
        legs=[round(float(leg),4) for leg in legs[:,-2]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        plot_from_npz(filenames, xl, yl, legends=legs, saveName='./EJPH/plots/greedy_action.pdf')

        data = np.load('./EJPH/experimental-vs-random/_random_.npz')
        data2 = np.load('./EJPH/experimental-vs-random/_experimental_.npz')
        meanRew=np.mean(data["results"],axis=1)/EP_STEPS
        meanRew2=np.mean(data2["results"],axis=1,keepdims=False)/EP_STEPS
        stdRew=np.std(data["results"],axis=1)/EP_STEPS
        stdRew2=np.std(data2["results"],axis=1)/EP_STEPS
        sns.set_context("paper")
        sns.set_style("whitegrid")
        fillBetween=False
        plt.plot(timesteps, meanRew, 'ro-')
        plt.plot(timesteps, meanRew2, 'bo--')
        if fillBetween:
            plt.fill_between(timesteps,meanRew + stdRew, meanRew - stdRew, facecolor='red', alpha=0.2)
            plt.fill_between(timesteps,meanRew2 + stdRew2, meanRew2 - stdRew2, facecolor='blue', alpha=0.2)
        plt.xlabel('timesteps')
        plt.ylabel('Rewards')
        # plt.title('Effect of initialisation on the "greedy policy" reward from experimental state')#random
        plt.legend(['random','experimental'])
        plt.savefig('./EJPH/plots/exp-vs-rand-greedy.pdf')
        plt.show()
    #helper fcn
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
    if TENSION_PLOT:
        scoreArr = np.zeros_like(TENSION_RANGE)
        stdArr = np.zeros_like(TENSION_RANGE)

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
                env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')#CartPoleButter(tensionMax=tension,resetMode='experimental')
                model = DQN.load(f'./EJPH/tension-perf/tension_sim_{tension}_V__best.zip', env=env)
                theta = 0
                cosThetaIni = np.cos(theta)
                sinThetaIni = np.sin(theta)
                rewArr = []
                obs = env.reset(costheta=cosThetaIni, sintheta=sinThetaIni)
                # env.reset()
                thetaArr, thetaDotArr, xArr, xDotArr = [], [], [], []
                for j in range(EP_STEPS):
                    act,_ = model.predict(obs,deterministic=True)
                    obs, rew, done, _ = env.step(act)
                    rewArr.append(rew)
                    # if tension==4.7:
                    #     env.render()
                    angle, count_tours = calculate_angle(prev_angle_value, obs[2], obs[3], count_tours)
                    prev_angle_value = angle
                    thetaArr.append(angle+count_tours*np.pi*2)
                    thetaDotArr.append(obs[4])
                    xArr.append(obs[0])
                    xDotArr.append(obs[1])
                    if done:
                        print(f'ended episode {tension} with {count_tours} tours and {np.sum(rewArr)} reward')
                        ax1.plot(thetaArr, '.')
                        fig.add_scatter(x=np.linspace(1,EP_STEPS,EP_STEPS), y=thetaArr, name=f'volt: {tension}')
                        fig2.add_scatter(x=np.linspace(1,EP_STEPS,EP_STEPS), y=xArr, name=f'volt: {tension}')
                        break
                        # ax1.savefig(logdir+'/thetaA.pdf')
                ax2.plot(moving_average(rewArr,20), color = colorPalette[i])
                a[1][0].plot(moving_average(rewArr,20), color = colorPalette[i])
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
            ax1.legend([str(t)+'V' for t in TENSION_RANGE], loc='upper right')
            ax2.legend([str(t)+'V' for t in TENSION_RANGE], loc='upper right')
            # a[1][0].legend([str(t)+'V' for t in TENSION_RANGE], loc='upper right')
            ax1.set_xlabel('timesteps')
            ax2.set_xlabel('timesteps')
            a[1][0].set_xlabel('timesteps')
            ax2.set_ylabel('Rewards')
            a[1][0].set_ylabel('Rewards')
            a[1][0].grid()
            a[1][0].set_title('Episode reward per step')
            # plt.title('Effect of the applied tension on the "greedy policy" reward')
            figm2.savefig('./EJPH/plots/episode_rew_tension.pdf')
            figm2.show()
            figm1.savefig(f'./EJPH/plots/episode_theta{theta/np.pi*180}.pdf')
            figm1.show()
        else:
            tensionMax = np.array(TENSION_RANGE)
            plt.plot(tensionMax, scoreArr, 'ro-')
            plt.fill_between(tensionMax, scoreArr + stdArr, scoreArr - stdArr, facecolor='red', alpha=0.5)
            plt.xlabel('Tension (V)')
            plt.ylabel('Rewards')
            plt.title('Effect of the applied tension on the "greedy policy" reward')
            plt.savefig('./EJPH/plots/episode_rew_10000eps')
            plt.show()
            np.savez(
                './EJPH/plots/tension-perf10000ep',
                tensionRange=tensionMax,
                results=scoreArr,
                resultsStd=stdArr
            )
        print('done inference on voltages')
        #BOXPLOT
        TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
        Te = 0.05
        EP_STEPS = 800
        scoreArr = np.zeros_like(TENSION_RANGE)
        stdArr = np.zeros_like(TENSION_RANGE)
        episodeArr = []
        for i, tension in enumerate(TENSION_RANGE):
            env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=tension,
                                 resetMode='experimental', sparseReward=False)
            model = DQN.load(f'./EJPH/tension-perf/tension_sim_{tension}_V__best', env=env)
            # episode_rewards, episode_lengths = evaluate_policy_episodes(env=env,model=model,n_eval_episodes=100,episode_steps=EP_STEPS)
            THETA_DOT_THRESHOLD = 0
            N_TRIALS = 10
            THETA_THRESHOLD = np.pi/18

            if THETA_DOT_THRESHOLD != 0:
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
            # NOT USED
            scoreArr[i] = np.mean(episode_rewards[:, -200:])
            stdArr[i] = np.std(episode_rewards[:, -200:])
            ##boxplot
            episodeArr.append(episode_rewards[:, -200:].flatten())  # taking the mean of 10 episodes in a steady state
            # epArr = [np.mean(s, axis=0) for s in episodeArr]
            print(scoreArr[i])
        a[1][1].boxplot(episodeArr, positions=TENSION_RANGE, patch_artist=True)
        a[1][1].grid()
        # sns.boxplot(x=TENSION_RANGE,data=episodeArr)
        a[1][1].set_ylabel('mean reward per step')
        a[1][1].set_xlabel('Applied DC motor Tension (V)')
        a[1][1].set_title('Reward in a steady state')
        # figT.legend(legsT, #loc="center right",   # Position of legend
        #    borderaxespad=0.1,    # Small spacing around legend box
        #    title="Tension color"  # Title for the legend
        #    ,bbox_to_anchor=(1.05, 1))
        # figT.legend(legsT,bbox_to_anchor=(1, -0.25, 1., .102),title="Voltage",
        #            ncol=8, mode="expand", borderaxespad=0.)
        # figT.legend(legsT,loc='upper center', bbox_to_anchor=(0., 1.05, 1., .102),
        #            )

        figT.tight_layout()
        shrink = 0.05
        for tax,bax in zip(a[0],a[1]):
            tbox = tax.get_position()
            bbox = bax.get_position()
            tax.set_position([tbox.x0, tbox.y0-shrink*bbox.height, tbox.width, (1-shrink)*tbox.height])
            bax.set_position([bbox.x0, bbox.y0, bbox.width, (1-shrink)*bbox.height])

        figT.legend(legsT, loc='upper center', bbox_to_anchor=(0.5, 1), title="Voltage",
                   ncol=len(legsT))

        # handles, labels = a[1][1].get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center')
        figT.savefig('./EJPH/plots/tension_all.pdf')
        figT.show()









#D subplot: apprentissage, inference, boxplot
diffSeed = False
if diffSeed:
    xArr, yArr, legs = plot_results('./EJPH/tension-perf-seed',only_return_data=True)  # ,title=t1) #'Effect of varying tension on the learning'
    legs = [float(leg) for leg in legs[:, -3]]
    xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
    save_show_fig(xArr, yArr, legs, saveName='./EJPH/plots/tension-seed.pdf')  # ,title=t1

    filenames = sorted(glob.glob(dirTension + '-seed/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs = [float(leg) for leg in legs[:, -3]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    legs = [str(leg) + 'V' for leg in legs]
    plot_from_npz(filenames, xl, yl, legends=legs, saveName='./EJPH/plots/greedy_tension_seed.pdf')

RAINBOW = False
if RAINBOW:
    print('plotting in rainbow for different voltages applied')
    EP_LENGTH = 800
    scoreArr1 = np.zeros_like(TENSION_RANGE)
    scoreArr2 = np.zeros_like(TENSION_RANGE)
    scoreArr3 = np.zeros_like(TENSION_RANGE)
    scoreArr4 = np.zeros_like(TENSION_RANGE)
    scoreArr5 = np.zeros_like(TENSION_RANGE)
    p1, p2, p3, p4, p5 = 0.2, 0.4, 0.6, 0.8, 1
    for j, tension in enumerate(TENSION_RANGE):
        env = CartPoleButter(Te=Te, N_STEPS=EP_LENGTH, discreteActions=True, tensionMax=tension,
                             resetMode='experimental', sparseReward=False)
        model = DQN.load(f'./EJPH/tension-perf/tension_sim_{tension}_V__best', env=env)
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
    plt.fill_between(TENSION_RANGE, scoreArr1 / EP_LENGTH, fillArr, facecolor=colorPalette[0], alpha=0.5)
    plt.plot(TENSION_RANGE, scoreArr2 / EP_LENGTH, 'o-b')
    plt.fill_between(TENSION_RANGE, scoreArr2 / EP_LENGTH, scoreArr1 / EP_LENGTH, facecolor=colorPalette[1], alpha=0.5)
    plt.plot(TENSION_RANGE, scoreArr3 / EP_LENGTH, 'o-g')
    plt.fill_between(TENSION_RANGE, scoreArr3 / EP_LENGTH, scoreArr2 / EP_LENGTH, facecolor=colorPalette[2], alpha=0.5)
    plt.plot(TENSION_RANGE, scoreArr4 / EP_LENGTH, 'o-c')
    plt.fill_between(TENSION_RANGE, scoreArr4 / EP_LENGTH, scoreArr3 / EP_LENGTH, facecolor=colorPalette[3], alpha=0.5)
    plt.plot(TENSION_RANGE, scoreArr5 / EP_LENGTH, 'o-y')
    plt.fill_between(TENSION_RANGE, scoreArr5 / EP_LENGTH, scoreArr4 / EP_LENGTH, facecolor=colorPalette[4], alpha=0.5)
    plt.hlines(y=1, xmin=min(TENSION_RANGE), xmax=max(TENSION_RANGE), linestyles='--')
    # plt.plot(TENSION_RANGE, scoreArr1, 'o-r')
    # plt.fill_between(TENSION_RANGE, scoreArr1, fillArr, facecolor=colorPalette[0], alpha=0.5)
    # plt.plot(TENSION_RANGE, scoreArr2, 'o-b')
    # plt.fill_between(TENSION_RANGE, scoreArr2, scoreArr1, facecolor=colorPalette[1], alpha=0.5)
    # plt.plot(TENSION_RANGE, scoreArr3, 'o-g')
    # plt.fill_between(TENSION_RANGE, scoreArr3, scoreArr2, facecolor=colorPalette[2], alpha=0.5)
    # plt.plot(TENSION_RANGE, scoreArr4, 'o-c')
    # plt.fill_between(TENSION_RANGE, scoreArr4, scoreArr3, facecolor=colorPalette[3], alpha=0.5)
    # plt.plot(TENSION_RANGE, scoreArr5, 'o-y')
    # plt.fill_between(TENSION_RANGE, scoreArr5, scoreArr4, facecolor=colorPalette[4], alpha=0.5)
    # plt.hlines(y=EP_LENGTH,xmin=min(TENSION_RANGE),xmax=max(TENSION_RANGE),linestyles='--')
    plt.grid()
    plt.xlabel('Tension (V)')
    plt.ylabel('Rewards')
    # plt.title('Effect of the applied tension on the "greedy policy" reward')

    # for p
    plt.legend([f'{int(p1 * 100)}% of episode', f'{int(p2 * 100)}% of episode', f'{int(p3 * 100)}% of episode',
                f'{int(p4 * 100)}% of episode', f'{int(p5 * 100)}% of episode'],
               loc='best')
    plt.savefig('./EJPH/plots/episode_rainbow.pdf')
    plt.show()
