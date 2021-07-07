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
from bokeh.palettes import d3
from custom_callbacks import plot_results

PLOT_TRAINING_REWARD=False
PLOT_EVAL_REWARD=True

#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 100
colorPalette = d3['Category20'][20]

#UNCOMMENT SNS for nicer visual plots, comment above plt
# sns.set_context("paper")
# sns.set_style("whitegrid")
# sns.set(style='ticks',rc={"font.size": 10, 'font.family': ['sans-serif'], 'axes.grid': True, 'font.sans-serif': 'Times New Roman'})

def save_show_fig(xArr,yArr,legs,title=None,savename=None):
    for i in range(len(xArr)):
        sns.lineplot(y=yArr[i], x=xArr[i],palette=colorPalette[i])
    if title is not None:
        plt.title(title)
    plt.xlabel('timesteps',)
    plt.ylabel('Rewards')
    plt.legend(legs, loc='best')
    try:
        plt.savefig(savename)
    except:
        print('provide avename for this plot')
    plt.show()

def generate_legends(legends):
    legends = np.array([legend.split('_') for legend in legends])
    return legends


logdir='./plots'
STEPS_TO_TRAIN=100000
EP_STEPS=800
Te=0.05
#FOLDER DIRS
dirTension = './EJPH/tension-perf'
dirStatic = './EJPH/static-friction'
dirDynamic = './EJPH/dynamic-friction'
dirNoise = './EJPH/encoder-noise'
dirAction = './EJPH/plots/action-noise'
dirReset = './EJPH/experimental-vs-random'
#TITLES IF NEEDED
t1="Effect of applied tension on training reward"
t2='Effect of static friction on training reward'
t3='Effect of viscous friction of a pendulum on training reward'
t4='Effect of measurement noise on training reward'
t5='Effect of action noise on training reward (std in %)'
t6='Effect of initialisation on training reward'


def plot_from_npz(filenames, xlabel, ylabel, legends, title=None, plot_std=False,saveName=None):

    for i,filename in enumerate(filenames):
        data = np.load(filename)
        meanRew, stdRew = np.mean(data["results"], axis=1), np.std(data["results"], axis=1, keepdims=False)
        plt.plot(timesteps, meanRew,'o-',color=colorPalette[i])
        if plot_std:
            plt.fill_between(timesteps, meanRew + stdRew, meanRew - stdRew, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend(legends)
    if saveName is not None:
        plt.savefig(saveName)
    plt.show()

    # return filenames


'''
PLOT THE TRAINING reward from csv log, namely monitor files
'''
def reaarange_arr_from_idx(xArr,yArr,idx):
    return [xArr[i] for i in idx],[yArr[i] for i in idx]
def sort_arr_from_legs(xArr,yArr,legs):
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    return [xArr[i] for i in idx],[yArr[i] for i in idx],legs
if PLOT_TRAINING_REWARD:
    xArr,yArr,legs=plot_results('./EJPH/tension-perf (copy)',only_return_data=True)#,title=t1) #'Effect of varying tension on the learning'
    legs = [float(leg) for leg in legs[:,-3]]
    xArr, yArr, legs = sort_arr_from_legs(xArr,yArr,legs)
    save_show_fig(xArr,yArr,legs,savename='./EJPH/plots/tension.pdf')#,title=t1

    xArr,yArr,legs=plot_results('./EJPH/static-friction',title=t2,only_return_data=True)
    legs=[round(float(leg[1:]),4) for leg in legs[:,-2]]
    xArr, yArr, legs = sort_arr_from_legs(xArr,yArr,legs)
    save_show_fig(xArr,yArr,legs,savename='./EJPH/plots/static.pdf')#,title=t2

    xArr,yArr,legs= plot_results('./EJPH/dynamic-friction',title=t3,only_return_data=True)
    legs=[float(leg) for leg in legs[:,-2]]
    xArr, yArr, legs = sort_arr_from_legs(xArr,yArr,legs)
    save_show_fig(xArr,yArr,legs,savename='./EJPH/plots/dynamic.pdf')#,title=t3

    xArr,yArr,legs= plot_results(dirNoise,title=t4, only_return_data=True)
    legs=[round(float(leg),4) for leg in legs[:,-3]]
    xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
    save_show_fig(xArr,yArr,legs,savename='./EJPH/plots/noise.pdf') #,title=t4

    xArr,yArr,legs= plot_results('./EJPH/action-noise',title=t5, only_return_data=True)
    legs=[float(leg)for leg in legs[:,-2]]
    xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
    save_show_fig(xArr,yArr,legs,savename='./EJPH/plots/action_noise.pdf')#,title=t5

    xArr,yArr,legs= plot_results('./EJPH/experimental-vs-random',title=t6, only_return_data=True)
    legs=[leg for leg in legs[:,-2]]
    save_show_fig(xArr,yArr,legs,savename='./EJPH/plots/exp-vs-rand.pdf')#,title=t6

    xArr,yArr,legs= plot_results('./EJPH/seeds',title=t6, only_return_data=True)
    legs = legs[:,-1]
    xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
    save_show_fig(xArr,yArr,[leg[0] for leg in legs], savename='./EJPH/plots/seeds.pdf') #,title=t6
'''
PLOT THE inference reward from .npz, namely EvalCallback logs
'''
if PLOT_EVAL_REWARD:
    title = 'Effect of applied tension on the "greedy policy" reward'

    NUM_TIMESTEPS = 150000
    EVAL_NUM_STEPS = 5000
    timesteps = np.linspace(EVAL_NUM_STEPS, NUM_TIMESTEPS, int(NUM_TIMESTEPS / EVAL_NUM_STEPS))

    xl  ='Timesteps'
    yl  ='Rewards'

    filenames = sorted(glob.glob(dirTension + '/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs=[float(leg) for leg in legs[:,-3]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    legs = [str(leg) + 'V' for leg in legs]
    plot_from_npz(filenames,xl,yl,legends=legs,saveName='./EJPH/plots/greedy_tension.pdf')

    filenames = sorted(glob.glob(dirDynamic + '/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs=[round(float(leg),4) for leg in legs[:,-2]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    plot_from_npz(filenames, xl, yl,legends=legs, saveName='./EJPH/plots/greedy_dynamic.pdf')

    filenames = sorted(glob.glob(dirStatic + '/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs=[round(float(leg[1:]),4) for leg in legs[:,-2]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    plot_from_npz(filenames, xl, yl,legends=legs, saveName='./EJPH/plots/greedy_static.pdf')



    filenames = sorted(glob.glob(dirNoise + '/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs=[round(float(leg),4) for leg in legs[:,-3]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    plot_from_npz(filenames, xl, yl, legends=legs, saveName='./EJPH/plots/greedy_noise.pdf')

    filenames = sorted(glob.glob(dirAction + '/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs=[round(float(leg),4) for leg in legs[:,-3]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    plot_from_npz(filenames, xl, yl, legends=legs, saveName='./EJPH/plots/greedy_action.pdf')

    data = np.load('./EJPH/experimental-vs-random/random.npz')
    data2 = np.load('./EJPH/experimental-vs-random/experimental.npz')
    meanRew=np.mean(data["results"],axis=1)
    meanRew2=np.mean(data2["results"],axis=1,keepdims=False)
    stdRew=np.std(data["results"],axis=1)
    stdRew2=np.std(data2["results"],axis=1)
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
    plt.savefig('./EJPH/exp-vs-rand.pdf')
    plt.show()


#TODO subplot: apprentissage, inference, boxplot