'''
2 modes: PLOT_TRAINING_REWARD: plots the training reward from the .csv files
PLOT_EVAL_PARAMS: plots evaluation reward from .npz files
'''
import sys
import os
import pickle
import numpy as np
sys.path.append(os.path.abspath('/'))
sys.path.append(os.path.abspath('../tmp'))
import glob
from src.custom_callbacks import moving_average
from matplotlib import pyplot as plt
from src.custom_callbacks import plot_results
import subprocess
from src.utils import inferenceResCartpole, HandlerColormap,label_legend_save_2,putIndInside_2_2
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle

cmap = plt.cm.brg
dqn7real = './deepRl/real-cartpole/dqn_7.1V'
filenames_expe = ['./deepRl/real-cartpole/dqn_7.1V/inference_results.npz', './weights/dqn50-real/pwm51/inference_results.npz']

PLOT_TRAINING_REWARD = True
PLOT_EVAL_PARAMS = True
TENSION_PLOT = True#plot with tension
PLOT_ACTION_NOISE = False
CROP = True#crop PDFs
PLOT_DOUBLE_PARAMS = False
TENSION_RANGE = np.array([2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12])




f_aAr = 20.75180095541654,  # -21.30359185798466,
f_bAr = 1.059719258572224,  # 1.1088617953891196,
f_cAr = 1.166390864012042 * np.array([0, 0.1, 1, 10]),  # -0.902272006611719,
f_d = 0.09727843708918459,  # 0.0393516077401241, #0.0,#
wAngular = 4.881653071189049,
kPendViscousAr = 0.0706*np.array([0, 0.1, 1, 10]).T
legsStatic = np.array([np.round(f_cc,4) for f_cc in f_cAr]).T# 0.0,#
legsVisc = [round(kPendViscous,4) for kPendViscous in kPendViscousAr]

fontsize = 20
FONT_SIZE_LABEL = 20
FONT_SIZE_AXIS = 20
#ffc6c4,#f4a3a8,#e38191,#cc607d,#ad466c,#8b3058,#672044
#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = fontsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 200
legendfontsize = 14
# colorPalette = d3['Category20'][20]
# set labels outside




NUM_Timesteps = 150000
EVAL_NUM_STEPS = 5000
Timesteps = np.linspace(EVAL_NUM_STEPS, NUM_Timesteps, int(NUM_Timesteps / EVAL_NUM_STEPS))

xl = 'Time step'
yl = 'Normalized return'

logdir='./deepRl/plots'
os.makedirs(logdir, exist_ok=True)
STEPS_TO_TRAIN=100000
EP_STEPS=800
Te=0.05
#FOLDER DIRS
dirTension = './deepRl/tension-perf'
dirStatic = './deepRl/static-friction'
dirDynamic = './deepRl/dynamic-friction'
dirNoise = './deepRl/encoder-noise'
dirAction = './deepRl/action-noise'
dirReset = './deepRl/experimental-vs-random'
#TITLES IF NEEDED
t1="Effect of Applied voltage on training reward"
t2='Effect of static friction on training reward'
t3='Effect of viscous friction of a pendulum on training reward'
t4='Effect of measurement noise on training reward'
t5='Effect of action noise on training reward (std in %)'
t6='Effect of initialisation on training reward'
#TITLES of legends IF NEEDED
legsNoiseTitle = "$\sigma_\Theta$ $\in$ [0.0, 17.45] $mrad$"
# legsNoiseTitle = "$\sigma_\Theta$ $\in$ [2.4, 9.4] $mrad$"
legsViscTitle = "Viscous friction $\in$ [0.0, 0.7] $N.s.rad^{-1}$ "
legsStaticTitle = "Static friction $\in$ [0.0, 1.7] $N.kg^{-1}$"
legsActionTitle = "$\sigma_U/U$ $\in$ [0.0, 10.0] %"

#plots
SCALE = 1.2 #0.8 # figsize (7.5,...)
WIDTH=12.5
figT, a     = plt.subplots(nrows=2, ncols=2, figsize=(SCALE*WIDTH,SCALE*8))#Tension
figP, aP    = plt.subplots(nrows=2, ncols=2, figsize=(SCALE*WIDTH,SCALE*8)) #parametric study
figSt, axSt = plt.subplots(nrows=2, ncols=1, figsize=(SCALE*6,SCALE*2*3.7125))
figDy, axDy = plt.subplots(nrows=2, ncols=1, figsize=(SCALE*6,SCALE*2*3.7125))
figNo, axNo = plt.subplots(nrows=2, ncols=1, figsize=(SCALE*6,SCALE*2*3.7125))
figAc, axAc = plt.subplots(nrows=2, ncols=1, figsize=(SCALE*6,SCALE*2*3.7125))

for i in range(2):
    for j in range(2):
        aP[i][j].xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
def save_show_fig(xArr,yArr,legs=None,title=None,saveName=None, ax=None, fig=None, true_value_index=None,experimental_value_index=None):
    # PLOT THE TRAIN REWARD IN THE AXIS
    if ax is None:
        fig, ax = plt.subplots(figsize=(SCALE*6,SCALE*3.7125))
    #standardisize in format
    xArr, yArr = np.array(xArr), np.array(yArr)
    if len(xArr.shape)==1 and type(xArr[0])==np.int64:
    # if len(xArr.shape)==1 and xArr[0].shape[0]==1:
        xArr = xArr.reshape(1, -1)
        yArr = yArr.reshape(1, -1)
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    if len(xArr)>18:
        print('too many values to print, funct error')
        raise IOError
    i_max=xArr.shape[0]
    for i in range(xArr.shape[0]):
        if i==true_value_index:
            ax.plot(xArr[i], yArr[i] / EP_STEPS, legscolor=cmap(i / i_max))
        elif i==experimental_value_index:
            ax.plot(xArr[i], yArr[i] / EP_STEPS, color=cmap(i / i_max),linewidth=4.0)
        else:
            ax.plot(xArr[i], yArr[i] / EP_STEPS, color=cmap(i / i_max))
    if title is not None:
        ax.set_title(title, fontsize=FONT_SIZE_LABEL)
    
    ax.set_xlabel('Time step', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(yl, fontsize=FONT_SIZE_LABEL)

    if legs is not None:
        ax.legend(legs, loc='best')#,bbox_to_anchor=(1.01, 1))
    if fig is not None:
        fig.tight_layout()
    
    try:
        if saveName!=None and fig is not None:
            fig.savefig(saveName)
            #plt.show()
    except:
        print('provide saveName for this plot')

def findInd(array,elem): #FIND INDX OF ELEMENT IN ARR
    try:
        for i, elArr in enumerate(array):
            if elem==elArr:
                return i
        return -1
    except:
        print('error occured in findInd func')


#helper fcs
def plot_from_npz(filenames, xlabel, ylabel, legends=None, legends_title=None, title=None, plot_std=False,saveName=None, ax=None, fig=None, true_value_index=None, target = None, infExpDir = None):
    if target is not None:
        tensionRange = np.array([name.split('_') for name in filenames])
        tensionRange = [round(float(seg), 2) for seg in tensionRange[:, -3]]
        filenames = [filenames[findInd(tensionRange,target)]] #trim
    i_max=len(filenames)#max colormap
    for i,filename in enumerate(filenames):
        data = np.load(filename)
        meanRew, stdRew = np.mean(data["results"], axis=1)/EP_STEPS, np.std(data["results"], axis=1, keepdims=False)/EP_STEPS
        if ax is None:
            fig, ax = plt.subplots(figsize=(SCALE*6,SCALE*3.7125))
        if i == true_value_index:
            ax.plot(Timesteps, meanRew, color=cmap(i / i_max))
            # ax.plot(Timesteps, meanRew, data['results'] color=cmap(i / i_max))
            # ax.plot(Timesteps, meanRew, 'o--', fillstyle='none', color=cmap(i / i_max))
        else:
            ax.plot(Timesteps, meanRew, color=cmap(i / i_max))
        if plot_std:
            plt.fill_between(Timesteps, meanRew + stdRew, meanRew - stdRew, alpha=0.2)
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    
    if title is not None:
        ax.set_title(title, fontsize=FONT_SIZE_LABEL)
    if legends_title is not None:
        cmap_handles = [Rectangle((0, 0), 1, 1)]  # [Rectangle((0, 0), 1, 1) for _ in cmap]
        handler_map = dict(zip(cmap_handles, [HandlerColormap(cmap, num_stripes=i_max)]))
        ax.legend(handles=cmap_handles, labels=[legends_title],
                       handler_map=handler_map, fontsize=legendfontsize, loc='lower right')
    if fig is not None:
        fig.tight_layout()
    if saveName is not None and fig is not None:
        fig.savefig(saveName)
    #plt.show()

# return filenames
def reaarange_arr_from_idx(xArr, yArr, idx): # THE array indexed to be returned
    return [xArr[i] for i in idx], [yArr[i] for i in idx]

def sort_arr_from_legs(xArr, yArr, legs): #sort the legs and corresponding arrays with it : x,y
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    return [xArr[i] for i in idx], [yArr[i] for i in idx], legs



if __name__=='__main__':
    '''
    PLOT THE inference reward from .npz, namely EvalCallback logs
    '''

    if PLOT_EVAL_PARAMS:
        title = 'Effect of Applied voltage on the "greedy policy" reward'
        def plot_inf_curve(dirCsv, ax, save_name=None,legsTitle=None, true_value_index=4,xl='Time step',yl='Normalized return'):
            filenames = sorted(glob.glob(dirCsv + '/*.npz'))
            legs = np.array([legend.split('_') for legend in filenames])
            legs = [round(float(leg), 4) for leg in legs[:, -3]]
            idx = sorted(range(len(legs)), key=lambda k: legs[k])
            legs = [legs[i] for i in idx]
            filenames = [filenames[i] for i in idx]
            plot_from_npz(filenames, xl, yl, ax=ax, legends=legs, legends_title=legsTitle,
                          saveName=f'./deepRl/plots/{save_name}.pdf', true_value_index=true_value_index)




        # plot_inf_curve(dirTension,a[0][1],'greedy_noise',legsNoiseTitle)
        # params physical figure
        filenames = sorted(glob.glob(dirStatic + '/*.npz'))
        legs = np.array([legend.split('_') for legend in filenames])
        legs = [round(float(leg[1:]), 4) for leg in legs[:, -2]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        plot_from_npz(filenames, xl, yl, ax=aP[0][0], legends=legs, legends_title=legsStaticTitle, saveName='./deepRl/plots/greedy_static.pdf', true_value_index=2)

        filenames = sorted(glob.glob(dirNoise + '/*.npz'))
        legs = np.array([legend.split('_') for legend in filenames])
        legs = [round(float(leg), 4) for leg in legs[:, -3]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        plot_from_npz(filenames, xl, yl, ax=aP[1][0], legends=legs, legends_title=legsNoiseTitle,
                      saveName='./deepRl/plots/greedy_noise.pdf', true_value_index=4)

        # plot_inf_curve(dirTension, a[0][1], 'greedy_noise', legsNoiseTitle)
        filenames = sorted(glob.glob(dirAction + '/*.npz'))
        legs = np.array([legend.split('_') for legend in filenames])
        legs = [round(float(leg), 4) for leg in legs[:, -2]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        plot_from_npz(filenames, xl, yl, ax=aP[1][1], legends=legs, legends_title=legsActionTitle,
                      saveName='./deepRl/plots/greedy_action.pdf', true_value_index=4)

        # inference
        filenames = sorted(glob.glob(dirDynamic + '/*.npz'))
        legs = np.array([legend.split('_') for legend in filenames])
        legs = [round(float(leg), 4) for leg in legs[:, -2]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        plot_from_npz(filenames, xl, yl, ax=aP[0][1], legends=legs, legends_title=legsViscTitle, saveName='./deepRl/plots/greedy_dynamic.pdf', true_value_index=2)








        print('generated train/inf')

    if TENSION_PLOT:
        '''
            PLOT THE TRAINING reward from csv log, namely monitor files
            '''
        xArr, yArr, legs = plot_results('./deepRl/tension-perf',
                                        only_return_data=True)  # ,title=t1) #'Effect of varying tension on the learning'
        legs = [float(leg) for leg in legs[:, -3]]
        xArrT, yArrT, legsT = sort_arr_from_legs(xArr, yArr, legs)  # ,title=t1
        save_show_fig(xArrT, yArrT, ax=a[0][0])
        # experimental training150pwm
        dcVoltage1 = 150 / 255 * 12
        dcVoltage2 = 12
        # experimental setup training
        # 7.1V
        legsT.append(f'{float(round(dcVoltage1, 2))}(experiment 1)')
        legsT.append(f'{float(round(dcVoltage1, 2))}(experiment 2)')
        xArrEx, yArrEx, _ = plot_results(dqn7real, only_return_data=True)

        # xArrT.append(xArrEx[0])
        # yArrT.append(yArrEx[0])
        a[0][0].plot(xArrEx[0], yArrEx[0] / EP_STEPS, color=cmap(7.1 / 12), linewidth=4.0)
        # 12V
        # xArrEx, yArrEx, _ = plot_results('./weights/dqn12V/continue', only_return_data=True)
        # xArrT.append(xArrEx[0])
        # legsT.append(f'{float(round(dcVoltage2,2))}(experiment 3)')
        # a[0][0].plot(xArrEx[0], yArrEx[0]/EP_STEPS, data['results'] color=colorPalette[np.where(TENSION_RANGE == 12)[0][0]])
        # 2.4V
        # 3.5V

        #inference
        # plot_inf_curve(dirTension,a[0][1])
        filenames = sorted(glob.glob(dirTension + '/*.npz'))
        legs = np.array([legend.split('_')[-3] for legend in filenames])
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        legs = [str(leg) + 'V' for leg in legs]
        plot_from_npz(filenames,xl,yl, ax=a[0][1], true_value_index=4)
        #moving average##################
        #TODO figure theta_x avec 4 courbes
        scoreArr = np.zeros_like(TENSION_RANGE)
        stdArr = np.zeros_like(TENSION_RANGE)

        PLOT_EPISODE_REWARD = True
        figm1, ax1 = plt.subplots()
        figm2, ax2 = plt.subplots()

        with open('./deepRl/data/inference_tensions_sim.pickle', 'rb') as f:
            rewArr_tensions = pickle.load(f)
        for i,rewArr in enumerate(rewArr_tensions):
            a[1][0].plot(moving_average(np.array(rewArr), 20), color=cmap(TENSION_RANGE[i] / len(TENSION_RANGE)))
        if PLOT_EPISODE_REWARD:
            a[1][0].set_xlabel('Time step', fontsize=FONT_SIZE_LABEL)
            a[1][0].set_ylabel(yl, fontsize=FONT_SIZE_LABEL)

        else:
            tensionMax = np.array(TENSION_RANGE)
            plt.plot(tensionMax, scoreArr, '-')
            plt.fill_between(tensionMax, scoreArr + stdArr, scoreArr - stdArr, facecolor='red', alpha=0.5)
            plt.rcParams['font.size'] = FONT_SIZE_LABEL
            plt.xlabel('Tension (V)')
            plt.ylabel(yl)
            plt.title('Effect of the Applied voltage on the "greedy policy" reward', fontsize=FONT_SIZE_LABEL)
            plt.rcParams['font.size'] = FONT_SIZE_AXIS
            #plt.show()
            np.savez(
                './deepRl/plots/tension-perf10000ep',
                tensionRange=tensionMax,
                results=scoreArr,
                resultsStd=stdArr
            )
        print('done inference on voltages')
        #indexes 12,14 are best found by theta_x_experiment.py
        #old #filenames = ['./deepRl/real-cartpole/dqn_7.1V/inference_results.npz', './weights/dqn2.4V/inference_results.npz']
        PLOT_SMALL_REAL_TENSION=True
        if PLOT_SMALL_REAL_TENSION:
            data = np.load(filenames_expe[1])
            data.allow_pickle=True
            rewsArr = data["modelRewArr"]
            a[1][0].plot(moving_average(rewsArr[9], 20), linewidth=4.0, color=cmap(0))
            #train
            dcVoltage3 = 2.4
            xArrEx2, yArrEx2, _ = plot_results('./weights/dqn50-real/pwm51', only_return_data=True)
            a[0][0].plot(xArrEx2[0], yArrEx2[0] / EP_STEPS, color=cmap(0), linewidth=4.0)


        data = np.load(filenames_expe[0])
        data.allow_pickle = True
        rewsArr = data['modelRewArr']
        a[1][0].plot(moving_average(rewsArr[14], 20), linewidth=4.0, color=cmap(7.1/12))



        #BOXPLOT
        Te = 0.05
        EP_STEPS = 800
        scoreArr = np.zeros_like(TENSION_RANGE)
        stdArr = np.zeros_like(TENSION_RANGE)
        episodeArr = []

        with open('./deepRl/data/boxplot.pickle', 'rb') as f:
            # pickle.dump(episodeArr, f)
            episodeArr=pickle.load(f)
        a[1][1].boxplot(episodeArr, positions=TENSION_RANGE, patch_artist=True)

        a[1][1].set_ylabel(yl, fontsize=FONT_SIZE_LABEL)
        a[1][1].set_xlabel('Applied DC motor Tension (V)', fontsize=FONT_SIZE_LABEL)


        # experimental inference
        # adding inference
        EXPERIMENTAL_INFERENCE=True
        if EXPERIMENTAL_INFERENCE:
            Timesteps7, epRew7 = inferenceResCartpole('./deepRl/real-cartpole/dqn_7.1V/inference_results.npz',
                                                      monitorFileName='./deepRl/real-cartpole/dqn_7.1V/monitor.csv')
            a[0][1].plot(Timesteps7, epRew7/EP_STEPS,color=cmap(7.1/12),linewidth=4.0)
            # 2.4 V
            Timesteps3, epRew3 = inferenceResCartpole('./weights/dqn50-real/pwm51/inference_results.npz',
                                                      monitorFileName='./weights/dqn50-real/pwm51/monitor.csv')
            a[0][1].plot(Timesteps3, epRew3/EP_STEPS,color=cmap(2.4/12),linewidth=4.0)

        #shrink the y axis to fit legend
        # figT.tight_layout()
        # shrink = 0.07
        # for tax,bax in zip(a[0],a[1]):
        #     tbox = tax.get_position()
        #     bbox = bax.get_position()
        #     tax.set_position([tbox.x0, tbox.y0-shrink*bbox.height, tbox.width, (1-shrink)*tbox.height])
        #     bax.set_position([bbox.x0, bbox.y0, bbox.width, (1-shrink)*bbox.height])
        cmap_labels = ["Applied voltage $\in$ [2.4, 12] V"]
        cmap_handles = [Rectangle((0, 0), 2, 2)]  # [Rectangle((0, 0), 1, 1) for _ in cmap]
        handler_map = dict(zip(cmap_handles, [HandlerColormap(cmap, num_stripes=len(TENSION_RANGE))]))
        figT.legend(handles=cmap_handles,
                       labels=cmap_labels,
                       handler_map=handler_map, fontsize=legendfontsize*1.5, loc='upper center', bbox_to_anchor=(0.5, 0.97))
        # figT.legend(legsT, loc='upper center', bbox_to_anchor=(0.5, 1), title="Applied voltage", ncol=len(legsT))
    #shrink x axis to fit legend

    # shrink = 0.01
    # for tax,bax in zip(aP[0], aP[1]):
    #     tbox = tax.get_position()
    #     bbox = bax.get_position()
    #     tax.set_position([tbox.x0, tbox.y0, (1-shrink)*tbox.width, tbox.height])
    #     bax.set_position([bbox.x0, bbox.y0, (1-shrink)*bbox.width, bbox.height])
    # figT.legend(legsT, loc='upper center', bbox_to_anchor=(0.5, 1), title="Applied voltage", ncol=len(legsT))

    figP.tight_layout()

    putIndInside_2_2(a)
    figT.savefig('./deepRl/plots/tension_all.pdf')
    figT.show()


    putIndInside_2_2(aP)

    figP.savefig('./deepRl/plots/param_analysis.pdf')
    figP.show()






#D subplot: apprentissage, inference, boxplot



dirTensionNoise = './deepRl/tensions-noise'
dirTensionSeed = './deepRl/tension-perf-seed'
dirTensionVar = './deepRl/tension-7-exp'

saveName = ['./deepRl/plots/greedy_tension_seed.pdf', './deepRl/plots/tension_noise_1.pdf', './deepRl/plots/tension_6.5-7.0.pdf', './deepRl/plots/tension_70.pdf']



if PLOT_ACTION_NOISE:
    def plot_train_inference(dir, saveName, figsize=(10, 15), legendTitle="Applied voltage", target=None,
                             dirExperiment=None, infExpDir=None):
        '''
        plots train inference at the double plot in 2 rows
        :param dir:
        :param saveName:
        :param figsize:
        :param legendTitle:
        :param target:
        :param dirExperiment:
        :param infExpDir:
        :return:
        '''
        filenames = sorted(glob.glob(dir + '/*.npz'))
        fTension, ax = plt.subplots(2, figsize=figsize)
        xArr, yArr, legs = plot_results(dir, only_return_data=True)
        legs = np.array([round(float(leg), 2) for leg in legs[:, -3]])
        xArrT, yArrT, legsT = sort_arr_from_legs(xArr, yArr, legs)
        if target != None:
            index = findInd(legsT, target)
            legsT = legsT[index]
            xArrT = xArrT[index]
            yArrT = yArrT[index]
        save_show_fig(xArrT, yArrT, ax=ax[0])

        if dirExperiment != None:  # plot the training part
            xArrEx, yArrEx, _ = plot_results(dirExperiment, only_return_data=True)
            ax[0].plot(xArrEx[0], yArrEx[0] / EP_STEPS, color=cmap(7.1/12),
                       linewidth=4.0)

        legs = np.array([legend.split('_') for legend in filenames])
        legs = [round(float(leg), 2) for leg in legs[:, -3]]
        idx = sorted(range(len(legs)), key=lambda k: legs[k])
        legs = [legs[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        legs = [str(leg) + 'V' for leg in legs]
        plot_from_npz(filenames, xl, yl, ax=ax[1], target=target)  # plot the inference part

        if dirExperiment != None and infExpDir != None:
            Timesteps7, epRew7 = inferenceResCartpole(infExpDir, monitorFileName=dirExperiment + '/monitor.csv')
            ax[1].plot(Timesteps7, epRew7 / EP_STEPS, data['results'], color=cmap(7.1/12), linewidth=4.0)
        legs.append('7.06 experiment')
        fTension.legend(legs, loc='upper center', bbox_to_anchor=(0.5, 1), title=legendTitle, ncol=min(len(legs), 6))
        fTension.savefig(saveName)


    # plot_train_inference(dirTensionNoise, saveName[1])
    # plot_train_inference(dirTensionNoise + '/original', saveName[2])
    targetRange = np.arange(7.01,7.15,0.01)
    for target in targetRange:
        target = round(target,2)
        plot_train_inference(dirTensionVar, f'./deepRl/plots/tension_{target}.pdf', dirExperiment = dqn7real,target=target, infExpDir=filenames[0])
    # plot_train_inference(dirTensionVar, saveName[3], dirExperiment = dqn7real, infExpDir=filenames[0])

if CROP:
    os.chdir('./deepRl/plots')
    subprocess.call(['sh', './crop.sh']) #call shell script for pdfcrop
# sys.exit(0)
if PLOT_DOUBLE_PARAMS:#not use in article


    # static friciton
    xArr, yArr, legsSt = plot_results('./deepRl/static-friction', title=t2, only_return_data=True)
    legsSt = [round(float(leg[1:]), 4) for leg in legsSt[:, -2]]
    xArr, yArr, legsSt = sort_arr_from_legs(xArr, yArr, legsSt)
    # save_show_fig(xArr, yArr, legsSt, saveName='./deepRl/plots/static.pdf', true_value_index=2)  # ,title=t2
    save_show_fig(xArr, yArr, ax=axSt[0], true_value_index=-1)  # ,title=t2

    xArr, yArr, legsDy = plot_results('./deepRl/dynamic-friction', title=t3, only_return_data=True)
    legsDy = [round(float(leg), 4) for leg in legsDy[:, -2]]
    xArr, yArr, legsDy = sort_arr_from_legs(xArr, yArr, legsDy)
    # save_show_fig(xArr, yArr, legsDy, saveName='./deepRl/plots/dynamic.pdf', true_value_index=2)  # ,title=t3
    save_show_fig(xArr, yArr, ax=axDy[0], true_value_index=2)  # ,title=t3

    xArr, yArr, legsNo = plot_results(dirNoise, title=t4, only_return_data=True)
    legsNo = [round(float(leg), 4) for leg in legsNo[:, -3]]
    xArr, yArr, legsNo = sort_arr_from_legs(xArr, yArr, legsNo)
    # save_show_fig(xArr, yArr, legsNo, saveName='./deepRl/plots/noise.pdf', true_value_index=4)  # ,title=t4
    save_show_fig(xArr, yArr, ax=axNo[0], true_value_index=4)  # ,title=t4

    xArr, yArr, legsAc = plot_results('./deepRl/action-noise', title=t5, only_return_data=True)
    legsAc = [float(leg) for leg in legsAc[:, -2]]
    xArr, yArr, legsAc = sort_arr_from_legs(xArr, yArr, legsAc)
    # save_show_fig(xArr, yArr, legsAc, saveName='./deepRl/plots/action_noise.pdf', true_value_index=2)  # ,title=t5
    save_show_fig(xArr, yArr, ax=axAc[0])  # , true_value_index=1)  # ,title=t5

    xArr, yArr, legs = plot_results('./deepRl/experimental-vs-random', title=t6, only_return_data=True)
    legs = [leg for leg in legs[:, -2]]
    save_show_fig(xArr, yArr, legs, saveName='./deepRl/plots/exp-vs-rand.pdf')  # ,title=t6

    xArr, yArr, legs = plot_results('./deepRl/seeds', title=t6, only_return_data=True)
    legs = legs[:, -1]
    xArr, yArr, legs = sort_arr_from_legs(xArr, yArr, legs)
    save_show_fig(xArr, yArr, [leg[0] for leg in legs], saveName='./deepRl/plots/seeds.pdf')  # ,title=t6

    #inference
    filenames = sorted(glob.glob(dirDynamic + '/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs=[round(float(leg),4) for leg in legs[:,-2]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    # plot_from_npz(filenames, xl, yl, ax=aP[0][1], legends=legs, legends_title=legsViscTitle, saveName='./deepRl/plots/greedy_dynamic.pdf', true_value_index=2)
    plot_from_npz(filenames, xl, yl, ax=axDy[1],  true_value_index=2)

    filenames = sorted(glob.glob(dirStatic + '/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs=[round(float(leg[1:]),4) for leg in legs[:,-2]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    # plot_from_npz(filenames, xl, yl, ax=aP[0][0], legends=legs, legends_title=legsStaticTitle, saveName='./deepRl/plots/greedy_static.pdf', true_value_index=2)
    plot_from_npz(filenames, xl, yl, ax=axSt[1],  true_value_index=2)

    filenames = sorted(glob.glob(dirNoise + '/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs=[round(float(leg),4) for leg in legs[:,-3]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    # plot_from_npz(filenames, xl, yl, ax=aP[1][0], legends=legs, legends_title=legsNoiseTitle, saveName='./deepRl/plots/greedy_noise.pdf', true_value_index=4)
    plot_from_npz(filenames, xl, yl, ax=axNo[1],  true_value_index=4)

    filenames = sorted(glob.glob(dirAction + '/*.npz'))
    legs = np.array([legend.split('_') for legend in filenames])
    legs = [round(float(leg),4) for leg in legs[:,-2]]
    idx = sorted(range(len(legs)), key=lambda k: legs[k])
    legs = [legs[i] for i in idx]
    filenames = [filenames[i] for i in idx]
    # plot_from_npz(filenames, xl, yl, ax=aP[1][1], legends=legs, legends_title=legsActionTitle, saveName='./deepRl/plots/greedy_action.pdf', true_value_index=4)
    plot_from_npz(filenames, xl, yl, ax=axAc[1],  true_value_index=1)
    #save_AB_PLOTS TRAIN-INFERENCE
    label_legend_save_2(figNo,axNo,legsNo,legsNoiseTitle,'noise_all')
    label_legend_save_2(figDy,axDy,legsVisc,legsViscTitle,'dynamic_all')
    label_legend_save_2(figSt,axSt,legsStatic,legsStaticTitle,'static_all')
    label_legend_save_2(figAc,axAc,legsAc,legsActionTitle,'noise_all')