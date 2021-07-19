from glob import glob

from tcp_envV2 import CartPoleCosSinRpiDiscrete3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from utils import linear_schedule, plot
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager, CheckPointEpisode
import socket
import time
import os
from utils import read_hyperparameters
import numpy as np
from custom_callbacks import plot_results
from env_custom import CartPoleButter
env = CartPoleButter(tensionMax=7.06,x_threshold=0.355)
INFERENCE_PATH='./EJPH/real-cartpole/dqn'
modelsObsArr, modelActArr = [],[]
filenames = (sorted(glob(os.path.join(INFERENCE_PATH, "*" + '.zip')), key=os.path.getmtime))
obs = env.reset()
for modelName in filenames:
    print(f'loading {modelName}')
    model = DQN.load(modelName,env=env)
    done = False
    obsArr,actArr,rewArr = [],[],[]
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, _ = env.step(action)
        obsArr.append(obs)
        actArr.append(action)
        rewArr.append(rewards)
        if done:
            obs = env.reset()
            break
    modelsObsArr.append(obsArr)
    modelActArr.append(actArr)
    print(np.sum(rewArr))
np.savez('./EJPH/real-cartpole/dqn/inference_results.npz',modelsObsArr=modelsObsArr,modelActArr=modelActArr,filenames=filenames)



#find files
# path='./EJPH/real-cartpole/dqn'
# print(sorted(glob(os.path.join(path, "*" + '.zip')),key=os.path.getmtime))


#ZOOM PLOT
# # Create main container
# fig = plt.figure()
#
# # Set the random seed
# np.random.seed(100)
#
# # Create mock data
# x = np.random.normal(400, 50, 10_000)
# y = np.random.normal(300, 50, 10_000)
# c = np.random.rand(10_000)
#
# def setlabel(ax, label, loc=2, borderpad=0.6, **kwargs):
#     legend = ax.get_legend()
#     if legend:
#         ax.add_artist(legend)
#     line, = ax.plot(np.NaN,np.NaN,color='none',label=label)
#     label_legend = ax.legend(handles=[line],loc=loc,handlelength=0,handleheight=0,handletextpad=0,borderaxespad=0,borderpad=borderpad,frameon=False,**kwargs)
#     label_legend.remove()
#     ax.add_artist(label_legend)
#     ax.text(-0.2,0.9,chr(98)+')', transform=ax.transAxes)
#     line.remove()
#
# # Create zoom-in plot
# ax = plt.scatter(x, y, s = 5, c = c)
# plt.xlim(400, 500)
# plt.ylim(350, 400)
# plt.xlabel('x', labelpad = 15)
# plt.ylabel('y', labelpad = 15)
#
# # Create zoom-out plot
# ax_new = fig.add_axes([0.6, 0.6, 0.2, 0.2]) # the position of zoom-out plot compare to the ratio of zoom-in plot
# plt.scatter(x, y, s = 1, c = c)
#
# # Save figure with nice margin
# setlabel(ax_new, '(a)')
# plt.show()