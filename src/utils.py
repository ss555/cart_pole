from typing import Callable
from matplotlib import pyplot as plt
import plotly.express as px
import numpy as np
import glob
import sys
import os
import yaml
import gym
from typing import Any, Callable, Dict, Union, List, Optional, Tuple
from collections import OrderedDict
from stable_baselines3.common.vec_env import VecEnv
# Import seaborn
import seaborn as sns
from env_wrappers import load_results, load_data_from_csv
from bokeh.palettes import d3


def rungekutta4(f, y0, t, args=()):
    '''

    :param f: 1st order derivative
    :param y0: ini conditions
    :param t: time interval ex: [0,Te]
    :param args: arguments for f ex: fs,fv...
    :return: y (including y0)
    '''
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y

def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
    # Save hyperparams
    with open(os.path.join(self.params_path, "config.yml"), "w") as f:
        yaml.dump(saved_hyperparams, f)

    # save command line arguments
    with open(os.path.join(self.params_path, "args.yml"), "w") as f:
        ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
        yaml.dump(ordered_args, f)

    print(f"Log path: {self.save_path}")

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
def inferenceResCartpole(filename: str = '', monitorFileName: str = ''):
    '''
    :param filename: name of .npz file
    :return: timeArray, epsiodeReward corresponding to inference
    NOTE: the weights are saved after nth episodes, that's why we also need to open monitor file to see the correspondance between episodes and Timesteps
    '''
    dataInf = np.load(filename)
    dataInf.allow_pickle = True
    # monitor file
    data, name = load_data_from_csv(monitorFileName)  # './EJPH/real-cartpole/dqn/monitor.csv')
    rewsArr = dataInf["modelRewArr"]
    obsArr = dataInf["modelsObsArr"]
    actArr = dataInf["modelActArr"]
    nameArr = dataInf["filenames"]
    Timesteps = np.zeros(len(obsArr))
    epReward = np.zeros(len(obsArr))
    for i in range(0, len(obsArr)):
        print()
        obs = obsArr[i]
        act = actArr[i]
        epReward[i] = np.sum(rewsArr[i])
        Timesteps[i] = np.sum(data['l'][:((i+1) * 10)])
        print(f'it {i} and {epReward[i]}')

    return Timesteps, epReward

def evaluate_policy_episodes(
        env: Union[gym.Env, VecEnv],
        model:"base_class.BaseAlgorithm",
        n_eval_episodes:int=1,
        episode_steps=800):
    '''
    :param env:
    :param model:
    :param n_eval_episodes:
    :return: array of episode rewards
    '''
    episodeRewArr=np.zeros((n_eval_episodes,episode_steps),dtype=np.float32)
    lengthArr=np.zeros(n_eval_episodes,dtype=np.float32)
    for i in range(n_eval_episodes):
        episodeLength=0
        done = False
        obs=env.reset()
        while not done:
            action,_state=model.predict(obs,deterministic=True)
            obs,reward,done,_=env.step(action)
            episodeRewArr[i,episodeLength]=reward
            episodeLength+=1
        lengthArr[i] = episodeLength
    return episodeRewArr, lengthArr

def read_hyperparameters(name, path="parameters.yml",additional_params=None):
    with open(path, "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        hyperparams = hyperparams_dict[name]
        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])
            print('parameters loaded')

        if "n_timesteps" in hyperparams:
            del hyperparams["n_timesteps"]
        # Convert to python object if needed
        try:
            if isinstance(hyperparams["policy_kwargs"], str):
                hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
        except:
            pass
        # Overwrite hyperparams if needed
        # hyperparams.update(self.custom_hyperparams)
    return hyperparams

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    '''
    :param initial_value: initial value of a learning rate that decays, used with SB3
    :return:
    '''
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func
def plot(observations = [],timeArr=[], actArr=[], save=True,plotlyUse=False,PLOT_TIME_DIFF=True,savePath='./tmp/',paperMode=False):
    '''
    :param observations: experience in form of N,5 observations=np.array(observations)
    :param timeArr: time when observation happened
    :param actArr: array of actions
    :param save: save plt plot or not
    :param plotlyUse: plot in nice figures in browser
    :return:
    '''
    try:
        if not paperMode:

            fig1=plt.figure(figsize=(12, 12))
            plt.subplot(221)
            plt.title('X')
            plt.xlabel('time (s)')
            plt.ylabel('distance (m)')
            #plt.axvline(0.2, 0, 1) #vertical line
            # plt.plot(timeArr,observations[:,0], 'r')
            plt.plot(timeArr,observations[:,0], 'r.')
            plt.subplot(223)
            plt.title('X_dot')
            plt.xlabel('time (s)')
            plt.ylabel('speed (m/s)')
            # plt.plot(timeArr,observations[:,1], 'g')
            plt.plot(timeArr,observations[:,1], 'g.')
            plt.subplot(222)
            plt.title('theta')
            plt.xlabel('time (s)')
            plt.ylabel('angle (rad)')
            theta=np.arctan2(observations[:,3],observations[:,2])
            # plt.plot(timeArr,theta, 'r')
            plt.plot(timeArr,theta, 'r.')
            plt.subplot(224)
            plt.title('theta_dot')
            plt.xlabel('time (s)')
            plt.ylabel('angular speed (rad/s)')
            # plt.plot(timeArr,observations[:,4], 'g')
            plt.plot(timeArr,observations[:,4], 'g.')
            plt.savefig(savePath+'/observations.png', dpi=200)
            plt.close(fig1)

        else:
            sns.set_context("paper")
            sns.set_style("whitegrid")
            sns.lineplot(x=timeArr, y=observations[:, 0])
            ym=np.mean(observations[-500:, 0])
            plt.plot([timeArr[0],timeArr[-1]],[ym,ym],'b--')
            plt.xlabel('time (s)')
            plt.ylabel('distance (m)')
            plt.show()
            sns.lineplot(x=timeArr, y=np.arctan2(observations[:,3],observations[:,2]))
            ym=np.mean(np.arctan2(observations[:,3],observations[:,2]))
            plt.plot([timeArr[0],timeArr[-1]],[ym,ym],'b--')
            plt.xlabel('time (s)')
            plt.ylabel('angle (rad)')
            plt.show()

    except:
        print('err occured')
    ##FOR FINE tuned position/acceleration...
    if plotlyUse:
        fig = px.scatter(x=timeArr, y=observations[:, 0], title='observations through time')
        # fig.add_scatter(x=timeArr[:, 0], y=observations[:, 4], name='theta_dot through time')
        fig.add_scatter(x=timeArr, y=observations[:, 4], name='theta_dot through time')
        theta=np.arctan2(observations[:, 3],observations[:, 2])
        # fig.add_scatter(x=timeArr[:, 0], y=theta, name='theta through time')
        fig.add_scatter(x=timeArr, y=theta, name='theta through time')
        fig.show()
    if PLOT_TIME_DIFF:
        try:
            # LOOK NOISE IN TIME
            fig2 = plt.figure(figsize=(12, 12))
            plt.plot(np.diff(timeArr))
            plt.savefig(savePath+'time_diff.png',dpi=200)
            plt.close(fig2)
            fig3 = plt.figure(figsize=(12, 12))
            plt.plot(timeArr, actArr, '.')
            plt.savefig('./tmp/time_action.png', dpi=200)
            plt.close(fig3)
        except:
            print('err of time plot')
    if save:
        np.savetxt('./tmp/obs.csv',observations, delimiter=",")
        np.savetxt('./tmp/time.csv',timeArr, delimiter=",")
        np.savetxt('./tmp/actArr.csv',actArr, delimiter=",")


def plot_line(observations = [],timeArr=[]):
    observations=np.array(observations)
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.title('X')
    plt.xlabel('time (s)')
    plt.ylabel('distance (m)')
    #plt.axvline(0.2, 0, 1) #vertical line
    plt.plot(timeArr,observations[:,0], 'r')
    # plt.plot(timeArr,observations[:,0], 'r.')
    plt.subplot(223)
    plt.title('X_dot')
    plt.xlabel('time (s)')
    plt.ylabel('speed (m/s)')
    plt.plot(timeArr,observations[:,1], 'g')
    # plt.plot(timeArr,observations[:,1], 'g.')
    plt.subplot(222)
    plt.title('theta')
    plt.xlabel('time (s)')
    plt.ylabel('angle (rad)')
    theta=np.arctan2(observations[:,3],observations[:,2])
    # plt.plot(timeArr,theta, 'r')
    plt.plot(timeArr,theta, 'r.')
    plt.subplot(224)
    plt.title('theta_dot')
    plt.xlabel('time (s)')
    plt.ylabel('angular speed (rad/s)')
    plt.plot(timeArr,observations[:,4], 'g')
    # plt.plot(timeArr,observations[:,4], 'g.')
    plt.show()
    plt.savefig('./tmp/observations.png')
    plt.plot(np.diff(timeArr))
    plt.show()
    plt.savefig('./tmp/time_diff.png')



def printAllFilesCD(extension,absPath):
    '''
    prints all files with this extension
    :param extension: extension of file searched
    :return: 0 (prints the names in console
    '''
    sys.path.append(os.path.abspath('../'))
    namesRaw = glob.glob(absPath + './*.csv')