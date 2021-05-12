from typing import Callable
from matplotlib import pyplot as plt
import plotly.express as px
import numpy as np
import glob
import sys
import os
import yaml
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
from pprint import pprint
# Import seaborn
import seaborn as sns
from env_wrappers import load_results

def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
    # Save hyperparams
    with open(os.path.join(self.params_path, "config.yml"), "w") as f:
        yaml.dump(saved_hyperparams, f)

    # save command line arguments
    with open(os.path.join(self.params_path, "args.yml"), "w") as f:
        ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
        yaml.dump(ordered_args, f)

    print(f"Log path: {self.save_path}")

def read_hyperparameters(name, path="parameters.yml",additional_params=None):

    with open(path, "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        hyperparams = hyperparams_dict[name]
        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])
            print('parameters loaded')
        # Convert to python object if needed
        if isinstance(hyperparams["policy_kwargs"], str):
            hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
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
def plot(observations = [],timeArr=[], actArr=[], save=True,plotlyUse=False,PLOT_TIME_DIFF=True):
    '''
    :param observations: experience in form of N,5
    :param timeArr: time when observation happened
    :param actArr: array of actions
    :param save: save plt plot or not
    :param plotlyUse: plot in nice figures in browser
    :return:
    '''
    observations=np.array(observations)
    try:
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
        plt.savefig('./tmp/observations.png', dpi=200)
        plt.close(fig1)
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
        # LOOK NOISE IN TIME
        fig2 = plt.figure(figsize=(12, 12))
        plt.plot(np.diff(timeArr[:,0]))
        # plt.show()
        plt.savefig('./tmp/time_diff.png',dpi=200)
        plt.close(fig2)

    fig3 = plt.figure(figsize=(12, 12))
    plt.plot(timeArr,actArr,'.')
    # plt.show()
    plt.savefig('./tmp/time_action.png',dpi=200)
    plt.close(fig3)
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
    sys.path.append(os.path.abspath('./'))
    namesRaw = glob.glob(absPath + './*.csv')