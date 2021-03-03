from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func
def plot(observations = [],timeArr=[], actArr=[], save=True):

    observations=np.array(observations)
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
    # plt.show()
    plt.savefig('./tmp/observations.png', dpi=200)
    plt.close(fig1)


    fig2 = plt.figure(figsize=(12, 12))
    plt.plot(np.diff(timeArr))
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