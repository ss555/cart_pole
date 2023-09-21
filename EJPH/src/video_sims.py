'''
records the video of the training on openai gym
'''

import sys
import os
# sys.path.append(os.path.abspath('./..'))
import gym
import cartpole
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./../..'))
sys.path.append(os.path.abspath('./..'))
import sys
import os
# sys.path.append(os.path.abspath('./..'))
import gym
import cartpole
sys.path.append(os.path.abspath('./'))
from stable_baselines3 import DQN
from src.env_custom import CartPoleRK4 #,CartPoleContinous,CartPoleDiscreteHistory#,CartPoleDiscreteButter2
import argparse
from utils import read_hyperparameters, evaluate_policy_episodes
from pathlib import Path
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from glob import glob
from src.env_wrappers import VideoRecorderWrapper
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'
#FOLDER DIRS
dirTension = './EJPH/tension-perf'
dirStatic = './EJPH/static-friction'
dirDynamic = './EJPH/dynamic-friction'
dirNoise = './EJPH/encoder-noise'
dirAction = './EJPH/action-noise'
dirReset = './EJPH/experimental-vs-random'
EXT = '.zip'

def evaluate_policy(env,model,steps):
    data=[]
    obs=env.reset()
    for i in range(int(steps)):
        action=model.predict(obs,deterministic=True)
        obs,rew,done,_=env.step(action[0])
        data.append((obs,rew,done))
        if done:
            break
    return data
def videoSimulate(video_folder,env,file,video_name):
    os.makedirs(video_folder, exist_ok=True)
    # # Start the video at step=0 and record 500 steps
    eval_env = VideoRecorderWrapper(env, video_folder=video_folder, record_video_trigger=lambda step: step == 0,
                                    name_prefix=video_name)
    model = DQN.load(file, env=eval_env)
    evaluate_policy(env=eval_env, model=model, steps=num_steps_inference)
    # Close the video recorder
    eval_env.close()
if __name__=='__main__':
    EP_STEPS = 10
    TRAIN = True

    Te = 0.05
    num_steps = 6e2
    num_steps_inference = 4e2
    #
    filenames = sorted(glob(os.path.join(dirTension, "*" + EXT)), key=os.path.getmtime)
    for file in filenames:
        TENSION = float(file.split('_')[2])
        video_name = f'{TENSION}'
        env = CartPoleRK4(tensionMax=TENSION, title_to_show = video_name)
        video_folder = './EJPH/video/tension/'
        videoSimulate(video_folder, env, file, video_name)


    # filenames = sorted(glob(os.path.join(dirStatic, "*" + EXT)), key=os.path.getmtime)
    # for file in filenames:
    #     f_c = round(float(file.split('_')[-3]),4)
    #     video_name = str(f_c)
    #     env = CartPoleRK4(f_c = f_c, title_to_show = video_name)
    #     video_folder = './EJPH/video/static/'
    #     videoSimulate(video_folder, env, file, video_name)

    # filenames = sorted(glob(os.path.join(dirDynamic, "*" + EXT)), key=os.path.getmtime)
    # from matplotlib import pyplot as plt
    # for file in filenames:
    #     k_pend = round(float(file.split('_')[3]),5)
    #     # video_name = f'{k_pend}'
    #     video_name = f'{k_pend}'
    #     # video_name = str(k_pend)+" [$N*s*rad^{-1}$]"
    #     env = CartPoleRK4(kPendViscous=k_pend, title_to_show=video_name)
    #     video_folder = './EJPH/video/viscous/'
    #     videoSimulate(video_folder, env, file, video_name = video_name)
    #
    # filenames = sorted(glob(os.path.join(dirNoise, "*" + EXT)), key=os.path.getmtime)
    # for file in filenames:
    #     Km = round(float(file.split('_')[-4]),6)
    #     video_name = str(Km)
    #     env = CartPoleRK4(Km = Km, title_to_show = video_name)
    #     video_folder = './EJPH/video/noise/'
    #     videoSimulate(video_folder, env, file, video_name)
