from tcp_envV2 import CartPoleRPI
import gym
import numpy as np
import os
from stable_baselines3 import SAC
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import argparse
from stable_baselines3.sac.policies import MlpPolicy
import time
from typing import Callable
import socket
HOST = '169.254.161.71'#'c0:3e:ba:6c:9e:eb'#'10.42.0.1'#wifiHot #'127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432
# Use deterministic actions for evaluation and SAVE the best model
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=20000, save_path='./logs/', name_prefix='rl_model')
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func
STEPS_TO_TRAIN = 30000

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            env = CartPoleRPI(pi_conn=conn)
            # Load the saved statistics
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load('envNorm.pkl', env)
            #  do not update them at test time
            env.training = False
            # reward normalization is not needed at test time
            env.norm_reward = False


            model = SAC.load("./logs/best_model", env=env)
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                time.sleep(0.03)
                if dones:
                    env.reset()

        conn.close()
finally:

    # model.save("cartpole_pi_sac")
    # model.save_replay_buffer("sac_swingup_buffer")
    pass