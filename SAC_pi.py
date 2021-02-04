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
from typing import Callable
import socket
HOST = '169.254.161.71'#'c0:3e:ba:6c:9e:eb'#'10.42.0.1'#wifiHot #'127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432
# Use deterministic actions for evaluation and SAVE the best model
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=20000, save_path='./logs/', name_prefix='rl_model')
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func
STEPS_TO_TRAIN = 300000

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            env = CartPoleRPI(pi_conn=conn)
            # The noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

            model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(0.001), ent_coef='auto',
                        verbose=0, action_noise=action_noise, batch_size=4048, learning_starts=0, policy_kwargs=dict(net_arch=[256, 256]),
                        tensorboard_log="./sac_cartpole_tensorboard/", train_freq=-1, n_episodes_rollout=1,
                        gradient_steps=-1)
            model.load("cartpole_pi_sac")
            model.load_replay_buffer("sac_swingup_buffer")
            with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
                model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, checkpoint_callback])

        conn.close()
    # model.save("cartpole_pi_sac")
    # model.save_replay_buffer("sac_swingup_buffer")
finally:

    # model.save("cartpole_pi_sac")
    # model.save_replay_buffer("sac_swingup_buffer")
    pass