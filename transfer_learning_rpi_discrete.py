from tcp_envV2 import CartPoleCosSinRpiDiscrete3
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from utils import linear_schedule, plot
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
import socket
import time
from sb3_contrib import QRDQN


import numpy as np
from custom_callbacks import plot_results
HOST = '169.254.161.71'#'255.255.0.0'#wifiHot #'127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432
logdir='./Transfer_learning'
# Use deterministic actions for evaluation and SAVE the best model
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# Save a checkpoint every 1000 steps
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
STEPS_TO_TRAIN = 150000
#lr schedule
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            env = CartPoleCosSinRpiDiscrete3(pi_conn=conn)
            env0 = Monitor(env, logdir)
            # model = QRDQN.load("./logs/other_algo/actions3/best_model", env=env)
            eval_callback = EvalCallback(env0, best_model_save_path='./logs/', n_eval_episodes=2,
                                         log_path=logdir, eval_freq=5000,
                                         deterministic=True, render=False)
            model = DQN.load("./logs/best_model", env=env)
            model.load_replay_buffer("dqn_pi_swingup_bufferN")
            model.learning_starts=0
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[eval_callback, callbackSave])
            obs = env.reset()
            obsArr=[obs]
            actArr=[0.0]
            timeArr=[0.0]
            start_time = time.time()
            for i in range(1000):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = env.step(action)
                obsArr.append(obs)
                actArr.append(action)
                timeArr.append(time.time()-start_time)
                if dones:
                    env.reset()
            env.reset()
        conn.close()
finally:
    plot(obsArr, timeArr, actArr)
    model.save("cartpole_pi_dqnN")
    model.save_replay_buffer("dqn_pi_swingup_bufferN")
    # WHEN NORMALISING
    conn.close()

