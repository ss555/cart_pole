from tcp_envV2 import CartPoleCosSinRPIv2
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from utils import linear_schedule, plot
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
import socket
import time
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
from custom_callbacks import plot_results
HOST = '169.254.161.71'#'255.255.0.0'#wifiHot #'127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432
logdir='./Transfer_learning'
# Use deterministic actions for evaluation and SAVE the best model
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# Save a checkpoint every 1000 steps
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
STEPS_TO_TRAIN = 150000
manual_seed = 5
#lr schedule

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            env = CartPoleCosSinRPIv2(pi_conn=conn)
            env0 = Monitor(env, logdir)
            n_actions = env.action_space.shape[-1]
            ## Automatically normalize the input features and reward
            env1 = DummyVecEnv([lambda: env0])
            # env=env.unwrapped
            env = VecNormalize.load('envNorm.pkl', env1)#VecNormalize.load(logdir+'/envNorm.pkl', env1)
            # env = VecNormalize(env1, norm_obs=True, norm_reward=True, clip_obs=10000, clip_reward=10000)
            envEval = VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=10000, clip_reward=10000)
            # Stop training when the model reaches the reward threshold
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1800, verbose=1)
            # Use deterministic actions for evaluation and SAVE the best model
            eval_callback = EvalCallback(envEval, best_model_save_path='./logs/', n_eval_episodes=1,
                                         log_path=logdir, eval_freq=15000, callback_on_new_best=callback_on_best,
                                         deterministic=True, render=False)

            ##action_noise
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
            # model = SAC.load("cartpole_pi_sac", env=env)



            # model.load_replay_buffer("sac_pi_swingup_buffer")
            # model=SAC.load("cartpole_pi_sac")
            # model.load_replay_buffefr("sac_pi_swingup_buffer")

            # model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100, callback=[eval_callback, callbackSave])
            # # WHEN NORMALISING
            # env.save('envRpiNorm.pkl')
            env.training = False
            model = SAC.load("./logs/best_model", env=env)  # SAC.load(logdir+"/best_model.zip", env=env)
            obs = env.reset()
            obsArr=[env.get_original_obs()[0]]
            actArr=[0.0]
            timeArr=[0.0]
            start_time = time.time()
            for i in range(300):
                action, _states = model.predict(obs, deterministic=False)
                obs, rewards, dones, _ = env.step(action)
                obsArr.append(env.get_original_obs()[0])
                actArr.append(action[0,0])
                timeArr.append(time.time()-start_time)
                if dones:
                    env.reset()
            env.reset()
        conn.close()
finally:
    plot(obsArr, timeArr,actArr)
    model.save("cartpole_pi_sac")
    model.save_replay_buffer("sac_pi_swingup_buffer")
    # WHEN NORMALISING
    env.save('envRpiNorm.pkl')
    conn.close()

