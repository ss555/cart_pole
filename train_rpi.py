from tcp_envV2 import CartPoleCosSinRPIhistory
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from typing import Callable
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
import socket
import numpy as np
from typing import Callable
from custom_callbacks import plot_results
from utils import linear_schedule, plot
HOST = '169.254.161.71'#'255.255.0.0'#wifiHot #'127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432
import time
logdir='./logs/rpi/'
resumeDir='./Transfer_learning/backup'
# Use deterministic actions for evaluation and SAVE the best model
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# Save a checkpoint every 1000 steps
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
STEPS_TO_TRAIN = 150000
manual_seed = 5



try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            env = CartPoleCosSinRPIhistory(pi_conn=conn)
            env0 = Monitor(env, logdir)
            n_actions = env.action_space.shape[-1]
            ## Automatically normalize the input features and reward
            env1 = DummyVecEnv([lambda: env0])
            # env=env.unwrapped
            env = VecNormalize.load(resumeDir+'/envRpiNorm.pkl', env1)
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
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
            # model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(float(1e-3)), buffer_size=300000,
            #             batch_size=512, ent_coef='auto', gamma=0.98, tau=0.02, n_episodes_rollout=-1, gradient_steps=-1,
            #             learning_starts=10000, seed=manual_seed, use_sde=True,
            #             policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]))
            env.training = True
            env.norm_reward = True

            model=SAC.load(resumeDir+"/cartpole_pi_sac",env=env)
            # model.load_replay_buffer(resumeDir+"/sac_pi_swingup_buffer")
            # model.action_noise=None#action_noise
            model.learning_starts = 0
            # model.learning_rate = float(7.3e-4)
            env.seed(manual_seed)
            model.train_freq=-1
            model.gradient_steps=-1
            model.n_episodes_rollout=1
            # model.target_update_interval=2
            model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100, callback=[eval_callback, callbackSave])
            # # WHEN NORMALISING
            env.training = False
            # reward normalization is not needed at test time
            env.norm_reward = False
            # plot_results(logdir)
            obs = env.reset()
            obsArr = [env.get_original_obs()[0]]
            actArr = [0.0]
            timeArr = [0.0]
            start_time = time.time()
            for i in range(200):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = env.step(action)
                obsArr.append(env.get_original_obs()[0])
                actArr.append(action[0, 0])
                timeArr.append(time.time() - start_time)
                if dones:
                    env.reset()
            plot(obsArr, timeArr, actArr)
        conn.close()
finally:
    conn.close()
    model.save("cartpole_pi_sac")
    model.save_replay_buffer("sac_pi_swingup_buffer")
    # WHEN NORMALISING
    env.save('envRpiNorm.pkl')