from tcp_envV2 import CartPoleRPI,CartPoleCosSinRPI
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from typing import Callable
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
import socket
from custom_callbacks import plot_results
HOST = '169.254.161.71'#'c0:3e:ba:6c:9e:eb'#'10.42.0.1'#wifiHot #'127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432
logdir='./logs/rpi/'
# Use deterministic actions for evaluation and SAVE the best model
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# Save a checkpoint every 1000 steps
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
STEPS_TO_TRAIN = 120000

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            env = CartPoleCosSinRPI(pi_conn=conn)
            env0 = Monitor(env, logdir)
            n_actions = env.action_space.shape[-1]
            ## Automatically normalize the input features and reward
            env1 = DummyVecEnv([lambda: env0])
            # env=env.unwrapped
            env = VecNormalize.load('envNorm.pkl', env1)
            #env = VecNormalize(env1, norm_obs=True, norm_reward=True, clip_obs=10000, clip_reward=10000)
            envEval = VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=10000, clip_reward=10000)
            # Stop training when the model reaches the reward threshold
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1700, verbose=1)
            # Use deterministic actions for evaluation and SAVE the best model
            eval_callback = EvalCallback(envEval, best_model_save_path='./logs/', n_eval_episodes=1,
                                         log_path=logdir, eval_freq=15000, callback_on_new_best=callback_on_best,
                                         deterministic=True, render=False)
            # model = SAC.load("cartpole_pi_sac", env=env)
            model = SAC.load("./logs/best_model_training.zip", env=env)
            manual_seed = 5
            env.seed(manual_seed)
            # model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100, callback=[eval_callback, callbackSave])
            # # WHEN NORMALISING
            # env.save('envRpiNorm.pkl')
            env.training = False
            # reward normalization is not needed at test time
            env.norm_reward = False
            # plot_results(logdir)

            obs = env.reset()
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = env.step(action)
                if dones:
                    env.reset()
        conn.close()
finally:

    model.save("cartpole_pi_sac")
    model.save_replay_buffer("sac_pi_swingup_buffer")
    # WHEN NORMALISING
    env.save('envRpiNorm.pkl')