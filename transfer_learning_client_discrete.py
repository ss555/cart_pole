from tcp_envV2 import CartPoleCosSinRpiDiscrete3
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from utils import linear_schedule, plot
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
import socket
import time
from utils import read_hyperparameters
HOST = '134.59.131.77'
LOAD_MODEL_PATH=None#"./logs/best_model"
LOAD_BUFFER_PATH=None#"dqn_pi_swingup_bufferN"
import numpy as np
from custom_callbacks import plot_results
PORT = 65432
logdir='./logs/dqn/'
# Use deterministic actions for evaluation and SAVE the best model
checkpoint = CheckpointCallback(save_freq=10000, save_path=logdir)
STEPS_TO_TRAIN = 200000
#lr schedule
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        env = CartPoleCosSinRpiDiscrete3(pi_conn=s)
        env0 = Monitor(env, logdir)
        eval_callback = EvalCallback(env, best_model_save_path=logdir, n_eval_episodes=2,
                                     log_path=logdir, eval_freq=10000,
                                     deterministic=True, render=False)
        hyperparams = read_hyperparameters('dqn_cartpole_50')
        model = DQN(env=env0, **hyperparams)
        if LOAD_MODEL_PATH!=None:
            model = DQN.load(LOAD_MODEL_PATH, env=env0)
            model.learning_starts=0
            model.gradient_steps=-1
            model.train_freq=-1
            model.n_episodes_rollout=1
            model.exploration_final_eps=0.05
        if LOAD_BUFFER_PATH!=None:
            model.load_replay_buffer(LOAD_BUFFER_PATH)
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
finally:

    model.save("cartpole_pi_dqnN")
    model.save_replay_buffer("dqn_pi_swingup_bufferN")
    s.close()

    try: #if training was interrupted
        plot_results(logdir)
        plot(obsArr, timeArr, actArr)
    finally:
        pass
