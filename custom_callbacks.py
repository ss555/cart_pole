import csv

from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from env_wrappers import load_results, ts2xy, load_data_from_csv
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import gym
import warnings
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common import base_class
from env_custom import CartPoleButter

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar
    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps or every episode)
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved. It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int = 0, log_dir: str = '/logs', monitor_filename:str = 'monitor.csv', verbose=0):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.monitor_filename = monitor_filename
        self.save_path = os.path.join(log_dir, 'best_model_training')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        pass
        # if self.save_path is not None:
        #     os.makedirs(self.save_path, exist_ok=True)

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        #Those are reference and will be updated automatically
        print('initialised callback Save')
    def check_save(self):
        # Retrieve training reward
        data, _ = load_data_from_csv(self.monitor_filename)
        x, y = ts2xy(data, 'timesteps')
        if len(x) > 0:
            # Mean training reward over the last 10 episodes
            mean_reward = np.mean(y[-10:])
            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                self.model.save(self.save_path)
    def _on_step(self) -> bool:
        # self.model.
        if self.check_freq==0:
            try:
                if self.locals['done']==True:
                    self.check_save()
            except:

                print('Define COUNTER of steps in an episode in your environement')
        else:
            if self.n_calls % self.check_freq == 0:
                self.check_save()
        return True


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, window_size=30, title='Learning Curve',only_return_data=False, paperMode=False):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """  # x, y = ts2xy(load_results(log_folder), 'walltime_hrs')#'timesteps')
    x_varArr=[]
    y_varArr=[]
    data_frame, legends = load_results(log_folder)
    for data in data_frame:
        # Convert to hours
        # x_var = data.t.values / 3600.0
        x_var = np.cumsum(data.l.values)
        y_var = data.r.values
        try:
            y_var = moving_average(y_var, window=window_size)
        except:
            print('empty file')
        # Truncate x
        x_var = x_var[len(x_var) - len(y_var):]
        x_var -= x_var[0]
        if not only_return_data:
            if paperMode:
                sns.set_context("paper")
                sns.set_style("whitegrid")
            else:
                sns.set_style("whitegrid")
                sns.set(font='serif typeface', rc={"font.size": 10})
                sns.lineplot(y=y_var, x=x_var)
        x_varArr.append(x_var)
        y_varArr.append(y_var)

    legends = np.array([legend.split('_') for legend in legends])
    legs=[]
    try:
        for i,counter in enumerate(legends[:,-2]):
            legs.append(legends[i,-3]+legends[i,-2])
    except:
        pass
    if not only_return_data:
        plt.legend(legs,loc='best')
        plt.xlabel('timesteps')
        plt.ylabel('Rewards')
        plt.title(title)
        plt.savefig(log_folder+'/plot.png')
        plt.show()
        print(f'saved to {log_folder}')
    return x_varArr,y_varArr,legends

class CheckPointEpisode(BaseCallback):
    '''
    callback for saving every n episodes
    '''
    def __init__(self, save_freq_ep: int,
                 save_path: str=None):
        super(CheckPointEpisode, self).__init__()
        self.save_path = save_path
        self.save_freq_ep = save_freq_ep
        self.n_episodes=0
    def _init_callback(self) -> None:
        assert self.save_path is not None, "path for checkpoint callback is not specified"
    def _on_rollout_start(self) -> None:
        self.n_episodes += 1
        if self.n_episodes%self.save_freq_ep == 0 and self.n_episodes!=0:
            try:
                self.model.save(self.save_path)
            except:
                print('error occured while saving checkpoint')
    def _on_step(self) -> bool:
        pass
class EvalCustomCallback(EventCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: filename for evaluations, with automatically added (``.npz``) if not exist
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str=None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        stop_training_reward_threshold : bool = np.inf,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(EvalCustomCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.log_path=log_path
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.evaluations_results = []
        self.evaluations_length = []
        self.evaluations_timesteps = []
        self.stop_training_reward_threshold = stop_training_reward_threshold

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)


    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
            )
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward
            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_length.append(episode_lengths)
                self.evaluations_results.append(episode_rewards)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_length=self.evaluations_length
                )
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")


            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(self.best_model_save_path)
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
                if mean_reward > self.stop_training_reward_threshold:
                    self.model._total_timesteps=1e7#end

        return True

class EvalThetaDotMetric(EventCallback):
    """
    Callback for evaluating a cartpole systematically on a given metric(grid on THETA and THETA_DOT.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: filename for evaluations, with automatically added (``.npz``) if not exist
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 0,
        warn: bool = True,
        THETA_DOT_THRESHOLD : float = 0.0,
        N_BINS : int = 10
    ):
        super(EvalThetaDotMetric, self).__init__(callback_on_new_best, verbose=verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.log_path=log_path
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.evaluations_results = []
        self.evaluations_length = []
        self.evaluations_timesteps = []
        self.THETA_DOT_THRESHOLD = THETA_DOT_THRESHOLD
        self.THETA_THRESHOLD = -np.pi/18
        self.N_BINS=N_BINS
        if THETA_DOT_THRESHOLD != 0:
            theta_dot = np.linspace(-self.THETA_DOT_THRESHOLD, self.THETA_DOT_THRESHOLD, self.N_BINS)
        else:
            theta_dot = [0]

        theta = np.linspace(self.THETA_THRESHOLD, self.THETA_THRESHOLD, self.N_BINS)

        self.arrTest = np.transpose([np.tile(theta, len(theta_dot)), np.repeat(theta_dot, len(theta))])

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        # if not isinstance(self.training_env, type(self.eval_env)):
        #     warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)


    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            episode_rewards = np.zeros(self.arrTest.shape[0], dtype=np.float32)
            episode_lengths = np.zeros(self.arrTest.shape[0], dtype=np.float32)
            for i,elem in enumerate(self.arrTest):
                done=False
                obs=self.eval_env.reset(costheta=np.cos(elem[0]),sintheta=np.sin(elem[0]), theta_ini_speed=elem[1])
                while not done:
                    action,_state = self.model.predict(obs,deterministic=True)
                    obs,reward,done,_ = self.eval_env.step(action)
                    episode_rewards[i] += reward
                    episode_lengths[i] += 1
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward
            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_length.append(episode_lengths)
                self.evaluations_results.append(episode_rewards)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_length=self.evaluations_length
                )
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")


            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(self.best_model_save_path)
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

            self.eval_env.reset()

        return True
