from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
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
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=0):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model_training')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
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


def plot_results(log_folder, window_size=50, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'walltime_hrs')#'timesteps')
    y = moving_average(y, window=window_size)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    # plt.xlabel('Number of Timesteps')
    plt.xlabel('Hours')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
    plt.savefig(log_folder + title + ".png")
# class PlottingCallback(BaseCallback):
#     """
#     Callback for plotting the performance in realtime.
#
#     :param verbose: (int)
#     """
#     def __init__(self, verbose=1):
#         super(PlottingCallback, self).__init__(verbose)
#         self._plot = None
#
#     def _on_step(self) -> bool:
#         # get the monitor's data-old
#         x, y = ts2xy(load_results(log_dir), 'timesteps')
#       if self._plot is None: # make the plot
#           plt.ion()
#           fig = plt.figure(figsize=(6,3))
#           ax = fig.add_subplot(111)
#           line, = ax.plot(x, y)
#           self._plot = (line, ax, fig)
#           plt.show()
#       else: # update and rescale the plot
#           self._plot[0].set_data(x, y)
#           self._plot[-2].relim()
#           self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02,
#                                    self.locals["total_timesteps"] * 1.02])
#           self._plot[-2].autoscale_view(True,True,True)
#           self._plot[-1].canvas.draw()