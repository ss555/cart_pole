__all__ = ["Monitor", "ResultsWriter", "get_monitor_files", "load_results"]
from typing import Callable
from gym.wrappers.monitoring import video_recorder
import csv
import json
import os
import time
from glob import glob
from typing import Dict, List, Optional, Tuple, Union
import gym
import numpy as np
import pandas as pd
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
    ):
        super(Monitor, self).__init__(env=env)
        self.t_start = time.time()
        if filename is not None:
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": env.spec and env.spec.id},
                extra_keys=reset_keywords + info_keywords,
            )
        else:
            self.results_writer = None
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True
        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action
        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super(Monitor, self).close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps
        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes
        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes
        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes
        :return:
        """
        return self.episode_times

class CsvLogSaveBestCallback(gym.Wrapper):
    def __init__(self, env: gym.Env,
                 saveLogFile:str='./monitor.csv',
                 saveModelFile:str='./best.zip'):
        self.env = env
        self.saveLogFile = saveLogFile
        self.saveModelFile = saveModelFile
        self.fileHandler = open(saveLogFile,'wt')
        self.writer = csv.DictWriter(self.fileHandler,fieldnames=('r','l','t'))
        self.writer.writeheader()
        self.fileHandler.flush()
        self.rewards = []
        self.timesteps = 0
        # self.t = []
        self.t_start = time.time()
    def step(self, action):
        rew, obs, done, _ = self.env.step(action)
        self.rewards.append(rew)
        self.timesteps += 1
        if done:
            self.writer.writerow({'r': round(self.rewards, 6),'l': round(self.timesteps, 6),'t': round(time.time()-self.t_start, 6)})

    def reset(self, **kwargs):
        self.rewards = []
        self.timesteps = 0


class LoadMonitorResultsError(Exception):
    """
    Raised when loading the monitor log fails.
    """

    pass


class ResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class
    :param filename: the location to save a log file, can be None for no log
    :param header: the header dictionary object of the saved csv
    :param reset_keywords: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    """

    def __init__(
        self,
        filename: str = "",
        header: Dict[str, Union[float, str]] = None,
        extra_keys: Tuple[str, ...] = (),
    ):
        if header is None:
            header = {}
        if not filename.endswith(Monitor.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor.EXT)
            else:
                filename = filename + Monitor.EXT
                # filename = filename + "." + Monitor.EXT
        self.file_handler = open(filename, "wt")
        self.file_handler.write("#%s\n" % json.dumps(header))
        self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t") + extra_keys)
        self.logger.writeheader()
        self.file_handler.flush()

    def write_row(self, epinfo: Dict[str, Union[float, int]]) -> None:
        """
        Close the file handler
        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()

    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()


def get_monitor_files(path: str) -> List[str]:
    """
    get all the monitor files in the given path
    :param path: the logging folder
    :return: the log files
    """
    return sorted(glob(os.path.join(path, "*" + Monitor.EXT)),key=os.path.getmtime)
X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: the x and y output
    """

    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = data_frame.r.values
    else:
        raise NotImplementedError
    return x_var, y_var

def load_results(path: str) -> pd.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``
    :param path: the directory path containing the log file(s)
    :return: the logged data
    """
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{Monitor.EXT} found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files:
        data_frame,headers=load_data_from_csv(file_name)
        data_frames.append(data_frame)
    return data_frames, monitor_files
def load_data_from_csv(path_to_file: str):
    headers = []
    with open(path_to_file, "rt") as file_handler:
        first_line = file_handler.readline()
        assert first_line[0] == "#"
        header = json.loads(first_line[1:])
        data_frame = pd.read_csv(file_handler, index_col=None)
        headers.append(header)
        data_frame["t"] += header["t_start"]
    return data_frame, headers

# def play(eval_env_id, model, steps: int = 50, deterministic: bool =True, video_path:str='./logs/video/dqn.mp4'):
#     '''
#
#     :param eval_env_id: env id or environement
#     :param model: RL model
#     :param steps: num of steps for inference
#     :param deterministic: action?
#     :param video_path: save path
#     :return:
#     '''
#     num_episodes = 0
#     video_recorder = None
#     env0 = gym.make(eval_env_id)
#     env = DummyVecEnv([lambda: env0])
#     video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
#     obs = env.reset()
#     for i in range(steps):
#         env.unwrapped.render()
#         video_recorder.capture_frame()
#         action = model.predict(obs, deterministic=deterministic)
#         obs, rew, done, info = env.step(action)
#         if done:
#             obs = env.reset()
#     if video_recorder.enabled:
#         # save video of first episode
#         print("Saved video.")
#         video_recorder.close()
#         video_recorder.enabled = False

class VideoRecorderWrapper(gym.Wrapper):
    """
    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length:  Length of recorded videos (done for security if we forget to close the env to avoid memory leakage)
    :param name_prefix: Prefix to the video name
    """

    def __init__(
        self,
        venv: gym.Env,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 20000,
        name_prefix: str = "rl-video",
    ):


        super().__init__(venv)
        # self.env.metadata = venv.metadata
        self.venv = venv
        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0

    def reset(self):
        obs = self.venv.reset()
        self.start_video_recorder()
        return obs

    def start_video_recorder(self) -> None:
        self.close_video_recorder()

        video_name = f"{self.name_prefix}"
        # video_name = f"{self.name_prefix}-step-{self.step_id}-to-step-{self.step_id + self.video_length}"
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.venv, base_path=base_path, metadata={"step_id": self.step_id}
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.step_id)

    def step(self, action):
        obs, rews, dones, infos = self.venv.step(action)

        self.step_id += 1
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                print(f"Saving video to {self.video_recorder.path}")
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()

        return obs, rews, dones, infos

    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def close(self) -> None:
        self.env.close()
        self.close_video_recorder()

    def __del__(self):
        self.close()
