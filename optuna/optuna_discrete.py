"""
"""
import pickle as pkl
from typing import Any, Dict
import os
import sys

sys.path.append(os.path.abspath('./'))
from env_custom import CartPoleButter
import gym
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

N_TRIALS = 1000
N_JOBS = 3
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 3
N_TIMESTEPS = int(9e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 1
TIMEOUT = int(60 * 15)  # 15 minutes
Te = 0.05
EP_STEPS = 800


# def q_learning(trial: optuna.Trial) -> Dict[str,Any]:
#     gamma = trial.suggest_uniform("gamma",0.8,0.9999)
#     lr =
def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])#16, 32, 64, 100,
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(5e5)])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 1500, 200])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    train_freq = (1, "episode")
    gradient_steps = -1

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])

    net_arch = {"small": [128, 128], "medium": [256, 256]}[net_arch]

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy"
}
class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 1,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:

    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_dqn_params(trial))
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706,
                              resetMode='experimental', sparseReward=False)  # ,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)

    # Create the RL model
    model = DQN(**kwargs, env=env)
    # Create env used for evaluation
    eval_env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode='experimental', sparseReward=False)#,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

    with open("study.pkl", "wb+") as f:
        pkl.dump(study, f)

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()