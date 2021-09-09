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
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from custom_callbacks import EvalX_ThetaMetric
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
N_TRIALS = 500
N_JOBS = 4
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 4
N_TIMESTEPS = int(6e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3
#TIMEOUT = int(60 * 15)  # 15 minutes in study.optimize

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(5e5)])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 10000, 20000])
    train_freq = (1, "episode")
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05])
    gradient_steps = -1
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # You can comment that out when not using gSDE
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [128, 128],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "verybig": [256, 256, 256],
    }[net_arch]

    target_entropy = "auto"

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy"
}
class TrialEvalCallback(EvalX_ThetaMetric):
    """Callback used for evaluating and reporting a trial.
    CHOICE: EvalThetaDotMetric or EvalX_ThetaMetric(not working)
    INHERITS FROM EvalX_ThetaMetric, so evaluates at different x and theta
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            eval_freq=eval_freq,
            # X_THRESHOLD=0.2,
            # N_BINS=4,
            deterministic=deterministic,
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
    Te = 0.05
    EP_STEPS = 1e5
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_sac_params(trial))
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=False, tensionMax=12, resetMode='experimental')
    # Create the RL model
    model = SAC(**kwargs, env=env)
    # Create env used for evaluation
    eval_env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=False, tensionMax=12, resetMode='experimental')
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        eval_freq = EVAL_FREQ,
        deterministic = True,
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
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=3)
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
    study.trials_dataframe().to_csv("study_results_continous_cartpole.csv")

    with open("study.pkl", "wb+") as f:
        pkl.dump(study, f)
    
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    
    fig1.show()
    fig2.show()

LOAD_PLOT = False

if LOAD_PLOT:
    with open("./optuna/study.pkl", "rb") as f:
        study = pkl.load(f)


fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)

fig1.show()
fig2.show()