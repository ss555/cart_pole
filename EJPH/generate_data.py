import sys
import os
import numpy as np
import time
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule, plot
from custom_callbacks import plot_results
from env_wrappers import Monitor
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, SAC
from env_custom import CartPoleButter
from utils import read_hyperparameters
from pathlib import Path
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from custom_callbacks import EvalCustomCallback, EvalThetaDotMetric, moving_average
from matplotlib import rcParams, pyplot as plt
import plotly.express as px
from bokeh.palettes import d3
STEPS_TO_TRAIN = 150000
EP_STEPS = 800
Te = 0.05
MANUAL_SEED = 0
# simulation results
#TODO 6.1 mettre:
#TODO fa,fb,fc,fd dans identificaiton
#Done episode reward for seed 0,5 + inference

DYNAMIC_FRICTION_SIM = True  # True
STATIC_FRICTION_SIM = True
encNoiseVarSim = True
ACTION_NOISE_SIM = False
RESET_EFFECT = False  # True#False
EVAL_TENSION_FINAL_PERF = False  # evaluate final PERFORMANCE of a cartpole for different voltages
SEED_TRAIN = True
# other
PLOT_FINAL_PERFORMANCE_STD = False  # False#
qLearningVsDQN = False  # compare q-learn and dqn
EVAL_TENSION_FINAL_PERF_seed = False  # evaluate final PERFORMANCE of a cartpole for different voltages
logdir = './EJPH/'
hyperparams = read_hyperparameters('dqn_cartpole_50')

# DONE temps d’apprentissage et note en fonction du coefficient de friction statique 4 valeurs du coefficient:Ksc,virt= 0,0.1∗Ksc,manip,Ksc,manip,10∗Ksc,manipDiscussion
STATIC_FRICTION_CART = -1.166390864012042
STATIC_FRICTION_ARR = np.array([0, 0.1, 1, 10]) * STATIC_FRICTION_CART

# DONE temps d’apprentissage et note en fonction de l’amplitude du controle
TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]#

# f_a = -20.75180095541654,  # -21.30359185798466,
# f_b = 1.059719258572224,  # 1.1088617953891196,
# f_c = -1.166390864012042,  # -0.902272006611719,
# f_d = -0.09727843708918459,  # 0.0393516077401241, #0.0,#
# wAngular = 4.881653071189049,
# kPendViscous = 0.07035332644615992,  # 0.0,#
# DONE temps  d’apprentissage  et  note  en  fonction  du  coefficient  de frottement dynamique
DYNAMIC_FRICTION_PENDULUM = 0.07035332644615992
DYNAMIC_FRICTION_ARR = np.array([0, 0.1, 1, 10]) * DYNAMIC_FRICTION_PENDULUM
#TODO : line real in full circle and other in empty
# DONE encoder noise
NOISE_TABLE = np.array([0, 0.01, 0.05, 0.1, 0.15, 0.5, 1, 5, 10]) * np.pi / 180

#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 100

colorPalette = d3['Category20'][8]


# sns.set_context("paper")
# sns.set_style("whitegrid")


# DONE graphique la fonction de recompense qui depends de la tension a 40000 pas
# DONE valeur de MAX recompense en fonction de tension
if EVAL_TENSION_FINAL_PERF:
    Path('./EJPH/tension-perf').mkdir(parents=True, exist_ok=True)
    filenames = []
    # train to generate data
    # inference to test the models
    # rainbow to plot in inference at different timesteps
    for i, tension in enumerate(TENSION_RANGE):
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
        filename = logdir + f'/tension-perf/tension_sim_{tension}_V_'
        env = Monitor(env, filename=filename)
        model = DQN(env=env, **hyperparams, seed=MANUAL_SEED)
        # eval_callback = EvalThetaDotMetric(envEval, best_model_save_path=filename[:-2], log_path=filename, eval_freq=5000, deterministic=True)
        eval_callback = EvalThetaDotMetric(envEval, log_path=filename, eval_freq=5000, deterministic=True)
        print(f'simulation for {tension} V')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
        # scoreArr[i] = eval_callback.best_mean_reward # eval_callback.evaluations_results
    plot_results(logdir + 'tension-perf', paperMode=True)



if STATIC_FRICTION_SIM:
    Path('./EJPH/static-friction').mkdir(exist_ok=True)
    filenames = []
    for frictionValue in STATIC_FRICTION_ARR:
        env0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, resetMode='experimental', f_c=frictionValue)
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, resetMode='experimental', f_c=frictionValue)

        filename = logdir + f'static-friction/static_friction_sim_{frictionValue}_'
        eval_callback = EvalThetaDotMetric(envEval, log_path=filename, eval_freq=5000)
        # filenames.append(filename)#NOT USED
        env = Monitor(env0, filename=filename)
        model = DQN(env=env, **hyperparams)
        print(f'simulation with static friciton {frictionValue} coef')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
    plot_results(logdir + 'static-friction')

if DYNAMIC_FRICTION_SIM:
    Path('./EJPH/dynamic-friction').mkdir(exist_ok=True)
    filenames = []
    for frictionValue in DYNAMIC_FRICTION_ARR:
        env0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', sparseReward=False, kPendViscous=frictionValue)
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', sparseReward=False, kPendViscous=frictionValue)
        filename = logdir + f'dynamic-friction/dynamic_friction_sim_{frictionValue}_'
        # filenames.append(filename)#NOT USED
        env = Monitor(env0, filename=filename)
        model = DQN(env=env, **hyperparams)
        eval_callback = EvalThetaDotMetric(envEval, log_path=filename, eval_freq=5000)
        print(f'simulation with dynamic friciton {frictionValue} coef')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
    plot_results(logdir + 'dynamic-friction')

if encNoiseVarSim:
    Path('./EJPH/encoder-noise').mkdir(exist_ok=True)
    filenames = []
    for encNoise in NOISE_TABLE:
        env0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', sparseReward=False, Km=encNoise)
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', sparseReward=False, Km=encNoise)
        filename = logdir + f'encoder-noise/enc_noise_sim_{encNoise}_rad_'
        # filenames.append(filename)#NOT USED
        env = Monitor(env0, filename=filename)
        eval_callback = EvalThetaDotMetric(envEval, log_path=filename, eval_freq=5000)
        model = DQN(env=env, **hyperparams)
        print(f'simulation with noise {encNoise * 180 / np.pi} degree')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
    plot_results(logdir + 'encoder-noise')


if RESET_EFFECT:
    Path('./EJPH/experimental-vs-random').mkdir(exist_ok=True)
    filenames = []
    filename = logdir + f'experimental-vs-random/_experimental_'
    env0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
    envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='random', sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
    env = Monitor(env0, filename=filename)
    eval_callback = EvalThetaDotMetric(envEval, log_path=filename, eval_freq=5000, deterministic=True)
    model = DQN(env=env, **hyperparams)
    print(f'simulation with experimental reset')
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
    env.close()

    print(f'simulation with random reset')
    filename = logdir + f'experimental-vs-random/_random_'
    env0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='random', sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
    envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='random', sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
    env = Monitor(env0, filename=filename)
    eval_callback2 = EvalThetaDotMetric(envEval, log_path=filename, eval_freq=5000, deterministic=True)
    model = DQN(env=env, **hyperparams)
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback2])
    env0.close()
    del model


    # filename = logdir + f'experimental-vs-random/iniThetaDot'
    # envTheta0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', thetaDotReset=13, sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
    # envTheta = Monitor(envTheta0, filename=filename)
    # # with ini speed 3rad/s
    # model = DQN(env=envTheta, **hyperparams)
    # eval_callback3 = EvalThetaDotMetric(envTheta0, log_path=filename, eval_freq=5000, deterministic=True)
    # print(f'simulation with random reset')
    # with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
    #     model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback3])

    plot_results(logdir + 'experimental-vs-random')

# DONE bruit sur action [0, 0.1%, 1%, 10%]
if ACTION_NOISE_SIM:#Action noise in % for standart deviation
    FORCE_STD_ARR = [0, 0.1, 1, 10]
    savePath = './EJPH/action-noise'
    Path(savePath).mkdir(exist_ok=True)
    for forceStd in FORCE_STD_ARR:
        filename = savePath+f'/force_std%_{forceStd}_'
        env0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental',sparseReward=False, forceStd=forceStd)
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental',sparseReward=False, forceStd=forceStd)
        env = Monitor(env0, filename=logdir + f'action-noise/actionStd%_{forceStd}_')
        eval_callback = EvalThetaDotMetric(envEval, log_path=filename, eval_freq=5000)
        model = DQN(env=env, **hyperparams)
        print(f'simulation with action noise in {forceStd}%')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
    plot_results(logdir + 'action-noise')
    env0.close()

if SEED_TRAIN:#basic model with default parameters
    # DONE figure:  note as a function of steps.  3 curves:  1 DQN, 1 Q-learning with learning, 1Q-learning without learning
    Path('./EJPH/seeds').mkdir(parents=True, exist_ok=True)
    for seed in range(10):
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, resetMode='experimental')  # ,integrator='semi-euler')#,integrator='rk4')
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, resetMode='experimental')  # ,integrator='semi-euler')#,integrator='rk4')
        filename = logdir + f'seeds/basic_{seed}'
        env = Monitor(env, filename)
        eval_callback = EvalThetaDotMetric(envEval, log_path=filename, eval_freq=5000)
        model = DQN(env=env, **hyperparams, seed=seed)
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
    plot_results(logdir + 'seeds')

#TODO 5.9V optuna params for 12V;


# DONE standard deviation of xas a function of the control amplitude in steadystate
# DONE standard deviation of θas a function of the control amplitude in steadystate
if PLOT_FINAL_PERFORMANCE_STD:
    Te = 0.05
    RENDER = False
    Path(f'./EJPH/final-PERFORMANCE{Te}').mkdir(exist_ok=True)
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', sparseReward=False)
    model = DQN.load('./weights/dqn50-sim/best_model.zip', env=env, seed=MANUAL_SEED)
    obs = env.reset()
    obsArr = [obs]
    start_time = time.time()
    actArr = [0.0]
    timeArr = [0.0]
    if RENDER:
        env.render()
    for i in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(action)
        if RENDER:
            env.render()
        obsArr.append(obs)
        actArr.append(action)
        timeArr.append(time.time() - start_time)
    obsArr = np.array(obsArr)
    plot(obsArr, timeArr, actArr, plotlyUse=False, savePath=logdir + f'/final-PERFORMANCE{Te}', paperMode=True)
    indexStart = 1000
    print(f'mean x : {np.mean(obsArr[indexStart:, 0])}')
    print(f'std on x: {np.std(obsArr[indexStart:, 0])}')
    print(f'mean theta : {np.mean(np.arctan2(obsArr[indexStart:, 3], obsArr[indexStart:, 2]))}')
    print(f'std on theta: {np.std(obsArr[indexStart:, 0])}')

if qLearningVsDQN:
    # DONE figure:  note as a function of steps.  3 curves:  1 DQN, 1 Q-learning with learning, 1Q - learning without learning
    Path('./EJPH/basic').mkdir(parents=True, exist_ok=True)
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, sparseReward=False, f_a=0, f_c=0, f_d=0, kPendViscous=0.0)  # ,integrator='semi-euler')#,integrator='rk4')
    # envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.47, resetMode='random', sparseReward=False,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)#,integrator='semi-euler')#,integrator='rk4')
    env = Monitor(env, filename=logdir + 'basic/basic_simulation_')
    model = DQN(env=env, **hyperparams, seed=MANUAL_SEED)
    eval_callback = EvalCustomCallback(env, eval_freq=5000, n_eval_episodes=1, deterministic=True)

    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])

    # DONE Q_learning
    plot_results(logdir + 'basic')


if EVAL_TENSION_FINAL_PERF_seed:
    saveFolder='./EJPH/tension-perf-seed'
    Path(saveFolder).mkdir(parents=True, exist_ok=True)
    filenames = []
    seed=0
    print(f'starting with seed{seed}')
    # train to generate data
    # inference to test the models
    # rainbow to plot in inference at different timesteps
    for i, tension in enumerate(TENSION_RANGE):
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
        filename = saveFolder + f'/tension_sim_{tension}_V_'
        env = Monitor(env, filename=filename)
        model = DQN(env=env, **hyperparams, seed=seed)
        # eval_callback = EvalThetaDotMetric(envEval, best_model_save_path=filename[:-2], log_path=filename, eval_freq=5000, deterministic=True)
        eval_callback = EvalThetaDotMetric(envEval, log_path=filename, eval_freq=5000, deterministic=True)
        print(f'simulation for {tension} V')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
        # scoreArr[i] = eval_callback.best_mean_reward # eval_callback.evaluations_results
    plot_results(saveFolder, paperMode=True)