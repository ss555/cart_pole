import sys
import os
import numpy as np
import seaborn as sns
import time
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule,plot
from custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from env_custom import CartPoleButter
from utils import read_hyperparameters
from pathlib import Path
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from custom_callbacks import EvalCustomCallback
from matplotlib import pyplot as plt
STEPS_TO_TRAIN=100000
EP_STEPS=800
Te=0.05
MANUAL_SEED=5
#simulation results
qLearningVsDQN=False #compare q-learn and dqn
DYNAMIC_FRICTION_SIM=True#True
STATIC_FRICTION_SIM=True
encNoiseVarSim=True
ACTION_NOISE_SIM=True
RESET_EFFECT=True
EVAL_TENSION_FINAL_PERF=True#evaluate final perfomance of a cartpole for different voltages
PLOT_FINAL_PERFOMANCE_STD=False#False#
SEED_TRAIN=False

logdir='./EJPH/'
hyperparams=read_hyperparameters('dqn_cartpole_50')

#DONE temps d’apprentissage et note en fonction du coefficient de friction statique 4 valeurs du coefficient:Ksc,virt= 0,0.1∗Ksc,manip,Ksc,manip,10∗Ksc,manipDiscussion
STATIC_FRICTION_CART=-0.902272006611719
STATIC_FRICTION_ARR=np.array([0,0.1,1,10])*STATIC_FRICTION_CART

#DONE temps d’apprentissage et note en fonction de l’amplitude du controle
tensionMax = [2.4, 3.5, 4.7, 5.9, 7.1, 8.47, 9.4, 12]

#DONE temps  d’apprentissage  et  note  en  fonction  du  coefficient  de frottement dynamique
DYNAMIC_FRICTION_PENDULUM=0.11963736650935591
DYNAMIC_FRICTION_ARR=np.array([0,0.1,1,10])*DYNAMIC_FRICTION_PENDULUM

#DONE encoder noise
NOISE_TABLE=np.array([0,0.01,0.05,0.1,0.15,0.5,1,5,10])*np.pi/180


if qLearningVsDQN:
    #DONE figure:  note as a function of steps.  3 curves:  1 DQN, 1 Q-learning with learning, 1Q-learning without learning
    Path('./EJPH/basic').mkdir(parents=True, exist_ok=True)
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.47, resetMode='experimental', sparseReward=False,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)#,integrator='ode')#,integrator='rk4')
    # envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.47, resetMode='random', sparseReward=False,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)#,integrator='ode')#,integrator='rk4')
    env = Monitor(env, filename=logdir+'basic/basic_simulation_')
    model = DQN(env=env,**hyperparams, seed=MANUAL_SEED)
    eval_callback = EvalCustomCallback(env, eval_freq=15000, n_eval_episodes=1, deterministic=True)

    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback,eval_callback])

    #TODO Q_learning
    plot_results(logdir+'basic')

#DONE graphique la fonction de recompense qui depends de la tension a 40000 pas
#DONE valeur de MAX recompense en fonction de tension
if EVAL_TENSION_FINAL_PERF:
    Path('./EJPH/tension-perf').mkdir(parents=True, exist_ok=True)
    filenames=[]
    scoreArr=np.zeros_like(tensionMax)
    stdArr=np.zeros_like(tensionMax)
    TRAIN=True
    if TRAIN:
        for i,tension in enumerate(tensionMax):
            env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=tension,resetMode='experimental',
                                  sparseReward=False)
            envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=tension,
                                 resetMode='random_theta_thetaDot', sparseReward=False)
            filename=logdir + f'/tension-perf/tension_sim_{tension}_V_'
            env = Monitor(env, filename=filename)
            model = DQN(env=env, **hyperparams, seed=MANUAL_SEED)
            eval_callback = EvalCustomCallback(envEval, log_path=filename, eval_freq=15000, n_eval_episodes=25, deterministic=True)
            print(f'simulation for {tension} V')
            with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
                model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
            # scoreArr[i] = eval_callback.best_mean_reward # eval_callback.evaluations_results
    else:
        for i,tension in enumerate(tensionMax):
            env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=tension,resetMode='random_theta_thetaDot',sparseReward=False)
            model = DQN.load(logdir + f'/tension-perf/tension-perf{tension}', env=env)
            episode_rewards, episode_lengths = evaluate_policy(
                model,
                env,
                n_eval_episodes=10000,
                deterministic=True,
                return_episode_rewards=True,
            )
            scoreArr[i]=np.mean(episode_rewards)
            stdArr[i]=np.std(episode_rewards)
            plot_results(logdir+'tension-perf',paperMode=True)
            sns.set_context("paper")
            sns.set_style("whitegrid")
            tensionMax=np.array(tensionMax)
            plt.plot(tensionMax,scoreArr,'ro-')
            plt.fill_between(tensionMax, scoreArr+stdArr, scoreArr-stdArr, facecolor='red', alpha=0.5)
            plt.xlabel('Tension (V)')
            plt.ylabel('Rewards')
            plt.title('Effect of the applied tension on the "greedy policy" reward')
            plt.show()

if STATIC_FRICTION_SIM:
    Path('./EJPH/static-friction').mkdir(exist_ok=True)
    filenames=[]
    for frictionValue in STATIC_FRICTION_ARR:
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental',sparseReward=False,f_c=frictionValue)  # ,integrator='ode')#,integrator='rk4')
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='random', sparseReward=False, f_c=frictionValue)
        eval_callback = EvalCustomCallback(envEval, log_path=filename, eval_freq=15000, n_eval_episodes=25, deterministic=True)
        filename = logdir + f'static-friction/static_friction_sim_{frictionValue}_'
        #filenames.append(filename)#NOT USED
        env = Monitor(env, filename=filename)
        model = DQN(env=env, **hyperparams)
        print(f'simulation with static friciton {frictionValue} coef')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    plot_results(logdir+'static-friction')

if DYNAMIC_FRICTION_SIM:
    Path('./EJPH/dynamic-friction').mkdir(exist_ok=True)
    filenames=[]
    for frictionValue in DYNAMIC_FRICTION_ARR:
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental', sparseReward=False,kPendViscous=frictionValue)  # ,integrator='ode')#,integrator='rk4')
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='random', sparseReward=False,kPendViscous=frictionValue)  # ,integrator='ode')#,integrator='rk4')
        filename = logdir + f'dynamic-friction/dynamic_friction_sim_{frictionValue}_'
        #filenames.append(filename)#NOT USED
        env = Monitor(env, filename=filename)
        model = DQN(env=env, **hyperparams)
        eval_callback = EvalCustomCallback(envEval, log_path=filename,eval_freq=15000, n_eval_episodes=25, deterministic=True)
        print(f'simulation with dynamic friciton {frictionValue} coef')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback,eval_callback])
    plot_results(logdir+'dynamic-friction')

if encNoiseVarSim:
    Path('./EJPH/encoder-noise').mkdir(exist_ok=True)
    filenames=[]
    for encNoise in NOISE_TABLE:
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental', sparseReward=False,Km=encNoise)  # ,integrator='ode')#,integrator='rk4')
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='random', sparseReward=False,Km=encNoise)  # ,integrator='ode')#,integrator='rk4')
        filename = logdir + f'encoder-noise/enc_noise_sim_{encNoise}_rad_'
        #filenames.append(filename)#NOT USED
        env = Monitor(env, filename=filename)
        eval_callback = EvalCustomCallback(envEval, log_path=filename, eval_freq=15000,
                                           n_eval_episodes=25, deterministic=True)

        model = DQN(env=env, **hyperparams)
        print(f'simulation with noise {encNoise*180/np.pi} degree')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    plot_results(logdir+'encoder-noise')
#TODO effect of initialisation with eval callback
if RESET_EFFECT:
    Path('./EJPH/experimental-vs-random').mkdir(exist_ok=True)
    filenames=[]
    filename = logdir + f'experimental-vs-random/experimental'
    env0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental', sparseReward=False)  # ,integrator='ode')#,integrator='rk4')
    env = Monitor(env0, filename=filename)
    envEval0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='random', sparseReward=False)
    envEval = Monitor(envEval0, filename=filename)

    #filenames.append(filename)#NOT USED
    eval_callback = EvalCustomCallback(envEval0, log_path=filename,eval_freq=15000,n_eval_episodes=25, deterministic=True)
    model = DQN(env=env, **hyperparams)
    print(f'simulation with experimental reset')
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])

    print(f'simulation with random reset')
    filename = logdir + f'experimental-vs-random/random'

    eval_callback2 = EvalCustomCallback(envEval0, log_path=filename, eval_freq=15000,n_eval_episodes=25, deterministic=True)

    model = DQN(env=envEval, **hyperparams)
    print(f'simulation with random reset')
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback,eval_callback2])
    plot_results(logdir+'experimental-vs-random')

#DONE bruit sur action [0, 0.1%, 1%, 10%]
if ACTION_NOISE_SIM:
    FORCE_STD_ARR=[0, 0.1, 1, 10]
    for forceStd in FORCE_STD_ARR:
        Path('./EJPH/action-noise').mkdir(exist_ok=True)
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental',sparseReward=False,forceStd=forceStd)
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='random',sparseReward=False,forceStd=forceStd)
        env=Monitor(env,filename=logdir+f'action-noise/actionStd%-{forceStd}')
        eval_callback = EvalCustomCallback(envEval, log_path=filename, eval_freq=15000,
                                           n_eval_episodes=25, deterministic=True)
        model=DQN(env=env,**hyperparams)
        print(f'simulation with action noise in {forceStd}%')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN,callback=[cus_callback,eval_callback])
    plot_results(logdir+'action-noise')


if SEED_TRAIN:
    # DONE figure:  note as a function of steps.  3 curves:  1 DQN, 1 Q-learning with learning, 1Q-learning without learning
    Path('./EJPH/seeds').mkdir(parents=True, exist_ok=True)
    for seed in range(10):
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode='experimental', sparseReward=False)#,integrator='ode')#,integrator='rk4')
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode='random', sparseReward=False)#,integrator='ode')#,integrator='rk4')
        filename = logdir + f'seeds/basic_{seed}'
        env = Monitor(env, filename)
        eval_callback = EvalCustomCallback(envEval, log_path=filename,eval_freq=15000, n_eval_episodes=25, deterministic=True)
        model = DQN(env=env,**hyperparams, seed=seed)
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback,eval_callback])
    plot_results(logdir + 'seeds')

#TODO standard deviation of xas a function of the control amplitude in steadystate
#TODO standard deviation of θas a function of the control amplitude in steadystate
if PLOT_FINAL_PERFOMANCE_STD:
    Te = 0.05
    RENDER=False
    Path(f'./EJPH/final-perfomance{Te}').mkdir(exist_ok=True)
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', sparseReward=False)
    model=DQN.load('./weights/dqn50-sim/best_model.zip',env=env,seed=MANUAL_SEED)
    obs=env.reset()
    obsArr = [obs]
    start_time=time.time()
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
    plot(obsArr, timeArr, actArr, plotlyUse=False,savePath=logdir+f'/final-perfomance{Te}',paperMode=True)
    indexStart = 1000
    print(f'mean x : {np.mean(obsArr[indexStart:, 0])}')
    print(f'std on x: {np.std(obsArr[indexStart:, 0])}')
    print(f'mean theta : {np.mean(np.arctan2(obsArr[indexStart:, 3], obsArr[indexStart:, 2]))}')
    print(f'std on theta: {np.std(obsArr[indexStart:, 0])}')

#TODO initialise random, inference graph
#TODO l'apprentissage/l'inferance/varie_temps_finale

