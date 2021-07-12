import sys
import os
import numpy as np
import seaborn as sns
import time
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule, plot
from custom_callbacks import plot_results
from env_wrappers import Monitor
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
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
MANUAL_SEED = 5
# simulation results
qLearningVsDQN = False  # compare q-learn and dqn
DYNAMIC_FRICTION_SIM = False  # True
STATIC_FRICTION_SIM = False
encNoiseVarSim = False
ACTION_NOISE_SIM = False
RESET_EFFECT = False  # True#False
EVAL_TENSION_FINAL_PERF = False  # evaluate final PERFORMANCE of a cartpole for different voltages
PLOT_FINAL_PERFORMANCE_STD = False  # False#
SEED_TRAIN = False
logdir = './EJPH/'
hyperparams = read_hyperparameters('dqn_cartpole_50')

# DONE temps d’apprentissage et note en fonction du coefficient de friction statique 4 valeurs du coefficient:Ksc,virt= 0,0.1∗Ksc,manip,Ksc,manip,10∗Ksc,manipDiscussion
STATIC_FRICTION_CART = -0.902272006611719
STATIC_FRICTION_ARR = np.array([0, 0.1, 1, 10]) * STATIC_FRICTION_CART

# DONE temps d’apprentissage et note en fonction de l’amplitude du controle
# TENSION_RANGE = [9.4]
TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]#
colorArr = ['red', 'blue', 'green', 'cyan', 'yellow', 'tan', 'navy', 'black']
# DONE temps  d’apprentissage  et  note  en  fonction  du  coefficient  de frottement dynamique
DYNAMIC_FRICTION_PENDULUM = 0.11963736650935591
DYNAMIC_FRICTION_ARR = np.array([0, 0.1, 1, 10]) * DYNAMIC_FRICTION_PENDULUM

# DONE encoder noise
NOISE_TABLE = np.array([0, 0.01, 0.05, 0.1, 0.15, 0.5, 1, 5, 10]) * np.pi / 180

#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 100

colorPalette = d3['Category20'][8]

if qLearningVsDQN:
    # DONE figure:  note as a function of steps.  3 curves:  1 DQN, 1 Q-learning with learning, 1Q - learning without learning
    Path('./EJPH/basic').mkdir(parents=True, exist_ok=True)
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.47, resetMode='experimental',
                         sparseReward=False, f_a=0, f_c=0, f_d=0,
                         kPendViscous=0.0)  # ,integrator='semi-euler')#,integrator='rk4')
    # envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.47, resetMode='random', sparseReward=False,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)#,integrator='semi-euler')#,integrator='rk4')
    env = Monitor(env, filename=logdir + 'basic/basic_simulation_')
    model = DQN(env=env, **hyperparams, seed=MANUAL_SEED)
    eval_callback = EvalCustomCallback(env, eval_freq=5000, n_eval_episodes=1, deterministic=True)

    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])

    # DONE Q_learning
    plot_results(logdir + 'basic')
# sns.set_context("paper")
# sns.set_style("whitegrid")
#TODO plot the x, theta component of reward
#TODO normalise for 1

# DONE graphique la fonction de recompense qui depends de la tension a 40000 pas
# DONE valeur de MAX recompense en fonction de tension
if EVAL_TENSION_FINAL_PERF:
    Path('./EJPH/tension-perf').mkdir(parents=True, exist_ok=True)
    filenames = []
    scoreArr = np.zeros_like(TENSION_RANGE)
    stdArr = np.zeros_like(TENSION_RANGE)
    # train to generate data
    # inference to test the models
    # rainbow to plot in inference at different timesteps
    MODE = 'RAINBOW'   # 'TRAIN' 'INFERENCE' 'RAINBOW'
    if MODE == 'TRAIN':
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
    elif MODE == 'INFERENCE':
        def calculate_angle(prev_value,cos,sin,count=0):
            '''
            :param prev_value:
            :param cos: cosinus
            :param sin: sinus
            :return:
            '''
            if prev_value - np.arctan2(sin,cos) > np.pi:
                count += 1
                return np.arctan2(sin, cos), count
            elif np.arctan2(sin,cos) - prev_value > np.pi:
                count -= 1
                return np.arctan2(sin, cos), count
            return np.arctan2(sin, cos), count
        PLOT_EPISODE_REWARD = True
        figm1, ax1 = plt.subplots()
        figm2, ax2 = plt.subplots()
        fig = px.scatter()
        fig2 = px.scatter()
        # fig = px.scatter(x=[0], y=[0])
        for i, tension in enumerate(TENSION_RANGE):
            prev_angle_value = 0.0
            count_tours = 0
            done = False
            if PLOT_EPISODE_REWARD:
                # env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=tension, resetMode='experimental', sparseReward=False)
                env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')#CartPoleButter(tensionMax=tension,resetMode='experimental')
                model = DQN.load(logdir + f'/tension-perf/tension_sim_{tension}_V__best.zip', env=env)
                theta = 0
                cosThetaIni = np.cos(theta)
                sinThetaIni = np.sin(theta)
                rewArr = []
                obs = env.reset(costheta=cosThetaIni, sintheta=sinThetaIni)
                # env.reset()
                thetaArr, thetaDotArr, xArr, xDotArr = [], [], [], []
                for j in range(EP_STEPS):
                    act,_ = model.predict(obs,deterministic=True)
                    obs, rew, done, _ = env.step(act)
                    rewArr.append(rew)
                    # if tension==4.7:
                    #     env.render()
                    angle, count_tours = calculate_angle(prev_angle_value, obs[2], obs[3], count_tours)
                    prev_angle_value = angle
                    thetaArr.append(angle+count_tours*np.pi*2)
                    thetaDotArr.append(obs[4])
                    xArr.append(obs[0])
                    xDotArr.append(obs[1])
                    if done:
                        print(f'ended episode {tension} with {count_tours} tours and {np.sum(rewArr)} reward')
                        ax1.plot(thetaArr, '.')
                        fig.add_scatter(x=np.linspace(1,EP_STEPS,EP_STEPS), y=thetaArr, name=f'volt: {tension}')
                        fig2.add_scatter(x=np.linspace(1,EP_STEPS,EP_STEPS), y=xArr, name=f'volt: {tension}')
                        break
                        # ax1.savefig(logdir+'/thetaA.pdf')
                #TODO theta tensions
                #TODO time at the training
                ax2.plot(moving_average(rewArr,20), color = colorPalette[i])
            else:
                episode_rewards, episode_lengths = evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=100,
                    deterministic=True,
                    return_episode_rewards=True,
                )
                scoreArr[i] = np.mean(episode_rewards)
                stdArr[i] = np.std(episode_rewards)
                print('done')
        if PLOT_EPISODE_REWARD:
            fig.show()
            fig2.show()

            ax1.legend([str(t)+'V' for t in TENSION_RANGE], loc='upper right')
            ax2.legend([str(t)+'V' for t in TENSION_RANGE], loc='upper right')
            ax1.set_xlabel('timesteps')
            ax2.set_xlabel('timesteps')
            ax2.set_ylabel('Rewards')
            # plt.title('Effect of the applied tension on the "greedy policy" reward')
            figm2.savefig('./EJPH/plots/episode_rew_tension.pdf')
            figm2.show()
            figm1.savefig(f'./EJPH/plots/episode_theta{theta/np.pi*180}.pdf')
            figm1.show()
        else:
            tensionMax = np.array(TENSION_RANGE)
            plt.plot(tensionMax, scoreArr, 'ro-')
            plt.fill_between(tensionMax, scoreArr + stdArr, scoreArr - stdArr, facecolor='red', alpha=0.5)
            plt.xlabel('Tension (V)')
            plt.ylabel('Rewards')
            plt.title('Effect of the applied tension on the "greedy policy" reward')
            plt.savefig('./EJPH/plots/episode_rew_10000eps')
            plt.show()
            np.savez(
                './EJPH/plots/tension-perf10000ep',
                tensionRange=tensionMax,
                results=scoreArr,
                resultsStd=stdArr
            )
        print('done inference on voltages')
    elif MODE == 'RAINBOW':
        print('plotting in rainbow for different voltages applied')
        EP_LENGTH = 800
        scoreArr1 = np.zeros_like(TENSION_RANGE)
        scoreArr2 = np.zeros_like(TENSION_RANGE)
        scoreArr3 = np.zeros_like(TENSION_RANGE)
        scoreArr4 = np.zeros_like(TENSION_RANGE)
        scoreArr5 = np.zeros_like(TENSION_RANGE)
        p1, p2, p3, p4, p5 = 0.2, 0.4, 0.6, 0.8, 1
        for j, tension in enumerate(TENSION_RANGE):
            env = CartPoleButter(Te=Te, N_STEPS=EP_LENGTH, discreteActions=True, tensionMax=tension, resetMode='experimental', sparseReward=False)
            model = DQN.load(logdir + f'/tension-perf/tension_sim_{tension}_V__best', env=env)
            # model = DQN.load(logdir + f'/tension-perf/thetaDot10/tension_sim_{tension}_V_.zip_2', env=env)
            episode_rewards = 0
            # THETA_THRESHOLD = np.pi/18
            # theta = np.linspace(-THETA_THRESHOLD, THETA_THRESHOLD, N_TRIALS)
            obs = env.reset(costheta=0.984807753012208,sintheta=-0.17364817766693033)
            for i in range(EP_LENGTH):
                action, _state = model.predict(obs)
                obs, cost, done, _ = env.step(action)
                episode_rewards += cost
                if i == int(EP_LENGTH * p1 - 1):
                    scoreArr1[j] = episode_rewards  # np.mean(episode_rewards)
                elif i == int(EP_LENGTH * p2 - 1):
                    scoreArr2[j] = episode_rewards
                elif i == int(EP_LENGTH * p3 - 1):
                    scoreArr3[j] = episode_rewards
                elif i == int(EP_LENGTH * p4 - 1):
                    scoreArr4[j] = episode_rewards
                elif i == int(EP_LENGTH * p5 - 1):
                    scoreArr5[j] = episode_rewards
                if done:
                    print(f'observations: {obs} and i: {i}')
                    break

            print('done')


        fillArr = np.zeros_like(scoreArr1)
        plt.plot(TENSION_RANGE, scoreArr1/EP_LENGTH, 'o-r')
        plt.fill_between(TENSION_RANGE, scoreArr1/EP_LENGTH, fillArr, facecolor=colorArr[0], alpha=0.5)
        plt.plot(TENSION_RANGE, scoreArr2/EP_LENGTH, 'o-b')
        plt.fill_between(TENSION_RANGE, scoreArr2/EP_LENGTH, scoreArr1/EP_LENGTH, facecolor=colorArr[1], alpha=0.5)
        plt.plot(TENSION_RANGE, scoreArr3/EP_LENGTH, 'o-g')
        plt.fill_between(TENSION_RANGE, scoreArr3/EP_LENGTH, scoreArr2/EP_LENGTH, facecolor=colorArr[2], alpha=0.5)
        plt.plot(TENSION_RANGE, scoreArr4/EP_LENGTH, 'o-c')
        plt.fill_between(TENSION_RANGE, scoreArr4/EP_LENGTH, scoreArr3/EP_LENGTH, facecolor=colorArr[3], alpha=0.5)
        plt.plot(TENSION_RANGE, scoreArr5/EP_LENGTH, 'o-y')
        plt.fill_between(TENSION_RANGE, scoreArr5/EP_LENGTH, scoreArr4/EP_LENGTH, facecolor=colorArr[4], alpha=0.5)
        plt.hlines(y=1,xmin=min(TENSION_RANGE),xmax=max(TENSION_RANGE),linestyles='--')
        # plt.plot(TENSION_RANGE, scoreArr1, 'o-r')
        # plt.fill_between(TENSION_RANGE, scoreArr1, fillArr, facecolor=colorArr[0], alpha=0.5)
        # plt.plot(TENSION_RANGE, scoreArr2, 'o-b')
        # plt.fill_between(TENSION_RANGE, scoreArr2, scoreArr1, facecolor=colorArr[1], alpha=0.5)
        # plt.plot(TENSION_RANGE, scoreArr3, 'o-g')
        # plt.fill_between(TENSION_RANGE, scoreArr3, scoreArr2, facecolor=colorArr[2], alpha=0.5)
        # plt.plot(TENSION_RANGE, scoreArr4, 'o-c')
        # plt.fill_between(TENSION_RANGE, scoreArr4, scoreArr3, facecolor=colorArr[3], alpha=0.5)
        # plt.plot(TENSION_RANGE, scoreArr5, 'o-y')
        # plt.fill_between(TENSION_RANGE, scoreArr5, scoreArr4, facecolor=colorArr[4], alpha=0.5)
        # plt.hlines(y=EP_LENGTH,xmin=min(TENSION_RANGE),xmax=max(TENSION_RANGE),linestyles='--')
        plt.grid()
        plt.xlabel('Tension (V)')
        plt.ylabel('Rewards')
        # plt.title('Effect of the applied tension on the "greedy policy" reward')

        # for p
        plt.legend([f'{int(p1 * 100)}% of episode', f'{int(p2 * 100)}% of episode', f'{int(p3 * 100)}% of episode',f'{int(p4 * 100)}% of episode',f'{int(p5 * 100)}% of episode'],
                   loc='best')
        plt.savefig('./EJPH/plots/episode_rainbow.pdf')
        plt.show()
    # elif MODE == '':

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

# TODO effect of initialisation with eval callback
if RESET_EFFECT:
    Path('./EJPH/experimental-vs-random').mkdir(exist_ok=True)
    filenames = []
    filename = logdir + f'experimental-vs-random/_experimental_'
    env0 = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
    envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='experimental', sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
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
if ACTION_NOISE_SIM:
    FORCE_STD_ARR = [0, 0.1, 1, 10]
    for forceStd in FORCE_STD_ARR:
        Path('./EJPH/action-noise').mkdir(exist_ok=True)
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
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode='experimental',
                             sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode='random_theta_thetaDot',
                                 sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
        filename = logdir + f'seeds/basic_{seed}'
        env = Monitor(env, filename)
        eval_callback = EvalCustomCallback(envEval, log_path=filename, eval_freq=5000, n_eval_episodes=51,
                                           deterministic=True)
        model = DQN(env=env, **hyperparams, seed=seed)
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
    plot_results(logdir + 'seeds')
#
#TODO courbe avec seed 0
#TODO courbe 14,13,boxplots pour 0
#TODO 5.9V optuna params for 12V;
#rainbow
if SEED_TRAIN:#basic model
    # DONE figure:  note as a function of steps.  3 curves:  1 DQN, 1 Q-learning with learning, 1Q-learning without learning
    Path('./EJPH/seeds').mkdir(parents=True, exist_ok=True)
    for seed in range(10):
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=12, resetMode='experimental',
                             sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
        envEval = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode='random_theta_thetaDot',
                                 sparseReward=False)  # ,integrator='semi-euler')#,integrator='rk4')
        filename = logdir + f'seeds/basic_{seed}'
        env = Monitor(env, filename)
        eval_callback = EvalCustomCallback(envEval, log_path=filename, eval_freq=5000, n_eval_episodes=51,
                                           deterministic=True)
        model = DQN(env=env, **hyperparams, seed=seed)
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
    plot_results(logdir + 'seeds')

# TODO standard deviation of xas a function of the control amplitude in steadystate
# TODO standard deviation of θas a function of the control amplitude in steadystate
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

# TODO initialise random, inference graph
# TODO l'apprentissage/l'inferance/varie_temps_finale