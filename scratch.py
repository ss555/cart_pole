#tensorboard --logdir ./sac_cartpole_tensorboard/

from sb3_contrib.common.wrappers import TimeFeatureWrapper
from typing import Callable
from stable_baselines3.common import results_plotter
from custom_callbacks import plot_results
# logdir='./logs/'
# plot_results(logdir)
'''
#copysign(INFINITY,-2.0) = -inf
float fA = 14.3, fB = 0.76, fC = 0.44;
Tension = (fcontrol + fA * fvit_chariot + copysignf (fC, fvit_chariot)) / fB;
Ensuite il faut convertir la tension en fonction du signe et il faut borner entre 0 et 255

statique pour bouger le chariot: 40/255*12=
example: appliquer 1N a la vitesse -0.05m/s : U=(1+14.3*0.5 -0.44)/0.76=10.144
fPWM= 10.144/12*255=216

'''
fcontrol=2
fA = 14.3, fB = 0.76, fC = 0.44;
Tension = (fcontrol + fA * fvit_chariot + copysignf (fC, fvit_chariot)) / fB;